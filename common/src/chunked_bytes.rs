use std::io::{self, BufRead, Read, Seek, SeekFrom};
use std::ops::Range;
use std::sync::Arc;

use ownedbytes::OwnedBytes;

#[derive(Debug)]
struct Chunk {
    start: usize,
    bytes: OwnedBytes,
}

impl Chunk {
    fn end(&self) -> usize {
        self.start + self.bytes.len()
    }
}

fn chunk_index(chunks: &[Chunk], position: usize) -> usize {
    chunks.partition_point(|chunk| chunk.end() <= position)
}

/// An immutable view over a list of buffers, read without copying them together.
#[derive(Clone, Debug)]
pub struct ChunkedBytes(Repr);

// Most reads fit in one buffer, so that case is stored inline without allocating.
#[derive(Clone, Debug)]
enum Repr {
    Contiguous(OwnedBytes),
    Chunked {
        chunks: Arc<[Chunk]>,
        range: Range<usize>,
    },
}

impl ChunkedBytes {
    /// Builds a view over `chunks`, in order. Empty chunks are skipped.
    pub fn new(chunks: Vec<OwnedBytes>) -> Self {
        let mut chunks: Vec<_> = chunks
            .into_iter()
            .filter(|bytes| !bytes.is_empty())
            .collect();
        if chunks.len() <= 1 {
            return chunks.pop().unwrap_or_else(OwnedBytes::empty).into();
        }
        let mut len = 0;
        let chunks = chunks
            .into_iter()
            .map(|bytes| {
                let start = len;
                len += bytes.len();
                Chunk { start, bytes }
            })
            .collect();
        Self(Repr::Chunked {
            chunks,
            range: 0..len,
        })
    }

    /// Returns the number of bytes in the view.
    pub fn len(&self) -> usize {
        match &self.0 {
            Repr::Contiguous(bytes) => bytes.len(),
            Repr::Chunked { range, .. } => range.len(),
        }
    }

    /// Returns true if the view has no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a sub-view without copying.
    ///
    /// Panics if `range` exceeds the view.
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end && range.end <= self.len());
        match &self.0 {
            Repr::Contiguous(bytes) => Self(Repr::Contiguous(bytes.slice(range))),
            Repr::Chunked {
                chunks,
                range: current,
            } => Self(Repr::Chunked {
                chunks: chunks.clone(),
                range: current.start + range.start..current.start + range.end,
            }),
        }
    }

    /// Returns a sub-view, or `None` if `range` exceeds the view.
    pub fn get(&self, range: Range<usize>) -> Option<Self> {
        (range.start <= range.end && range.end <= self.len()).then(|| self.slice(range))
    }

    /// Borrows `range` when it lies within a single chunk.
    ///
    /// Panics if `range` exceeds the view.
    pub fn contiguous_slice(&self, range: Range<usize>) -> Option<&[u8]> {
        assert!(range.start <= range.end && range.end <= self.len());
        let (chunks, current) = match &self.0 {
            Repr::Contiguous(bytes) => return Some(&bytes[range]),
            Repr::Chunked { chunks, range } => (chunks, range),
        };
        if range.is_empty() {
            return Some(&[]);
        }
        let start = current.start + range.start;
        let end = current.start + range.end;
        let chunk = &chunks[chunk_index(chunks, start)];
        (end <= chunk.end()).then(|| &chunk.bytes[start - chunk.start..end - chunk.start])
    }

    /// Iterates over the buffers covering the view, trimmed to its range.
    pub fn chunks(&self) -> impl Iterator<Item = OwnedBytes> + '_ {
        let (contiguous, chunked) = match &self.0 {
            Repr::Contiguous(bytes) => ((!bytes.is_empty()).then(|| bytes.clone()), None),
            Repr::Chunked { chunks, range } => (None, Some((chunks, range))),
        };
        contiguous.into_iter().chain(
            chunked
                .into_iter()
                .flat_map(|(chunks, range)| {
                    chunks[chunk_index(chunks, range.start)..]
                        .iter()
                        .take_while(move |chunk| !range.is_empty() && chunk.start < range.end)
                        .map(move |chunk| (chunk, range))
                })
                .map(|(chunk, range)| {
                    let start = range.start.saturating_sub(chunk.start);
                    let end = (range.end - chunk.start).min(chunk.bytes.len());
                    chunk.bytes.slice(start..end)
                }),
        )
    }

    fn borrowed_slices(&self, range: Range<usize>) -> impl Iterator<Item = &[u8]> {
        assert!(range.start <= range.end && range.end <= self.len());
        let (contiguous, chunked) = match &self.0 {
            Repr::Contiguous(bytes) => (Some(&bytes[range]), None),
            Repr::Chunked {
                chunks,
                range: current,
            } => (
                None,
                Some((
                    chunks,
                    current.start + range.start..current.start + range.end,
                )),
            ),
        };
        contiguous
            .into_iter()
            .chain(chunked.into_iter().flat_map(|(chunks, range)| {
                chunks[chunk_index(chunks, range.start)..]
                    .iter()
                    .take_while(move |chunk| range.start < range.end && chunk.start < range.end)
                    .map(move |chunk| {
                        let start = range.start.saturating_sub(chunk.start);
                        let end = (range.end - chunk.start).min(chunk.bytes.len());
                        &chunk.bytes[start..end]
                    })
            }))
    }

    /// Copies a range into `output` without cloning the underlying buffers.
    ///
    /// Panics if `range` exceeds the view or its length differs from `output.len()`.
    pub fn copy_to(&self, range: Range<usize>, mut output: &mut [u8]) {
        assert_eq!(range.len(), output.len());
        for slice in self.borrowed_slices(range) {
            let (head, tail) = output.split_at_mut(slice.len());
            head.copy_from_slice(slice);
            output = tail;
        }
    }

    /// Appends a range to `output` without cloning the underlying buffers.
    ///
    /// Panics if `range` exceeds the view.
    pub fn extend_into(&self, range: Range<usize>, output: &mut Vec<u8>) {
        let len = range.len();
        let slices = self.borrowed_slices(range);
        output.reserve(len);
        for slice in slices {
            output.extend_from_slice(slice);
        }
    }

    /// Returns the bytes as one buffer, copying only when they span several chunks.
    pub fn to_contiguous(&self) -> OwnedBytes {
        match &self.0 {
            Repr::Contiguous(bytes) => bytes.clone(),
            Repr::Chunked { chunks, range } => {
                if range.is_empty() {
                    return OwnedBytes::empty();
                }
                let chunk = &chunks[chunk_index(chunks, range.start)];
                if range.end <= chunk.end() {
                    return chunk
                        .bytes
                        .slice(range.start - chunk.start..range.end - chunk.start);
                }
                OwnedBytes::new(self.to_vec())
            }
        }
    }

    /// Like [`ChunkedBytes::to_contiguous`], but takes a single buffer without cloning it.
    pub fn into_contiguous(self) -> OwnedBytes {
        match self.0 {
            Repr::Contiguous(bytes) => bytes,
            Repr::Chunked { .. } => self.to_contiguous(),
        }
    }

    /// Copies the bytes into a new vector.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut contents = Vec::with_capacity(self.len());
        self.extend_into(0..self.len(), &mut contents);
        contents
    }

    /// Returns a reader over the view.
    pub fn into_reader(self) -> ChunkedBytesReader {
        let index = match &self.0 {
            Repr::Contiguous(_) => 0,
            Repr::Chunked { chunks, range } => chunk_index(chunks, range.start),
        };
        ChunkedBytesReader {
            bytes: self,
            index,
            position: 0,
        }
    }
}

impl From<OwnedBytes> for ChunkedBytes {
    fn from(bytes: OwnedBytes) -> Self {
        Self(Repr::Contiguous(bytes))
    }
}

/// A `BufRead + Read + Seek` reader over [`ChunkedBytes`] that never copies the buffers.
#[derive(Clone, Debug)]
pub struct ChunkedBytesReader {
    bytes: ChunkedBytes,
    index: usize,
    position: u64,
}

impl ChunkedBytesReader {
    fn offset(&self) -> Option<usize> {
        usize::try_from(self.position)
            .ok()
            .filter(|position| *position < self.bytes.len())
    }

    fn set_position(&mut self, position: u64) {
        self.position = position;
        let Repr::Chunked { chunks, range } = &self.bytes.0 else {
            return;
        };
        let Some(offset) = self.offset() else {
            return;
        };
        let absolute = range.start + offset;
        if !chunks
            .get(self.index)
            .is_some_and(|chunk| chunk.start <= absolute && absolute < chunk.end())
        {
            self.index = chunk_index(chunks, absolute);
        }
    }
}

impl BufRead for ChunkedBytesReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        let Some(offset) = self.offset() else {
            return Ok(&[]);
        };
        match &self.bytes.0 {
            Repr::Contiguous(bytes) => Ok(&bytes[offset..]),
            Repr::Chunked { chunks, range } => {
                let absolute = range.start + offset;
                let chunk = &chunks[self.index];
                let end = range.end.min(chunk.end());
                Ok(&chunk.bytes[absolute - chunk.start..end - chunk.start])
            }
        }
    }

    fn consume(&mut self, amount: usize) {
        self.set_position(self.position.saturating_add(amount as u64));
    }
}

impl Read for ChunkedBytesReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let input = self.fill_buf()?;
        let len = input.len().min(output.len());
        output[..len].copy_from_slice(&input[..len]);
        self.consume(len);
        Ok(len)
    }
}

impl Seek for ChunkedBytesReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let (base, offset) = match position {
            SeekFrom::Start(position) => (position, 0),
            SeekFrom::End(offset) => (self.bytes.len() as u64, offset),
            SeekFrom::Current(offset) => (self.position, offset),
        };
        let position = base.checked_add_signed(offset).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            )
        })?;
        self.set_position(position);
        Ok(position)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_remaining(reader: &mut ChunkedBytesReader) -> Vec<u8> {
        let mut output = Vec::new();
        reader.read_to_end(&mut output).unwrap();
        output
    }

    #[test]
    fn slices_and_readers_retain_original_chunks() {
        let first = OwnedBytes::new(vec![0, 1, 2]);
        let second = OwnedBytes::new(vec![3, 4, 5, 6]);
        let first_ptr = first.as_ptr();
        let second_ptr = second.as_ptr();
        let bytes = ChunkedBytes::new(vec![OwnedBytes::empty(), first, second]);
        for start in 0..=bytes.len() {
            for end in start..=bytes.len() {
                let sliced = bytes.slice(start..end);
                assert_eq!(
                    sliced.chunks().map(|chunk| chunk.len()).sum::<usize>(),
                    end - start
                );
                if start == end {
                    assert_eq!(sliced.chunks().count(), 0);
                }
                assert_eq!(bytes.get(start..end).unwrap().to_vec(), sliced.to_vec());
                assert!(bytes.get(start..bytes.len() + 1).is_none());
                assert_eq!(
                    sliced.to_vec(),
                    (start as u8..end as u8).collect::<Vec<_>>()
                );
                let within_chunk = [0, 3, bytes.len()]
                    .iter()
                    .all(|&boundary| boundary <= start || boundary >= end);
                assert_eq!(
                    sliced.contiguous_slice(0..end - start),
                    bytes.contiguous_slice(start..end)
                );
                match bytes.contiguous_slice(start..end) {
                    Some(slice) => {
                        assert!(within_chunk);
                        assert_eq!(slice, (start as u8..end as u8).collect::<Vec<_>>());
                        if let Some(chunk) = sliced.chunks().next() {
                            assert_eq!(slice.as_ptr(), chunk.as_ptr());
                        }
                    }
                    None => assert!(!within_chunk),
                }
                let contiguous = sliced.to_contiguous();
                assert_eq!(
                    contiguous.as_slice(),
                    (start as u8..end as u8).collect::<Vec<_>>()
                );
                if sliced.chunks().count() == 1 {
                    assert_eq!(
                        contiguous.as_ptr(),
                        sliced.chunks().next().unwrap().as_ptr()
                    );
                }
                assert_eq!(
                    read_remaining(&mut sliced.into_reader()),
                    (start as u8..end as u8).collect::<Vec<_>>()
                );
            }
        }
        let sliced = bytes.slice(1..6).slice(1..4);
        drop(bytes);
        let mut reader = sliced.into_reader();
        assert_eq!(
            reader.fill_buf().unwrap().as_ptr(),
            first_ptr.wrapping_add(2)
        );
        assert_eq!(reader.fill_buf().unwrap(), &[2]);
        reader.consume(1);
        assert_eq!(reader.fill_buf().unwrap().as_ptr(), second_ptr);
        assert_eq!(reader.fill_buf().unwrap(), &[3, 4]);
        reader.consume(2);
        assert!(reader.fill_buf().unwrap().is_empty());
        reader.consume(0);
        assert_eq!(reader.read(&mut [0; 4]).unwrap(), 0);
    }

    #[test]
    fn single_buffers_behave_like_chunked_buffers() {
        let data: Vec<u8> = (0..9).collect();
        let single = ChunkedBytes::new(vec![OwnedBytes::empty(), OwnedBytes::new(data.clone())]);
        let single_ptr = single.chunks().next().unwrap().as_ptr();
        let chunked = ChunkedBytes::new(vec![
            OwnedBytes::new(data[..4].to_vec()),
            OwnedBytes::new(data[4..].to_vec()),
        ]);
        assert!(ChunkedBytes::new(Vec::new()).is_empty());
        assert_eq!(
            ChunkedBytes::new(vec![OwnedBytes::empty()])
                .chunks()
                .count(),
            0
        );
        for start in 0..=data.len() {
            for end in start..=data.len() {
                let expected = &data[start..end];
                for bytes in [&single, &chunked] {
                    let sliced = bytes.slice(start..end);
                    assert_eq!(sliced.len(), expected.len());
                    assert_eq!(sliced.to_vec(), expected);
                    let mut copied = vec![0; expected.len()];
                    bytes.copy_to(start..end, &mut copied);
                    assert_eq!(copied, expected);
                    sliced.copy_to(0..sliced.len(), &mut copied);
                    assert_eq!(copied, expected);
                    let mut extended = vec![255];
                    sliced.extend_into(0..sliced.len(), &mut extended);
                    assert_eq!(&extended[1..], expected);
                    assert_eq!(extended[0], 255);
                    assert_eq!(sliced.to_contiguous().as_slice(), expected);
                    assert_eq!(sliced.clone().into_contiguous().as_slice(), expected);
                    assert_eq!(
                        sliced
                            .chunks()
                            .map(|chunk| chunk.to_vec())
                            .collect::<Vec<_>>()
                            .concat(),
                        expected
                    );
                    assert_eq!(bytes.get(start..end).unwrap().to_vec(), expected);
                    assert!(bytes.get(start..data.len() + 1).is_none());
                    let mut reader = sliced.into_reader();
                    for target in (0..=expected.len()).rev() {
                        reader.seek(SeekFrom::Start(target as u64)).unwrap();
                        assert_eq!(read_remaining(&mut reader), expected[target..]);
                    }
                }
                let sliced = single.slice(start..end);
                assert_eq!(sliced.contiguous_slice(0..end - start), Some(expected));
                assert_eq!(sliced.chunks().count(), usize::from(start < end));
                if start < end {
                    assert_eq!(
                        sliced.into_contiguous().as_ptr(),
                        single_ptr.wrapping_add(start)
                    );
                }
            }
        }
    }

    #[test]
    fn copies_reject_invalid_ranges_and_output_lengths() {
        for range in [Range { start: 2, end: 1 }, 0..4, usize::MAX..usize::MAX] {
            assert!(std::panic::catch_unwind(|| {
                let bytes =
                    ChunkedBytes::new(vec![OwnedBytes::new(vec![1, 2]), OwnedBytes::new(vec![3])]);
                bytes.copy_to(range, &mut []);
            })
            .is_err());
        }
        assert!(std::panic::catch_unwind(|| {
            let bytes = ChunkedBytes::from(OwnedBytes::new(vec![1, 2, 3]));
            bytes.copy_to(0..2, &mut [0]);
        })
        .is_err());
    }

    #[test]
    fn readers_seek_across_chunks_without_copying() {
        let chunks = [vec![0, 1, 2], vec![3], vec![4, 5, 6, 7]];
        let ptrs: Vec<_> = chunks.iter().map(|chunk| chunk.as_ptr()).collect();
        let bytes = ChunkedBytes::new(
            chunks
                .into_iter()
                .flat_map(|chunk| [OwnedBytes::empty(), OwnedBytes::new(chunk)])
                .collect(),
        );
        let chunk_starts = [0, 3, 4];
        let expected: Vec<u8> = (0..8).collect();
        for start in 0..=bytes.len() {
            for end in start..=bytes.len() {
                let len = end - start;
                let mut reader = bytes.slice(start..end).into_reader();
                for target in (0..=len + 2).rev() {
                    let previous = reader.stream_position().unwrap();
                    for seek in [
                        SeekFrom::Current(target as i64 - previous as i64),
                        SeekFrom::Start(target as u64),
                        SeekFrom::End(target as i64 - len as i64),
                    ] {
                        assert_eq!(reader.seek(seek).unwrap(), target as u64);
                        assert_eq!(reader.stream_position().unwrap(), target as u64);
                        let buffer = reader.fill_buf().unwrap();
                        if target >= len {
                            assert!(buffer.is_empty());
                            continue;
                        }
                        let absolute = start + target;
                        let chunk = chunk_starts.partition_point(|&s| s <= absolute) - 1;
                        assert_eq!(
                            buffer.as_ptr(),
                            ptrs[chunk].wrapping_add(absolute - chunk_starts[chunk])
                        );
                        assert_eq!(buffer[0], expected[absolute]);
                    }
                    assert_eq!(
                        read_remaining(&mut reader),
                        expected[(start + target).min(end)..end]
                    );
                    assert_eq!(reader.stream_position().unwrap(), target.max(len) as u64);
                }
                for invalid in [SeekFrom::End(-(len as i64) - 1), SeekFrom::Current(-1)] {
                    reader.rewind().unwrap();
                    assert_eq!(
                        reader.seek(invalid).unwrap_err().kind(),
                        io::ErrorKind::InvalidInput
                    );
                    assert_eq!(reader.stream_position().unwrap(), 0);
                }
            }
        }
    }

    #[test]
    fn readers_outlive_their_source_and_resume_after_seeking() {
        let bytes = ChunkedBytes::new(vec![
            OwnedBytes::new(vec![0, 1, 2, 3]),
            OwnedBytes::new(vec![4, 5, 6, 7]),
        ]);
        let mut reader = std::thread::spawn({
            let bytes = bytes.slice(1..7);
            move || {
                let mut reader = bytes.into_reader();
                let mut prefix = [0; 4];
                reader.read_exact(&mut prefix).unwrap();
                assert_eq!(prefix, [1, 2, 3, 4]);
                reader
            }
        })
        .join()
        .unwrap();
        drop(bytes);
        assert_eq!(reader.stream_position().unwrap(), 4);
        assert_eq!(read_remaining(&mut reader), [5, 6]);
        reader.seek(SeekFrom::Current(-3)).unwrap();
        let mut lines = Vec::new();
        assert_eq!(reader.read_until(5, &mut lines).unwrap(), 2);
        assert_eq!(lines, [4, 5]);
        reader.seek(SeekFrom::Start(u64::MAX)).unwrap();
        assert!(read_remaining(&mut reader).is_empty());
        assert!(reader.seek(SeekFrom::Current(1)).is_err());
    }
}
