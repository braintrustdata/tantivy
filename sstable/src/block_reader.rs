use std::borrow::Cow;
use std::io::{self, Read};
use std::ops::Range;

use common::ChunkedBytes;
use zstd::bulk::Decompressor;

pub struct BlockReader {
    buffer: Vec<u8>,
    decompressor: Option<Decompressor<'static>>,
    reader: ChunkedBytes,
    position: usize,
    offset: usize,
}

impl BlockReader {
    pub fn new(reader: impl Into<ChunkedBytes>) -> BlockReader {
        BlockReader {
            buffer: Vec::new(),
            decompressor: None,
            reader: reader.into(),
            position: 0,
            offset: 0,
        }
    }

    pub(crate) fn reset(&mut self, reader: impl Into<ChunkedBytes>) {
        self.buffer.clear();
        self.reader = reader.into();
        self.position = 0;
        self.offset = 0;
    }

    fn remaining(&self) -> usize {
        self.reader.len() - self.position
    }

    fn take_range(&mut self, len: usize) -> Range<usize> {
        let range = self.position..self.position + len;
        self.position = range.end;
        range
    }

    fn take_array<const N: usize>(&mut self) -> [u8; N] {
        let range = self.take_range(N);
        let mut bytes = [0u8; N];
        match self.reader.contiguous_slice(range.clone()) {
            Some(slice) => bytes.copy_from_slice(slice),
            None => self
                .reader
                .slice(range)
                .into_reader()
                .read_exact(&mut bytes)
                .expect("the taken range has exactly N bytes"),
        }
        bytes
    }

    pub fn deserialize_u64(&mut self) -> u64 {
        let (num_bytes, val) = super::vint::deserialize_read(self.buffer());
        self.advance(num_bytes);
        val
    }

    #[inline(always)]
    pub fn buffer_from_to(&self, range: Range<usize>) -> &[u8] {
        &self.buffer[range]
    }

    pub fn read_block(&mut self) -> io::Result<bool> {
        self.offset = 0;
        self.buffer.clear();

        let block_len = match self.remaining() {
            0 => return Ok(false),
            1..=3 => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to read block_len",
                ))
            }
            _ => u32::from_le_bytes(self.take_array()) as usize,
        };
        if block_len <= 1 {
            return Ok(false);
        }
        let [compress] = self.take_array();
        let block_len = block_len - 1;

        if self.remaining() < block_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to read block content",
            ));
        }
        let block = self.take_range(block_len);
        if compress == 1 {
            let compressed = match self.reader.contiguous_slice(block.clone()) {
                Some(compressed) => Cow::Borrowed(compressed),
                None => Cow::Owned(self.reader.slice(block).to_vec()),
            };
            let required_capacity = Decompressor::upper_bound(&compressed).unwrap_or(1024 * 1024);
            self.buffer.reserve(required_capacity);
            let decompressor = match &mut self.decompressor {
                Some(decompressor) => decompressor,
                slot @ None => slot.insert(Decompressor::new()?),
            };
            decompressor.decompress_to_buffer(&compressed[..], &mut self.buffer)?;
        } else if let Some(block) = self.reader.contiguous_slice(block.clone()) {
            self.buffer.extend_from_slice(block);
        } else {
            self.buffer.resize(block_len, 0u8);
            self.reader
                .slice(block)
                .into_reader()
                .read_exact(&mut self.buffer[..])?;
        }

        Ok(true)
    }

    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline(always)]
    pub fn advance(&mut self, num_bytes: usize) {
        self.offset += num_bytes;
    }

    #[inline(always)]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer[self.offset..]
    }
}

impl io::Read for BlockReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let len = self.buffer().read(buf)?;
        self.advance(len);
        Ok(len)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let len = self.buffer.len();
        buf.extend_from_slice(self.buffer());
        self.advance(len);
        Ok(len)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.buffer().read_exact(buf)?;
        self.advance(buf.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use common::OwnedBytes;

    use super::*;

    fn block(data: &[u8], compressed: bool) -> OwnedBytes {
        let payload = if compressed {
            zstd::bulk::compress(data, 3).unwrap()
        } else {
            data.to_vec()
        };
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&u32::try_from(payload.len() + 1).unwrap().to_le_bytes());
        encoded.push(u8::from(compressed));
        encoded.extend_from_slice(&payload);
        OwnedBytes::new(encoded)
    }

    #[test]
    fn test_reset_reuses_storage_across_independent_blocks() {
        let large = vec![b'a'; 32_768];
        let mut reader = BlockReader::new(block(&large, true));
        assert!(reader.read_block().unwrap());
        assert_eq!(reader.buffer(), large);
        let capacity = reader.buffer.capacity();
        let buffer_ptr = reader.buffer.as_ptr();
        reader.advance(17);

        for (data, compressed) in [
            (b"plain".as_slice(), false),
            (b"different".as_slice(), true),
        ] {
            reader.reset(block(data, compressed));
            assert!(reader.buffer().is_empty());
            assert_eq!(reader.offset(), 0);
            assert!(reader.read_block().unwrap());
            assert_eq!(reader.buffer(), data);
            assert_eq!(reader.buffer.capacity(), capacity);
            assert_eq!(reader.buffer.as_ptr(), buffer_ptr);
            assert!(reader.decompressor.is_some());
            assert!(!reader.read_block().unwrap());
        }

        reader.reset(OwnedBytes::empty());
        assert!(!reader.read_block().unwrap());
        reader.reset(block(&large, true));
        assert!(reader.read_block().unwrap());
        assert_eq!(reader.buffer(), large);
    }

    #[test]
    fn test_blocks_split_across_chunks_read_like_contiguous_blocks() {
        let blocks = [
            (b"plain block".to_vec(), false),
            ((0..4096).map(|i| (i % 7) as u8).collect::<Vec<_>>(), true),
            (b"tail".to_vec(), false),
        ];
        let encoded: Vec<u8> = blocks
            .iter()
            .flat_map(|(data, compressed)| block(data, *compressed).to_vec())
            .collect();
        for chunk_len in [1, 2, 3, 5, 7, 64, 1000, encoded.len()] {
            let mut reader = BlockReader::new(ChunkedBytes::new(
                encoded
                    .chunks(chunk_len)
                    .map(|chunk| OwnedBytes::new(chunk.to_vec()))
                    .collect(),
            ));
            for (data, _) in &blocks {
                assert!(reader.read_block().unwrap(), "chunk_len={chunk_len}");
                assert_eq!(reader.buffer(), data.as_slice(), "chunk_len={chunk_len}");
            }
            assert!(!reader.read_block().unwrap());
        }
    }

    #[test]
    fn test_reset_after_decompression_error() {
        let mut invalid = block(b"not a zstd frame", false).to_vec();
        invalid[4] = 1;
        let mut reader = BlockReader::new(OwnedBytes::new(invalid));
        assert!(reader.read_block().is_err());
        reader.reset(block(b"valid independent frame", true));
        assert!(reader.read_block().unwrap());
        assert_eq!(reader.buffer(), b"valid independent frame");
    }
}
