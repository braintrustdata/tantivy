use std::io;

mod merger;

use std::iter::ExactSizeIterator;

use common::{write_vint_branchless, VInt};
use sstable::value::{ValueReader, ValueWriter};
use sstable::SSTable;
use tantivy_fst::automaton::AlwaysMatch;

pub use self::merger::TermMerger;
use crate::postings::TermInfo;

/// The term dictionary contains all of the terms in
/// `tantivy index` in a sorted manner.
///
/// The `Fst` crate is used to associate terms to their
/// respective `TermOrdinal`. The `TermInfoStore` then makes it
/// possible to fetch the associated `TermInfo`.
pub type TermDictionary = sstable::Dictionary<TermSSTable>;

/// Builder for the new term dictionary.
pub type TermDictionaryBuilder<W> = sstable::Writer<W, TermInfoValueWriter>;

/// `TermStreamer` acts as a cursor over a range of terms of a segment.
/// Terms are guaranteed to be sorted.
pub type TermStreamer<'a, A = AlwaysMatch> = sstable::Streamer<'a, TermSSTable, A>;

/// SSTable used to store TermInfo objects.
pub struct TermSSTable;

pub type TermStreamerBuilder<'a, A = AlwaysMatch> = sstable::StreamerBuilder<'a, TermSSTable, A>;

impl SSTable for TermSSTable {
    type Value = TermInfo;
    type ValueReader = TermInfoValueReader;
    type ValueWriter = TermInfoValueWriter;
}

#[derive(Default)]
pub struct TermInfoValueReader {
    term_infos: Vec<TermInfo>,
}

impl ValueReader for TermInfoValueReader {
    type Value = TermInfo;

    #[inline(always)]
    fn value(&self, idx: usize) -> &TermInfo {
        &self.term_infos[idx]
    }

    fn load(&mut self, mut data: &[u8]) -> io::Result<usize> {
        let len_before = data.len();
        self.term_infos.clear();
        let num_els = VInt::deserialize_u64(&mut data)?;
        let mut postings_start = VInt::deserialize_u64(&mut data)? as usize;
        let mut positions_start = VInt::deserialize_u64(&mut data)? as usize;
        for _ in 0..num_els {
            let doc_freq = VInt::deserialize_u64(&mut data)? as u32;
            let postings_num_bytes = VInt::deserialize_u64(&mut data)?;
            let positions_num_bytes = VInt::deserialize_u64(&mut data)?;
            let postings_end = postings_start + postings_num_bytes as usize;
            let positions_end = positions_start + positions_num_bytes as usize;
            let term_info = TermInfo {
                doc_freq,
                postings_range: postings_start..postings_end,
                positions_range: positions_start..positions_end,
            };
            self.term_infos.push(term_info);
            postings_start = postings_end;
            positions_start = positions_end;
        }
        let consumed_len = len_before - data.len();
        Ok(consumed_len)
    }
}

#[derive(Default)]
pub struct TermInfoValueWriter {
    buffer: Vec<u8>,
    num_terms: u32,
}

impl ValueWriter for TermInfoValueWriter {
    type Value = TermInfo;

    #[inline(always)]
    fn write(&mut self, term_info: &TermInfo) {
        if self.buffer.capacity() - self.buffer.len() < 56 {
            self.buffer.reserve(1024);
        }
        unsafe {
            let base = self.buffer.as_mut_ptr();
            let mut ptr = base.add(self.buffer.len());
            if self.num_terms == 0 {
                ptr = write_vint_branchless(term_info.postings_range.start as u64, ptr);
                ptr = write_vint_branchless(term_info.positions_range.start as u64, ptr);
            }
            let df = u64::from(term_info.doc_freq);
            let post_len = term_info.postings_range.len() as u64;
            let pos_len = term_info.positions_range.len() as u64;
            if (df | post_len | pos_len) < 128 {
                let packed =
                    df as u32 | ((post_len as u32) << 8) | ((pos_len as u32) << 16) | 0x0080_8080;
                ptr.cast::<u32>().write_unaligned(packed.to_le());
                ptr = ptr.add(3);
            } else {
                ptr = write_vint_branchless(df, ptr);
                ptr = write_vint_branchless(post_len, ptr);
                ptr = write_vint_branchless(pos_len, ptr);
            }
            self.buffer.set_len(ptr.offset_from(base) as usize);
        }
        self.num_terms = self
            .num_terms
            .checked_add(1)
            .expect("Too many terms in block");
    }

    #[inline(always)]
    fn serialize_block(&self, output: &mut Vec<u8>) {
        VInt(u64::from(self.num_terms)).serialize_into_vec(output);
        output.extend_from_slice(&self.buffer);
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.buffer.clear();
        self.num_terms = 0;
    }
}

#[cfg(test)]
mod tests {
    use sstable::value::{ValueReader, ValueWriter};

    use crate::postings::TermInfo;
    use crate::termdict::sstable_termdict::TermInfoValueReader;

    #[test]
    fn test_block_terminfos() {
        let mut term_info_writer = super::TermInfoValueWriter::default();
        term_info_writer.write(&TermInfo {
            doc_freq: 120u32,
            postings_range: 17..45,
            positions_range: 10..122,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 10u32,
            postings_range: 45..450,
            positions_range: 122..1100,
        });
        term_info_writer.write(&TermInfo {
            doc_freq: 17u32,
            postings_range: 450..462,
            positions_range: 1100..1302,
        });
        let mut buffer = Vec::new();
        term_info_writer.serialize_block(&mut buffer);
        let mut term_info_reader = TermInfoValueReader::default();
        let num_bytes: usize = term_info_reader.load(&buffer[..]).unwrap();
        assert_eq!(
            term_info_reader.value(0),
            &TermInfo {
                doc_freq: 120u32,
                postings_range: 17..45,
                positions_range: 10..122
            }
        );
        assert_eq!(buffer.len(), num_bytes);
    }

    #[test]
    fn test_preencoded_term_info_matches_original_bytes() {
        fn append(value: u64, out: &mut Vec<u8>) {
            let mut bytes = [0; 10];
            let len = common::VInt(value).serialize_into(&mut bytes);
            out.extend_from_slice(&bytes[..len]);
        }
        let mut writer = super::TermInfoValueWriter::default();
        for start in [0, 127, 128, 16_384, 1u64 << 56, 1u64 << 63]
            .into_iter()
            .filter_map(|start| usize::try_from(start).ok())
        {
            for count in [0, 1, 127, 128, 1024] {
                writer.clear();
                let mut expected = vec![0xa5];
                append(count as u64, &mut expected);
                if count > 0 {
                    append(start as u64, &mut expected);
                    append(start as u64, &mut expected);
                }
                let mut post = start;
                let mut pos = start;
                for i in 0..count {
                    let df = [0, 1, 127, 128, 16_383, 16_384, u32::MAX][i % 7];
                    let post_len = [0, 1, 7, 127, 128, 16_384, 1 << 21][i % 7];
                    let pos_len = [0, 1, 127, 128, 16_384, 1 << 21, 1 << 22][i % 7];
                    writer.write(&TermInfo {
                        doc_freq: df,
                        postings_range: post..post + post_len,
                        positions_range: pos..pos + pos_len,
                    });
                    append(u64::from(df), &mut expected);
                    append(post_len as u64, &mut expected);
                    append(pos_len as u64, &mut expected);
                    post += post_len;
                    pos += pos_len;
                }
                for _ in 0..2 {
                    let mut actual = vec![0xa5];
                    writer.serialize_block(&mut actual);
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
