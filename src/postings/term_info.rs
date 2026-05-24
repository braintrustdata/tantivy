use std::io;
use std::ops::Range;

use common::{BinarySerializable, FixedSize};

/// `TermInfo` wraps the metadata associated with a Term.
/// It is segment-local.
#[derive(Debug, Default, Eq, PartialEq, Clone)]
pub struct TermInfo {
    /// Number of documents in the segment containing the term
    pub doc_freq: u32,
    /// Byte range of this term within the postings (`.idx`) file.
    ///
    /// This range may include codec extension bytes at the tail. Use
    /// [`Self::normal_postings_range`] when reading Tantivy's standard postings
    /// list.
    pub postings_range: Range<usize>,
    /// Byte range of codec extension bytes stored at the tail of
    /// [`Self::postings_range`].
    pub repeated_postings_range: Range<usize>,
    /// Byte range of the positions of this terms in the positions (`.pos`) file.
    pub positions_range: Range<usize>,
}

impl TermInfo {
    /// Returns the byte range for Tantivy's standard postings list, excluding
    /// codec extension bytes appended to the term.
    pub fn normal_postings_range(&self) -> Range<usize> {
        if self.repeated_postings_range.end == self.postings_range.end
            && self.repeated_postings_range.start >= self.postings_range.start
            && self.repeated_postings_range.start <= self.postings_range.end
        {
            self.postings_range.start..self.repeated_postings_range.start
        } else {
            self.postings_range.clone()
        }
    }

    pub(crate) fn posting_num_bytes(&self) -> u32 {
        let num_bytes = self.postings_range.len();
        assert!(num_bytes <= u32::MAX as usize);
        num_bytes as u32
    }

    pub(crate) fn repeated_postings_start_offset(&self) -> u64 {
        assert!(self.repeated_postings_range.start >= self.postings_range.start);
        assert!(self.repeated_postings_range.start <= self.postings_range.end);
        assert_eq!(self.repeated_postings_range.end, self.postings_range.end);
        self.repeated_postings_range.start as u64
    }

    pub(crate) fn positions_num_bytes(&self) -> u32 {
        let num_bytes = self.positions_range.len();
        assert!(num_bytes <= u32::MAX as usize);
        num_bytes as u32
    }
}

impl FixedSize for TermInfo {
    /// Size required for the binary serialization of a `TermInfo` object.
    /// This is large, but in practise, `TermInfo` are encoded in blocks and
    /// only the first `TermInfo` of a block is serialized uncompressed.
    /// The subsequent `TermInfo` are delta encoded and bitpacked.
    const SIZE_IN_BYTES: usize = 3 * u32::SIZE_IN_BYTES + 3 * u64::SIZE_IN_BYTES;
}

impl BinarySerializable for TermInfo {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        self.doc_freq.serialize(writer)?;
        (self.postings_range.start as u64).serialize(writer)?;
        self.posting_num_bytes().serialize(writer)?;
        self.repeated_postings_start_offset().serialize(writer)?;
        (self.positions_range.start as u64).serialize(writer)?;
        self.positions_num_bytes().serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let doc_freq = u32::deserialize(reader)?;
        let postings_start_offset = u64::deserialize(reader)? as usize;
        let postings_num_bytes = u32::deserialize(reader)? as usize;
        let postings_end_offset = postings_start_offset + postings_num_bytes;
        let repeated_postings_start_offset = u64::deserialize(reader)? as usize;
        let positions_start_offset = u64::deserialize(reader)? as usize;
        let positions_num_bytes = u32::deserialize(reader)? as usize;
        let positions_end_offset = positions_start_offset + positions_num_bytes;
        Ok(TermInfo {
            doc_freq,
            postings_range: postings_start_offset..postings_end_offset,
            repeated_postings_range: repeated_postings_start_offset..postings_end_offset,
            positions_range: positions_start_offset..positions_end_offset,
        })
    }
}

#[cfg(test)]
mod tests {

    use super::TermInfo;
    use crate::tests::fixed_size_test;

    // TODO add serialize/deserialize test for terminfo

    #[test]
    fn test_fixed_size() {
        fixed_size_test::<TermInfo>();
    }
}
