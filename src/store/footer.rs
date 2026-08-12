use std::io;

use common::{BinarySerializable, FixedSize, HasLen};

use super::{Decompressor, DOC_STORE_VERSION};
use crate::directory::FileSlice;

/// Bit in [`DocStoreFooter::decompressor_options`] recording whether the segment's blocks
/// were compressed against a dictionary. Merges compare this against the *current* index-wide
/// dictionary setting to decide whether raw block-stacking is safe -- see
/// `IndexMerger::write_storable_fields` in `indexer/merger.rs`.
const DICTIONARY_USED_FLAG: u8 = 1 << 0;

#[derive(Debug, Clone, PartialEq)]
pub struct DocStoreFooter {
    pub offset: u64,
    pub decompressor: Decompressor,
    pub decompressor_options: u8,
}

/// Serialises the footer to a byte-array
/// - offset : 8 bytes
/// - compressor id: 1 byte
/// - decompressor options (bit 0: dictionary used): 1 byte
/// - reserved for future use: 14 bytes
impl BinarySerializable for DocStoreFooter {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        BinarySerializable::serialize(&DOC_STORE_VERSION, writer)?;
        BinarySerializable::serialize(&self.offset, writer)?;
        BinarySerializable::serialize(&self.decompressor.get_id(), writer)?;
        BinarySerializable::serialize(&self.decompressor_options, writer)?;
        writer.write_all(&[0; 14])?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let doc_store_version = u32::deserialize(reader)?;
        if doc_store_version != DOC_STORE_VERSION {
            panic!("actual doc store version: {doc_store_version}, expected: {DOC_STORE_VERSION}");
        }
        let offset = u64::deserialize(reader)?;
        let compressor_id = u8::deserialize(reader)?;
        let decompressor_options = u8::deserialize(reader)?;
        let mut skip_buf = [0; 14];
        reader.read_exact(&mut skip_buf)?;
        Ok(DocStoreFooter {
            offset,
            decompressor: Decompressor::from_id(compressor_id),
            decompressor_options,
        })
    }
}

impl FixedSize for DocStoreFooter {
    const SIZE_IN_BYTES: usize = 28;
}

impl DocStoreFooter {
    pub fn new(offset: u64, decompressor: Decompressor, dictionary_used: bool) -> Self {
        let decompressor_options = if dictionary_used {
            DICTIONARY_USED_FLAG
        } else {
            0
        };
        DocStoreFooter {
            offset,
            decompressor,
            decompressor_options,
        }
    }

    /// Whether the segment's blocks were compressed against a dictionary at write time.
    pub fn dictionary_used(&self) -> bool {
        self.decompressor_options & DICTIONARY_USED_FLAG != 0
    }

    pub fn extract_footer(file: FileSlice) -> io::Result<(DocStoreFooter, FileSlice)> {
        if file.len() < DocStoreFooter::SIZE_IN_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "File corrupted. The file is smaller than Footer::SIZE_IN_BYTES (len={}).",
                    file.len()
                ),
            ));
        }
        let (body, footer_slice) = file.split_from_end(DocStoreFooter::SIZE_IN_BYTES);
        let mut footer_bytes = footer_slice.read_bytes()?;
        let footer = DocStoreFooter::deserialize(&mut footer_bytes)?;
        Ok((footer, body))
    }
}

#[test]
fn doc_store_footer_test() {
    // This test is just to safe guard changes on the footer.
    // When the doc store footer is updated, make sure to update also the serialize/deserialize
    // methods
    assert_eq!(core::mem::size_of::<DocStoreFooter>(), 16);
}
