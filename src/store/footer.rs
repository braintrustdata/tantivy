use std::io;

use common::{BinarySerializable, FixedSize, HasLen};

use super::{Decompressor, DOC_STORE_VERSION};
use crate::directory::FileSlice;

#[derive(Debug, Clone, PartialEq)]
pub struct DocStoreFooter {
    pub offset: u64,
    pub decompressor: Decompressor,
    /// Content hash (see `ZstdDictionaryDescriptor::hash_bytes`) of the zstd dictionary this
    /// doc store was compressed with, or 0 if none. This intentionally does *not* bump
    /// `DOC_STORE_VERSION`: every doc store ever written before this field existed has these
    /// bytes zero-filled already (they were reserved-but-unused), so 0 continues to mean
    /// exactly what it always meant here -- no dictionary.
    pub dictionary_content_hash: u64,
}

/// Serialises the footer to a byte-array
/// - offset : 8 bytes
/// - compressor id: 1 byte
/// - dictionary content hash (0 if none): 8 bytes
/// - reserved for future use: 7 bytes
impl BinarySerializable for DocStoreFooter {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        BinarySerializable::serialize(&DOC_STORE_VERSION, writer)?;
        BinarySerializable::serialize(&self.offset, writer)?;
        BinarySerializable::serialize(&self.decompressor.get_id(), writer)?;
        BinarySerializable::serialize(&self.dictionary_content_hash, writer)?;
        writer.write_all(&[0; 7])?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let doc_store_version = u32::deserialize(reader)?;
        if doc_store_version != DOC_STORE_VERSION {
            panic!("actual doc store version: {doc_store_version}, expected: {DOC_STORE_VERSION}");
        }
        let offset = u64::deserialize(reader)?;
        let compressor_id = u8::deserialize(reader)?;
        let dictionary_content_hash = u64::deserialize(reader)?;
        let mut skip_buf = [0; 7];
        reader.read_exact(&mut skip_buf)?;
        Ok(DocStoreFooter {
            offset,
            decompressor: Decompressor::from_id(compressor_id),
            dictionary_content_hash,
        })
    }
}

impl FixedSize for DocStoreFooter {
    const SIZE_IN_BYTES: usize = 28;
}

impl DocStoreFooter {
    pub fn new(offset: u64, decompressor: Decompressor, dictionary_content_hash: u64) -> Self {
        DocStoreFooter {
            offset,
            decompressor,
            dictionary_content_hash,
        }
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
    assert_eq!(core::mem::size_of::<DocStoreFooter>(), 24);
}
