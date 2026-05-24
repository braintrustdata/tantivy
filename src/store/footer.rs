use std::io;

use common::{BinarySerializable, FixedSize, HasLen};

use super::{Decompressor, DOC_STORE_VERSION};
use crate::directory::FileSlice;

const DOC_STORE_EXTENSION_MAGIC: &[u8; 4] = b"STEX";
const DOC_STORE_EXTENSION_VERSION: u8 = 1;
const DOC_STORE_EXTENSION_OFFSET_RANGE: std::ops::Range<usize> = 4..12;
const DOC_STORE_RESERVED_BYTES: usize = 15;

#[derive(Debug, Clone, PartialEq)]
pub struct DocStoreFooter {
    pub offset: u64,
    pub decompressor: Decompressor,
    pub extension_offset: Option<u64>,
}

/// Serialises the footer to a byte-array
/// - offset : 8 bytes
/// - compressor id: 1 byte
/// - reserved for future use: 15 bytes
impl BinarySerializable for DocStoreFooter {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        BinarySerializable::serialize(&DOC_STORE_VERSION, writer)?;
        BinarySerializable::serialize(&self.offset, writer)?;
        BinarySerializable::serialize(&self.decompressor.get_id(), writer)?;
        if let Some(extension_offset) = self.extension_offset {
            writer.write_all(DOC_STORE_EXTENSION_MAGIC)?;
            BinarySerializable::serialize(&extension_offset, writer)?;
            BinarySerializable::serialize(&DOC_STORE_EXTENSION_VERSION, writer)?;
            writer.write_all(&[0; 2])?;
        } else {
            writer.write_all(&[0; DOC_STORE_RESERVED_BYTES])?;
        }
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let doc_store_version = u32::deserialize(reader)?;
        if doc_store_version != DOC_STORE_VERSION {
            panic!("actual doc store version: {doc_store_version}, expected: {DOC_STORE_VERSION}");
        }
        let offset = u64::deserialize(reader)?;
        let compressor_id = u8::deserialize(reader)?;
        let mut reserved = [0; DOC_STORE_RESERVED_BYTES];
        reader.read_exact(&mut reserved)?;
        let extension_offset = if reserved.iter().all(|byte| *byte == 0) {
            None
        } else {
            if &reserved[..DOC_STORE_EXTENSION_MAGIC.len()] != DOC_STORE_EXTENSION_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "doc store footer contains unknown extension metadata",
                ));
            }
            if reserved[12] != DOC_STORE_EXTENSION_VERSION {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "actual doc store extension version: {}, expected: {DOC_STORE_EXTENSION_VERSION}",
                        reserved[12]
                    ),
                ));
            }
            let mut extension_offset_bytes = &reserved[DOC_STORE_EXTENSION_OFFSET_RANGE][..];
            Some(<u64 as BinarySerializable>::deserialize(
                &mut extension_offset_bytes,
            )?)
        };
        Ok(DocStoreFooter {
            offset,
            decompressor: Decompressor::from_id(compressor_id),
            extension_offset,
        })
    }
}

impl FixedSize for DocStoreFooter {
    const SIZE_IN_BYTES: usize = 28;
}

impl DocStoreFooter {
    pub fn new(offset: u64, decompressor: Decompressor) -> Self {
        DocStoreFooter {
            offset,
            decompressor,
            extension_offset: None,
        }
    }

    pub fn with_extension_offset(
        offset: u64,
        decompressor: Decompressor,
        extension_offset: u64,
    ) -> Self {
        DocStoreFooter {
            offset,
            decompressor,
            extension_offset: Some(extension_offset),
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
    assert_eq!(DocStoreFooter::SIZE_IN_BYTES, 28);
}
