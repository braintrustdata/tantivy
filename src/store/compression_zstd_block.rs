use std::io;

use zstd::bulk::{compress_to_buffer, decompress_to_buffer, Compressor, Decompressor};
use zstd::DEFAULT_COMPRESSION_LEVEL;

#[inline]
pub fn compress(
    uncompressed: &[u8],
    compressed: &mut Vec<u8>,
    compression_level: Option<i32>,
    dictionary: Option<&[u8]>,
) -> io::Result<()> {
    let count_size = std::mem::size_of::<u32>();
    let max_size = zstd::zstd_safe::compress_bound(uncompressed.len()) + count_size;

    compressed.clear();
    compressed.resize(max_size, 0);

    let level = compression_level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let compressed_size = if let Some(dict) = dictionary {
        // Blocks compressed against a dictionary carry a zstd frame checksum. It's verified
        // automatically by the decompressor (a few bytes + a fast XXH64 pass per block) and is
        // what catches decompressing with a missing/mismatched/corrupted dictionary -- cheaper
        // than separately hashing the (multi-megabyte) dictionary itself on every doc store open.
        let mut compressor = Compressor::with_dictionary(level, dict)?;
        compressor.set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true))?;
        compressor.compress_to_buffer(uncompressed, &mut compressed[count_size..])?
    } else {
        compress_to_buffer(uncompressed, &mut compressed[count_size..], level)?
    };

    compressed[0..count_size].copy_from_slice(&(uncompressed.len() as u32).to_le_bytes());
    compressed.resize(compressed_size + count_size, 0);

    Ok(())
}

/// Compresses a whole (small-ish, memory-resident) blob for standalone storage -- used for the
/// docstore dictionary file itself, not the block format above (no length-prefix framing, since
/// there's no skip-index seeking into this file, just one atomic_read/atomic_write).
pub fn compress_whole(bytes: &[u8]) -> io::Result<Vec<u8>> {
    zstd::stream::encode_all(bytes, DEFAULT_COMPRESSION_LEVEL)
}

/// Inverse of [`compress_whole`].
pub fn decompress_whole(bytes: &[u8]) -> io::Result<Vec<u8>> {
    zstd::stream::decode_all(bytes)
}

#[inline]
pub fn decompress(
    compressed: &[u8],
    decompressed: &mut Vec<u8>,
    dictionary: Option<&[u8]>,
) -> io::Result<()> {
    let count_size = std::mem::size_of::<u32>();
    let uncompressed_size = u32::from_le_bytes(
        compressed
            .get(..count_size)
            .ok_or(io::ErrorKind::InvalidData)?
            .try_into()
            .unwrap(),
    ) as usize;

    decompressed.clear();
    decompressed.resize(uncompressed_size, 0);

    let decompressed_size = if let Some(dict) = dictionary {
        Decompressor::with_dictionary(dict)?
            .decompress_to_buffer(&compressed[count_size..], decompressed)?
    } else {
        decompress_to_buffer(&compressed[count_size..], decompressed)?
    };

    if decompressed_size != uncompressed_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "doc store block not completely decompressed, data corruption".to_string(),
        ));
    }

    Ok(())
}
