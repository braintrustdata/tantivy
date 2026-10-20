use std::io::{self, Read};
use std::ops::Range;

use common::OwnedBytes;
use zstd::bulk::Decompressor;

pub struct BlockReader {
    buffer: Vec<u8>,
    reader: OwnedBytes,
    offset: usize,
    decompressor: Option<Decompressor<'static>>,
}

impl BlockReader {
    pub fn new(reader: OwnedBytes) -> BlockReader {
        BlockReader {
            buffer: Vec::new(),
            reader,
            offset: 0,
            decompressor: None,
        }
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

        let block_len = match self.reader.len() {
            0 => return Ok(false),
            1..=3 => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to read block_len",
                ))
            }
            _ => self.reader.read_u32() as usize,
        };
        if block_len <= 1 {
            return Ok(false);
        }
        let compress = self.reader.read_u8();
        let block_len = block_len - 1;

        if self.reader.len() < block_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "failed to read block content",
            ));
        }
        if compress == 1 {
            let required_capacity =
                Decompressor::upper_bound(&self.reader[..block_len]).unwrap_or(1024 * 1024);
            self.buffer.reserve(required_capacity);
            let decompressor = match &mut self.decompressor {
                Some(decompressor) => decompressor,
                slot @ None => slot.insert(Decompressor::new()?),
            };
            decompressor.decompress_to_buffer(&self.reader[..block_len], &mut self.buffer)?;

            self.reader.advance(block_len);
        } else {
            self.buffer.resize(block_len, 0u8);
            self.reader.read_exact(&mut self.buffer[..])?;
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
    use super::*;

    #[test]
    fn test_read_mixed_blocks_with_reused_decompressor() -> io::Result<()> {
        let blocks = [
            (b"raw".repeat(8), false),
            (b"first compressed block".repeat(1024), true),
            (b"raw middle".repeat(8), false),
            (b"larger compressed block".repeat(4096), true),
            (b"small compressed block".repeat(128), true),
        ];
        let mut encoded = Vec::new();
        for (block, compressed) in &blocks {
            let payload = if *compressed {
                zstd::bulk::compress(block, 3)?
            } else {
                block.clone()
            };
            encoded.extend_from_slice(&(payload.len() as u32 + 1).to_le_bytes());
            encoded.push(u8::from(*compressed));
            encoded.extend_from_slice(&payload);
        }

        let mut reader = BlockReader::new(OwnedBytes::new(encoded));
        assert!(reader.decompressor.is_none());
        for (i, (block, _)) in blocks.iter().enumerate() {
            assert!(reader.read_block()?);
            assert_eq!(reader.buffer(), block);
            assert_eq!(reader.decompressor.is_some(), i > 0);
            reader.advance(block.len());
        }
        assert!(!reader.read_block()?);
        Ok(())
    }

    #[test]
    fn test_invalid_compressed_block_returns_error() {
        let mut reader = BlockReader::new(OwnedBytes::new(vec![4, 0, 0, 0, 1, 0, 0, 0]));
        assert!(reader.read_block().is_err());
    }
}
