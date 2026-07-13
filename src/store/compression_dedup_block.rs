use std::io;

use rustc_hash::FxHashMap;

use super::footer::CompressionDictionaryFooter;

const BLOCK_MAGIC: &[u8; 4] = b"TDD1";
const DICTIONARY_MAGIC: &[u8; 8] = b"TDDICT1\0";
const VERSION: u32 = 1;
const CHUNK_SIZE: usize = 1024;
const MIN_CHUNK_SIZE: usize = 64;
const TAG_LITERAL: u8 = 0;
const TAG_REF: u8 = 1;

pub(crate) struct DedupCompressor {
    dictionary: Vec<Vec<u8>>,
    dictionary_index: FxHashMap<u64, Vec<u32>>,
    frame_buffer: Vec<u8>,
    dictionary_buffer: Vec<u8>,
}

pub(crate) struct DedupDecompressor {
    dictionary: Vec<Vec<u8>>,
}

impl DedupCompressor {
    pub(crate) fn new() -> Self {
        Self {
            dictionary: Vec::new(),
            dictionary_index: FxHashMap::default(),
            frame_buffer: Vec::new(),
            dictionary_buffer: Vec::new(),
        }
    }

    pub(crate) fn compress_block(
        &mut self,
        uncompressed: &[u8],
        compressed: &mut Vec<u8>,
    ) -> io::Result<()> {
        self.frame_buffer.clear();
        self.frame_buffer.extend_from_slice(BLOCK_MAGIC);
        write_u32(&mut self.frame_buffer, VERSION);
        write_u32(&mut self.frame_buffer, checked_u32(uncompressed.len())?);
        write_u32(&mut self.frame_buffer, checked_u32(CHUNK_SIZE)?);
        let op_count_pos = self.frame_buffer.len();
        write_u32(&mut self.frame_buffer, 0);

        let mut op_count = 0u32;
        for chunk in uncompressed.chunks(CHUNK_SIZE) {
            if chunk.len() >= MIN_CHUNK_SIZE {
                let dict_id = self.intern(chunk)?;
                self.frame_buffer.push(TAG_REF);
                write_u32(&mut self.frame_buffer, dict_id);
            } else {
                self.frame_buffer.push(TAG_LITERAL);
                write_u32(&mut self.frame_buffer, checked_u32(chunk.len())?);
                self.frame_buffer.extend_from_slice(chunk);
            }
            op_count = op_count.checked_add(1).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "too many dedup block ops")
            })?;
        }
        self.frame_buffer[op_count_pos..op_count_pos + 4].copy_from_slice(&op_count.to_le_bytes());

        super::compression_zstd_block::compress(&self.frame_buffer, compressed, None)
    }

    pub(crate) fn serialize_dictionary(&mut self) -> io::Result<Vec<u8>> {
        self.dictionary_buffer.clear();
        self.dictionary_buffer.extend_from_slice(DICTIONARY_MAGIC);
        write_u32(&mut self.dictionary_buffer, VERSION);
        write_u32(
            &mut self.dictionary_buffer,
            checked_u32(self.dictionary.len())?,
        );
        for entry in &self.dictionary {
            write_u32(&mut self.dictionary_buffer, checked_u32(entry.len())?);
            self.dictionary_buffer.extend_from_slice(entry);
        }

        let mut compressed = Vec::new();
        super::compression_zstd_block::compress(&self.dictionary_buffer, &mut compressed, None)?;
        Ok(compressed)
    }

    fn intern(&mut self, bytes: &[u8]) -> io::Result<u32> {
        let key = dictionary_key(bytes);
        if let Some(candidate_ids) = self.dictionary_index.get(&key) {
            for &candidate_id in candidate_ids {
                if self.dictionary[candidate_id as usize] == bytes {
                    return Ok(candidate_id);
                }
            }
        }

        let id = checked_u32(self.dictionary.len())?;
        self.dictionary.push(bytes.to_vec());
        self.dictionary_index.entry(key).or_default().push(id);
        Ok(id)
    }
}

impl DedupDecompressor {
    pub(crate) fn open(compressed_dictionary: &[u8]) -> io::Result<Self> {
        let mut dictionary_bytes = Vec::new();
        super::compression_zstd_block::decompress(compressed_dictionary, &mut dictionary_bytes)?;
        let mut input = dictionary_bytes.as_slice();
        read_exact(&mut input, DICTIONARY_MAGIC)?;
        let version = read_u32(&mut input)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dedup dictionary version {version}"),
            ));
        }
        let num_entries = read_u32(&mut input)? as usize;
        let mut dictionary = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            let len = read_u32(&mut input)? as usize;
            dictionary.push(read_bytes(&mut input, len)?.to_vec());
        }
        if !input.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing bytes in dedup dictionary",
            ));
        }
        Ok(Self { dictionary })
    }

    pub(crate) fn decompress_block(
        &self,
        compressed: &[u8],
        decompressed: &mut Vec<u8>,
    ) -> io::Result<()> {
        let mut frame = Vec::new();
        super::compression_zstd_block::decompress(compressed, &mut frame)?;
        let mut input = frame.as_slice();
        read_exact(&mut input, BLOCK_MAGIC)?;
        let version = read_u32(&mut input)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported dedup block version {version}"),
            ));
        }
        let uncompressed_len = read_u32(&mut input)? as usize;
        let _chunk_size = read_u32(&mut input)?;
        let op_count = read_u32(&mut input)?;

        decompressed.clear();
        decompressed.reserve(uncompressed_len);
        for _ in 0..op_count {
            let tag = read_u8(&mut input)?;
            match tag {
                TAG_LITERAL => {
                    let len = read_u32(&mut input)? as usize;
                    decompressed.extend_from_slice(read_bytes(&mut input, len)?);
                }
                TAG_REF => {
                    let id = read_u32(&mut input)? as usize;
                    let entry = self.dictionary.get(id).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("dedup dictionary id {id} out of range"),
                        )
                    })?;
                    decompressed.extend_from_slice(entry);
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown dedup block tag {tag}"),
                    ));
                }
            }
        }
        if !input.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing bytes in dedup block",
            ));
        }
        if decompressed.len() != uncompressed_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "dedup block not completely decompressed, data corruption",
            ));
        }
        Ok(())
    }
}

pub(crate) fn dictionary_footer(
    offset: usize,
    bytes: &[u8],
) -> io::Result<CompressionDictionaryFooter> {
    Ok(CompressionDictionaryFooter {
        offset: offset as u64,
        length: checked_u32(bytes.len())?,
    })
}

fn dictionary_key(bytes: &[u8]) -> u64 {
    ((bytes.len() as u64) << 32) ^ crc32fast::hash(bytes) as u64
}

fn checked_u32(value: usize) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "dedup value exceeds u32"))
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u8(input: &mut &[u8]) -> io::Result<u8> {
    let byte = input
        .first()
        .copied()
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "dedup frame truncated"))?;
    *input = &input[1..];
    Ok(byte)
}

fn read_u32(input: &mut &[u8]) -> io::Result<u32> {
    let bytes = read_bytes(input, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_exact(input: &mut &[u8], expected: &[u8]) -> io::Result<()> {
    let bytes = read_bytes(input, expected.len())?;
    if bytes != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid dedup magic",
        ));
    }
    Ok(())
}

fn read_bytes<'a>(input: &mut &'a [u8], len: usize) -> io::Result<&'a [u8]> {
    if input.len() < len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "dedup frame truncated",
        ));
    }
    let (head, tail) = input.split_at(len);
    *input = tail;
    Ok(head)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_block_roundtrip() {
        let mut compressor = DedupCompressor::new();
        let first = b"first unique block ".repeat(128);
        let second = b"first unique block ".repeat(128);
        let mut first_compressed = Vec::new();
        let mut second_compressed = Vec::new();
        compressor
            .compress_block(&first, &mut first_compressed)
            .unwrap();
        compressor
            .compress_block(&second, &mut second_compressed)
            .unwrap();
        let dictionary = compressor.serialize_dictionary().unwrap();
        let decompressor = DedupDecompressor::open(&dictionary).unwrap();

        let mut decompressed = Vec::new();
        decompressor
            .decompress_block(&first_compressed, &mut decompressed)
            .unwrap();
        assert_eq!(decompressed, first);
        decompressor
            .decompress_block(&second_compressed, &mut decompressed)
            .unwrap();
        assert_eq!(decompressed, second);
    }
}
