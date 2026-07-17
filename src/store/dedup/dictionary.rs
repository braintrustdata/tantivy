use std::io;

/// Magic prefixing the raw (decompressed) dictionary payload.
const DICTIONARY_MAGIC: &[u8; 8] = b"TDDICT1\0";
/// On-disk dictionary format version.
const VERSION: u32 = 1;

/// A per-segment dictionary of repeated byte sequences used by the `Dedup` docstore codec.
///
/// Entry ids are the indices `0..len()`. The serialized form is the `TDDICT1` container
/// consumed by the `tantivy-cli dump-dedup-dictionary` reader: a zstd block
/// (`[u32 LE uncompressed_size][zstd bulk]`) wrapping
/// `magic ++ version:u32 ++ num_entries:u32 ++ (len:u32 ++ bytes)*`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DedupDictionary {
    entries: Vec<Vec<u8>>,
}

impl DedupDictionary {
    /// Build a dictionary from its entries, in id order.
    pub fn new(entries: Vec<Vec<u8>>) -> DedupDictionary {
        DedupDictionary { entries }
    }

    /// The entries, indexed by id.
    pub fn entries(&self) -> &[Vec<u8>] {
        &self.entries
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the dictionary has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The bytes for `id`, or `None` if out of range.
    pub fn get(&self, id: usize) -> Option<&[u8]> {
        self.entries.get(id).map(Vec::as_slice)
    }

    /// Serialize into the `TDDICT1` container.
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut raw = Vec::new();
        raw.extend_from_slice(DICTIONARY_MAGIC);
        raw.extend_from_slice(&VERSION.to_le_bytes());
        let num_entries: u32 = self
            .entries
            .len()
            .try_into()
            .map_err(|_| invalid("too many dedup dictionary entries"))?;
        raw.extend_from_slice(&num_entries.to_le_bytes());
        for entry in &self.entries {
            let len: u32 = entry
                .len()
                .try_into()
                .map_err(|_| invalid("dedup dictionary entry too large"))?;
            raw.extend_from_slice(&len.to_le_bytes());
            raw.extend_from_slice(entry);
        }

        let mut compressed = Vec::new();
        super::super::compression_zstd_block::compress(&raw, &mut compressed, None)?;
        Ok(compressed)
    }

    /// Parse a `TDDICT1` container produced by [`serialize`](Self::serialize).
    pub fn deserialize(bytes: &[u8]) -> io::Result<DedupDictionary> {
        let mut raw = Vec::new();
        super::super::compression_zstd_block::decompress(bytes, &mut raw)?;

        let mut input: &[u8] = &raw;
        read_expected(&mut input, DICTIONARY_MAGIC)?;
        let version = read_u32(&mut input)?;
        if version != VERSION {
            return Err(invalid(&format!(
                "unsupported dedup dictionary version {version}"
            )));
        }
        let num_entries = read_u32(&mut input)? as usize;
        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            let len = read_u32(&mut input)? as usize;
            entries.push(read_bytes(&mut input, len)?.to_vec());
        }
        if !input.is_empty() {
            return Err(invalid("trailing bytes in dedup dictionary"));
        }
        Ok(DedupDictionary { entries })
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn read_bytes<'a>(input: &mut &'a [u8], len: usize) -> io::Result<&'a [u8]> {
    if input.len() < len {
        return Err(invalid("dedup dictionary is truncated"));
    }
    let (head, tail) = input.split_at(len);
    *input = tail;
    Ok(head)
}

fn read_u32(input: &mut &[u8]) -> io::Result<u32> {
    let bytes = read_bytes(input, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_expected(input: &mut &[u8], expected: &[u8]) -> io::Result<()> {
    if read_bytes(input, expected.len())? != expected {
        return Err(invalid("invalid dedup dictionary magic"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(entries: Vec<Vec<u8>>) {
        let dict = DedupDictionary::new(entries);
        let bytes = dict.serialize().unwrap();
        assert_eq!(DedupDictionary::deserialize(&bytes).unwrap(), dict);
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(vec![]);
    }

    #[test]
    fn roundtrip_single() {
        roundtrip(vec![b"\"model\":\"gpt-5-mini\"".to_vec()]);
    }

    #[test]
    fn roundtrip_multi_and_binary() {
        roundtrip(vec![
            b"".to_vec(),
            b"\"model\":\"gpt-5-mini\"".to_vec(),
            vec![0x00, 0xFE, 0xFF, 0x01, 0xFE],
            vec![0xFE; 1024],
        ]);
    }

    #[test]
    fn accessors() {
        let dict = DedupDictionary::new(vec![b"a".to_vec(), b"bb".to_vec()]);
        assert_eq!(dict.len(), 2);
        assert!(!dict.is_empty());
        assert_eq!(dict.get(0), Some(b"a".as_slice()));
        assert_eq!(dict.get(1), Some(b"bb".as_slice()));
        assert_eq!(dict.get(2), None);
        assert!(DedupDictionary::default().is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut raw = b"XXXXXXXX".to_vec();
        raw.extend_from_slice(&VERSION.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        let mut compressed = Vec::new();
        super::super::super::compression_zstd_block::compress(&raw, &mut compressed, None).unwrap();
        assert!(DedupDictionary::deserialize(&compressed).is_err());
    }

    #[test]
    fn rejects_bad_version() {
        let mut raw = DICTIONARY_MAGIC.to_vec();
        raw.extend_from_slice(&2u32.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        let mut compressed = Vec::new();
        super::super::super::compression_zstd_block::compress(&raw, &mut compressed, None).unwrap();
        let err = DedupDictionary::deserialize(&compressed).unwrap_err();
        assert!(err.to_string().contains("version 2"));
    }

    #[test]
    fn rejects_truncated_entry() {
        let mut raw = DICTIONARY_MAGIC.to_vec();
        raw.extend_from_slice(&VERSION.to_le_bytes());
        raw.extend_from_slice(&1u32.to_le_bytes());
        raw.extend_from_slice(&16u32.to_le_bytes());
        raw.extend_from_slice(b"only-4");
        let mut compressed = Vec::new();
        super::super::super::compression_zstd_block::compress(&raw, &mut compressed, None).unwrap();
        assert!(DedupDictionary::deserialize(&compressed).is_err());
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut raw = DICTIONARY_MAGIC.to_vec();
        raw.extend_from_slice(&VERSION.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(b"extra");
        let mut compressed = Vec::new();
        super::super::super::compression_zstd_block::compress(&raw, &mut compressed, None).unwrap();
        assert!(DedupDictionary::deserialize(&compressed).is_err());
    }
}
