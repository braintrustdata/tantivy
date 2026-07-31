use std::hash::Hasher as _;
use std::io;

use serde::{Deserialize, Deserializer, Serialize};

/// Compressor can be used on `IndexSettings` to choose
/// the compressor used to compress the doc store.
///
/// The default is Lz4Block, but also depends on the enabled feature flags.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum Compressor {
    /// No compression
    None,
    /// Use the lz4 compressor (block format)
    #[cfg(feature = "lz4-compression")]
    Lz4,
    /// Use the zstd compressor
    #[cfg(feature = "zstd-compression")]
    Zstd(ZstdCompressor),
}

impl Serialize for Compressor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        match *self {
            Compressor::None => serializer.serialize_str("none"),
            #[cfg(feature = "lz4-compression")]
            Compressor::Lz4 => serializer.serialize_str("lz4"),
            #[cfg(feature = "zstd-compression")]
            Compressor::Zstd(zstd) => serializer.serialize_str(&zstd.ser_to_string()),
        }
    }
}

impl<'de> Deserialize<'de> for Compressor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        let buf = String::deserialize(deserializer)?;
        let compressor = match buf.as_str() {
            "none" => Compressor::None,
            #[cfg(feature = "lz4-compression")]
            "lz4" => Compressor::Lz4,
            #[cfg(not(feature = "lz4-compression"))]
            "lz4" => {
                return Err(serde::de::Error::custom(
                    "unsupported variant `lz4`, please enable Tantivy's `lz4-compression` feature",
                ))
            }
            #[cfg(feature = "zstd-compression")]
            _ if buf.starts_with("zstd") => Compressor::Zstd(
                ZstdCompressor::deser_from_str(&buf).map_err(serde::de::Error::custom)?,
            ),
            #[cfg(not(feature = "zstd-compression"))]
            _ if buf.starts_with("zstd") => {
                return Err(serde::de::Error::custom(
                    "unsupported variant `zstd`, please enable Tantivy's `zstd-compression` \
                     feature",
                ))
            }
            _ => {
                return Err(serde::de::Error::unknown_variant(
                    &buf,
                    &[
                        "none",
                        #[cfg(feature = "lz4-compression")]
                        "lz4",
                        #[cfg(feature = "zstd-compression")]
                        "zstd",
                        #[cfg(feature = "zstd-compression")]
                        "zstd(compression_level=5)",
                    ],
                ));
            }
        };

        Ok(compressor)
    }
}

#[derive(Clone, Default, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// A small, self-describing reference to the zstd dictionary a doc store was compressed with.
/// The dictionary's raw bytes are not part of this descriptor (and are not part of `meta.json`
/// at all) -- callers resolve the actual bytes out-of-band from `content_hash`.
pub struct ZstdDictionaryDescriptor {
    /// Content hash of the dictionary bytes, used both to look up the dictionary out-of-band
    /// and to detect a mismatched/missing dictionary at read time.
    pub content_hash: u64,
}

impl ZstdDictionaryDescriptor {
    /// Computes the content hash used to identify a zstd dictionary's bytes. This is the single
    /// source of truth for how the hash is derived -- callers resolving a dictionary by hash
    /// (e.g. from content-addressed storage) and the doc store footer's write-time integrity
    /// check (see `store::footer`) must agree on this function.
    pub fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut hasher = fnv::FnvHasher::default();
        hasher.write(bytes);
        hasher.finish()
    }
}

#[derive(Clone, Default, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// The Zstd compressor, with optional compression level and dictionary.
pub struct ZstdCompressor {
    /// The compression level, if unset defaults to zstd::DEFAULT_COMPRESSION_LEVEL = 3
    pub compression_level: Option<i32>,
    /// If set, the doc store was (and must continue to be) compressed against this dictionary.
    #[serde(default)]
    pub dictionary: Option<ZstdDictionaryDescriptor>,
}

#[cfg(feature = "zstd-compression")]
impl ZstdCompressor {
    fn deser_from_str(val: &str) -> Result<ZstdCompressor, String> {
        if !val.starts_with("zstd") {
            return Err(format!("needs to start with zstd, but got {val}"));
        }
        if val == "zstd" {
            return Ok(ZstdCompressor::default());
        }
        let options = &val["zstd".len() + 1..val.len() - 1];

        let mut compressor = ZstdCompressor::default();
        for option in options.split(',') {
            let (opt_name, value) = option
                .split_once('=')
                .ok_or_else(|| format!("no '=' found in option {option:?}"))?;

            match opt_name {
                "compression_level" => {
                    let value = value.parse::<i32>().map_err(|err| {
                        format!("Could not parse value {value} of option {opt_name}, e: {err}")
                    })?;
                    if value >= 15 {
                        warn!(
                            "High zstd compression level detected: {:?}. High compression levels \
                             (>=15) are slow and will limit indexing speed.",
                            value
                        )
                    }
                    compressor.compression_level = Some(value);
                }
                "dictionary_hash" => {
                    let content_hash = u64::from_str_radix(value, 16).map_err(|err| {
                        format!("Could not parse value {value} of option {opt_name}, e: {err}")
                    })?;
                    compressor.dictionary = Some(ZstdDictionaryDescriptor { content_hash });
                }
                _ => {
                    return Err(format!("unknown zstd option {opt_name:?}"));
                }
            }
        }
        Ok(compressor)
    }
    fn ser_to_string(&self) -> String {
        let mut opts = Vec::new();
        if let Some(compression_level) = self.compression_level {
            opts.push(format!("compression_level={compression_level}"));
        }
        if let Some(dictionary) = &self.dictionary {
            opts.push(format!("dictionary_hash={:x}", dictionary.content_hash));
        }
        if opts.is_empty() {
            "zstd".to_string()
        } else {
            format!("zstd({})", opts.join(","))
        }
    }
}

impl Default for Compressor {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(feature = "lz4-compression")]
        return Compressor::Lz4;

        #[cfg(feature = "zstd-compression")]
        return Compressor::Zstd(ZstdCompressor::default());

        Compressor::None
    }
}

impl Compressor {
    #[inline]
    pub(crate) fn compress_into(
        &self,
        uncompressed: &[u8],
        compressed: &mut Vec<u8>,
        dictionary: Option<&[u8]>,
    ) -> io::Result<()> {
        match self {
            Self::None => {
                compressed.clear();
                compressed.extend_from_slice(uncompressed);
                Ok(())
            }
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => super::compression_lz4_block::compress(uncompressed, compressed),
            #[cfg(feature = "zstd-compression")]
            Self::Zstd(_zstd_compressor) => super::compression_zstd_block::compress(
                uncompressed,
                compressed,
                _zstd_compressor.compression_level,
                dictionary,
            ),
        }
    }
}

#[cfg(all(feature = "zstd-compression", test))]
mod tests {
    use super::*;

    #[test]
    fn zstd_serde_roundtrip() {
        let compressor = ZstdCompressor {
            compression_level: Some(15),
            dictionary: None,
        };

        assert_eq!(
            ZstdCompressor::deser_from_str(&compressor.ser_to_string()).unwrap(),
            compressor
        );

        assert_eq!(
            ZstdCompressor::deser_from_str(&ZstdCompressor::default().ser_to_string()).unwrap(),
            ZstdCompressor::default()
        );

        let compressor_with_dict = ZstdCompressor {
            compression_level: Some(15),
            dictionary: Some(ZstdDictionaryDescriptor {
                content_hash: 0xdeadbeef,
            }),
        };
        assert_eq!(
            ZstdCompressor::deser_from_str(&compressor_with_dict.ser_to_string()).unwrap(),
            compressor_with_dict
        );

        let dict_only = ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor { content_hash: 42 }),
        };
        assert_eq!(
            ZstdCompressor::deser_from_str(&dict_only.ser_to_string()).unwrap(),
            dict_only
        );
    }

    #[test]
    fn deser_zstd_test() {
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd").unwrap(),
            ZstdCompressor::default()
        );

        assert!(ZstdCompressor::deser_from_str("zzstd").is_err());
        assert!(ZstdCompressor::deser_from_str("zzstd()").is_err());
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=15)").unwrap(),
            ZstdCompressor {
                compression_level: Some(15),
                dictionary: None,
            }
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(dictionary_hash=2a)").unwrap(),
            ZstdCompressor {
                compression_level: None,
                dictionary: Some(ZstdDictionaryDescriptor { content_hash: 42 }),
            }
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=15,dictionary_hash=2a)")
                .unwrap(),
            ZstdCompressor {
                compression_level: Some(15),
                dictionary: Some(ZstdDictionaryDescriptor { content_hash: 42 }),
            }
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compresion_level=15)").unwrap_err(),
            "unknown zstd option \"compresion_level\""
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level->2)").unwrap_err(),
            "no '=' found in option \"compression_level->2\""
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=over9000)").unwrap_err(),
            "Could not parse value over9000 of option compression_level, e: invalid digit found \
             in string"
        );
    }
}
