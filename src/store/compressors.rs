use std::io;

use serde::{Deserialize, Deserializer, Serialize};

/// Compressor can be used on `IndexSettings` to choose
/// the compressor used to compress the doc store.
///
/// The default is Lz4Block, but also depends on the enabled feature flags.
#[derive(Clone, Debug, PartialEq, Eq)]
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
        match self {
            Compressor::None => serializer.serialize_str("none"),
            #[cfg(feature = "lz4-compression")]
            Compressor::Lz4 => serializer.serialize_str("lz4"),
            #[cfg(feature = "zstd-compression")]
            Compressor::Zstd(zstd) => {
                if let Some(dictionary) = &zstd.dictionary {
                    if !is_safe_and_encodable_relative_path(&dictionary.path) {
                        return Err(serde::ser::Error::custom(format!(
                            "dictionary_path must be a plain path relative to the index \
                             directory (no leading '/', no '..' components, and no ','), got \
                             {:?}",
                            dictionary.path
                        )));
                    }
                }
                serializer.serialize_str(&zstd.ser_to_string())
            }
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

/// Rejects absolute paths, `..`/`.` components (would escape the index directory via
/// `Directory::resolve_path`'s unchecked `root_path.join`), and `,` (collides with
/// `zstd(opt=val,...)`'s option separator).
#[cfg(feature = "zstd-compression")]
fn is_safe_and_encodable_relative_path(path: &str) -> bool {
    let as_path = std::path::Path::new(path);
    !as_path.is_absolute()
        && as_path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
        && !path.contains(',')
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// A small, self-describing reference to the zstd dictionary a doc store was compressed with.
/// The dictionary's raw bytes are not part of this descriptor (and are not part of `meta.json`
/// at all) -- `path` is a `Directory`-relative path (from the index root) that callers resolve
/// via `Directory::atomic_read`. There is deliberately no content hash here: a dictionary can be
/// multiple megabytes, and hashing it on every doc store open (i.e. every segment, every reader
/// reload) would be wasted work. Integrity/mismatch detection instead relies on zstd's own frame
/// checksum (see `compression_zstd_block`), which is verified as a side effect of decompression
/// at effectively no extra cost.
///
/// `SegmentUpdater::list_files` protects `path` from garbage collection for any `Directory` that
/// routes GC through it (i.e. anything wrapped in `ManagedDirectory`, which is every `Index`).
/// Directory implementations that run their own GC outside of that path are not covered and must
/// protect the dictionary file themselves.
pub struct ZstdDictionaryDescriptor {
    /// Path (relative to the index's own directory) of the file holding this dictionary's bytes,
    /// stored zstd-compressed on disk.
    pub path: String,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
                "dictionary_path" => {
                    if !is_safe_and_encodable_relative_path(value) {
                        return Err(format!(
                            "dictionary_path must be a plain path relative to the index \
                             directory (no leading '/', no '..' components, and no ','), \
                             got {value:?}"
                        ));
                    }
                    compressor.dictionary = Some(ZstdDictionaryDescriptor {
                        path: value.to_string(),
                    });
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
            opts.push(format!("dictionary_path={}", dictionary.path));
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

    /// Whether this compressor is configured with a zstd dictionary. Merges use this to fall
    /// back from raw block-stacking (which never decompresses, and so never exercises the
    /// dictionary's own zstd checksum) to a decompress/recompress path -- see
    /// `IndexMerger::write_storable_fields` in `indexer/merger.rs`.
    pub fn has_dictionary(&self) -> bool {
        match self {
            Self::None => false,
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => false,
            #[cfg(feature = "zstd-compression")]
            Self::Zstd(zstd_compressor) => zstd_compressor.dictionary.is_some(),
        }
    }

    /// The `Directory`-relative path of this compressor's dictionary, if any. Used by
    /// `SegmentUpdater::list_files` to protect the dictionary file from garbage collection --
    /// see that function for why this can't be inferred from segment metadata alone.
    pub(crate) fn dictionary_path(&self) -> Option<&str> {
        match self {
            Self::None => None,
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => None,
            #[cfg(feature = "zstd-compression")]
            Self::Zstd(zstd_compressor) => {
                zstd_compressor.dictionary.as_ref().map(|d| d.path.as_str())
            }
        }
    }

    /// If this compressor names a zstd dictionary, fetches it from `directory` at the path
    /// recorded in the descriptor (stored zstd-compressed on disk) and decompresses it. Returns
    /// `Ok(None)` if this compressor doesn't use a dictionary.
    pub fn resolve_dictionary(
        &self,
        directory: &dyn crate::Directory,
    ) -> io::Result<Option<std::sync::Arc<[u8]>>> {
        match self {
            Self::None => Ok(None),
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => Ok(None),
            #[cfg(feature = "zstd-compression")]
            Self::Zstd(zstd_compressor) => {
                let Some(descriptor) = &zstd_compressor.dictionary else {
                    return Ok(None);
                };
                // Defense in depth: `ZstdDictionaryDescriptor::path` is `pub`, so a caller can
                // construct one directly and skip `deser_from_str`'s validation. Re-check here,
                // right before it's ever handed to a `Directory`, so an absolute path or `..`
                // traversal can't escape the index directory (see `resolve_path` on
                // `MmapDirectory` et al., which is a plain, unchecked `root_path.join(..)`), and
                // so a `,` in the path can't have snuck in and desynced a subsequent
                // `ser_to_string`/`deser_from_str` round-trip.
                if !is_safe_and_encodable_relative_path(&descriptor.path) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "dictionary_path must be a plain path relative to the index \
                             directory (no leading '/', no '..' components, and no ','), got \
                             {:?}",
                            descriptor.path
                        ),
                    ));
                }
                let compressed = directory
                    .atomic_read(std::path::Path::new(&descriptor.path))
                    .map_err(|err| {
                        // Preserve `NotFound`/the original `io::Error`'s kind, rather than
                        // flattening every failure mode to `ErrorKind::Other` -- a misconfigured
                        // `dictionary_path` should be as easy to diagnose as a typical
                        // missing-file error.
                        let kind = match &err {
                            crate::directory::error::OpenReadError::FileDoesNotExist(_) => {
                                io::ErrorKind::NotFound
                            }
                            crate::directory::error::OpenReadError::IoError { io_error, .. } => {
                                io_error.kind()
                            }
                            crate::directory::error::OpenReadError::IncompatibleIndex(_) => {
                                io::ErrorKind::InvalidData
                            }
                        };
                        io::Error::new(kind, err.to_string())
                    })?;
                let bytes = super::compression_zstd_block::decompress_whole(&compressed)?;
                Ok(Some(std::sync::Arc::from(bytes)))
            }
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
                path: "dict.bin.zst".to_string(),
            }),
        };
        assert_eq!(
            ZstdCompressor::deser_from_str(&compressor_with_dict.ser_to_string()).unwrap(),
            compressor_with_dict
        );

        let dict_only = ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor {
                path: "dict.bin.zst".to_string(),
            }),
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
            ZstdCompressor::deser_from_str("zstd(dictionary_path=dict.bin.zst)").unwrap(),
            ZstdCompressor {
                compression_level: None,
                dictionary: Some(ZstdDictionaryDescriptor {
                    path: "dict.bin.zst".to_string(),
                }),
            }
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=15,dictionary_path=dict.bin.zst)")
                .unwrap(),
            ZstdCompressor {
                compression_level: Some(15),
                dictionary: Some(ZstdDictionaryDescriptor {
                    path: "dict.bin.zst".to_string(),
                }),
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

    #[test]
    fn deser_zstd_rejects_unsafe_dictionary_paths() {
        // Absolute path -- `PathBuf::join` fully replaces `root_path` when the joined path is
        // absolute, so this would otherwise read/write outside the index directory entirely.
        assert!(ZstdCompressor::deser_from_str("zstd(dictionary_path=/etc/passwd)")
            .unwrap_err()
            .contains("dictionary_path must be a plain path"));

        // `..` traversal -- passed straight through for the OS to resolve at open time.
        assert!(ZstdCompressor::deser_from_str(
            "zstd(dictionary_path=../../../../etc/passwd)"
        )
        .unwrap_err()
        .contains("dictionary_path must be a plain path"));

        // A plain relative path is still accepted.
        assert!(ZstdCompressor::deser_from_str("zstd(dictionary_path=dict.bin.zst)").is_ok());
    }

    #[test]
    fn dictionary_path_with_comma_is_rejected_not_silently_mis_encoded() {
        // A comma is a perfectly safe, non-traversing relative path component, but it collides
        // with `zstd(opt=val,opt=val)`'s `,`-based option separator. `ZstdDictionaryDescriptor`'s
        // `path` field is `pub`, so a directly-constructed descriptor can carry one in even
        // though `deser_from_str` would reject it in text form. Note this can *not* be caught by
        // adding a check inside `deser_from_str`'s "dictionary_path" arm: `options.split(',')`
        // fragments the string into separate options *before* any option-specific value is ever
        // isolated, so a value containing ',' never reaches that arm as one piece -- it instead
        // surfaces downstream as the unrelated, unhelpful `no '=' found in option "b.zst"`.
        // Serializing (e.g. for `meta.json`) is the first point after direct construction where
        // this is actually reachable and worth catching, so that's where it's checked.
        let compressor_with_comma = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor {
                path: "a,b.zst".to_string(),
            }),
        });
        let err = serde_json::to_string(&compressor_with_comma)
            .expect_err("a ',' in dictionary_path should fail to serialize, not mis-encode");
        assert!(
            err.to_string().contains("dictionary_path must be a plain path"),
            "unexpected error: {err}"
        );

        // A directly-constructed descriptor with a safe path still serializes fine.
        let safe = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor {
                path: "dict.bin.zst".to_string(),
            }),
        });
        assert!(serde_json::to_string(&safe).is_ok());
    }

    #[test]
    fn resolve_dictionary_reads_and_decompresses_from_path() {
        use crate::directory::{Directory, RamDirectory};

        let directory = RamDirectory::create();
        let raw_dict = b"a dictionary's worth of bytes, stored zstd-compressed on disk".to_vec();
        let compressed = super::super::compression_zstd_block::compress_whole(&raw_dict).unwrap();
        directory
            .atomic_write(std::path::Path::new("dict.bin.zst"), &compressed)
            .unwrap();

        let compressor = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor {
                path: "dict.bin.zst".to_string(),
            }),
        });
        let resolved = compressor.resolve_dictionary(&directory).unwrap();
        assert_eq!(resolved.as_deref(), Some(raw_dict.as_slice()));
    }

    #[test]
    fn resolve_dictionary_is_none_without_a_dictionary() {
        use crate::directory::RamDirectory;

        let directory = RamDirectory::create();
        let compressor = Compressor::Zstd(ZstdCompressor::default());
        assert!(compressor.resolve_dictionary(&directory).unwrap().is_none());
        assert!(Compressor::None.resolve_dictionary(&directory).unwrap().is_none());
    }

    #[test]
    fn resolve_dictionary_missing_file_surfaces_not_found_not_other() {
        use crate::directory::RamDirectory;

        let directory = RamDirectory::create();
        let compressor = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor {
                path: "does_not_exist.bin.zst".to_string(),
            }),
        });
        let err = compressor.resolve_dictionary(&directory).unwrap_err();
        // A misconfigured `dictionary_path` should be as easy to diagnose as any other
        // missing-file error -- not flattened to `ErrorKind::Other`.
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn resolve_dictionary_rejects_unsafe_paths_even_when_constructed_directly() {
        // `ZstdDictionaryDescriptor::path` is `pub`, so a caller can bypass
        // `deser_from_str`'s validation entirely by building the struct literal directly.
        // `resolve_dictionary` must still refuse to hand an absolute/traversing path to the
        // `Directory`.
        use crate::directory::RamDirectory;

        let directory = RamDirectory::create();

        let absolute = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor {
                path: "/etc/passwd".to_string(),
            }),
        });
        let err = absolute.resolve_dictionary(&directory).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let traversal = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor {
                path: "../../../../etc/passwd".to_string(),
            }),
        });
        let err = traversal.resolve_dictionary(&directory).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let with_comma = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionaryDescriptor {
                path: "a,b.zst".to_string(),
            }),
        });
        let err = with_comma.resolve_dictionary(&directory).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
