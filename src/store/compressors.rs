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
                    if !ZstdDictionary::is_safe_and_encodable_relative_path(dictionary.path()) {
                        return Err(serde::ser::Error::custom(format!(
                            "dictionary_path must be a plain path relative to the index \
                             directory (no leading '/', no '..' components, and no ','), got \
                             {:?}",
                            dictionary.path()
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

/// A zstd dictionary a doc store was (or should be) compressed with. Owns everything about how a
/// dictionary is stored, named, and resolved -- callers hand over raw bytes (`seed`/`from_bytes`)
/// or a recorded path (via `meta.json`, deserialized as `Path`); they never need to know the
/// on-disk naming convention or write the file themselves.
///
/// The dictionary's raw bytes are never part of `meta.json` -- only a `Directory`-relative path
/// is ever persisted (see `Serialize`/`Deserialize` below), resolved via `Directory::atomic_read`.
/// Integrity of the bytes at that path relies on zstd's own frame checksum (see
/// `compression_zstd_block`), verified as a side effect of decompression at effectively no extra
/// cost -- this type does not re-verify content on every load.
///
/// The path itself *is* content-addressed (SHA-256 of the raw bytes, see `seed`), which is a
/// separate, cheap, one-time cost paid only when a dictionary is first seeded, not on every open.
/// This matters for correctness, not just dedup: two different dictionaries must never compare
/// equal as `IndexSettings` (`ZstdDictionary`'s `PartialEq` is path-based), since merges -- both
/// `IndexMerger::write_storable_fields` and any embedder comparing `IndexSettings` before merging
/// indices from possibly-different sources -- rely on that equality to detect a real dictionary
/// mismatch rather than silently combining segments compressed against different dictionaries.
///
/// `SegmentUpdater::list_files` protects `path()` from garbage collection for any `Directory` that
/// routes GC through it (i.e. anything wrapped in `ManagedDirectory`, which is every `Index`).
/// Directory implementations that run their own GC outside of that path are not covered and must
/// protect the dictionary file themselves.
#[derive(Clone, Debug)]
pub enum ZstdDictionary {
    /// Not yet resolved to bytes; `path` is where to load them from via `Directory::atomic_read`.
    /// This is what deserializing from `meta.json` always produces.
    Path(String),
    /// Bytes already known -- e.g. just supplied to `seed`/`from_bytes`, or already loaded once.
    Loaded {
        /// Where these bytes are (or will be) persisted, relative to the index directory.
        path: String,
        /// The dictionary's raw (uncompressed) bytes.
        bytes: std::sync::Arc<[u8]>,
    },
}

// Not feature-gated: `ZstdDictionary` sits behind `Option<ZstdDictionary>` on the never-gated
// `ZstdCompressor` struct, whose derived `PartialEq`/`Eq`/`Serialize`/`Deserialize` therefore
// need `ZstdDictionary` to implement those traits (and thus need `path()`) unconditionally, not
// just when the `zstd-compression` feature happens to be enabled.
impl ZstdDictionary {
    /// The `Directory`-relative path of this dictionary's file, regardless of whether the bytes
    /// have been loaded yet.
    pub fn path(&self) -> &str {
        match self {
            ZstdDictionary::Path(path) => path,
            ZstdDictionary::Loaded { path, .. } => path,
        }
    }
}

/// Process-wide cache of decompressed dictionary bytes, keyed by path. Safe to key by path alone
/// (no separate directory identity in the key) because `load_internal` *verifies*, not assumes,
/// that a path is genuinely content-addressed before ever inserting into this map (recomputes
/// the hash of what was actually loaded and rejects a mismatch) -- so an entry that made it in
/// here is guaranteed unique to its content, regardless of which `Directory`/index it came from.
///

/// Weak-referenced, not LRU: an entry serves cached bytes for as long as *something else* (a
/// `SegmentReader`, `StoreReader`, etc.) still holds a strong `Arc` to them; once every such
/// holder drops, the entry naturally goes dead and the next lookup re-decompresses. This trades
/// away an explicit memory ceiling for simplicity -- no eviction policy to tune, memory tracks
/// actual live usage. Mirrors `MmapDirectory`'s own `MmapCache` (`directory/mmap_directory.rs`),
/// the only other path-keyed weak-ref cache in this fork.
///
/// This only helps when dictionary usage overlaps in time (concurrently open segments, or opens
/// in quick succession before the last reference drops) -- fully sequential open/close/open never
/// benefits, since nothing is left alive to serve the next lookup. That's expected, not a bug.
#[cfg(feature = "zstd-compression")]
static DICTIONARY_CACHE: once_cell::sync::Lazy<
    std::sync::RwLock<std::collections::HashMap<String, std::sync::Weak<[u8]>>>,
> = once_cell::sync::Lazy::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

// Test-only isolation for `DICTIONARY_CACHE`: it's a process-wide static, and several tests
// (here and in `store/mod.rs`) reuse the same literal path (e.g. "dict.bin.zst") with different
// content across different `RamDirectory` instances. Without clearing between tests, parallel
// test execution could serve one test's cached bytes to another under the same path. Mirrors
// `indexer::merger::take_stacked_segments_for_test`'s same-purpose reset.
#[cfg(all(feature = "zstd-compression", test))]
pub(crate) fn clear_dictionary_cache_for_test() {
    DICTIONARY_CACHE.write().unwrap().clear();
}

#[cfg(feature = "zstd-compression")]
impl ZstdDictionary {
    /// Rejects absolute paths, `..`/`.` components (would escape the index directory via
    /// `Directory::resolve_path`'s unchecked `root_path.join`), and `,` (collides with
    /// `zstd(opt=val,...)`'s option separator). `ZstdDictionary` is the only thing that ever
    /// opens a dictionary file (`load`/`load_internal`/`seed`), so it's the natural place to own
    /// what counts as a safe path -- every other caller of this check (`Compressor::serialize`,
    /// `ZstdCompressor::deser_from_str`) is validating a path *before* it becomes a `ZstdDictionary`.
    pub(crate) fn is_safe_and_encodable_relative_path(path: &str) -> bool {
        let as_path = std::path::Path::new(path);
        !as_path.is_absolute()
            && as_path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)))
            && !path.contains(',')
    }

    /// Returns the dictionary's raw bytes, loading and decompressing them from `directory` if not
    /// already known.
    pub fn load(&self, directory: &dyn crate::Directory) -> io::Result<std::sync::Arc<[u8]>> {
        match self {
            ZstdDictionary::Loaded { bytes, .. } => Ok(bytes.clone()),
            ZstdDictionary::Path(path) => Self::load_internal(directory, path),
        }
    }

    /// Reads and decompresses dictionary bytes from `path` via `directory`. The mechanical core
    /// shared by `load` (this dictionary's own path) and by segment-open resolution, which must
    /// load from a specific *segment's own recorded* path (`SegmentMeta::docstore_dictionary_path`)
    /// rather than this dictionary's path -- the two can differ if an index's dictionary setting
    /// ever changes after a segment was written, and a segment must always be read back with
    /// whatever it was actually compressed against, not the index's current setting.
    pub(crate) fn load_internal(
        directory: &dyn crate::Directory,
        path: &str,
    ) -> io::Result<std::sync::Arc<[u8]>> {
        if !Self::is_safe_and_encodable_relative_path(path) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "dictionary_path must be a plain path relative to the index directory (no \
                     leading '/', no '..' components, and no ','), got {path:?}"
                ),
            ));
        }

        if let Some(bytes) = DICTIONARY_CACHE
            .read()
            .unwrap()
            .get(path)
            .and_then(std::sync::Weak::upgrade)
        {
            return Ok(bytes);
        }

        let compressed = directory
            .atomic_read(std::path::Path::new(path))
            .map_err(|err| {
                // Preserve `NotFound`/the original `io::Error`'s kind, rather than flattening
                // every failure mode to `ErrorKind::Other` -- a misconfigured dictionary path
                // should be as easy to diagnose as a typical missing-file error.
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
        let bytes: std::sync::Arc<[u8]> = std::sync::Arc::from(super::decompress_whole(&compressed)?);

        // The cache's whole safety argument is "the path is content-addressed, so path equality
        // implies content equality" -- but nothing stops a caller from constructing
        // `ZstdDictionary::Path` with an arbitrary string (it's a public variant), and a fixed
        // literal name was literally this fork's previous design. Don't just assume the
        // invariant holds: verify it, on every cold load, by recomputing the content-addressed
        // name for what was actually read and comparing it against `path`. A mismatch means
        // caching under `path` would be unsafe (two different directories could then poison each
        // other's cache entry under the same literal name) -- refuse rather than cache it.
        let expected_path = content_addressed_dictionary_path(&bytes);
        if expected_path != path {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "dictionary at {path:?} does not hash to its own path (expected \
                     {expected_path:?}) -- refusing to cache it under a name that doesn't \
                     uniquely identify its content, since the dictionary cache is process-wide \
                     and keyed by path alone"
                ),
            ));
        }

        // Racing with another thread's concurrent miss for the same path is fine: both decompress
        // independently and both inserts are content-identical (verified above), so simply
        // overwriting is correct -- no need for single-flight coordination here.
        DICTIONARY_CACHE
            .write()
            .unwrap()
            .insert(path.to_string(), std::sync::Arc::downgrade(&bytes));

        Ok(bytes)
    }

    /// Seeds a brand-new index's dictionary: derives a content-addressed path from a SHA-256 of
    /// `bytes`, writes them zstd-compressed to `directory` at that path, and returns the
    /// resulting `ZstdDictionary` (already `Loaded`, since the caller's bytes are used directly
    /// rather than immediately reading them back). This is the *only* thing tantivy can't do on
    /// its own when creating a new index: the actual dictionary bytes have to come from the
    /// embedding application. Everything else -- naming, writing, and later resolving -- is
    /// entirely this fork's own business.
    pub fn seed(directory: &dyn crate::Directory, bytes: std::sync::Arc<[u8]>) -> io::Result<Self> {
        let compressed = super::compress_whole(&bytes)?;
        let path = content_addressed_dictionary_path(&bytes);
        directory.atomic_write(std::path::Path::new(&path), &compressed)?;
        Ok(ZstdDictionary::Loaded { path, bytes })
    }

    /// Constructs a dictionary directly from bytes already in hand, with no I/O -- for callers
    /// that already know both the path and the bytes (e.g. reusing a dictionary previously seeded
    /// elsewhere, or tests).
    pub fn from_bytes(path: String, bytes: std::sync::Arc<[u8]>) -> Self {
        ZstdDictionary::Loaded { path, bytes }
    }
}

/// Derives a content-addressed filename for a dictionary's raw bytes: identical bytes always
/// produce the same name (safe to re-seed/overwrite idempotently), different bytes produce
/// different names with overwhelming probability (SHA-256), so `ZstdDictionary`/`IndexSettings`
/// equality actually reflects dictionary identity rather than just "some caller reused the same
/// literal path."
#[cfg(feature = "zstd-compression")]
fn content_addressed_dictionary_path(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(bytes);
    let hex: String = hash.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("dict-{hex}.bin.zst")
}

impl PartialEq for ZstdDictionary {
    /// Path-based: two dictionaries are the same iff their (content-addressed) paths match,
    /// regardless of whether either side happens to already have bytes loaded. Resolution state
    /// is incidental/lazy and must not affect equality -- `IndexSettings` comparisons (merges)
    /// rely on this being a pure statement about *which* dictionary, not about caching state.
    fn eq(&self, other: &Self) -> bool {
        self.path() == other.path()
    }
}
impl Eq for ZstdDictionary {}

impl Serialize for ZstdDictionary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ZstdDictionary", 1)?;
        state.serialize_field("path", self.path())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ZstdDictionary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        #[derive(Deserialize)]
        struct Repr {
            path: String,
        }
        let repr = Repr::deserialize(deserializer)?;
        Ok(ZstdDictionary::Path(repr.path))
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// The Zstd compressor, with optional compression level and dictionary.
pub struct ZstdCompressor {
    /// The compression level, if unset defaults to zstd::DEFAULT_COMPRESSION_LEVEL = 3
    pub compression_level: Option<i32>,
    /// If set, the doc store was (and must continue to be) compressed against this dictionary.
    #[serde(default)]
    pub dictionary: Option<ZstdDictionary>,
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
                    if !ZstdDictionary::is_safe_and_encodable_relative_path(value) {
                        return Err(format!(
                            "dictionary_path must be a plain path relative to the index \
                             directory (no leading '/', no '..' components, and no ','), \
                             got {value:?}"
                        ));
                    }
                    compressor.dictionary = Some(ZstdDictionary::Path(value.to_string()));
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
            opts.push(format!("dictionary_path={}", dictionary.path()));
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
            Self::Zstd(zstd_compressor) => zstd_compressor.dictionary.as_ref().map(|d| d.path()),
        }
    }

    /// If this compressor names a zstd dictionary, loads its bytes via `directory` (see
    /// `ZstdDictionary::load`). Returns `Ok(None)` if this compressor doesn't use a dictionary.
    pub fn resolve_dictionary(
        &self,
        directory: &dyn crate::Directory,
    ) -> io::Result<Option<std::sync::Arc<[u8]>>> {
        match self {
            Self::None => Ok(None),
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => Ok(None),
            #[cfg(feature = "zstd-compression")]
            Self::Zstd(zstd_compressor) => zstd_compressor
                .dictionary
                .as_ref()
                .map(|dictionary| dictionary.load(directory))
                .transpose(),
        }
    }
}

/// Loads dictionary bytes for a specific *segment's own recorded* path
/// (`SegmentMeta::docstore_dictionary_path`), rather than a `Compressor`'s currently-configured
/// one. Opening an existing segment must always resolve whatever dictionary it was actually
/// compressed against, not the index's current setting -- the two can differ if the index's
/// dictionary setting ever changes after the segment was written, which is exactly the scenario
/// the merge-eligibility checks in `indexer/merger.rs` exist to catch. Returns `Ok(None)` if the
/// segment has no recorded dictionary path.
pub(crate) fn resolve_segment_dictionary(
    directory: &dyn crate::Directory,
    segment_dictionary_path: Option<&str>,
) -> io::Result<Option<std::sync::Arc<[u8]>>> {
    let Some(path) = segment_dictionary_path else {
        return Ok(None);
    };
    #[cfg(feature = "zstd-compression")]
    {
        Ok(Some(ZstdDictionary::load_internal(directory, path)?))
    }
    #[cfg(not(feature = "zstd-compression"))]
    {
        let _ = (directory, path);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "segment has a docstore dictionary path recorded, but this build lacks the \
             zstd-compression feature",
        ))
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
            dictionary: Some(ZstdDictionary::Path("dict.bin.zst".to_string())),
        };
        assert_eq!(
            ZstdCompressor::deser_from_str(&compressor_with_dict.ser_to_string()).unwrap(),
            compressor_with_dict
        );

        let dict_only = ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionary::Path("dict.bin.zst".to_string())),
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
                dictionary: Some(ZstdDictionary::Path("dict.bin.zst".to_string())),
            }
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=15,dictionary_path=dict.bin.zst)")
                .unwrap(),
            ZstdCompressor {
                compression_level: Some(15),
                dictionary: Some(ZstdDictionary::Path("dict.bin.zst".to_string())),
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
        // with `zstd(opt=val,opt=val)`'s `,`-based option separator. `ZstdDictionary` can be
        // constructed directly (`Path`/`from_bytes`/`seed`), so a caller can carry one in even
        // though `deser_from_str` would reject it in text form. Note this can *not* be caught by
        // adding a check inside `deser_from_str`'s "dictionary_path" arm: `options.split(',')`
        // fragments the string into separate options *before* any option-specific value is ever
        // isolated, so a value containing ',' never reaches that arm as one piece -- it instead
        // surfaces downstream as the unrelated, unhelpful `no '=' found in option "b.zst"`.
        // Serializing (e.g. for `meta.json`) is the first point after direct construction where
        // this is actually reachable and worth catching, so that's where it's checked.
        let compressor_with_comma = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionary::Path("a,b.zst".to_string())),
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
            dictionary: Some(ZstdDictionary::Path("dict.bin.zst".to_string())),
        });
        assert!(serde_json::to_string(&safe).is_ok());
    }

    #[test]
    fn resolve_dictionary_reads_and_decompresses_from_path() {
        use crate::directory::RamDirectory;
        use std::sync::Arc;

        clear_dictionary_cache_for_test();
        let directory = RamDirectory::create();
        let raw_dict: Arc<[u8]> =
            Arc::from(b"a dictionary's worth of bytes, stored zstd-compressed on disk".to_vec());
        // `seed` writes the compressed bytes under the correct content-addressed name; construct
        // a fresh `Path` (not the `Loaded` value `seed` returns) so `resolve_dictionary` actually
        // exercises the read-from-`Directory` path this test means to cover.
        let seeded = ZstdDictionary::seed(&directory, raw_dict.clone()).unwrap();

        let compressor = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionary::Path(seeded.path().to_string())),
        });
        let resolved = compressor.resolve_dictionary(&directory).unwrap();
        assert_eq!(resolved.as_deref(), Some(raw_dict.as_ref()));
    }

    #[test]
    fn zstd_dictionary_cache_shares_bytes_while_a_strong_ref_is_alive() {
        use crate::directory::RamDirectory;
        use std::sync::Arc;

        clear_dictionary_cache_for_test();
        let directory = RamDirectory::create();
        let raw: Arc<[u8]> = Arc::from(b"shared dictionary content for cache hit test".to_vec());
        let seeded = ZstdDictionary::seed(&directory, raw).unwrap();

        let dict = ZstdDictionary::Path(seeded.path().to_string());
        let first = dict.load(&directory).unwrap();
        let second = dict.load(&directory).unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "a second load while the first Arc is still alive should return the cached \
             allocation, not decompress again"
        );
    }

    #[test]
    fn zstd_dictionary_cache_reloads_after_last_strong_ref_drops() {
        use crate::directory::{Directory, RamDirectory};
        use std::sync::Arc;

        clear_dictionary_cache_for_test();
        let directory = RamDirectory::create();
        let raw: Arc<[u8]> = Arc::from(b"dictionary content for cache eviction test".to_vec());
        let seeded = ZstdDictionary::seed(&directory, raw.clone()).unwrap();
        let path = seeded.path().to_string();

        let dict = ZstdDictionary::Path(path.clone());
        let first = dict.load(&directory).unwrap();
        assert_eq!(first.as_ref(), raw.as_ref());
        drop(first); // last strong ref -- the cache entry is now dead

        // Delete the underlying file. Overwriting it with *different* content, the way this test
        // used to prove non-staleness, no longer works: `load_internal` now verifies a loaded
        // dictionary hashes back to its own path, so mismatched content at the same path would
        // just be (correctly) rejected -- that's a different code path than the one this test
        // means to exercise. Deleting instead proves the same thing more directly: if the cache
        // incorrectly served a stale entry instead of reloading, this would still succeed with
        // the old (correct) content; since it correctly reloads, it must now fail with `NotFound`.
        directory.delete(std::path::Path::new(&path)).unwrap();
        let err = dict.load(&directory).unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::NotFound,
            "once the only strong Arc drops, a later load must reload fresh (and see the file is \
             gone), not serve a stale cached entry"
        );
    }

    #[test]
    fn zstd_dictionary_load_rejects_content_that_does_not_hash_to_its_own_path() {
        // The cache's safety argument is "the path is content-addressed, so path equality implies
        // content equality" -- but `ZstdDictionary::Path` is a public variant, constructible with
        // any string, and a fixed literal name was this fork's own previous design (and remains
        // possible for any embedder, legacy data, or manual `meta.json` edit). `load_internal`
        // must not just assume the invariant holds: it verifies it on every cold load, refusing
        // to cache (or return) bytes whose hash doesn't match the path they were loaded from.
        use crate::directory::{Directory, RamDirectory};

        clear_dictionary_cache_for_test();
        let directory = RamDirectory::create();
        let content = b"this content does not hash to the literal path below".to_vec();
        let compressed = super::super::compression_zstd_block::compress_whole(&content).unwrap();
        directory
            .atomic_write(std::path::Path::new("not-actually-a-hash.bin.zst"), &compressed)
            .unwrap();

        let dict = ZstdDictionary::Path("not-actually-a-hash.bin.zst".to_string());
        let err = dict.load(&directory).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("does not hash to its own path"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn zstd_dictionary_shared_literal_path_across_directories_never_serves_cross_contaminated_bytes()
     {
        // Direct regression test for a reviewer-flagged scenario: two different `Directory`
        // instances (simulating two different open indexes in one process) using the same
        // non-content-addressed literal path, each with their own different bytes at that path.
        // Before hash verification, whichever loaded first could populate the process-wide cache
        // under that shared literal key, and the other could then silently receive the wrong
        // index's bytes on its own lookup for the same key, for as long as that Arc stayed alive.
        // With verification, neither load can ever succeed: a non-hash-derived shared literal can
        // never accumulate a valid cache entry in the first place (every insert attempt is
        // checked against its own content and rejected if it doesn't match) -- so there's no
        // window in which a correct entry could exist for the other directory to be incorrectly
        // served.
        use crate::directory::{Directory, RamDirectory};

        clear_dictionary_cache_for_test();
        let shared_literal_path = "dict.bin.zst"; // fixed literal, not content-addressed

        let directory_a = RamDirectory::create();
        directory_a
            .atomic_write(
                std::path::Path::new(shared_literal_path),
                &super::super::compression_zstd_block::compress_whole(b"index A's dictionary")
                    .unwrap(),
            )
            .unwrap();

        let directory_b = RamDirectory::create();
        directory_b
            .atomic_write(
                std::path::Path::new(shared_literal_path),
                &super::super::compression_zstd_block::compress_whole(
                    b"index B's completely different dictionary",
                )
                .unwrap(),
            )
            .unwrap();

        let dict = ZstdDictionary::Path(shared_literal_path.to_string());
        assert_eq!(
            dict.load(&directory_a).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData,
            "index A's own load should fail rather than populate the shared cache key"
        );
        assert_eq!(
            dict.load(&directory_b).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData,
            "index B's load must independently fail too -- it must never receive index A's \
             bytes from the shared cache key, since nothing valid was ever cached there"
        );
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
            dictionary: Some(ZstdDictionary::Path("does_not_exist.bin.zst".to_string())),
        });
        let err = compressor.resolve_dictionary(&directory).unwrap_err();
        // A misconfigured `dictionary_path` should be as easy to diagnose as any other
        // missing-file error -- not flattened to `ErrorKind::Other`.
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn zstd_dictionary_is_safe_and_encodable_relative_path() {
        // Direct unit coverage of the check itself, now owned by `ZstdDictionary` (the only
        // entity that ever opens a dictionary file) rather than floating as a free function.
        assert!(!ZstdDictionary::is_safe_and_encodable_relative_path(
            "/etc/passwd"
        ));
        assert!(!ZstdDictionary::is_safe_and_encodable_relative_path(
            "../../../../etc/passwd"
        ));
        assert!(!ZstdDictionary::is_safe_and_encodable_relative_path(
            "a,b.zst"
        ));
        assert!(ZstdDictionary::is_safe_and_encodable_relative_path(
            "dict.bin.zst"
        ));
    }

    #[test]
    fn zstd_dictionary_load_rejects_unsafe_paths_even_when_constructed_directly() {
        // `ZstdDictionary::Path` can be constructed directly, so a caller can bypass
        // `deser_from_str`'s validation entirely. `ZstdDictionary::load` -- the only place that
        // ever hands a dictionary path to a `Directory` -- must still refuse to do so for an
        // absolute/traversing/comma-containing path.
        use crate::directory::RamDirectory;

        let directory = RamDirectory::create();

        let err = ZstdDictionary::Path("/etc/passwd".to_string())
            .load(&directory)
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = ZstdDictionary::Path("../../../../etc/passwd".to_string())
            .load(&directory)
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = ZstdDictionary::Path("a,b.zst".to_string())
            .load(&directory)
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
