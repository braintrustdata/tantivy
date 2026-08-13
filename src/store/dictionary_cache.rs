//! Process-wide cache of decompressed dictionary bytes, keyed by path.
//!
//! Weak-referenced, not LRU: an entry serves cached bytes for as long as *something else* (a
//! `SegmentReader`, `StoreReader`, etc.) still holds a strong `Arc` to them; once every such
//! holder drops, the entry naturally goes dead and the next lookup re-decompresses. This trades
//! away an explicit memory ceiling for simplicity -- no eviction policy to tune, memory tracks
//! actual live usage. Mirrors `MmapDirectory`'s own `MmapCache` (`directory/mmap_directory.rs`),
//! the only other path-keyed weak-ref cache in this fork.
//!
//! This only helps when dictionary usage overlaps in time (concurrently open segments, or opens
//! in quick succession before the last reference drops) -- fully sequential open/close/open never
//! benefits, since nothing is left alive to serve the next lookup. That's expected, not a bug.
//!
//! Safe to key by bare path, with no directory identity in the key, only because the caller (see
//! `ZstdDictionary::load_internal`) verifies -- not assumes -- that a path is genuinely
//! content-addressed before ever calling `insert`. This module doesn't know or care about
//! hashing; it trusts `insert`'s caller to only ever pass a path that really does uniquely
//! identify `bytes`.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};

use once_cell::sync::Lazy;

static CACHE: Lazy<RwLock<HashMap<String, Weak<[u8]>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Returns cached bytes for `path`, if a strong reference is still alive somewhere.
pub(crate) fn get(path: &str) -> Option<Arc<[u8]>> {
    CACHE.read().unwrap().get(path).and_then(Weak::upgrade)
}

/// Records `bytes` under `path`. Racing with another thread's concurrent insert for the same
/// path is fine to just overwrite: the caller is responsible for having already verified that
/// `path` uniquely identifies `bytes`, so any two concurrent inserts for the same path are
/// content-identical.
pub(crate) fn insert(path: &str, bytes: &Arc<[u8]>) {
    CACHE
        .write()
        .unwrap()
        .insert(path.to_string(), Arc::downgrade(bytes));
}

/// Test-only isolation: this cache is a process-wide static, and several tests use different
/// `RamDirectory` instances that could otherwise cross-contaminate under parallel test execution
/// if they ever reused a path. Mirrors `indexer::merger::take_stacked_segments_for_test`'s
/// same-purpose reset.
#[cfg(test)]
pub(crate) fn clear_for_test() {
    CACHE.write().unwrap().clear();
}
