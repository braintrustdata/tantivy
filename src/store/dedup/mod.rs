//! Experimental docstore dedup codec.
//!
//! See `brainstore/tantivy-repeated-postings-codec.md`. Phase 1 is the per-segment
//! dictionary type and its on-disk (de)serialization; the block codec and store wiring land
//! in later phases.

mod dictionary;

pub use dictionary::DedupDictionary;
