//! Compressed/slow/row-oriented storage for documents.
//!
//! A field needs to be marked as stored in the schema in
//! order to be handled in the `Store`.
//!
//! Internally, documents (or rather their stored fields) are serialized to a buffer.
//! When the buffer exceeds `block_size` (defaults to 16K), the buffer is compressed
//! using LZ4 or Zstd and the resulting block is written to disk.
//!
//! One can then request for a specific `DocId`.
//! A skip list helps navigating to the right block,
//! decompresses it entirely and returns the document within it.
//!
//! If the last document requested was in the same block,
//! the reader is smart enough to avoid decompressing
//! the block a second time, but their is no real
//! uncompressed block* cache.
//!
//! A typical use case for the store is, once
//! the search result page has been computed, returning
//! the actual content of the 10 best document.
//!
//! # Usage
//!
//! Most users should not access the `StoreReader` directly
//! and should rely on either
//!
//! - at the segment level, the
//! [`SegmentReader`'s `doc` method](../struct.SegmentReader.html#method.doc)
//! - at the index level, the [`Searcher::doc()`](crate::Searcher::doc) method

mod compressors;
mod decompressors;
mod footer;
mod index;
mod reader;
mod writer;
pub use self::compressors::{Compressor, ZstdCompressor, ZstdDictionary};
pub(crate) use self::compressors::resolve_segment_dictionary;
pub use self::decompressors::Decompressor;
pub(crate) use self::reader::DOCSTORE_CACHE_CAPACITY;
pub use self::reader::{CacheStats, StoreReader};
pub use self::writer::StoreWriter;
mod store_compressor;

/// Doc store version in footer to handle format changes.
pub(crate) const DOC_STORE_VERSION: u32 = 1;

#[cfg(feature = "lz4-compression")]
mod compression_lz4_block;

#[cfg(feature = "zstd-compression")]
mod compression_zstd_block;
#[cfg(feature = "zstd-compression")]
pub use self::compression_zstd_block::{compress_whole, decompress_whole};

#[cfg(test)]
pub mod tests {

    use std::path::Path;
    use std::sync::Arc;

    use super::*;
    use crate::directory::{Directory, RamDirectory, WritePtr};
    use crate::fastfield::AliveBitSet;
    use crate::schema::document::Value;
    use crate::schema::{
        self, Schema, TantivyDocument, TextFieldIndexing, TextOptions, STORED, TEXT,
    };
    use crate::{Index, IndexSettings, IndexWriter, Term};

    const LOREM: &str = "Doc Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do \
                         eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad \
                         minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip \
                         ex ea commodo consequat. Duis aute irure dolor in reprehenderit in \
                         voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur \
                         sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt \
                         mollit anim id est laborum.";

    const BLOCK_SIZE: usize = 16_384;

    pub fn write_lorem_ipsum_store(
        writer: WritePtr,
        num_docs: usize,
        compressor: Compressor,
        blocksize: usize,
        separate_thread: bool,
    ) -> Schema {
        write_lorem_ipsum_store_with_dictionary(
            writer,
            num_docs,
            compressor,
            blocksize,
            separate_thread,
            None,
        )
    }

    pub fn write_lorem_ipsum_store_with_dictionary(
        writer: WritePtr,
        num_docs: usize,
        compressor: Compressor,
        blocksize: usize,
        separate_thread: bool,
        dictionary: Option<std::sync::Arc<[u8]>>,
    ) -> Schema {
        let mut schema_builder = Schema::builder();
        let field_body = schema_builder.add_text_field("body", TextOptions::default().set_stored());
        let field_title =
            schema_builder.add_text_field("title", TextOptions::default().set_stored());
        let schema = schema_builder.build();
        {
            let mut store_writer =
                StoreWriter::new(writer, compressor, blocksize, separate_thread, dictionary)
                    .unwrap();
            for i in 0..num_docs {
                let mut doc = TantivyDocument::default();
                doc.add_field_value(field_body, LOREM.to_string());
                doc.add_field_value(field_title, format!("Doc {i}"));
                store_writer.store(&doc, &schema).unwrap();
            }
            store_writer.close().unwrap();
        }
        schema
    }

    const NUM_DOCS: usize = 1_000;
    #[test]
    fn test_doc_store_iter_with_delete_bug_1077() -> crate::Result<()> {
        // this will cover deletion of the first element in a checkpoint
        let deleted_doc_ids = (200..300).collect::<Vec<_>>();
        let alive_bitset =
            AliveBitSet::for_test_from_deleted_docs(&deleted_doc_ids, NUM_DOCS as u32);

        let path = Path::new("store");
        let directory = RamDirectory::create();
        let store_wrt = directory.open_write(path)?;
        let schema =
            write_lorem_ipsum_store(store_wrt, NUM_DOCS, Compressor::Lz4, BLOCK_SIZE, true);
        let field_title = schema.get_field("title").unwrap();
        let store_file = directory.open_read(path)?;
        let store = StoreReader::open(store_file, 10, None)?;
        for i in 0..NUM_DOCS as u32 {
            assert_eq!(
                *store
                    .get::<TantivyDocument>(i)?
                    .get_first(field_title)
                    .unwrap()
                    .as_str()
                    .unwrap(),
                format!("Doc {i}")
            );
        }

        for doc in store.iter::<TantivyDocument>(Some(&alive_bitset)) {
            let doc = doc?;
            let title_content = doc.get_first(field_title).unwrap().as_str().unwrap();
            if !title_content.starts_with("Doc ") {
                panic!("unexpected title_content {title_content}");
            }

            let id = title_content
                .strip_prefix("Doc ")
                .unwrap()
                .parse::<u32>()
                .unwrap();
            if alive_bitset.is_deleted(id) {
                panic!("unexpected deleted document {id}");
            }
        }

        Ok(())
    }

    fn test_store(
        compressor: Compressor,
        blocksize: usize,
        separate_thread: bool,
    ) -> crate::Result<()> {
        let path = Path::new("store");
        let directory = RamDirectory::create();
        let store_wrt = directory.open_write(path)?;
        let schema =
            write_lorem_ipsum_store(store_wrt, NUM_DOCS, compressor, blocksize, separate_thread);
        let field_title = schema.get_field("title").unwrap();
        let store_file = directory.open_read(path)?;
        let store = StoreReader::open(store_file, 10, None)?;
        for i in 0..NUM_DOCS as u32 {
            assert_eq!(
                *store
                    .get::<TantivyDocument>(i)?
                    .get_first(field_title)
                    .unwrap()
                    .as_str()
                    .unwrap(),
                format!("Doc {i}")
            );
        }
        for (i, doc) in store.iter::<TantivyDocument>(None).enumerate() {
            assert_eq!(
                *doc?.get_first(field_title).unwrap().as_str().unwrap(),
                format!("Doc {i}")
            );
        }
        Ok(())
    }

    #[test]
    fn test_store_no_compression_same_thread() -> crate::Result<()> {
        test_store(Compressor::None, BLOCK_SIZE, false)
    }

    #[test]
    fn test_store_no_compression() -> crate::Result<()> {
        test_store(Compressor::None, BLOCK_SIZE, true)
    }

    #[cfg(feature = "lz4-compression")]
    #[test]
    fn test_store_lz4_block() -> crate::Result<()> {
        test_store(Compressor::Lz4, BLOCK_SIZE, true)
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_store_zstd() -> crate::Result<()> {
        test_store(
            Compressor::Zstd(ZstdCompressor::default()),
            BLOCK_SIZE,
            true,
        )
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_store_zstd_with_dictionary() -> crate::Result<()> {
        let dictionary: Arc<[u8]> = LOREM.as_bytes().to_vec().into();
        let path = Path::new("store");
        let directory = RamDirectory::create();
        let store_wrt = directory.open_write(path)?;
        let schema = write_lorem_ipsum_store_with_dictionary(
            store_wrt,
            NUM_DOCS,
            Compressor::Zstd(ZstdCompressor::default()),
            BLOCK_SIZE,
            true,
            Some(dictionary.clone()),
        );
        let field_title = schema.get_field("title").unwrap();
        let store_file = directory.open_read(path)?;
        let store = StoreReader::open(store_file, 10, Some(dictionary))?;
        for i in 0..NUM_DOCS as u32 {
            assert_eq!(
                *store
                    .get::<TantivyDocument>(i)?
                    .get_first(field_title)
                    .unwrap()
                    .as_str()
                    .unwrap(),
                format!("Doc {i}")
            );
        }
        Ok(())
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_store_zstd_dictionary_mismatch_errors() -> crate::Result<()> {
        let dictionary: Arc<[u8]> = LOREM.as_bytes().to_vec().into();
        let wrong_dictionary: Arc<[u8]> = b"not the right dictionary bytes at all"
            .to_vec()
            .into();
        let path = Path::new("store");
        let directory = RamDirectory::create();
        let store_wrt = directory.open_write(path)?;
        write_lorem_ipsum_store_with_dictionary(
            store_wrt,
            10,
            Compressor::Zstd(ZstdCompressor::default()),
            BLOCK_SIZE,
            true,
            Some(dictionary),
        );
        let store_file = directory.open_read(path)?;

        // StoreReader::open no longer eagerly checks the dictionary (that would mean hashing a
        // potentially multi-megabyte dictionary on every open) -- a missing/mismatched
        // dictionary is instead caught by zstd's own frame checksum the first time a block is
        // actually decompressed.
        let reader_without_dict = StoreReader::open(store_file.clone(), 10, None)?;
        assert!(reader_without_dict.get::<TantivyDocument>(0).is_err());

        let reader_with_wrong_dict = StoreReader::open(store_file, 10, Some(wrong_dictionary))?;
        assert!(reader_with_wrong_dict.get::<TantivyDocument>(0).is_err());
        Ok(())
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_store_reader_ignores_an_unused_supplied_dictionary() -> crate::Result<()> {
        // A store written *without* a dictionary must open and read fine even if the caller
        // happens to supply one anyway (e.g. because the directory's settings named one, but
        // this particular store was written before/without it) -- blocks that were never
        // compressed against a dictionary don't reference it, so a dictionary the decompressor
        // never needed is simply unused, not an error.
        let unused_dictionary: Arc<[u8]> = LOREM.as_bytes().to_vec().into();
        let path = Path::new("store");
        let directory = RamDirectory::create();
        let store_wrt = directory.open_write(path)?;
        let schema = write_lorem_ipsum_store(
            store_wrt,
            10,
            Compressor::Zstd(ZstdCompressor::default()),
            BLOCK_SIZE,
            true,
        );
        let field_title = schema.get_field("title").unwrap();
        let store_file = directory.open_read(path)?;
        let store = StoreReader::open(store_file, 10, Some(unused_dictionary))?;
        assert_eq!(
            store
                .get::<TantivyDocument>(0)?
                .get_first(field_title)
                .unwrap()
                .as_str()
                .unwrap(),
            "Doc 0"
        );
        Ok(())
    }

    #[test]
    fn test_store_with_delete() -> crate::Result<()> {
        let mut schema_builder = schema::Schema::builder();

        let text_field_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_index_option(schema::IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();
        let text_field = schema_builder.add_text_field("text_field", text_field_options);
        let schema = schema_builder.build();
        let index_builder = Index::builder().schema(schema);

        let index = index_builder.create_in_ram()?;

        {
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            index_writer.add_document(doc!(text_field=> "deleteme"))?;
            index_writer.add_document(doc!(text_field=> "deletemenot"))?;
            index_writer.add_document(doc!(text_field=> "deleteme"))?;
            index_writer.add_document(doc!(text_field=> "deletemenot"))?;
            index_writer.add_document(doc!(text_field=> "deleteme"))?;

            index_writer.delete_term(Term::from_field_text(text_field, "deleteme"));
            index_writer.commit()?;
        }

        let searcher = index.reader()?.searcher();
        let reader = searcher.segment_reader(0);
        let store = reader.get_store_reader(10)?;
        for doc in store.iter::<TantivyDocument>(reader.alive_bitset()) {
            assert_eq!(
                *doc?.get_first(text_field).unwrap().as_str().unwrap(),
                "deletemenot".to_string()
            );
        }
        Ok(())
    }

    #[cfg(feature = "lz4-compression")]
    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_merge_with_changed_compressor() -> crate::Result<()> {
        let mut schema_builder = schema::Schema::builder();

        let text_field = schema_builder.add_text_field("text_field", TEXT | STORED);
        let schema = schema_builder.build();
        let index_builder = Index::builder().schema(schema);

        let mut index = index_builder.create_in_ram().unwrap();
        index.settings_mut().docstore_compression = Compressor::Lz4;
        {
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            // put enough data create enough blocks in the doc store to be considered for stacking
            for _ in 0..200 {
                index_writer.add_document(doc!(text_field=> LOREM))?;
            }
            assert!(index_writer.commit().is_ok());
            for _ in 0..200 {
                index_writer.add_document(doc!(text_field=> LOREM))?;
            }
            assert!(index_writer.commit().is_ok());
        }
        assert_eq!(
            index.reader().unwrap().searcher().segment_readers()[0]
                .get_store_reader(10)
                .unwrap()
                .decompressor(),
            Decompressor::Lz4
        );
        // Change compressor, this disables stacking on merging
        let index_settings = index.settings_mut();
        index_settings.docstore_compression = Compressor::Zstd(Default::default());
        // Merging the segments
        {
            let segment_ids = index
                .searchable_segment_ids()
                .expect("Searchable segments failed.");
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            assert!(index_writer.merge(&segment_ids).wait().is_ok());
            assert!(index_writer.wait_merging_threads().is_ok());
        }

        let searcher = index.reader().unwrap().searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let reader = searcher.segment_readers().iter().last().unwrap();
        let store = reader.get_store_reader(10).unwrap();

        for doc in store
            .iter::<TantivyDocument>(reader.alive_bitset())
            .take(50)
        {
            assert_eq!(
                *doc?.get_first(text_field).and_then(|v| v.as_str()).unwrap(),
                LOREM.to_string()
            );
        }
        assert_eq!(store.decompressor(), Decompressor::Zstd);

        Ok(())
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_merge_recompresses_correctly_when_dictionary_rotates() -> crate::Result<()> {
        // A dictionary is meant to be fixed for an index's life, but if it's ever changed anyway
        // (bug, race, manual meta.json edit, or a deliberate migration), a merge over segments
        // written under the old dictionary must never raw-copy their blocks (stacking) into a
        // merged segment declared under the new one -- `SegmentReader::open` resolves each
        // segment's *own* recorded dictionary correctly (`SegmentMeta::docstore_dictionary_path`),
        // so decompressing old segments still works; what must not happen is skipping that
        // decompression step and shipping dict_a-compressed bytes under a dict_b label. The
        // eligibility check forces the decompress/recompress path whenever a segment's recorded
        // dictionary doesn't match the merge target's, so the merge below succeeds *and*
        // correctly migrates every document to the new dictionary -- this is only correct
        // because recompression actually happens; assert that explicitly via the stacked-segment
        // count, not just that the merge didn't error.
        let mut schema_builder = schema::Schema::builder();
        let text_field = schema_builder.add_text_field("text_field", TEXT | STORED);
        let schema = schema_builder.build();

        // Write the dictionaries to the *raw* directory, before it's wrapped in a
        // `ManagedDirectory` by `open_or_create` below -- exactly as Brainstore's own
        // `seed_dictionary_for_create` does (writes via the unwrapped `PrefixDirectory`, before
        // `IndexBuilder::open_or_create` wraps it). This matters: a file written *through* the
        // `ManagedDirectory` gets registered and becomes eligible for the automatic
        // post-commit GC (`SegmentUpdater::list_files` only protects segment files + meta.json),
        // which would otherwise delete an unrelated sibling file like the dictionary blob.
        let ram_directory = RamDirectory::create();
        let dict_a = super::compression_zstd_block::compress_whole(LOREM.as_bytes())?;
        let dict_b =
            super::compression_zstd_block::compress_whole(b"an entirely different dictionary")?;
        ram_directory.atomic_write(Path::new("dict_a.bin.zst"), &dict_a)?;
        ram_directory.atomic_write(Path::new("dict_b.bin.zst"), &dict_b)?;

        let settings = IndexSettings {
            docstore_compression: Compressor::Zstd(ZstdCompressor {
                compression_level: None,
                dictionary: Some(ZstdDictionary::Path("dict_a.bin.zst".to_string())),
            }),
            ..Default::default()
        };
        let mut index = Index::builder()
            .schema(schema)
            .settings(settings)
            .open_or_create(ram_directory)?;
        {
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            // Put enough data in each segment to be considered for stacking.
            for _ in 0..200 {
                index_writer
                    .add_document(doc!(text_field=> LOREM))
                    .expect("add_document 1 failed");
            }
            index_writer.commit().expect("commit 1 failed");
            for _ in 0..200 {
                index_writer
                    .add_document(doc!(text_field=> LOREM))
                    .expect("add_document 2 failed");
            }
            index_writer.commit().expect("commit 2 failed");
        }

        // Rotate the dictionary -- same compressor family, different dictionary. A real
        // Brainstore-side bug/race is what this is meant to simulate; the fork itself has no
        // way to prevent this in-process.
        index.settings_mut().docstore_compression = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: Some(ZstdDictionary::Path("dict_b.bin.zst".to_string())),
        });

        // Drive the merge directly through `IndexMerger`/`SegmentSerializer`, the same building
        // blocks `SegmentUpdater::merge` uses (`indexer/segment_updater.rs`), rather than through
        // `IndexWriter::merge`'s scheduling: under `cfg(test)`, a scheduled merge that returns an
        // `Err` deliberately panics inside a rayon worker instead of surfacing a `Result`
        // (`segment_updater.rs`'s `if cfg!(test) { panic!(...) }`), which is orthogonal to what
        // this test wants to observe.
        let segments = index.searchable_segments().expect("Searchable segments failed.");
        let merger = crate::indexer::merger::IndexMerger::open(
            index.schema(),
            index.settings().clone(),
            &segments[..],
        )?;
        let merged_segment = index.new_segment();
        let segment_serializer =
            crate::indexer::SegmentSerializer::for_segment(merged_segment, true)?;
        crate::indexer::merger::take_stacked_segments_for_test(); // reset from any prior test
        let merge_result = merger.write(segment_serializer, None);
        assert!(
            merge_result.is_ok(),
            "merge should succeed -- decompressing each segment with its own recorded \
             dictionary and recompressing under the new one is a correct migration, not \
             corruption; got {merge_result:?}"
        );
        assert_eq!(
            crate::indexer::merger::take_stacked_segments_for_test(),
            0,
            "correctness here depends entirely on the recompress path actually running -- \
             neither segment may take the raw-copy stacking shortcut when its recorded \
             dictionary doesn't match the target's"
        );
        Ok(())
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_merge_recompresses_correctly_when_dictionary_removed() -> crate::Result<()> {
        // Mirror image of `test_merge_recompresses_correctly_when_dictionary_rotates`: the
        // stacking-eligibility guard used to only check whether the merge *target* currently has
        // a dictionary configured (`store_writer.compressor().has_dictionary()`), so
        // Some(dict) -> None went undetected -- the target's `has_dictionary()` is `false`, so
        // the guard never fired, and blocks physically compressed against `dict_a` got raw-copied
        // into a merged segment whose settings now said "no dictionary" -- real corruption, since
        // nothing would have decompressed them correctly. The fix compares the segment's own
        // recorded dictionary path/flag against the target's current setting, in both
        // directions, forcing recompress here too -- which now means each segment is correctly
        // decompressed with its own dict_a and recompressed with no dictionary at all: a correct
        // migration, not corruption. Assert that recompression actually ran, not just success.
        let mut schema_builder = schema::Schema::builder();
        let text_field = schema_builder.add_text_field("text_field", TEXT | STORED);
        let schema = schema_builder.build();

        // See the rotation test above for why this is written to the raw, unwrapped directory.
        let ram_directory = RamDirectory::create();
        let dict_a = super::compression_zstd_block::compress_whole(LOREM.as_bytes())?;
        ram_directory.atomic_write(Path::new("dict_a.bin.zst"), &dict_a)?;

        let settings = IndexSettings {
            docstore_compression: Compressor::Zstd(ZstdCompressor {
                compression_level: None,
                dictionary: Some(ZstdDictionary::Path("dict_a.bin.zst".to_string())),
            }),
            ..Default::default()
        };
        let mut index = Index::builder()
            .schema(schema)
            .settings(settings)
            .open_or_create(ram_directory)?;
        {
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            // Put enough data in each segment to be considered for stacking.
            for _ in 0..200 {
                index_writer
                    .add_document(doc!(text_field=> LOREM))
                    .expect("add_document 1 failed");
            }
            index_writer.commit().expect("commit 1 failed");
            for _ in 0..200 {
                index_writer
                    .add_document(doc!(text_field=> LOREM))
                    .expect("add_document 2 failed");
            }
            index_writer.commit().expect("commit 2 failed");
        }

        // Remove the dictionary -- same compressor family (Zstd), no dictionary. A real
        // Brainstore-side bug/race is what this is meant to simulate; the fork itself has no
        // way to prevent this in-process.
        index.settings_mut().docstore_compression = Compressor::Zstd(ZstdCompressor {
            compression_level: None,
            dictionary: None,
        });

        // Drive the merge directly through `IndexMerger`/`SegmentSerializer`; see the rotation
        // test above for why this bypasses `IndexWriter::merge`'s scheduling.
        let segments = index.searchable_segments().expect("Searchable segments failed.");
        let merger = crate::indexer::merger::IndexMerger::open(
            index.schema(),
            index.settings().clone(),
            &segments[..],
        )?;
        let merged_segment = index.new_segment();
        let segment_serializer =
            crate::indexer::SegmentSerializer::for_segment(merged_segment, true)?;
        crate::indexer::merger::take_stacked_segments_for_test(); // reset from any prior test
        let merge_result = merger.write(segment_serializer, None);
        assert!(
            merge_result.is_ok(),
            "merge should succeed -- decompressing each segment with its own recorded \
             dict_a and recompressing with no dictionary is a correct migration, not \
             corruption; got {merge_result:?}"
        );
        assert_eq!(
            crate::indexer::merger::take_stacked_segments_for_test(),
            0,
            "correctness here depends entirely on the recompress path actually running -- \
             neither segment may take the raw-copy stacking shortcut when its recorded \
             dictionary doesn't match the target's"
        );
        Ok(())
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_merge_stacks_when_both_segments_share_the_same_dictionary() -> crate::Result<()> {
        // Segments compressed against the *same* dictionary (the common case: a dictionary
        // configured once, unchanged, for the index's whole life) should still fast-merge
        // (raw block-stacking, no decompress/recompress) -- the eligibility guard forcing
        // recompress for any dictionary-configured target was overly blunt; comparing the
        // segment's own recorded dictionary path (`SegmentMeta::docstore_dictionary_path`)
        // against the target's current one lets same-dictionary merges take the fast path while
        // still catching real mismatches (see the rotated/removed dictionary tests above).
        let mut schema_builder = schema::Schema::builder();
        let text_field = schema_builder.add_text_field("text_field", TEXT | STORED);
        let schema = schema_builder.build();

        // See the rotation test above for why this is written to the raw, unwrapped directory.
        let ram_directory = RamDirectory::create();
        let dict = super::compression_zstd_block::compress_whole(LOREM.as_bytes())?;
        ram_directory.atomic_write(Path::new("dict.bin.zst"), &dict)?;

        let settings = IndexSettings {
            docstore_compression: Compressor::Zstd(ZstdCompressor {
                compression_level: None,
                dictionary: Some(ZstdDictionary::Path("dict.bin.zst".to_string())),
            }),
            ..Default::default()
        };
        let mut index = Index::builder()
            .schema(schema)
            .settings(settings)
            .open_or_create(ram_directory)?;
        {
            let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
            // Put enough data in each segment to be considered for stacking.
            for _ in 0..200 {
                index_writer
                    .add_document(doc!(text_field=> LOREM))
                    .expect("add_document 1 failed");
            }
            index_writer.commit().expect("commit 1 failed");
            for _ in 0..200 {
                index_writer
                    .add_document(doc!(text_field=> LOREM))
                    .expect("add_document 2 failed");
            }
            index_writer.commit().expect("commit 2 failed");
        }

        // Dictionary setting is untouched -- both segments and the merge target agree.
        let segments = index.searchable_segments().expect("Searchable segments failed.");
        assert_eq!(segments.len(), 2);
        let merger = crate::indexer::merger::IndexMerger::open(
            index.schema(),
            index.settings().clone(),
            &segments[..],
        )?;
        let merged_segment = index.new_segment();
        let segment_serializer =
            crate::indexer::SegmentSerializer::for_segment(merged_segment, true)?;
        crate::indexer::merger::take_stacked_segments_for_test(); // reset from any prior test
        let merge_result = merger.write(segment_serializer, None);
        assert!(
            merge_result.is_ok(),
            "merge of same-dictionary segments should succeed; got {merge_result:?}"
        );
        assert_eq!(
            crate::indexer::merger::take_stacked_segments_for_test(),
            2,
            "both segments share the merge target's dictionary and should take the fast \
             block-stacking path, not decompress/recompress"
        );
        Ok(())
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_single_segment_index_writer_records_dictionary_path() -> crate::Result<()> {
        // `SingleSegmentIndexWriter::finalize` is one of four call sites that stamp
        // `SegmentMeta::docstore_dictionary_path` at commit time (alongside the normal commit
        // path and both merge paths) -- exercise it directly rather than relying on the other
        // three sites' coverage by inspection, since nothing else in the crate touches this
        // writer at all.
        let mut schema_builder = schema::Schema::builder();
        let text_field = schema_builder.add_text_field("text_field", TEXT | STORED);
        let schema = schema_builder.build();

        // See the rotation test above for why this is written to the raw, unwrapped directory.
        let ram_directory = RamDirectory::create();
        let dict = super::compression_zstd_block::compress_whole(LOREM.as_bytes())?;
        ram_directory.atomic_write(Path::new("dict.bin.zst"), &dict)?;

        let settings = IndexSettings {
            docstore_compression: Compressor::Zstd(ZstdCompressor {
                compression_level: None,
                dictionary: Some(ZstdDictionary::Path("dict.bin.zst".to_string())),
            }),
            ..Default::default()
        };

        let mut writer = Index::builder()
            .schema(schema)
            .settings(settings)
            .single_segment_index_writer::<TantivyDocument>(
                ram_directory,
                crate::indexer::index_writer::MEMORY_BUDGET_NUM_BYTES_MIN,
            )?;
        writer.add_document(doc!(text_field=> LOREM))?;
        let index = writer.finalize()?;

        let segments = index.searchable_segments()?;
        assert_eq!(segments.len(), 1);
        assert_eq!(
            segments[0].meta().docstore_dictionary_path(),
            Some("dict.bin.zst")
        );

        // Sanity: the document is actually readable back through the recorded dictionary.
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.num_docs(), 1);
        let doc: TantivyDocument = searcher.doc(crate::DocAddress::new(0u32, 0u32))?;
        assert_eq!(
            *doc.get_first(text_field).and_then(|v| v.as_str()).unwrap(),
            LOREM.to_string()
        );
        Ok(())
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_dictionary_written_through_managed_directory_survives_post_commit_gc()
    -> crate::Result<()> {
        // Regression test for a managed-directory GC hazard: seeding the dictionary blob via
        // `Index::directory()` -- a `ManagedDirectory`, and the natural, public way to write a
        // sibling asset once you hold an open `Index` -- registers the file for garbage
        // collection (`ManagedDirectory::atomic_write`/`open_write` unconditionally call
        // `register_file_as_managed` before delegating). `SegmentUpdater::list_files`, the GC's
        // live-file set, only knows about segment component files and `meta.json`; it has no
        // notion of `index_settings.docstore_compression`'s dictionary path. `schedule_commit`
        // runs `garbage_collect_files` unconditionally after every commit, so a dictionary
        // written this way is deleted out from under the index on the very next commit.
        //
        // Contrast with `test_merge_with_rotated_dictionary_fails_loud_instead_of_stacking_silently`
        // above, which deliberately writes its dictionaries to the *raw*, unwrapped directory to
        // dodge exactly this hazard.
        let mut schema_builder = schema::Schema::builder();
        let text_field = schema_builder.add_text_field("text_field", TEXT | STORED);
        let schema = schema_builder.build();

        let dict_path = "dict.bin.zst";
        let dict_bytes = super::compression_zstd_block::compress_whole(LOREM.as_bytes())?;

        let settings = IndexSettings {
            docstore_compression: Compressor::Zstd(ZstdCompressor {
                compression_level: None,
                dictionary: Some(ZstdDictionary::Path(dict_path.to_string())),
            }),
            ..Default::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()?;

        // Seed the dictionary the way an integrator naturally would once they hold an open
        // `Index`: through `Index::directory()`, i.e. the managed directory -- not through the
        // raw pre-wrap directory the way the rotation test above does.
        index
            .directory()
            .atomic_write(Path::new(dict_path), &dict_bytes)?;

        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            for _ in 0..200 {
                index_writer.add_document(doc!(text_field=> LOREM))?;
            }
            // The unconditional post-commit `garbage_collect_files` call is the mechanism under
            // test: it should not sweep up a file it doesn't know is load-bearing.
            index_writer.commit()?;
        }

        assert!(
            index.directory().exists(Path::new(dict_path))?,
            "dictionary file seeded through the managed directory was garbage collected on the \
             very next commit"
        );

        Ok(())
    }

    #[test]
    fn test_merge_of_small_segments() -> crate::Result<()> {
        let mut schema_builder = schema::Schema::builder();

        let text_field = schema_builder.add_text_field("text_field", TEXT | STORED);
        let schema = schema_builder.build();
        let index_builder = Index::builder().schema(schema);

        let index = index_builder.create_in_ram().unwrap();

        {
            let mut index_writer = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=> "1"))?;
            index_writer.commit()?;
            index_writer.add_document(doc!(text_field=> "2"))?;
            index_writer.commit()?;
            index_writer.add_document(doc!(text_field=> "3"))?;
            index_writer.commit()?;
            index_writer.add_document(doc!(text_field=> "4"))?;
            index_writer.commit()?;
            index_writer.add_document(doc!(text_field=> "5"))?;
            index_writer.commit()?;
        }
        // Merging the segments
        {
            let segment_ids = index.searchable_segment_ids()?;
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.merge(&segment_ids).wait()?;
            index_writer.wait_merging_threads()?;
        }

        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let reader = searcher.segment_readers().iter().last().unwrap();
        let store = reader.get_store_reader(10)?;
        assert_eq!(store.block_checkpoints().count(), 1);
        Ok(())
    }
}

#[cfg(all(test, feature = "unstable"))]
mod bench {

    use std::path::Path;

    use test::Bencher;

    use super::tests::write_lorem_ipsum_store;
    use crate::directory::{Directory, RamDirectory};
    use crate::store::{Compressor, StoreReader};
    use crate::TantivyDocument;

    #[bench]
    #[cfg(feature = "mmap")]
    fn bench_store_encode(b: &mut Bencher) {
        let directory = RamDirectory::create();
        let path = Path::new("store");
        b.iter(|| {
            write_lorem_ipsum_store(
                directory.open_write(path).unwrap(),
                1_000,
                Compressor::default(),
                16_384,
                true,
            );
            directory.delete(path).unwrap();
        });
    }

    #[bench]
    fn bench_store_decode(b: &mut Bencher) {
        let directory = RamDirectory::create();
        let path = Path::new("store");
        write_lorem_ipsum_store(
            directory.open_write(path).unwrap(),
            1_000,
            Compressor::default(),
            16_384,
            true,
        );
        let store_file = directory.open_read(path).unwrap();
        let store = StoreReader::open(store_file, 10, None).unwrap();
        b.iter(|| store.iter::<TantivyDocument>(None).collect::<Vec<_>>());
    }
}
