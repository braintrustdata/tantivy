//! Store as a [`SegmentPlugin`] implementation.
//!
//! This wraps the existing `StoreWriter` and `StoreReader` types behind the plugin
//! interface so that the document store participates in the unified plugin lifecycle.

use std::any::Any;
use std::collections::BTreeMap;

use measure_time::debug_time;

use crate::directory::Directory;
use crate::index::{SegmentComponent, SegmentReader};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::schema::document::Document;
use crate::schema::Schema;
use crate::space_usage::{ComponentSpaceUsage, STORE};
use crate::store::{StoreReader, StoreWriter};
use crate::Segment;

/// Built-in segment plugin that stores and merges stored documents.
pub struct StorePlugin;

// Test-only observability into the store-merge fast path (`store_writer.stack`) below, so tests
// can assert stacking actually happened rather than falling back to decompress/recompress --
// e.g. that two segments compressed against the *same* dictionary still fast-merge. Thread-local
// rather than a shared global: tests drive merges with no merge thread pool, which runs the store
// merge synchronously on the calling (test) thread, so this stays isolated per test even when
// `cargo test` runs tests in parallel.
#[cfg(test)]
thread_local! {
    static STACKED_SEGMENTS_FOR_TEST: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn take_stacked_segments_for_test() -> usize {
    STACKED_SEGMENTS_FOR_TEST.with(|count| count.replace(0))
}

impl SegmentPlugin for StorePlugin {
    fn extensions(&self) -> &[&str] {
        &["store"]
    }

    fn create_writer(&self, _ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        unimplemented!("StorePlugin is a built-in; use StorePluginWriter::new")
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        debug_time!("write-storable-fields");
        debug!("write-storable-fields");

        let store_span = tracing::debug_span!(
            "write_storable_fields",
            doc_id_mapping_is_trivial = ctx.doc_id_mapping.is_trivial(),
            stacked_segments = tracing::field::Empty,
            copied_segments = tracing::field::Empty
        );
        let _enter = store_span.enter();
        let mut stacked_segments = 0usize;
        let mut copied_segments = 0usize;

        let path = ctx.target_segment.relative_path(SegmentComponent::Store);
        let directory = ctx.target_segment.index().directory();
        let store_write = directory.open_write(&path)?;
        let settings = ctx.settings;
        let dictionary = settings
            .docstore_compression
            .resolve_dictionary(directory)?;
        let mut store_writer = StoreWriter::new(
            store_write,
            settings.docstore_compression.clone(),
            settings.docstore_blocksize,
            settings.docstore_compress_dedicated_thread,
            dictionary,
        )?;

        if !ctx.doc_id_mapping.is_trivial() {
            debug!("non-trivial-doc-id-mapping");

            let store_readers: Vec<_> = ctx
                .readers
                .iter()
                .map(|reader| reader.get_store_reader(50))
                .collect::<Result<_, _>>()?;

            let mut document_iterators: Vec<_> = store_readers
                .iter()
                .enumerate()
                .map(|(i, store)| {
                    store.iter_raw_current_version(ctx.readers[i].alive_bitset(), ctx.schema)
                })
                .collect();

            for old_doc_addr in ctx.doc_id_mapping.iter_old_doc_addrs() {
                let doc_bytes_it = &mut document_iterators[old_doc_addr.segment_ord as usize];
                if let Some(doc_bytes_res) = doc_bytes_it.next() {
                    let doc_bytes = doc_bytes_res?;
                    store_writer.store_bytes(&doc_bytes)?;
                } else {
                    return Err(crate::error::DataCorruption::comment_only(format!(
                        "unexpected missing document in docstore on merge, doc address \
                         {old_doc_addr:?}",
                    ))
                    .into());
                }
            }
            copied_segments = ctx.readers.len();
        } else {
            debug!("trivial-doc-id-mapping");
            for reader in ctx.readers {
                let store_reader = reader.get_store_reader(1)?;
                if reader.has_deletes()
                    // If there is not enough data in the store, we avoid stacking in order to
                    // avoid creating many small blocks in the doc store. Once we have 5 full blocks,
                    // we start stacking. In the worst case 2/7 of the blocks would be very small.
                    || store_reader.block_checkpoints().take(7).count() < 6
                    || store_reader.decompressor() != store_writer.compressor().into()
                    // Byte-for-byte stacking below never decompresses, so it can't exercise the
                    // dictionary's own zstd checksum -- the one thing that would catch the
                    // dictionary invariant (this segment's blocks and the merge target must agree
                    // on the exact same dictionary) ever being violated (bug, race, manual
                    // meta.json edit). Compare what this segment actually recorded when its doc
                    // store was written (`SegmentMeta::docstore_dictionary_path`, frozen at write
                    // time) against the target's *current* dictionary: equal (including both
                    // being "no dictionary") allows the fast stacking path below; any mismatch --
                    // dictionary added, removed, or rotated to a different one -- forces
                    // decompress/recompress instead, so a violation fails loudly here rather than
                    // silently, on some unrelated later read.
                    || reader.docstore_dictionary_path() != store_writer.compressor().dictionary_path()
                    // Defense in depth: also cross-check the flag recorded directly in the
                    // `.store` file's own footer (self-contained in the file bytes, so it stays
                    // correct even if `meta.json` is ever hand-edited or drifts independently)
                    // rather than trusting `SegmentMeta` alone.
                    || store_reader.dictionary_used() != store_writer.compressor().has_dictionary()
                    // Stacking copies blocks verbatim, so every stacked segment must share the
                    // target's doc store version.
                    || store_reader.doc_store_version() != crate::store::DOC_STORE_VERSION
                {
                    for doc_bytes_res in
                        store_reader.iter_raw_current_version(reader.alive_bitset(), ctx.schema)
                    {
                        let doc_bytes = doc_bytes_res?;
                        store_writer.store_bytes(&doc_bytes)?;
                    }
                    copied_segments += 1;
                } else {
                    store_writer.stack(store_reader)?;
                    stacked_segments += 1;
                    #[cfg(test)]
                    STACKED_SEGMENTS_FOR_TEST.with(|count| count.set(count.get() + 1));
                }
            }
        }
        tracing::Span::current().record("stacked_segments", stacked_segments);
        tracing::Span::current().record("copied_segments", copied_segments);
        tracing::debug_span!("close_store_writer").in_scope(|| store_writer.close())?;
        Ok(())
    }

    fn space_usage(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<BTreeMap<String, ComponentSpaceUsage>> {
        let store = segment_reader.get_store_reader(0)?;
        Ok(BTreeMap::from([(
            STORE.to_string(),
            ComponentSpaceUsage::Store(store.space_usage()),
        )]))
    }
}

/// Accumulates and serializes stored documents for a segment.
pub struct StorePluginWriter {
    store_writer: Option<StoreWriter>,
    remapping_required: bool,
}

impl StorePluginWriter {
    pub(crate) fn new(ctx: &PluginWriterContext) -> crate::Result<Self> {
        let settings = ctx.segment.index().settings();
        let directory = ctx.segment.index().directory();
        let remapping_required = settings.sort_by_field.is_some() || settings.manual_doc_id_mapping;

        let store_writer = if remapping_required {
            let path = ctx.segment.relative_path(SegmentComponent::TempStore);
            let store_write = directory.open_write(&path)?;
            StoreWriter::new(
                store_write,
                crate::store::Compressor::None,
                // We want fast random access on the docs, so we choose a small block size.
                // If this is zero, the skip index will contain too many checkpoints and
                // therefore will be relatively slow.
                16000,
                settings.docstore_compress_dedicated_thread,
                None,
            )?
        } else {
            let path = ctx.segment.relative_path(SegmentComponent::Store);
            let store_write = directory.open_write(&path)?;
            let dictionary = settings.docstore_compression.resolve_dictionary(directory)?;
            StoreWriter::new(
                store_write,
                settings.docstore_compression.clone(),
                settings.docstore_blocksize,
                settings.docstore_compress_dedicated_thread,
                dictionary,
            )?
        };

        Ok(StorePluginWriter {
            store_writer: Some(store_writer),
            remapping_required,
        })
    }

    /// Serializes and stores a document according to the supplied schema.
    pub fn store<D: Document>(&mut self, document: &D, schema: &Schema) -> crate::Result<()> {
        if let Some(ref mut writer) = self.store_writer {
            writer
                .store(document, schema)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }

    /// Stores an already serialized document.
    pub fn store_bytes(&mut self, serialized_document: &[u8]) -> crate::Result<()> {
        if let Some(ref mut writer) = self.store_writer {
            writer
                .store_bytes(serialized_document)
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }
}

impl PluginWriter for StorePluginWriter {
    fn serialize(
        mut self: Box<Self>,
        segment: &Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        // For sorted indexes (remapping_required), we need to close the temp store
        // and reorder documents into the final store.
        if self.remapping_required {
            if let Some(doc_id_map) = doc_id_map {
                debug!("resort-docstore");
                // Close the temp store writer first
                if let Some(old_writer) = self.store_writer.take() {
                    old_writer
                        .close()
                        .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
                }

                // Open the final store writer
                let settings = segment.index().settings().clone();
                let store_write = segment.open_write(SegmentComponent::Store)?;
                let dictionary = settings
                    .docstore_compression
                    .resolve_dictionary(segment.index().directory())?;
                let mut store_writer = StoreWriter::new(
                    store_write,
                    settings.docstore_compression,
                    settings.docstore_blocksize,
                    settings.docstore_compress_dedicated_thread,
                    dictionary,
                )
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;

                // Read from temp store and write in new order
                let store_read = StoreReader::open(
                    segment.open_read(SegmentComponent::TempStore)?,
                    1, /* The docstore is configured to have one doc per block, and each doc is
                        * accessed only once: we don't need caching. */
                    None, // TempStore is always written with Compressor::None, no dictionary.
                )?;
                for old_doc_id in doc_id_map.iter_old_doc_ids() {
                    let doc_bytes = store_read.get_document_bytes(old_doc_id)?;
                    store_writer
                        .store_bytes(&doc_bytes)
                        .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
                }

                // Replace the writer with the final store writer
                self.store_writer = Some(store_writer);
            }
        }
        // For non-sorted indexes, data was already written incrementally.
        if let Some(writer) = self.store_writer {
            writer
                .close()
                .map_err(|e| crate::TantivyError::InternalError(e.to_string()))?;
        }
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        let Some(writer) = self.store_writer.as_ref() else {
            return 0;
        };
        writer.mem_usage()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
