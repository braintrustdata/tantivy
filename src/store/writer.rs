use std::io;

use common::BinarySerializable;

use super::compressors::Compressor;
use super::StoreReader;
use crate::directory::WritePtr;
use crate::schema::document::{BinaryDocumentSerializer, Document};
use crate::schema::Schema;
use crate::store::store_compressor::BlockCompressor;
use crate::DocId;

/// Write tantivy's [`Store`](./index.html)
///
/// Contrary to the other components of `tantivy`,
/// the store is written to disc as document as being added,
/// as opposed to when the segment is getting finalized.
///
/// The skip list index on the other hand, is built in memory.
pub struct StoreWriter {
    compressor: Compressor,
    block_size: usize,
    num_docs_in_current_block: DocId,
    current_block: Vec<u8>,
    doc_pos: Vec<u32>,
    block_compressor: BlockCompressor,
}

struct StoreReaderStats {
    total_store_bytes: u64,
    data_bytes: u64,
    offsets_bytes: u64,
    num_docs: u64,
    num_blocks: usize,
}

fn store_reader_stats(store_reader: &StoreReader) -> StoreReaderStats {
    let space_usage = store_reader.space_usage();
    let mut num_docs = 0u64;
    let mut num_blocks = 0usize;
    for checkpoint in store_reader.block_checkpoints() {
        num_docs = u64::from(checkpoint.doc_range.end);
        num_blocks += 1;
    }
    StoreReaderStats {
        total_store_bytes: space_usage.total().get_bytes(),
        data_bytes: space_usage.data_usage().get_bytes(),
        offsets_bytes: space_usage.offsets_usage().get_bytes(),
        num_docs,
        num_blocks,
    }
}

impl StoreWriter {
    /// Create a store writer.
    ///
    /// The store writer will writes blocks on disc as
    /// document are added.
    pub fn new(
        writer: WritePtr,
        compressor: Compressor,
        block_size: usize,
        dedicated_thread: bool,
    ) -> io::Result<StoreWriter> {
        let block_compressor = BlockCompressor::new(compressor, writer, dedicated_thread)?;
        Ok(StoreWriter {
            compressor,
            block_size,
            num_docs_in_current_block: 0,
            doc_pos: Vec::new(),
            current_block: Vec::new(),
            block_compressor,
        })
    }

    pub(crate) fn compressor(&self) -> Compressor {
        self.compressor
    }

    /// The memory used (inclusive childs)
    pub fn mem_usage(&self) -> usize {
        self.current_block.capacity() + self.doc_pos.capacity() * std::mem::size_of::<u32>()
    }

    /// Checks if the current block is full, and if so, compresses and flushes it.
    fn check_flush_block(&mut self) -> io::Result<()> {
        // this does not count the VInt storing the index lenght itself, but it is negligible in
        // front of everything else.
        let index_len = self.doc_pos.len() * std::mem::size_of::<usize>();
        if self.current_block.len() + index_len > self.block_size {
            self.send_current_block_to_compressor()?;
        }
        Ok(())
    }

    /// Flushes current uncompressed block and sends to compressor.
    fn send_current_block_to_compressor(&mut self) -> io::Result<()> {
        // We don't do anything if the current block is empty to begin with.
        if self.current_block.is_empty() {
            return Ok(());
        }

        let flush_span = tracing::info_span!(
            "flush_stored_fields_block",
            compressor = ?self.compressor,
            block_size = self.block_size,
            uncompressed_block_bytes = self.current_block.len(),
            num_docs_in_block = self.num_docs_in_current_block,
            num_offsets_in_block = self.doc_pos.len()
        );
        let _enter = flush_span.enter();

        let size_of_u32 = std::mem::size_of::<u32>();
        self.current_block
            .reserve((self.doc_pos.len() + 1) * size_of_u32);

        for pos in self.doc_pos.iter() {
            pos.serialize(&mut self.current_block)?;
        }
        (self.doc_pos.len() as u32).serialize(&mut self.current_block)?;

        self.block_compressor
            .compress_block_and_write(&self.current_block, self.num_docs_in_current_block)?;
        self.doc_pos.clear();
        self.current_block.clear();
        self.num_docs_in_current_block = 0;
        Ok(())
    }

    /// Store a new document.
    ///
    /// The document id is implicitly the current number
    /// of documents.
    pub fn store<D: Document>(&mut self, document: &D, schema: &Schema) -> io::Result<()> {
        self.doc_pos.push(self.current_block.len() as u32);

        let mut serializer = BinaryDocumentSerializer::new(&mut self.current_block, schema);
        serializer.serialize_doc(document)?;

        self.num_docs_in_current_block += 1;
        self.check_flush_block()?;
        Ok(())
    }

    /// Store bytes of a serialized document.
    ///
    /// The document id is implicitly the current number
    /// of documents.
    pub fn store_bytes(&mut self, serialized_document: &[u8]) -> io::Result<()> {
        self.doc_pos.push(self.current_block.len() as u32);
        self.current_block.extend_from_slice(serialized_document);
        self.num_docs_in_current_block += 1;
        self.check_flush_block()?;
        Ok(())
    }

    /// Stacks a store reader on top of the documents written so far.
    /// This method is an optimization compared to iterating over the documents
    /// in the store and adding them one by one, as the store's data will
    /// not be decompressed and then recompressed.
    pub fn stack(&mut self, store_reader: StoreReader) -> io::Result<()> {
        let stats = store_reader_stats(&store_reader);
        tracing::info_span!(
            "stack_store_reader",
            compressor = ?self.compressor,
            block_size = self.block_size,
            current_block_bytes = self.current_block.len(),
            current_block_docs = self.num_docs_in_current_block,
            store_bytes = stats.total_store_bytes,
            store_data_bytes = stats.data_bytes,
            store_offsets_bytes = stats.offsets_bytes,
            num_docs = stats.num_docs,
            num_blocks = stats.num_blocks
        )
        .in_scope(|| {
            // We flush the current block first before stacking
            self.send_current_block_to_compressor()?;
            self.block_compressor.stack_reader(store_reader)?;
            Ok(())
        })
    }

    /// Finalized the store writer.
    ///
    /// Compress the last unfinished block if any,
    /// and serializes the skip list index on disc.
    pub fn close(mut self) -> io::Result<()> {
        tracing::info_span!(
            "store_writer_close",
            compressor = ?self.compressor,
            block_size = self.block_size,
            current_block_bytes = self.current_block.len(),
            current_block_docs = self.num_docs_in_current_block
        )
        .in_scope(|| {
            tracing::info_span!("flush_pending_store_block")
                .in_scope(|| self.send_current_block_to_compressor())?;
            tracing::info_span!("close_block_compressor")
                .in_scope(|| self.block_compressor.close())?;
            Ok(())
        })
    }
}
