use std::io::Write;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;
use std::{io, thread};

use common::{BinarySerializable, CountingWriter, TerminatingWrite};

use crate::directory::WritePtr;
use crate::store::footer::DocStoreFooter;
use crate::store::index::{Checkpoint, SkipIndexBuilder};
use crate::store::{Compressor, Decompressor, StoreReader};
use crate::DocId;

pub struct BlockCompressor(BlockCompressorVariants);

#[derive(Clone, Copy)]
struct StackReaderStats {
    total_store_bytes: u64,
    data_bytes: u64,
    offsets_bytes: u64,
    num_docs: u64,
    num_blocks: usize,
}

impl StackReaderStats {
    fn from_store_reader(store_reader: &StoreReader) -> Self {
        let space_usage = store_reader.space_usage();
        let mut num_docs = 0u64;
        let mut num_blocks = 0usize;
        for checkpoint in store_reader.block_checkpoints() {
            num_docs = u64::from(checkpoint.doc_range.end);
            num_blocks += 1;
        }
        Self {
            total_store_bytes: space_usage.total().get_bytes(),
            data_bytes: space_usage.data_usage().get_bytes(),
            offsets_bytes: space_usage.offsets_usage().get_bytes(),
            num_docs,
            num_blocks,
        }
    }
}

// The struct wrapping an enum is just here to keep the
// impls private.
enum BlockCompressorVariants {
    SameThread(BlockCompressorImpl),
    DedicatedThread(DedicatedThreadBlockCompressorImpl),
}

impl BlockCompressor {
    pub fn new(compressor: Compressor, wrt: WritePtr, dedicated_thread: bool) -> io::Result<Self> {
        let block_compressor_impl = BlockCompressorImpl::new(compressor, wrt);
        if dedicated_thread {
            let dedicated_thread_compressor =
                DedicatedThreadBlockCompressorImpl::new(block_compressor_impl)?;
            Ok(BlockCompressor(BlockCompressorVariants::DedicatedThread(
                dedicated_thread_compressor,
            )))
        } else {
            Ok(BlockCompressor(BlockCompressorVariants::SameThread(
                block_compressor_impl,
            )))
        }
    }

    pub fn compress_block_and_write(
        &mut self,
        bytes: &[u8],
        num_docs_in_block: u32,
    ) -> io::Result<()> {
        match &mut self.0 {
            BlockCompressorVariants::SameThread(block_compressor) => {
                tracing::info_span!(
                    "compress_store_block",
                    uncompressed_block_bytes = bytes.len(),
                    num_docs_in_block = num_docs_in_block,
                    compressed_block_bytes = tracing::field::Empty,
                    write_start_offset = tracing::field::Empty,
                    write_end_offset = tracing::field::Empty
                )
                .in_scope(|| block_compressor.compress_block_and_write(bytes, num_docs_in_block))?;
            }
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor
                    .compress_block_and_write(bytes, num_docs_in_block)?;
            }
        }
        Ok(())
    }

    pub fn stack_reader(&mut self, store_reader: StoreReader) -> io::Result<()> {
        let stats = StackReaderStats::from_store_reader(&store_reader);
        match &mut self.0 {
            BlockCompressorVariants::SameThread(block_compressor) => {
                tracing::info_span!(
                    "stack_store_reader_blocks",
                    store_bytes = stats.total_store_bytes,
                    store_data_bytes = stats.data_bytes,
                    store_offsets_bytes = stats.offsets_bytes,
                    num_docs = stats.num_docs,
                    num_blocks = stats.num_blocks,
                    write_start_offset = tracing::field::Empty,
                    write_end_offset = tracing::field::Empty
                )
                .in_scope(|| block_compressor.stack(store_reader))?;
            }
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor.stack_reader(store_reader, stats)?;
            }
        }
        Ok(())
    }

    pub fn close(self) -> io::Result<()> {
        let imp = self.0;
        match imp {
            BlockCompressorVariants::SameThread(block_compressor) => block_compressor.close(),
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                tracing::info_span!("wait_for_docstore_compressor_thread")
                    .in_scope(|| different_thread_block_compressor.close())
            }
        }
    }
}

struct BlockCompressorImpl {
    compressor: Compressor,
    first_doc_in_block: DocId,
    offset_index_writer: SkipIndexBuilder,
    intermediary_buffer: Vec<u8>,
    writer: CountingWriter<WritePtr>,
}

impl BlockCompressorImpl {
    fn new(compressor: Compressor, writer: WritePtr) -> Self {
        Self {
            compressor,
            first_doc_in_block: 0,
            offset_index_writer: SkipIndexBuilder::new(),
            intermediary_buffer: Vec::new(),
            writer: CountingWriter::wrap(writer),
        }
    }

    fn compress_block_and_write(&mut self, data: &[u8], num_docs_in_block: u32) -> io::Result<()> {
        assert!(num_docs_in_block > 0);
        self.intermediary_buffer.clear();
        self.compressor
            .compress_into(data, &mut self.intermediary_buffer)?;

        let start_offset = self.writer.written_bytes() as usize;
        self.writer.write_all(&self.intermediary_buffer)?;
        let end_offset = self.writer.written_bytes() as usize;
        tracing::Span::current().record("compressed_block_bytes", self.intermediary_buffer.len());
        tracing::Span::current().record("write_start_offset", start_offset as u64);
        tracing::Span::current().record("write_end_offset", end_offset as u64);

        self.register_checkpoint(Checkpoint {
            doc_range: self.first_doc_in_block..self.first_doc_in_block + num_docs_in_block,
            byte_range: start_offset..end_offset,
        });
        Ok(())
    }

    fn register_checkpoint(&mut self, checkpoint: Checkpoint) {
        self.offset_index_writer.insert(checkpoint.clone());
        self.first_doc_in_block = checkpoint.doc_range.end;
    }

    /// Stacks a store reader on top of the documents written so far.
    /// This method is an optimization compared to iterating over the documents
    /// in the store and adding them one by one, as the store's data will
    /// not be decompressed and then recompressed.
    fn stack(&mut self, store_reader: StoreReader) -> io::Result<()> {
        let doc_shift = self.first_doc_in_block;
        let start_shift = self.writer.written_bytes() as usize;

        // just bulk write all of the block of the given reader.
        self.writer
            .write_all(store_reader.block_data()?.as_slice())?;
        let end_shift = self.writer.written_bytes() as usize;
        tracing::Span::current().record("write_start_offset", start_shift as u64);
        tracing::Span::current().record("write_end_offset", end_shift as u64);

        // concatenate the index of the `store_reader`, after translating
        // its start doc id and its start file offset.
        for mut checkpoint in store_reader.block_checkpoints() {
            checkpoint.doc_range.start += doc_shift;
            checkpoint.doc_range.end += doc_shift;
            checkpoint.byte_range.start += start_shift;
            checkpoint.byte_range.end += start_shift;
            self.register_checkpoint(checkpoint);
        }
        Ok(())
    }

    fn close(mut self) -> io::Result<()> {
        let header_offset: u64 = self.writer.written_bytes();
        let docstore_footer =
            DocStoreFooter::new(header_offset, Decompressor::from(self.compressor));
        self.offset_index_writer.serialize_into(&mut self.writer)?;
        docstore_footer.serialize(&mut self.writer)?;
        self.writer.terminate()
    }
}

// ---------------------------------
enum BlockCompressorMessage {
    CompressBlockAndWrite {
        block_data: Vec<u8>,
        num_docs_in_block: u32,
        parent_span: tracing::Span,
    },
    Stack {
        store_reader: StoreReader,
        stats: StackReaderStats,
        parent_span: tracing::Span,
    },
}

struct DedicatedThreadBlockCompressorImpl {
    join_handle: Option<JoinHandle<io::Result<()>>>,
    tx: SyncSender<BlockCompressorMessage>,
}

impl DedicatedThreadBlockCompressorImpl {
    fn new(mut block_compressor: BlockCompressorImpl) -> io::Result<Self> {
        let (tx, rx): (
            SyncSender<BlockCompressorMessage>,
            Receiver<BlockCompressorMessage>,
        ) = sync_channel(3);
        let parent_span = tracing::Span::current();
        let join_handle = thread::Builder::new()
            .name("docstore-compressor-thread".to_string())
            .spawn(move || {
                let _parent_span_guard = parent_span.enter();
                while let Ok(packet) = rx.recv() {
                    match packet {
                        BlockCompressorMessage::CompressBlockAndWrite {
                            block_data,
                            num_docs_in_block,
                            parent_span,
                        } => {
                            let _parent_span_guard = parent_span.enter();
                            tracing::info_span!(
                                "compress_store_block",
                                uncompressed_block_bytes = block_data.len(),
                                num_docs_in_block = num_docs_in_block,
                                compressed_block_bytes = tracing::field::Empty,
                                write_start_offset = tracing::field::Empty,
                                write_end_offset = tracing::field::Empty
                            )
                            .in_scope(|| {
                                block_compressor
                                    .compress_block_and_write(&block_data[..], num_docs_in_block)
                            })?;
                        }
                        BlockCompressorMessage::Stack {
                            store_reader,
                            stats,
                            parent_span,
                        } => {
                            let _parent_span_guard = parent_span.enter();
                            tracing::info_span!(
                                "stack_store_reader_blocks",
                                store_bytes = stats.total_store_bytes,
                                store_data_bytes = stats.data_bytes,
                                store_offsets_bytes = stats.offsets_bytes,
                                num_docs = stats.num_docs,
                                num_blocks = stats.num_blocks,
                                write_start_offset = tracing::field::Empty,
                                write_end_offset = tracing::field::Empty
                            )
                            .in_scope(|| block_compressor.stack(store_reader))?;
                        }
                    }
                }
                block_compressor.close()?;
                Ok(())
            })?;
        Ok(DedicatedThreadBlockCompressorImpl {
            join_handle: Some(join_handle),
            tx,
        })
    }

    fn compress_block_and_write(&mut self, bytes: &[u8], num_docs_in_block: u32) -> io::Result<()> {
        self.send(BlockCompressorMessage::CompressBlockAndWrite {
            block_data: bytes.to_vec(),
            num_docs_in_block,
            parent_span: tracing::Span::current(),
        })
    }

    fn stack_reader(
        &mut self,
        store_reader: StoreReader,
        stats: StackReaderStats,
    ) -> io::Result<()> {
        self.send(BlockCompressorMessage::Stack {
            store_reader,
            stats,
            parent_span: tracing::Span::current(),
        })
    }

    fn send(&mut self, msg: BlockCompressorMessage) -> io::Result<()> {
        if self.tx.send(msg).is_err() {
            harvest_thread_result(self.join_handle.take())?;
            return Err(io::Error::new(io::ErrorKind::Other, "Unidentified error."));
        }
        Ok(())
    }

    fn close(self) -> io::Result<()> {
        drop(self.tx);
        harvest_thread_result(self.join_handle)
    }
}

/// Wait for the thread result to terminate and returns its result.
///
/// If the thread panicked, or if the result has already been harvested,
/// returns an explicit error.
fn harvest_thread_result(join_handle_opt: Option<JoinHandle<io::Result<()>>>) -> io::Result<()> {
    let join_handle = join_handle_opt
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Thread already joined."))?;
    join_handle
        .join()
        .map_err(|_err| io::Error::new(io::ErrorKind::Other, "Compressing thread panicked."))?
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::path::Path;

    use crate::directory::RamDirectory;
    use crate::store::store_compressor::BlockCompressor;
    use crate::store::Compressor;
    use crate::Directory;

    fn populate_block_compressor(mut block_compressor: BlockCompressor) -> io::Result<()> {
        block_compressor.compress_block_and_write(b"hello", 1)?;
        block_compressor.compress_block_and_write(b"happy", 1)?;
        block_compressor.close()?;
        Ok(())
    }

    #[test]
    fn test_block_store_compressor_impls_yield_the_same_result() {
        let ram_directory = RamDirectory::default();
        let path1 = Path::new("path1");
        let path2 = Path::new("path2");
        let wrt1 = ram_directory.open_write(path1).unwrap();
        let wrt2 = ram_directory.open_write(path2).unwrap();
        let block_compressor1 = BlockCompressor::new(Compressor::None, wrt1, true).unwrap();
        let block_compressor2 = BlockCompressor::new(Compressor::None, wrt2, false).unwrap();
        populate_block_compressor(block_compressor1).unwrap();
        populate_block_compressor(block_compressor2).unwrap();
        let data1 = ram_directory.open_read(path1).unwrap();
        let data2 = ram_directory.open_read(path2).unwrap();
        assert_eq!(data1.read_bytes().unwrap(), data2.read_bytes().unwrap());
    }
}
