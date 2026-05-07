use std::io::{self, Write};

use common::{BinarySerializable, CountingWriter};
use once_cell::sync::Lazy;
use tantivy_fst::raw::Fst;
use tantivy_fst::Automaton;

use super::term_info_store::{TermInfoStore, TermInfoStoreWriter};
use super::{TermStreamer, TermStreamerBuilder};
use crate::directory::{FileSlice, OwnedBytes};
use crate::postings::TermInfo;
use crate::termdict::TermOrdinal;

fn convert_fst_error(e: tantivy_fst::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
}

const FST_VERSION: u32 = 1;

/// Builder for the new term dictionary.
///
/// Inserting must be done in the order of the `keys`.
pub struct TermDictionaryBuilder<W> {
    fst_builder: tantivy_fst::MapBuilder<W>,
    term_info_store_writer: TermInfoStoreWriter,
    term_ord: u64,
}

impl<W> TermDictionaryBuilder<W>
where
    W: Write,
{
    /// Creates a new `TermDictionaryBuilder`
    pub fn create(w: W) -> io::Result<Self> {
        let fst_builder = tantivy_fst::MapBuilder::new(w).map_err(convert_fst_error)?;
        Ok(TermDictionaryBuilder {
            fst_builder,
            term_info_store_writer: TermInfoStoreWriter::new(),
            term_ord: 0,
        })
    }

    /// Inserts a `(key, value)` pair in the term dictionary.
    ///
    /// *Keys have to be inserted in order.*
    pub fn insert<K: AsRef<[u8]>>(&mut self, key_ref: K, value: &TermInfo) -> io::Result<()> {
        let key = key_ref.as_ref();
        self.insert_key(key)?;
        self.insert_value(value)?;
        Ok(())
    }

    /// # Warning
    /// Horribly dangerous internal API
    ///
    /// If used, it must be used by systematically alternating calls
    /// to insert_key and insert_value.
    ///
    /// Prefer using `.insert(key, value)`
    pub fn insert_key(&mut self, key: &[u8]) -> io::Result<()> {
        self.fst_builder
            .insert(key, self.term_ord)
            .map_err(convert_fst_error)?;
        self.term_ord += 1;
        Ok(())
    }

    /// # Warning
    ///
    /// Horribly dangerous internal API. See `.insert_key(...)`.
    pub fn insert_value(&mut self, term_info: &TermInfo) -> io::Result<()> {
        self.term_info_store_writer.write_term_info(term_info)?;
        Ok(())
    }

    /// Finalize writing the builder, and returns the underlying
    /// `Write` object.
    pub fn finish(mut self) -> io::Result<W> {
        let mut file = self.fst_builder.into_inner().map_err(convert_fst_error)?;
        {
            let mut counting_writer = CountingWriter::wrap(&mut file);
            self.term_info_store_writer
                .serialize(&mut counting_writer)?;
            let footer_size = counting_writer.written_bytes();
            footer_size.serialize(&mut counting_writer)?;
            FST_VERSION.serialize(&mut counting_writer)?;
        }
        Ok(file)
    }
}

fn open_fst_index(fst_file: FileSlice) -> io::Result<tantivy_fst::Map<OwnedBytes>> {
    let span = tracing::info_span!(
        "op",
        otel.name = "FST term dictionary open fst index",
        span_lineage_metrics = true,
        fst_termdict_fst_bytes = fst_file.num_bytes().get_bytes(),
    );
    let _guard = span.enter();
    let bytes = tracing::info_span!(
        "op",
        otel.name = "FST term dictionary read fst bytes",
        span_lineage_metrics = true,
        fst_termdict_fst_bytes = fst_file.num_bytes().get_bytes(),
    )
    .in_scope(|| fst_file.read_bytes())?;
    let fst = tracing::info_span!(
        "op",
        otel.name = "FST term dictionary parse fst bytes",
        span_lineage_metrics = true,
        fst_termdict_fst_bytes = bytes.len() as u64,
    )
    .in_scope(|| {
        Fst::new(bytes).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Fst data is corrupted: {:?}", err),
            )
        })
    })?;
    Ok(tantivy_fst::Map::from(fst))
}

static EMPTY_TERM_DICT_FILE: Lazy<FileSlice> = Lazy::new(|| {
    let term_dictionary_data: Vec<u8> = TermDictionaryBuilder::create(Vec::<u8>::new())
        .expect("Creating a TermDictionaryBuilder in a Vec<u8> should never fail")
        .finish()
        .expect("Writing in a Vec<u8> should never fail");
    FileSlice::from(term_dictionary_data)
});

/// The term dictionary contains all of the terms in
/// `tantivy index` in a sorted manner.
///
/// The `Fst` crate is used to associate terms to their
/// respective `TermOrdinal`. The `TermInfoStore` then makes it
/// possible to fetch the associated `TermInfo`.
pub struct TermDictionary {
    fst_index: tantivy_fst::Map<OwnedBytes>,
    term_info_store: TermInfoStore,
}

impl TermDictionary {
    /// Opens a `TermDictionary`.
    pub fn open(file: FileSlice) -> io::Result<Self> {
        let span = tracing::info_span!(
            "op",
            otel.name = "FST term dictionary open",
            span_lineage_metrics = true,
            fst_termdict_file_bytes = file.num_bytes().get_bytes(),
            fst_termdict_footer_bytes = 12u64,
            fst_termdict_fst_bytes = tracing::field::Empty,
            fst_termdict_values_bytes = tracing::field::Empty,
        );
        let _guard = span.enter();
        let (main_slice, footer_len_slice) = file.split_from_end(12);
        let mut footer_len_bytes = tracing::info_span!(
            "op",
            otel.name = "FST term dictionary read footer",
            span_lineage_metrics = true,
            fst_termdict_footer_bytes = footer_len_slice.num_bytes().get_bytes(),
        )
        .in_scope(|| footer_len_slice.read_bytes())?;
        let footer_size = u64::deserialize(&mut footer_len_bytes)?;
        let version = u32::deserialize(&mut footer_len_bytes)?;
        if version != FST_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Unsuported fst version, expected {version}, found {FST_VERSION}",),
            ));
        }

        let (fst_file_slice, values_file_slice) = main_slice.split_from_end(footer_size as usize);
        span.record(
            "fst_termdict_fst_bytes",
            fst_file_slice.num_bytes().get_bytes(),
        );
        span.record(
            "fst_termdict_values_bytes",
            values_file_slice.num_bytes().get_bytes(),
        );
        let fst_index = open_fst_index(fst_file_slice)?;
        let term_info_store = TermInfoStore::open(values_file_slice)?;
        Ok(TermDictionary {
            fst_index,
            term_info_store,
        })
    }

    /// Creates an empty term dictionary which contains no terms.
    pub fn empty() -> Self {
        TermDictionary::open(EMPTY_TERM_DICT_FILE.clone()).unwrap()
    }

    /// Returns the number of terms in the dictionary.
    /// Term ordinals range from 0 to `num_terms() - 1`.
    pub fn num_terms(&self) -> usize {
        self.term_info_store.num_terms()
    }

    /// Returns the ordinal associated with a given term.
    pub fn term_ord<K: AsRef<[u8]>>(&self, key: K) -> io::Result<Option<TermOrdinal>> {
        Ok(self.fst_index.get(key))
    }

    /// Stores the term associated with a given term ordinal in
    /// a `bytes` buffer.
    ///
    /// Term ordinals are defined as the position of the term in
    /// the sorted list of terms.
    ///
    /// Returns true if and only if the term has been found.
    ///
    /// Regardless of whether the term is found or not,
    /// the buffer may be modified.
    pub fn ord_to_term(&self, mut ord: TermOrdinal, bytes: &mut Vec<u8>) -> io::Result<bool> {
        bytes.clear();
        let fst = self.fst_index.as_fst();
        let mut node = fst.root();
        while ord != 0 || !node.is_final() {
            if let Some(transition) = node
                .transitions()
                .take_while(|transition| transition.out.value() <= ord)
                .last()
            {
                ord -= transition.out.value();
                bytes.push(transition.inp);
                let new_node_addr = transition.addr;
                node = fst.node(new_node_addr);
            } else {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Returns the number of terms in the dictionary.
    pub fn term_info_from_ord(&self, term_ord: TermOrdinal) -> TermInfo {
        self.term_info_store.get(term_ord)
    }

    /// Lookups the value corresponding to the key.
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> io::Result<Option<TermInfo>> {
        let key = key.as_ref();
        let span = tracing::info_span!(
            "op",
            otel.name = "FST term dictionary get",
            span_lineage_metrics = true,
            fst_termdict_key_bytes = key.len() as u64,
            fst_termdict_result = tracing::field::Empty,
            fst_termdict_term_ord = tracing::field::Empty,
            fst_termdict_doc_freq = tracing::field::Empty,
            fst_termdict_postings_bytes = tracing::field::Empty,
            fst_termdict_positions_bytes = tracing::field::Empty,
        );
        let _guard = span.enter();
        let term_ord = tracing::info_span!(
            "op",
            otel.name = "FST term dictionary term ord",
            span_lineage_metrics = true,
            fst_termdict_key_bytes = key.len() as u64,
        )
        .in_scope(|| self.term_ord(key))?;
        let Some(term_ord) = term_ord else {
            span.record("fst_termdict_result", "missing");
            return Ok(None);
        };
        span.record("fst_termdict_term_ord", term_ord);
        let term_info = tracing::info_span!(
            "op",
            otel.name = "FST term dictionary term info from ord",
            span_lineage_metrics = true,
            fst_termdict_term_ord = term_ord,
        )
        .in_scope(|| self.term_info_from_ord(term_ord));
        span.record("fst_termdict_result", "found");
        span.record("fst_termdict_doc_freq", term_info.doc_freq as u64);
        span.record(
            "fst_termdict_postings_bytes",
            term_info.postings_range.len() as u64,
        );
        span.record(
            "fst_termdict_positions_bytes",
            term_info.positions_range.len() as u64,
        );
        Ok(Some(term_info))
    }

    /// Returns a range builder, to stream all of the terms
    /// within an interval.
    pub fn range(&self) -> TermStreamerBuilder<'_> {
        TermStreamerBuilder::new(self, self.fst_index.range())
    }

    /// A stream of all the sorted terms.
    pub fn stream(&self) -> io::Result<TermStreamer<'_>> {
        self.range().into_stream()
    }

    /// Returns a search builder, to stream all of the terms
    /// within the Automaton
    pub fn search<'a, A: Automaton + 'a>(&'a self, automaton: A) -> TermStreamerBuilder<'a, A> {
        let stream_builder = self.fst_index.search(automaton);
        TermStreamerBuilder::<A>::new(self, stream_builder)
    }
}
