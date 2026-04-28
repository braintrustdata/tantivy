use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::BitSet;
use tantivy_fst::Automaton;

use super::phrase_prefix_query::prefix_end;
use crate::index::SegmentReader;
use crate::query::{BitSetDocSet, ConstScorer, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::termdict::{TermDictionary, TermStreamer};
use crate::{DocId, Score, TantivyError};

/// A weight struct for Fuzzy Term and Regex Queries
pub struct AutomatonWeight<A> {
    field: Field,
    automaton: Arc<A>,
    // For JSON fields, the term dictionary include terms from all paths.
    // We apply additional filtering based on the given JSON path, when searching within the term
    // dictionary. This prevents terms from unrelated paths from matching the search criteria.
    json_path_bytes: Option<Box<[u8]>>,
}

impl<A> AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    /// Create a new AutomationWeight
    pub fn new<IntoArcA: Into<Arc<A>>>(field: Field, automaton: IntoArcA) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: None,
        }
    }

    /// Create a new AutomationWeight for a json path
    pub fn new_for_json_path<IntoArcA: Into<Arc<A>>>(
        field: Field,
        automaton: IntoArcA,
        json_path_bytes: &[u8],
    ) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: Some(json_path_bytes.to_vec().into_boxed_slice()),
        }
    }

    fn automaton_stream<'a>(
        &'a self,
        term_dict: &'a TermDictionary,
    ) -> io::Result<TermStreamer<'a, &'a A>> {
        let automaton: &A = &self.automaton;
        let mut term_stream_builder = term_dict.search(automaton);

        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }

        term_stream_builder.into_stream()
    }
}

impl<A> Weight for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let span = tracing::info_span!(
            "tantivy_automaton_weight_scorer",
            field_id = self.field.field_id(),
            max_doc,
            automaton_type = std::any::type_name::<A>(),
            has_json_path = self.json_path_bytes.is_some(),
            json_path_len = self
                .json_path_bytes
                .as_ref()
                .map(|path| path.len())
                .unwrap_or(0),
            matched_terms = tracing::field::Empty,
            posting_lists_read = tracing::field::Empty,
            posting_blocks = tracing::field::Empty,
            posting_docs = tracing::field::Empty,
            bitset_len = tracing::field::Empty,
            inverted_index_open_ns = tracing::field::Empty,
            term_stream_open_ns = tracing::field::Empty,
            term_stream_advance_ns = tracing::field::Empty,
            postings_read_ns = tracing::field::Empty,
            postings_docs_ns = tracing::field::Empty,
            postings_advance_ns = tracing::field::Empty,
            bitset_build_ns = tracing::field::Empty,
            bitset_to_docset_ns = tracing::field::Empty,
        );
        let _guard = span.enter();

        let mut doc_bitset = BitSet::with_max_value(max_doc);
        let inverted_index_start = Instant::now();
        let inverted_index = reader.inverted_index(self.field)?;
        let inverted_index_elapsed = inverted_index_start.elapsed();
        let term_dict = inverted_index.terms();
        let term_stream_start = Instant::now();
        let mut term_stream = self.automaton_stream(term_dict)?;
        let term_stream_elapsed = term_stream_start.elapsed();

        let mut matched_terms = 0u64;
        let mut posting_lists_read = 0u64;
        let mut posting_blocks = 0u64;
        let mut posting_docs = 0u64;
        let mut term_stream_advance_elapsed = Duration::default();
        let mut postings_read_elapsed = Duration::default();
        let mut postings_docs_elapsed = Duration::default();
        let mut postings_advance_elapsed = Duration::default();
        let mut bitset_build_elapsed = Duration::default();

        loop {
            let advance_start = Instant::now();
            let advanced = term_stream.advance();
            term_stream_advance_elapsed += advance_start.elapsed();
            if !advanced {
                break;
            }

            matched_terms += 1;
            let term_info = term_stream.value();
            let postings_read_start = Instant::now();
            let mut block_segment_postings = inverted_index
                .read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
            postings_read_elapsed += postings_read_start.elapsed();
            posting_lists_read += 1;

            loop {
                let docs_start = Instant::now();
                let docs = block_segment_postings.docs();
                postings_docs_elapsed += docs_start.elapsed();
                if docs.is_empty() {
                    break;
                }
                posting_blocks += 1;
                posting_docs += docs.len() as u64;

                let bitset_start = Instant::now();
                for &doc in docs {
                    doc_bitset.insert(doc);
                }
                bitset_build_elapsed += bitset_start.elapsed();

                let postings_advance_start = Instant::now();
                block_segment_postings.advance();
                postings_advance_elapsed += postings_advance_start.elapsed();
            }
        }

        span.record("matched_terms", matched_terms);
        span.record("posting_lists_read", posting_lists_read);
        span.record("posting_blocks", posting_blocks);
        span.record("posting_docs", posting_docs);
        span.record("bitset_len", doc_bitset.len() as u64);
        span.record(
            "inverted_index_open_ns",
            inverted_index_elapsed.as_nanos() as u64,
        );
        span.record("term_stream_open_ns", term_stream_elapsed.as_nanos() as u64);
        span.record(
            "term_stream_advance_ns",
            term_stream_advance_elapsed.as_nanos() as u64,
        );
        span.record("postings_read_ns", postings_read_elapsed.as_nanos() as u64);
        span.record("postings_docs_ns", postings_docs_elapsed.as_nanos() as u64);
        span.record(
            "postings_advance_ns",
            postings_advance_elapsed.as_nanos() as u64,
        );
        span.record("bitset_build_ns", bitset_build_elapsed.as_nanos() as u64);

        let bitset_to_docset_start = Instant::now();
        let doc_bitset = BitSetDocSet::from(doc_bitset);
        span.record(
            "bitset_to_docset_ns",
            bitset_to_docset_start.elapsed().as_nanos() as u64,
        );
        let const_scorer = ConstScorer::new(doc_bitset, boost);
        Ok(Box::new(const_scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) == doc {
            Ok(Explanation::new("AutomatonScorer", 1.0))
        } else {
            Err(TantivyError::InvalidArgument(
                "Document does not exist".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use tantivy_fst::Automaton;

    use super::AutomatonWeight;
    use crate::docset::TERMINATED;
    use crate::query::Weight;
    use crate::schema::{Schema, STRING};
    use crate::{Index, IndexWriter};

    fn create_index() -> crate::Result<Index> {
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(title=>"abc"))?;
        index_writer.add_document(doc!(title=>"bcd"))?;
        index_writer.add_document(doc!(title=>"abcd"))?;
        index_writer.commit()?;
        Ok(index)
    }

    #[derive(Clone, Copy)]
    enum State {
        Start,
        NotMatching,
        AfterA,
    }

    struct PrefixedByA;

    impl Automaton for PrefixedByA {
        type State = State;

        fn start(&self) -> Self::State {
            State::Start
        }

        fn is_match(&self, state: &Self::State) -> bool {
            matches!(*state, State::AfterA)
        }

        fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
            match *state {
                State::Start => {
                    if byte == b'a' {
                        State::AfterA
                    } else {
                        State::NotMatching
                    }
                }
                State::AfterA => State::AfterA,
                State::NotMatching => State::NotMatching,
            }
        }
    }

    #[test]
    fn test_automaton_weight() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), 2u32);
        assert_eq!(scorer.doc(), 2u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_boost() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.32)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.32);
        Ok(())
    }
}
