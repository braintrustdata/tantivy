use std::time::Instant;

use super::term_scorer::TermScorer;
use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN};
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentReader;
use crate::postings::SegmentPostings;
use crate::query::bm25::Bm25Weight;
use crate::query::explanation::does_not_match;
use crate::query::weight::{for_each_docset_buffered, for_each_scorer};
use crate::query::{Explanation, Scorer, Weight};
use crate::schema::IndexRecordOption;
use crate::{DocId, Score, Term};

pub struct TermWeight {
    term: Term,
    index_record_option: IndexRecordOption,
    similarity_weight: Bm25Weight,
    scoring_enabled: bool,
}

impl Weight for TermWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let term_scorer = self.specialized_scorer(reader, boost)?;
        Ok(Box::new(term_scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.specialized_scorer(reader, 1.0)?;
        if scorer.doc() > doc || scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let mut explanation = scorer.explain();
        explanation.add_context(format!("Term={:?}", self.term,));
        Ok(explanation)
    }

    fn count(&self, reader: &SegmentReader) -> crate::Result<u32> {
        if let Some(alive_bitset) = reader.alive_bitset() {
            Ok(self.scorer(reader, 1.0)?.count(alive_bitset))
        } else {
            let field = self.term.field();
            let inv_index = reader.inverted_index(field)?;
            let term_info = inv_index.get_term_info(&self.term)?;
            Ok(term_info.map(|term_info| term_info.doc_freq).unwrap_or(0))
        }
    }

    /// Iterates through all of the document matched by the DocSet
    /// `DocSet` and push the scored documents to the collector.
    fn for_each(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score),
    ) -> crate::Result<()> {
        let mut scorer = self.specialized_scorer(reader, 1.0)?;
        for_each_scorer(&mut scorer, callback);
        Ok(())
    }

    /// Iterates through all of the document matched by the DocSet
    /// `DocSet` and push the scored documents to the collector.
    fn for_each_no_score(
        &self,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(&[DocId]),
    ) -> crate::Result<()> {
        let field = self.term.field();
        let span = tracing::info_span!(
            "tantivy_term_weight_for_each_no_score",
            field_id = field.field_id(),
            term_type = ?self.term.typ(),
            term_value_bytes = self.term.serialized_value_bytes().len(),
            max_doc = reader.max_doc(),
            scoring_enabled = self.scoring_enabled,
            index_record_option = ?self.index_record_option,
            scorer_build_ns = tracing::field::Empty,
            iterate_ns = tracing::field::Empty,
            blocks_seen = tracing::field::Empty,
            docs_seen = tracing::field::Empty,
        );
        let _guard = span.enter();

        let scorer_start = Instant::now();
        let mut scorer = self.specialized_scorer(reader, 1.0)?;
        span.record("scorer_build_ns", scorer_start.elapsed().as_nanos() as u64);

        let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
        let mut blocks_seen = 0u64;
        let mut docs_seen = 0u64;
        let iterate_start = Instant::now();
        for_each_docset_buffered(&mut scorer, &mut buffer, &mut |docs: &[DocId]| {
            blocks_seen += 1;
            docs_seen += docs.len() as u64;
            callback(docs);
        });
        span.record("iterate_ns", iterate_start.elapsed().as_nanos() as u64);
        span.record("blocks_seen", blocks_seen);
        span.record("docs_seen", docs_seen);
        Ok(())
    }

    /// Calls `callback` with all of the `(doc, score)` for which score
    /// is exceeding a given threshold.
    ///
    /// This method is useful for the TopDocs collector.
    /// For all docsets, the blanket implementation has the benefit
    /// of prefiltering (doc, score) pairs, avoiding the
    /// virtual dispatch cost.
    ///
    /// More importantly, it makes it possible for scorers to implement
    /// important optimization (e.g. BlockWAND for union).
    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        let scorer = self.specialized_scorer(reader, 1.0)?;
        crate::query::boolean_query::block_wand_single_scorer(scorer, threshold, callback);
        Ok(())
    }
}

impl TermWeight {
    pub fn new(
        term: Term,
        index_record_option: IndexRecordOption,
        similarity_weight: Bm25Weight,
        scoring_enabled: bool,
    ) -> TermWeight {
        TermWeight {
            term,
            index_record_option,
            similarity_weight,
            scoring_enabled,
        }
    }

    pub fn term(&self) -> &Term {
        &self.term
    }

    pub(crate) fn specialized_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<TermScorer> {
        let field = self.term.field();
        let span = tracing::info_span!(
            "tantivy_term_weight_scorer",
            field_id = field.field_id(),
            term_type = ?self.term.typ(),
            term_value_bytes = self.term.serialized_value_bytes().len(),
            max_doc = reader.max_doc(),
            scoring_enabled = self.scoring_enabled,
            index_record_option = ?self.index_record_option,
            found = tracing::field::Empty,
            doc_freq = tracing::field::Empty,
            postings_bytes = tracing::field::Empty,
            positions_bytes = tracing::field::Empty,
            inverted_index_open_ns = tracing::field::Empty,
            fieldnorm_open_ns = tracing::field::Empty,
            term_info_ns = tracing::field::Empty,
            postings_read_ns = tracing::field::Empty,
            scorer_build_ns = tracing::field::Empty,
        );
        let _guard = span.enter();

        let inverted_index_start = Instant::now();
        let inverted_index = reader.inverted_index(field)?;
        span.record(
            "inverted_index_open_ns",
            inverted_index_start.elapsed().as_nanos() as u64,
        );

        let fieldnorm_start = Instant::now();
        let fieldnorm_reader_opt = if self.scoring_enabled {
            reader.fieldnorms_readers().get_field(field)?
        } else {
            None
        };
        let fieldnorm_reader =
            fieldnorm_reader_opt.unwrap_or_else(|| FieldNormReader::constant(reader.max_doc(), 1));
        span.record(
            "fieldnorm_open_ns",
            fieldnorm_start.elapsed().as_nanos() as u64,
        );

        let similarity_weight = self.similarity_weight.boost_by(boost);

        let term_info_start = Instant::now();
        let term_info_opt = inverted_index.get_term_info(&self.term)?;
        span.record("term_info_ns", term_info_start.elapsed().as_nanos() as u64);

        if let Some(term_info) = term_info_opt {
            span.record("found", true);
            span.record("doc_freq", term_info.doc_freq);
            span.record("postings_bytes", term_info.postings_range.len() as u64);
            span.record("positions_bytes", term_info.positions_range.len() as u64);

            let postings_start = Instant::now();
            let segment_postings =
                inverted_index.read_postings_from_terminfo(&term_info, self.index_record_option)?;
            span.record(
                "postings_read_ns",
                postings_start.elapsed().as_nanos() as u64,
            );

            let scorer_build_start = Instant::now();
            let scorer = TermScorer::new(segment_postings, fieldnorm_reader, similarity_weight);
            span.record(
                "scorer_build_ns",
                scorer_build_start.elapsed().as_nanos() as u64,
            );
            Ok(scorer)
        } else {
            span.record("found", false);
            let scorer_build_start = Instant::now();
            let scorer = TermScorer::new(
                SegmentPostings::empty(),
                fieldnorm_reader,
                similarity_weight,
            );
            span.record(
                "scorer_build_ns",
                scorer_build_start.elapsed().as_nanos() as u64,
            );
            Ok(scorer)
        }
    }
}
