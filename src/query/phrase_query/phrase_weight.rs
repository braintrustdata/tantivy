use super::{phrase_telemetry, PhraseScorer};
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentReader;
use crate::postings::{SegmentPostings, TermInfo};
use crate::query::bm25::Bm25Weight;
use crate::query::explanation::does_not_match;
use crate::query::{EmptyScorer, Explanation, Intersection, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption, Term};
use crate::{DocId, DocSet, Score, TERMINATED};

const PHRASE_PREFLIGHT_MIN_TERMS: usize = 8;

pub struct PhraseWeight {
    phrase_terms: Vec<(usize, Term)>,
    similarity_weight_opt: Option<Bm25Weight>,
    slop: u32,
}

impl PhraseWeight {
    /// Creates a new phrase weight.
    /// If `similarity_weight_opt` is None, then scoring is disabled
    pub fn new(
        phrase_terms: Vec<(usize, Term)>,
        similarity_weight_opt: Option<Bm25Weight>,
    ) -> PhraseWeight {
        let slop = 0;
        PhraseWeight {
            phrase_terms,
            similarity_weight_opt,
            slop,
        }
    }

    fn fieldnorm_reader(&self, reader: &SegmentReader) -> crate::Result<FieldNormReader> {
        let field = self.phrase_terms[0].1.field();
        if self.similarity_weight_opt.is_some() {
            if let Some(fieldnorm_reader) = reader.fieldnorms_readers().get_field(field)? {
                return Ok(fieldnorm_reader);
            }
        }
        Ok(FieldNormReader::constant(reader.max_doc(), 1))
    }

    fn term_infos(
        &self,
        reader: &SegmentReader,
    ) -> crate::Result<Option<Vec<(usize, Field, TermInfo)>>> {
        let mut term_infos = Vec::with_capacity(self.phrase_terms.len());
        for &(offset, ref term) in &self.phrase_terms {
            let field = term.field();
            let inverted_index = reader.inverted_index(field)?;
            let Some(term_info) = inverted_index.get_term_info(term)? else {
                phrase_telemetry::record_missing_term();
                return Ok(None);
            };
            term_infos.push((offset, field, term_info));
        }
        Ok(Some(term_infos))
    }

    fn has_candidate_doc(
        &self,
        reader: &SegmentReader,
        term_infos: &[(usize, Field, TermInfo)],
    ) -> crate::Result<bool> {
        if term_infos.len() < PHRASE_PREFLIGHT_MIN_TERMS {
            return Ok(true);
        }
        phrase_telemetry::record_preflight_attempt(term_infos.len());

        let mut postings = Vec::with_capacity(term_infos.len());
        for (_, field, term_info) in term_infos {
            let inverted_index = reader.inverted_index(*field)?;
            postings.push(
                inverted_index.read_postings_from_terminfo(term_info, IndexRecordOption::Basic)?,
            );
        }

        let has_candidate = Intersection::new(postings).doc() != TERMINATED;
        if has_candidate {
            phrase_telemetry::record_preflight_candidate_found();
        } else {
            phrase_telemetry::record_preflight_no_candidate();
        }
        Ok(has_candidate)
    }

    pub(crate) fn phrase_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<Option<PhraseScorer<SegmentPostings>>> {
        phrase_telemetry::record_scorer_attempt(self.phrase_terms.len());
        let similarity_weight_opt = self
            .similarity_weight_opt
            .as_ref()
            .map(|similarity_weight| similarity_weight.boost_by(boost));
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let Some(term_infos) = self.term_infos(reader)? else {
            return Ok(None);
        };
        if !self.has_candidate_doc(reader, &term_infos)? {
            return Ok(None);
        }

        let mut term_postings_list = Vec::new();
        phrase_telemetry::record_positions_load(term_infos.len());
        for (offset, field, term_info) in term_infos {
            let postings = reader.inverted_index(field)?.read_postings_from_terminfo(
                &term_info,
                IndexRecordOption::WithFreqsAndPositions,
            )?;
            term_postings_list.push((offset, postings));
        }
        Ok(Some(PhraseScorer::new(
            term_postings_list,
            similarity_weight_opt,
            fieldnorm_reader,
            self.slop,
        )))
    }

    pub fn slop(&mut self, slop: u32) {
        self.slop = slop;
    }
}

impl Weight for PhraseWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        if let Some(scorer) = self.phrase_scorer(reader, boost)? {
            Ok(Box::new(scorer))
        } else {
            Ok(Box::new(EmptyScorer))
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let scorer_opt = self.phrase_scorer(reader, 1.0)?;
        if scorer_opt.is_none() {
            return Err(does_not_match(doc));
        }
        let mut scorer = scorer_opt.unwrap();
        if scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let fieldnorm_id = fieldnorm_reader.fieldnorm_id(doc);
        let phrase_count = scorer.phrase_count();
        let mut explanation = Explanation::new("Phrase Scorer", scorer.score());
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            explanation.add_detail(similarity_weight.explain(fieldnorm_id, phrase_count));
        }
        Ok(explanation)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::create_index;
    use crate::docset::TERMINATED;
    use crate::query::{EnableScoring, PhraseQuery};
    use crate::{DocSet, Term};

    #[test]
    pub fn test_phrase_count() -> crate::Result<()> {
        let index = create_index(&["a c", "a a b d a b c", " a b"])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
        ]);
        let enable_scoring = EnableScoring::enabled_from_searcher(&searcher);
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        let mut phrase_scorer = phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .unwrap();
        assert_eq!(phrase_scorer.doc(), 1);
        assert_eq!(phrase_scorer.phrase_count(), 2);
        assert_eq!(phrase_scorer.advance(), 2);
        assert_eq!(phrase_scorer.doc(), 2);
        assert_eq!(phrase_scorer.phrase_count(), 1);
        assert_eq!(phrase_scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    pub fn test_long_phrase_without_candidate_doc() -> crate::Result<()> {
        let index = create_index(&["alpha bravo charlie delta", "echo foxtrot golf hotel"])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(
            [
                "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
            ]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect(),
        );
        let enable_scoring = EnableScoring::disabled_from_schema(searcher.schema());
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        assert!(phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .is_none());
        Ok(())
    }

    #[test]
    pub fn test_long_phrase_with_candidate_doc() -> crate::Result<()> {
        let index = create_index(&[
            "alpha bravo charlie delta echo foxtrot golf hotel",
            "alpha bravo charlie delta",
            "echo foxtrot golf hotel",
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(
            [
                "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
            ]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect(),
        );
        let enable_scoring = EnableScoring::disabled_from_schema(searcher.schema());
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        let mut phrase_scorer = phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .unwrap();
        assert_eq!(phrase_scorer.doc(), 0);
        assert_eq!(phrase_scorer.advance(), TERMINATED);
        Ok(())
    }
}
