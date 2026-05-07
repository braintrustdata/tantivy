use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use super::{PhraseQueryStats, PhraseScorer};
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentReader;
use crate::postings::{Postings, SegmentPostings, TermInfo};
use crate::query::bm25::Bm25Weight;
use crate::query::explanation::does_not_match;
use crate::query::{EmptyScorer, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption, Term};
use crate::{DocId, DocSet, Score, TERMINATED};

const DEFAULT_PREFLIGHT_MIN_TERMS: usize = 8;
const ADAPTIVE_LOOKUP_MIN_OBSERVATIONS: u64 = 16;
const ADAPTIVE_LOOKUP_MIN_MISS_NUMERATOR: u64 = 1;
const ADAPTIVE_LOOKUP_MIN_MISS_DENOMINATOR: u64 = 2;
const ATOMIC_ORDERING: Ordering = Ordering::Relaxed;

#[derive(Default)]
struct PhraseTermLookupStats {
    lookups: AtomicU64,
    misses: AtomicU64,
}

struct PreloadedPhrasePair {
    left_idx: usize,
    right_idx: usize,
    left_postings: Option<SegmentPostings>,
    right_postings: Option<SegmentPostings>,
}

enum PhrasePairPreflight {
    Skipped,
    NoCandidate,
    Candidate(PreloadedPhrasePair),
}

struct PhrasePairCandidateCheck {
    has_candidate: bool,
    candidate_docs: u64,
    position_reads: u64,
    position_values: u64,
}

pub struct PhraseWeight {
    phrase_terms: Vec<(usize, Term)>,
    lookup_stats: Vec<PhraseTermLookupStats>,
    similarity_weight_opt: Option<Bm25Weight>,
    slop: u32,
    preflight_min_terms: usize,
    stats: Arc<PhraseQueryStats>,
}

impl PhraseWeight {
    /// Creates a new phrase weight.
    /// If `similarity_weight_opt` is None, then scoring is disabled
    pub fn new(
        phrase_terms: Vec<(usize, Term)>,
        similarity_weight_opt: Option<Bm25Weight>,
        stats: Arc<PhraseQueryStats>,
    ) -> PhraseWeight {
        let slop = 0;
        let lookup_stats = (0..phrase_terms.len())
            .map(|_| PhraseTermLookupStats::default())
            .collect();
        PhraseWeight {
            phrase_terms,
            lookup_stats,
            similarity_weight_opt,
            slop,
            preflight_min_terms: DEFAULT_PREFLIGHT_MIN_TERMS,
            stats,
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
        let span = tracing::info_span!(
            "op",
            otel.name = "Phrase query term infos",
            span_lineage_metrics = true,
            phrase_query_terms = self.phrase_terms.len() as u64,
            phrase_query_lookup_steps = tracing::field::Empty,
            phrase_query_reused_terms = tracing::field::Empty,
            phrase_query_terms_found = tracing::field::Empty,
            phrase_query_missing_term_idx = tracing::field::Empty,
            phrase_query_result = tracing::field::Empty,
        );
        let _guard = span.enter();
        let mut term_infos_by_position: Vec<Option<(usize, Field, TermInfo)>> =
            (0..self.phrase_terms.len()).map(|_| None).collect();
        let mut found_term_infos: Vec<(&Term, Field, TermInfo)> = Vec::new();
        let lookup_order = term_info_lookup_order(&self.phrase_terms, &self.lookup_stats);
        let mut lookup_steps = 0u64;
        let mut reused_terms = 0u64;
        for term_idx in lookup_order {
            let (offset, term) = &self.phrase_terms[term_idx];
            if let Some((_, field, term_info)) = found_term_infos
                .iter()
                .find(|(found_term, _, _)| *found_term == term)
            {
                reused_terms += 1;
                term_infos_by_position[term_idx] = Some((*offset, *field, term_info.clone()));
                continue;
            }
            let field = term.field();
            lookup_steps += 1;
            let inverted_index = reader.inverted_index(field)?;
            let term_info = inverted_index.get_term_info(term)?;
            self.lookup_stats[term_idx]
                .lookups
                .fetch_add(1, ATOMIC_ORDERING);
            let Some(term_info) = term_info else {
                self.lookup_stats[term_idx]
                    .misses
                    .fetch_add(1, ATOMIC_ORDERING);
                self.stats.term_info_missing.fetch_add(1, ATOMIC_ORDERING);
                span.record("phrase_query_lookup_steps", lookup_steps);
                span.record("phrase_query_reused_terms", reused_terms);
                span.record("phrase_query_terms_found", found_term_infos.len() as u64);
                span.record("phrase_query_missing_term_idx", term_idx as u64);
                span.record("phrase_query_result", "missing");
                return Ok(None);
            };
            found_term_infos.push((term, field, term_info.clone()));
            term_infos_by_position[term_idx] = Some((*offset, field, term_info));
        }
        span.record("phrase_query_lookup_steps", lookup_steps);
        span.record("phrase_query_reused_terms", reused_terms);
        span.record("phrase_query_terms_found", found_term_infos.len() as u64);
        span.record("phrase_query_result", "found");
        Ok(Some(
            term_infos_by_position
                .into_iter()
                .map(|term_info| term_info.expect("term info populated for every phrase term"))
                .collect(),
        ))
    }

    fn phrase_pair_preflight(
        &self,
        reader: &SegmentReader,
        term_infos: &[(usize, Field, TermInfo)],
    ) -> crate::Result<PhrasePairPreflight> {
        let span = tracing::info_span!(
            "op",
            otel.name = "Phrase query pair preflight",
            span_lineage_metrics = true,
            phrase_query_terms = term_infos.len() as u64,
            phrase_query_slop = self.slop as u64,
            phrase_query_preflight_min_terms = self.preflight_min_terms as u64,
            phrase_query_preflight_result = tracing::field::Empty,
            phrase_query_preflight_left_idx = tracing::field::Empty,
            phrase_query_preflight_right_idx = tracing::field::Empty,
            phrase_query_preflight_postings_bytes = tracing::field::Empty,
            phrase_query_preflight_positions_bytes = tracing::field::Empty,
            phrase_query_preflight_candidate_docs = tracing::field::Empty,
            phrase_query_preflight_position_reads = tracing::field::Empty,
            phrase_query_preflight_position_values = tracing::field::Empty,
        );
        let _guard = span.enter();
        if self.slop != 0 {
            self.stats
                .preflight_skipped_slop
                .fetch_add(1, ATOMIC_ORDERING);
            span.record("phrase_query_preflight_result", "skipped_slop");
            return Ok(PhrasePairPreflight::Skipped);
        }
        if term_infos.len() < self.preflight_min_terms {
            self.stats
                .preflight_skipped_too_short
                .fetch_add(1, ATOMIC_ORDERING);
            span.record("phrase_query_preflight_result", "skipped_too_short");
            return Ok(PhrasePairPreflight::Skipped);
        }

        let Some((left_idx, right_idx)) = select_cheapest_phrase_pair(term_infos) else {
            span.record("phrase_query_preflight_result", "skipped_no_pair");
            return Ok(PhrasePairPreflight::Skipped);
        };
        self.stats.preflight_attempts.fetch_add(1, ATOMIC_ORDERING);

        let (left_offset, left_field, left_term_info) = &term_infos[left_idx];
        let (right_offset, right_field, right_term_info) = &term_infos[right_idx];
        span.record("phrase_query_preflight_left_idx", left_idx as u64);
        span.record("phrase_query_preflight_right_idx", right_idx as u64);
        span.record(
            "phrase_query_preflight_postings_bytes",
            (left_term_info.postings_range.len() + right_term_info.postings_range.len()) as u64,
        );
        span.record(
            "phrase_query_preflight_positions_bytes",
            (left_term_info.positions_range.len() + right_term_info.positions_range.len()) as u64,
        );
        let left_inverted_index = reader.inverted_index(*left_field)?;
        let right_inverted_index = reader.inverted_index(*right_field)?;
        let left_postings = left_inverted_index.read_postings_from_terminfo(
            left_term_info,
            IndexRecordOption::WithFreqsAndPositions,
        )?;
        let right_postings = right_inverted_index.read_postings_from_terminfo(
            right_term_info,
            IndexRecordOption::WithFreqsAndPositions,
        )?;
        let left_postings_for_scorer = left_postings.clone();
        let right_postings_for_scorer = right_postings.clone();
        let has_candidate = phrase_pair_has_candidate(
            left_postings,
            right_postings,
            *left_offset as u32,
            *right_offset as u32,
        );
        span.record(
            "phrase_query_preflight_candidate_docs",
            has_candidate.candidate_docs,
        );
        span.record(
            "phrase_query_preflight_position_reads",
            has_candidate.position_reads,
        );
        span.record(
            "phrase_query_preflight_position_values",
            has_candidate.position_values,
        );
        if has_candidate.has_candidate {
            self.stats.preflight_candidate.fetch_add(1, ATOMIC_ORDERING);
            span.record("phrase_query_preflight_result", "candidate");
            Ok(PhrasePairPreflight::Candidate(PreloadedPhrasePair {
                left_idx,
                right_idx,
                left_postings: Some(left_postings_for_scorer),
                right_postings: Some(right_postings_for_scorer),
            }))
        } else {
            self.stats
                .preflight_no_candidate
                .fetch_add(1, ATOMIC_ORDERING);
            span.record("phrase_query_preflight_result", "no_candidate");
            Ok(PhrasePairPreflight::NoCandidate)
        }
    }

    pub(crate) fn phrase_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<Option<PhraseScorer<SegmentPostings>>> {
        let scorer_span = tracing::info_span!(
            "op",
            otel.name = "Phrase query scorer",
            span_lineage_metrics = true,
            phrase_query_terms = self.phrase_terms.len() as u64,
            phrase_query_slop = self.slop as u64,
            phrase_query_preflight_min_terms = self.preflight_min_terms as u64,
            phrase_query_result = tracing::field::Empty,
        );
        let _scorer_guard = scorer_span.enter();
        let similarity_weight_opt = self
            .similarity_weight_opt
            .as_ref()
            .map(|similarity_weight| similarity_weight.boost_by(boost));
        self.stats.scorer_attempts.fetch_add(1, ATOMIC_ORDERING);
        let fieldnorm_reader = tracing::info_span!(
            "op",
            otel.name = "Phrase query fieldnorm reader",
            span_lineage_metrics = true,
        )
        .in_scope(|| self.fieldnorm_reader(reader))?;
        let Some(term_infos) = self.term_infos(reader)? else {
            scorer_span.record("phrase_query_result", "missing_terms");
            return Ok(None);
        };
        let mut preloaded_pair = match self.phrase_pair_preflight(reader, &term_infos)? {
            PhrasePairPreflight::Skipped => None,
            PhrasePairPreflight::NoCandidate => {
                scorer_span.record("phrase_query_result", "preflight_no_candidate");
                return Ok(None);
            }
            PhrasePairPreflight::Candidate(preloaded_pair) => Some(preloaded_pair),
        };

        let mut term_postings_list = Vec::new();
        let postings_span = tracing::info_span!(
            "op",
            otel.name = "Phrase query open scorer postings",
            span_lineage_metrics = true,
            phrase_query_terms = term_infos.len() as u64,
            phrase_query_terms_opened = tracing::field::Empty,
            phrase_query_terms_preloaded = tracing::field::Empty,
            phrase_query_postings_bytes = tracing::field::Empty,
            phrase_query_positions_bytes = tracing::field::Empty,
            phrase_query_fresh_postings_bytes = tracing::field::Empty,
            phrase_query_fresh_positions_bytes = tracing::field::Empty,
        );
        {
            let _postings_guard = postings_span.enter();
            let mut terms_opened = 0u64;
            let mut terms_preloaded = 0u64;
            let mut postings_bytes = 0u64;
            let mut positions_bytes = 0u64;
            let mut fresh_postings_bytes = 0u64;
            let mut fresh_positions_bytes = 0u64;
            for (idx, (offset, field, term_info)) in term_infos.into_iter().enumerate() {
                postings_bytes += term_info.postings_range.len() as u64;
                positions_bytes += term_info.positions_range.len() as u64;
                let postings = match preloaded_pair.as_mut() {
                    Some(pair) if idx == pair.left_idx => {
                        terms_preloaded += 1;
                        pair.left_postings.take().expect("left postings preloaded")
                    }
                    Some(pair) if idx == pair.right_idx => {
                        terms_preloaded += 1;
                        pair.right_postings
                            .take()
                            .expect("right postings preloaded")
                    }
                    _ => {
                        terms_opened += 1;
                        fresh_postings_bytes += term_info.postings_range.len() as u64;
                        fresh_positions_bytes += term_info.positions_range.len() as u64;
                        reader.inverted_index(field)?.read_postings_from_terminfo(
                            &term_info,
                            IndexRecordOption::WithFreqsAndPositions,
                        )?
                    }
                };
                term_postings_list.push((offset, postings));
            }
            postings_span.record("phrase_query_terms_opened", terms_opened);
            postings_span.record("phrase_query_terms_preloaded", terms_preloaded);
            postings_span.record("phrase_query_postings_bytes", postings_bytes);
            postings_span.record("phrase_query_positions_bytes", positions_bytes);
            postings_span.record("phrase_query_fresh_postings_bytes", fresh_postings_bytes);
            postings_span.record("phrase_query_fresh_positions_bytes", fresh_positions_bytes);
        }
        let phrase_scorer = PhraseScorer::new(
            term_postings_list,
            similarity_weight_opt,
            fieldnorm_reader,
            self.slop,
        );
        if phrase_scorer.doc() == TERMINATED {
            scorer_span.record("phrase_query_result", "terminated");
            Ok(None)
        } else {
            scorer_span.record("phrase_query_result", "ready");
            Ok(Some(phrase_scorer))
        }
    }

    pub fn slop(&mut self, slop: u32) {
        self.slop = slop;
    }

    /// Sets the minimum phrase length for the exact-pair preflight check.
    ///
    /// The default is 8 terms. Values less than 2 are clamped to 2.
    pub fn set_preflight_min_terms(&mut self, value: usize) {
        self.preflight_min_terms = value.max(2);
    }
}

fn static_term_info_lookup_order(phrase_terms: &[(usize, Term)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..phrase_terms.len()).collect();
    order.sort_by(|&left_idx, &right_idx| {
        let left_term = &phrase_terms[left_idx].1;
        let right_term = &phrase_terms[right_idx].1;
        right_term
            .serialized_value_bytes()
            .len()
            .cmp(&left_term.serialized_value_bytes().len())
            .then_with(|| left_idx.cmp(&right_idx))
    });
    order
}

fn term_info_lookup_order(
    phrase_terms: &[(usize, Term)],
    lookup_stats: &[PhraseTermLookupStats],
) -> Vec<usize> {
    let mut order = static_term_info_lookup_order(phrase_terms);
    order.sort_by(|&left_idx, &right_idx| {
        let left_priority =
            adaptive_lookup_priority(&phrase_terms[left_idx].1, &lookup_stats[left_idx]);
        let right_priority =
            adaptive_lookup_priority(&phrase_terms[right_idx].1, &lookup_stats[right_idx]);
        right_priority
            .cmp(&left_priority)
            .then_with(|| left_idx.cmp(&right_idx))
    });
    order
}

fn adaptive_lookup_priority(term: &Term, stats: &PhraseTermLookupStats) -> (u8, u128, usize) {
    let lookups = stats.lookups.load(ATOMIC_ORDERING);
    let misses = stats.misses.load(ATOMIC_ORDERING);
    let term_len = term.serialized_value_bytes().len();
    if is_adaptive_lookup_term_from_counts(lookups, misses) {
        return (
            1,
            (misses as u128) * 1_000_000u128 / (lookups as u128),
            term_len,
        );
    }
    (0, term_len as u128, term_len)
}

fn is_adaptive_lookup_term_from_counts(lookups: u64, misses: u64) -> bool {
    lookups >= ADAPTIVE_LOOKUP_MIN_OBSERVATIONS
        && misses.saturating_mul(ADAPTIVE_LOOKUP_MIN_MISS_DENOMINATOR)
            >= lookups.saturating_mul(ADAPTIVE_LOOKUP_MIN_MISS_NUMERATOR)
}

fn select_cheapest_phrase_pair(term_infos: &[(usize, Field, TermInfo)]) -> Option<(usize, usize)> {
    let mut best_pair = None;
    let mut best_cost = usize::MAX;
    for left_idx in 0..term_infos.len() {
        for right_idx in left_idx + 1..term_infos.len() {
            let left_info = &term_infos[left_idx].2;
            let right_info = &term_infos[right_idx].2;
            let cost = left_info.positions_range.len()
                + right_info.positions_range.len()
                + left_info.postings_range.len()
                + right_info.postings_range.len();
            if cost < best_cost {
                best_cost = cost;
                best_pair = Some((left_idx, right_idx));
            }
        }
    }
    best_pair
}

fn phrase_pair_has_candidate(
    mut left: SegmentPostings,
    mut right: SegmentPostings,
    left_offset: u32,
    right_offset: u32,
) -> PhrasePairCandidateCheck {
    let max_offset = left_offset.max(right_offset);
    let left_position_offset = max_offset - left_offset;
    let right_position_offset = max_offset - right_offset;
    let mut left_positions = Vec::new();
    let mut right_positions = Vec::new();
    let mut candidate_docs = 0u64;
    let mut position_reads = 0u64;
    let mut position_values = 0u64;
    let mut candidate = left.doc().max(right.doc());
    loop {
        if candidate == TERMINATED {
            return PhrasePairCandidateCheck {
                has_candidate: false,
                candidate_docs,
                position_reads,
                position_values,
            };
        }
        let left_doc = left.seek(candidate);
        let right_doc = right.seek(candidate);
        if left_doc == TERMINATED || right_doc == TERMINATED {
            return PhrasePairCandidateCheck {
                has_candidate: false,
                candidate_docs,
                position_reads,
                position_values,
            };
        }
        if left_doc == right_doc {
            candidate_docs += 1;
            left.positions_with_offset(left_position_offset, &mut left_positions);
            position_reads += 1;
            position_values += left_positions.len() as u64;
            right.positions_with_offset(right_position_offset, &mut right_positions);
            position_reads += 1;
            position_values += right_positions.len() as u64;
            if positions_intersect(&left_positions, &right_positions) {
                return PhrasePairCandidateCheck {
                    has_candidate: true,
                    candidate_docs,
                    position_reads,
                    position_values,
                };
            }
            candidate = left.advance().max(right.doc());
        } else {
            candidate = left_doc.max(right_doc);
        }
    }
}

fn positions_intersect(left: &[u32], right: &[u32]) -> bool {
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Greater => right_index += 1,
        }
    }
    false
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
