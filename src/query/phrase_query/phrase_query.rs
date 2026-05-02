use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use super::PhraseWeight;
use crate::query::bm25::Bm25Weight;
use crate::query::{EnableScoring, Query, Weight};
use crate::schema::{Field, IndexRecordOption, Term};

/// `PhraseQuery` matches a specific sequence of words.
///
/// For instance the phrase query for `"part time"` will match
/// the sentence
///
/// **Alan just got a part time job.**
///
/// On the other hand it will not match the sentence.
///
/// **This is my favorite part of the job.**
///
/// [Slop](PhraseQuery::set_slop) allows leniency in term proximity
/// for some performance tradeof.
///
/// Using a `PhraseQuery` on a field requires positions
/// to be indexed for this field.
#[derive(Clone, Debug)]
pub struct PhraseQuery {
    field: Field,
    phrase_terms: Vec<(usize, Term)>,
    slop: u32,
    preflight_min_terms: usize,
    stats: Arc<PhraseQueryStats>,
}

const DEFAULT_PREFLIGHT_MIN_TERMS: usize = 8;
const ATOMIC_ORDERING: Ordering = Ordering::Relaxed;

#[derive(Default, Debug)]
pub(crate) struct PhraseQueryStats {
    pub(crate) scorer_attempts: AtomicU64,
    pub(crate) term_info_missing: AtomicU64,
    pub(crate) preflight_skipped_slop: AtomicU64,
    pub(crate) preflight_skipped_too_short: AtomicU64,
    pub(crate) preflight_attempts: AtomicU64,
    pub(crate) preflight_candidate: AtomicU64,
    pub(crate) preflight_no_candidate: AtomicU64,
}

/// Point-in-time counters for phrase query execution.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhraseQueryStatsSnapshot {
    /// Number of segment-level phrase scorer construction attempts.
    pub scorer_attempts: u64,
    /// Number of scorer attempts that stopped because at least one phrase term was missing.
    pub term_info_missing: u64,
    /// Number of preflight checks skipped because the phrase uses slop.
    pub preflight_skipped_slop: u64,
    /// Number of preflight checks skipped because the phrase is shorter than the configured threshold.
    pub preflight_skipped_too_short: u64,
    /// Number of exact-pair preflight checks attempted.
    pub preflight_attempts: u64,
    /// Number of preflight checks that found a possible phrase candidate.
    pub preflight_candidate: u64,
    /// Number of preflight checks that found no possible phrase candidate.
    pub preflight_no_candidate: u64,
}

impl PhraseQueryStats {
    pub(crate) fn snapshot(&self) -> PhraseQueryStatsSnapshot {
        PhraseQueryStatsSnapshot {
            scorer_attempts: self.scorer_attempts.load(ATOMIC_ORDERING),
            term_info_missing: self.term_info_missing.load(ATOMIC_ORDERING),
            preflight_skipped_slop: self.preflight_skipped_slop.load(ATOMIC_ORDERING),
            preflight_skipped_too_short: self.preflight_skipped_too_short.load(ATOMIC_ORDERING),
            preflight_attempts: self.preflight_attempts.load(ATOMIC_ORDERING),
            preflight_candidate: self.preflight_candidate.load(ATOMIC_ORDERING),
            preflight_no_candidate: self.preflight_no_candidate.load(ATOMIC_ORDERING),
        }
    }
}

impl PhraseQueryStatsSnapshot {
    /// Returns the non-negative difference between two snapshots.
    pub fn saturating_sub(self, other: PhraseQueryStatsSnapshot) -> PhraseQueryStatsSnapshot {
        PhraseQueryStatsSnapshot {
            scorer_attempts: self.scorer_attempts.saturating_sub(other.scorer_attempts),
            term_info_missing: self
                .term_info_missing
                .saturating_sub(other.term_info_missing),
            preflight_skipped_slop: self
                .preflight_skipped_slop
                .saturating_sub(other.preflight_skipped_slop),
            preflight_skipped_too_short: self
                .preflight_skipped_too_short
                .saturating_sub(other.preflight_skipped_too_short),
            preflight_attempts: self
                .preflight_attempts
                .saturating_sub(other.preflight_attempts),
            preflight_candidate: self
                .preflight_candidate
                .saturating_sub(other.preflight_candidate),
            preflight_no_candidate: self
                .preflight_no_candidate
                .saturating_sub(other.preflight_no_candidate),
        }
    }
}

impl PhraseQuery {
    /// Creates a new `PhraseQuery` given a list of terms.
    ///
    /// There must be at least two terms, and all terms
    /// must belong to the same field.
    /// Offset for each term will be same as index in the Vector
    pub fn new(terms: Vec<Term>) -> PhraseQuery {
        let terms_with_offset = terms.into_iter().enumerate().collect();
        PhraseQuery::new_with_offset(terms_with_offset)
    }

    /// Creates a new `PhraseQuery` given a list of terms and their offsets.
    ///
    /// Can be used to provide custom offset for each term.
    pub fn new_with_offset(terms: Vec<(usize, Term)>) -> PhraseQuery {
        PhraseQuery::new_with_offset_and_slop(terms, 0)
    }

    /// Creates a new `PhraseQuery` given a list of terms, their offsets and a slop
    pub fn new_with_offset_and_slop(mut terms: Vec<(usize, Term)>, slop: u32) -> PhraseQuery {
        assert!(
            terms.len() > 1,
            "A phrase query is required to have strictly more than one term."
        );
        terms.sort_by_key(|&(offset, _)| offset);
        let field = terms[0].1.field();
        assert!(
            terms[1..].iter().all(|term| term.1.field() == field),
            "All terms from a phrase query must belong to the same field"
        );
        PhraseQuery {
            field,
            phrase_terms: terms,
            slop,
            preflight_min_terms: DEFAULT_PREFLIGHT_MIN_TERMS,
            stats: Arc::new(PhraseQueryStats::default()),
        }
    }

    /// Slop allowed for the phrase.
    ///
    /// The query will match if its terms are separated by `slop` terms at most.
    /// The slop can be considered a budget between all terms.
    /// E.g. "A B C" with slop 1 allows "A X B C", "A B X C", but not "A X B X C".
    ///
    /// Transposition costs 2, e.g. "A B" with slop 1 will not match "B A" but it would with slop 2
    /// Transposition is not a special case, in the example above A is moved 1 position and B is
    /// moved 1 position, so the slop is 2.
    ///
    /// As a result slop works in both directions, so the order of the terms may changed as long as
    /// they respect the slop.
    ///
    /// By default the slop is 0 meaning query terms need to be adjacent.
    pub fn set_slop(&mut self, value: u32) {
        self.slop = value;
    }

    /// Sets the minimum phrase length for the exact-pair preflight check.
    ///
    /// The default is 8 terms. Values less than 2 are clamped to 2.
    pub fn set_preflight_min_terms(&mut self, value: usize) {
        self.preflight_min_terms = value.max(2);
    }

    /// Slop allowed for the phrase.
    pub fn slop(&self) -> u32 {
        self.slop
    }

    /// Minimum phrase length required before exact-pair preflight is attempted.
    pub fn preflight_min_terms(&self) -> usize {
        self.preflight_min_terms
    }

    /// Returns point-in-time execution counters for this phrase query.
    pub fn stats_snapshot(&self) -> PhraseQueryStatsSnapshot {
        self.stats.snapshot()
    }

    /// The [`Field`] this `PhraseQuery` is targeting.
    pub fn field(&self) -> Field {
        self.field
    }

    /// `Term`s in the phrase without the associated offsets.
    pub fn phrase_terms(&self) -> Vec<Term> {
        self.phrase_terms
            .iter()
            .map(|(_, term)| term.clone())
            .collect::<Vec<Term>>()
    }

    /// Returns the [`PhraseWeight`] for the given phrase query given a specific `searcher`.
    ///
    /// This function is the same as [`Query::weight()`] except it returns
    /// a specialized type [`PhraseWeight`] instead of a Boxed trait.
    pub(crate) fn phrase_weight(
        &self,
        enable_scoring: EnableScoring<'_>,
    ) -> crate::Result<PhraseWeight> {
        let schema = enable_scoring.schema();
        let field_entry = schema.get_field_entry(self.field);
        let has_positions = field_entry
            .field_type()
            .get_index_record_option()
            .map(IndexRecordOption::has_positions)
            .unwrap_or(false);
        if !has_positions {
            let field_name = field_entry.name();
            return Err(crate::TantivyError::SchemaError(format!(
                "Applied phrase query on field {field_name:?}, which does not have positions \
                 indexed"
            )));
        }
        let terms = self.phrase_terms();
        let bm25_weight_opt = match enable_scoring {
            EnableScoring::Enabled {
                statistics_provider,
                ..
            } => Some(Bm25Weight::for_terms(statistics_provider, &terms)?),
            EnableScoring::Disabled { .. } => None,
        };
        let mut weight = PhraseWeight::new(
            self.phrase_terms.clone(),
            bm25_weight_opt,
            self.stats.clone(),
        );
        if self.slop > 0 {
            weight.slop(self.slop);
        }
        weight.set_preflight_min_terms(self.preflight_min_terms);
        Ok(weight)
    }
}

impl Query for PhraseQuery {
    /// Create the weight associated with a query.
    ///
    /// See [`Weight`].
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let phrase_weight = self.phrase_weight(enable_scoring)?;
        Ok(Box::new(phrase_weight))
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        for (_, term) in &self.phrase_terms {
            visitor(term, true);
        }
    }
}
