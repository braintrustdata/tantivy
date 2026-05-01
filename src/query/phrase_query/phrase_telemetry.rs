use std::sync::atomic::{AtomicU64, Ordering};

const ORDERING: Ordering = Ordering::Relaxed;

static SCORER_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static SCORER_TERMS: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_LOOKUPS: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_FOUND: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_MISSING_LOOKUP_DEPTH: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_MISSING_FIRST_LOOKUP: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_MISSING_SECOND_LOOKUP: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_MISSING_LATER_LOOKUP: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_ADAPTIVE_FIRST_LOOKUPS: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_ADAPTIVE_FIRST_LOOKUP_MISSES: AtomicU64 = AtomicU64::new(0);
static TERM_INFO_ADAPTIVE_ORDER_CHANGES: AtomicU64 = AtomicU64::new(0);
static MISSING_TERMS: AtomicU64 = AtomicU64::new(0);
static PREFLIGHT_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PREFLIGHT_TERMS: AtomicU64 = AtomicU64::new(0);
static PREFLIGHT_NO_CANDIDATE: AtomicU64 = AtomicU64::new(0);
static PREFLIGHT_CANDIDATE_FOUND: AtomicU64 = AtomicU64::new(0);
static PAIR_PREFLIGHT_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PAIR_PREFLIGHT_NO_CANDIDATE: AtomicU64 = AtomicU64::new(0);
static PAIR_PREFLIGHT_CANDIDATE_FOUND: AtomicU64 = AtomicU64::new(0);
static PAIR_PREFLIGHT_POSITIONS_LOADS: AtomicU64 = AtomicU64::new(0);
static PAIR_PREFLIGHT_POSTINGS_BYTES: AtomicU64 = AtomicU64::new(0);
static PAIR_PREFLIGHT_POSITIONS_BYTES: AtomicU64 = AtomicU64::new(0);
static POSITIONS_LOADS: AtomicU64 = AtomicU64::new(0);
static POSITIONS_TERMS: AtomicU64 = AtomicU64::new(0);
static POSITIONS_POSTINGS_BYTES: AtomicU64 = AtomicU64::new(0);
static POSITIONS_BYTES: AtomicU64 = AtomicU64::new(0);

/// Process-wide counters for phrase query scorer initialization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhraseQueryTelemetrySnapshot {
    /// Number of phrase scorer initialization attempts.
    pub scorer_attempts: u64,
    /// Total number of terms across phrase scorer initialization attempts.
    pub scorer_terms: u64,
    /// Number of term dictionary lookups performed while initializing phrase scorers.
    pub term_info_lookups: u64,
    /// Number of term dictionary lookups that found a term.
    pub term_info_found: u64,
    /// Number of repeated phrase terms served from the phrase initialization cache.
    pub term_info_cache_hits: u64,
    /// Sum of dictionary lookup depths at which phrase scorer attempts found a missing term.
    pub term_info_missing_lookup_depth: u64,
    /// Number of phrase scorer attempts that missed on the first dictionary lookup.
    pub term_info_missing_first_lookup: u64,
    /// Number of phrase scorer attempts that missed on the second dictionary lookup.
    pub term_info_missing_second_lookup: u64,
    /// Number of phrase scorer attempts that missed after the second dictionary lookup.
    pub term_info_missing_later_lookup: u64,
    /// Number of phrase scorer attempts whose first lookup used mature adaptive miss-rate stats.
    pub term_info_adaptive_first_lookups: u64,
    /// Number of adaptive first lookups that missed.
    pub term_info_adaptive_first_lookup_misses: u64,
    /// Number of phrase scorer attempts whose adaptive first lookup differed from the static order.
    pub term_info_adaptive_order_changes: u64,
    /// Number of phrase scorer attempts that found at least one missing term.
    pub missing_terms: u64,
    /// Number of long phrase preflight checks.
    pub preflight_attempts: u64,
    /// Total number of terms across long phrase preflight checks.
    pub preflight_terms: u64,
    /// Number of preflight checks that found no document containing all phrase terms.
    pub preflight_no_candidate: u64,
    /// Number of preflight checks that found at least one candidate document.
    pub preflight_candidate_found: u64,
    /// Number of two-term positional preflight checks.
    pub pair_preflight_attempts: u64,
    /// Number of two-term positional preflight checks that found no candidate phrase position.
    pub pair_preflight_no_candidate: u64,
    /// Number of two-term positional preflight checks that found a candidate phrase position.
    pub pair_preflight_candidate_found: u64,
    /// Number of positional postings loaded for two-term preflight checks.
    pub pair_preflight_positions_loads: u64,
    /// Total postings bytes loaded for two-term preflight checks.
    pub pair_preflight_postings_bytes: u64,
    /// Total positions bytes loaded for two-term preflight checks.
    pub pair_preflight_positions_bytes: u64,
    /// Number of phrase scorer attempts that loaded positional postings.
    pub positions_loads: u64,
    /// Total number of terms whose positional postings were loaded.
    pub positions_terms: u64,
    /// Total postings bytes loaded for final phrase scorers.
    pub positions_postings_bytes: u64,
    /// Total positions bytes loaded for final phrase scorers.
    pub positions_bytes: u64,
}

impl PhraseQueryTelemetrySnapshot {
    /// Returns the saturating difference between this snapshot and an earlier snapshot.
    pub fn delta_since(self, before: PhraseQueryTelemetrySnapshot) -> Self {
        PhraseQueryTelemetrySnapshot {
            scorer_attempts: self.scorer_attempts.saturating_sub(before.scorer_attempts),
            scorer_terms: self.scorer_terms.saturating_sub(before.scorer_terms),
            term_info_lookups: self
                .term_info_lookups
                .saturating_sub(before.term_info_lookups),
            term_info_found: self.term_info_found.saturating_sub(before.term_info_found),
            term_info_cache_hits: self
                .term_info_cache_hits
                .saturating_sub(before.term_info_cache_hits),
            term_info_missing_lookup_depth: self
                .term_info_missing_lookup_depth
                .saturating_sub(before.term_info_missing_lookup_depth),
            term_info_missing_first_lookup: self
                .term_info_missing_first_lookup
                .saturating_sub(before.term_info_missing_first_lookup),
            term_info_missing_second_lookup: self
                .term_info_missing_second_lookup
                .saturating_sub(before.term_info_missing_second_lookup),
            term_info_missing_later_lookup: self
                .term_info_missing_later_lookup
                .saturating_sub(before.term_info_missing_later_lookup),
            term_info_adaptive_first_lookups: self
                .term_info_adaptive_first_lookups
                .saturating_sub(before.term_info_adaptive_first_lookups),
            term_info_adaptive_first_lookup_misses: self
                .term_info_adaptive_first_lookup_misses
                .saturating_sub(before.term_info_adaptive_first_lookup_misses),
            term_info_adaptive_order_changes: self
                .term_info_adaptive_order_changes
                .saturating_sub(before.term_info_adaptive_order_changes),
            missing_terms: self.missing_terms.saturating_sub(before.missing_terms),
            preflight_attempts: self
                .preflight_attempts
                .saturating_sub(before.preflight_attempts),
            preflight_terms: self.preflight_terms.saturating_sub(before.preflight_terms),
            preflight_no_candidate: self
                .preflight_no_candidate
                .saturating_sub(before.preflight_no_candidate),
            preflight_candidate_found: self
                .preflight_candidate_found
                .saturating_sub(before.preflight_candidate_found),
            pair_preflight_attempts: self
                .pair_preflight_attempts
                .saturating_sub(before.pair_preflight_attempts),
            pair_preflight_no_candidate: self
                .pair_preflight_no_candidate
                .saturating_sub(before.pair_preflight_no_candidate),
            pair_preflight_candidate_found: self
                .pair_preflight_candidate_found
                .saturating_sub(before.pair_preflight_candidate_found),
            pair_preflight_positions_loads: self
                .pair_preflight_positions_loads
                .saturating_sub(before.pair_preflight_positions_loads),
            pair_preflight_postings_bytes: self
                .pair_preflight_postings_bytes
                .saturating_sub(before.pair_preflight_postings_bytes),
            pair_preflight_positions_bytes: self
                .pair_preflight_positions_bytes
                .saturating_sub(before.pair_preflight_positions_bytes),
            positions_loads: self.positions_loads.saturating_sub(before.positions_loads),
            positions_terms: self.positions_terms.saturating_sub(before.positions_terms),
            positions_postings_bytes: self
                .positions_postings_bytes
                .saturating_sub(before.positions_postings_bytes),
            positions_bytes: self.positions_bytes.saturating_sub(before.positions_bytes),
        }
    }

    /// Returns true if all counters are zero.
    pub fn is_empty(&self) -> bool {
        *self == PhraseQueryTelemetrySnapshot::default()
    }
}

/// Returns a process-wide phrase query telemetry snapshot.
pub fn phrase_query_telemetry_snapshot() -> PhraseQueryTelemetrySnapshot {
    PhraseQueryTelemetrySnapshot {
        scorer_attempts: SCORER_ATTEMPTS.load(ORDERING),
        scorer_terms: SCORER_TERMS.load(ORDERING),
        term_info_lookups: TERM_INFO_LOOKUPS.load(ORDERING),
        term_info_found: TERM_INFO_FOUND.load(ORDERING),
        term_info_cache_hits: TERM_INFO_CACHE_HITS.load(ORDERING),
        term_info_missing_lookup_depth: TERM_INFO_MISSING_LOOKUP_DEPTH.load(ORDERING),
        term_info_missing_first_lookup: TERM_INFO_MISSING_FIRST_LOOKUP.load(ORDERING),
        term_info_missing_second_lookup: TERM_INFO_MISSING_SECOND_LOOKUP.load(ORDERING),
        term_info_missing_later_lookup: TERM_INFO_MISSING_LATER_LOOKUP.load(ORDERING),
        term_info_adaptive_first_lookups: TERM_INFO_ADAPTIVE_FIRST_LOOKUPS.load(ORDERING),
        term_info_adaptive_first_lookup_misses: TERM_INFO_ADAPTIVE_FIRST_LOOKUP_MISSES
            .load(ORDERING),
        term_info_adaptive_order_changes: TERM_INFO_ADAPTIVE_ORDER_CHANGES.load(ORDERING),
        missing_terms: MISSING_TERMS.load(ORDERING),
        preflight_attempts: PREFLIGHT_ATTEMPTS.load(ORDERING),
        preflight_terms: PREFLIGHT_TERMS.load(ORDERING),
        preflight_no_candidate: PREFLIGHT_NO_CANDIDATE.load(ORDERING),
        preflight_candidate_found: PREFLIGHT_CANDIDATE_FOUND.load(ORDERING),
        pair_preflight_attempts: PAIR_PREFLIGHT_ATTEMPTS.load(ORDERING),
        pair_preflight_no_candidate: PAIR_PREFLIGHT_NO_CANDIDATE.load(ORDERING),
        pair_preflight_candidate_found: PAIR_PREFLIGHT_CANDIDATE_FOUND.load(ORDERING),
        pair_preflight_positions_loads: PAIR_PREFLIGHT_POSITIONS_LOADS.load(ORDERING),
        pair_preflight_postings_bytes: PAIR_PREFLIGHT_POSTINGS_BYTES.load(ORDERING),
        pair_preflight_positions_bytes: PAIR_PREFLIGHT_POSITIONS_BYTES.load(ORDERING),
        positions_loads: POSITIONS_LOADS.load(ORDERING),
        positions_terms: POSITIONS_TERMS.load(ORDERING),
        positions_postings_bytes: POSITIONS_POSTINGS_BYTES.load(ORDERING),
        positions_bytes: POSITIONS_BYTES.load(ORDERING),
    }
}

pub(crate) fn record_scorer_attempt(term_count: usize) {
    SCORER_ATTEMPTS.fetch_add(1, ORDERING);
    SCORER_TERMS.fetch_add(term_count as u64, ORDERING);
}

pub(crate) fn record_term_info_lookup(found: bool) {
    TERM_INFO_LOOKUPS.fetch_add(1, ORDERING);
    if found {
        TERM_INFO_FOUND.fetch_add(1, ORDERING);
    }
}

pub(crate) fn record_term_info_cache_hit() {
    TERM_INFO_CACHE_HITS.fetch_add(1, ORDERING);
}

pub(crate) fn record_adaptive_first_lookup(order_changed: bool) {
    TERM_INFO_ADAPTIVE_FIRST_LOOKUPS.fetch_add(1, ORDERING);
    if order_changed {
        TERM_INFO_ADAPTIVE_ORDER_CHANGES.fetch_add(1, ORDERING);
    }
}

pub(crate) fn record_adaptive_first_lookup_miss() {
    TERM_INFO_ADAPTIVE_FIRST_LOOKUP_MISSES.fetch_add(1, ORDERING);
}

pub(crate) fn record_missing_term(lookup_depth: usize) {
    MISSING_TERMS.fetch_add(1, ORDERING);
    TERM_INFO_MISSING_LOOKUP_DEPTH.fetch_add(lookup_depth as u64, ORDERING);
    match lookup_depth {
        0 | 1 => {
            TERM_INFO_MISSING_FIRST_LOOKUP.fetch_add(1, ORDERING);
        }
        2 => {
            TERM_INFO_MISSING_SECOND_LOOKUP.fetch_add(1, ORDERING);
        }
        _ => {
            TERM_INFO_MISSING_LATER_LOOKUP.fetch_add(1, ORDERING);
        }
    }
}

pub(crate) fn record_pair_preflight_attempt() {
    PAIR_PREFLIGHT_ATTEMPTS.fetch_add(1, ORDERING);
}

pub(crate) fn record_pair_preflight_no_candidate() {
    PAIR_PREFLIGHT_NO_CANDIDATE.fetch_add(1, ORDERING);
}

pub(crate) fn record_pair_preflight_candidate_found() {
    PAIR_PREFLIGHT_CANDIDATE_FOUND.fetch_add(1, ORDERING);
}

pub(crate) fn record_pair_preflight_positions_load(postings_bytes: usize, positions_bytes: usize) {
    PAIR_PREFLIGHT_POSITIONS_LOADS.fetch_add(1, ORDERING);
    PAIR_PREFLIGHT_POSTINGS_BYTES.fetch_add(postings_bytes as u64, ORDERING);
    PAIR_PREFLIGHT_POSITIONS_BYTES.fetch_add(positions_bytes as u64, ORDERING);
}

pub(crate) fn record_positions_load(
    term_count: usize,
    postings_bytes: usize,
    positions_bytes: usize,
) {
    POSITIONS_LOADS.fetch_add(1, ORDERING);
    POSITIONS_TERMS.fetch_add(term_count as u64, ORDERING);
    POSITIONS_POSTINGS_BYTES.fetch_add(postings_bytes as u64, ORDERING);
    POSITIONS_BYTES.fetch_add(positions_bytes as u64, ORDERING);
}
