use std::sync::atomic::{AtomicU64, Ordering};

const ORDERING: Ordering = Ordering::Relaxed;

static SCORER_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static SCORER_TERMS: AtomicU64 = AtomicU64::new(0);
static MISSING_TERMS: AtomicU64 = AtomicU64::new(0);
static PREFLIGHT_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PREFLIGHT_TERMS: AtomicU64 = AtomicU64::new(0);
static PREFLIGHT_NO_CANDIDATE: AtomicU64 = AtomicU64::new(0);
static PREFLIGHT_CANDIDATE_FOUND: AtomicU64 = AtomicU64::new(0);
static POSITIONS_LOADS: AtomicU64 = AtomicU64::new(0);
static POSITIONS_TERMS: AtomicU64 = AtomicU64::new(0);

/// Process-wide counters for phrase query scorer initialization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhraseQueryTelemetrySnapshot {
    /// Number of phrase scorer initialization attempts.
    pub scorer_attempts: u64,
    /// Total number of terms across phrase scorer initialization attempts.
    pub scorer_terms: u64,
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
    /// Number of phrase scorer attempts that loaded positional postings.
    pub positions_loads: u64,
    /// Total number of terms whose positional postings were loaded.
    pub positions_terms: u64,
}

impl PhraseQueryTelemetrySnapshot {
    /// Returns the saturating difference between this snapshot and an earlier snapshot.
    pub fn delta_since(self, before: PhraseQueryTelemetrySnapshot) -> Self {
        PhraseQueryTelemetrySnapshot {
            scorer_attempts: self.scorer_attempts.saturating_sub(before.scorer_attempts),
            scorer_terms: self.scorer_terms.saturating_sub(before.scorer_terms),
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
            positions_loads: self.positions_loads.saturating_sub(before.positions_loads),
            positions_terms: self.positions_terms.saturating_sub(before.positions_terms),
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
        missing_terms: MISSING_TERMS.load(ORDERING),
        preflight_attempts: PREFLIGHT_ATTEMPTS.load(ORDERING),
        preflight_terms: PREFLIGHT_TERMS.load(ORDERING),
        preflight_no_candidate: PREFLIGHT_NO_CANDIDATE.load(ORDERING),
        preflight_candidate_found: PREFLIGHT_CANDIDATE_FOUND.load(ORDERING),
        positions_loads: POSITIONS_LOADS.load(ORDERING),
        positions_terms: POSITIONS_TERMS.load(ORDERING),
    }
}

pub(crate) fn record_scorer_attempt(term_count: usize) {
    SCORER_ATTEMPTS.fetch_add(1, ORDERING);
    SCORER_TERMS.fetch_add(term_count as u64, ORDERING);
}

pub(crate) fn record_missing_term() {
    MISSING_TERMS.fetch_add(1, ORDERING);
}

pub(crate) fn record_preflight_attempt(term_count: usize) {
    PREFLIGHT_ATTEMPTS.fetch_add(1, ORDERING);
    PREFLIGHT_TERMS.fetch_add(term_count as u64, ORDERING);
}

pub(crate) fn record_preflight_no_candidate() {
    PREFLIGHT_NO_CANDIDATE.fetch_add(1, ORDERING);
}

pub(crate) fn record_preflight_candidate_found() {
    PREFLIGHT_CANDIDATE_FOUND.fetch_add(1, ORDERING);
}

pub(crate) fn record_positions_load(term_count: usize) {
    POSITIONS_LOADS.fetch_add(1, ORDERING);
    POSITIONS_TERMS.fetch_add(term_count as u64, ORDERING);
}
