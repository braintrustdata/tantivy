# Async Search Implementation Plan

## Context

- Current async support is hidden behind the `quickwit` feature (async readers, async store, async term dictionary, etc.).  
- `Searcher` still exposes only sync search APIs; weights, collectors, and postings rely on synchronous I/O.  
- Goal: introduce a fully async search pipeline (query ⟶ weight ⟶ scorer ⟶ collector) without breaking existing sync users or the `quickwit` consumers that already expect async doc retrieval helpers.

## Guiding Principles

- Keep the sync API as-is; async becomes opt-in behind `#[cfg(feature = "quickwit")]`.  
- Avoid trait duplication when possible by factoring shared logic into helper methods.  
- Reuse async I/O primitives already available under `quickwit` (term dictionary, file slices, store, composite files).  
- Maintain MSRV 1.63 by returning `BoxFuture<'_, Result<…>>` instead of using `async fn` in traits.  
- Preserve compatibility with existing collectors/weights; provide async shims even if they internally delegate to sync code at first, then incrementally replace blocking paths.

## Work Breakdown

### 1. Public API Surface

- Add `Searcher::search_async` and `Searcher::search_with_statistics_provider_async` (and possibly `search_with_executor_async`) in `src/core/searcher.rs`.  
- Implement the async versions behind `cfg(feature = "quickwit")`; reuse synchronous orchestration where possible but await per-segment futures (initially sequential; add parallelization once the plumbing exists).  
- Extend `EnableScoring` with async helpers (or accept that async weights still consume the sync statistics provider but perform async I/O internally).

### 2. Async Trait Layer

- Define `AsyncQuery`, `AsyncWeight`, `AsyncCollector`, and `AsyncSegmentCollector` traits in `src/query/query.rs`, `src/query/weight.rs`, and `src/collector/mod.rs`.  
- Use associated `type Future` or `BoxFuture<crate::Result<…>>` signatures to avoid nightly features.  
- Provide blanket implementations so existing sync types can implement async traits by delegating to their sync logic; this allows incremental migration.

### 3. Async Postings & Inverted Index

- Add async counterparts:
  - `BlockSegmentPostings::open_async`, factoring out the shared initialization.  
  - `SegmentPostings::from_block_postings_async` if needed, or reuse the existing constructor.  
  - `InvertedIndexReader::read_block_postings_async`, `read_postings_async`, `read_postings_no_deletes_async`.  
- Ensure these methods await `FileSlice::read_bytes_slice_async` and reuse existing decoding code once the data is in memory.  
- Update `TermDictionary` range/stream helpers as required (most async plumbing already exists via SSTables).

### 4. Async Weight Implementations

- Start with `TermWeight`:
  - Use the new async postings calls.  
  - Return `BoxFuture<'_, crate::Result<Box<dyn AsyncScorer>>>` (or a sync scorer wrapped in `BoxFuture::from(async move { … })` until scorers need to become async).  
- Expand to other commonly used weights (boolean, phrase, range, automaton, fast field range, etc.).  
- For combinators (boolean, union, intersection), orchestrate child futures and reuse the sync score-combiner logic once all child scorers are available.

### 5. Async Collectors

- Introduce `AsyncCollector::collect_segment_async` that mirrors the sync `collect_segment`.  
- Implement a baseline async `Count` collector plus a no-op collector suitable for testing.  
- Initially, collectors may internally call sync methods after awaiting scorer construction; add true async loops once scorers expose async iteration helpers.

### 6. Scorers & Iteration Helpers

- Evaluate whether scorers need async iteration (likely no—posting iteration stays CPU-bound once data is in memory).  
- Expose async wrappers around `for_each_scorer` / `for_each_docset_buffered` that return `BoxFuture` but call the synchronous helpers internally.

### 7. Testing & QA

- Add feature-gated async tests under `#[cfg(feature = "quickwit")]`:
  - Smoke test for `search_async` with scoring disabled and enabled.  
  - Tests for async doc frequency, async term weights, and async collector integration.  
- Run tests in CI with the `quickwit` feature enabled.  
- Add doc examples (e.g., async `tokio` integration) in `doc/` and update `CHANGELOG.md`.

### 8. Follow-up & Performance

- Profile async search to ensure we don’t regress sync performance (no hidden blocking).  
- Expand async collector coverage (`TopDocs`, facet collectors, aggregations).  
- Evaluate adding per-segment parallelism via `FuturesUnordered` or an async-aware executor abstraction.

## Estimated Effort

| Scope | Estimate |
| --- | --- |
| API & trait scaffolding | 1.5–2 days |
| Async postings plumbing | ~1 day |
| Initial weight migration (Term + common weights) | 2–3 days |
| Async collectors + smoke tests | ~1 day |
| Follow-up (docs, more collectors, perf) | As needed |

## Risks & Mitigations

- **Trait complexity**: returning boxed futures can get noisy. Mitigate by centralizing helper types (`type WeightFuture<'a> = BoxFuture<'a, crate::Result<Box<dyn AsyncScorer>>>`).  
- **Sync/async divergence**: ensure helper functions encapsulate shared logic so fixes apply to both paths.  
- **Partial migration**: maintain clear interim state (e.g., panic if async search hits a weight that hasn’t been ported) to avoid silent blocking in async contexts.  
- **Testing coverage**: rely on feature-gated tests and add dedicated async integration tests to avoid regressions.

