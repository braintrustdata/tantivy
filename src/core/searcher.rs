use std::collections::BTreeMap;
use std::sync::Arc;
use std::{fmt, io};
use tracing::instrument;

use crate::collector::Collector;
#[cfg(feature = "quickwit")]
use crate::collector::{AsyncCollector, SegmentCollector, SyncCollectorAdapter};
use crate::core::Executor;
#[cfg(feature = "quickwit")]
use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::index::SegmentReader;
use crate::query::{Bm25StatisticsProvider, EnableScoring, Query, Weight};
use crate::schema::document::DocumentDeserialize;
use crate::schema::{Schema, Term};
use crate::space_usage::SearcherSpaceUsage;
use crate::store::{CacheStats, StoreReader};
#[cfg(feature = "quickwit")]
use crate::TERMINATED;
use crate::{DocAddress, Index, Opstamp, SegmentId, TrackedObject};
use futures::future::try_join_all;

/// Identifies the searcher generation accessed by a [`Searcher`].
///
/// While this might seem redundant, a [`SearcherGeneration`] contains
/// both a `generation_id` AND a list of `(SegmentId, DeleteOpstamp)`.
///
/// This is on purpose. This object is used by the [`Warmer`](crate::reader::Warmer) API.
/// Having both information makes it possible to identify which
/// artifact should be refreshed or garbage collected.
///
/// Depending on the use case, `Warmer`'s implementers can decide to
/// produce artifacts per:
/// - `generation_id` (e.g. some searcher level aggregates)
/// - `(segment_id, delete_opstamp)` (e.g. segment level aggregates)
/// - `segment_id` (e.g. for immutable document level information)
/// - `(generation_id, segment_id)` (e.g. for consistent dynamic column)
/// - ...
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SearcherGeneration {
    segments: BTreeMap<SegmentId, Option<Opstamp>>,
    generation_id: u64,
}

impl SearcherGeneration {
    pub(crate) fn from_segment_readers(
        segment_readers: &[SegmentReader],
        generation_id: u64,
    ) -> Self {
        let mut segment_id_to_del_opstamp = BTreeMap::new();
        for segment_reader in segment_readers {
            segment_id_to_del_opstamp
                .insert(segment_reader.segment_id(), segment_reader.delete_opstamp());
        }
        Self {
            segments: segment_id_to_del_opstamp,
            generation_id,
        }
    }

    /// Returns the searcher generation id.
    pub fn generation_id(&self) -> u64 {
        self.generation_id
    }

    /// Return a `(SegmentId -> DeleteOpstamp)` mapping.
    pub fn segments(&self) -> &BTreeMap<SegmentId, Option<Opstamp>> {
        &self.segments
    }
}

/// Holds a list of `SegmentReader`s ready for search.
///
/// It guarantees that the `Segment` will not be removed before
/// the destruction of the `Searcher`.
#[derive(Clone)]
pub struct Searcher {
    inner: Arc<SearcherInner>,
}

impl Searcher {
    /// Returns the `Index` associated with the `Searcher`
    pub fn index(&self) -> &Index {
        &self.inner.index
    }

    /// [`SearcherGeneration`] which identifies the version of the snapshot held by this `Searcher`.
    pub fn generation(&self) -> &SearcherGeneration {
        self.inner.generation.as_ref()
    }

    /// Fetches a document from tantivy's store given a [`DocAddress`].
    ///
    /// The searcher uses the segment ordinal to route the
    /// request to the right `Segment`.
    pub fn doc<D: DocumentDeserialize>(&self, doc_address: DocAddress) -> crate::Result<D> {
        let store_reader = &self.inner.store_readers[doc_address.segment_ord as usize];
        store_reader.get(doc_address.doc_id)
    }

    /// The cache stats for the underlying store reader.
    ///
    /// Aggregates the sum for each segment store reader.
    pub fn doc_store_cache_stats(&self) -> CacheStats {
        let cache_stats: CacheStats = self
            .inner
            .store_readers
            .iter()
            .map(|reader| reader.cache_stats())
            .sum();
        cache_stats
    }

    /// Fetches a document in an asynchronous manner.
    #[cfg(feature = "quickwit")]
    pub async fn doc_async<D: DocumentDeserialize>(
        &self,
        doc_address: DocAddress,
    ) -> crate::Result<D> {
        let store_reader = &self.inner.store_readers[doc_address.segment_ord as usize];
        store_reader.get_async(doc_address.doc_id).await
    }

    /// Access the schema associated with the index of this searcher.
    pub fn schema(&self) -> &Schema {
        &self.inner.schema
    }

    /// Returns the overall number of documents in the index.
    pub fn num_docs(&self) -> u64 {
        self.inner
            .segment_readers
            .iter()
            .map(|segment_reader| u64::from(segment_reader.num_docs()))
            .sum::<u64>()
    }

    /// Return the overall number of documents containing
    /// the given term.
    pub fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
        let mut total_doc_freq = 0;
        for segment_reader in &self.inner.segment_readers {
            let inverted_index = segment_reader.inverted_index(term.field())?;
            let doc_freq = inverted_index.doc_freq(term)?;
            total_doc_freq += u64::from(doc_freq);
        }
        Ok(total_doc_freq)
    }

    /// Return the overall number of documents containing
    /// the given term in an asynchronous manner.
    #[cfg(feature = "quickwit")]
    pub async fn doc_freq_async(&self, term: &Term) -> crate::Result<u64> {
        let mut total_doc_freq = 0;
        for segment_reader in &self.inner.segment_readers {
            let inverted_index = segment_reader.inverted_index_async(term.field()).await?;
            let doc_freq = inverted_index.doc_freq_async(term).await?;
            total_doc_freq += u64::from(doc_freq);
        }
        Ok(total_doc_freq)
    }

    /// Minimal asynchronous search entry point.
    #[cfg(feature = "quickwit")]
    pub async fn search_async<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
    ) -> crate::Result<C::Fruit> {
        self.search_async_internal(query, collector, None).await
    }

    /// Async search with a custom [Bm25StatisticsProvider].
    #[cfg(feature = "quickwit")]
    #[instrument(skip(self, query, collector, statistics_provider))]
    pub async fn search_with_statistics_provider_async<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
        statistics_provider: &dyn Bm25StatisticsProvider,
    ) -> crate::Result<C::Fruit> {
        self.search_async_internal(query, collector, Some(statistics_provider))
            .await
    }

    #[cfg(feature = "quickwit")]
    #[instrument(skip(self, query, collector, statistics_provider))]
    pub async fn search_with_async_collector_and_statistics<C>(
        &self,
        query: &dyn Query,
        collector: &C,
        statistics_provider: &dyn Bm25StatisticsProvider,
    ) -> crate::Result<<C as AsyncCollector>::Fruit>
    where
        C: Collector + AsyncCollector,
        <C as Collector>::Fruit: Send,
        <C as AsyncCollector>::Fruit: Send,
        <C as Collector>::Child: Send + 'static,
        <C as AsyncCollector>::Child: Send + 'static,
    {
        let requires_scoring = AsyncCollector::requires_scoring(collector);
        let enabled_scoring = if requires_scoring {
            EnableScoring::enabled_from_statistics_provider(statistics_provider, self)
        } else {
            EnableScoring::disabled_from_searcher(self)
        };

        let weight = query.weight(enabled_scoring)?;
        if weight.as_async_weight().is_none() {
            return Err(crate::TantivyError::InternalError(format!(
                "Query does not support async search: {:?}",
                query
            )));
        }

        self.search_async_with_async_collector(collector, weight, requires_scoring)
            .await
    }

    #[cfg(feature = "quickwit")]
    #[instrument(skip(self, query, collector, statistics_provider))]
    async fn search_async_internal<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
        statistics_provider: Option<&dyn Bm25StatisticsProvider>,
    ) -> crate::Result<C::Fruit> {
        let requires_scoring = collector.requires_scoring();
        let enabled_scoring = if requires_scoring {
            if let Some(statistics_provider) = statistics_provider {
                EnableScoring::enabled_from_statistics_provider(statistics_provider, self)
            } else {
                EnableScoring::enabled_from_searcher(self)
            }
        } else {
            EnableScoring::disabled_from_searcher(self)
        };

        let weight = query.weight(enabled_scoring)?;
        if weight.as_async_weight().is_none() {
            eprintln!("NO ASYNC WEIGHT FOR THIS QUERY: {:?}", query);
            return if let Some(statistics_provider) = statistics_provider {
                tracing::info_span!("synchronous search with statistics").in_scope(|| {
                    self.search_with_statistics_provider(query, collector, statistics_provider)
                })
            } else {
                tracing::info_span!("synchronous search").in_scope(|| self.search(query, collector))
            };
        }

        if let Some(custom_async_collector) = collector.as_async_collector() {
            return self
                .search_async_with_async_collector(custom_async_collector, weight, requires_scoring)
                .await;
        }

        let adapter = SyncCollectorAdapter::new(collector);
        self.search_async_with_async_collector(&adapter, weight, requires_scoring)
            .await
    }

    #[cfg(feature = "quickwit")]
    #[instrument(skip(self, collector, weight))]
    async fn search_async_with_async_collector<FruitT, Child>(
        &self,
        collector: &(dyn AsyncCollector<Fruit = FruitT, Child = Child> + '_),
        weight: Box<dyn Weight>,
        requires_scoring: bool,
    ) -> crate::Result<FruitT>
    where
        FruitT: crate::collector::Fruit,
        Child: SegmentCollector + Send + 'static,
    {
        use tracing::Instrument;

        let async_weight = weight
            .as_async_weight()
            .expect("checked for async support in caller");

        // Create all segment collectors and scorers concurrently
        // We can't use tokio::spawn here because of lifetime constraints,
        // but try_join_all will poll all futures concurrently
        let segment_futures: Vec<_> = self
            .segment_readers()
            .iter()
            .cloned()
            .enumerate()
            .map(|(segment_ord, segment_reader)| {
                let segment_reader_clone = segment_reader.clone();
                async move {
                    let segment_collector = collector
                        .for_segment_async(segment_ord as u32, &segment_reader)
                        .instrument(tracing::info_span!("create_segment_collector"))
                        .await?;
                    let scorer = async_weight
                        .scorer_async(segment_reader_clone, 1.0)
                        .instrument(tracing::info_span!("scorer_async"))
                        .await?;
                    Ok::<_, crate::TantivyError>((segment_collector, scorer, segment_reader))
                }
            })
            .collect();

        let segment_tasks = futures::future::try_join_all(segment_futures)
            .instrument(tracing::info_span!("prepare_segment_tasks"))
            .await?;

        // Now spawn each segment task for parallel execution
        let mut handles = Vec::new();
        for (mut segment_collector, mut scorer, segment_reader) in segment_tasks {
            let handle = tokio::spawn(
                async move {
                    let alive_bitset_opt = segment_reader.alive_bitset().cloned();
                    if requires_scoring {
                        let scorer_mut = scorer.as_mut();
                        let mut doc = scorer_mut.doc();
                        match alive_bitset_opt {
                            Some(alive_bitset) => {
                                while doc != TERMINATED {
                                    let score = scorer_mut.score();
                                    if alive_bitset.is_alive(doc) {
                                        segment_collector.collect(doc, score);
                                    }
                                    doc = scorer_mut.advance();
                                }
                            }
                            None => {
                                while doc != TERMINATED {
                                    let score = scorer_mut.score();
                                    segment_collector.collect(doc, score);
                                    doc = scorer_mut.advance();
                                }
                            }
                        }
                    } else {
                        let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
                        let scorer_mut = scorer.as_mut();
                        match alive_bitset_opt {
                            Some(alive_bitset) => loop {
                                let num_items = scorer_mut.fill_buffer(&mut buffer);
                                for doc in buffer[..num_items].iter().copied() {
                                    if alive_bitset.is_alive(doc) {
                                        segment_collector.collect(doc, 0.0);
                                    }
                                }
                                if num_items != buffer.len() {
                                    break;
                                }
                            },
                            None => loop {
                                let num_items = scorer_mut.fill_buffer(&mut buffer);
                                segment_collector.collect_block(&buffer[..num_items]);
                                if num_items != buffer.len() {
                                    break;
                                }
                            },
                        }
                    }
                    segment_collector.harvest()
                }
                .instrument(tracing::info_span!("segment_search")),
            );
            handles.push(handle);
        }

        // Wait for all parallel tasks to complete
        let mut fruits = Vec::new();
        for handle in handles {
            let fruit = handle.await.map_err(|e| {
                crate::TantivyError::InternalError(format!("Task join error: {}", e))
            })?;
            fruits.push(fruit);
        }

        collector.merge_fruits_async(fruits).await
    }

    /// Return the list of segment readers
    pub fn segment_readers(&self) -> &[SegmentReader] {
        &self.inner.segment_readers
    }

    /// Returns the segment_reader associated with the given segment_ord
    pub fn segment_reader(&self, segment_ord: u32) -> &SegmentReader {
        &self.inner.segment_readers[segment_ord as usize]
    }

    /// Runs a query on the segment readers wrapped by the searcher.
    ///
    /// Search works as follows :
    ///
    ///  First the weight object associated with the query is created.
    ///
    ///  Then, the query loops over the segments and for each segment :
    ///  - setup the collector and informs it that the segment being processed has changed.
    ///  - creates a SegmentCollector for collecting documents associated with the segment
    ///  - creates a `Scorer` object associated for this segment
    ///  - iterate through the matched documents and push them to the segment collector.
    ///
    ///  Finally, the Collector merges each of the child collectors into itself for result usability
    ///  by the caller.
    pub fn search<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
    ) -> crate::Result<C::Fruit> {
        self.search_with_statistics_provider(query, collector, self)
    }

    /// Same as [`search(...)`](Searcher::search) but allows specifying
    /// a [Bm25StatisticsProvider].
    ///
    /// This can be used to adjust the statistics used in computing BM25
    /// scores.
    pub fn search_with_statistics_provider<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
        statistics_provider: &dyn Bm25StatisticsProvider,
    ) -> crate::Result<C::Fruit> {
        let enabled_scoring = if collector.requires_scoring() {
            EnableScoring::enabled_from_statistics_provider(statistics_provider, self)
        } else {
            EnableScoring::disabled_from_searcher(self)
        };
        let executor = self.inner.index.search_executor();
        self.search_with_executor(query, collector, executor, enabled_scoring)
    }

    /// Same as [`search(...)`](Searcher::search) but multithreaded.
    ///
    /// The current implementation is rather naive :
    /// multithreading is by splitting search into as many task
    /// as there are segments.
    ///
    /// It is powerless at making search faster if your index consists in
    /// one large segment.
    ///
    /// Also, keep in my multithreading a single query on several
    /// threads will not improve your throughput. It can actually
    /// hurt it. It will however, decrease the average response time.
    pub fn search_with_executor<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
        executor: &Executor,
        enabled_scoring: EnableScoring,
    ) -> crate::Result<C::Fruit> {
        let weight = query.weight(enabled_scoring)?;
        let segment_readers = self.segment_readers();
        let fruits = executor.map(
            |(segment_ord, segment_reader)| {
                collector.collect_segment(weight.as_ref(), segment_ord as u32, segment_reader)
            },
            segment_readers.iter().enumerate(),
        )?;
        collector.merge_fruits(fruits)
    }

    /// Summarize total space usage of this searcher.
    pub fn space_usage(&self) -> io::Result<SearcherSpaceUsage> {
        let mut space_usage = SearcherSpaceUsage::new();
        for segment_reader in self.segment_readers() {
            space_usage.add_segment(segment_reader.space_usage()?);
        }
        Ok(space_usage)
    }
}

impl From<Arc<SearcherInner>> for Searcher {
    fn from(inner: Arc<SearcherInner>) -> Self {
        Searcher { inner }
    }
}

/// Holds a list of `SegmentReader`s ready for search.
///
/// It guarantees that the `Segment` will not be removed before
/// the destruction of the `Searcher`.
pub(crate) struct SearcherInner {
    schema: Schema,
    index: Index,
    segment_readers: Vec<SegmentReader>,
    store_readers: Vec<StoreReader>,
    generation: TrackedObject<SearcherGeneration>,
}

impl SearcherInner {
    /// Creates a new `Searcher`
    pub(crate) fn new(
        schema: Schema,
        index: Index,
        segment_readers: Vec<SegmentReader>,
        generation: TrackedObject<SearcherGeneration>,
        doc_store_cache_num_blocks: usize,
    ) -> io::Result<SearcherInner> {
        assert_eq!(
            &segment_readers
                .iter()
                .map(|reader| (reader.segment_id(), reader.delete_opstamp()))
                .collect::<BTreeMap<_, _>>(),
            generation.segments(),
            "Set of segments referenced by this Searcher and its SearcherGeneration must match"
        );
        let store_readers: Vec<StoreReader> = segment_readers
            .iter()
            .map(|segment_reader| segment_reader.get_store_reader(doc_store_cache_num_blocks))
            .collect::<io::Result<Vec<_>>>()?;

        Ok(SearcherInner {
            schema,
            index,
            segment_readers,
            store_readers,
            generation,
        })
    }

    pub(crate) async fn new_async(
        schema: Schema,
        index: Index,
        segment_readers: Vec<SegmentReader>,
        generation: TrackedObject<SearcherGeneration>,
        doc_store_cache_num_blocks: usize,
    ) -> io::Result<SearcherInner> {
        assert_eq!(
            &segment_readers
                .iter()
                .map(|reader| (reader.segment_id(), reader.delete_opstamp()))
                .collect::<BTreeMap<_, _>>(),
            generation.segments(),
            "Set of segments referenced by this Searcher and its SearcherGeneration must match"
        );
        let store_reader_futures = segment_readers.iter().map(|segment_reader| {
            segment_reader.get_store_reader_async(doc_store_cache_num_blocks)
        });
        let store_readers = try_join_all(store_reader_futures).await?;

        Ok(SearcherInner {
            schema,
            index,
            segment_readers,
            store_readers,
            generation,
        })
    }
}

impl fmt::Debug for Searcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let segment_ids = self
            .segment_readers()
            .iter()
            .map(SegmentReader::segment_id)
            .collect::<Vec<_>>();
        write!(f, "Searcher({segment_ids:?})")
    }
}

#[cfg(all(test, feature = "quickwit"))]
mod tests {
    use futures::executor::block_on;

    use crate::collector::{Collector, SegmentCollector};
    use crate::query::TermQuery;
    use crate::schema::{IndexRecordOption, Schema, TEXT};
    use crate::{doc, DocAddress, Index, Term};

    #[test]
    fn test_search_async_term_query_count() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(doc!(text_field => "hello world"))?;
            writer.add_document(doc!(text_field => "goodbye world"))?;
            writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let term_query = TermQuery::new(
            Term::from_field_text(text_field, "hello"),
            IndexRecordOption::Basic,
        );
        let docs = block_on(searcher.search_async(&term_query, &DocIdsCollector))?;
        assert_eq!(docs, vec![DocAddress::new(0, 0)]);
        Ok(())
    }

    #[derive(Default)]
    struct DocIdsCollector;

    impl Collector for DocIdsCollector {
        type Fruit = Vec<DocAddress>;

        type Child = DocIdsSegmentCollector;

        fn for_segment(
            &self,
            segment_local_id: crate::SegmentOrdinal,
            _segment: &crate::SegmentReader,
        ) -> crate::Result<Self::Child> {
            Ok(DocIdsSegmentCollector {
                segment_ord: segment_local_id,
                docs: Vec::new(),
            })
        }

        fn requires_scoring(&self) -> bool {
            false
        }

        fn merge_fruits(
            &self,
            segment_fruits: Vec<<Self::Child as SegmentCollector>::Fruit>,
        ) -> crate::Result<Self::Fruit> {
            Ok(segment_fruits.into_iter().flatten().collect())
        }
    }

    struct DocIdsSegmentCollector {
        segment_ord: crate::SegmentOrdinal,
        docs: Vec<DocAddress>,
    }

    impl SegmentCollector for DocIdsSegmentCollector {
        type Fruit = Vec<DocAddress>;

        fn collect(&mut self, doc: crate::DocId, _score: crate::Score) {
            self.docs.push(DocAddress::new(self.segment_ord, doc));
        }

        fn collect_block(&mut self, docs: &[crate::DocId]) {
            self.docs.extend(
                docs.iter()
                    .copied()
                    .map(|doc| DocAddress::new(self.segment_ord, doc)),
            );
        }

        fn harvest(self) -> Self::Fruit {
            self.docs
        }
    }

    #[test]
    fn test_search_async_multiple_docs() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(doc!(text_field => "hello world"))?;
            writer.add_document(doc!(text_field => "hello rust"))?;
            writer.add_document(doc!(text_field => "goodbye world"))?;
            writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let term_query = TermQuery::new(
            Term::from_field_text(text_field, "hello"),
            IndexRecordOption::Basic,
        );
        let docs = block_on(searcher.search_async(&term_query, &DocIdsCollector))?;
        assert_eq!(docs, vec![DocAddress::new(0, 0), DocAddress::new(0, 1)]);
        Ok(())
    }
}
