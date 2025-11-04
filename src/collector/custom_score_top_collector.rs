use crate::collector::top_collector::{TopCollector, TopSegmentCollector};
use crate::collector::{Collector, SegmentCollector};
use crate::{DocAddress, DocId, Score, SegmentReader};

#[cfg(feature = "quickwit")]
use futures::future::BoxFuture;
#[cfg(feature = "quickwit")]
use crate::collector::AsyncCollector;

pub struct CustomScoreTopCollector<TCustomScorer, TScore = Score> {
    pub custom_scorer: TCustomScorer,
    pub collector: TopCollector<TScore>,
}

impl<TCustomScorer, TScore> CustomScoreTopCollector<TCustomScorer, TScore>
where TScore: Clone + PartialOrd
{
    pub(crate) fn new(
        custom_scorer: TCustomScorer,
        collector: TopCollector<TScore>,
    ) -> CustomScoreTopCollector<TCustomScorer, TScore> {
        CustomScoreTopCollector {
            custom_scorer,
            collector,
        }
    }
}

/// A custom segment scorer makes it possible to define any kind of score
/// for a given document belonging to a specific segment.
///
/// It is the segment local version of the [`CustomScorer`].
pub trait CustomSegmentScorer<TScore>: 'static {
    /// Computes the score of a specific `doc`.
    fn score(&mut self, doc: DocId) -> TScore;
}

/// `CustomScorer` makes it possible to define any kind of score.
///
/// The `CustomerScorer` itself does not make much of the computation itself.
/// Instead, it helps constructing `Self::Child` instances that will compute
/// the score at a segment scale.
pub trait CustomScorer<TScore>: Sync {
    /// Type of the associated [`CustomSegmentScorer`].
    type Child: CustomSegmentScorer<TScore> + Send;
    /// Builds a child scorer for a specific segment. The child scorer is associated with
    /// a specific segment.
    fn segment_scorer(&self, segment_reader: &SegmentReader) -> crate::Result<Self::Child>;
}

#[cfg(feature = "quickwit")]
pub trait AsyncCustomScorer<TScore>: Sync {
    /// Type of the associated [`CustomSegmentScorer`].
    type Child: CustomSegmentScorer<TScore> + Send;
    /// Builds a child scorer for a specific segment asynchronously.
    fn segment_scorer_async<'a>(
        &'a self,
        segment_reader: &'a SegmentReader,
    ) -> BoxFuture<'a, crate::Result<Self::Child>>;
}

impl<TCustomScorer, TScore> Collector for CustomScoreTopCollector<TCustomScorer, TScore>
where
    TCustomScorer: CustomScorer<TScore> + Send + Sync,
    TScore: 'static + PartialOrd + Clone + Send + Sync,
{
    type Fruit = Vec<(TScore, DocAddress)>;

    type Child = CustomScoreTopSegmentCollector<TCustomScorer::Child, TScore>;

    fn for_segment(
        &self,
        segment_local_id: u32,
        segment_reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        let segment_collector = self.collector.for_segment(segment_local_id, segment_reader);
        let segment_scorer = self.custom_scorer.segment_scorer(segment_reader)?;
        Ok(CustomScoreTopSegmentCollector {
            segment_collector,
            segment_scorer,
        })
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(&self, segment_fruits: Vec<Self::Fruit>) -> crate::Result<Self::Fruit> {
        self.collector.merge_fruits(segment_fruits)
    }
}

#[cfg(feature = "quickwit")]
impl<TCustomScorer, TScore> AsyncCollector for CustomScoreTopCollector<TCustomScorer, TScore>
where
    TCustomScorer: AsyncCustomScorer<TScore> + Send + Sync,
    TScore: 'static + PartialOrd + Clone + Send + Sync,
{
    type Fruit = Vec<(TScore, DocAddress)>;
    type Child = CustomScoreTopSegmentCollector<TCustomScorer::Child, TScore>;

    fn requires_scoring(&self) -> bool {
        false
    }

    fn for_segment_async<'a>(
        &'a self,
        segment_local_id: u32,
        segment_reader: &'a SegmentReader,
    ) -> BoxFuture<'a, crate::Result<Self::Child>> {
        use tracing::Instrument;
        Box::pin(async move {
            let segment_collector = self.collector.for_segment(segment_local_id, segment_reader);
            let segment_scorer = self.custom_scorer.segment_scorer_async(segment_reader)
                .instrument(tracing::info_span!("custom_scorer_segment_scorer_async"))
                .await?;
            Ok(CustomScoreTopSegmentCollector {
                segment_collector,
                segment_scorer,
            })
        }.instrument(tracing::info_span!("for_segment_async", segment_ord=%segment_local_id)))
    }

    fn merge_fruits_async(
        &self,
        segment_fruits: Vec<Self::Fruit>,
    ) -> BoxFuture<'_, crate::Result<Self::Fruit>> {
        Box::pin(async move { self.collector.merge_fruits(segment_fruits) })
    }
}

pub struct CustomScoreTopSegmentCollector<T, TScore>
where
    TScore: 'static + PartialOrd + Clone + Send + Sync + Sized,
    T: CustomSegmentScorer<TScore>,
{
    segment_collector: TopSegmentCollector<TScore>,
    segment_scorer: T,
}

impl<T, TScore> SegmentCollector for CustomScoreTopSegmentCollector<T, TScore>
where
    TScore: 'static + PartialOrd + Clone + Send + Sync,
    T: 'static + CustomSegmentScorer<TScore> + Send,
{
    type Fruit = Vec<(TScore, DocAddress)>;

    fn collect(&mut self, doc: DocId, _score: Score) {
        let score = self.segment_scorer.score(doc);
        self.segment_collector.collect(doc, score);
    }

    fn harvest(self) -> Vec<(TScore, DocAddress)> {
        self.segment_collector.harvest()
    }
}

impl<F, TScore, T> CustomScorer<TScore> for F
where
    F: 'static + Send + Sync + Fn(&SegmentReader) -> T,
    T: CustomSegmentScorer<TScore> + Send,
{
    type Child = T;

    fn segment_scorer(&self, segment_reader: &SegmentReader) -> crate::Result<Self::Child> {
        Ok((self)(segment_reader))
    }
}

impl<F, TScore> CustomSegmentScorer<TScore> for F
where F: 'static + FnMut(DocId) -> TScore
{
    fn score(&mut self, doc: DocId) -> TScore {
        (self)(doc)
    }
}
