use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN, TERMINATED};
use crate::index::SegmentReader;
use crate::query::boost_query::BoostScorer;
use crate::query::explanation::does_not_match;
use crate::query::{EnableScoring, Explanation, Query, Scorer, Weight};
use crate::{DocId, Score};

/// Query that matches all of the documents.
///
/// All of the document get the score 1.0.
#[derive(Clone, Debug)]
pub struct AllQuery;

impl Query for AllQuery {
    fn weight(&self, _: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(AllWeight))
    }
}

/// Weight associated with the `AllQuery` query.
pub struct AllWeight;

impl Weight for AllWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let all_scorer = AllScorer {
            doc: 0u32,
            max_doc: reader.max_doc(),
        };
        Ok(Box::new(BoostScorer::new(all_scorer, boost)))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        if doc >= reader.max_doc() {
            return Err(does_not_match(doc));
        }
        Ok(Explanation::new("AllQuery", 1.0))
    }
}

/// Scorer associated with the `AllQuery` query.
pub struct AllScorer {
    doc: DocId,
    max_doc: DocId,
}

impl DocSet for AllScorer {
    #[inline(always)]
    fn advance(&mut self) -> DocId {
        if self.doc + 1 >= self.max_doc {
            self.doc = TERMINATED;
            return TERMINATED;
        }
        self.doc += 1;
        self.doc
    }

    #[inline(always)]
    fn seek(&mut self, target: DocId) -> DocId {
        let target = self.doc.max(target);
        self.doc = if target >= self.max_doc {
            TERMINATED
        } else {
            target
        };
        self.doc
    }

    fn fill_buffer(&mut self, buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN]) -> usize {
        if self.doc() == TERMINATED {
            return 0;
        }
        let is_safe_distance = self.doc() + (buffer.len() as u32) < self.max_doc;
        if is_safe_distance {
            let num_items = buffer.len();
            for buffer_val in buffer {
                *buffer_val = self.doc();
                self.doc += 1;
            }
            num_items
        } else {
            for (i, buffer_val) in buffer.iter_mut().enumerate() {
                *buffer_val = self.doc();
                if self.advance() == TERMINATED {
                    return i + 1;
                }
            }
            buffer.len()
        }
    }

    #[inline(always)]
    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        self.max_doc
    }
}

impl Scorer for AllScorer {
    fn score(&mut self) -> Score {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::AllQuery;
    use crate::docset::{DocSet, COLLECT_BLOCK_BUFFER_LEN, TERMINATED};
    use crate::query::{
        intersect_scorers, AllScorer, ConstScorer, EnableScoring, Query, VecDocSet,
    };
    use crate::schema::{Schema, TEXT};
    use crate::{Index, IndexWriter};

    fn create_test_index() -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(field=>"aaa"))?;
        index_writer.add_document(doc!(field=>"bbb"))?;
        index_writer.commit()?;
        index_writer.add_document(doc!(field=>"ccc"))?;
        index_writer.commit()?;
        Ok(index)
    }

    #[test]
    fn test_all_query() -> crate::Result<()> {
        let index = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let weight = AllQuery.weight(EnableScoring::disabled_from_schema(&index.schema()))?;
        {
            let reader = searcher.segment_reader(0);
            let mut scorer = weight.scorer(reader, 1.0)?;
            assert_eq!(scorer.doc(), 0u32);
            assert_eq!(scorer.advance(), 1u32);
            assert_eq!(scorer.doc(), 1u32);
            assert_eq!(scorer.advance(), TERMINATED);
        }
        {
            let reader = searcher.segment_reader(1);
            let mut scorer = weight.scorer(reader, 1.0)?;
            assert_eq!(scorer.doc(), 0u32);
            assert_eq!(scorer.advance(), TERMINATED);
        }
        Ok(())
    }

    #[test]
    fn test_all_query_with_boost() -> crate::Result<()> {
        let index = create_test_index()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let weight = AllQuery.weight(EnableScoring::disabled_from_schema(searcher.schema()))?;
        let reader = searcher.segment_reader(0);
        {
            let mut scorer = weight.scorer(reader, 2.0)?;
            assert_eq!(scorer.doc(), 0u32);
            assert_eq!(scorer.score(), 2.0);
        }
        {
            let mut scorer = weight.scorer(reader, 1.5)?;
            assert_eq!(scorer.doc(), 0u32);
            assert_eq!(scorer.score(), 1.5);
        }
        Ok(())
    }

    #[test]
    fn test_seek() {
        let mut scorer = AllScorer {
            doc: 0,
            max_doc: 100,
        };
        for target in [0, 1, 42, 42, 98] {
            assert_eq!(scorer.seek(target), target);
            assert_eq!(scorer.doc(), target);
        }
        assert_eq!(scorer.advance(), 99);
        assert_eq!(scorer.seek(100), TERMINATED);
        assert_eq!(scorer.seek(0), TERMINATED);
        assert_eq!(scorer.seek(TERMINATED), TERMINATED);
    }

    #[test]
    fn test_seek_boundaries() {
        for max_doc in [0, 1, 100, TERMINATED] {
            for target in [max_doc, TERMINATED] {
                let mut scorer = AllScorer { doc: 0, max_doc };
                assert_eq!(scorer.seek(target), TERMINATED);
                assert_eq!(scorer.doc(), TERMINATED);
            }
            if max_doc > 0 {
                let mut scorer = AllScorer { doc: 0, max_doc };
                assert_eq!(scorer.seek(max_doc - 1), max_doc - 1);
                assert_eq!(scorer.seek(max_doc), TERMINATED);
            }
        }
    }

    #[test]
    fn test_seek_sparse_intersection() {
        let mut scorer = intersect_scorers(vec![
            Box::new(AllScorer {
                doc: 0,
                max_doc: TERMINATED,
            }),
            Box::new(ConstScorer::new(
                VecDocSet::from(vec![1_000_000_000, 2_000_000_000]),
                1.0,
            )),
        ]);
        assert_eq!(scorer.doc(), 1_000_000_000);
        assert_eq!(scorer.seek(1_500_000_000), 2_000_000_000);
        assert_eq!(scorer.advance(), TERMINATED);
    }

    #[test]
    pub fn test_fill_buffer() {
        let mut postings = AllScorer {
            doc: 0u32,
            max_doc: COLLECT_BLOCK_BUFFER_LEN as u32 * 2 + 9,
        };
        let mut buffer = [0u32; COLLECT_BLOCK_BUFFER_LEN];
        assert_eq!(postings.fill_buffer(&mut buffer), COLLECT_BLOCK_BUFFER_LEN);
        for i in 0u32..COLLECT_BLOCK_BUFFER_LEN as u32 {
            assert_eq!(buffer[i as usize], i);
        }
        assert_eq!(postings.fill_buffer(&mut buffer), COLLECT_BLOCK_BUFFER_LEN);
        for i in 0u32..COLLECT_BLOCK_BUFFER_LEN as u32 {
            assert_eq!(buffer[i as usize], i + COLLECT_BLOCK_BUFFER_LEN as u32);
        }
        assert_eq!(postings.fill_buffer(&mut buffer), 9);
    }
}
