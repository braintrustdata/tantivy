use std::cmp::Ordering;

use crate::query::{EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::Field;
use crate::{DocId, Score, SegmentReader, TantivyError, TERMINATED};

#[derive(Debug, Clone)]
pub struct VectorAnnQuery {
    field: Field,
    vector_id: String,
    query: Vec<f32>,
    limit: usize,
    min_similarity: Option<f32>,
}

impl VectorAnnQuery {
    pub fn new(
        field: Field,
        vector_id: impl Into<String>,
        query: Vec<f32>,
        limit: usize,
        min_similarity: Option<f32>,
    ) -> Self {
        Self {
            field,
            vector_id: vector_id.into(),
            query,
            limit,
            min_similarity,
        }
    }
}

impl Query for VectorAnnQuery {
    fn weight(&self, _enable_scoring: EnableScoring) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(VectorAnnWeight {
            field: self.field,
            vector_id: self.vector_id.clone(),
            query: self.query.clone(),
            limit: self.limit,
            min_similarity: self.min_similarity,
        }))
    }
}

#[derive(Debug, Clone)]
struct VectorAnnWeight {
    field: Field,
    vector_id: String,
    query: Vec<f32>,
    limit: usize,
    min_similarity: Option<f32>,
}

impl Weight for VectorAnnWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        if self.limit == 0 {
            return Ok(Box::new(VectorAnnScorer::empty()));
        }

        let mut docs = search_segment(
            reader,
            self.field,
            &self.vector_id,
            &self.query,
            self.limit,
            self.min_similarity,
        )?;
        docs.sort_by_key(|(doc, _)| *doc);
        Ok(Box::new(VectorAnnScorer::new(docs, boost)))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.doc() > doc {
            return Err(TantivyError::InvalidArgument(format!(
                "VectorAnnQuery: doc {doc} is out of order"
            )));
        }
        scorer.seek(doc);
        if scorer.doc() != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "VectorAnnQuery: doc {doc} not found"
            )));
        }
        Ok(Explanation::new("VectorAnnQuery", scorer.score()))
    }

    fn count(&self, reader: &SegmentReader) -> crate::Result<u32> {
        let mut scorer = self.scorer(reader, 1.0)?;
        let mut count = 0u32;
        while scorer.doc() != TERMINATED {
            count += 1;
            scorer.advance();
        }
        Ok(count)
    }
}

fn search_segment(
    reader: &SegmentReader,
    field: Field,
    vector_id: &str,
    query: &[f32],
    limit: usize,
    min_similarity: Option<f32>,
) -> crate::Result<Vec<(DocId, Score)>> {
    if let Some(vector_ann_reader) = reader.vector_ann_reader() {
        return vector_ann_reader
            .search(field, vector_id, query, limit, min_similarity)
            .map_err(Into::into);
    }

    let vector_reader = match reader.vector_reader(field) {
        Some(reader) => reader,
        None => return Ok(Vec::new()),
    };

    let mut scored: Vec<(DocId, Score)> = Vec::new();
    let query_norm = l2_norm(query);
    if query_norm == 0.0 {
        return Ok(Vec::new());
    }

    for (doc_id, vector) in vector_reader.iter_vectors(field, vector_id) {
        if let Some(sim) = cosine_similarity(query, vector.as_ref(), query_norm) {
            if min_similarity.map_or(true, |min| sim >= min) {
                scored.push((doc_id, sim));
            }
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    if scored.len() > limit {
        scored.truncate(limit);
    }
    Ok(scored)
}

fn l2_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn cosine_similarity(query: &[f32], vector: &[f32], query_norm: f32) -> Option<f32> {
    if query.len() != vector.len() || vector.is_empty() {
        return None;
    }
    let vector_norm = l2_norm(vector);
    if vector_norm == 0.0 {
        return None;
    }
    let dot = query.iter().zip(vector.iter()).map(|(a, b)| a * b).sum::<f32>();
    Some(dot / (query_norm * vector_norm))
}

#[derive(Debug)]
struct VectorAnnScorer {
    docs: Vec<(DocId, Score)>,
    cursor: usize,
    boost: Score,
}

impl VectorAnnScorer {
    fn new(docs: Vec<(DocId, Score)>, boost: Score) -> Self {
        Self {
            docs,
            cursor: 0,
            boost,
        }
    }

    fn empty() -> Self {
        Self {
            docs: Vec::new(),
            cursor: 0,
            boost: 1.0,
        }
    }

    fn current_doc(&self) -> DocId {
        self.docs
            .get(self.cursor)
            .map(|(doc, _)| *doc)
            .unwrap_or(TERMINATED)
    }
}

impl crate::DocSet for VectorAnnScorer {
    fn advance(&mut self) -> DocId {
        if self.cursor < self.docs.len() {
            self.cursor += 1;
        }
        self.current_doc()
    }

    fn doc(&self) -> DocId {
        self.current_doc()
    }

    fn size_hint(&self) -> u32 {
        self.docs.len() as u32
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current_doc() >= target {
            return self.current_doc();
        }
        let slice = &self.docs[self.cursor..];
        match slice.binary_search_by(|(doc, _)| doc.cmp(&target)) {
            Ok(idx) => {
                self.cursor += idx;
                self.current_doc()
            }
            Err(idx) => {
                self.cursor += idx;
                self.current_doc()
            }
        }
    }

}

impl Scorer for VectorAnnScorer {
    fn score(&mut self) -> Score {
        self.docs
            .get(self.cursor)
            .map(|(_, score)| score * self.boost)
            .unwrap_or(0.0)
    }
}
