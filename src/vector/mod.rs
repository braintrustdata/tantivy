//! Vector storage module for Tantivy.
//!
//! This module provides vector field support for storing named embedding vectors.
//! Each document can have multiple named vectors per field, stored in columnar format
//! optimized for batch access patterns like clustering and similarity search.
//!
//! # Design Goals
//!
//! 1. **Efficient batch retrieval** - Optimized for ML workloads that process all vectors
//!    with a given ID (e.g., clustering, computing centroids, batch similarity)
//! 2. **Sparse storage** - Only stores vectors that exist; uses presence bitsets
//! 3. **Multiple encodings** - f32 (full precision), f16 (50% smaller), int8 (75% smaller)
//! 4. **Segment lifecycle** - Vector files merge/GC with Tantivy's segment management
//!
//! # Features
//!
//! - **Named vectors**: Each document can have multiple vectors identified by string IDs
//!   (e.g., "chunk_0", "chunk_1", "summary")
//! - **Columnar storage**: All vectors with the same ID stored contiguously
//! - **Fixed dimensions per vector_id**: Required for efficient columnar layout
//! - **Optional vectors**: Documents may omit any or all vector IDs (sparse)
//! - **Presence bitset**: O(1) check for whether a doc has a given vector
//!
//! # File Format
//!
//! ```text
//! Header:
//!   magic: u32 ("TVEC")
//!   version: u8 (1)
//!   encoding: u8 (0=f32, 1=f16, 2=int8)
//!   num_fields: u32
//!   field_ids: [u32; num_fields]
//!   num_docs: u32
//!
//! Per field, per vector_id (sorted alphabetically):
//!   vector_id_len: u32
//!   vector_id: [u8; len]
//!   dimensions: u32               # fixed for this column
//!   presence_bitset: [u64; ⌈num_docs/64⌉]
//!   [if int8: scale: f32, zero_point: f32]
//!   vectors: [encoded; dims × popcount(bitset)]  # contiguous
//!
//! Footer: Tantivy's standard file footer
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | `iter_vectors(field, id)` | O(n) | Primary pattern: sequential scan, cache-friendly |
//! | `get(field, id, doc)` | O(1) | Random access via bitset + offset calculation |
//! | `has_vector(field, id, doc)` | O(1) | Single bit check |
//! | `dimensions(field, id)` | O(1) | Stored in header |
//! | `count(field, id)` | O(1) | Popcount of bitset (cached) |
//!
//! ## What's Fast
//!
//! - **Batch iteration**: `iter_vectors` reads contiguous memory, ideal for SIMD/vectorized ops
//! - **Clustering**: Extract all vectors → feed to K-means/DBSCAN (see example below)
//! - **Similarity search**: Scan all vectors, compute distances in batch
//! - **Presence checks**: Bitset operations are extremely fast
//!
//! ## What's Slower
//!
//! - **Single vector lookup**: `get()` works but involves bitset math; if you need many
//!   random accesses, consider iterating instead
//! - **Cross-segment queries**: Each segment has its own file; aggregate across segments
//!
//! # Trade-offs
//!
//! | Choice | Benefit | Cost |
//! |--------|---------|------|
//! | Fixed dims per vector_id | Contiguous storage, no per-vector length | Can't mix dimensions |
//! | Columnar by vector_id | Fast batch access for ML | Slower single-doc retrieval |
//! | Presence bitset | Sparse storage, O(1) checks | 1 bit per doc overhead |
//! | f16 encoding | 50% storage reduction | ~0.1% precision loss |
//! | int8 encoding | 75% storage reduction | ~1-2% precision loss |
//!
//! # Usage Examples
//!
//! ## Schema Definition
//!
//! ```rust
//! use tantivy::schema::SchemaBuilder;
//!
//! let mut schema_builder = SchemaBuilder::new();
//! let embedding = schema_builder.add_vector_map_field("embedding", ());
//! let schema = schema_builder.build();
//! ```
//!
//! ## Indexing Documents
//!
//! ```rust,ignore
//! let mut doc = TantivyDocument::new();
//! doc.add_named_vector(embedding, "chunk_0", vec![0.1, 0.2, 0.3]);
//! doc.add_named_vector(embedding, "summary", vec![1.0, 2.0, 3.0]);
//! index_writer.add_document(doc)?;
//! ```
//!
//! ## Columnar Access for ML
//!
//! ```rust,ignore
//! // Collect all "chunk_0" vectors for clustering
//! let mut all_vectors: Vec<Vec<f32>> = Vec::new();
//! let mut doc_ids: Vec<DocId> = Vec::new();
//!
//! for segment_reader in searcher.segment_readers() {
//!     if let Some(vr) = segment_reader.vector_reader(embedding) {
//!         for (doc_id, vec) in vr.iter_vectors(embedding, "chunk_0") {
//!             doc_ids.push(doc_id);
//!             all_vectors.push(vec.into_owned());
//!         }
//!     }
//! }
//! // Now feed `all_vectors` to your clustering algorithm
//! ```
//!
//! ## Clustering Example (with ndarray)
//!
//! ```rust,ignore
//! use ndarray::{Array2, Axis};
//!
//! // Convert vectors to ndarray matrix
//! let n_samples = all_vectors.len();
//! let n_dims = all_vectors[0].len();
//! let flat: Vec<f32> = all_vectors.into_iter().flatten().collect();
//! let matrix = Array2::from_shape_vec((n_samples, n_dims), flat).unwrap();
//!
//! // Simple K-means (conceptual - use linfa-clustering in practice)
//! fn kmeans_step(data: &Array2<f32>, k: usize) -> Vec<usize> {
//!     // 1. Initialize k random centroids
//!     // 2. Assign each point to nearest centroid
//!     // 3. Update centroids to mean of assigned points
//!     // 4. Repeat until convergence
//!     todo!()
//! }
//! ```

pub mod format;
mod reader;
mod writer;

pub use format::{VectorEncoding, Int8QuantParams, PresenceBitset, VECTOR_MAGIC, VECTOR_VERSION};
pub use reader::VectorReader;
pub use writer::VectorFieldsWriter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::RamDirectory;
    use crate::schema::document::Document;
    use crate::schema::{Field, SchemaBuilder, STORED, TEXT};
    use crate::{Index, TantivyDocument};
    use std::collections::BTreeMap;
    use std::io::Cursor;

    #[test]
    fn test_vector_fields_roundtrip() {
        // Create a schema with a vector field
        let mut schema_builder = SchemaBuilder::new();
        let vec_field = schema_builder.add_vector_map_field("embedding", ());
        let schema = schema_builder.build();

        let mut writer = VectorFieldsWriter::from_schema(&schema);

        // Add doc with named vectors
        let mut doc = TantivyDocument::new();
        doc.add_named_vector(vec_field, "chunk_0", vec![1.0, 2.0]);
        doc.add_named_vector(vec_field, "summary", vec![10.0, 20.0, 30.0]);
        writer.add_document(&doc);

        // Verify we can retrieve the vectors
        let chunk_vecs = writer.get_vectors(vec_field, "chunk_0").unwrap();
        assert_eq!(chunk_vecs.get(&0), Some(&vec![1.0, 2.0]));
    }

    #[test]
    fn test_vector_binary_format_roundtrip() {
        use crate::vector::format::{VECTOR_MAGIC, VECTOR_VERSION, VectorEncoding};
        
        // Manually write v2 columnar binary data and read it back
        let mut buffer = Vec::new();

        // V2 Header
        buffer.extend_from_slice(&VECTOR_MAGIC.to_le_bytes());
        buffer.push(VECTOR_VERSION);
        buffer.push(VectorEncoding::F32 as u8);
        
        // 1 field
        buffer.extend_from_slice(&1u32.to_le_bytes());
        // Field ID: 0
        buffer.extend_from_slice(&0u32.to_le_bytes());
        // Num docs: 2
        buffer.extend_from_slice(&2u32.to_le_bytes());

        // Field 0 has 1 vector ID: "main"
        buffer.extend_from_slice(&1u32.to_le_bytes());

        // Vector ID "main"
        let main_id = b"main";
        buffer.extend_from_slice(&(main_id.len() as u32).to_le_bytes());
        buffer.extend_from_slice(main_id);
        // Dimensions: 3 (fixed for this vector_id)
        buffer.extend_from_slice(&3u32.to_le_bytes());
        // Presence bitset: both docs have "main" (bits 0,1 set = 0b11)
        let mut bitset_bytes = [0u8; 8]; // Padded to u64
        bitset_bytes[0] = 0b00000011;
        buffer.extend_from_slice(&bitset_bytes);
        // Vectors (contiguous): [0.1, 0.2, 0.3], [1.0, 2.0, 3.0]
        buffer.extend_from_slice(&0.1f32.to_le_bytes());
        buffer.extend_from_slice(&0.2f32.to_le_bytes());
        buffer.extend_from_slice(&0.3f32.to_le_bytes());
        buffer.extend_from_slice(&1.0f32.to_le_bytes());
        buffer.extend_from_slice(&2.0f32.to_le_bytes());
        buffer.extend_from_slice(&3.0f32.to_le_bytes());

        // Read them back
        let reader = VectorReader::open(Cursor::new(&buffer)).unwrap();
        let field = Field::from_field_id(0);

        assert_eq!(reader.num_docs(), 2);
        assert_eq!(reader.encoding(), VectorEncoding::F32);
        assert_eq!(
            reader.get(field, "main", 0).map(|c| c.into_owned()),
            Some(vec![0.1f32, 0.2, 0.3])
        );
        assert_eq!(reader.get(field, "main", 1).map(|c| c.into_owned()), Some(vec![1.0f32, 2.0, 3.0]));
        assert!(reader.get(field, "main", 2).is_none()); // out of bounds
        assert!(reader.get(field, "nonexistent", 0).is_none()); // no such ID
    }

    /// End-to-end test: Create an index with named vectors, add documents, and read vectors back
    #[test]
    fn test_vector_index_integration() {
        use crate::directory::Directory;
        use crate::schema::document::{ReferenceValue, ReferenceValueLeaf, Value};
        use crate::SegmentComponent;

        // Create schema with a text field and a vector field
        let mut schema_builder = SchemaBuilder::new();
        let title_field = schema_builder.add_text_field("title", TEXT | STORED);
        let embedding_field = schema_builder.add_vector_map_field("embedding", ());
        let schema = schema_builder.build();

        // First, test that named vectors are stored correctly in TantivyDocument
        let mut doc1 = TantivyDocument::new();
        doc1.add_text(title_field, "First document");
        doc1.add_named_vector(embedding_field, "chunk_0", vec![0.1, 0.2, 0.3]);
        doc1.add_named_vector(embedding_field, "summary", vec![10.0, 20.0]);

        // Verify the vector map is in the document
        let mut found_vector = false;
        for (field, value) in doc1.iter_fields_and_values() {
            if field == embedding_field {
                if let ReferenceValue::Leaf(ReferenceValueLeaf::VectorMap(vec_map)) = value.as_value()
                {
                    assert_eq!(vec_map.get("chunk_0"), Some(&vec![0.1f32, 0.2, 0.3]));
                    assert_eq!(vec_map.get("summary"), Some(&vec![10.0f32, 20.0]));
                    found_vector = true;
                }
            }
        }
        assert!(found_vector, "Vector not found in document");

        // Create index in RAM
        let directory = RamDirectory::create();
        let index = Index::create(directory, schema.clone(), Default::default()).unwrap();

        // Add documents with named vectors
        let mut index_writer = index.writer::<TantivyDocument>(50_000_000).unwrap();

        // Document 1: with chunk_0 and summary vectors
        let mut doc1 = TantivyDocument::new();
        doc1.add_text(title_field, "First document");
        doc1.add_named_vector(embedding_field, "chunk_0", vec![0.1, 0.2, 0.3]);
        doc1.add_named_vector(embedding_field, "summary", vec![10.0, 20.0, 30.0]);
        index_writer.add_document(doc1).unwrap();

        // Document 2: only chunk_0 (same dimensions as doc1's chunk_0)
        let mut doc2 = TantivyDocument::new();
        doc2.add_text(title_field, "Second document");
        doc2.add_named_vector(embedding_field, "chunk_0", vec![1.0, 2.0, 3.0]);
        index_writer.add_document(doc2).unwrap();

        // Document 3: no vectors
        let mut doc3 = TantivyDocument::new();
        doc3.add_text(title_field, "Third document");
        index_writer.add_document(doc3).unwrap();

        // Commit and drop the index_writer to release the lock
        index_writer.commit().unwrap();
        drop(index_writer);

        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        // Verify we can read vectors back
        assert_eq!(searcher.num_docs(), 3);

        // Get segment readers and verify columnar access
        // Collect all "chunk_0" vectors across segments (primary access pattern)
        let mut chunk0_count = 0;
        let mut summary_count = 0;

        for segment_reader in searcher.segment_readers() {
            if let Some(vector_reader) = segment_reader.vector_reader(embedding_field) {
                // Count vectors using columnar iteration
                for (_doc_id, _vec) in vector_reader.iter_vectors(embedding_field, "chunk_0") {
                    chunk0_count += 1;
                }
                for (_doc_id, _vec) in vector_reader.iter_vectors(embedding_field, "summary") {
                    summary_count += 1;
                }
            }
        }

        // Doc 1 has chunk_0, Doc 2 has chunk_0 = 2 total
        assert_eq!(chunk0_count, 2, "Expected 2 chunk_0 vectors");
        // Doc 1 has summary = 1 total
        assert_eq!(summary_count, 1, "Expected 1 summary vector");

        // Now test segment merging
        let segment_ids: Vec<_> = index
            .searchable_segments()
            .unwrap()
            .iter()
            .map(|s| s.id())
            .collect();
        let mut index_writer: crate::IndexWriter =
            index.writer_with_num_threads(1, 15_000_000).unwrap();
        index_writer.merge(&segment_ids).wait().unwrap();
        index_writer.commit().unwrap();
        drop(index_writer);

        // Reload the reader after merge
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        // After merge, should have 1 segment with all 3 documents
        assert_eq!(
            searcher.segment_readers().len(),
            1,
            "Expected 1 segment after merge"
        );

        let segment_reader = searcher.segment_reader(0);
        let vector_reader = segment_reader.vector_reader(embedding_field);
        assert!(vector_reader.is_some(), "Expected vector reader after merge");
        let vector_reader = vector_reader.unwrap();

        // Verify columnar access after merge
        // All chunk_0 vectors should be contiguous
        let chunk0_vecs: Vec<_> = vector_reader
            .iter_vectors(embedding_field, "chunk_0")
            .collect();
        assert_eq!(chunk0_vecs.len(), 2, "Expected 2 chunk_0 vectors after merge");

        let summary_vecs: Vec<_> = vector_reader
            .iter_vectors(embedding_field, "summary")
            .collect();
        assert_eq!(
            summary_vecs.len(),
            1,
            "Expected 1 summary vector after merge"
        );

        // Verify vector IDs are preserved
        let ids: Vec<_> = vector_reader.vector_ids(embedding_field).unwrap().collect();
        assert!(ids.contains(&"chunk_0"), "Missing chunk_0 ID");
        assert!(ids.contains(&"summary"), "Missing summary ID");
    }

    /// Demonstrates using vector storage for K-means clustering.
    /// 
    /// This test shows the primary use case: extracting all vectors with a given ID
    /// across documents and feeding them to a clustering algorithm.
    #[test]
    fn test_vector_clustering_workflow() {
        use ndarray::{Array1, Array2, Axis};
        use rand::Rng;

        // Create schema with vector field
        let mut schema_builder = SchemaBuilder::new();
        let embedding_field = schema_builder.add_vector_map_field("embedding", ());
        let schema = schema_builder.build();

        // Create index
        let directory = RamDirectory::create();
        let index = Index::create(directory, schema.clone(), Default::default()).unwrap();
        let mut index_writer = index.writer::<TantivyDocument>(50_000_000).unwrap();

        // Generate 100 random 8-dimensional vectors in 3 clusters
        let mut rng = rand::thread_rng();
        let cluster_centers = [
            [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0f32, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [-5.0f32, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
        ];

        for i in 0..100 {
            let cluster_idx = i % 3;
            let center = &cluster_centers[cluster_idx];
            
            // Add noise to cluster center
            let vec: Vec<f32> = center
                .iter()
                .map(|&c| c + rng.gen_range(-0.5..0.5))
                .collect();

            let mut doc = TantivyDocument::new();
            doc.add_named_vector(embedding_field, "embedding", vec);
            index_writer.add_document(doc).unwrap();
        }

        index_writer.commit().unwrap();
        drop(index_writer);

        // === CLUSTERING WORKFLOW ===
        // Step 1: Extract all vectors using columnar access
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let mut all_vectors: Vec<Vec<f32>> = Vec::new();
        let mut doc_ids: Vec<u32> = Vec::new();

        for segment_reader in searcher.segment_readers() {
            if let Some(vector_reader) = segment_reader.vector_reader(embedding_field) {
                for (doc_id, vec) in vector_reader.iter_vectors(embedding_field, "embedding") {
                    doc_ids.push(doc_id);
                    all_vectors.push(vec.into_owned());
                }
            }
        }

        assert_eq!(all_vectors.len(), 100, "Should have 100 vectors");
        assert_eq!(all_vectors[0].len(), 8, "Each vector should be 8-dimensional");

        // Step 2: Convert to ndarray for clustering
        let n_samples = all_vectors.len();
        let n_dims = all_vectors[0].len();
        let flat: Vec<f32> = all_vectors.into_iter().flatten().collect();
        let data = Array2::from_shape_vec((n_samples, n_dims), flat).unwrap();

        // Step 3: Simple K-means clustering (k=3)
        let k = 3;
        let assignments = simple_kmeans(&data, k, 10);

        // Verify we got 3 clusters with reasonable distribution
        let mut cluster_counts = [0usize; 3];
        for &a in &assignments {
            cluster_counts[a] += 1;
        }

        // Each cluster should have roughly 33 points (with some tolerance)
        for (i, &count) in cluster_counts.iter().enumerate() {
            assert!(
                count >= 20 && count <= 50,
                "Cluster {} has {} points, expected ~33",
                i,
                count
            );
        }

        // Step 4: Compute cluster centroids
        let centroids = compute_centroids(&data, &assignments, k);
        assert_eq!(centroids.nrows(), k);
        assert_eq!(centroids.ncols(), n_dims);

        // Verify centroids are near original cluster centers (within tolerance)
        for i in 0..k {
            let centroid = centroids.row(i);
            let mut min_dist = f32::MAX;
            for center in &cluster_centers {
                let dist = centroid
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                min_dist = min_dist.min(dist);
            }
            assert!(
                min_dist < 2.0,
                "Centroid {} is too far from any cluster center: dist={}",
                i,
                min_dist
            );
        }
    }

    /// Simple K-means implementation for testing.
    /// Returns cluster assignments for each sample.
    fn simple_kmeans(data: &ndarray::Array2<f32>, k: usize, max_iters: usize) -> Vec<usize> {
        use ndarray::{Array1, Axis};

        let n_samples = data.nrows();
        let n_dims = data.ncols();

        // Initialize centroids using first k points
        let mut centroids: Vec<Array1<f32>> = (0..k)
            .map(|i| data.row(i * n_samples / k).to_owned())
            .collect();

        let mut assignments = vec![0usize; n_samples];

        for _ in 0..max_iters {
            // Assign each point to nearest centroid
            let mut changed = false;
            for (i, row) in data.rows().into_iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_dist = f32::MAX;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist: f32 = row
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();

                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            for c in 0..k {
                let mut sum = Array1::<f32>::zeros(n_dims);
                let mut count = 0;

                for (i, row) in data.rows().into_iter().enumerate() {
                    if assignments[i] == c {
                        sum = sum + &row.to_owned();
                        count += 1;
                    }
                }

                if count > 0 {
                    centroids[c] = sum / count as f32;
                }
            }
        }

        assignments
    }

    /// Compute centroids from data and assignments.
    fn compute_centroids(
        data: &ndarray::Array2<f32>,
        assignments: &[usize],
        k: usize,
    ) -> ndarray::Array2<f32> {
        use ndarray::{Array1, Array2};

        let n_dims = data.ncols();
        let mut centroids = Array2::<f32>::zeros((k, n_dims));

        for c in 0..k {
            let mut sum = Array1::<f32>::zeros(n_dims);
            let mut count = 0;

            for (i, row) in data.rows().into_iter().enumerate() {
                if assignments[i] == c {
                    sum = sum + &row.to_owned();
                    count += 1;
                }
            }

            if count > 0 {
                centroids.row_mut(c).assign(&(sum / count as f32));
            }
        }

        centroids
    }

    /// Test that vectors with different dimensions for the same vector_id
    /// are handled gracefully (currently they log a warning and skip).
    #[test]
    fn test_vector_dimension_mismatch() {
        // Create schema with vector field
        let mut schema_builder = SchemaBuilder::new();
        let embedding_field = schema_builder.add_vector_map_field("embedding", ());
        let schema = schema_builder.build();

        // Create index
        let directory = RamDirectory::create();
        let index = Index::create(directory, schema.clone(), Default::default()).unwrap();
        let mut index_writer = index.writer::<TantivyDocument>(50_000_000).unwrap();

        // Document 1: 3-dimensional vector
        let mut doc1 = TantivyDocument::new();
        doc1.add_named_vector(embedding_field, "chunk", vec![0.1, 0.2, 0.3]);
        index_writer.add_document(doc1).unwrap();

        // Document 2: 5-dimensional vector (DIFFERENT dimensions for same vector_id)
        let mut doc2 = TantivyDocument::new();
        doc2.add_named_vector(embedding_field, "chunk", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        index_writer.add_document(doc2).unwrap();

        // Document 3: 3-dimensional vector (same as doc1)
        let mut doc3 = TantivyDocument::new();
        doc3.add_named_vector(embedding_field, "chunk", vec![0.4, 0.5, 0.6]);
        index_writer.add_document(doc3).unwrap();

        // Commit - this should not panic, but may log warnings about dimension mismatches
        index_writer.commit().unwrap();
        drop(index_writer);

        // Read vectors back
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let mut vectors: Vec<(u32, Vec<f32>)> = Vec::new();
        for segment_reader in searcher.segment_readers() {
            if let Some(vector_reader) = segment_reader.vector_reader(embedding_field) {
                for (doc_id, vec) in vector_reader.iter_vectors(embedding_field, "chunk") {
                    vectors.push((doc_id, vec.into_owned()));
                }
            }
        }

        // The current behavior is that only vectors with matching dimensions are kept.
        // Doc1 has 3 dims, Doc2 has 5 dims (skipped), Doc3 has 3 dims.
        // So we should have 2 vectors, both 3-dimensional.
        assert_eq!(vectors.len(), 2, "Expected 2 vectors (dimension mismatch skipped)");
        for (doc_id, vec) in &vectors {
            assert_eq!(vec.len(), 3, "Doc {} has wrong dimensions: {:?}", doc_id, vec);
        }
    }
}

