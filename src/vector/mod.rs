//! Vector storage module for Tantivy.
//!
//! This module provides vector field support for storing embedding vectors (arrays of f32).
//! Vectors are stored in a separate `.vec` file per segment and are fully integrated with
//! Tantivy's segment lifecycle (flush, merge, garbage collection).
//!
//! # Features
//!
//! - **Variable-length vectors**: Each document can have vectors of different dimensions
//! - **Optional vectors**: Documents may omit vectors for any vector field
//! - **Segment lifecycle integration**: Vector files are automatically managed during
//!   segment merges and garbage collection
//! - **Efficient binary format**: Vectors stored in compact binary format (header + raw f32 data)
//!
//! # Usage
//!
//! ## Schema Definition
//!
//! ```rust
//! use tantivy::schema::{SchemaBuilder, STORED, TEXT};
//!
//! let mut schema_builder = SchemaBuilder::new();
//! let title = schema_builder.add_text_field("title", TEXT | STORED);
//! let embedding = schema_builder.add_vector_field("embedding", ());
//! let schema = schema_builder.build();
//! ```
//!
//! ## Indexing Documents with Vectors
//!
//! ```rust,ignore
//! use tantivy::{Index, TantivyDocument};
//!
//! let index = Index::create_in_ram(schema.clone());
//! let mut index_writer = index.writer(50_000_000)?;
//!
//! // Document with a vector
//! let mut doc = TantivyDocument::new();
//! doc.add_text(title, "My document");
//! doc.add_field_value(embedding, vec![0.1f32, 0.2, 0.3, 0.4]);
//! index_writer.add_document(doc)?;
//!
//! // Document without a vector (optional)
//! let mut doc2 = TantivyDocument::new();
//! doc2.add_text(title, "Another document");
//! index_writer.add_document(doc2)?;
//!
//! index_writer.commit()?;
//! ```
//!
//! ## Reading Vectors
//!
//! ```rust,ignore
//! let reader = index.reader()?;
//! let searcher = reader.searcher();
//!
//! for segment_reader in searcher.segment_readers() {
//!     if let Some(vector_reader) = segment_reader.vector_reader()?.as_ref() {
//!         // Get vector for doc_id 0
//!         if let Some(vec) = vector_reader.get(embedding, 0) {
//!             println!("Vector: {:?}", vec);
//!         }
//!     }
//! }
//! ```
//!
//! # File Format
//!
//! The `.vec` file uses a simple binary format:
//!
//! ```text
//! Header:
//!   - num_fields: u32 (number of vector fields)
//!   - field_ids: [u32; num_fields] (field IDs in order)
//!   - num_docs: u32 (total documents in segment)
//!
//! Per-field data (in field_id order):
//!   For each document (in doc_id order):
//!     - vector_len: u32 (0 if no vector)
//!     - vector_data: [f32; vector_len]
//!
//! Footer:
//!   - 8-byte magic number (Tantivy file format requirement)
//! ```
//!
//! # Future Work
//!
//! Future versions may add:
//! - Integration with vector search libraries (FAISS, USearch)
//! - Approximate nearest neighbor (ANN) search
//! - Quantization for memory efficiency

mod reader;
mod writer;

pub use reader::VectorReader;
pub use writer::{VectorFieldsWriter, VectorWriter};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::RamDirectory;
    use crate::schema::document::Document;
    use crate::schema::{Field, SchemaBuilder, STORED, TEXT};
    use crate::{Index, IndexWriter, TantivyDocument};
    use std::io::Cursor;

    #[test]
    fn test_vector_fields_roundtrip() {
        // Create a schema with a vector field
        let mut schema_builder = SchemaBuilder::new();
        let _vec_field = schema_builder.add_vector_field("embedding", ());
        let _schema = schema_builder.build();

        // VectorFieldsWriter is created from schema and used during indexing
        // The full integration is tested in test_vector_index_integration
    }

    #[test]
    fn test_vector_binary_format_roundtrip() {
        // Manually write binary data and read it back
        let mut buffer = Vec::new();

        // Header: 1 field
        buffer.extend_from_slice(&1u32.to_le_bytes());
        // Field ID: 0
        buffer.extend_from_slice(&0u32.to_le_bytes());
        // Num docs: 3
        buffer.extend_from_slice(&3u32.to_le_bytes());

        // Doc 0: [0.1, 0.2, 0.3]
        buffer.extend_from_slice(&3u32.to_le_bytes());
        buffer.extend_from_slice(&0.1f32.to_le_bytes());
        buffer.extend_from_slice(&0.2f32.to_le_bytes());
        buffer.extend_from_slice(&0.3f32.to_le_bytes());

        // Doc 1: [] (empty)
        buffer.extend_from_slice(&0u32.to_le_bytes());

        // Doc 2: [1.0, 2.0]
        buffer.extend_from_slice(&2u32.to_le_bytes());
        buffer.extend_from_slice(&1.0f32.to_le_bytes());
        buffer.extend_from_slice(&2.0f32.to_le_bytes());

        // Read them back
        let reader = VectorReader::open(Cursor::new(&buffer)).unwrap();
        let field = Field::from_field_id(0);

        assert_eq!(reader.num_docs(), 3);
        assert_eq!(reader.get(field, 0), Some(&[0.1f32, 0.2, 0.3][..]));
        assert_eq!(reader.get(field, 1), Some(&[][..])); // empty vector
        assert_eq!(reader.get(field, 2), Some(&[1.0f32, 2.0][..]));
        assert_eq!(reader.get(field, 3), None); // out of bounds
    }

    /// End-to-end test: Create an index with vectors, add documents, and read vectors back
    #[test]
    fn test_vector_index_integration() {
        use crate::directory::Directory;
        use crate::schema::document::{ReferenceValue, ReferenceValueLeaf, Value};
        use crate::SegmentComponent;
        
        // Create schema with a text field and a vector field
        let mut schema_builder = SchemaBuilder::new();
        let title_field = schema_builder.add_text_field("title", TEXT | STORED);
        let embedding_field = schema_builder.add_vector_field("embedding", ());
        let schema = schema_builder.build();

        // First, test that vectors are stored correctly in TantivyDocument
        let mut doc1 = TantivyDocument::new();
        doc1.add_text(title_field, "First document");
        doc1.add_field_value(embedding_field, vec![0.1f32, 0.2, 0.3]);
        
        // Verify the vector is in the document
        let mut found_vector = false;
        for (field, value) in doc1.iter_fields_and_values() {
            if field == embedding_field {
                if let ReferenceValue::Leaf(ReferenceValueLeaf::Vector(vec)) = value.as_value() {
                    assert_eq!(vec, &[0.1f32, 0.2, 0.3]);
                    found_vector = true;
                }
            }
        }
        assert!(found_vector, "Vector not found in document");

        // Create index in RAM
        let directory = RamDirectory::create();
        let index = Index::create(directory, schema.clone(), Default::default()).unwrap();

        // Add documents with vectors
        let mut index_writer = index.writer::<TantivyDocument>(50_000_000).unwrap();

        // Document 1: with a 3-dimensional vector
        let mut doc1 = TantivyDocument::new();
        doc1.add_text(title_field, "First document");
        doc1.add_field_value(embedding_field, vec![0.1f32, 0.2, 0.3]);
        index_writer.add_document(doc1).unwrap();

        // Document 2: with a 2-dimensional vector (different size!)
        let mut doc2 = TantivyDocument::new();
        doc2.add_text(title_field, "Second document");
        doc2.add_field_value(embedding_field, vec![1.0f32, 2.0]);
        index_writer.add_document(doc2).unwrap();

        // Document 3: no vector (testing optional vectors)
        let mut doc3 = TantivyDocument::new();
        doc3.add_text(title_field, "Third document");
        index_writer.add_document(doc3).unwrap();

        // Commit and drop the index_writer to release the lock
        index_writer.commit().unwrap();
        drop(index_writer);
        
        // Debug: Check what files exist
        let segments = index.searchable_segments().unwrap();
        eprintln!("Number of segments: {}", segments.len());
        for segment in &segments {
            eprintln!("Segment: {:?}", segment.id());
            let vec_path = segment.meta().relative_path(SegmentComponent::Vectors);
            let exists = index.directory().exists(&vec_path).unwrap();
            eprintln!("Vector file {} exists: {}", vec_path.display(), exists);
        }
        
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        // Verify we can read vectors back
        assert_eq!(searcher.num_docs(), 3);

        // Get segment readers and verify vectors
        // Note: Each document is in a separate segment (3 segments total)
        // We need to find which segment has which vector
        let mut found_vectors = vec![false; 3];
        
        for segment_reader in searcher.segment_readers() {
            let vector_reader = segment_reader.vector_reader().unwrap();
            assert!(vector_reader.is_some(), "Expected vector reader to exist");
            let vector_reader = vector_reader.unwrap();
            
            // Each segment has 1 document (doc_id = 0)
            let vec = vector_reader.get(embedding_field, 0);
            
            // Match vector to expected document
            if let Some(v) = vec {
                if v == &[0.1f32, 0.2, 0.3] {
                    found_vectors[0] = true;
                } else if v == &[1.0f32, 2.0] {
                    found_vectors[1] = true;
                } else if v.is_empty() {
                    found_vectors[2] = true;
                }
            }
        }
        
        assert!(found_vectors[0], "Vector [0.1, 0.2, 0.3] not found");
        assert!(found_vectors[1], "Vector [1.0, 2.0] not found");
        assert!(found_vectors[2], "Empty vector not found");
        
        // Now test segment merging
        let segment_ids: Vec<_> = index.searchable_segments().unwrap()
            .iter()
            .map(|s| s.id())
            .collect();
        let mut index_writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
        index_writer.merge(&segment_ids).wait().unwrap();
        index_writer.commit().unwrap();
        drop(index_writer);
        
        // Reload the reader after merge
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        
        // After merge, should have 1 segment with all 3 documents
        eprintln!("After merge: {} segments", searcher.segment_readers().len());
        assert_eq!(searcher.segment_readers().len(), 1, "Expected 1 segment after merge");
        
        let segment_reader = searcher.segment_reader(0);
        let vector_reader = segment_reader.vector_reader().unwrap();
        assert!(vector_reader.is_some(), "Expected vector reader after merge");
        let vector_reader = vector_reader.unwrap();
        
        // Verify all vectors are present in the merged segment
        // Note: doc order may have changed after merge
        let mut merged_vectors: Vec<Option<Vec<f32>>> = (0..3)
            .map(|i| vector_reader.get(embedding_field, i).map(|v| v.to_vec()))
            .collect();
        
        // Sort to compare consistently
        merged_vectors.sort_by(|a, b| {
            match (a, b) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(a), Some(b)) => {
                    let len_cmp = a.len().cmp(&b.len());
                    if len_cmp != std::cmp::Ordering::Equal {
                        len_cmp
                    } else {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    }
                }
            }
        });
        
        eprintln!("Merged vectors: {:?}", merged_vectors);
        
        // Should have: empty vec, [1.0, 2.0], [0.1, 0.2, 0.3]
        assert_eq!(merged_vectors.len(), 3);
        assert!(merged_vectors.iter().any(|v| v.as_ref().map_or(false, |v| v.is_empty())), "Missing empty vector");
        assert!(merged_vectors.iter().any(|v| v.as_ref().map_or(false, |v| v == &[1.0, 2.0])), "Missing [1.0, 2.0]");
        assert!(merged_vectors.iter().any(|v| v.as_ref().map_or(false, |v| v == &[0.1, 0.2, 0.3])), "Missing [0.1, 0.2, 0.3]");
    }
}

