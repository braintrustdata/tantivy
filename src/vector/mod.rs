//! Vector storage module for Tantivy.
//!
//! This module provides vector field support for storing named embedding vectors.
//! Each document can have multiple named vectors per field, stored in columnar format
//! for efficient batch access.
//!
//! # Features
//!
//! - **Named vectors**: Each document can have multiple vectors identified by string IDs
//!   (e.g., "chunk_0", "chunk_1", "summary")
//! - **Columnar storage**: Vectors with the same ID are stored contiguously for efficient
//!   batch retrieval across documents
//! - **Variable-length vectors**: Each vector can have different dimensions
//! - **Optional vectors**: Documents may omit any or all vector IDs
//! - **Segment lifecycle integration**: Vector files are automatically managed during
//!   segment merges and garbage collection
//!
//! # Usage
//!
//! ## Schema Definition
//!
//! ```rust
//! use tantivy::schema::SchemaBuilder;
//!
//! let mut schema_builder = SchemaBuilder::new();
//! let embedding = schema_builder.add_vector_field("embedding", ());
//! let schema = schema_builder.build();
//! ```
//!
//! ## Indexing Documents with Named Vectors
//!
//! ```rust,ignore
//! use tantivy::{Index, TantivyDocument};
//!
//! let index = Index::create_in_ram(schema.clone());
//! let mut index_writer = index.writer(50_000_000)?;
//!
//! // Document with multiple named vectors
//! let mut doc = TantivyDocument::new();
//! doc.add_named_vector(embedding, "chunk_0", vec![0.1, 0.2, 0.3]);
//! doc.add_named_vector(embedding, "chunk_1", vec![0.4, 0.5, 0.6]);
//! doc.add_named_vector(embedding, "summary", vec![1.0, 2.0]);
//! index_writer.add_document(doc)?;
//!
//! // Another document with only some vectors
//! let mut doc2 = TantivyDocument::new();
//! doc2.add_named_vector(embedding, "chunk_0", vec![0.7, 0.8, 0.9]);
//! // No chunk_1 or summary for this doc
//! index_writer.add_document(doc2)?;
//!
//! index_writer.commit()?;
//! ```
//!
//! ## Reading Vectors (Columnar Access)
//!
//! ```rust,ignore
//! let reader = index.reader()?;
//! let searcher = reader.searcher();
//!
//! for segment_reader in searcher.segment_readers() {
//!     if let Some(vector_reader) = segment_reader.vector_reader(embedding) {
//!         // Primary access pattern: get all vectors with a given ID across documents
//!         for (doc_id, vec) in vector_reader.iter_vectors("chunk_0") {
//!             println!("Doc {}: {:?}", doc_id, vec);
//!         }
//!
//!         // Or get a specific document's vector
//!         if let Some(vec) = vector_reader.get(embedding, "summary", doc_id) {
//!             println!("Summary: {:?}", vec);
//!         }
//!     }
//! }
//! ```
//!
//! # File Format (Columnar by Vector ID)
//!
//! The `.vec` file stores vectors grouped by vector ID for efficient batch access:
//!
//! ```text
//! Header:
//!   - num_fields: u32
//!   - field_ids: [u32; num_fields]
//!   - num_docs: u32
//!
//! Per field (in field_id order):
//!   - num_vector_ids: u32 (distinct string IDs for this field)
//!   - For each vector_id (sorted alphabetically):
//!     - vector_id_len: u32
//!     - vector_id: [u8; vector_id_len]
//!     - For each doc (0..num_docs, in doc_id order):
//!       - vector_len: u32 (0 if doc doesn't have this vector_id)
//!       - vector_data: [f32; vector_len] (if vector_len > 0)
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
        let vec_field = schema_builder.add_vector_field("embedding", ());
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
        // Manually write columnar binary data and read it back
        let mut buffer = Vec::new();

        // Header: 1 field
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
        // Doc 0: [0.1, 0.2, 0.3]
        buffer.extend_from_slice(&3u32.to_le_bytes());
        buffer.extend_from_slice(&0.1f32.to_le_bytes());
        buffer.extend_from_slice(&0.2f32.to_le_bytes());
        buffer.extend_from_slice(&0.3f32.to_le_bytes());
        // Doc 1: [1.0, 2.0]
        buffer.extend_from_slice(&2u32.to_le_bytes());
        buffer.extend_from_slice(&1.0f32.to_le_bytes());
        buffer.extend_from_slice(&2.0f32.to_le_bytes());

        // Read them back
        let reader = VectorReader::open(Cursor::new(&buffer)).unwrap();
        let field = Field::from_field_id(0);

        assert_eq!(reader.num_docs(), 2);
        assert_eq!(
            reader.get(field, "main", 0),
            Some(&[0.1f32, 0.2, 0.3][..])
        );
        assert_eq!(reader.get(field, "main", 1), Some(&[1.0f32, 2.0][..]));
        assert_eq!(reader.get(field, "main", 2), None); // out of bounds
        assert_eq!(reader.get(field, "nonexistent", 0), None); // no such ID
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
        let embedding_field = schema_builder.add_vector_field("embedding", ());
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
                if let ReferenceValue::Leaf(ReferenceValueLeaf::Vector(vec_map)) = value.as_value()
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
        doc1.add_named_vector(embedding_field, "summary", vec![10.0, 20.0]);
        index_writer.add_document(doc1).unwrap();

        // Document 2: only chunk_0
        let mut doc2 = TantivyDocument::new();
        doc2.add_text(title_field, "Second document");
        doc2.add_named_vector(embedding_field, "chunk_0", vec![1.0, 2.0]);
        index_writer.add_document(doc2).unwrap();

        // Document 3: no vectors
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
        eprintln!(
            "After merge: {} segments",
            searcher.segment_readers().len()
        );
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
        eprintln!("Merged chunk_0 vectors: {:?}", chunk0_vecs);
        assert_eq!(chunk0_vecs.len(), 2, "Expected 2 chunk_0 vectors after merge");

        let summary_vecs: Vec<_> = vector_reader
            .iter_vectors(embedding_field, "summary")
            .collect();
        eprintln!("Merged summary vectors: {:?}", summary_vecs);
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
}

