//! Vector writer for writing embedding vectors to segment files.
//!
//! The `VectorFieldsWriter` collects vectors during indexing and serializes them
//! to the segment's `.vec` file during flush. It handles:
//!
//! - Collecting named vectors from documents during `add_document`
//! - Columnar storage: vectors with the same string ID are stored together
//! - Serializing to binary format with proper doc_id mapping
//! - Supporting document reordering during segment merges
//!
//! ## Storage Format (Columnar by Vector ID)
//!
//! The format is optimized for accessing all vectors with a given ID across documents:
//!
//! ```text
//! Header:
//!   - num_fields: u32
//!   - field_ids: [u32; num_fields]
//!   - num_docs: u32
//!
//! Per field (ordered by field_id):
//!   - num_vector_ids: u32 (number of distinct string IDs)
//!   - For each vector_id (sorted alphabetically):
//!     - vector_id_len: u32
//!     - vector_id: [u8; vector_id_len]
//!     - For each doc (0..num_docs, in doc_id order):
//!       - vector_len: u32 (0 if doc doesn't have this vector_id)
//!       - vector_data: [f32; vector_len] (if vector_len > 0)
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{self, Write};

use common::TerminatingWrite;

use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::schema::document::{Document, ReferenceValue, ReferenceValueLeaf, Value};
use crate::schema::{Field, FieldType, Schema};
use crate::DocId;

/// High-level writer for managing vector fields during indexing.
///
/// Collects named vectors from documents and writes them in columnar format
/// where all vectors with the same string ID are stored contiguously.
pub struct VectorFieldsWriter {
    /// Map from field_id to (vector_id -> (doc_id -> vector))
    /// Organized for efficient columnar serialization
    field_vectors: HashMap<u32, BTreeMap<String, HashMap<DocId, Vec<f32>>>>,
    /// List of vector field IDs
    vector_fields: Vec<Field>,
    num_docs: DocId,
}

impl VectorFieldsWriter {
    /// Creates a new VectorFieldsWriter from a schema.
    pub fn from_schema(schema: &Schema) -> Self {
        let vector_fields: Vec<Field> = schema
            .fields()
            .filter_map(|(field, entry)| {
                if matches!(entry.field_type(), FieldType::Vector(_)) {
                    Some(field)
                } else {
                    None
                }
            })
            .collect();

        let field_vectors: HashMap<u32, BTreeMap<String, HashMap<DocId, Vec<f32>>>> = vector_fields
            .iter()
            .map(|f| (f.field_id(), BTreeMap::new()))
            .collect();

        VectorFieldsWriter {
            field_vectors,
            vector_fields,
            num_docs: 0,
        }
    }

    /// Returns true if there are any vector fields in the schema.
    pub fn has_vector_fields(&self) -> bool {
        !self.vector_fields.is_empty()
    }

    /// Indexes vectors from a document.
    ///
    /// Vectors are stored as a map of string IDs to f32 arrays.
    pub fn add_document<D: Document>(&mut self, doc: &D) {
        let doc_id = self.num_docs;

        // Extract vectors from the document
        for (field, value) in doc.iter_fields_and_values() {
            if let Some(field_data) = self.field_vectors.get_mut(&field.field_id()) {
                let value_access = value as D::Value<'_>;
                if let ReferenceValue::Leaf(ReferenceValueLeaf::Vector(vector_map)) =
                    value_access.as_value()
                {
                    // Add each named vector to the columnar storage
                    for (vector_id, vec) in vector_map {
                        field_data
                            .entry(vector_id.clone())
                            .or_default()
                            .insert(doc_id, vec.clone());
                    }
                }
            }
        }

        self.num_docs += 1;
    }

    /// Returns the memory usage estimate.
    pub fn mem_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        for field_data in self.field_vectors.values() {
            for (vector_id, doc_vectors) in field_data {
                total += vector_id.capacity();
                total += doc_vectors.len()
                    * (std::mem::size_of::<DocId>() + std::mem::size_of::<Vec<f32>>());
                for vec in doc_vectors.values() {
                    total += vec.capacity() * std::mem::size_of::<f32>();
                }
            }
        }
        total
    }

    /// Serializes all vectors to the writer in columnar format.
    ///
    /// If `doc_id_map` is provided, vectors are remapped to the new doc_id order.
    /// The writer is terminated with a footer after serialization.
    pub fn serialize<W: TerminatingWrite>(
        &self,
        mut wrt: W,
        doc_id_map: Option<&DocIdMapping>,
    ) -> io::Result<()> {
        // Write header: field count and field IDs
        let num_fields = self.vector_fields.len() as u32;
        wrt.write_all(&num_fields.to_le_bytes())?;

        for field in &self.vector_fields {
            wrt.write_all(&field.field_id().to_le_bytes())?;
        }

        // Write number of docs
        let effective_num_docs = doc_id_map
            .map(|m| m.iter_old_doc_ids().count() as u32)
            .unwrap_or(self.num_docs);
        wrt.write_all(&effective_num_docs.to_le_bytes())?;

        // Write vectors for each field in columnar format
        for field in &self.vector_fields {
            let field_data = self.field_vectors.get(&field.field_id()).unwrap();

            // Write number of vector IDs for this field
            let num_vector_ids = field_data.len() as u32;
            wrt.write_all(&num_vector_ids.to_le_bytes())?;

            // Write each vector ID's data (BTreeMap is already sorted)
            for (vector_id, doc_vectors) in field_data {
                // Write vector ID string
                let id_bytes = vector_id.as_bytes();
                wrt.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
                wrt.write_all(id_bytes)?;

                // Write vectors for all docs in order
                if let Some(doc_id_mapping) = doc_id_map {
                    for old_doc_id in doc_id_mapping.iter_old_doc_ids() {
                        Self::write_vector_entry(&mut wrt, doc_vectors.get(&old_doc_id))?;
                    }
                } else {
                    for doc_id in 0..self.num_docs {
                        Self::write_vector_entry(&mut wrt, doc_vectors.get(&doc_id))?;
                    }
                }
            }
        }

        // Terminate with footer (adds magic bytes required by tantivy's file reading)
        wrt.terminate()
    }

    fn write_vector_entry<W: Write>(wrt: &mut W, vector: Option<&Vec<f32>>) -> io::Result<()> {
        match vector {
            Some(vec) => {
                // Write vector length followed by vector data
                let len = vec.len() as u32;
                wrt.write_all(&len.to_le_bytes())?;
                for v in vec {
                    wrt.write_all(&v.to_le_bytes())?;
                }
            }
            None => {
                // Write length 0 for missing vector
                wrt.write_all(&0u32.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Gets all vector IDs across all fields (for merging).
    pub fn all_vector_ids(&self) -> BTreeSet<String> {
        let mut ids = BTreeSet::new();
        for field_data in self.field_vectors.values() {
            ids.extend(field_data.keys().cloned());
        }
        ids
    }

    /// Gets vectors for a specific field and vector ID.
    pub fn get_vectors(
        &self,
        field: Field,
        vector_id: &str,
    ) -> Option<&HashMap<DocId, Vec<f32>>> {
        self.field_vectors
            .get(&field.field_id())
            .and_then(|field_data| field_data.get(vector_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::SchemaBuilder;
    use crate::TantivyDocument;
    use std::collections::BTreeMap;

    #[test]
    fn test_columnar_vector_writer() {
        let mut schema_builder = SchemaBuilder::new();
        let vec_field = schema_builder.add_vector_field("embedding", ());
        let schema = schema_builder.build();

        let mut writer = VectorFieldsWriter::from_schema(&schema);

        // Doc 0: two named vectors
        let mut doc0 = TantivyDocument::new();
        let mut vectors0 = BTreeMap::new();
        vectors0.insert("chunk_0".to_string(), vec![1.0, 2.0]);
        vectors0.insert("summary".to_string(), vec![10.0, 20.0, 30.0]);
        doc0.add_field_value(vec_field, vectors0);
        writer.add_document(&doc0);

        // Doc 1: only chunk_0
        let mut doc1 = TantivyDocument::new();
        let mut vectors1 = BTreeMap::new();
        vectors1.insert("chunk_0".to_string(), vec![3.0, 4.0]);
        doc1.add_field_value(vec_field, vectors1);
        writer.add_document(&doc1);

        // Doc 2: only summary
        let mut doc2 = TantivyDocument::new();
        let mut vectors2 = BTreeMap::new();
        vectors2.insert("summary".to_string(), vec![40.0, 50.0, 60.0]);
        doc2.add_field_value(vec_field, vectors2);
        writer.add_document(&doc2);

        // Verify columnar structure
        let chunk0_vecs = writer.get_vectors(vec_field, "chunk_0").unwrap();
        assert_eq!(chunk0_vecs.get(&0), Some(&vec![1.0, 2.0]));
        assert_eq!(chunk0_vecs.get(&1), Some(&vec![3.0, 4.0]));
        assert_eq!(chunk0_vecs.get(&2), None); // Doc 2 doesn't have chunk_0

        let summary_vecs = writer.get_vectors(vec_field, "summary").unwrap();
        assert_eq!(summary_vecs.get(&0), Some(&vec![10.0, 20.0, 30.0]));
        assert_eq!(summary_vecs.get(&1), None); // Doc 1 doesn't have summary
        assert_eq!(summary_vecs.get(&2), Some(&vec![40.0, 50.0, 60.0]));
    }

    #[test]
    fn test_empty_writer() {
        let schema = SchemaBuilder::new().build();
        let writer = VectorFieldsWriter::from_schema(&schema);
        assert!(!writer.has_vector_fields());
    }
}

