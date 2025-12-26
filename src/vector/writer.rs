//! Vector writer for writing embedding vectors to segment files.
//!
//! The `VectorFieldsWriter` collects vectors during indexing and serializes them
//! to the segment's `.vec` file during flush. It handles:
//!
//! - Collecting named vectors from documents during `add_document`
//! - Columnar storage: vectors with the same string ID are stored together
//! - Presence bitsets for efficient sparse storage
//! - Multiple encoding formats (f32, f16, int8)
//! - Supporting document reordering during segment merges
//!
//! ## Storage Format (Columnar)
//!
//! ```text
//! Header:
//!   - magic: u32 (0x43455654 = "TVEC")
//!   - version: u8 (1)
//!   - encoding: u8 (0=f32, 1=f16, 2=int8)
//!   - num_fields: u32
//!   - field_ids: [u32; num_fields]
//!   - num_docs: u32
//!
//! Per field (ordered by field_id):
//!   - num_vector_ids: u32
//!   - For each vector_id (sorted alphabetically):
//!     - vector_id_len: u32
//!     - vector_id: [u8; vector_id_len]
//!     - dimensions: u32 (fixed for this vector_id)
//!     - presence_bitset: [u8; ceil(num_docs/8)] padded to u64
//!     - [if int8: scale: f32, zero_point: f32]
//!     - vectors: [encoded; dims × popcount(bitset)] (contiguous)
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;

use common::TerminatingWrite;

use super::format::{
    encode_vector, Int8QuantParams, PresenceBitset, VectorEncoding, VECTOR_MAGIC, VECTOR_VERSION,
};
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
    /// Encoding to use for serialization
    encoding: VectorEncoding,
}

impl VectorFieldsWriter {
    /// Creates a new VectorFieldsWriter from a schema.
    pub fn from_schema(schema: &Schema) -> Self {
        Self::from_schema_with_encoding(schema, VectorEncoding::default())
    }

    /// Creates a new VectorFieldsWriter with a specific encoding.
    pub fn from_schema_with_encoding(schema: &Schema, encoding: VectorEncoding) -> Self {
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
            encoding,
        }
    }

    /// Returns true if there are any vector fields in the schema.
    pub fn has_vector_fields(&self) -> bool {
        !self.vector_fields.is_empty()
    }

    /// Returns the encoding being used.
    pub fn encoding(&self) -> VectorEncoding {
        self.encoding
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

    /// Serializes all vectors to the writer in optimized columnar format.
    ///
    /// If `doc_id_map` is provided, vectors are remapped to the new doc_id order.
    /// The writer is terminated with a footer after serialization.
    pub fn serialize<W: TerminatingWrite>(
        &self,
        mut wrt: W,
        doc_id_map: Option<&DocIdMapping>,
    ) -> io::Result<()> {
        // Write header
        wrt.write_all(&VECTOR_MAGIC.to_le_bytes())?;
        wrt.write_all(&[VECTOR_VERSION])?;
        wrt.write_all(&[self.encoding as u8])?;

        // Write field count and field IDs
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

                // Determine dimensions (from first vector)
                let dimensions = doc_vectors
                    .values()
                    .next()
                    .map(|v| v.len() as u32)
                    .unwrap_or(0);
                wrt.write_all(&dimensions.to_le_bytes())?;

                // Build presence bitset and collect vectors in order
                let mut bitset = PresenceBitset::new(effective_num_docs);
                let mut ordered_vectors: Vec<&Vec<f32>> = Vec::new();

                if let Some(doc_id_mapping) = doc_id_map {
                    for (new_doc_id, old_doc_id) in doc_id_mapping.iter_old_doc_ids().enumerate() {
                        if let Some(vec) = doc_vectors.get(&old_doc_id) {
                            bitset.set(new_doc_id as u32);
                            ordered_vectors.push(vec);
                        }
                    }
                } else {
                    for doc_id in 0..self.num_docs {
                        if let Some(vec) = doc_vectors.get(&doc_id) {
                            bitset.set(doc_id);
                            ordered_vectors.push(vec);
                        }
                    }
                }

                // Write presence bitset
                let bitset_bytes = bitset.as_bytes();
                wrt.write_all(&bitset_bytes)?;

                // For int8 encoding, compute and write quantization params
                let quant_params = if self.encoding == VectorEncoding::Int8 {
                    let params =
                        Int8QuantParams::from_vectors(ordered_vectors.iter().map(|v| v.as_slice()));
                    wrt.write_all(&params.scale.to_le_bytes())?;
                    wrt.write_all(&params.zero_point.to_le_bytes())?;
                    Some(params)
                } else {
                    None
                };

                // Write vectors contiguously
                for vec in ordered_vectors {
                    let encoded = encode_vector(vec, self.encoding, quant_params.as_ref());
                    wrt.write_all(&encoded)?;
                }
            }
        }

        // Terminate with footer (adds magic bytes required by tantivy's file reading)
        wrt.terminate()
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
    pub fn get_vectors(&self, field: Field, vector_id: &str) -> Option<&HashMap<DocId, Vec<f32>>> {
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
