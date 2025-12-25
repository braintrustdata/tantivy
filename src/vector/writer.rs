//! Vector writer for writing embedding vectors to segment files.
//!
//! The `VectorFieldsWriter` collects vectors during indexing and serializes them
//! to the segment's `.vec` file during flush. It handles:
//!
//! - Collecting vectors from documents during `add_document`
//! - Serializing to binary format with proper doc_id mapping
//! - Supporting document reordering during segment merges

use std::collections::HashMap;
use std::io::{self, Write};

use common::TerminatingWrite;

use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::schema::document::{Document, ReferenceValue, ReferenceValueLeaf, Value};
use crate::schema::{Field, FieldType, Schema};
use crate::DocId;

/// Low-level writer for vector data in JSON Lines format.
///
/// Each document's vector is written as a JSON array on a single line.
/// Documents without vectors get an empty array `[]`.
pub struct VectorWriter<W: Write> {
    writer: W,
    num_docs: u32,
}

impl<W: Write> VectorWriter<W> {
    /// Creates a new VectorWriter.
    pub fn new(writer: W) -> Self {
        VectorWriter { writer, num_docs: 0 }
    }

    /// Adds a vector for the next document.
    pub fn add_vector(&mut self, vector: &[f32]) -> io::Result<()> {
        // Write as JSON array
        write!(self.writer, "[")?;
        for (i, v) in vector.iter().enumerate() {
            if i > 0 {
                write!(self.writer, ",")?;
            }
            write!(self.writer, "{}", v)?;
        }
        writeln!(self.writer, "]")?;
        self.num_docs += 1;
        Ok(())
    }

    /// Adds an empty vector placeholder for a document without a vector.
    pub fn add_empty(&mut self) -> io::Result<()> {
        writeln!(self.writer, "[]")?;
        self.num_docs += 1;
        Ok(())
    }

    /// Returns the number of documents written so far.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Finishes writing and flushes the underlying writer.
    pub fn finish(mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Returns the inner writer, consuming this VectorWriter.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// High-level writer for managing vector fields during indexing.
///
/// Collects vectors from documents and writes them to the segment file
/// during serialization.
pub struct VectorFieldsWriter {
    /// Map from field to vectors by doc_id
    /// Key is field_id, value is a map from doc_id to vector
    field_vectors: HashMap<u32, Vec<Option<Vec<f32>>>>,
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

        let field_vectors: HashMap<u32, Vec<Option<Vec<f32>>>> = vector_fields
            .iter()
            .map(|f| (f.field_id(), Vec::new()))
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
    pub fn add_document<D: Document>(&mut self, doc: &D) {
        let doc_id = self.num_docs;

        // Ensure we have space for this doc_id in all field vectors
        for vectors in self.field_vectors.values_mut() {
            if vectors.len() <= doc_id as usize {
                vectors.resize(doc_id as usize + 1, None);
            }
        }

        // Extract vectors from the document
        for (field, value) in doc.iter_fields_and_values() {
            if let Some(vectors) = self.field_vectors.get_mut(&field.field_id()) {
                let value_access = value as D::Value<'_>;
                if let ReferenceValue::Leaf(ReferenceValueLeaf::Vector(vec)) = value_access.as_value() {
                    vectors[doc_id as usize] = Some(vec.to_vec());
                }
            }
        }

        self.num_docs += 1;
    }

    /// Returns the memory usage estimate.
    pub fn mem_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        for vectors in self.field_vectors.values() {
            total += vectors.capacity() * std::mem::size_of::<Option<Vec<f32>>>();
            for opt_vec in vectors {
                if let Some(vec) = opt_vec {
                    total += vec.capacity() * std::mem::size_of::<f32>();
                }
            }
        }
        total
    }

    /// Serializes all vectors to the writer.
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
        wrt.write_all(&self.num_docs.to_le_bytes())?;

        // Write vectors for each field, in doc_id order
        for field in &self.vector_fields {
            let vectors = self.field_vectors.get(&field.field_id()).unwrap();

            if let Some(doc_id_mapping) = doc_id_map {
                // Remap doc IDs
                for old_doc_id in doc_id_mapping.iter_old_doc_ids() {
                    Self::write_vector_entry(&mut wrt, vectors.get(old_doc_id as usize).and_then(|v| v.as_ref()))?;
                }
            } else {
                // No remapping needed
                for i in 0..self.num_docs {
                    Self::write_vector_entry(&mut wrt, vectors.get(i as usize).and_then(|v| v.as_ref()))?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_vectors() {
        let mut buffer = Vec::new();
        let mut writer = VectorWriter::new(&mut buffer);

        writer.add_vector(&[1.0, 2.0, 3.0]).unwrap();
        writer.add_empty().unwrap();
        writer.add_vector(&[4.5]).unwrap();
        writer.finish().unwrap();

        let output = String::from_utf8(buffer).unwrap();
        assert_eq!(output, "[1,2,3]\n[]\n[4.5]\n");
    }

    #[test]
    fn test_empty_writer() {
        let mut buffer = Vec::new();
        let writer = VectorWriter::new(&mut buffer);
        writer.finish().unwrap();

        assert!(buffer.is_empty());
    }
}

