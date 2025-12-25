//! Vector reader for reading embedding vectors from segment files.
//!
//! The `VectorReader` provides access to vectors stored in columnar format,
//! where all vectors with the same string ID are stored contiguously.
//!
//! # Columnar Access Pattern
//!
//! The primary access pattern is retrieving all vectors for a given vector ID
//! across all documents in the segment:
//!
//! ```rust,ignore
//! let segment_reader = searcher.segment_reader(0);
//! if let Some(vector_reader) = segment_reader.vector_reader(embedding_field) {
//!     // Get all "chunk_0" vectors across all documents
//!     for (doc_id, vec) in vector_reader.iter_vectors("chunk_0") {
//!         println!("Doc {}: {:?}", doc_id, vec);
//!     }
//!     
//!     // Or get a specific document's vector by ID
//!     let vec = vector_reader.get("chunk_0", doc_id);
//! }
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{self, Read};

use crate::schema::Field;
use crate::DocId;

/// Reader for vector data stored in columnar format.
///
/// Vectors are organized by vector ID (string) for efficient batch access:
/// - `field -> vector_id -> [doc0_vec, doc1_vec, ...]`
///
/// This allows efficient retrieval of all vectors with a given ID across documents.
pub struct VectorReader {
    /// Map from field_id to (vector_id -> (doc_id -> vector))
    /// Using BTreeMap for vector_id to maintain sorted order
    field_vectors: HashMap<u32, BTreeMap<String, Vec<Option<Vec<f32>>>>>,
    /// Number of documents
    num_docs: u32,
}

impl VectorReader {
    /// Opens a VectorReader from a binary reader.
    ///
    /// Reads all vectors into memory for fast random access.
    pub fn open<R: Read>(mut reader: R) -> io::Result<Self> {
        // Read header
        let num_fields = Self::read_u32(&mut reader)?;

        let mut field_ids = Vec::with_capacity(num_fields as usize);
        for _ in 0..num_fields {
            field_ids.push(Self::read_u32(&mut reader)?);
        }

        let num_docs = Self::read_u32(&mut reader)?;

        // Read vectors for each field in columnar format
        let mut field_vectors = HashMap::new();
        for field_id in field_ids {
            let num_vector_ids = Self::read_u32(&mut reader)?;
            let mut vector_id_map = BTreeMap::new();

            for _ in 0..num_vector_ids {
                // Read vector ID string
                let id_len = Self::read_u32(&mut reader)? as usize;
                let mut id_bytes = vec![0u8; id_len];
                reader.read_exact(&mut id_bytes)?;
                let vector_id = String::from_utf8(id_bytes)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

                // Read vectors for all docs
                let mut doc_vectors = Vec::with_capacity(num_docs as usize);
                for _ in 0..num_docs {
                    let vec_len = Self::read_u32(&mut reader)?;
                    if vec_len > 0 {
                        let mut vec = Vec::with_capacity(vec_len as usize);
                        for _ in 0..vec_len {
                            vec.push(Self::read_f32(&mut reader)?);
                        }
                        doc_vectors.push(Some(vec));
                    } else {
                        doc_vectors.push(None);
                    }
                }

                vector_id_map.insert(vector_id, doc_vectors);
            }

            field_vectors.insert(field_id, vector_id_map);
        }

        Ok(VectorReader {
            field_vectors,
            num_docs,
        })
    }

    fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    /// Creates an empty VectorReader (for segments with no vector data).
    pub fn empty() -> Self {
        VectorReader {
            field_vectors: HashMap::new(),
            num_docs: 0,
        }
    }

    /// Returns the number of documents in this reader.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Gets the vector for a given field, vector ID, and doc_id.
    ///
    /// Returns `None` if field, vector_id, or doc_id is not found,
    /// or if the document doesn't have this vector ID.
    pub fn get(&self, field: Field, vector_id: &str, doc_id: DocId) -> Option<&[f32]> {
        self.field_vectors
            .get(&field.field_id())
            .and_then(|vector_id_map| vector_id_map.get(vector_id))
            .and_then(|doc_vectors| doc_vectors.get(doc_id as usize))
            .and_then(|opt_vec| opt_vec.as_ref())
            .map(|v| v.as_slice())
    }

    /// Returns all vector IDs available for a given field.
    pub fn vector_ids(&self, field: Field) -> Option<impl Iterator<Item = &str>> {
        self.field_vectors
            .get(&field.field_id())
            .map(|vector_id_map| vector_id_map.keys().map(|s| s.as_str()))
    }

    /// Returns all vector IDs across all fields.
    pub fn all_vector_ids(&self) -> BTreeSet<&str> {
        let mut ids = BTreeSet::new();
        for vector_id_map in self.field_vectors.values() {
            ids.extend(vector_id_map.keys().map(|s| s.as_str()));
        }
        ids
    }

    /// Iterates over all vectors for a given field and vector ID.
    ///
    /// Yields (doc_id, &[f32]) pairs for documents that have this vector.
    /// This is the primary access pattern for columnar vector storage.
    pub fn iter_vectors(
        &self,
        field: Field,
        vector_id: &str,
    ) -> impl Iterator<Item = (DocId, &[f32])> {
        self.field_vectors
            .get(&field.field_id())
            .and_then(|vector_id_map| vector_id_map.get(vector_id))
            .into_iter()
            .flat_map(|doc_vectors| {
                doc_vectors
                    .iter()
                    .enumerate()
                    .filter_map(|(doc_id, opt_vec)| {
                        opt_vec
                            .as_ref()
                            .map(|vec| (doc_id as DocId, vec.as_slice()))
                    })
            })
    }

    /// Gets all vectors for a document across all vector IDs for a field.
    ///
    /// Returns a map of vector_id -> &[f32] for the given document.
    pub fn get_doc_vectors(&self, field: Field, doc_id: DocId) -> BTreeMap<&str, &[f32]> {
        let mut result = BTreeMap::new();
        if let Some(vector_id_map) = self.field_vectors.get(&field.field_id()) {
            for (vector_id, doc_vectors) in vector_id_map {
                if let Some(Some(vec)) = doc_vectors.get(doc_id as usize) {
                    result.insert(vector_id.as_str(), vec.as_slice());
                }
            }
        }
        result
    }

    /// Returns an iterator over all field IDs that have vectors.
    pub fn field_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.field_vectors.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_columnar_test_data() -> Vec<u8> {
        let mut data = Vec::new();
        // Header: 1 field
        data.extend_from_slice(&1u32.to_le_bytes());
        // Field ID: 0
        data.extend_from_slice(&0u32.to_le_bytes());
        // Num docs: 3
        data.extend_from_slice(&3u32.to_le_bytes());

        // Field 0 has 2 vector IDs: "chunk_0" and "summary"
        data.extend_from_slice(&2u32.to_le_bytes());

        // Vector ID "chunk_0" (sorted alphabetically comes first)
        let chunk_id = b"chunk_0";
        data.extend_from_slice(&(chunk_id.len() as u32).to_le_bytes());
        data.extend_from_slice(chunk_id);
        // Doc 0: [1.0, 2.0]
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        // Doc 1: [3.0, 4.0]
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&4.0f32.to_le_bytes());
        // Doc 2: empty (no chunk_0)
        data.extend_from_slice(&0u32.to_le_bytes());

        // Vector ID "summary"
        let summary_id = b"summary";
        data.extend_from_slice(&(summary_id.len() as u32).to_le_bytes());
        data.extend_from_slice(summary_id);
        // Doc 0: [10.0, 20.0, 30.0]
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&10.0f32.to_le_bytes());
        data.extend_from_slice(&20.0f32.to_le_bytes());
        data.extend_from_slice(&30.0f32.to_le_bytes());
        // Doc 1: empty (no summary)
        data.extend_from_slice(&0u32.to_le_bytes());
        // Doc 2: [40.0, 50.0, 60.0]
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&40.0f32.to_le_bytes());
        data.extend_from_slice(&50.0f32.to_le_bytes());
        data.extend_from_slice(&60.0f32.to_le_bytes());

        data
    }

    #[test]
    fn test_read_columnar_vectors() {
        let data = write_columnar_test_data();
        let reader = VectorReader::open(Cursor::new(data)).unwrap();
        let field = Field::from_field_id(0);

        assert_eq!(reader.num_docs(), 3);

        // Test individual access
        assert_eq!(reader.get(field, "chunk_0", 0), Some(&[1.0f32, 2.0][..]));
        assert_eq!(reader.get(field, "chunk_0", 1), Some(&[3.0f32, 4.0][..]));
        assert_eq!(reader.get(field, "chunk_0", 2), None);

        assert_eq!(
            reader.get(field, "summary", 0),
            Some(&[10.0f32, 20.0, 30.0][..])
        );
        assert_eq!(reader.get(field, "summary", 1), None);
        assert_eq!(
            reader.get(field, "summary", 2),
            Some(&[40.0f32, 50.0, 60.0][..])
        );

        // Test columnar iteration (primary access pattern)
        let chunk_vecs: Vec<_> = reader.iter_vectors(field, "chunk_0").collect();
        assert_eq!(chunk_vecs.len(), 2);
        assert_eq!(chunk_vecs[0], (0, &[1.0f32, 2.0][..]));
        assert_eq!(chunk_vecs[1], (1, &[3.0f32, 4.0][..]));

        let summary_vecs: Vec<_> = reader.iter_vectors(field, "summary").collect();
        assert_eq!(summary_vecs.len(), 2);
        assert_eq!(summary_vecs[0], (0, &[10.0f32, 20.0, 30.0][..]));
        assert_eq!(summary_vecs[1], (2, &[40.0f32, 50.0, 60.0][..]));

        // Test get all vectors for a document
        let doc0_vecs = reader.get_doc_vectors(field, 0);
        assert_eq!(doc0_vecs.len(), 2);
        assert_eq!(doc0_vecs.get("chunk_0"), Some(&&[1.0f32, 2.0][..]));
        assert_eq!(doc0_vecs.get("summary"), Some(&&[10.0f32, 20.0, 30.0][..]));

        // Test vector IDs
        let ids: Vec<_> = reader.vector_ids(field).unwrap().collect();
        assert_eq!(ids, vec!["chunk_0", "summary"]);
    }

    #[test]
    fn test_empty_reader() {
        let reader = VectorReader::empty();
        let field = Field::from_field_id(0);
        assert_eq!(reader.num_docs(), 0);
        assert_eq!(reader.get(field, "any", 0), None);
    }
}

