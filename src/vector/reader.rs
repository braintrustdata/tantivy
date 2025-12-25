//! Vector reader for reading embedding vectors from segment files.
//!
//! The `VectorReader` provides access to vectors stored in a segment's `.vec` file.
//! Vectors are loaded into memory on open for fast random access by document ID.
//!
//! # Example
//!
//! ```rust,ignore
//! let segment_reader = searcher.segment_reader(0);
//! if let Some(vector_reader) = segment_reader.vector_reader()?.as_ref() {
//!     // Get vector for document 0
//!     let vec: Option<&[f32]> = vector_reader.get(embedding_field, 0);
//! }
//! ```

use std::collections::HashMap;
use std::io::{self, Read};

use crate::schema::Field;

/// Reader for vector data stored in binary format.
///
/// Binary format:
/// - Header: num_fields (u32), field_ids (u32 each), num_docs (u32)
/// - For each field, for each doc: vector_len (u32), vector_data (f32 each)
///
/// Vectors are loaded into memory on open for fast random access by doc_id.
pub struct VectorReader {
    /// Map from field_id to vectors by doc_id
    field_vectors: HashMap<u32, Vec<Vec<f32>>>,
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
        
        // Read vectors for each field
        let mut field_vectors = HashMap::new();
        for field_id in field_ids {
            let mut vectors = Vec::with_capacity(num_docs as usize);
            for _ in 0..num_docs {
                let vec_len = Self::read_u32(&mut reader)?;
                let mut vec = Vec::with_capacity(vec_len as usize);
                for _ in 0..vec_len {
                    vec.push(Self::read_f32(&mut reader)?);
                }
                vectors.push(vec);
            }
            field_vectors.insert(field_id, vectors);
        }

        Ok(VectorReader { field_vectors, num_docs })
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

    /// Gets the vector for a given field and doc_id.
    ///
    /// Returns `None` if field or doc_id is not found.
    /// Returns `Some(&[])` if the document has no vector (empty).
    pub fn get(&self, field: Field, doc_id: u32) -> Option<&[f32]> {
        self.field_vectors
            .get(&field.field_id())
            .and_then(|vectors| vectors.get(doc_id as usize))
            .map(|v| v.as_slice())
    }

    /// Gets all vectors for a given field.
    pub fn get_field_vectors(&self, field: Field) -> Option<&Vec<Vec<f32>>> {
        self.field_vectors.get(&field.field_id())
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

    fn write_test_data() -> Vec<u8> {
        let mut data = Vec::new();
        // Header: 1 field
        data.extend_from_slice(&1u32.to_le_bytes());
        // Field ID: 0
        data.extend_from_slice(&0u32.to_le_bytes());
        // Num docs: 3
        data.extend_from_slice(&3u32.to_le_bytes());
        
        // Doc 0: [1.0, 2.0, 3.0]
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        
        // Doc 1: [] (empty)
        data.extend_from_slice(&0u32.to_le_bytes());
        
        // Doc 2: [4.5, 5.5]
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&4.5f32.to_le_bytes());
        data.extend_from_slice(&5.5f32.to_le_bytes());
        
        data
    }

    #[test]
    fn test_read_vectors() {
        let data = write_test_data();
        let reader = VectorReader::open(Cursor::new(data)).unwrap();
        let field = Field::from_field_id(0);

        assert_eq!(reader.num_docs(), 3);
        assert_eq!(reader.get(field, 0), Some(&[1.0f32, 2.0, 3.0][..]));
        assert_eq!(reader.get(field, 1), Some(&[][..]));
        assert_eq!(reader.get(field, 2), Some(&[4.5f32, 5.5][..]));
        assert_eq!(reader.get(field, 3), None);
    }

    #[test]
    fn test_empty_reader() {
        let reader = VectorReader::empty();
        let field = Field::from_field_id(0);
        assert_eq!(reader.num_docs(), 0);
        assert_eq!(reader.get(field, 0), None);
    }
}

