//! Vector reader for reading embedding vectors from segment files.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{self, Read};

use super::format::{
    decode_vector, Int8QuantParams, PresenceBitset, VectorEncoding, VECTOR_MAGIC, VECTOR_VERSION,
};
use crate::schema::Field;
use crate::DocId;

/// Data for a single vector column (one vector_id within a field).
#[derive(Debug)]
struct VectorColumn {
    /// Dimensions of vectors in this column
    dimensions: u32,
    /// Which docs have this vector
    presence: PresenceBitset,
    /// Raw encoded vector data (contiguous)
    data: Vec<u8>,
    /// Quantization params (only for Int8 encoding)
    quant_params: Option<Int8QuantParams>,
}

impl VectorColumn {
    /// Get vector for a doc_id, decoding on the fly.
    fn get(&self, doc_id: DocId, encoding: VectorEncoding) -> Option<Cow<'_, [f32]>> {
        if !self.presence.get(doc_id) {
            return None;
        }

        let index = self.presence.count_ones_before(doc_id) as usize;
        let bytes_per_vec = self.dimensions as usize * encoding.bytes_per_dim();
        let start = index * bytes_per_vec;
        let end = start + bytes_per_vec;

        if end > self.data.len() {
            return None;
        }

        let bytes = &self.data[start..end];
        Some(decode_vector(bytes, encoding, self.quant_params.as_ref()))
    }

    /// Iterate over all vectors in this column.
    fn iter<'a>(
        &'a self,
        encoding: VectorEncoding,
    ) -> impl Iterator<Item = (DocId, Cow<'a, [f32]>)> + 'a {
        let bytes_per_vec = self.dimensions as usize * encoding.bytes_per_dim();

        self.presence.iter_ones().enumerate().map(move |(idx, doc_id)| {
            let start = idx * bytes_per_vec;
            let end = start + bytes_per_vec;
            let bytes = &self.data[start..end];
            (doc_id, decode_vector(bytes, encoding, self.quant_params.as_ref()))
        })
    }

    /// Number of vectors in this column.
    fn count(&self) -> u32 {
        self.presence.count_ones()
    }
}

/// Reader for vector data stored in optimized columnar format.
///
/// Vectors are organized by vector ID (string) for efficient batch access:
/// - `field -> vector_id -> VectorColumn`
///
/// This allows efficient retrieval of all vectors with a given ID across documents.
pub struct VectorReader {
    /// Map from field_id to (vector_id -> VectorColumn)
    field_vectors: HashMap<u32, HashMap<String, VectorColumn>>,
    /// Number of documents
    num_docs: u32,
    /// Encoding used in this file
    encoding: VectorEncoding,
}

impl VectorReader {
    /// Opens a VectorReader from a binary reader.
    pub fn open<R: Read>(mut reader: R) -> io::Result<Self> {
        // Read and verify magic number
        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes)?;
        let magic = u32::from_le_bytes(magic_bytes);

        if magic != VECTOR_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid vector file magic: expected {:x}, got {:x}", VECTOR_MAGIC, magic),
            ));
        }

        Self::read_format(reader)
    }

    /// Read the vector format.
    fn read_format<R: Read>(mut reader: R) -> io::Result<Self> {
        // Read version and encoding
        let mut version_byte = [0u8; 1];
        reader.read_exact(&mut version_byte)?;
        let version = version_byte[0];

        if version != VECTOR_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported vector format version: {}", version),
            ));
        }

        let mut encoding_byte = [0u8; 1];
        reader.read_exact(&mut encoding_byte)?;
        let encoding = VectorEncoding::from_u8(encoding_byte[0]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown vector encoding: {}", encoding_byte[0]),
            )
        })?;

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
            let num_vector_ids = Self::read_u32(&mut reader)?;
            let mut vector_id_map = HashMap::new();

            for _ in 0..num_vector_ids {
                // Read vector ID string
                let id_len = Self::read_u32(&mut reader)? as usize;
                let mut id_bytes = vec![0u8; id_len];
                reader.read_exact(&mut id_bytes)?;
                let vector_id = String::from_utf8(id_bytes)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

                // Read dimensions
                let dimensions = Self::read_u32(&mut reader)?;

                // Read presence bitset
                let bitset_len = ((num_docs as usize + 63) / 64) * 8;
                let mut bitset_bytes = vec![0u8; bitset_len];
                reader.read_exact(&mut bitset_bytes)?;
                let presence = PresenceBitset::from_bytes(&bitset_bytes, num_docs);

                // Read quantization params if int8
                let quant_params = if encoding == VectorEncoding::Int8 {
                    let scale = Self::read_f32(&mut reader)?;
                    let zero_point = Self::read_f32(&mut reader)?;
                    Some(Int8QuantParams { scale, zero_point })
                } else {
                    None
                };

                // Read vector data
                let num_vectors = presence.count_ones() as usize;
                let bytes_per_vec = dimensions as usize * encoding.bytes_per_dim();
                let total_bytes = num_vectors * bytes_per_vec;
                let mut data = vec![0u8; total_bytes];
                reader.read_exact(&mut data)?;

                vector_id_map.insert(
                    vector_id,
                    VectorColumn {
                        dimensions,
                        presence,
                        data,
                        quant_params,
                    },
                );
            }

            field_vectors.insert(field_id, vector_id_map);
        }

        Ok(VectorReader {
            field_vectors,
            num_docs,
            encoding,
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
            encoding: VectorEncoding::F32,
        }
    }

    /// Returns the number of documents in this reader.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Returns the encoding used in this file.
    pub fn encoding(&self) -> VectorEncoding {
        self.encoding
    }

    /// Gets the vector for a given field, vector ID, and doc_id.
    ///
    /// Returns `None` if field, vector_id, or doc_id is not found,
    /// or if the document doesn't have this vector ID.
    pub fn get(&self, field: Field, vector_id: &str, doc_id: DocId) -> Option<Cow<'_, [f32]>> {
        self.field_vectors
            .get(&field.field_id())
            .and_then(|vector_id_map| vector_id_map.get(vector_id))
            .and_then(|column| column.get(doc_id, self.encoding))
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
    /// Yields (doc_id, vector) pairs for documents that have this vector.
    /// This is the primary access pattern - vectors are stored contiguously by ID.
    pub fn iter_vectors(
        &self,
        field: Field,
        vector_id: &str,
    ) -> impl Iterator<Item = (DocId, Cow<'_, [f32]>)> {
        let encoding = self.encoding;
        self.field_vectors
            .get(&field.field_id())
            .and_then(|vector_id_map| vector_id_map.get(vector_id))
            .into_iter()
            .flat_map(move |column| column.iter(encoding))
    }

    /// Gets the dimensions of vectors for a given field and vector ID.
    pub fn dimensions(&self, field: Field, vector_id: &str) -> Option<u32> {
        self.field_vectors
            .get(&field.field_id())
            .and_then(|vector_id_map| vector_id_map.get(vector_id))
            .map(|column| column.dimensions)
    }

    /// Gets the number of vectors for a given field and vector ID.
    pub fn count(&self, field: Field, vector_id: &str) -> Option<u32> {
        self.field_vectors
            .get(&field.field_id())
            .and_then(|vector_id_map| vector_id_map.get(vector_id))
            .map(|column| column.count())
    }

    /// Checks if a document has a vector for a given field and vector ID.
    pub fn has_vector(&self, field: Field, vector_id: &str, doc_id: DocId) -> bool {
        self.field_vectors
            .get(&field.field_id())
            .and_then(|vector_id_map| vector_id_map.get(vector_id))
            .map(|column| column.presence.get(doc_id))
            .unwrap_or(false)
    }

    /// Gets all vectors for a document across all vector IDs for a field.
    ///
    /// Returns a map of vector_id -> vector for the given document.
    pub fn get_doc_vectors(&self, field: Field, doc_id: DocId) -> BTreeMap<&str, Cow<'_, [f32]>> {
        let mut result = BTreeMap::new();
        if let Some(vector_id_map) = self.field_vectors.get(&field.field_id()) {
            for (vector_id, column) in vector_id_map {
                if let Some(vec) = column.get(doc_id, self.encoding) {
                    result.insert(vector_id.as_str(), vec);
                }
            }
        }
        result
    }

    /// Batch retrieves vectors for multiple documents and multiple vector IDs.
    ///
    /// More efficient than calling `get()` repeatedly when you know the
    /// doc_ids and vector_ids upfront.
    pub fn get_batch(
        &self,
        field: Field,
        doc_ids: &[DocId],
        vector_ids: &[&str],
    ) -> Vec<BTreeMap<String, Vec<f32>>> {
        let mut results = Vec::with_capacity(doc_ids.len());

        // Single field lookup
        let vector_id_map = match self.field_vectors.get(&field.field_id()) {
            Some(map) => map,
            None => {
                results.resize_with(doc_ids.len(), BTreeMap::new);
                return results;
            }
        };

        // Pre-lookup the columns we need
        let columns: Vec<_> = vector_ids
            .iter()
            .filter_map(|&vid| vector_id_map.get(vid).map(|col| (vid.to_string(), col)))
            .collect();

        for &doc_id in doc_ids {
            let mut doc_result = BTreeMap::new();
            for (vector_id, column) in &columns {
                if let Some(vec) = column.get(doc_id, self.encoding) {
                    doc_result.insert(vector_id.clone(), vec.into_owned());
                }
            }
            results.push(doc_result);
        }

        results
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

        // Header
        data.extend_from_slice(&VECTOR_MAGIC.to_le_bytes());
        data.push(VECTOR_VERSION);
        data.push(VectorEncoding::F32 as u8);

        // 1 field
        data.extend_from_slice(&1u32.to_le_bytes());
        // Field ID: 0
        data.extend_from_slice(&0u32.to_le_bytes());
        // Num docs: 3
        data.extend_from_slice(&3u32.to_le_bytes());

        // Field 0 has 2 vector IDs
        data.extend_from_slice(&2u32.to_le_bytes());

        // Vector ID "chunk_0"
        let chunk_id = b"chunk_0";
        data.extend_from_slice(&(chunk_id.len() as u32).to_le_bytes());
        data.extend_from_slice(chunk_id);
        // Dimensions: 2
        data.extend_from_slice(&2u32.to_le_bytes());
        // Presence bitset: docs 0 and 1 have chunk_0 (bits 0,1 set = 0b11 = 3)
        let mut bitset_bytes = [0u8; 8]; // Padded to u64
        bitset_bytes[0] = 0b00000011;
        data.extend_from_slice(&bitset_bytes);
        // Vectors: [1.0, 2.0], [3.0, 4.0]
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&4.0f32.to_le_bytes());

        // Vector ID "summary"
        let summary_id = b"summary";
        data.extend_from_slice(&(summary_id.len() as u32).to_le_bytes());
        data.extend_from_slice(summary_id);
        // Dimensions: 3
        data.extend_from_slice(&3u32.to_le_bytes());
        // Presence bitset: docs 0 and 2 have summary (bits 0,2 set = 0b101 = 5)
        let mut bitset_bytes = [0u8; 8];
        bitset_bytes[0] = 0b00000101;
        data.extend_from_slice(&bitset_bytes);
        // Vectors: [10.0, 20.0, 30.0], [40.0, 50.0, 60.0]
        data.extend_from_slice(&10.0f32.to_le_bytes());
        data.extend_from_slice(&20.0f32.to_le_bytes());
        data.extend_from_slice(&30.0f32.to_le_bytes());
        data.extend_from_slice(&40.0f32.to_le_bytes());
        data.extend_from_slice(&50.0f32.to_le_bytes());
        data.extend_from_slice(&60.0f32.to_le_bytes());

        data
    }

    #[test]
    fn test_read_vectors() {
        let data = write_test_data();
        let reader = VectorReader::open(Cursor::new(data)).unwrap();
        let field = Field::from_field_id(0);

        assert_eq!(reader.num_docs(), 3);
        assert_eq!(reader.encoding(), VectorEncoding::F32);

        // Test individual access
        assert_eq!(
            reader.get(field, "chunk_0", 0).map(|c| c.into_owned()),
            Some(vec![1.0f32, 2.0])
        );
        assert_eq!(
            reader.get(field, "chunk_0", 1).map(|c| c.into_owned()),
            Some(vec![3.0f32, 4.0])
        );
        assert!(reader.get(field, "chunk_0", 2).is_none());

        assert_eq!(
            reader.get(field, "summary", 0).map(|c| c.into_owned()),
            Some(vec![10.0f32, 20.0, 30.0])
        );
        assert!(reader.get(field, "summary", 1).is_none());
        assert_eq!(
            reader.get(field, "summary", 2).map(|c| c.into_owned()),
            Some(vec![40.0f32, 50.0, 60.0])
        );

        // Test columnar iteration
        let chunk_vecs: Vec<_> = reader
            .iter_vectors(field, "chunk_0")
            .map(|(id, v)| (id, v.into_owned()))
            .collect();
        assert_eq!(chunk_vecs.len(), 2);
        assert_eq!(chunk_vecs[0], (0, vec![1.0f32, 2.0]));
        assert_eq!(chunk_vecs[1], (1, vec![3.0f32, 4.0]));

        // Test metadata
        assert_eq!(reader.dimensions(field, "chunk_0"), Some(2));
        assert_eq!(reader.dimensions(field, "summary"), Some(3));
        assert_eq!(reader.count(field, "chunk_0"), Some(2));
        assert_eq!(reader.count(field, "summary"), Some(2));

        // Test has_vector
        assert!(reader.has_vector(field, "chunk_0", 0));
        assert!(reader.has_vector(field, "chunk_0", 1));
        assert!(!reader.has_vector(field, "chunk_0", 2));
    }

    #[test]
    fn test_empty_reader() {
        let reader = VectorReader::empty();
        let field = Field::from_field_id(0);
        assert_eq!(reader.num_docs(), 0);
        assert!(reader.get(field, "any", 0).is_none());
    }

    #[test]
    fn test_get_batch() {
        let data = write_test_data();
        let reader = VectorReader::open(Cursor::new(data)).unwrap();
        let field = Field::from_field_id(0);

        // Batch get for all docs, both vector IDs
        let results = reader.get_batch(field, &[0, 1, 2], &["chunk_0", "summary"]);
        assert_eq!(results.len(), 3);

        // Doc 0: has both
        assert_eq!(results[0].get("chunk_0"), Some(&vec![1.0f32, 2.0]));
        assert_eq!(results[0].get("summary"), Some(&vec![10.0f32, 20.0, 30.0]));

        // Doc 1: has only chunk_0
        assert_eq!(results[1].get("chunk_0"), Some(&vec![3.0f32, 4.0]));
        assert!(results[1].get("summary").is_none());

        // Doc 2: has only summary
        assert!(results[2].get("chunk_0").is_none());
        assert_eq!(results[2].get("summary"), Some(&vec![40.0f32, 50.0, 60.0]));

        // Batch get with filtered vector IDs
        let results = reader.get_batch(field, &[0, 2], &["summary"]);
        assert_eq!(results.len(), 2);
        assert!(results[0].get("chunk_0").is_none());
        assert_eq!(results[0].get("summary"), Some(&vec![10.0f32, 20.0, 30.0]));
        assert_eq!(results[1].get("summary"), Some(&vec![40.0f32, 50.0, 60.0]));
    }
}
