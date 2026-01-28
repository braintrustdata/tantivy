use std::collections::HashMap;
use std::io;
use std::ops::Range;

use half::f16 as half_f16;
use once_cell::sync::OnceCell;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use common::{HasLen, OwnedBytes, TerminatingWrite};

use crate::directory::FileSlice;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::schema::Field;
use crate::vector::format::VectorEncoding;
use crate::DocId;

pub const VECTOR_ANN_MAGIC: u32 = 0x4E415654; // "TVAN"
pub const VECTOR_ANN_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VectorAnnMetric {
    Cos = 0,
}

impl VectorAnnMetric {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(VectorAnnMetric::Cos),
            _ => None,
        }
    }
}

struct VectorAnnIndex {
    index: Index,
    _backing: OwnedBytes,
}

struct VectorAnnIndexMeta {
    dimensions: u32,
    index_range: Range<usize>,
    index: OnceCell<VectorAnnIndex>,
}

impl VectorAnnIndexMeta {
    fn new(dimensions: u32, index_range: Range<usize>) -> Self {
        Self {
            dimensions,
            index_range,
            index: OnceCell::new(),
        }
    }

    fn load_index(
        &self,
        file: &FileSlice,
        metric: VectorAnnMetric,
    ) -> io::Result<&VectorAnnIndex> {
        self.index.get_or_try_init(|| {
            let bytes = file.read_bytes_slice(self.index_range.clone())?;
            let index = build_usearch_view(self.dimensions as usize, metric, bytes.as_slice())?;
            Ok(VectorAnnIndex {
                index,
                _backing: bytes,
            })
        })
    }
}

pub struct VectorAnnReader {
    file: FileSlice,
    fields: HashMap<u32, HashMap<String, VectorAnnIndexMeta>>,
    encoding: VectorEncoding,
    metric: VectorAnnMetric,
}

impl VectorAnnReader {
    pub fn open(file: FileSlice) -> io::Result<Self> {
        let mut cursor = FileSliceCursor::new(file.clone());
        let magic = cursor.read_u32()?;
        if magic != VECTOR_ANN_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid vector ANN magic: expected {:x}, got {:x}",
                    VECTOR_ANN_MAGIC, magic
                ),
            ));
        }

        let version = cursor.read_u8()?;
        if version != VECTOR_ANN_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported vector ANN version: {}", version),
            ));
        }

        let encoding = VectorEncoding::from_u8(cursor.read_u8()?).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Unknown vector ANN encoding")
        })?;
        let metric = VectorAnnMetric::from_u8(cursor.read_u8()?).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Unknown vector ANN metric")
        })?;
        cursor.read_u8()?; // reserved

        let num_fields = cursor.read_u32()? as usize;
        let mut field_ids = Vec::with_capacity(num_fields);
        for _ in 0..num_fields {
            field_ids.push(cursor.read_u32()?);
        }

        let mut fields = HashMap::new();
        for field_id in field_ids {
            let num_vector_ids = cursor.read_u32()? as usize;
            let mut vector_map = HashMap::new();
            for _ in 0..num_vector_ids {
                let id_len = cursor.read_u32()? as usize;
                let id_bytes = cursor.read_bytes(id_len)?;
                let vector_id = String::from_utf8(id_bytes.to_vec()).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid vector id: {}", e))
                })?;
                let dimensions = cursor.read_u32()?;
                let index_len = cursor.read_u64()? as usize;
                let index_start = cursor.offset();
                cursor.skip(index_len)?;
                let index_range = index_start..(index_start + index_len);
                vector_map.insert(vector_id, VectorAnnIndexMeta::new(dimensions, index_range));
            }
            fields.insert(field_id, vector_map);
        }

        Ok(VectorAnnReader {
            file,
            fields,
            encoding,
            metric,
        })
    }

    pub fn search(
        &self,
        field: Field,
        vector_id: &str,
        query: &[f32],
        limit: usize,
        min_similarity: Option<f32>,
    ) -> io::Result<Vec<(DocId, f32)>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let Some(field_map) = self.fields.get(&field.field_id()) else {
            return Ok(Vec::new());
        };
        let Some(meta) = field_map.get(vector_id) else {
            return Ok(Vec::new());
        };
        if meta.dimensions as usize != query.len() {
            return Ok(Vec::new());
        }
        if self.encoding != VectorEncoding::F16 {
            return Ok(Vec::new());
        }

        let index = meta.load_index(&self.file, self.metric)?;
        let query_bits = f32_to_f16_bits(query);
        let query_f16 = usearch::f16::from_i16s(&query_bits);
        let matches = index
            .index
            .search(query_f16, limit)
            .map_err(usearch_err)?;

        let mut results = Vec::with_capacity(matches.keys.len());
        for (key, distance) in matches.keys.iter().zip(matches.distances.iter()) {
            if *key > u32::MAX as u64 {
                continue;
            }
            let similarity = match self.metric {
                VectorAnnMetric::Cos => 1.0 - distance,
            };
            if let Some(min) = min_similarity {
                if similarity < min {
                    continue;
                }
            }
            results.push((*key as u32, similarity));
        }
        results.sort_by_key(|(doc_id, _)| *doc_id);
        Ok(results)
    }
}

pub struct VectorAnnWriter;

impl VectorAnnWriter {
    pub fn serialize_from_writer<W: TerminatingWrite>(
        writer: &crate::vector::writer::VectorFieldsWriter,
        mut out: W,
        doc_id_map: Option<&DocIdMapping>,
    ) -> io::Result<()> {
        out.write_all(&VECTOR_ANN_MAGIC.to_le_bytes())?;
        out.write_all(&[VECTOR_ANN_VERSION])?;
        out.write_all(&[VectorEncoding::F16 as u8])?;
        out.write_all(&[VectorAnnMetric::Cos as u8])?;
        out.write_all(&[0u8])?;

        let vector_fields = writer.vector_fields();
        out.write_all(&(vector_fields.len() as u32).to_le_bytes())?;
        for field in vector_fields {
            out.write_all(&field.field_id().to_le_bytes())?;
        }

        for field in vector_fields {
            let vector_ids = writer.vector_ids_for_field(*field).unwrap_or_default();
            let mut entries: Vec<(String, u32, Vec<(DocId, &[f32])>)> = Vec::new();
            for vector_id in vector_ids {
                let Some(doc_vectors) = writer.get_vectors(*field, &vector_id) else {
                    continue;
                };
                let Some(dimensions) = doc_vectors.values().next().map(|v| v.len() as u32) else {
                    continue;
                };
                let mut ordered: Vec<(DocId, &[f32])> = Vec::new();
                if let Some(mapping) = doc_id_map {
                    for (new_doc_id, old_doc_id) in mapping.iter_old_doc_ids().enumerate() {
                        if let Some(vec) = doc_vectors.get(&old_doc_id) {
                            ordered.push((new_doc_id as u32, vec.as_slice()));
                        }
                    }
                } else {
                    let mut entries: Vec<_> = doc_vectors.iter().collect();
                    entries.sort_by_key(|(doc_id, _)| *doc_id);
                    for (doc_id, vec) in entries {
                        ordered.push((*doc_id, vec.as_slice()));
                    }
                }
                if !ordered.is_empty() && dimensions > 0 {
                    entries.push((vector_id, dimensions, ordered));
                }
            }

            out.write_all(&(entries.len() as u32).to_le_bytes())?;
            for (vector_id, dimensions, ordered) in entries {
                let index_bytes = build_ann_index_bytes(dimensions as usize, ordered)?;
                let id_bytes = vector_id.as_bytes();
                out.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
                out.write_all(id_bytes)?;
                out.write_all(&dimensions.to_le_bytes())?;
                out.write_all(&(index_bytes.len() as u64).to_le_bytes())?;
                out.write_all(&index_bytes)?;
            }
        }

        out.terminate()
    }

    pub fn serialize_from_merged<W: TerminatingWrite>(
        mut out: W,
        vector_fields: &[Field],
        vectors: Vec<(Field, String, u32, Vec<(DocId, Vec<f32>)>)>,
    ) -> io::Result<()> {
        out.write_all(&VECTOR_ANN_MAGIC.to_le_bytes())?;
        out.write_all(&[VECTOR_ANN_VERSION])?;
        out.write_all(&[VectorEncoding::F16 as u8])?;
        out.write_all(&[VectorAnnMetric::Cos as u8])?;
        out.write_all(&[0u8])?;

        out.write_all(&(vector_fields.len() as u32).to_le_bytes())?;
        for field in vector_fields {
            out.write_all(&field.field_id().to_le_bytes())?;
        }

        let mut by_field: HashMap<u32, Vec<(String, u32, Vec<(DocId, Vec<f32>)>)>> =
            HashMap::new();
        for (field, vector_id, dimensions, vectors) in vectors {
            by_field
                .entry(field.field_id())
                .or_default()
                .push((vector_id, dimensions, vectors));
        }

        for field in vector_fields {
            let mut entries = by_field.remove(&field.field_id()).unwrap_or_default();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            out.write_all(&(entries.len() as u32).to_le_bytes())?;
            for (vector_id, dimensions, vectors) in entries {
                let index_bytes = build_ann_index_bytes(
                    dimensions as usize,
                    vectors.iter().map(|(doc_id, vec)| (*doc_id, vec.as_slice())),
                )?;
                let id_bytes = vector_id.as_bytes();
                out.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
                out.write_all(id_bytes)?;
                out.write_all(&dimensions.to_le_bytes())?;
                out.write_all(&(index_bytes.len() as u64).to_le_bytes())?;
                out.write_all(&index_bytes)?;
            }
        }

        out.terminate()
    }
}

fn build_usearch_view(
    dimensions: usize,
    metric: VectorAnnMetric,
    buffer: &[u8],
) -> io::Result<Index> {
    let mut options = IndexOptions::default();
    options.dimensions = dimensions;
    options.metric = match metric {
        VectorAnnMetric::Cos => MetricKind::Cos,
    };
    options.quantization = ScalarKind::F16;
    let index = Index::new(&options).map_err(usearch_err)?;
    unsafe { index.view_from_buffer(buffer).map_err(usearch_err)? };
    Ok(index)
}

fn build_ann_index_bytes<'a, I>(dimensions: usize, vectors: I) -> io::Result<Vec<u8>>
where
    I: IntoIterator<Item = (DocId, &'a [f32])>,
{
    let vectors: Vec<(DocId, &'a [f32])> = vectors.into_iter().collect();
    let mut options = IndexOptions::default();
    options.dimensions = dimensions;
    options.metric = MetricKind::Cos;
    options.quantization = ScalarKind::F16;
    let index = Index::new(&options).map_err(usearch_err)?;
    index
        .reserve(vectors.len())
        .map_err(usearch_err)?;
    for (doc_id, vec) in vectors {
        let bits = f32_to_f16_bits(vec);
        let f16_slice = usearch::f16::from_i16s(&bits);
        index.add(doc_id as u64, f16_slice).map_err(usearch_err)?;
    }
    let mut buffer = vec![0u8; index.serialized_length()];
    index.save_to_buffer(&mut buffer).map_err(usearch_err)?;
    Ok(buffer)
}

fn f32_to_f16_bits(vec: &[f32]) -> Vec<i16> {
    vec.iter()
        .map(|&v| {
            let bits = half_f16::from_f32(v).to_bits();
            i16::from_le_bytes(bits.to_le_bytes())
        })
        .collect()
}

fn usearch_err<E: std::fmt::Display>(err: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("usearch error: {}", err))
}

struct FileSliceCursor {
    file: FileSlice,
    offset: usize,
    len: usize,
}

impl FileSliceCursor {
    fn new(file: FileSlice) -> Self {
        let len = file.len();
        Self { file, offset: 0, len }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn read_exact(&mut self, len: usize) -> io::Result<OwnedBytes> {
        let start = self.offset;
        let end = start + len;
        if end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected EOF reading vector ANN file",
            ));
        }
        let bytes = self.file.read_bytes_slice(start..end)?;
        self.offset = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        Ok(self.read_exact(1)?.as_slice()[0])
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes(bytes.as_slice().try_into().unwrap()))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes(bytes.as_slice().try_into().unwrap()))
    }

    fn read_bytes(&mut self, len: usize) -> io::Result<OwnedBytes> {
        self.read_exact(len)
    }

    fn skip(&mut self, len: usize) -> io::Result<()> {
        let next = self.offset + len;
        if next > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected EOF skipping vector ANN file",
            ));
        }
        self.offset = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directory::Directory;
    use crate::directory::RamDirectory;
    use crate::schema::{SchemaBuilder, STORED, TEXT};
    use crate::{Index, TantivyDocument};

    #[test]
    fn test_vector_ann_roundtrip() {
        let mut schema_builder = SchemaBuilder::new();
        let title = schema_builder.add_text_field("title", TEXT | STORED);
        let embedding = schema_builder.add_vector_map_field("vectors", ());
        let schema = schema_builder.build();

        let directory = RamDirectory::create();
        let index = Index::create(directory, schema.clone(), Default::default()).unwrap();
        let mut writer = index.writer::<TantivyDocument>(50_000_000).unwrap();

        let mut doc1 = TantivyDocument::new();
        doc1.add_text(title, "doc1");
        doc1.add_named_vector(embedding, "embedding", vec![0.1, 0.2, 0.3]);
        writer.add_document(doc1).unwrap();

        let mut doc2 = TantivyDocument::new();
        doc2.add_text(title, "doc2");
        doc2.add_named_vector(embedding, "embedding", vec![0.2, 0.1, 0.0]);
        writer.add_document(doc2).unwrap();

        writer.commit().unwrap();
        drop(writer);

        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0);
        let ann_reader = segment_reader.vector_ann_reader().unwrap();
        let results = ann_reader
            .search(embedding, "embedding", &[0.15, 0.15, 0.15], 2, None)
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|(doc_id, _)| *doc_id <= 1));
    }

    #[test]
    fn test_vector_ann_range_reads() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        #[derive(Debug)]
        struct CountingFileHandle {
            inner: Arc<dyn crate::directory::FileHandle>,
            bytes: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl crate::directory::FileHandle for CountingFileHandle {
            fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
                self.bytes.fetch_add(range.len(), Ordering::Relaxed);
                self.inner.read_bytes(range)
            }
        }

        impl HasLen for CountingFileHandle {
            fn len(&self) -> usize {
                self.inner.len()
            }
        }

        let mut schema_builder = SchemaBuilder::new();
        let embedding = schema_builder.add_vector_map_field("vectors", ());
        let schema = schema_builder.build();

        let directory = RamDirectory::create();
        let index = Index::create(directory.clone(), schema.clone(), Default::default()).unwrap();
        let mut writer = index.writer::<TantivyDocument>(50_000_000).unwrap();
        for i in 0..100 {
            let mut doc = TantivyDocument::new();
            doc.add_named_vector(embedding, "embedding", vec![i as f32, 0.0, 1.0]);
            doc.add_named_vector(embedding, "embedding_alt", vec![0.0, i as f32, 1.0]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        drop(writer);

        let segment = index.searchable_segments().unwrap().pop().unwrap();
        let ann_path = segment.relative_path(crate::SegmentComponent::VectorAnn);
        let handle = directory.get_file_handle(&ann_path).unwrap();
        let bytes_read = Arc::new(AtomicUsize::new(0));
        let counting = CountingFileHandle {
            inner: handle,
            bytes: bytes_read.clone(),
        };
        let file_slice = FileSlice::new(Arc::new(counting));
        let ann_reader = VectorAnnReader::open(file_slice).unwrap();
        let results = ann_reader
            .search(embedding, "embedding", &[1.0, 0.0, 1.0], 5, None)
            .unwrap();
        assert!(!results.is_empty());

        let file_len = ann_reader.file.len();
        let read_len = bytes_read.load(Ordering::Relaxed);
        assert!(read_len < file_len, "expected range reads only");
    }

    #[test]
    #[ignore]
    fn bench_vector_ann_size_estimate() {
        let dims = 768;
        let count = 100_000;

        let mut options = IndexOptions::default();
        options.dimensions = dims;
        options.metric = MetricKind::Cos;
        options.quantization = ScalarKind::F16;

        let index = usearch::Index::new(&options).unwrap();
        index.reserve(count).unwrap();

        let mut vector = vec![0.0f32; dims];
        for doc_id in 0..count {
            for (idx, value) in vector.iter_mut().enumerate() {
                let raw = ((doc_id + idx) % 1000) as f32 / 1000.0;
                *value = raw;
            }
            let bits = f32_to_f16_bits(&vector);
            let f16 = usearch::f16::from_i16s(&bits);
            index.add(doc_id as u64, f16).unwrap();
        }

        let bytes = index.serialized_length();
        let mib = bytes as f64 / (1024.0 * 1024.0);
        println!(
            "ANN size ({} vectors x {} dims f16): {} bytes ({:.2} MiB)",
            count, dims, bytes, mib
        );
    }

    #[test]
    #[ignore]
    fn bench_vector_ann_range_reads_fs() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        use crate::directory::MmapDirectory;
        use tempfile::tempdir;

        #[derive(Debug)]
        struct CountingFileHandle {
            inner: Arc<dyn crate::directory::FileHandle>,
            bytes: Arc<AtomicUsize>,
        }

        #[async_trait::async_trait]
        impl crate::directory::FileHandle for CountingFileHandle {
            fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
                self.bytes.fetch_add(range.len(), Ordering::Relaxed);
                self.inner.read_bytes(range)
            }
        }

        impl HasLen for CountingFileHandle {
            fn len(&self) -> usize {
                self.inner.len()
            }
        }

        let tmp = tempdir().unwrap();
        let directory = MmapDirectory::open(tmp.path()).unwrap();

        let mut schema_builder = SchemaBuilder::new();
        let embedding = schema_builder.add_vector_map_field("vectors", ());
        let schema = schema_builder.build();

        let index = Index::create(directory.clone(), schema.clone(), Default::default()).unwrap();
        let mut writer = index.writer::<TantivyDocument>(50_000_000).unwrap();
        for i in 0..2_000 {
            let mut doc = TantivyDocument::new();
            doc.add_named_vector(embedding, "embedding", vec![i as f32, 0.0, 1.0]);
            doc.add_named_vector(embedding, "embedding_alt", vec![0.0, i as f32, 1.0]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        drop(writer);

        let segment = index.searchable_segments().unwrap().pop().unwrap();
        let ann_path = segment.relative_path(crate::SegmentComponent::VectorAnn);
        let handle = directory.get_file_handle(&ann_path).unwrap();
        let bytes_read = Arc::new(AtomicUsize::new(0));
        let counting = CountingFileHandle {
            inner: handle,
            bytes: bytes_read.clone(),
        };
        let file_slice = FileSlice::new(Arc::new(counting));
        let ann_reader = VectorAnnReader::open(file_slice).unwrap();
        let results = ann_reader
            .search(embedding, "embedding", &[1.0, 0.0, 1.0], 5, None)
            .unwrap();
        assert!(!results.is_empty());

        let file_len = ann_reader.file.len();
        let read_len = bytes_read.load(Ordering::Relaxed);
        println!(
            "ANN range read bytes: {} of {} ({:.2}%)",
            read_len,
            file_len,
            (read_len as f64 / file_len as f64) * 100.0
        );
        assert!(read_len < file_len, "expected range reads only");
    }
}
