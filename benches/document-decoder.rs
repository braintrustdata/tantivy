use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tantivy::schema::document::{
    ArrayAccess, BinaryDocumentDeserializer, DeserializeError, DocumentDeserialize,
    DocumentDeserializer, ObjectAccess, ValueDeserializeSeed, ValueDeserializer, ValueVisitor,
};
use tantivy::schema::{Schema, STORED};
use tantivy::{DateTime, Index, TantivyDocument};

#[derive(Clone, Copy)]
struct StringBytes;

impl ValueDeserializeSeed for StringBytes {
    type Value = usize;

    fn deserialize<'de, D: ValueDeserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<usize, DeserializeError> {
        deserializer.deserialize_any(self)
    }
}

impl ValueVisitor for StringBytes {
    type Value = usize;

    fn visit_borrowed_str(&self, value: &str) -> Result<usize, DeserializeError> {
        Ok(value.len())
    }

    fn visit_string(&self, value: String) -> Result<usize, DeserializeError> {
        Ok(value.len())
    }

    fn visit_null(&self) -> Result<usize, DeserializeError> {
        Ok(0)
    }

    fn visit_u64(&self, _value: u64) -> Result<usize, DeserializeError> {
        Ok(0)
    }

    fn visit_i64(&self, _value: i64) -> Result<usize, DeserializeError> {
        Ok(0)
    }

    fn visit_f64(&self, _value: f64) -> Result<usize, DeserializeError> {
        Ok(0)
    }

    fn visit_bool(&self, _value: bool) -> Result<usize, DeserializeError> {
        Ok(0)
    }

    fn visit_datetime(&self, _value: DateTime) -> Result<usize, DeserializeError> {
        Ok(0)
    }

    fn visit_array<'de, A: ArrayAccess<'de>>(
        &self,
        mut access: A,
    ) -> Result<usize, DeserializeError> {
        let mut bytes = access.size_hint();
        while let Some(value) = access.next_element_seed(*self)? {
            bytes += value;
        }
        Ok(bytes)
    }

    fn visit_object<'de, A: ObjectAccess<'de>>(
        &self,
        mut access: A,
    ) -> Result<usize, DeserializeError> {
        let mut bytes = access.size_hint();
        while let Some((key, value)) = access.next_entry_seed(*self, *self)? {
            bytes += key + value;
        }
        Ok(bytes)
    }
}

struct DocumentStringBytes(usize);

impl DocumentDeserialize for DocumentStringBytes {
    fn deserialize<'de, D: DocumentDeserializer<'de>>(
        mut deserializer: D,
    ) -> Result<Self, DeserializeError> {
        let mut bytes = deserializer.size_hint();
        while let Some((_field, value)) = deserializer.next_field_seed(StringBytes)? {
            bytes += value;
        }
        Ok(Self(bytes))
    }
}

fn benchmark(c: &mut Criterion) {
    for (name, corpus) in [
        ("hdfs", include_str!("hdfs.json")),
        ("github", include_str!("gh.json")),
        ("wiki", include_str!("wiki.json")),
    ] {
        let mut schema = Schema::builder();
        let field = schema.add_json_field("payload", STORED);
        let schema = schema.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_with_num_threads(1, 15_000_000).unwrap();
        let documents: Vec<_> = corpus
            .lines()
            .take(64)
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                let mut document = TantivyDocument::new();
                document.add_field_value(field, value);
                writer.add_document(document.clone()).unwrap();
                document
            })
            .collect();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let store = searcher.segment_reader(0).get_store_reader(64).unwrap();
        let encoded: Vec<_> = (0..documents.len())
            .map(|doc_id| store.get_document_bytes(doc_id as u32).unwrap())
            .collect();
        for (doc_id, (bytes, expected)) in encoded.iter().zip(&documents).enumerate() {
            let mut input: &[u8] = bytes;
            let document = TantivyDocument::deserialize(
                BinaryDocumentDeserializer::from_slice(&mut input).unwrap(),
            )
            .unwrap();
            assert_eq!(&document, expected);
            assert!(input.is_empty());
            let mut input: &[u8] = bytes;
            let count = DocumentStringBytes::deserialize(
                BinaryDocumentDeserializer::from_slice(&mut input).unwrap(),
            )
            .unwrap();
            assert_eq!(
                count.0,
                store.get::<DocumentStringBytes>(doc_id as u32).unwrap().0
            );
            assert!(input.is_empty());
        }
        let mut input: &[u8] = &encoded[0];
        let deserializer = BinaryDocumentDeserializer::from_slice(&mut input).unwrap();
        eprintln!(
            "{name}: decoder size {} bytes",
            std::mem::size_of_val(&deserializer)
        );
        let bytes: usize = encoded.iter().map(|bytes| bytes.len()).sum();
        eprintln!(
            "{name}: {} documents, {bytes} stored bytes",
            documents.len()
        );
        let mut group = c.benchmark_group(name);
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function("borrowed_visit", |b| {
            b.iter(|| {
                for bytes in &encoded {
                    let mut input: &[u8] = black_box(bytes);
                    black_box(
                        DocumentStringBytes::deserialize(
                            BinaryDocumentDeserializer::from_slice(&mut input).unwrap(),
                        )
                        .unwrap()
                        .0,
                    );
                }
            });
        });
        group.bench_function("owned_slice", |b| {
            b.iter(|| {
                for bytes in &encoded {
                    let mut input: &[u8] = black_box(bytes);
                    black_box(
                        TantivyDocument::deserialize(
                            BinaryDocumentDeserializer::from_slice(&mut input).unwrap(),
                        )
                        .unwrap(),
                    );
                }
            });
        });
        group.bench_function("owned_store_get", |b| {
            b.iter(|| {
                for doc_id in 0..documents.len() {
                    black_box(
                        store
                            .get::<TantivyDocument>(black_box(doc_id as u32))
                            .unwrap(),
                    );
                }
            });
        });
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = benchmark
}
criterion_main!(benches);
