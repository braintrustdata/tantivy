use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tantivy::directory::RamDirectory;
use tantivy::indexer::merge_indices;
use tantivy::schema::{Schema, TantivyDocument, TEXT};
use tantivy::{Index, IndexWriter};

fn merge_scratch(c: &mut Criterion) {
    let mut schema = Schema::builder();
    schema.add_text_field("url", TEXT);
    schema.add_text_field("title", TEXT);
    schema.add_text_field("body", TEXT);
    let schema = schema.build();
    let documents: Vec<_> = include_str!("wiki.json")
        .lines()
        .map(|line| TantivyDocument::parse_json(&schema, line).unwrap())
        .collect();
    let indices: Vec<_> = documents
        .chunks(documents.len().div_ceil(4))
        .map(|documents| {
            let index = Index::create_in_ram(schema.clone());
            let mut writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000).unwrap();
            for document in documents {
                writer.add_document(document.clone()).unwrap();
            }
            writer.commit().unwrap();
            writer.wait_merging_threads().unwrap();
            assert_eq!(index.searchable_segment_ids().unwrap().len(), 1);
            index
        })
        .collect();
    let merged = merge_indices(&indices, RamDirectory::default()).unwrap();
    let reader = merged.reader().unwrap();
    assert_eq!(reader.searcher().num_docs(), documents.len() as u64);
    assert_eq!(merged.searchable_segment_ids().unwrap().len(), 1);
    drop(reader);
    drop(merged);

    let mut group = c.benchmark_group("merge-scratch");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(documents.len() as u64));
    group.bench_function("wiki-four-segments", |b| {
        b.iter(|| black_box(merge_indices(&indices, RamDirectory::default()).unwrap()));
    });
    group.finish();
}

criterion_group!(benches, merge_scratch);
criterion_main!(benches);
