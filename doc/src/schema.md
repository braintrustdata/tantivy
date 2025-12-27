# Defining your schema

Tantivy requires you to define a schema before creating an index. The schema defines the fields that documents can have and how they should be indexed.

## Field Types

Tantivy supports the following field types:

- **Text**: Full-text searchable content with configurable tokenization
- **Numeric**: `u64`, `i64`, `f64` for numbers and dates
- **Bool**: Boolean values
- **Bytes**: Raw byte arrays
- **IP Address**: IPv4/IPv6 addresses
- **Facet**: Hierarchical categorization
- **JSON**: Dynamic JSON objects
- **VectorMap**: Embedding vectors - maps of string IDs to f32 arrays

## VectorMap Fields

VectorMap fields allow you to store **named embedding vectors** alongside your documents. Each document can have multiple vectors per field, identified by string IDs (e.g., "chunk_0", "chunk_1", "summary"). This is useful for:

- Chunked document embeddings (store vectors for each chunk)
- Multi-representation embeddings (e.g., title vs body vs summary)
- Semantic search and recommendation systems

### Creating a VectorMap Field

```rust
use tantivy::schema::SchemaBuilder;

let mut schema_builder = SchemaBuilder::new();
let embedding = schema_builder.add_vector_map_field("embedding", ());
let schema = schema_builder.build();
```

### Adding Named Vectors to Documents

```rust
use tantivy::TantivyDocument;

let mut doc = TantivyDocument::new();
// Add multiple named vectors
doc.add_named_vector(embedding, "chunk_0", vec![0.1, 0.2, 0.3]);
doc.add_named_vector(embedding, "chunk_1", vec![0.4, 0.5, 0.6]);
doc.add_named_vector(embedding, "summary", vec![1.0, 2.0]);
```

### Key Features

- **Named vectors**: Each document can have multiple vectors identified by string IDs
- **Columnar storage**: Vectors with the same ID are stored contiguously for efficient batch access
- **Variable dimensions**: Each vector can have different sizes
- **Optional**: Documents don't need to have all vector IDs
- **Segment-integrated**: Vector files are managed alongside other segment files during merges and garbage collection

### Reading Vectors (Columnar Access)

The primary access pattern is retrieving all vectors with a given ID across documents:

```rust
let reader = index.reader()?;
let searcher = reader.searcher();

for segment_reader in searcher.segment_readers() {
    if let Some(vector_reader) = segment_reader.vector_reader(embedding) {
        // Primary access pattern: iterate all "chunk_0" vectors
        for (doc_id, vec) in vector_reader.iter_vectors(embedding, "chunk_0") {
            println!("Doc {}: {:?}", doc_id, vec);
        }
        
        // Or get a specific document's vector
        let vec = vector_reader.get(embedding, "summary", doc_id);
        // vec is Option<&[f32]>
        
        // Get all vector IDs for a field
        for id in vector_reader.vector_ids(embedding).unwrap() {
            println!("Vector ID: {}", id);
        }
    }
}
```
