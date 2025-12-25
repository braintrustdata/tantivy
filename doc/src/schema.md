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
- **Vector**: Embedding vectors (arrays of f32)

## Vector Fields

Vector fields allow you to store embedding vectors alongside your documents. This is useful for semantic search, recommendation systems, and other ML-powered features.

### Creating a Vector Field

```rust
use tantivy::schema::SchemaBuilder;

let mut schema_builder = SchemaBuilder::new();
let embedding = schema_builder.add_vector_field("embedding", ());
let schema = schema_builder.build();
```

### Adding Vectors to Documents

```rust
use tantivy::TantivyDocument;

let mut doc = TantivyDocument::new();
doc.add_field_value(embedding, vec![0.1f32, 0.2, 0.3, 0.4]);
```

### Key Features

- **Variable dimensions**: Each document can have vectors of different sizes
- **Optional**: Documents don't need to have a vector for every vector field
- **Segment-integrated**: Vector files are managed alongside other segment files during merges and garbage collection

### Reading Vectors

```rust
// After committing and creating a reader
let reader = index.reader()?;
let searcher = reader.searcher();

for segment_reader in searcher.segment_readers() {
    if let Some(vector_reader) = segment_reader.vector_reader()? {
        let vec = vector_reader.get(embedding, doc_id);
        // vec is Option<&[f32]>
    }
}
```
