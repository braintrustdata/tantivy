use crate::artifact::DocumentArtifact;
use crate::query::Weight;
use crate::schema::document::Document;
use crate::schema::{TantivyDocument, Term};
use crate::Opstamp;

/// Timestamped Delete operation.
pub struct DeleteOperation {
    pub opstamp: Opstamp,
    pub target: Box<dyn Weight>,
}

/// Timestamped Add operation.
#[derive(Eq, PartialEq, Debug)]
pub struct AddOperation<D: Document = TantivyDocument> {
    /// Operation timestamp assigned by the writer.
    pub opstamp: Opstamp,
    /// Document to add.
    pub document: D,
    /// Optional document to write to the doc store instead of `document`.
    pub stored_document: Option<D>,
    /// Per-document artifact payloads.
    pub artifacts: Vec<DocumentArtifact>,
    /// Per-document doc store extension payloads.
    pub store_extensions: Vec<DocumentArtifact>,
}

/// UserOperation is an enum type that encapsulates other operation types.
#[derive(Eq, PartialEq, Debug)]
pub enum UserOperation<D: Document = TantivyDocument> {
    /// Add operation
    Add(D),
    /// Add operation with segment artifacts
    AddWithArtifacts {
        /// Document to add.
        document: D,
        /// Per-document artifact payloads.
        artifacts: Vec<DocumentArtifact>,
    },
    /// Add operation with separate stored document and store extension payloads.
    AddWithStoreExtensions {
        /// Document to index.
        document: D,
        /// Optional document to store.
        stored_document: Option<D>,
        /// Per-document artifact payloads.
        artifacts: Vec<DocumentArtifact>,
        /// Per-document doc store extension payloads.
        store_extensions: Vec<DocumentArtifact>,
    },
    /// Delete operation
    Delete(Term),
}
