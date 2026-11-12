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
    /// Per-document artifact payloads.
    pub artifacts: Vec<DocumentArtifact>,
    /// Document written to the doc store instead of `document`, if set. Indexing and fast fields
    /// still come from `document`.
    pub stored_document: Option<D>,
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
    /// Add operation that writes `stored_document` to the doc store instead of `document`.
    AddWithStoredDocument {
        /// Document to index.
        document: D,
        /// Document to write to the doc store.
        stored_document: D,
        /// Per-document artifact payloads.
        artifacts: Vec<DocumentArtifact>,
    },
    /// Delete operation
    Delete(Term),
}
