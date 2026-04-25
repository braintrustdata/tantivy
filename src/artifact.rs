//! Extension points for segment-owned files written outside Tantivy's core codecs.

use crate::directory::WritePtr;
use crate::index::Segment;
pub use crate::indexer::doc_id_mapping::{DocIdMapping, SegmentDocIdMapping};
use crate::DocId;

/// Per-document artifact payload routed to a registered segment artifact provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocumentArtifact {
    /// Identifier returned by the provider that should receive the payload.
    pub provider_id: String,
    /// Provider-specific bytes for the document.
    pub payload: Vec<u8>,
}

impl DocumentArtifact {
    /// Creates a new document artifact payload for a provider.
    pub fn new(provider_id: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            provider_id: provider_id.into(),
            payload,
        }
    }
}

/// Provider for a segment-owned artifact file.
pub trait SegmentArtifactProvider: Send + Sync {
    /// Stable provider identifier used by document artifact payloads.
    fn id(&self) -> &str;

    /// File extension used for the provider's segment artifact.
    fn file_extension(&self) -> &str;

    /// Creates a writer for a new segment.
    fn make_writer(&self) -> Box<dyn SegmentArtifactWriter>;

    /// Returns whether a merge should ask this provider to produce an output artifact.
    fn has_merge_input(&self, _segments: &[Segment]) -> crate::Result<bool> {
        Ok(true)
    }

    /// Writes this provider's merged artifact for a segment merge.
    fn merge(&self, context: SegmentArtifactMergeContext<'_>) -> crate::Result<()>;
}

/// Writer for one provider's artifact data within a new segment.
pub trait SegmentArtifactWriter: Send {
    /// Records provider-specific bytes for a document.
    fn record(&mut self, doc_id: DocId, payload: &[u8]) -> crate::Result<()>;

    /// Returns whether the writer has any document payloads to serialize.
    fn has_documents(&self) -> bool;

    /// Estimated heap usage for the writer.
    fn mem_usage(&self) -> usize {
        0
    }

    /// Serializes the artifact for a completed segment.
    fn serialize(
        &mut self,
        output: WritePtr,
        doc_id_map: Option<&DocIdMapping>,
        max_doc: DocId,
    ) -> crate::Result<()>;
}

/// Context passed to a provider while merging segment artifacts.
pub struct SegmentArtifactMergeContext<'a> {
    /// Destination writer for the merged artifact.
    pub output: WritePtr,
    /// Source segments participating in the merge.
    pub segments: &'a [Segment],
    /// Mapping from new document ids to source segment document addresses.
    pub doc_id_mapping: &'a SegmentDocIdMapping,
    /// Number of documents in the merged segment.
    pub max_doc: DocId,
}
