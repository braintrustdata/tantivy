use serde::{Deserialize, Serialize};

/// Define how a VectorMap field should be handled by tantivy.
///
/// VectorMap fields store a map of named embedding vectors (String -> Vec<f32>) per document.
/// They are always stored in a separate `.vec` file per segment.
///
/// Note: Vectors are not indexed for search - they are stored for retrieval
/// and can be used for vector similarity search via external libraries
/// like FAISS or Usearch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VectorMapOptions {
    // Marker type - no configuration needed for now.
    // Future options might include:
    // - stored: bool (whether to include in doc store)
    // - normalize: bool (whether to L2 normalize vectors)
}

impl VectorMapOptions {
    /// Creates a new `VectorMapOptions` with default settings.
    pub fn new() -> Self {
        Self::default()
    }
}

impl From<()> for VectorMapOptions {
    fn from(_: ()) -> VectorMapOptions {
        VectorMapOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_map_options_serialization() {
        let options = VectorMapOptions::default();
        let json = serde_json::to_string(&options).unwrap();
        assert_eq!(json, "{}");

        let deserialized: VectorMapOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(deserialized, options);
    }
}
