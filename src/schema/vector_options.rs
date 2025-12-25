use serde::{Deserialize, Serialize};

/// Define how a vector field should be handled by tantivy.
///
/// Vector fields store embedding vectors (arrays of f32) per document.
/// They are always stored in a separate `.vec` file per segment.
///
/// Note: Vectors are not indexed for search - they are stored for retrieval
/// and can be used for vector similarity search via external libraries
/// like FAISS or Usearch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VectorOptions {
    // Marker type - no configuration needed for now.
    // Future options might include:
    // - stored: bool (whether to include in doc store)
    // - normalize: bool (whether to L2 normalize vectors)
}

impl VectorOptions {
    /// Creates a new `VectorOptions` with default settings.
    pub fn new() -> Self {
        Self::default()
    }
}

impl From<()> for VectorOptions {
    fn from(_: ()) -> VectorOptions {
        VectorOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_options_serialization() {
        let options = VectorOptions::default();
        let json = serde_json::to_string(&options).unwrap();
        assert_eq!(json, "{}");

        let deserialized: VectorOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(deserialized, options);
    }
}

