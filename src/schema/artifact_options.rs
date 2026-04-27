use serde::{Deserialize, Serialize};

/// Options for fields backed by external segment artifact providers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ArtifactOptions {}

impl ArtifactOptions {
    /// Creates default artifact options.
    pub fn new() -> Self {
        Self::default()
    }
}

impl From<()> for ArtifactOptions {
    fn from(_: ()) -> ArtifactOptions {
        ArtifactOptions::default()
    }
}
