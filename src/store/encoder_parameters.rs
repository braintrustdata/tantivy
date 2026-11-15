/// Runtime-only Zstd overrides. Unset fields preserve the index level and Zstd defaults.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ZstdEncoderParameters {
    /// Overrides the compression level selected by index settings.
    pub compression_level: Option<i32>,
    /// Match-finding strategy.
    pub strategy: Option<ZstdStrategy>,
    /// Base-two logarithm of the primary probe table entry count.
    pub hash_log: Option<u32>,
    /// Base-two logarithm of the chain or secondary probe table entry count.
    pub chain_log: Option<u32>,
    /// Minimum searched match length.
    pub min_match: Option<u32>,
    /// Strategy-dependent search target or fast-strategy sampling distance.
    pub target_length: Option<u32>,
}

/// Zstd match-finding strategies supported by runtime encoder overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZstdStrategy {
    /// Single-table fast search.
    Fast,
    /// Two-table fast search.
    DoubleFast,
    /// Greedy match search.
    Greedy,
    /// Lazy match search.
    Lazy,
}
