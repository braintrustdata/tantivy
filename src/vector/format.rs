//! Vector storage format definitions.

use std::borrow::Cow;

use half::f16;

/// Magic number for vector files: "TVEC" in little-endian
pub const VECTOR_MAGIC: u32 = 0x43455654; // "TVEC"

/// Current format version
pub const VECTOR_VERSION: u8 = 1;

/// Encoding type for vector data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum VectorEncoding {
    /// 32-bit float (4 bytes per dimension) - full precision
    #[default]
    F32 = 0,
    /// 16-bit float (2 bytes per dimension) - ~0.1% accuracy loss
    F16 = 1,
    /// 8-bit quantized (1 byte per dimension) - ~1-2% accuracy loss
    /// Requires scale/zero_point for dequantization
    Int8 = 2,
}

impl VectorEncoding {
    /// Bytes per dimension for this encoding
    pub fn bytes_per_dim(&self) -> usize {
        match self {
            VectorEncoding::F32 => 4,
            VectorEncoding::F16 => 2,
            VectorEncoding::Int8 => 1,
        }
    }

    /// Parse from u8
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(VectorEncoding::F32),
            1 => Some(VectorEncoding::F16),
            2 => Some(VectorEncoding::Int8),
            _ => None,
        }
    }
}

/// Int8 quantization parameters for a vector column.
#[derive(Debug, Clone, Copy)]
pub struct Int8QuantParams {
    /// Scale factor: original = quantized * scale + zero_point
    pub scale: f32,
    /// Zero point offset
    pub zero_point: f32,
}

impl Int8QuantParams {
    /// Compute quantization parameters from a set of vectors.
    pub fn from_vectors<'a>(vectors: impl Iterator<Item = &'a [f32]>) -> Self {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for vec in vectors {
            for &v in vec {
                min = min.min(v);
                max = max.max(v);
            }
        }

        if min == f32::MAX {
            // No vectors
            return Self {
                scale: 1.0,
                zero_point: 0.0,
            };
        }

        let range = max - min;
        if range < f32::EPSILON {
            Self {
                scale: 1.0,
                zero_point: min,
            }
        } else {
            Self {
                scale: range / 255.0,
                zero_point: min,
            }
        }
    }

    /// Quantize a f32 value to u8
    #[inline]
    pub fn quantize(&self, v: f32) -> u8 {
        ((v - self.zero_point) / self.scale)
            .round()
            .clamp(0.0, 255.0) as u8
    }

    /// Dequantize a u8 value to f32
    #[inline]
    pub fn dequantize(&self, v: u8) -> f32 {
        v as f32 * self.scale + self.zero_point
    }

    /// Quantize a vector
    pub fn quantize_vec(&self, vec: &[f32]) -> Vec<u8> {
        vec.iter().map(|&v| self.quantize(v)).collect()
    }

    /// Dequantize a vector
    pub fn dequantize_vec(&self, bytes: &[u8]) -> Vec<f32> {
        bytes.iter().map(|&v| self.dequantize(v)).collect()
    }
}

/// Encode f32 vectors to f16 bytes.
pub fn encode_f16(vec: &[f32]) -> Vec<u8> {
    vec.iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

/// Decode f16 bytes to f32 vector.
pub fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

/// Encode f32 vector to bytes based on encoding.
pub fn encode_vector(vec: &[f32], encoding: VectorEncoding, quant: Option<&Int8QuantParams>) -> Vec<u8> {
    match encoding {
        VectorEncoding::F32 => {
            vec.iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect()
        }
        VectorEncoding::F16 => encode_f16(vec),
        VectorEncoding::Int8 => {
            let quant = quant.expect("Int8 encoding requires quantization params");
            quant.quantize_vec(vec)
        }
    }
}

/// Decode bytes to f32 vector based on encoding.
/// Returns borrowed slice for f32 (zero-copy), owned Vec for f16/int8.
pub fn decode_vector<'a>(bytes: &'a [u8], encoding: VectorEncoding, quant: Option<&Int8QuantParams>) -> Cow<'a, [f32]> {
    match encoding {
        VectorEncoding::F32 => {
            Cow::Borrowed(bytemuck::cast_slice(bytes))
        }
        VectorEncoding::F16 => Cow::Owned(decode_f16(bytes)),
        VectorEncoding::Int8 => {
            let quant = quant.expect("Int8 encoding requires quantization params");
            Cow::Owned(quant.dequantize_vec(bytes))
        }
    }
}

/// Builder for constructing a PresenceBitset.
#[derive(Debug, Clone)]
pub struct PresenceBitsetBuilder {
    bits: Vec<u64>,
}

impl PresenceBitsetBuilder {
    /// Create a new builder with capacity for `len` bits (rounded up to 64).
    pub fn new(len: u32) -> Self {
        let num_words = (len as usize + 63) / 64;
        Self {
            bits: vec![0; num_words],
        }
    }

    /// Set bit at index.
    pub fn set(&mut self, index: u32) {
        let word = index as usize / 64;
        if word < self.bits.len() {
            let bit = index % 64;
            self.bits[word] |= 1 << bit;
        }
    }

    /// Build the final PresenceBitset, computing the cumulative popcount cache.
    pub fn build(self) -> PresenceBitset {
        let mut cumulative_counts = Vec::with_capacity(self.bits.len());
        let mut cumulative = 0u32;
        for &word in &self.bits {
            cumulative_counts.push(cumulative);
            cumulative += word.count_ones();
        }
        PresenceBitset {
            bits: self.bits,
            cumulative_counts,
        }
    }

    /// Get the raw bytes (little-endian u64 words) for serialization.
    pub fn as_bytes(&self) -> Vec<u8> {
        self.bits
            .iter()
            .flat_map(|&w| w.to_le_bytes())
            .collect()
    }
}

/// A read-only bitset for tracking presence of vectors.
///
/// Maintains a cumulative popcount cache for O(1) rank queries.
/// Construct via `PresenceBitsetBuilder::build()` or `PresenceBitset::from_bytes()`.
#[derive(Debug, Clone)]
pub struct PresenceBitset {
    bits: Vec<u64>,
    /// Cumulative popcount at each u64 boundary for O(1) rank queries.
    cumulative_counts: Vec<u32>,
}

impl PresenceBitset {
    /// Create from raw bytes (little-endian u64 words).
    pub fn from_bytes(bytes: &[u8], len: u32) -> Self {
        let num_words = (len as usize + 63) / 64;
        let mut bits = vec![0u64; num_words];
        for (i, chunk) in bytes.chunks(8).enumerate() {
            if i < num_words {
                let mut word_bytes = [0u8; 8];
                word_bytes[..chunk.len()].copy_from_slice(chunk);
                bits[i] = u64::from_le_bytes(word_bytes);
            }
        }

        // Build cumulative popcount cache
        let mut cumulative_counts = Vec::with_capacity(num_words);
        let mut cumulative = 0u32;
        for &word in &bits {
            cumulative_counts.push(cumulative);
            cumulative += word.count_ones();
        }

        Self {
            bits,
            cumulative_counts,
        }
    }

    /// Check if bit is set at index.
    pub fn get(&self, index: u32) -> bool {
        let word = index as usize / 64;
        if word >= self.bits.len() {
            return false;
        }
        let bit = index % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Count number of set bits (popcount).
    pub fn count_ones(&self) -> u32 {
        if self.bits.is_empty() {
            return 0;
        }
        let last_idx = self.bits.len() - 1;
        self.cumulative_counts[last_idx] + self.bits[last_idx].count_ones()
    }

    /// Count number of set bits before index (for computing vector offset).
    #[inline]
    pub fn count_ones_before(&self, index: u32) -> u32 {
        let word_idx = (index / 64) as usize;
        let bit_offset = index % 64;

        let base_count = if word_idx < self.cumulative_counts.len() {
            self.cumulative_counts[word_idx]
        } else {
            return self.count_ones();
        };

        if bit_offset == 0 {
            return base_count;
        }

        let mask = (1u64 << bit_offset) - 1;
        base_count + (self.bits[word_idx] & mask).count_ones()
    }

    /// Iterate over indices where bit is set.
    pub fn iter_ones(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits.iter().enumerate().flat_map(|(word_idx, &word)| {
            let base = (word_idx * 64) as u32;
            (0..64).filter_map(move |bit| {
                if (word >> bit) & 1 == 1 {
                    Some(base + bit)
                } else {
                    None
                }
            })
        })
    }

    /// Capacity in bits (rounded up to multiple of 64).
    pub fn capacity(&self) -> u32 {
        (self.bits.len() * 64) as u32
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip() {
        let original = vec![0.0, 1.0, -1.0, 0.5, 100.0, -100.0];
        let encoded = encode_f16(&original);
        let decoded = decode_f16(&encoded);

        assert_eq!(encoded.len(), original.len() * 2);
        for (o, d) in original.iter().zip(decoded.iter()) {
            assert!((o - d).abs() < 0.01, "f16 roundtrip: {} vs {}", o, d);
        }
    }

    #[test]
    fn test_int8_quantization() {
        let vectors = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.25, 0.75, 0.9],
        ];
        let params = Int8QuantParams::from_vectors(vectors.iter().map(|v| v.as_slice()));

        for vec in &vectors {
            let quantized = params.quantize_vec(vec);
            let dequantized = params.dequantize_vec(&quantized);

            for (o, d) in vec.iter().zip(dequantized.iter()) {
                assert!((o - d).abs() < 0.01, "int8 roundtrip: {} vs {}", o, d);
            }
        }
    }

    #[test]
    fn test_presence_bitset() {
        let mut builder = PresenceBitsetBuilder::new(100);

        builder.set(0);
        builder.set(50);
        builder.set(99);

        let bitset = builder.build();

        assert!(bitset.get(0));
        assert!(bitset.get(50));
        assert!(bitset.get(99));
        assert!(!bitset.get(1));
        assert!(!bitset.get(98));

        assert_eq!(bitset.count_ones(), 3);
        assert_eq!(bitset.count_ones_before(0), 0);
        assert_eq!(bitset.count_ones_before(1), 1);
        assert_eq!(bitset.count_ones_before(50), 1);
        assert_eq!(bitset.count_ones_before(51), 2);
        assert_eq!(bitset.count_ones_before(100), 3);

        let ones: Vec<_> = bitset.iter_ones().collect();
        assert_eq!(ones, vec![0, 50, 99]);
    }

    #[test]
    fn test_bitset_serialization() {
        let mut builder = PresenceBitsetBuilder::new(150);
        builder.set(0);
        builder.set(63);
        builder.set(64);
        builder.set(127);
        builder.set(149);

        let bytes = builder.as_bytes();
        let restored = PresenceBitset::from_bytes(&bytes, 150);

        assert!(restored.get(0));
        assert!(restored.get(63));
        assert!(restored.get(64));
        assert!(restored.get(127));
        assert!(restored.get(149));
        assert!(!restored.get(1));
        assert!(!restored.get(148));

        // Verify cache was built correctly on deserialization
        assert_eq!(restored.count_ones_before(0), 0);
        assert_eq!(restored.count_ones_before(1), 1);
        assert_eq!(restored.count_ones_before(64), 2);
        assert_eq!(restored.count_ones_before(65), 3);
        assert_eq!(restored.count_ones_before(150), 5);
    }
}

