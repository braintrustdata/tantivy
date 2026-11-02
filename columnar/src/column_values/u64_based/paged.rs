use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

use common::{ChunkedBytes, OwnedBytes};
use tantivy_bitpacker::BitUnpacker;

/// Bytes that bitpacked column values are read from.
pub trait ColumnData: Clone + Send + Sync + 'static {
    /// Returns the value at `idx` of the bitpacked run that starts at `byte_offset`.
    fn get_bits(&self, bit_unpacker: &BitUnpacker, byte_offset: usize, idx: u32) -> u64;

    /// Same as [`BitUnpacker::get_ids_for_value_range`].
    fn get_ids_for_value_range(
        &self,
        bit_unpacker: &BitUnpacker,
        range: RangeInclusive<u64>,
        id_range: Range<u32>,
        positions: &mut Vec<u32>,
    );
}

impl ColumnData for OwnedBytes {
    #[inline(always)]
    fn get_bits(&self, bit_unpacker: &BitUnpacker, byte_offset: usize, idx: u32) -> u64 {
        if byte_offset == 0 {
            return bit_unpacker.get(idx, self.as_slice());
        }
        bit_unpacker.get(idx, &self.as_slice()[byte_offset..])
    }

    fn get_ids_for_value_range(
        &self,
        bit_unpacker: &BitUnpacker,
        range: RangeInclusive<u64>,
        id_range: Range<u32>,
        positions: &mut Vec<u32>,
    ) {
        bit_unpacker.get_ids_for_value_range(range, id_range, self.as_slice(), positions);
    }
}

/// Column data kept as the page-cache pages it was read from, so opening a column does not
/// copy it. Every page but the first and the last holds `1 << page_shift` bytes.
#[derive(Clone)]
pub(crate) struct PagedBytes {
    pages: Arc<[OwnedBytes]>,
    // Bytes missing from the front of the first page, so that offsets map to pages by shifting.
    skip: usize,
    page_shift: u32,
    page_mask: usize,
    len: usize,
}

impl PagedBytes {
    /// Returns the bytes unchanged when they are one chunk or not split into equal pages.
    pub(crate) fn new(bytes: ChunkedBytes) -> Result<Self, ChunkedBytes> {
        let pages: Vec<OwnedBytes> = bytes.chunks().collect();
        let page_len = pages.iter().map(OwnedBytes::len).max().unwrap_or(0);
        let uniform = pages.len() > 1
            && page_len.is_power_of_two()
            && pages[1..pages.len() - 1]
                .iter()
                .all(|page| page.len() == page_len);
        if !uniform {
            return Err(bytes);
        }
        Ok(Self {
            skip: page_len - pages[0].len(),
            page_shift: page_len.trailing_zeros(),
            page_mask: page_len - 1,
            len: bytes.len(),
            pages: pages.into(),
        })
    }

    #[inline(always)]
    fn locate(&self, addr: usize) -> (usize, usize) {
        let position = addr + self.skip;
        let page = position >> self.page_shift;
        let offset = if page == 0 {
            addr
        } else {
            position & self.page_mask
        };
        (page, offset)
    }

    #[inline(always)]
    fn read_u64_le(&self, addr: usize) -> u64 {
        let (page, offset) = self.locate(addr);
        if let Some(bytes) = self
            .pages
            .get(page)
            .and_then(|page| page.as_slice().get(offset..offset + 8))
        {
            return u64::from_le_bytes(bytes.try_into().unwrap());
        }
        self.read_u64_le_across_pages(addr)
    }

    #[cold]
    #[inline(never)]
    fn read_u64_le_across_pages(&self, addr: usize) -> u64 {
        let mut bytes = [0u8; 8];
        for (slot, addr) in bytes.iter_mut().zip(addr..self.len) {
            let (page, offset) = self.locate(addr);
            *slot = self.pages[page].as_slice()[offset];
        }
        u64::from_le_bytes(bytes)
    }
}

impl ColumnData for PagedBytes {
    #[inline(always)]
    fn get_bits(&self, bit_unpacker: &BitUnpacker, byte_offset: usize, idx: u32) -> u64 {
        bit_unpacker.get_with(idx, |addr| self.read_u64_le(byte_offset + addr))
    }

    /// Decodes each page in batches. Values whose 8-value group crosses a page are read one by
    /// one.
    fn get_ids_for_value_range(
        &self,
        bit_unpacker: &BitUnpacker,
        range: RangeInclusive<u64>,
        id_range: Range<u32>,
        positions: &mut Vec<u32>,
    ) {
        positions.clear();
        let num_bits = bit_unpacker.bit_width() as usize;
        let push_one_by_one = |ids: Range<u32>, positions: &mut Vec<u32>| {
            for idx in ids {
                if range.contains(&self.get_bits(bit_unpacker, 0, idx)) {
                    positions.push(idx);
                }
            }
        };
        if num_bits == 0 {
            return push_one_by_one(id_range, positions);
        }
        let end = id_range.end.min((self.len * 8 / num_bits) as u32);
        let mut batch = Vec::new();
        let mut idx = id_range.start;
        while idx < end {
            // Every 8th value starts on a byte, so groups of 8 can be decoded from one page.
            let (page, _) = self.locate(idx as usize * num_bits / 8);
            let page_end = (((page + 1) << self.page_shift) - self.skip).min(self.len);
            let lo = idx.next_multiple_of(8);
            let hi = ((page_end * 8 / num_bits) as u32 & !7).min(end);
            if lo >= hi {
                let next = (lo + 8).min(end);
                push_one_by_one(idx..next, positions);
                idx = next;
                continue;
            }
            push_one_by_one(idx..lo, positions);
            let (_, offset) = self.locate(lo as usize * num_bits / 8);
            bit_unpacker.get_ids_for_value_range(
                range.clone(),
                0..hi - lo,
                &self.pages[page].as_slice()[offset..],
                &mut batch,
            );
            positions.extend(batch.iter().map(|id| id + lo));
            idx = hi;
        }
    }
}

#[cfg(test)]
mod tests {
    use common::{ChunkedBytes, OwnedBytes};

    use crate::column_values::u64_based::tests::get_codec_test_datasets;
    use crate::column_values::u64_based::{
        load_u64_based_column_values, load_u64_based_column_values_from_chunks,
        serialize_u64_based_column_values, ALL_U64_CODEC_TYPES,
    };

    fn paged(bytes: &OwnedBytes, offset: usize, page_size: usize) -> ChunkedBytes {
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < bytes.len() {
            let end = (((offset + start) / page_size + 1) * page_size - offset).min(bytes.len());
            chunks.push(bytes.slice(start..end));
            start = end;
        }
        ChunkedBytes::new(chunks)
    }

    #[test]
    fn values_read_from_pages_match_contiguous_values() {
        let mut datasets = get_codec_test_datasets();
        datasets.push(((0..5_000u64).map(|i| i % 7).collect(), "few bits"));
        datasets.push((
            (0..5_000u64)
                .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) >> 8)
                .collect(),
            "wide",
        ));
        for (values, name) in datasets {
            for codec in ALL_U64_CODEC_TYPES {
                let mut buffer = Vec::new();
                if serialize_u64_based_column_values(&&values[..], &[codec], &mut buffer).is_err() {
                    continue;
                }
                let bytes = OwnedBytes::new(buffer);
                let expected = load_u64_based_column_values::<u64>(bytes.clone()).unwrap();
                for page_size in [1usize, 2, 8, 64, 4096, 1 << 20] {
                    for offset in [0, 3, page_size.saturating_sub(1)] {
                        let actual = load_u64_based_column_values_from_chunks::<u64>(paged(
                            &bytes, offset, page_size,
                        ))
                        .unwrap();
                        assert_eq!(actual.num_vals(), expected.num_vals());
                        for idx in 0..values.len() as u32 {
                            assert_eq!(
                                actual.get_val(idx),
                                expected.get_val(idx),
                                "{name} {codec:?} page={page_size} offset={offset} idx={idx}"
                            );
                        }
                        let range = expected.get_val(values.len() as u32 / 3)
                            ..=expected
                                .get_val(values.len() as u32 / 2)
                                .max(expected.get_val(values.len() as u32 / 3));
                        let num_vals = values.len() as u32;
                        for ids in [0..num_vals, num_vals / 5..num_vals - 1] {
                            let (mut want, mut got) = (Vec::new(), Vec::new());
                            expected.get_row_ids_for_value_range(
                                range.clone(),
                                ids.clone(),
                                &mut want,
                            );
                            actual.get_row_ids_for_value_range(range.clone(), ids, &mut got);
                            assert_eq!(got, want, "{name} {codec:?} page={page_size}");
                        }
                    }
                }
            }
        }
    }
}
