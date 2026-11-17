use crate::column_index::optional_index::{SelectCursor, Set, SetCodec};

pub struct SparseBlockCodec;

impl SetCodec for SparseBlockCodec {
    type Item = u16;
    type Reader<'a> = SparseBlock<'a>;

    fn serialize(
        els: impl Iterator<Item = u16>,
        mut wrt: impl std::io::Write,
    ) -> std::io::Result<()> {
        for el in els {
            wrt.write_all(&el.to_le_bytes())?;
        }
        Ok(())
    }

    fn open(data: &[u8]) -> Self::Reader<'_> {
        SparseBlock(data)
    }
}

#[derive(Copy, Clone)]
pub struct SparseBlock<'a>(&'a [u8]);

impl<'a> SelectCursor<u16> for SparseBlock<'a> {
    #[inline]
    fn select(&mut self, rank: u16) -> u16 {
        <SparseBlock<'a> as Set<u16>>::select(self, rank)
    }
}

impl<'a> Set<u16> for SparseBlock<'a> {
    type SelectCursor<'b>
        = Self
    where Self: 'b;

    #[inline(always)]
    fn contains(&self, el: u16) -> bool {
        self.binary_search(el).is_ok()
    }

    #[inline(always)]
    fn rank_if_exists(&self, el: u16) -> Option<u16> {
        self.binary_search(el).ok()
    }

    #[inline(always)]
    fn rank(&self, el: u16) -> u16 {
        self.binary_search(el).unwrap_or_else(|el| el)
    }

    #[inline(always)]
    fn select(&self, rank: u16) -> u16 {
        let offset = rank as usize * 2;
        u16::from_le_bytes(self.0[offset..offset + 2].try_into().unwrap())
    }

    #[inline(always)]
    fn select_cursor(&self) -> Self::SelectCursor<'_> {
        *self
    }
}

#[inline(always)]
fn get_u16(data: &[u8], byte_position: usize) -> u16 {
    let bytes: [u8; 2] = data[byte_position..byte_position + 2].try_into().unwrap();
    u16::from_le_bytes(bytes)
}

impl<'a> SparseBlock<'a> {
    #[inline(always)]
    fn value_at_idx(&self, data: &[u8], idx: u16) -> u16 {
        let start_offset: usize = idx as usize * 2;
        get_u16(data, start_offset)
    }

    #[inline]
    fn num_vals(&self) -> u16 {
        (self.0.len() / 2) as u16
    }

    #[inline]
    pub(crate) fn rank_if_exists_with_cursor(
        &self,
        target: u16,
        cursor: &mut u16,
        block_id: &mut Option<u16>,
    ) -> Option<u16> {
        let num_vals = self.num_vals();
        let idx = (*cursor).min(num_vals);
        let result = if idx < num_vals {
            let upper = self.value_at_idx(self.0, idx);
            if target == upper {
                *cursor = idx + 1;
                return Some(idx);
            }
            if target < upper {
                if idx == 0 {
                    *cursor = idx;
                    return None;
                }
                let lower = self.value_at_idx(self.0, idx - 1);
                if lower < target {
                    *cursor = idx;
                    return None;
                }
                if lower == target {
                    return Some(idx - 1);
                }
                *block_id = None;
                self.binary_search_range(target, 0, idx)
            } else {
                if idx + 1 < num_vals {
                    let next = self.value_at_idx(self.0, idx + 1);
                    if target < next {
                        *cursor = idx + 1;
                        return None;
                    }
                    if target == next {
                        *cursor = idx + 2;
                        return Some(idx + 1);
                    }
                }
                self.binary_search_range(target, (idx + 2).min(num_vals), num_vals)
            }
        } else {
            if num_vals == 0 {
                return None;
            }
            let last = self.value_at_idx(self.0, num_vals - 1);
            if target > last {
                return None;
            }
            if target == last {
                return Some(num_vals - 1);
            }
            *block_id = None;
            self.binary_search_range(target, 0, num_vals - 1)
        };
        match result {
            Ok(rank) => {
                *cursor = rank + 1;
                Some(rank)
            }
            Err(rank) => {
                *cursor = rank;
                None
            }
        }
    }

    #[inline]
    fn binary_search(&self, target: u16) -> Result<u16, u16> {
        self.binary_search_range(target, 0, self.num_vals())
    }

    #[inline]
    #[allow(clippy::comparison_chain)]
    fn binary_search_range(&self, target: u16, mut left: u16, mut right: u16) -> Result<u16, u16> {
        while left < right {
            let mid = left + (right - left) / 2;
            let mid_val = self.value_at_idx(self.0, mid);
            if target > mid_val {
                left = mid + 1;
            } else if target < mid_val {
                right = mid;
            } else {
                return Ok(mid);
            }
        }
        Err(left)
    }
}

#[cfg(test)]
mod cursor_tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn rank_cursor_matches_stateless(
            rows in proptest::collection::btree_set(any::<u16>(), 0..5120),
            queries in proptest::collection::vec(any::<u16>(), 0..512),
            initial in any::<u16>(),
        ) {
            let mut bytes = Vec::new();
            SparseBlockCodec::serialize(rows.iter().copied(), &mut bytes).unwrap();
            let block = SparseBlockCodec::open(&bytes);
            let mut cursor = initial;
            let mut block_id = Some(0);
            for query in queries.into_iter().chain(rows.iter().copied()).chain(rows.iter().rev().copied()) {
                let expected = block.rank_if_exists(query);
                prop_assert_eq!(block.rank_if_exists_with_cursor(query, &mut cursor, &mut block_id), expected);
                prop_assert_eq!(block.rank_if_exists_with_cursor(query, &mut cursor, &mut block_id), expected);
            }
        }
    }

    #[test]
    fn cursor_boundaries() {
        for rows in [
            vec![],
            vec![0],
            vec![u16::MAX],
            (0..5119).map(|i| i * 12).collect(),
        ] {
            let mut bytes = Vec::new();
            SparseBlockCodec::serialize(rows.iter().copied(), &mut bytes).unwrap();
            let block = SparseBlockCodec::open(&bytes);
            let mut cursor = 0;
            let mut block_id = Some(0);
            for query in (0..=u16::MAX).chain((0..=u16::MAX).rev()) {
                assert_eq!(
                    block.rank_if_exists_with_cursor(query, &mut cursor, &mut block_id),
                    block.rank_if_exists(query)
                );
            }
        }
    }
}
