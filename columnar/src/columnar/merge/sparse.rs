use crate::column_index::SerializableColumnIndex;
use crate::iterable::Iterable;
use crate::{ColumnIndex, MergeRowOrder, ShuffleMergeOrder};

const INVERSE_ORDER_BUDGET: usize = 64 * 1024 * 1024;

struct InverseShuffleOrder {
    #[cfg(debug_assertions)]
    shuffle_order_addr: usize,
    segment_offsets: Vec<usize>,
    old_to_new: Vec<u32>,
}

impl InverseShuffleOrder {
    fn build(num_rows: &[u32], order: &ShuffleMergeOrder, budget: usize) -> Option<Self> {
        let total_rows = num_rows
            .iter()
            .try_fold(0usize, |sum, &rows| sum.checked_add(rows as usize))?;
        if total_rows > budget / std::mem::size_of::<u32>() {
            return None;
        }
        let mut segment_offsets = Vec::with_capacity(num_rows.len() + 1);
        segment_offsets.push(0);
        for &rows in num_rows {
            segment_offsets.push(segment_offsets.last().copied().unwrap() + rows as usize);
        }
        let mut old_to_new = vec![u32::MAX; total_rows];
        for (new_row, addr) in order.iter_new_to_old_row_addrs().enumerate() {
            debug_assert!(
                addr.row_id < num_rows[addr.segment_ord as usize],
                "shuffle row exceeds its source segment"
            );
            debug_assert!(
                new_row < u32::MAX as usize,
                "output row must not use the deleted-row sentinel"
            );
            let slot =
                &mut old_to_new[segment_offsets[addr.segment_ord as usize] + addr.row_id as usize];
            if *slot != u32::MAX {
                return None;
            }
            *slot = u32::try_from(new_row).ok()?;
        }
        Some(Self {
            #[cfg(debug_assertions)]
            shuffle_order_addr: order as *const ShuffleMergeOrder as usize,
            segment_offsets,
            old_to_new,
        })
    }

    fn segment(&self, segment: usize) -> &[u32] {
        &self.old_to_new[self.segment_offsets[segment]..self.segment_offsets[segment + 1]]
    }
}

#[derive(Default)]
pub(super) struct SparseMergeContext {
    inverse: Option<InverseShuffleOrder>,
    disabled: bool,
    packed: Vec<u64>,
}

impl SparseMergeContext {
    pub(super) fn prepare<'a, 'b>(
        &'a mut self,
        column_index: &SerializableColumnIndex<'_>,
        indexes: impl Iterator<Item = Option<&'b ColumnIndex>> + Clone,
        merge_order: &MergeRowOrder,
        num_rows_per_segment: &[u32],
        mut map_rank: impl FnMut(usize, u32) -> u32,
    ) -> Option<&'a [u64]> {
        if self.disabled || !matches!(column_index, SerializableColumnIndex::Optional { .. }) {
            return None;
        }
        let MergeRowOrder::Shuffled(order) = merge_order else {
            return None;
        };
        let mut value_count = 0u64;
        for index in indexes.clone().flatten() {
            match index {
                ColumnIndex::Empty { .. } => {}
                ColumnIndex::Optional(index) => value_count += u64::from(index.num_non_nulls()),
                _ => return None,
            }
        }
        if value_count == 0 || value_count > u64::from(order.num_rows() / 8) {
            return None;
        }
        if self.inverse.is_none() {
            self.inverse =
                InverseShuffleOrder::build(num_rows_per_segment, order, INVERSE_ORDER_BUDGET);
            if self.inverse.is_none() {
                self.disabled = true;
                return None;
            }
        }
        let inverse = self.inverse.as_ref().unwrap();
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                inverse.shuffle_order_addr, order as *const ShuffleMergeOrder as usize,
                "inverse shuffle order belongs to a different merge"
            );
            debug_assert_eq!(
                inverse.segment_offsets.len(),
                num_rows_per_segment.len() + 1
            );
            debug_assert!(
                inverse
                    .segment_offsets
                    .windows(2)
                    .zip(num_rows_per_segment)
                    .all(|(offsets, &rows)| offsets[1] - offsets[0] == rows as usize),
                "inverse shuffle order source sizes changed"
            );
        }
        self.packed.clear();
        self.packed.reserve_exact(value_count as usize);
        for (segment, index) in indexes.enumerate() {
            let Some(ColumnIndex::Optional(index)) = index else {
                continue;
            };
            let old_to_new = inverse.segment(segment);
            for (rank, row) in index.iter_rows().enumerate() {
                let new_row = old_to_new[row as usize];
                if new_row == u32::MAX {
                    continue;
                }
                debug_assert!(
                    rank < u32::MAX as usize,
                    "source value rank must fit in the packed reference"
                );
                debug_assert!(
                    {
                        let addr = order.new_row_id_to_old_row_id[new_row as usize];
                        addr.segment_ord == segment as u32 && addr.row_id == row
                    },
                    "inverse shuffle order must round-trip each surviving source row"
                );
                let value_ref = map_rank(segment, rank as u32);
                self.packed
                    .push((u64::from(new_row) << 32) | u64::from(value_ref));
            }
        }
        self.packed.sort_unstable();
        debug_assert!(
            self.packed
                .windows(2)
                .all(|pair| (pair[0] >> 32) < (pair[1] >> 32)),
            "packed Optional rows must be strictly increasing"
        );
        Some(&self.packed)
    }
}

pub(super) struct PackedRowIds<'a>(pub(super) &'a [u64]);

impl Iterable<u32> for PackedRowIds<'_> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        Box::new(self.0.iter().map(|&packed| (packed >> 32) as u32))
    }
}

#[cfg(test)]
mod sparse_tests {
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    use super::*;
    use crate::columnar::merge::merge_columnar_with_context;
    use crate::{ColumnType, ColumnarReader, ColumnarWriter, RowAddr};

    pub(super) fn reader(rows: u32, stride: u32, multi: bool) -> ColumnarReader {
        let mut writer = ColumnarWriter::default();
        for row in (0..rows).step_by(stride as usize) {
            writer.record_str(row, "str", &format!("term_{row}"));
            writer.record_bytes(row, "raw", &row.to_le_bytes());
            let value = u64::MAX - (u64::from(rows) << 32) - u64::from(row);
            writer.record_numerical(row, "a_u64", value);
            writer.record_numerical(row, "z_u64", value ^ (1u64 << 45));
            writer.record_numerical(
                row,
                "i64",
                i64::MIN + (i64::from(rows) << 32) + i64::from(row),
            );
            writer.record_numerical(
                row,
                "f64",
                -(f64::from(rows) * 1024.0 + f64::from(row) + 0.25),
            );
            writer.record_bool(row, "bool", (row / stride + rows) % 2 == 0);
            writer.record_datetime(
                row,
                "date",
                crate::DateTime::from_timestamp_micros(-i64::from(rows) * 1000000 - i64::from(row)),
            );
            if multi && row == 0 {
                writer.record_numerical(row, "a_u64", value - 1);
                writer.record_numerical(row, "z_u64", value - 1);
                writer.record_numerical(row, "i64", -1i64);
                writer.record_numerical(row, "f64", 0.25f64);
                writer.record_bool(row, "bool", false);
                writer.record_datetime(row, "date", crate::DateTime::from_timestamp_micros(0));
                writer.record_str(row, "str", "extra");
                writer.record_bytes(row, "raw", b"extra");
            }
        }
        let mut bytes = Vec::new();
        writer.serialize(rows, None, &mut bytes).unwrap();
        ColumnarReader::open(bytes).unwrap()
    }

    pub(super) fn compare(
        readers: &[&ColumnarReader],
        sizes: &[u32],
        addresses: Vec<RowAddr>,
    ) -> SparseMergeContext {
        let required = [("empty".to_owned(), ColumnType::Str)];
        let mut expected = Vec::new();
        merge_columnar_with_context(
            readers,
            &required,
            ShuffleMergeOrder::for_test(sizes, addresses.clone()).into(),
            &mut SparseMergeContext {
                disabled: true,
                ..Default::default()
            },
            &mut expected,
        )
        .unwrap();
        let mut context = SparseMergeContext::default();
        let mut actual = Vec::new();
        merge_columnar_with_context(
            readers,
            &required,
            ShuffleMergeOrder::for_test(sizes, addresses).into(),
            &mut context,
            &mut actual,
        )
        .unwrap();
        assert_eq!(actual, expected);
        context
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]
        #[test]
        fn fused_matches_streaming(rows in 32u32..2000, stride in 8u32..200, seed in any::<u64>(), deletes in any::<bool>()) {
            let sizes = [rows, 0, rows + 13];
            let readers: Vec<_> = sizes.iter().map(|&n| reader(n, stride, false)).collect();
            let refs: Vec<_> = readers.iter().collect();
            let mut addresses: Vec<_> = sizes.iter().enumerate().flat_map(|(segment, &n)| {
                (0..n).filter(move |&row| !deletes || row % 3 != 2).map(move |row_id| RowAddr { segment_ord: segment as u32, row_id })
            }).collect();
            addresses.shuffle(&mut StdRng::seed_from_u64(seed));
            let count = addresses.len();
            let upper_bound: u32 = sizes.iter().map(|&n| if n == 0 { 0 } else { (n - 1) / stride + 1 }).sum();
            let context = compare(&refs, &sizes, addresses);
            prop_assert_eq!(context.inverse.is_some(), u64::from(upper_bound) <= (count / 8) as u64);
        }
    }

    #[test]
    fn source_and_target_cardinality_guards() {
        let sparse = reader(1024, 100, false);
        let full = reader(1, 1, false);
        let addresses: Vec<_> = (0..1024)
            .map(|row_id| RowAddr {
                segment_ord: 0,
                row_id,
            })
            .chain(std::iter::once(RowAddr {
                segment_ord: 1,
                row_id: 0,
            }))
            .collect();
        assert!(compare(&[&sparse, &full], &[1024, 1], addresses)
            .inverse
            .is_none());
        let multi = reader(256, 100, true);
        let addresses: Vec<_> = (1..256)
            .map(|row_id| RowAddr {
                segment_ord: 0,
                row_id,
            })
            .chain((0..1024).map(|row_id| RowAddr {
                segment_ord: 1,
                row_id,
            }))
            .collect();
        assert!(compare(&[&multi, &sparse], &[256, 1024], addresses)
            .inverse
            .is_none());
        let addresses: Vec<_> = (0..1024)
            .step_by(100)
            .map(|row_id| RowAddr {
                segment_ord: 0,
                row_id,
            })
            .collect();
        assert!(compare(&[&sparse], &[1024], addresses).inverse.is_none());
    }

    #[test]
    fn duplicate_rows_preserve_streaming_behavior() {
        let sparse = reader(1024, 100, false);
        let addresses: Vec<_> = (0..1024)
            .map(|row_id| RowAddr {
                segment_ord: 0,
                row_id,
            })
            .chain(std::iter::once(RowAddr {
                segment_ord: 0,
                row_id: 0,
            }))
            .collect();
        let context = compare(&[&sparse], &[1024], addresses);
        assert!(context.inverse.is_none());
        assert!(context.disabled);
    }

    #[test]
    fn inverse_budget_and_deleted_rows() {
        let order = ShuffleMergeOrder::for_test(
            &[2, 0, 3],
            vec![
                RowAddr {
                    segment_ord: 2,
                    row_id: 2,
                },
                RowAddr {
                    segment_ord: 0,
                    row_id: 1,
                },
            ],
        );
        assert!(InverseShuffleOrder::build(&[2, 0, 3], &order, 19).is_none());
        let inverse = InverseShuffleOrder::build(&[2, 0, 3], &order, 20).unwrap();
        assert_eq!(inverse.segment(0), &[u32::MAX, 1]);
        assert_eq!(inverse.segment(1), &[]);
        assert_eq!(inverse.segment(2), &[u32::MAX, u32::MAX, 0]);
        assert!(
            InverseShuffleOrder::build(&[u32::MAX, u32::MAX], &order, INVERSE_ORDER_BUDGET)
                .is_none()
        );
    }

    #[test]
    fn packed_row_limits() {
        let packed = [
            u64::from(u32::MAX),
            (u64::from(u32::MAX - 1) << 32) | u64::from(u32::MAX),
        ];
        assert_eq!(
            PackedRowIds(&packed).boxed_iter().collect::<Vec<_>>(),
            [0, u32::MAX - 1]
        );
    }
    #[test]
    fn numeric_only_fusion_preserves_source_ranks() {
        let sizes = [128, 0, 256];
        let populated: [&[u32]; 3] = [&[3, 99], &[], &[0, 1, 200]];
        for kind in 0..6 {
            let readers: Vec<_> = sizes
                .iter()
                .enumerate()
                .map(|(segment, &rows)| {
                    let mut writer = ColumnarWriter::default();
                    for (rank, &row) in populated[segment].iter().enumerate() {
                        let n = ((segment as u64) << 20) | (rank as u64 + 3);
                        match kind {
                            0 => writer.record_numerical(row, "numeric", u64::MAX - n),
                            1 => writer.record_numerical(row, "numeric", i64::MIN + n as i64),
                            2 => writer.record_numerical(row, "numeric", -(n as f64 + 0.25)),
                            3 => writer.record_bool(row, "numeric", n % 2 == 0),
                            4 => writer.record_datetime(
                                row,
                                "numeric",
                                crate::DateTime::from_timestamp_micros(-(n as i64) * 100),
                            ),
                            _ if segment == 0 => {
                                writer.record_numerical(row, "numeric", -(n as i64))
                            }
                            _ => writer.record_numerical(row, "numeric", u64::MAX - n),
                        }
                    }
                    let mut bytes = Vec::new();
                    writer.serialize(rows, None, &mut bytes).unwrap();
                    ColumnarReader::open(bytes).unwrap()
                })
                .collect();
            let refs: Vec<_> = readers.iter().collect();
            let mut addresses: Vec<_> = sizes
                .iter()
                .enumerate()
                .flat_map(|(segment, &rows)| {
                    (0..rows).map(move |row_id| RowAddr {
                        segment_ord: segment as u32,
                        row_id,
                    })
                })
                .collect();
            addresses.shuffle(&mut StdRng::seed_from_u64(801));
            let expected: Vec<_> = addresses
                .iter()
                .enumerate()
                .filter_map(|(new_row, addr)| {
                    populated[addr.segment_ord as usize]
                        .iter()
                        .position(|&row| row == addr.row_id)
                        .map(|rank| ((new_row as u64) << 32) | rank as u64)
                })
                .collect();
            let context = compare(&refs, &sizes, addresses);
            assert!(context.inverse.is_some());
            assert_eq!(context.packed, expected);
        }
    }
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "shuffle row exceeds its source segment")]
    fn inverse_rejects_cross_segment_row_alias() {
        let order = ShuffleMergeOrder {
            new_row_id_to_old_row_id: vec![RowAddr {
                segment_ord: 0,
                row_id: 2,
            }],
            alive_bitsets: vec![None, None],
        };
        let _ = InverseShuffleOrder::build(&[2, 2], &order, 16);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "inverse shuffle order belongs to a different merge")]
    fn inverse_cannot_be_reused_across_merges() {
        let indexes = [ColumnIndex::Optional(
            crate::column_index::OptionalIndex::for_test(128, &[50]),
        )];
        let index = SerializableColumnIndex::Optional {
            non_null_row_ids: Box::new(0..1),
            num_rows: 128,
        };
        let first: MergeRowOrder = ShuffleMergeOrder::for_test(
            &[128],
            (0..128)
                .map(|row_id| RowAddr {
                    segment_ord: 0,
                    row_id,
                })
                .collect(),
        )
        .into();
        let second: MergeRowOrder = ShuffleMergeOrder::for_test(
            &[128],
            (0..128)
                .rev()
                .map(|row_id| RowAddr {
                    segment_ord: 0,
                    row_id,
                })
                .collect(),
        )
        .into();
        let mut context = SparseMergeContext::default();
        assert!(context
            .prepare(
                &index,
                indexes.iter().map(Some),
                &first,
                &[128],
                |_, rank| rank
            )
            .is_some());
        let _ = context.prepare(
            &index,
            indexes.iter().map(Some),
            &second,
            &[128],
            |_, rank| rank,
        );
    }
}

#[cfg(test)]
mod coverage_boundary_tests {
    use super::sparse_tests::{compare, reader};
    use crate::{ColumnarReader, ColumnarWriter, RowAddr};

    #[test]
    fn fusion_density_cutoff_and_small_output_boundaries() {
        for rows in [0, 1, 7, 8, 9, 15, 16, 17] {
            for count in [0, rows / 8, rows / 8 + 1] {
                if count > rows {
                    continue;
                }
                let mut writer = ColumnarWriter::default();
                for row in 0..count {
                    writer.record_numerical(row, "value", u64::MAX - u64::from(row));
                }
                let mut bytes = Vec::new();
                writer.serialize(rows, None, &mut bytes).unwrap();
                let reader = ColumnarReader::open(bytes).unwrap();
                let addresses = (0..rows)
                    .rev()
                    .map(|row_id| RowAddr {
                        segment_ord: 0,
                        row_id,
                    })
                    .collect();
                let context = compare(&[&reader], &[rows], addresses);
                assert_eq!(context.inverse.is_some(), count > 0 && count <= rows / 8);
            }
        }
    }

    #[test]
    fn fusion_with_dense_blocks_and_deleted_populated_rows() {
        let mut writer = ColumnarWriter::default();
        for row in (0..5120).chain([65536]) {
            writer.record_numerical(row, "a_u64", u64::MAX - u64::from(row));
            writer.record_str(row, "str", "dense");
        }
        let mut bytes = Vec::new();
        writer.serialize(65537, None, &mut bytes).unwrap();
        let dense = ColumnarReader::open(bytes).unwrap();
        let sparse = reader(65537, 100, false);
        let addresses = (0..2)
            .flat_map(|segment_ord| {
                (0..65537)
                    .rev()
                    .filter(|row| row % 11 != 0)
                    .map(move |row_id| RowAddr {
                        segment_ord,
                        row_id,
                    })
            })
            .collect();
        let context = compare(&[&dense, &sparse], &[65537, 65537], addresses);
        assert!(context.inverse.is_some());
    }
}

#[cfg(all(test, debug_assertions))]
mod inverse_guard_tests {
    use super::*;
    use crate::column_index::OptionalIndex;
    use crate::RowAddr;

    #[test]
    #[should_panic(expected = "inverse shuffle order must round-trip each surviving source row")]
    fn rejects_corrupted_inverse_segment() {
        let sizes = [128, 128];
        let order: MergeRowOrder = ShuffleMergeOrder::for_test(
            &sizes,
            (0..2)
                .flat_map(|segment_ord| {
                    (0..128).map(move |row_id| RowAddr {
                        segment_ord,
                        row_id,
                    })
                })
                .collect(),
        )
        .into();
        let MergeRowOrder::Shuffled(shuffle) = &order else {
            unreachable!()
        };
        let mut inverse =
            InverseShuffleOrder::build(&sizes, shuffle, INVERSE_ORDER_BUDGET).unwrap();
        inverse.old_to_new[50] = 128 + 50;
        let mut context = SparseMergeContext {
            inverse: Some(inverse),
            ..Default::default()
        };
        let indexes = [
            ColumnIndex::Optional(OptionalIndex::for_test(128, &[50])),
            ColumnIndex::Empty { num_docs: 128 },
        ];
        let index = SerializableColumnIndex::Optional {
            non_null_row_ids: Box::new(50..51),
            num_rows: 256,
        };
        let _ = context.prepare(
            &index,
            indexes.iter().map(Some),
            &order,
            &sizes,
            |_, rank| rank,
        );
    }
}

#[cfg(fuzzing)]
pub fn merge_columnar_for_fuzzing(
    columnar_readers: &[&crate::ColumnarReader],
    required_columns: &[(String, crate::ColumnType)],
    merge_row_order: MergeRowOrder,
    output: &mut impl std::io::Write,
) -> std::io::Result<()> {
    super::merge_columnar_with_context(
        columnar_readers,
        required_columns,
        merge_row_order,
        &mut SparseMergeContext {
            disabled: true,
            ..Default::default()
        },
        output,
    )
}
