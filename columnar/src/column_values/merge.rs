use std::fmt::Debug;
use std::sync::Arc;

use crate::column_index::Set;
use crate::iterable::Iterable;
use crate::{ColumnIndex, ColumnValues, MergeRowOrder};

pub(crate) struct MergedColumnValues<'a, T> {
    pub(crate) column_indexes: &'a [ColumnIndex],
    pub(crate) column_values: &'a [Option<Arc<dyn ColumnValues<T>>>],
    pub(crate) merge_row_order: &'a MergeRowOrder,
}

impl<'a, T: Copy + PartialOrd + Debug + 'static> Iterable<T> for MergedColumnValues<'a, T> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = T> + '_> {
        match self.merge_row_order {
            MergeRowOrder::Stack(_) => Box::new(
                self.column_values
                    .iter()
                    .flatten()
                    .flat_map(|column_value| column_value.iter()),
            ),
            MergeRowOrder::Shuffled(order) => {
                let mut cursors: Vec<_> = self
                    .column_indexes
                    .iter()
                    .map(|index| match index {
                        ColumnIndex::Optional(index) => index.cursor_for_sparse_blocks(),
                        _ => None,
                    })
                    .collect();
                let has_multivalued = self
                    .column_indexes
                    .iter()
                    .any(|index| matches!(index, ColumnIndex::Multivalued(_)));
                if !has_multivalued {
                    return Box::new(order.iter_new_to_old_row_addrs().filter_map(move |addr| {
                        let segment = addr.segment_ord as usize;
                        let values = self.column_values[segment].as_ref()?;
                        let rank = match &mut cursors[segment] {
                            Some(cursor) => cursor.rank_if_exists(addr.row_id)?,
                            None => match &self.column_indexes[segment] {
                                ColumnIndex::Full => addr.row_id,
                                ColumnIndex::Optional(index) => {
                                    index.rank_if_exists(addr.row_id)?
                                }
                                ColumnIndex::Empty { .. } => return None,
                                ColumnIndex::Multivalued(_) => unreachable!(),
                            },
                        };
                        Some(values.get_val(rank))
                    }));
                }
                Box::new(
                    order
                        .iter_new_to_old_row_addrs()
                        .filter_map(move |addr| {
                            let segment = addr.segment_ord as usize;
                            let values = self.column_values[segment].as_ref()?;
                            let range = match &mut cursors[segment] {
                                Some(cursor) => cursor
                                    .rank_if_exists(addr.row_id)
                                    .map(|rank| rank..rank + 1)
                                    .unwrap_or(0..0),
                                None => self.column_indexes[segment].value_row_ids(addr.row_id),
                            };
                            if range.is_empty() {
                                return None;
                            }
                            Some((range, values))
                        })
                        .flat_map(|(range, values)| range.map(move |rank| values.get_val(rank))),
                )
            }
        }
    }
}

impl MergedColumnValues<'_, u64> {
    pub(crate) fn with_sparse_rows<'a>(&'a self, packed: &'a [u64]) -> impl Iterable + 'a {
        let MergeRowOrder::Shuffled(order) = self.merge_row_order else {
            unreachable!();
        };
        SparseMergedColumnValues {
            packed,
            column_values: self.column_values,
            order,
        }
    }
}

struct SparseMergedColumnValues<'a> {
    packed: &'a [u64],
    column_values: &'a [Option<Arc<dyn ColumnValues>>],
    order: &'a crate::ShuffleMergeOrder,
}

impl Iterable for SparseMergedColumnValues<'_> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        Box::new(self.packed.iter().map(|&packed| {
            let new_row = (packed >> 32) as usize;
            let rank = packed as u32;
            let segment = self.order.new_row_id_to_old_row_id[new_row].segment_ord as usize;
            let values = self.column_values[segment].as_ref().unwrap();
            debug_assert!(
                rank < values.num_vals(),
                "packed source rank exceeds its column values"
            );
            values.get_val(rank)
        }))
    }
}

#[cfg(all(test, debug_assertions))]
mod sparse_reference_tests {
    use super::*;
    use crate::{ColumnarReader, ColumnarWriter, DynamicColumn, RowAddr, ShuffleMergeOrder};

    #[test]
    #[should_panic(expected = "packed source rank exceeds its column values")]
    fn rejects_out_of_range_packed_rank() {
        let mut writer = ColumnarWriter::default();
        writer.record_numerical(0, "value", u64::MAX);
        let mut bytes = Vec::new();
        writer.serialize(128, None, &mut bytes).unwrap();
        let reader = ColumnarReader::open(bytes).unwrap();
        let DynamicColumn::U64(column) = reader
            .read_columns("value")
            .unwrap()
            .remove(0)
            .open()
            .unwrap()
        else {
            panic!();
        };
        let columns = [Some(column.values)];
        let order = ShuffleMergeOrder::for_test(
            &[128],
            vec![RowAddr {
                segment_ord: 0,
                row_id: 0,
            }],
        );
        let values = SparseMergedColumnValues {
            packed: &[1],
            column_values: &columns,
            order: &order,
        };
        let _ = values.boxed_iter().next();
    }

    #[test]
    #[should_panic(expected = "internal error: entered unreachable code")]
    fn sparse_replay_rejects_stack_order() {
        let indexes = [ColumnIndex::Empty { num_docs: 0 }];
        let columns = [None];
        let order = crate::StackMergeOrder::stack_for_test(&[0]).into();
        let values = MergedColumnValues::<u64> {
            column_indexes: &indexes,
            column_values: &columns,
            merge_row_order: &order,
        };
        let _ = values.with_sparse_rows(&[]);
    }
}
