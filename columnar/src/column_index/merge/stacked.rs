use std::iter;

use crate::column_index::SerializableColumnIndex;
use crate::iterable::Iterable;
use crate::{Cardinality, ColumnIndex, RowId, StackMergeOrder};

/// Simple case:
/// The new mapping just consists in stacking the different column indexes.
///
/// There are no sort nor deletes involved.
pub fn merge_column_index_stacked<'a>(
    columns: &'a [ColumnIndex],
    cardinality_after_merge: Cardinality,
    stack_merge_order: &'a StackMergeOrder,
) -> SerializableColumnIndex<'a> {
    match cardinality_after_merge {
        Cardinality::Full => SerializableColumnIndex::Full,
        Cardinality::Optional => SerializableColumnIndex::Optional {
            non_null_row_ids: Box::new(StackedOptionalIndex {
                columns,
                stack_merge_order,
            }),
            num_rows: stack_merge_order.num_rows(),
        },
        Cardinality::Multivalued => {
            let stacked_multivalued_index = StackedMultivaluedIndex {
                columns,
                stack_merge_order,
            };
            SerializableColumnIndex::Multivalued(Box::new(stacked_multivalued_index))
        }
    }
}

struct StackedOptionalIndex<'a> {
    columns: &'a [ColumnIndex],
    stack_merge_order: &'a StackMergeOrder,
}

impl<'a> Iterable<RowId> for StackedOptionalIndex<'a> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = RowId> + 'a> {
        Box::new(
            self.columns
                .iter()
                .enumerate()
                .flat_map(|(columnar_id, column_index_opt)| {
                    let columnar_row_range = self.stack_merge_order.columnar_range(columnar_id);
                    let rows_it: Box<dyn Iterator<Item = RowId>> = match column_index_opt {
                        ColumnIndex::Full => Box::new(columnar_row_range),
                        ColumnIndex::Optional(optional_index) => Box::new(
                            optional_index
                                .iter_rows()
                                .map(move |row_id: RowId| columnar_row_range.start + row_id),
                        ),
                        ColumnIndex::Multivalued(_) => {
                            panic!("No multivalued index is allowed when stacking column index");
                        }
                        ColumnIndex::Empty { .. } => Box::new(std::iter::empty()),
                    };
                    rows_it
                }),
        )
    }
}

#[derive(Clone, Copy)]
struct StackedMultivaluedIndex<'a> {
    columns: &'a [ColumnIndex],
    stack_merge_order: &'a StackMergeOrder,
}

fn convert_column_opt_to_multivalued_index<'a>(
    column_index_opt: &'a ColumnIndex,
    num_rows: RowId,
) -> Box<dyn Iterator<Item = RowId> + 'a> {
    match column_index_opt {
        ColumnIndex::Empty { .. } => Box::new(iter::repeat(0u32).take(num_rows as usize + 1)),
        ColumnIndex::Full => Box::new(0..num_rows + 1),
        ColumnIndex::Optional(optional_index) => {
            let mut rows = optional_index.iter_rows();
            let mut next = rows.next();
            let mut rank = 0;
            Box::new(
                (0..num_rows)
                    .map(move |row| {
                        let start_offset = rank;
                        if next == Some(row) {
                            rank += 1;
                            next = rows.next();
                        }
                        start_offset
                    })
                    .chain(std::iter::once(optional_index.num_non_nulls())),
            )
        }
        ColumnIndex::Multivalued(multivalued_index) => multivalued_index.start_index_column.iter(),
    }
}

impl<'a> Iterable<RowId> for StackedMultivaluedIndex<'a> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = RowId> + '_> {
        let multivalued_indexes =
            self.columns
                .iter()
                .enumerate()
                .map(|(columnar_id, column_opt)| {
                    let num_rows =
                        self.stack_merge_order.columnar_range(columnar_id).len() as RowId;
                    convert_column_opt_to_multivalued_index(column_opt, num_rows)
                });
        stack_multivalued_indexes(multivalued_indexes)
    }
}

// Refactor me
fn stack_multivalued_indexes<'a>(
    mut multivalued_indexes: impl Iterator<Item = Box<dyn Iterator<Item = RowId> + 'a>> + 'a,
) -> Box<dyn Iterator<Item = RowId> + 'a> {
    let mut offset = 0;
    let mut last_row_id = 0;
    let mut current_it = multivalued_indexes.next();
    Box::new(std::iter::from_fn(move || loop {
        if let Some(row_id) = current_it.as_mut()?.next() {
            last_row_id = offset + row_id;
            return Some(last_row_id);
        }
        offset = last_row_id;
        loop {
            current_it = multivalued_indexes.next();
            if current_it.as_mut()?.next().is_some() {
                break;
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use crate::RowId;

    fn it<'a>(row_ids: &'a [RowId]) -> Box<dyn Iterator<Item = RowId> + 'a> {
        Box::new(row_ids.iter().copied())
    }

    #[test]
    fn test_stack() {
        let columns = [
            it(&[0u32, 0u32]),
            it(&[0u32, 1u32, 1u32, 4u32]),
            it(&[0u32, 3u32, 5u32]),
            it(&[0u32, 4u32]),
        ]
        .into_iter();
        let start_offsets: Vec<RowId> = super::stack_multivalued_indexes(columns).collect();
        assert_eq!(start_offsets, &[0, 0, 1, 1, 4, 7, 9, 13]);
    }
}

#[cfg(test)]
mod sparse_offset_tests {
    use proptest::prelude::*;

    use super::*;
    use crate::column_index::{OptionalIndex, Set};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn offsets_match_rank(num_docs in 0u32..2048, rows in proptest::collection::btree_set(0u32..2048, 0..256)) {
            let rows: Vec<_> = rows.into_iter().filter(|&row| row < num_docs).collect();
            let index = OptionalIndex::for_test(num_docs, &rows);
            let column = ColumnIndex::Optional(index.clone());
            let actual: Vec<_> = convert_column_opt_to_multivalued_index(&column, num_docs).collect();
            let expected: Vec<_> = (0..=num_docs).map(|row| index.rank(row)).collect();
            prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn empty_and_block_boundaries_keep_sentinel() {
        for num_docs in [0, 1, 65536, 131080] {
            let rows: Vec<_> = [0, 65535, 65536, 131079]
                .into_iter()
                .filter(|&row| row < num_docs)
                .collect();
            let index = OptionalIndex::for_test(num_docs, &rows);
            let column = ColumnIndex::Optional(index.clone());
            let actual: Vec<_> =
                convert_column_opt_to_multivalued_index(&column, num_docs).collect();
            assert_eq!(actual.len(), num_docs as usize + 1);
            assert_eq!(
                actual,
                (0..=num_docs)
                    .map(|row| index.rank(row))
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[cfg(test)]
mod coverage_boundary_tests {
    use super::*;

    #[test]
    fn stacked_offsets_skip_empty_iterators_and_preserve_sentinel() {
        for inputs in [
            vec![],
            vec![vec![]],
            vec![
                vec![0u32, 2],
                vec![],
                vec![],
                vec![0],
                vec![0, 0, 3],
                vec![],
            ],
        ] {
            let expected = if inputs.len() > 1 {
                vec![0, 2, 2, 5]
            } else {
                vec![]
            };
            let mut merged = stack_multivalued_indexes(
                inputs
                    .iter()
                    .map(|v| Box::new(v.iter().copied()) as Box<dyn Iterator<Item = u32>>),
            );
            assert_eq!(merged.by_ref().collect::<Vec<_>>(), expected);
            assert_eq!(merged.next(), None);
        }
    }
}
