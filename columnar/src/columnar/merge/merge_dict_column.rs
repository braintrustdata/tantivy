use std::io::{self, Write};

use common::{BitSet, CountingWriter, ReadOnlyBitSet};
use sstable::{SSTable, Streamer, TermOrdinal, VoidSSTable};

use super::sparse::{PackedRowIds, SparseMergeContext};
use super::term_merger::TermMerger;
use crate::column::serialize_column_mappable_to_u64;
use crate::column_index::{SerializableColumnIndex, Set};
use crate::iterable::Iterable;
use crate::{BytesColumn, ColumnIndex, MergeRowOrder, ShuffleMergeOrder};

// Serialize [Dictionary, Column, dictionary num bytes U32::LE]
// Column: [Column Index, Column Values, column index num bytes U32::LE]
pub fn merge_bytes_or_str_column(
    column_index: SerializableColumnIndex<'_>,
    bytes_columns: &[Option<BytesColumn>],
    merge_row_order: &MergeRowOrder,
    num_rows_per_segment: &[u32],
    sparse_context: &mut SparseMergeContext,
    output: &mut impl Write,
) -> io::Result<()> {
    // Serialize dict and generate mapping for values
    let mut output = CountingWriter::wrap(output);
    // TODO !!! Remove useless terms.
    let term_ord_mapping = serialize_merged_dict(bytes_columns, merge_row_order, &mut output)?;
    let dictionary_num_bytes: u32 = output.written_bytes() as u32;
    let output = output.finish();
    if let Some(packed) = prepare_sparse_dict(
        sparse_context,
        &column_index,
        bytes_columns,
        &term_ord_mapping,
        merge_row_order,
        num_rows_per_segment,
    ) {
        let sparse_index = SerializableColumnIndex::Optional {
            non_null_row_ids: Box::new(PackedRowIds(packed)),
            num_rows: merge_row_order.num_rows(),
        };
        serialize_column_mappable_to_u64(sparse_index, &PackedOrdinals(packed), output)?;
    } else {
        let remapped_term_ordinals_values = RemappedTermOrdinalsValues {
            bytes_columns,
            term_ord_mapping: &term_ord_mapping,
            merge_row_order,
        };
        serialize_column_mappable_to_u64(column_index, &remapped_term_ordinals_values, output)?;
    }
    output.write_all(&dictionary_num_bytes.to_le_bytes())?;
    Ok(())
}

struct RemappedTermOrdinalsValues<'a> {
    bytes_columns: &'a [Option<BytesColumn>],
    term_ord_mapping: &'a TermOrdinalMapping,
    merge_row_order: &'a MergeRowOrder,
}

impl<'a> Iterable for RemappedTermOrdinalsValues<'a> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self.merge_row_order {
            MergeRowOrder::Stack(_) => self.boxed_iter_stacked(),
            MergeRowOrder::Shuffled(shuffle_merge_order) => {
                self.boxed_iter_shuffled(shuffle_merge_order)
            }
        }
    }
}

impl<'a> RemappedTermOrdinalsValues<'a> {
    fn boxed_iter_stacked(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        let iter = self
            .bytes_columns
            .iter()
            .enumerate()
            .flat_map(|(seg_ord, bytes_column_opt)| {
                let bytes_column = bytes_column_opt.as_ref()?;
                Some((seg_ord, bytes_column))
            })
            .flat_map(move |(seg_ord, bytes_column)| {
                let term_ord_after_merge_mapping =
                    self.term_ord_mapping.get_segment(seg_ord as u32);
                bytes_column
                    .ords()
                    .values
                    .iter()
                    .map(move |term_ord| term_ord_after_merge_mapping[term_ord as usize])
            });
        Box::new(iter)
    }

    fn boxed_iter_shuffled<'b>(
        &'b self,
        shuffle_merge_order: &'b ShuffleMergeOrder,
    ) -> Box<dyn Iterator<Item = u64> + 'b> {
        let mut cursors: Vec<_> = self
            .bytes_columns
            .iter()
            .map(|column| match &column.as_ref()?.ords().index {
                ColumnIndex::Optional(index) => index.cursor_for_sparse_blocks(),
                _ => None,
            })
            .collect();
        let has_multivalued = self
            .bytes_columns
            .iter()
            .flatten()
            .any(|column| matches!(&column.ords().index, ColumnIndex::Multivalued(_)));
        if !has_multivalued {
            return Box::new(shuffle_merge_order.iter_new_to_old_row_addrs().filter_map(
                move |addr| {
                    let segment = addr.segment_ord as usize;
                    let column = self.bytes_columns[segment].as_ref()?;
                    let rank = match &mut cursors[segment] {
                        Some(cursor) => cursor.rank_if_exists(addr.row_id)?,
                        None => match &column.ords().index {
                            ColumnIndex::Full => addr.row_id,
                            ColumnIndex::Optional(index) => index.rank_if_exists(addr.row_id)?,
                            ColumnIndex::Empty { .. } => return None,
                            ColumnIndex::Multivalued(_) => unreachable!(),
                        },
                    };
                    let old_ord = column.ords().values.get_val(rank);
                    Some(self.term_ord_mapping.get_segment(addr.segment_ord)[old_ord as usize])
                },
            ));
        }
        Box::new(
            shuffle_merge_order
                .iter_new_to_old_row_addrs()
                .filter_map(move |addr| {
                    let segment = addr.segment_ord as usize;
                    let column = self.bytes_columns[segment].as_ref()?;
                    let range = match &mut cursors[segment] {
                        Some(cursor) => cursor
                            .rank_if_exists(addr.row_id)
                            .map(|rank| rank..rank + 1)
                            .unwrap_or(0..0),
                        None => column.ords().index.value_row_ids(addr.row_id),
                    };
                    if range.is_empty() {
                        return None;
                    }
                    Some((
                        range,
                        column,
                        self.term_ord_mapping.get_segment(addr.segment_ord),
                    ))
                })
                .flat_map(|(range, column, mapping)| {
                    range.map(move |rank| mapping[column.ords().values.get_val(rank) as usize])
                }),
        )
    }
}

fn compute_term_bitset(column: &BytesColumn, row_bitset: &ReadOnlyBitSet) -> BitSet {
    let num_terms = column.dictionary().num_terms();
    let mut term_bitset = BitSet::with_max_value(num_terms as u32);
    if let ColumnIndex::Optional(index) = &column.ords().index {
        if index.num_non_nulls() as usize <= row_bitset.len() {
            for (rank, row) in index.iter_rows().enumerate() {
                if row_bitset.contains(row) {
                    term_bitset.insert(column.ords().values.get_val(rank as u32) as u32);
                }
            }
        } else {
            let mut cursor = index.cursor();
            for row in row_bitset.iter() {
                if let Some(rank) = cursor.rank_if_exists(row) {
                    term_bitset.insert(column.ords().values.get_val(rank) as u32);
                }
            }
        }
        return term_bitset;
    }
    for row_id in row_bitset.iter() {
        for term_ord in column.term_ord_column.values_for_doc(row_id) {
            term_bitset.insert(term_ord as u32);
        }
    }
    term_bitset
}

fn is_term_present(bitsets: &[Option<BitSet>], term_merger: &TermMerger) -> bool {
    for (segment_ord, from_term_ord) in term_merger.matching_segments() {
        if let Some(bitset) = bitsets[segment_ord].as_ref() {
            if bitset.contains(from_term_ord as u32) {
                return true;
            }
        } else {
            return true;
        }
    }
    false
}

fn serialize_merged_dict(
    bytes_columns: &[Option<BytesColumn>],
    merge_row_order: &MergeRowOrder,
    output: &mut impl Write,
) -> io::Result<TermOrdinalMapping> {
    let mut term_ord_mapping = TermOrdinalMapping::default();

    let mut field_term_streams = Vec::new();
    for column_opt in bytes_columns.iter() {
        if let Some(column) = column_opt {
            term_ord_mapping.add_segment(column.dictionary.num_terms());
            let terms: Streamer<VoidSSTable> = column.dictionary.stream()?;
            field_term_streams.push(terms);
        } else {
            term_ord_mapping.add_segment(0);
            field_term_streams.push(Streamer::empty());
        }
    }

    let mut merged_terms = TermMerger::new(field_term_streams);
    let mut sstable_builder = sstable::VoidSSTable::writer(output);

    match merge_row_order {
        MergeRowOrder::Stack(_) => {
            let mut current_term_ord = 0;
            while merged_terms.advance() {
                let term_bytes: &[u8] = merged_terms.key();
                sstable_builder.insert(term_bytes, &())?;
                for (segment_ord, from_term_ord) in merged_terms.matching_segments() {
                    term_ord_mapping.register_from_to(segment_ord, from_term_ord, current_term_ord);
                }
                current_term_ord += 1;
            }
            term_ord_mapping.num_terms = current_term_ord;
            sstable_builder.finish()?;
        }
        MergeRowOrder::Shuffled(shuffle_merge_order) => {
            assert_eq!(shuffle_merge_order.alive_bitsets.len(), bytes_columns.len());
            let mut term_bitsets: Vec<Option<BitSet>> = Vec::with_capacity(bytes_columns.len());
            for (alive_bitset_opt, bytes_column_opt) in shuffle_merge_order
                .alive_bitsets
                .iter()
                .zip(bytes_columns.iter())
            {
                match (alive_bitset_opt, bytes_column_opt) {
                    (Some(alive_bitset), Some(bytes_column)) => {
                        let term_bitset = compute_term_bitset(bytes_column, alive_bitset);
                        term_bitsets.push(Some(term_bitset));
                    }
                    _ => {
                        term_bitsets.push(None);
                    }
                }
            }
            let mut current_term_ord = 0;
            while merged_terms.advance() {
                let term_bytes: &[u8] = merged_terms.key();
                if !is_term_present(&term_bitsets[..], &merged_terms) {
                    continue;
                }
                sstable_builder.insert(term_bytes, &())?;
                for (segment_ord, from_term_ord) in merged_terms.matching_segments() {
                    term_ord_mapping.register_from_to(segment_ord, from_term_ord, current_term_ord);
                }
                current_term_ord += 1;
            }
            term_ord_mapping.num_terms = current_term_ord;
            sstable_builder.finish()?;
        }
    }
    Ok(term_ord_mapping)
}

#[derive(Default, Debug)]
struct TermOrdinalMapping {
    num_terms: u64,
    per_segment_new_term_ordinals: Vec<Vec<TermOrdinal>>,
}

impl TermOrdinalMapping {
    fn add_segment(&mut self, max_term_ord: usize) {
        self.per_segment_new_term_ordinals
            .push(vec![TermOrdinal::default(); max_term_ord]);
    }

    fn register_from_to(&mut self, segment_ord: usize, from_ord: TermOrdinal, to_ord: TermOrdinal) {
        self.per_segment_new_term_ordinals[segment_ord][from_ord as usize] = to_ord;
    }

    fn get_segment(&self, segment_ord: u32) -> &[TermOrdinal] {
        &(self.per_segment_new_term_ordinals[segment_ord as usize])[..]
    }
}

fn prepare_sparse_dict<'a>(
    context: &'a mut SparseMergeContext,
    column_index: &SerializableColumnIndex<'_>,
    columns: &[Option<BytesColumn>],
    mapping: &TermOrdinalMapping,
    merge_order: &MergeRowOrder,
    num_rows_per_segment: &[u32],
) -> Option<&'a [u64]> {
    u32::try_from(mapping.num_terms).ok()?;
    context.prepare(
        column_index,
        columns
            .iter()
            .map(|column| column.as_ref().map(|column| &column.ords().index)),
        merge_order,
        num_rows_per_segment,
        |segment, rank| {
            let column = columns[segment].as_ref().unwrap();
            let old_ord = column.ords().values.get_val(rank);
            let new_ord = mapping.get_segment(segment as u32)[old_ord as usize];
            debug_assert!(
                new_ord < mapping.num_terms,
                "remapped ordinal must reference the merged dictionary"
            );
            u32::try_from(new_ord).unwrap()
        },
    )
}

struct PackedOrdinals<'a>(&'a [u64]);

impl Iterable for PackedOrdinals<'_> {
    fn boxed_iter(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        Box::new(self.0.iter().map(|&packed| u64::from(packed as u32)))
    }
}

#[cfg(test)]
mod sparse_dict_tests {
    use super::*;
    use crate::{ColumnarReader, ColumnarWriter, DynamicColumn, RowAddr};

    #[test]
    fn packed_ordinals_and_overflow_guard() {
        let packed = [u64::from(u32::MAX), (7u64 << 32) | u64::from(u32::MAX)];
        assert_eq!(
            PackedOrdinals(&packed).boxed_iter().collect::<Vec<_>>(),
            [u64::from(u32::MAX); 2]
        );
        let mut writer = ColumnarWriter::default();
        writer.record_str(0, "str", "value");
        let mut bytes = Vec::new();
        writer.serialize(128, None, &mut bytes).unwrap();
        let reader = ColumnarReader::open(bytes).unwrap();
        let DynamicColumn::Str(column) = reader
            .read_columns("str")
            .unwrap()
            .remove(0)
            .open()
            .unwrap()
        else {
            panic!();
        };
        let columns = [Some(column.into())];
        let index = SerializableColumnIndex::Optional {
            non_null_row_ids: Box::new(0..1),
            num_rows: 128,
        };
        let order = ShuffleMergeOrder::for_test(
            &[128],
            (0..128)
                .map(|row_id| RowAddr {
                    segment_ord: 0,
                    row_id,
                })
                .collect(),
        )
        .into();
        let mut context = SparseMergeContext::default();
        let mut mapping = TermOrdinalMapping {
            num_terms: u64::from(u32::MAX) + 1,
            per_segment_new_term_ordinals: vec![vec![0]],
        };
        assert!(
            prepare_sparse_dict(&mut context, &index, &columns, &mapping, &order, &[128]).is_none()
        );
        mapping.num_terms = 1;
        assert!(
            prepare_sparse_dict(&mut context, &index, &columns, &mapping, &order, &[128]).is_some()
        );
    }
}
