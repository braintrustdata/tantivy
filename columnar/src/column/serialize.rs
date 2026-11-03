use std::io;
use std::io::Write;
use std::sync::Arc;

use common::file_slice::FileSlice;
use common::{BinarySerializable, ChunkedBytes, OwnedBytes};
use sstable::Dictionary;

use crate::column::{BytesColumn, Column};
use crate::column_index::{serialize_column_index, SerializableColumnIndex};
use crate::column_values::{
    load_u64_based_column_values, load_u64_based_column_values_from_chunks,
    serialize_column_values_u128, serialize_u64_based_column_values, CodecType,
    MonotonicallyMappableToU128, MonotonicallyMappableToU64,
};
use crate::iterable::Iterable;
use crate::StrColumn;

pub fn serialize_column_mappable_to_u128<T: MonotonicallyMappableToU128>(
    column_index: SerializableColumnIndex<'_>,
    iterable: &dyn Iterable<T>,
    output: &mut impl Write,
) -> io::Result<()> {
    let column_index_num_bytes = serialize_column_index(column_index, output)?;
    serialize_column_values_u128(iterable, output)?;
    output.write_all(&column_index_num_bytes.to_le_bytes())?;
    Ok(())
}

pub fn serialize_column_mappable_to_u64<T: MonotonicallyMappableToU64>(
    column_index: SerializableColumnIndex<'_>,
    column_values: &impl Iterable<T>,
    output: &mut impl Write,
) -> io::Result<()> {
    let column_index_num_bytes = serialize_column_index(column_index, output)?;
    serialize_u64_based_column_values(
        column_values,
        &[CodecType::Bitpacked, CodecType::BlockwiseLinear],
        output,
    )?;
    output.write_all(&column_index_num_bytes.to_le_bytes())?;
    Ok(())
}

pub fn open_column_u64<T: MonotonicallyMappableToU64>(bytes: OwnedBytes) -> io::Result<Column<T>> {
    let (body, column_index_num_bytes_payload) = bytes.rsplit(4);
    let column_index_num_bytes = u32::from_le_bytes(
        column_index_num_bytes_payload
            .as_slice()
            .try_into()
            .unwrap(),
    );
    let (column_index_data, column_values_data) = body.split(column_index_num_bytes as usize);
    let column_index = crate::column_index::open_column_index(column_index_data.into())?;
    let column_values = load_u64_based_column_values(column_values_data)?;
    Ok(Column {
        index: column_index,
        values: column_values,
    })
}

/// Same as [`open_column_u64`], but keeps the values in the chunks they were read as.
pub fn open_column_u64_from_chunks<T: MonotonicallyMappableToU64>(
    bytes: ChunkedBytes,
) -> io::Result<Column<T>> {
    let invalid = || io::Error::new(io::ErrorKind::InvalidData, "column is too short");
    let body_len = bytes.len().checked_sub(4).ok_or_else(invalid)?;
    let column_index_num_bytes =
        u32::deserialize(&mut bytes.slice(body_len..bytes.len()).into_reader())? as usize;
    if column_index_num_bytes > body_len {
        return Err(invalid());
    }
    let column_index =
        crate::column_index::open_column_index(bytes.slice(0..column_index_num_bytes))?;
    let column_values =
        load_u64_based_column_values_from_chunks(bytes.slice(column_index_num_bytes..body_len))?;
    Ok(Column {
        index: column_index,
        values: column_values,
    })
}

pub fn open_column_u128<T: MonotonicallyMappableToU128>(
    bytes: OwnedBytes,
) -> io::Result<Column<T>> {
    let (body, column_index_num_bytes_payload) = bytes.rsplit(4);
    let column_index_num_bytes = u32::from_le_bytes(
        column_index_num_bytes_payload
            .as_slice()
            .try_into()
            .unwrap(),
    );
    let (column_index_data, column_values_data) = body.split(column_index_num_bytes as usize);
    let column_index = crate::column_index::open_column_index(column_index_data.into())?;
    let column_values = crate::column_values::open_u128_mapped(column_values_data)?;
    Ok(Column {
        index: column_index,
        values: column_values,
    })
}

/// Open the column as u64.
///
/// See [`open_u128_as_compact_u64`] for more details.
pub fn open_column_u128_as_compact_u64(bytes: OwnedBytes) -> io::Result<Column<u64>> {
    let (body, column_index_num_bytes_payload) = bytes.rsplit(4);
    let column_index_num_bytes = u32::from_le_bytes(
        column_index_num_bytes_payload
            .as_slice()
            .try_into()
            .unwrap(),
    );
    let (column_index_data, column_values_data) = body.split(column_index_num_bytes as usize);
    let column_index = crate::column_index::open_column_index(column_index_data.into())?;
    let column_values = crate::column_values::open_u128_as_compact_u64(column_values_data)?;
    Ok(Column {
        index: column_index,
        values: column_values,
    })
}

pub fn open_column_bytes(data: OwnedBytes) -> io::Result<BytesColumn> {
    let (body, dictionary_len_bytes) = data.rsplit(4);
    let dictionary_len = u32::from_le_bytes(dictionary_len_bytes.as_slice().try_into().unwrap());
    let (dictionary_bytes, column_bytes) = body.split(dictionary_len as usize);
    let dictionary = Arc::new(Dictionary::from_bytes(dictionary_bytes)?);
    let term_ord_column = crate::column::open_column_u64::<u64>(column_bytes)?;
    Ok(BytesColumn {
        dictionary,
        term_ord_column,
    })
}

/// Same as [`open_column_bytes`], but keeps the term ordinals in the chunks they were read as.
pub fn open_column_bytes_from_slice(file_slice: FileSlice) -> io::Result<BytesColumn> {
    let (body, dictionary_len) = file_slice.split_from_end(4);
    let dictionary_len = u32::deserialize(&mut dictionary_len.read_bytes()?)? as usize;
    let (dictionary, term_ord_column) = body.split(dictionary_len);
    Ok(BytesColumn {
        dictionary: Arc::new(Dictionary::from_bytes(dictionary.read_bytes()?)?),
        term_ord_column: open_column_u64_from_chunks(term_ord_column.read_chunks()?)?,
    })
}

pub fn open_column_str(data: OwnedBytes) -> io::Result<StrColumn> {
    let bytes_column = open_column_bytes(data)?;
    Ok(StrColumn::wrap(bytes_column))
}
