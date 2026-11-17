#![no_main]

use std::collections::BTreeMap;

use libfuzzer_sys::{fuzz_target, Corpus};
use tantivy_columnar::{
    merge_columnar, merge_columnar_for_fuzzing, ColumnType, ColumnarReader, ColumnarWriter,
    DateTime, DynamicColumn, MergeRowOrder, RowAddr, ShuffleMergeOrder, StackMergeOrder,
};

const NAMES: [&str; 3] = ["number", "raw", "str"];
type Model = [BTreeMap<u32, Vec<u64>>; 3];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut n = self.0;
        n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        n = (n ^ (n >> 27)).wrapping_mul(0x94d049bb133111eb);
        n ^ (n >> 31)
    }
}

fn float(n: u64) -> f64 {
    if n & 15 == 0 {
        -0.0
    } else {
        (n as i32) as f64 / 4.0 + 0.125
    }
}

fn term(n: u64) -> String {
    format!("t{n:016x}")
}

fn source(
    data: &[u8],
    segment: usize,
    rows: u32,
    kind: ColumnType,
    rng: &mut Rng,
) -> (ColumnarReader, Model) {
    let mut writer = ColumnarWriter::default();
    let mut model: Model = std::array::from_fn(|_| BTreeMap::new());
    for col in 0..3 {
        let shape = data[8 + segment * 3 + col] % 8;
        if shape == 0 {
            continue;
        }
        writer.record_column_type(
            NAMES[col],
            [kind, ColumnType::Bytes, ColumnType::Str][col],
            false,
        );
        let stride = match shape {
            1 => 8,
            2 => 9,
            3 => 100,
            5 => 20,
            _ => 1,
        };
        let offset = u32::from(data[20 + col]) % stride;
        for row in 0..rows {
            let present = match shape {
                4 => true,
                6 if rows > 5120 => row % 65536 < 5120,
                6 => row % 4 != 0,
                7 => (rng.next() & 63) == 0,
                _ => row % stride == offset,
            };
            if !present {
                continue;
            }
            let count = if shape == 5 { 4 + data[31] % 5 } else { 1 };
            let values = model[col].entry(row).or_default();
            for value in 0..count {
                let mut key = rng.next();
                if col > 0 {
                    key = match data[24] % 3 {
                        0 => key & 15,
                        1 => key & 2047,
                        _ => ((segment as u64) << 48) | (u64::from(row) << 8) | u64::from(value),
                    };
                }
                values.push(key);
                match col {
                    1 => writer.record_bytes(row, NAMES[col], &key.to_le_bytes()),
                    2 => writer.record_str(row, NAMES[col], &term(key)),
                    _ => match kind {
                        ColumnType::U64 => writer.record_numerical(row, NAMES[col], key),
                        ColumnType::I64 => writer.record_numerical(row, NAMES[col], key as i64),
                        ColumnType::F64 => writer.record_numerical(row, NAMES[col], float(key)),
                        ColumnType::Bool => writer.record_bool(row, NAMES[col], key & 1 != 0),
                        ColumnType::DateTime => writer.record_datetime(
                            row,
                            NAMES[col],
                            DateTime::from_timestamp_nanos(key as i64),
                        ),
                        _ => unreachable!(),
                    },
                }
            }
        }
    }
    let mut bytes = Vec::new();
    writer.serialize(rows, None, &mut bytes).unwrap();
    (ColumnarReader::open(bytes).unwrap(), model)
}

fn verify(bytes: Vec<u8>, addresses: &[RowAddr], models: &[Model]) {
    let reader = ColumnarReader::open(bytes).unwrap();
    assert_eq!(reader.num_rows() as usize, addresses.len());
    assert_eq!(reader.list_columns().unwrap().len(), 3);
    for (col, name) in NAMES.iter().enumerate() {
        let mut handles = reader.read_columns(name).unwrap();
        assert_eq!(handles.len(), 1);
        let column = handles.remove(0).open().unwrap();
        for (row, addr) in addresses.iter().enumerate() {
            let keys = models[addr.segment_ord as usize][col]
                .get(&addr.row_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let row = row as u32;
            match &column {
                DynamicColumn::U64(c) => {
                    assert_eq!(c.values_for_doc(row).collect::<Vec<_>>(), keys)
                }
                DynamicColumn::I64(c) => assert_eq!(
                    c.values_for_doc(row).collect::<Vec<_>>(),
                    keys.iter().map(|&k| k as i64).collect::<Vec<_>>()
                ),
                DynamicColumn::F64(c) => assert_eq!(
                    c.values_for_doc(row).map(f64::to_bits).collect::<Vec<_>>(),
                    keys.iter().map(|&k| float(k).to_bits()).collect::<Vec<_>>()
                ),
                DynamicColumn::Bool(c) => assert_eq!(
                    c.values_for_doc(row).collect::<Vec<_>>(),
                    keys.iter().map(|&k| k & 1 != 0).collect::<Vec<_>>()
                ),
                DynamicColumn::DateTime(c) => assert_eq!(
                    c.values_for_doc(row).collect::<Vec<_>>(),
                    keys.iter()
                        .map(|&k| DateTime::from_timestamp_nanos(k as i64))
                        .collect::<Vec<_>>()
                ),
                DynamicColumn::Bytes(c) => {
                    let actual = c
                        .term_ords(row)
                        .map(|ord| {
                            let mut value = Vec::new();
                            assert!(c.ord_to_bytes(ord, &mut value).unwrap());
                            value
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual,
                        keys.iter()
                            .map(|k| k.to_le_bytes().to_vec())
                            .collect::<Vec<_>>()
                    );
                }
                DynamicColumn::Str(c) => {
                    let actual = c
                        .term_ords(row)
                        .map(|ord| {
                            let mut value = String::new();
                            assert!(c.ord_to_str(ord, &mut value).unwrap());
                            value
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(actual, keys.iter().map(|&k| term(k)).collect::<Vec<_>>());
                }
                _ => unreachable!(),
            }
        }
    }
}

fuzz_target!(|data: &[u8]| -> Corpus {
    if data.len() < 40 {
        return Corpus::Reject;
    }
    let segments = usize::from(data[0] % 5);
    let kind = [
        ColumnType::U64,
        ColumnType::I64,
        ColumnType::F64,
        ColumnType::Bool,
        ColumnType::DateTime,
    ][usize::from(data[2] % 5)];
    let mut rng = Rng(u64::from_le_bytes(data[32..40].try_into().unwrap()));
    let sizes: Vec<_> = (0..segments)
        .map(|s| {
            if s == 0 && data[1] & 31 == 31 {
                [65535, 65536, 65537, 65540][usize::from(data[4] % 4)]
            } else {
                u32::from(data[4 + s])
            }
        })
        .collect();
    let (readers, models): (Vec<_>, Vec<_>) = sizes
        .iter()
        .enumerate()
        .map(|(s, &n)| source(data, s, n, kind, &mut rng))
        .unzip();
    let refs: Vec<_> = readers.iter().collect();
    let pattern = data[3] % 4;
    let mut addresses = Vec::new();
    for row in 0..sizes.iter().copied().max().unwrap_or(0) {
        for (segment, &size) in sizes.iter().enumerate() {
            if row >= size {
                continue;
            }
            let populated = models[segment][0].contains_key(&row);
            let keep = match data[23] % 8 {
                0 => true,
                1 => false,
                2 => row % 10 != 0,
                3 => row % 2 == 0,
                4 => populated,
                5 => row == u32::from(data[30]) % size,
                6 => row < 2,
                _ => !populated,
            };
            if pattern == 0 || keep {
                addresses.push(RowAddr {
                    segment_ord: segment as u32,
                    row_id: row,
                });
            }
        }
    }
    if pattern == 0 {
        addresses.sort_by_key(|addr| (addr.segment_ord, addr.row_id));
    } else if pattern == 2 {
        addresses.reverse();
    } else if pattern == 3 {
        for i in (1..addresses.len()).rev() {
            addresses.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
    }
    let order = || -> MergeRowOrder {
        if pattern == 0 {
            StackMergeOrder::stack(&refs).into()
        } else {
            ShuffleMergeOrder::for_test(&sizes, addresses.clone()).into()
        }
    };
    let required = [
        ("number".to_owned(), kind),
        ("raw".to_owned(), ColumnType::Bytes),
        ("str".to_owned(), ColumnType::Str),
    ];
    let mut fused = Vec::new();
    merge_columnar(&refs, &required, order(), &mut fused).unwrap();
    let mut streaming = Vec::new();
    merge_columnar_for_fuzzing(&refs, &required, order(), &mut streaming).unwrap();
    assert_eq!(fused, streaming);
    verify(fused, &addresses, &models);
    Corpus::Keep
});
