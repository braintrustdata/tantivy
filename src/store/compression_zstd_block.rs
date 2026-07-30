use std::fs::{File, OpenOptions};
use std::io;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use zstd::bulk::{compress_to_buffer, decompress_to_buffer, Compressor, Decompressor};
use zstd::DEFAULT_COMPRESSION_LEVEL;

/// Env var pointing to a raw zstd dictionary file. When set, doc store blocks are
/// compressed/decompressed against this dictionary. Experimental: loaded once per process.
const ZSTD_DICT_ENV_VAR: &str = "TANTIVY_ZSTD_DICT_PATH";

/// Env var pointing to a file that dictionary-usage traces are appended to.
/// Experimental: opened once per process, one line written per compress/decompress call
/// that actually used the dictionary.
const ZSTD_DICT_TRACE_ENV_VAR: &str = "TANTIVY_ZSTD_DICT_TRACE_PATH";

fn zstd_dictionary() -> Option<&'static [u8]> {
    static DICT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    DICT.get_or_init(|| {
        let path = std::env::var(ZSTD_DICT_ENV_VAR).ok()?;
        match std::fs::read(&path) {
            Ok(bytes) => {
                info!("Loaded zstd doc store dictionary from {path:?} ({} bytes)", bytes.len());
                Some(bytes)
            }
            Err(err) => {
                warn!("Failed to read zstd dictionary at {path:?} from {ZSTD_DICT_ENV_VAR}: {err}");
                None
            }
        }
    })
    .as_deref()
}

fn zstd_dict_trace_file() -> Option<&'static Mutex<File>> {
    static TRACE_FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    TRACE_FILE
        .get_or_init(|| {
            let path = std::env::var(ZSTD_DICT_TRACE_ENV_VAR).ok()?;
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(file) => Some(Mutex::new(file)),
                Err(err) => {
                    warn!(
                        "Failed to open zstd dictionary trace file at {path:?} from \
                         {ZSTD_DICT_TRACE_ENV_VAR}: {err}"
                    );
                    None
                }
            }
        })
        .as_ref()
}

fn trace_dict_usage(message: &str) {
    let Some(file) = zstd_dict_trace_file() else {
        return;
    };
    let timestamp_micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    if let Ok(mut file) = file.lock() {
        let _ = writeln!(file, "[{timestamp_micros}] {message}");
    }
}

#[inline]
pub fn compress(
    uncompressed: &[u8],
    compressed: &mut Vec<u8>,
    compression_level: Option<i32>,
) -> io::Result<()> {
    let count_size = std::mem::size_of::<u32>();
    let max_size = zstd::zstd_safe::compress_bound(uncompressed.len()) + count_size;

    compressed.clear();
    compressed.resize(max_size, 0);

    let level = compression_level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let compressed_size = if let Some(dict) = zstd_dictionary() {
        trace_dict_usage(&format!(
            "compress: {} bytes uncompressed",
            uncompressed.len()
        ));
        Compressor::with_dictionary(level, dict)?
            .compress_to_buffer(uncompressed, &mut compressed[count_size..])?
    } else {
        compress_to_buffer(uncompressed, &mut compressed[count_size..], level)?
    };

    compressed[0..count_size].copy_from_slice(&(uncompressed.len() as u32).to_le_bytes());
    compressed.resize(compressed_size + count_size, 0);

    Ok(())
}

#[inline]
pub fn decompress(compressed: &[u8], decompressed: &mut Vec<u8>) -> io::Result<()> {
    let count_size = std::mem::size_of::<u32>();
    let uncompressed_size = u32::from_le_bytes(
        compressed
            .get(..count_size)
            .ok_or(io::ErrorKind::InvalidData)?
            .try_into()
            .unwrap(),
    ) as usize;

    decompressed.clear();
    decompressed.resize(uncompressed_size, 0);

    let decompressed_size = if let Some(dict) = zstd_dictionary() {
        trace_dict_usage(&format!("decompress: {} bytes compressed", compressed.len()));
        Decompressor::with_dictionary(dict)?.decompress_to_buffer(&compressed[count_size..], decompressed)?
    } else {
        decompress_to_buffer(&compressed[count_size..], decompressed)?
    };

    if decompressed_size != uncompressed_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "doc store block not completely decompressed, data corruption".to_string(),
        ));
    }

    Ok(())
}
