use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::sync::Mutex;

use once_cell::sync::OnceCell;
use serde_json::json;

const TRACE_PATH_ENV: &str = "TANTIVY_STORE_IO_TRACE_PATH";
const PREVIEW_LEN: usize = 16;

static TRACE_FILE: OnceCell<Mutex<File>> = OnceCell::new();

#[derive(Clone, Copy)]
pub(crate) enum StoreIoOperation {
    Read,
    Write,
}

impl StoreIoOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

pub(crate) fn record(op: StoreIoOperation, offset: usize, data: &[u8]) -> io::Result<()> {
    let Some(trace_file) = trace_file()? else {
        return Ok(());
    };

    let line = json!({
        "op": op.as_str(),
        "offset": offset,
        "length": data.len(),
        "first_16_hex": hex_preview(data.iter().take(PREVIEW_LEN).copied()),
        "last_16_hex": hex_preview(data.iter().skip(data.len().saturating_sub(PREVIEW_LEN)).copied()),
    });

    let mut trace_file = trace_file
        .lock()
        .map_err(|_| io::Error::new(io::ErrorKind::Other, "store IO trace lock poisoned"))?;
    serde_json::to_writer(&mut *trace_file, &line)?;
    trace_file.write_all(b"\n")?;
    Ok(())
}

fn trace_file() -> io::Result<Option<&'static Mutex<File>>> {
    let Some(path) = std::env::var_os(TRACE_PATH_ENV) else {
        return Ok(None);
    };

    TRACE_FILE
        .get_or_try_init(|| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map(Mutex::new)
        })
        .map(Some)
}

fn hex_preview(bytes: impl IntoIterator<Item = u8>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::new();
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
