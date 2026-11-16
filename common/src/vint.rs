use std::io;
use std::io::{Read, Write};

use super::BinarySerializable;

/// Variable int serializes a u128 number
pub fn serialize_vint_u128(mut val: u128, output: &mut Vec<u8>) {
    loop {
        let next_byte: u8 = (val % 128u128) as u8;
        val /= 128u128;
        if val == 0 {
            output.push(next_byte | STOP_BIT);
            return;
        } else {
            output.push(next_byte);
        }
    }
}

///   Wrapper over a `u128` that serializes as a variable int.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VIntU128(pub u128);

impl BinarySerializable for VIntU128 {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        let mut buffer = vec![];
        serialize_vint_u128(self.0, &mut buffer);
        writer.write_all(&buffer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut bytes = reader.bytes();
        let mut result = 0u128;
        let mut shift = 0u64;
        loop {
            match bytes.next() {
                Some(Ok(b)) => {
                    result |= u128::from(b % 128u8) << shift;
                    if b >= STOP_BIT {
                        return Ok(VIntU128(result));
                    }
                    shift += 7;
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Reach end of buffer while reading VInt",
                    ));
                }
            }
        }
    }
}

///   Wrapper over a `u64` that serializes as a variable int.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VInt(pub u64);

const STOP_BIT: u8 = 128;

#[inline]
pub fn serialize_vint_u32(val: u32, buf: &mut [u8; 8]) -> &[u8] {
    const START_2: u64 = 1 << 7;
    const START_3: u64 = 1 << 14;
    const START_4: u64 = 1 << 21;
    const START_5: u64 = 1 << 28;

    const MASK_1: u64 = 127;
    const MASK_2: u64 = MASK_1 << 7;
    const MASK_3: u64 = MASK_2 << 7;
    const MASK_4: u64 = MASK_3 << 7;
    const MASK_5: u64 = MASK_4 << 7;

    let val = u64::from(val);
    const STOP_BIT: u64 = 128u64;
    let (res, num_bytes) = if val < START_2 {
        (val | STOP_BIT, 1)
    } else if val < START_3 {
        (
            (val & MASK_1) | ((val & MASK_2) << 1) | (STOP_BIT << (8)),
            2,
        )
    } else if val < START_4 {
        (
            (val & MASK_1) | ((val & MASK_2) << 1) | ((val & MASK_3) << 2) | (STOP_BIT << (8 * 2)),
            3,
        )
    } else if val < START_5 {
        (
            (val & MASK_1)
                | ((val & MASK_2) << 1)
                | ((val & MASK_3) << 2)
                | ((val & MASK_4) << 3)
                | (STOP_BIT << (8 * 3)),
            4,
        )
    } else {
        (
            (val & MASK_1)
                | ((val & MASK_2) << 1)
                | ((val & MASK_3) << 2)
                | ((val & MASK_4) << 3)
                | ((val & MASK_5) << 4)
                | (STOP_BIT << (8 * 4)),
            5,
        )
    };
    *buf = res.to_le_bytes();
    &buf[0..num_bytes]
}

/// Returns the number of bytes covered by a
/// serialized vint `u32`.
///
/// Expects a buffer data that starts
/// by the serialized `vint`, scans at most 5 bytes ahead until
/// it finds the vint final byte.
///
/// # May Panic
/// If the payload does not start by a valid `vint`
fn vint_len(data: &[u8]) -> usize {
    for (i, &val) in data.iter().enumerate().take(5) {
        if val >= STOP_BIT {
            return i + 1;
        }
    }
    panic!("Corrupted data. Invalid VInt 32");
}

/// Reads a vint `u32` from a buffer, and
/// consumes its payload data.
///
/// # Panics
///
/// If the buffer does not start by a valid
/// vint payload
pub fn read_u32_vint(data: &mut &[u8]) -> u32 {
    let (result, vlen) = read_u32_vint_no_advance(data);
    *data = &data[vlen..];
    result
}

pub fn read_u32_vint_no_advance(data: &[u8]) -> (u32, usize) {
    let vlen = vint_len(data);
    let mut result = 0u32;
    let mut shift = 0u64;
    for &b in &data[..vlen] {
        result |= u32::from(b & 127u8) << shift;
        shift += 7;
    }
    (result, vlen)
}
/// Write a `u32` as a vint payload.
pub fn write_u32_vint<W: io::Write>(val: u32, writer: &mut W) -> io::Result<()> {
    let mut buf = [0u8; 8];
    let data = serialize_vint_u32(val, &mut buf);
    writer.write_all(data)
}

/// Encodes a stop-bit VInt, speculatively writing ten bytes.
///
/// # Safety
/// `dst` must be valid for ten writable bytes. Only the returned prefix belongs
/// to the encoded value; the remaining bytes are overwritten as well.
#[inline]
pub unsafe fn encode_vint_branchless(v: u64, dst: *mut u8) -> usize {
    let bit_width = 64 - (v | 1).leading_zeros() as usize;
    let len = (bit_width * 9 + 64) >> 6;
    let lo_data = (v & 0x7f)
        | ((v & (0x7f << 7)) << 1)
        | ((v & (0x7f << 14)) << 2)
        | ((v & (0x7f << 21)) << 3)
        | ((v & (0x7f << 28)) << 4)
        | ((v & (0x7f << 35)) << 5)
        | ((v & (0x7f << 42)) << 6)
        | ((v & (0x7f << 49)) << 7);
    let hi_data = (((v >> 56) & 0x7f) | (((v >> 63) & 1) << 8)) as u16;
    let stop_lo = if len <= 8 {
        0x80u64 << ((len - 1) * 8)
    } else {
        0
    };
    let stop_hi = if len > 8 {
        0x80u16 << ((len - 9) * 8)
    } else {
        0
    };
    dst.cast::<u64>()
        .write_unaligned((lo_data | stop_lo).to_le());
    dst.add(8)
        .cast::<u16>()
        .write_unaligned((hi_data | stop_hi).to_le());
    len
}

/// Encodes a stop-bit VInt and advances past its logical bytes.
///
/// # Safety
/// `dst` must be valid for ten writable bytes, including speculative writes
/// beyond the returned pointer.
#[inline(always)]
pub unsafe fn write_vint_branchless(v: u64, dst: *mut u8) -> *mut u8 {
    dst.add(encode_vint_branchless(v, dst))
}

/// Encodes a stop-bit VInt with a one-byte fast path.
///
/// # Safety
/// `dst` must be valid for ten writable bytes, including speculative writes
/// beyond the returned pointer.
#[inline(always)]
pub unsafe fn write_vint_raw(v: u64, dst: *mut u8) -> *mut u8 {
    if v < u64::from(STOP_BIT) {
        dst.write(v as u8 | STOP_BIT);
        dst.add(1)
    } else {
        write_vint_branchless(v, dst)
    }
}

impl VInt {
    pub fn val(&self) -> u64 {
        self.0
    }

    pub fn deserialize_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
        VInt::deserialize(reader).map(|vint| vint.0)
    }

    #[inline]
    pub fn serialize_into_vec(&self, output: &mut Vec<u8>) {
        if self.0 < u64::from(STOP_BIT) {
            output.push(self.0 as u8 | STOP_BIT);
            return;
        }
        self.serialize_into_vec_slow(output);
    }

    #[cold]
    #[inline(never)]
    fn serialize_into_vec_slow(&self, output: &mut Vec<u8>) {
        if output.capacity() - output.len() < 10 {
            output.reserve(10);
        }
        let len = output.len();
        unsafe {
            let written = encode_vint_branchless(self.0, output.as_mut_ptr().add(len));
            output.set_len(len + written);
        }
    }

    #[cold]
    #[inline(never)]
    fn serialize_slow<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        let mut buffer = std::mem::MaybeUninit::<[u8; 10]>::uninit();
        unsafe {
            let ptr = buffer.as_mut_ptr().cast::<u8>();
            let len = encode_vint_branchless(self.0, ptr);
            writer.write_all(std::slice::from_raw_parts(ptr, len))
        }
    }

    pub fn serialize_into(&self, buffer: &mut [u8; 10]) -> usize {
        let mut remaining = self.0;
        for (i, b) in buffer.iter_mut().enumerate() {
            let next_byte: u8 = (remaining % 128u64) as u8;
            remaining /= 128u64;
            if remaining == 0u64 {
                *b = next_byte | STOP_BIT;
                return i + 1;
            } else {
                *b = next_byte;
            }
        }
        unreachable!();
    }
}

impl BinarySerializable for VInt {
    #[inline]
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        if self.0 < u64::from(STOP_BIT) {
            let byte = self.0 as u8 | STOP_BIT;
            return writer.write_all(std::slice::from_ref(&byte));
        }
        self.serialize_slow(writer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut bytes = reader.bytes();
        let mut result = 0u64;
        let mut shift = 0u64;
        loop {
            match bytes.next() {
                Some(Ok(b)) => {
                    result |= u64::from(b % 128u8) << shift;
                    if b >= STOP_BIT {
                        return Ok(VInt(result));
                    }
                    shift += 7;
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Reach end of buffer while reading VInt",
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{serialize_vint_u32, BinarySerializable, VInt};

    fn aux_test_vint(val: u64) {
        let mut v = [14u8; 10];
        let num_bytes = VInt(val).serialize_into(&mut v);
        for el in &v[num_bytes..10] {
            assert_eq!(el, &14u8);
        }
        assert!(num_bytes > 0);
        if num_bytes < 10 {
            assert!(1u64 << (7 * num_bytes) > val);
        }
        if num_bytes > 1 {
            assert!(1u64 << (7 * (num_bytes - 1)) <= val);
        }
        let serdeser_val = VInt::deserialize(&mut &v[..]).unwrap();
        assert_eq!(val, serdeser_val.0);
    }

    #[test]
    fn test_vint() {
        aux_test_vint(0);
        aux_test_vint(1);
        aux_test_vint(5);
        aux_test_vint(u64::MAX);
        for i in 1..9 {
            let power_of_128 = 1u64 << (7 * i);
            aux_test_vint(power_of_128 - 1u64);
            aux_test_vint(power_of_128);
            aux_test_vint(power_of_128 + 1u64);
        }
        aux_test_vint(10);
    }

    fn aux_test_serialize_vint_u32(val: u32) {
        let mut buffer = [0u8; 10];
        let mut buffer2 = [0u8; 8];
        let len_vint = VInt(val as u64).serialize_into(&mut buffer);
        let res2 = serialize_vint_u32(val, &mut buffer2);
        assert_eq!(&buffer[..len_vint], res2, "array wrong for {val}");
    }

    #[test]
    fn test_vint_u32() {
        aux_test_serialize_vint_u32(0);
        aux_test_serialize_vint_u32(1);
        aux_test_serialize_vint_u32(5);
        for i in 1..3 {
            let power_of_128 = 1u32 << (7 * i);
            aux_test_serialize_vint_u32(power_of_128 - 1u32);
            aux_test_serialize_vint_u32(power_of_128);
            aux_test_serialize_vint_u32(power_of_128 + 1u32);
        }
        aux_test_serialize_vint_u32(u32::MAX);
    }

    #[test]
    fn test_vint_writer_fast_paths_match_existing_encoding() {
        let mut values: Vec<u64> = (0..=u16::MAX).map(u64::from).collect();
        for bits in (7..64).step_by(7) {
            values.extend([(1u64 << bits) - 1, 1u64 << bits, (1u64 << bits) + 1]);
        }
        values.push(u64::MAX);
        let mut random = 0x627261696e73746fu64;
        for _ in 0..10_000 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            values.push(random);
        }
        let mut vector = Vec::with_capacity(16);
        let mut writer = Vec::with_capacity(16);
        for value in values {
            let mut expected = [0; 10];
            let len = VInt(value).serialize_into(&mut expected);
            vector.clear();
            writer.clear();
            vector.push(0xa5);
            writer.push(0xa5);
            VInt(value).serialize_into_vec(&mut vector);
            VInt(value).serialize(&mut writer).unwrap();
            assert_eq!(vector[0], 0xa5);
            assert_eq!(&vector[1..], &expected[..len], "value={value}");
            assert_eq!(writer, vector, "value={value}");
            assert_eq!(VInt::deserialize(&mut &vector[1..]).unwrap().0, value);
        }
        for value in [0, 1, 127] {
            let mut bytes = Vec::new();
            VInt(value).serialize_into_vec(&mut bytes);
            assert_eq!(bytes, [value as u8 | 0x80]);
        }
    }

    #[test]
    fn test_vint_vec_growth_preserves_existing_bytes() {
        for value in [0, 127, 128, 16_383, 16_384, u64::MAX] {
            let mut expected = [0; 10];
            let len = VInt(value).serialize_into(&mut expected);
            for prefix_len in [0, 1, 7, 31] {
                for spare in [0, 1, 9, 10, 16] {
                    let mut bytes = Vec::with_capacity(prefix_len + spare);
                    bytes.resize(prefix_len, 0xa5);
                    VInt(value).serialize_into_vec(&mut bytes);
                    assert!(bytes[..prefix_len].iter().all(|&byte| byte == 0xa5));
                    assert_eq!(&bytes[prefix_len..], &expected[..len]);
                }
            }
        }
    }

    #[test]
    fn test_vint_writer_handles_short_writes_and_errors() {
        struct ShortWriter(Vec<u8>);
        impl std::io::Write for ShortWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let len = bytes.len().min(1);
                self.0.extend_from_slice(&bytes[..len]);
                Ok(len)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        struct FailingWriter;
        impl std::io::Write for FailingWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for value in [0, 127, 128, 16_384, u64::MAX] {
            let mut expected = [0; 10];
            let len = VInt(value).serialize_into(&mut expected);
            let mut writer = ShortWriter(Vec::new());
            VInt(value)
                .serialize(&mut writer as &mut dyn std::io::Write)
                .unwrap();
            assert_eq!(writer.0, expected[..len]);
            assert_eq!(
                VInt(value)
                    .serialize(&mut FailingWriter)
                    .unwrap_err()
                    .kind(),
                std::io::ErrorKind::BrokenPipe
            );
        }
    }

    #[test]
    fn test_branchless_encoder_matches_reference_and_bounds() {
        let mut values: Vec<u64> = (0..=u16::MAX).map(u64::from).collect();
        for bits in 1..64 {
            values.extend([(1u64 << bits) - 1, 1u64 << bits, (1u64 << bits) + 1]);
        }
        values.push(u64::MAX);
        for value in values {
            let mut reference = [0; 10];
            let len = VInt(value).serialize_into(&mut reference);
            let mut bytes = [0xa5; 13];
            let actual_len =
                unsafe { super::encode_vint_branchless(value, bytes.as_mut_ptr().add(1)) };
            assert_eq!(actual_len, len, "value={value}");
            assert_eq!(&bytes[1..1 + len], &reference[..len]);
            assert_eq!(bytes[0], 0xa5);
            assert_eq!(&bytes[11..], &[0xa5; 2]);
            for write in [
                super::write_vint_raw as unsafe fn(u64, *mut u8) -> *mut u8,
                super::write_vint_branchless,
            ] {
                let mut bytes = [0xa5; 13];
                let base = unsafe { bytes.as_mut_ptr().add(1) };
                let end = unsafe { write(value, base) };
                assert_eq!(unsafe { end.offset_from(base) }, len as isize);
                assert_eq!(&bytes[1..1 + len], &reference[..len]);
                assert_eq!(bytes[0], 0xa5);
                assert_eq!(&bytes[11..], &[0xa5; 2]);
            }
        }
    }
}
