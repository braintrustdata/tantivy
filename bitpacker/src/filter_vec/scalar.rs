use std::ops::RangeInclusive;

#[cfg(test)]
pub fn filter_vec_in_place(range: RangeInclusive<u32>, offset: u32, output: &mut Vec<u32>) {
    let len = filter_slice_in_place(range, offset, output);
    output.truncate(len);
}

pub fn filter_slice_in_place(range: RangeInclusive<u32>, offset: u32, output: &mut [u32]) -> usize {
    // We restrict the accepted boundary, because unsigned integers & SIMD don't
    // play well.
    let mut output_cursor = 0;
    for i in 0..output.len() {
        let val = output[i];
        output[output_cursor] = offset + i as u32;
        output_cursor += if range.contains(&val) { 1 } else { 0 };
    }
    output_cursor
}
