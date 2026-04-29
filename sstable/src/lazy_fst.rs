use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use common::file_slice::FileSlice;
use common::{BinarySerializable, HasLen, OwnedBytes};

const FST_VERSION: u64 = 2;
const EMPTY_ADDRESS: usize = 0;
const TRANS_INDEX_THRESHOLD: usize = 32;
const PAGE_SIZE: usize = 16 * 1024;
const COMMON_INPUTS_INV: [u8; 64] = [
    b't', b'e', b'/', b'o', b'a', b's', b'r', b'i', b'p', b'c', b'n', b'w', b'.', b'h', b'l', b'm',
    b'-', b'd', b'u', b'0', b'1', b'2', b'g', b'=', b':', b'b', b'f', b'3', b'y', b'5', b'&', b'_',
    b'4', b'v', b'9', b'6', b'7', b'8', b'k', b'%', b'?', b'x', b'C', b'D', b'A', b'S', b'F', b'I',
    b'B', b'E', b'j', b'P', b'T', b'z', b'R', b'N', b'M', b'+', b'L', b'O', b'q', b'H', b'G', b'W',
];

#[derive(Debug, Clone)]
pub(crate) struct LazyFstIndex {
    fst_file: FileSlice,
    meta: LazyFstMeta,
    pages: Arc<Mutex<PageCache>>,
}

#[derive(Debug, Clone, Copy)]
struct LazyFstMeta {
    version: u64,
    root_addr: usize,
}

#[derive(Debug)]
struct PageCache {
    pages: HashMap<usize, OwnedBytes>,
}

impl LazyFstIndex {
    pub(crate) fn open(fst_file: FileSlice) -> io::Result<Self> {
        if fst_file.len() < 32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "FST data is corrupted",
            ));
        }

        let mut header = fst_file.read_bytes_slice(0..16)?;
        let version = u64::deserialize(&mut header)?;
        if version == 0 || version > FST_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "FST data is corrupted",
            ));
        }

        let mut footer = fst_file.read_bytes_slice(fst_file.len() - 16..fst_file.len())?;
        let _len = u64::deserialize(&mut footer)?;
        let root_addr = usize::try_from(u64::deserialize(&mut footer)?)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "FST data is corrupted"))?;

        if (root_addr == EMPTY_ADDRESS && fst_file.len() != 32)
            || (root_addr != EMPTY_ADDRESS && root_addr + 17 != fst_file.len())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "FST data is corrupted",
            ));
        }

        Ok(LazyFstIndex {
            fst_file,
            meta: LazyFstMeta { version, root_addr },
            pages: Arc::new(Mutex::new(PageCache {
                pages: HashMap::new(),
            })),
        })
    }

    pub(crate) fn lower_bound(&self, key: &[u8]) -> io::Result<Option<u64>> {
        let mut node = self.node(self.meta.root_addr)?;
        let mut out = 0u64;
        let mut frames = Vec::with_capacity(key.len());

        for &b in key {
            if let Some(transition_idx) = node.find_input(self, b)? {
                let transition = node.transition(self, transition_idx)?;
                frames.push(SearchFrame {
                    node,
                    out,
                    next_transition: node.next_transition(transition_idx),
                });
                out += transition.out;
                node = self.node(transition.addr)?;
            } else {
                if let Some(transition_idx) = node.first_transition_gt(self, b)? {
                    return self.first_output_from_transition(&node, out, transition_idx);
                }
                return self.backtrack(frames);
            }
        }

        if node.is_final {
            return Ok(Some(out + node.final_output));
        }
        if let Some(transition_idx) = node.first_transition() {
            return self.first_output_from_transition(&node, out, transition_idx);
        }
        self.backtrack(frames)
    }

    fn backtrack(&self, mut frames: Vec<SearchFrame>) -> io::Result<Option<u64>> {
        while let Some(frame) = frames.pop() {
            if let Some(transition_idx) = frame.next_transition {
                return self.first_output_from_transition(&frame.node, frame.out, transition_idx);
            }
        }
        Ok(None)
    }

    fn first_output_from_transition(
        &self,
        node: &LazyNode,
        out: u64,
        transition_idx: usize,
    ) -> io::Result<Option<u64>> {
        let transition = node.transition(self, transition_idx)?;
        let mut out = out + transition.out;
        let mut node = self.node(transition.addr)?;
        loop {
            if node.is_final {
                return Ok(Some(out + node.final_output));
            }
            let Some(transition_idx) = node.first_transition() else {
                return Ok(None);
            };
            let transition = node.transition(self, transition_idx)?;
            out += transition.out;
            node = self.node(transition.addr)?;
        }
    }

    fn node(&self, addr: usize) -> io::Result<LazyNode> {
        if addr == EMPTY_ADDRESS {
            return Ok(LazyNode::empty_final());
        }
        let state = self.read_byte(addr)?;
        match (state & 0b11_000000) >> 6 {
            0b11 => self.one_trans_next_node(addr, state),
            0b10 => self.one_trans_node(addr, state),
            _ => self.any_trans_node(addr, state),
        }
    }

    fn one_trans_next_node(&self, start: usize, state: u8) -> io::Result<LazyNode> {
        let input = common_input(state & 0b00_111111);
        let input_len = usize::from(input.is_none());
        let end = checked_sub(start, input_len)?;
        Ok(LazyNode {
            start,
            end,
            kind: LazyNodeKind::OneTransNext { input },
            is_final: false,
            ntrans: 1,
            sizes: PackSizes::new(),
            final_output: 0,
        })
    }

    fn one_trans_node(&self, start: usize, state: u8) -> io::Result<LazyNode> {
        let input = common_input(state & 0b00_111111);
        let input_len = usize::from(input.is_none());
        let sizes_pos = checked_sub(checked_sub(start, input_len)?, 1)?;
        let sizes = PackSizes::decode(self.read_byte(sizes_pos)?);
        let end = checked_sub(
            checked_sub(sizes_pos, sizes.transition_pack_size)?,
            sizes.output_pack_size,
        )?;
        Ok(LazyNode {
            start,
            end,
            kind: LazyNodeKind::OneTrans { input, input_len },
            is_final: false,
            ntrans: 1,
            sizes,
            final_output: 0,
        })
    }

    fn any_trans_node(&self, start: usize, state: u8) -> io::Result<LazyNode> {
        let ntrans_len = if state & 0b00_111111 == 0 { 1 } else { 0 };
        let ntrans = if ntrans_len == 0 {
            (state & 0b00_111111) as usize
        } else {
            match self.read_byte(checked_sub(start, 1)?)? as usize {
                1 => 256,
                ntrans => ntrans,
            }
        };
        let sizes_pos = checked_sub(checked_sub(start, ntrans_len)?, 1)?;
        let sizes = PackSizes::decode(self.read_byte(sizes_pos)?);
        let index_size = any_trans_index_size(self.meta.version, ntrans);
        let total_trans_size = ntrans + ntrans * sizes.transition_pack_size + index_size;
        let outputs_size = ntrans * sizes.output_pack_size;
        let is_final = state & 0b01_000000 == 0b01_000000;
        let final_output_size = if is_final { sizes.output_pack_size } else { 0 };
        let end = checked_sub(
            checked_sub(checked_sub(sizes_pos, total_trans_size)?, outputs_size)?,
            final_output_size,
        )?;
        let final_output = if is_final && sizes.output_pack_size > 0 {
            let output_start = checked_sub(
                checked_sub(sizes_pos, total_trans_size)?,
                outputs_size + sizes.output_pack_size,
            )?;
            self.unpack_uint(output_start..output_start + sizes.output_pack_size)?
        } else {
            0
        };
        Ok(LazyNode {
            start,
            end,
            kind: LazyNodeKind::AnyTrans {
                ntrans_len,
                index_size,
            },
            is_final,
            ntrans,
            sizes,
            final_output,
        })
    }

    fn read_byte(&self, pos: usize) -> io::Result<u8> {
        let bytes = self.read_range(pos..pos + 1)?;
        Ok(bytes.as_slice()[0])
    }

    fn unpack_uint(&self, range: Range<usize>) -> io::Result<u64> {
        let bytes = self.read_range(range)?;
        Ok(unpack_uint(bytes.as_slice()))
    }

    fn read_range(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        if range.end > self.fst_file.len() || range.start > range.end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "FST data is corrupted",
            ));
        }
        let start_page = range.start / PAGE_SIZE;
        let end_page = (range.end.saturating_sub(1)) / PAGE_SIZE;
        if start_page == end_page {
            let page = self.page(start_page)?;
            let page_start = start_page * PAGE_SIZE;
            return Ok(page.slice(range.start - page_start..range.end - page_start));
        }

        let mut bytes = Vec::with_capacity(range.end - range.start);
        for page_id in start_page..=end_page {
            let page = self.page(page_id)?;
            let page_start = page_id * PAGE_SIZE;
            let start = range.start.saturating_sub(page_start);
            let end = (range.end - page_start).min(page.len());
            bytes.extend_from_slice(&page.as_slice()[start..end]);
        }
        Ok(OwnedBytes::new(bytes))
    }

    fn page(&self, page_id: usize) -> io::Result<OwnedBytes> {
        let mut cache = self
            .pages
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "FST page cache poisoned"))?;
        if let Some(page) = cache.pages.get(&page_id) {
            return Ok(page.clone());
        }
        let start = page_id * PAGE_SIZE;
        let end = (start + PAGE_SIZE).min(self.fst_file.len());
        let page = self.fst_file.read_bytes_slice(start..end)?;
        cache.pages.insert(page_id, page.clone());
        Ok(page)
    }
}

#[derive(Debug, Clone, Copy)]
struct SearchFrame {
    node: LazyNode,
    out: u64,
    next_transition: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct LazyNode {
    start: usize,
    end: usize,
    kind: LazyNodeKind,
    is_final: bool,
    ntrans: usize,
    sizes: PackSizes,
    final_output: u64,
}

impl LazyNode {
    fn empty_final() -> Self {
        LazyNode {
            start: EMPTY_ADDRESS,
            end: EMPTY_ADDRESS,
            kind: LazyNodeKind::EmptyFinal,
            is_final: true,
            ntrans: 0,
            sizes: PackSizes::new(),
            final_output: 0,
        }
    }

    fn first_transition(&self) -> Option<usize> {
        if self.ntrans == 0 {
            None
        } else {
            Some(0)
        }
    }

    fn next_transition(&self, transition_idx: usize) -> Option<usize> {
        if transition_idx + 1 < self.ntrans {
            Some(transition_idx + 1)
        } else {
            None
        }
    }

    fn find_input(&self, index: &LazyFstIndex, input: u8) -> io::Result<Option<usize>> {
        match self.kind {
            LazyNodeKind::EmptyFinal => Ok(None),
            LazyNodeKind::OneTransNext { .. } | LazyNodeKind::OneTrans { .. } => {
                if self.input(index, 0)? == input {
                    Ok(Some(0))
                } else {
                    Ok(None)
                }
            }
            LazyNodeKind::AnyTrans { index_size, .. }
                if index.meta.version >= 2 && index_size > 0 =>
            {
                let index_start = checked_sub(
                    checked_sub(self.start, self.any_ntrans_len()?)?,
                    1 + index_size,
                )?;
                let idx = index.read_byte(index_start + input as usize)? as usize;
                if idx >= self.ntrans {
                    Ok(None)
                } else {
                    Ok(Some(idx))
                }
            }
            LazyNodeKind::AnyTrans { .. } => {
                let inputs_start = checked_sub(
                    checked_sub(self.start, self.any_ntrans_len()?)?,
                    1 + self.ntrans,
                )?;
                let inputs = index.read_range(inputs_start..inputs_start + self.ntrans)?;
                Ok(inputs
                    .as_slice()
                    .iter()
                    .position(|&candidate| candidate == input)
                    .map(|i| self.ntrans - i - 1))
            }
        }
    }

    fn first_transition_gt(&self, index: &LazyFstIndex, input: u8) -> io::Result<Option<usize>> {
        for transition_idx in 0..self.ntrans {
            if self.input(index, transition_idx)? > input {
                return Ok(Some(transition_idx));
            }
        }
        Ok(None)
    }

    fn transition(&self, index: &LazyFstIndex, transition_idx: usize) -> io::Result<Transition> {
        Ok(Transition {
            out: self.output(index, transition_idx)?,
            addr: self.transition_addr(index, transition_idx)?,
        })
    }

    fn input(&self, index: &LazyFstIndex, transition_idx: usize) -> io::Result<u8> {
        match self.kind {
            LazyNodeKind::EmptyFinal => Err(corruption_error()),
            LazyNodeKind::OneTransNext { input, .. } | LazyNodeKind::OneTrans { input, .. } => {
                if transition_idx != 0 {
                    return Err(corruption_error());
                }
                if let Some(input) = input {
                    Ok(input)
                } else {
                    index.read_byte(checked_sub(self.start, 1)?)
                }
            }
            LazyNodeKind::AnyTrans { index_size, .. } => {
                let at = checked_sub(
                    checked_sub(
                        checked_sub(self.start, self.any_ntrans_len()?)?,
                        1 + index_size + transition_idx,
                    )?,
                    1,
                )?;
                index.read_byte(at)
            }
        }
    }

    fn output(&self, index: &LazyFstIndex, transition_idx: usize) -> io::Result<u64> {
        if self.sizes.output_pack_size == 0 {
            return Ok(0);
        }
        match self.kind {
            LazyNodeKind::EmptyFinal | LazyNodeKind::OneTransNext { .. } => Ok(0),
            LazyNodeKind::OneTrans { input_len, .. } => {
                let output_start = checked_sub(
                    checked_sub(
                        checked_sub(checked_sub(self.start, input_len)?, 1)?,
                        self.sizes.transition_pack_size,
                    )?,
                    self.sizes.output_pack_size,
                )?;
                index.unpack_uint(output_start..output_start + self.sizes.output_pack_size)
            }
            LazyNodeKind::AnyTrans {
                ntrans_len,
                index_size,
            } => {
                let total_trans_size =
                    self.ntrans + self.ntrans * self.sizes.transition_pack_size + index_size;
                let output_start = checked_sub(
                    checked_sub(
                        checked_sub(self.start, ntrans_len + 1 + total_trans_size)?,
                        transition_idx * self.sizes.output_pack_size,
                    )?,
                    self.sizes.output_pack_size,
                )?;
                index.unpack_uint(output_start..output_start + self.sizes.output_pack_size)
            }
        }
    }

    fn transition_addr(&self, index: &LazyFstIndex, transition_idx: usize) -> io::Result<usize> {
        match self.kind {
            LazyNodeKind::EmptyFinal => Err(corruption_error()),
            LazyNodeKind::OneTransNext { .. } => checked_sub(self.end, 1),
            LazyNodeKind::OneTrans { input_len, .. } => {
                let transition_start = checked_sub(
                    checked_sub(checked_sub(self.start, input_len)?, 1)?,
                    self.sizes.transition_pack_size,
                )?;
                let delta = index.unpack_uint(
                    transition_start..transition_start + self.sizes.transition_pack_size,
                )?;
                unpack_delta(delta, self.end)
            }
            LazyNodeKind::AnyTrans {
                ntrans_len,
                index_size,
            } => {
                let transition_start = checked_sub(
                    checked_sub(
                        checked_sub(self.start, ntrans_len + 1 + index_size + self.ntrans)?,
                        transition_idx * self.sizes.transition_pack_size,
                    )?,
                    self.sizes.transition_pack_size,
                )?;
                let delta = index.unpack_uint(
                    transition_start..transition_start + self.sizes.transition_pack_size,
                )?;
                unpack_delta(delta, self.end)
            }
        }
    }

    fn any_ntrans_len(&self) -> io::Result<usize> {
        match self.kind {
            LazyNodeKind::AnyTrans { ntrans_len, .. } => Ok(ntrans_len),
            _ => Err(corruption_error()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum LazyNodeKind {
    EmptyFinal,
    OneTransNext {
        input: Option<u8>,
    },
    OneTrans {
        input: Option<u8>,
        input_len: usize,
    },
    AnyTrans {
        ntrans_len: usize,
        index_size: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct PackSizes {
    transition_pack_size: usize,
    output_pack_size: usize,
}

impl PackSizes {
    const fn new() -> Self {
        PackSizes {
            transition_pack_size: 0,
            output_pack_size: 0,
        }
    }

    fn decode(value: u8) -> Self {
        PackSizes {
            transition_pack_size: ((value & 0b1111_0000) >> 4) as usize,
            output_pack_size: (value & 0b0000_1111) as usize,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Transition {
    out: u64,
    addr: usize,
}

fn any_trans_index_size(version: u64, ntrans: usize) -> usize {
    if version >= 2 && ntrans > TRANS_INDEX_THRESHOLD {
        256
    } else {
        0
    }
}

fn common_input(idx: u8) -> Option<u8> {
    if idx == 0 {
        None
    } else {
        Some(COMMON_INPUTS_INV[(idx - 1) as usize])
    }
}

fn unpack_uint(bytes: &[u8]) -> u64 {
    let mut output = 0u64;
    for (i, byte) in bytes.iter().enumerate() {
        output |= (*byte as u64) << (i * 8);
    }
    output
}

fn unpack_delta(delta: u64, node_addr: usize) -> io::Result<usize> {
    let delta_addr = usize::try_from(delta)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "FST data is corrupted"))?;
    if delta_addr == EMPTY_ADDRESS {
        Ok(EMPTY_ADDRESS)
    } else {
        checked_sub(node_addr, delta_addr)
    }
}

fn checked_sub(left: usize, right: usize) -> io::Result<usize> {
    left.checked_sub(right).ok_or_else(corruption_error)
}

fn corruption_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "FST data is corrupted")
}
