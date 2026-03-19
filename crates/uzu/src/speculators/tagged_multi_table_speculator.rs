use std::{collections::HashMap, ops::Deref};

use bytemuck::cast_slice;
use xxhash_rust::xxh3::xxh3_64;
use zerocopy::{FromBytes, Immutable, KnownLayout};

use crate::speculators::speculator::Speculator;

#[cfg(not(target_endian = "little"))]
compile_error!("Only little endian is supported");

const MAX_CTX: usize = 15;

#[inline]
fn full_hash(seq: &[u32]) -> u64 {
    xxh3_64(cast_slice(seq))
}

/// Apply temperature scaling: p_i^(1/τ), then renormalize.
#[inline]
fn apply_temperature(probs: &mut HashMap<u64, f32>, inv_tau: f32) {
    let mut sum = 0.0f32;
    for v in probs.values_mut() {
        *v = v.powf(inv_tau);
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in probs.values_mut() {
            *v *= inv_sum;
        }
    }
}

#[repr(C)]
#[derive(FromBytes, KnownLayout, Immutable)]
struct TaggedTableHeader {
    hashtable_size: u32,
    top_k: u32,
    ngram_n: u32,
    ngram_pad: u32,
}

const HEADER_SIZE: usize = size_of::<TaggedTableHeader>();

struct TableLayout {
    hashtable_size: u32,
    top_k: u32,
    ngram_n: u32,
    ngram_pad: u32,
    tags: Box<[u64]>,
    keys_start: usize,
    values_start: usize,
}

impl TableLayout {
    fn parse(
        bytes: &[u8],
        offset: usize,
    ) -> (Self, usize) {
        let header = TaggedTableHeader::ref_from_bytes(&bytes[offset..offset + HEADER_SIZE]).unwrap();

        let hs = header.hashtable_size as usize;
        let k = header.top_k as usize;

        let mut off = offset + HEADER_SIZE;

        let tags_start = off;
        off += 8 * hs;
        let mut tags = vec![0u64; hs].into_boxed_slice();
        for i in 0..hs {
            let t = tags_start + i * 8;
            tags[i] = u64::from_le_bytes(bytes[t..t + 8].try_into().unwrap());
        }

        let keys_start = off;
        off += 4 * k * hs;

        let values_start = off;
        off += 4 * k * hs;

        // counts
        off += 4 * hs;

        // continuation dist (skip)
        let cont_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        off += 4 * cont_len;
        off += 4 * cont_len;

        let layout = Self {
            hashtable_size: header.hashtable_size,
            top_k: header.top_k,
            ngram_n: header.ngram_n,
            ngram_pad: header.ngram_pad,
            tags,
            keys_start,
            values_start,
        };

        (layout, off - offset)
    }

    #[inline]
    fn keys<'a>(
        &self,
        bytes: &'a [u8],
    ) -> &'a [u32] {
        let len = self.hashtable_size as usize * self.top_k as usize;
        cast_slice(&bytes[self.keys_start..self.keys_start + 4 * len])
    }

    #[inline]
    fn values<'a>(
        &self,
        bytes: &'a [u8],
    ) -> &'a [f32] {
        let len = self.hashtable_size as usize * self.top_k as usize;
        cast_slice(&bytes[self.values_start..self.values_start + 4 * len])
    }

    #[inline]
    fn lookup(
        &self,
        bytes: &[u8],
        prefix: &[u64],
    ) -> Option<HashMap<u64, f32>> {
        let ngram_ctx = (self.ngram_n - 1) as usize;

        let mut ctx_buf = [0u32; MAX_CTX];
        if ngram_ctx > 0 {
            let prefix_len = prefix.len();
            if prefix_len >= ngram_ctx {
                for i in 0..ngram_ctx {
                    ctx_buf[i] = prefix[prefix_len - ngram_ctx + i] as u32;
                }
            } else {
                let pad_count = ngram_ctx - prefix_len;
                for i in 0..pad_count {
                    ctx_buf[i] = self.ngram_pad;
                }
                for i in 0..prefix_len {
                    ctx_buf[pad_count + i] = prefix[i] as u32;
                }
            }
        }

        let hash = full_hash(&ctx_buf[..ngram_ctx]);
        let idx = (hash % self.hashtable_size as u64) as usize;
        let tag = hash / self.hashtable_size as u64;

        if self.tags[idx] != tag {
            return None;
        }

        let k = self.top_k as usize;
        let k_start = idx * k;
        let k_end = k_start + k;

        let keys = self.keys(bytes);
        let values = self.values(bytes);

        let mut result = HashMap::with_capacity(k);
        for i in k_start..k_end {
            result.insert(keys[i] as u64, values[i]);
        }

        Some(result)
    }
}

/// Tagged multi-table n-gram speculator with backoff.
///
/// Binary format (little-endian):
/// ```text
/// [max_order: u32]
/// For each table (1-gram .. max_order-gram):
///   [table_len: u64] [TaggedNGramTable bytes]
/// ```
///
/// Inference: try highest-order table first. On tag mismatch, backoff to next lower order.
pub struct TaggedMultiTableSpeculator<B: Deref<Target = [u8]> + Send + Sync> {
    bytes: B,
    tables: Vec<TableLayout>,
    inv_tau: f32,
}

impl<B: Deref<Target = [u8]> + Send + Sync> TaggedMultiTableSpeculator<B> {
    pub fn new(bytes: B) -> Self {
        let mut off = 0;

        let max_order = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;

        let mut tables = Vec::with_capacity(max_order as usize);
        for _ in 0..max_order {
            let table_len = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as usize;
            off += 8;

            let (layout, _) = TableLayout::parse(&bytes, off);
            tables.push(layout);
            off += table_len;
        }

        assert_eq!(off, bytes.len());

        Self {
            bytes,
            tables,
            inv_tau: 0.0,
        }
    }

    pub fn with_temperature(mut self, tau: f32) -> Self {
        self.inv_tau = if tau > 0.0 && tau != 1.0 { 1.0 / tau } else { 0.0 };
        self
    }
}

impl TaggedMultiTableSpeculator<memmap2::Mmap> {
    pub fn load(path: &str) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::default().map(&file).unwrap() };
        Self::new(mmap)
    }
}

impl<B: Deref<Target = [u8]> + Send + Sync> Speculator for TaggedMultiTableSpeculator<B> {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        for table in self.tables.iter().rev() {
            if let Some(mut result) = table.lookup(&self.bytes, prefix) {
                if self.inv_tau > 0.0 {
                    apply_temperature(&mut result, self.inv_tau);
                }
                return result;
            }
        }
        HashMap::new()
    }
}

/// KN multi-table n-gram speculator.
///
/// Binary format: `[max_order: u32, discount: f32]` followed by tagged tables.
pub struct KNMultiTableSpeculator<B: Deref<Target = [u8]> + Send + Sync> {
    bytes: B,
    _discount: f32,
    tables: Vec<TableLayout>,
    inv_tau: f32,
}

impl<B: Deref<Target = [u8]> + Send + Sync> KNMultiTableSpeculator<B> {
    pub fn new(bytes: B) -> Self {
        let mut off = 0;

        let max_order = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let discount = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        off += 8;

        let mut tables = Vec::with_capacity(max_order as usize);
        for _ in 0..max_order {
            let table_len = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as usize;
            off += 8;

            let (layout, _) = TableLayout::parse(&bytes, off);
            tables.push(layout);
            off += table_len;
        }

        assert_eq!(off, bytes.len());

        Self {
            bytes,
            _discount: discount,
            tables,
            inv_tau: 0.0,
        }
    }

    pub fn with_temperature(mut self, tau: f32) -> Self {
        self.inv_tau = if tau > 0.0 && tau != 1.0 { 1.0 / tau } else { 0.0 };
        self
    }
}

impl KNMultiTableSpeculator<memmap2::Mmap> {
    pub fn load(path: &str) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::default().map(&file).unwrap() };
        Self::new(mmap)
    }
}

impl<B: Deref<Target = [u8]> + Send + Sync> Speculator for KNMultiTableSpeculator<B> {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        for table in self.tables.iter().rev() {
            if let Some(mut result) = table.lookup(&self.bytes, prefix) {
                if self.inv_tau > 0.0 {
                    apply_temperature(&mut result, self.inv_tau);
                }
                return result;
            }
        }
        HashMap::new()
    }
}
