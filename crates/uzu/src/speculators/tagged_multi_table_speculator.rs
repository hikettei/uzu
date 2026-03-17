use std::{collections::HashMap, iter::repeat_n, ops::Deref};

use bytemuck::cast_slice;
use xxhash_rust::xxh3::xxh3_64;
use zerocopy::{FromBytes, Immutable, KnownLayout};

use crate::speculators::speculator::Speculator;

#[cfg(not(target_endian = "little"))]
compile_error!("Only little endian is supported");

fn full_hash(seq: &[u32]) -> u64 {
    xxh3_64(cast_slice(seq))
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

/// Layout of a single tagged n-gram table within the mmap'd buffer.
/// All offsets are absolute byte positions into the parent buffer.
struct TableLayout {
    hashtable_size: u32,
    top_k: u32,
    ngram_n: u32,
    ngram_pad: u32,
    tags_start: usize,
    keys_start: usize,
    values_start: usize,
}

impl TableLayout {
    /// Parse table layout starting at `offset`. Returns (layout, bytes_consumed).
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

        let keys_start = off;
        off += 4 * k * hs;

        let values_start = off;
        off += 4 * k * hs;

        // counts
        off += 4 * hs;

        // continuation dist (skip)
        let cont_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        off += 4 * cont_len; // cont_keys
        off += 4 * cont_len; // cont_vals

        let layout = Self {
            hashtable_size: header.hashtable_size,
            top_k: header.top_k,
            ngram_n: header.ngram_n,
            ngram_pad: header.ngram_pad,
            tags_start,
            keys_start,
            values_start,
        };

        (layout, off - offset)
    }

    #[inline]
    fn read_tag(
        &self,
        bytes: &[u8],
        idx: usize,
    ) -> u64 {
        let off = self.tags_start + idx * 8;
        u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn read_key(
        &self,
        bytes: &[u8],
        idx: usize,
    ) -> u32 {
        let off = self.keys_start + idx * 4;
        u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
    }

    #[inline]
    fn read_value(
        &self,
        bytes: &[u8],
        idx: usize,
    ) -> f32 {
        let off = self.values_start + idx * 4;
        f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
    }

    /// Hash context, check tag, return top-k if tag matches.
    fn lookup(
        &self,
        bytes: &[u8],
        prefix: &[u64],
    ) -> Option<HashMap<u64, f32>> {
        let ngram_ctx = (self.ngram_n - 1) as usize;

        let context: Vec<u32> = if ngram_ctx == 0 {
            vec![]
        } else if prefix.len() >= ngram_ctx {
            prefix[prefix.len() - ngram_ctx..].iter().map(|&x| x as u32).collect()
        } else {
            repeat_n(self.ngram_pad, ngram_ctx - prefix.len()).chain(prefix.iter().map(|&x| x as u32)).collect()
        };

        let hash = full_hash(&context);
        let idx = (hash % self.hashtable_size as u64) as usize;
        let tag = hash / self.hashtable_size as u64;

        if self.read_tag(bytes, idx) != tag {
            return None;
        }

        let k = self.top_k as usize;
        let k_start = idx * k;

        let mut result = HashMap::with_capacity(k);
        for i in 0..k {
            result.insert(self.read_key(bytes, k_start + i) as u64, self.read_value(bytes, k_start + i));
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
/// Each TaggedNGramTable:
/// ```text
/// [header: hashtable_size u32, top_k u32, ngram_n u32, ngram_pad u32]
/// [tags:   u64 * hashtable_size]
/// [keys:   u32 * hashtable_size * top_k]
/// [values: f32 * hashtable_size * top_k]
/// [counts: u32 * hashtable_size]
/// [cont_dist_len: u32] [cont_keys: u32 * len] [cont_vals: f32 * len]
/// ```
///
/// Inference: try highest-order table first. On tag mismatch, backoff to next lower order.
pub struct TaggedMultiTableSpeculator<B: Deref<Target = [u8]> + Send + Sync> {
    bytes: B,
    tables: Vec<TableLayout>,
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
        }
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
        // Try from highest order to lowest, first tag match wins
        for table in self.tables.iter().rev() {
            if let Some(result) = table.lookup(&self.bytes, prefix) {
                return result;
            }
        }
        HashMap::new()
    }
}
