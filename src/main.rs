// Shopping Sessions: native Rust pipeline.
// Reads $DATA_DIR, writes $OUTPUT_DIR/{user_daily,campaign_daily}/*.parquet.

use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use memmap2::Mmap;

#[cfg(feature = "diag")]
mod diag;
mod pq;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ---------------------------------------------------------------- hashing

#[inline(always)]
fn mix(v: u64) -> u64 {
    let mut x = v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 32;
    x = x.wrapping_mul(0xd6e8_feb8_6659_fd93);
    x ^= x >> 32;
    x
}

#[derive(Default, Clone, Copy)]
struct FastHasher(u64);
impl Hasher for FastHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0 = mix(hash_bytes(bytes).rotate_left(23) ^ self.0);
    }
    #[inline(always)]
    fn write_u64(&mut self, v: u64) {
        self.0 = mix(v);
    }
    #[inline(always)]
    fn write_u32(&mut self, v: u32) {
        self.0 = mix(v as u64);
    }
    #[inline(always)]
    fn write_usize(&mut self, v: usize) {
        self.0 = mix(v as u64);
    }
}
type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[inline(always)]
fn part_of(k: u64, mask: usize) -> usize {
    (mix(k) >> 40) as usize & mask
}

// ---------------------------------------------------------------- string keys
// A string of 1..=10 bytes from [0-9_a-z] encodes exactly as ten 6-bit codes
// (0 = padding), so numeric order equals byte-wise order. Anything else is
// interned and gets the top bit set; such keys compare by their bytes.
const KEY_LEN: usize = 10;

const fn build_enc() -> [u8; 256] {
    let mut t = [0x80u8; 256];
    let mut i = 0;
    while i < 10 {
        t[b'0' as usize + i] = 1 + i as u8;
        i += 1;
    }
    t[b'_' as usize] = 11;
    let mut i = 0;
    while i < 26 {
        t[b'a' as usize + i] = 12 + i as u8;
        i += 1;
    }
    t
}
static ENC: [u8; 256] = build_enc();
const fn build_dec() -> [u8; 38] {
    let mut t = [0u8; 38];
    let mut i = 0;
    while i < 10 {
        t[1 + i] = b'0' + i as u8;
        i += 1;
    }
    t[11] = b'_';
    let mut i = 0;
    while i < 26 {
        t[12 + i] = b'a' + i as u8;
        i += 1;
    }
    t
}
static DEC: [u8; 38] = build_dec();
const SLOW: u64 = 1 << 63;

#[inline(always)]
fn hash_bytes(s: &[u8]) -> u64 {
    let mut h: u64 = 0x243F_6A88_85A3_08D3 ^ s.len() as u64;
    let mut c = s.chunks_exact(8);
    for w in &mut c {
        h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    let r = c.remainder();
    if !r.is_empty() {
        let mut b = [0u8; 8];
        b[..r.len()].copy_from_slice(r);
        h = (h ^ u64::from_le_bytes(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    mix(h)
}

/// Branch-free fast_key: reads 10 bytes starting at s.as_ptr() (caller guarantees they are
/// readable), encodes all of them and masks positions >= len.
#[inline(always)]
unsafe fn fast_key_win(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if n.wrapping_sub(1) >= KEY_LEN {
        return None;
    }
    let p = s.as_ptr();
    let mut v = 0u64;
    let mut bad = 0u32;
    for i in 0..KEY_LEN {
        let c = ENC[unsafe { *p.add(i) } as usize];
        bad |= ((c >> 7) as u32) << i;
        v |= ((c & 0x3F) as u64) << (6 * (KEY_LEN - 1 - i));
    }
    let keep_bits = 6 * n as u32; // top bits that belong to the key
    let v = v & !(u64::MAX >> (64 - 6 * KEY_LEN as u32) >> keep_bits) & ((1u64 << (6 * KEY_LEN)) - 1);
    if bad & ((1u32 << n) - 1) != 0 {
        return None;
    }
    Some(v)
}

#[inline(always)]
fn fast_key(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if n >= 1 && n <= KEY_LEN {
        let mut v = 0u64;
        let mut flags = 0u8;
        for i in 0..KEY_LEN {
            let c = if i < n { ENC[s[i] as usize] } else { 0 };
            flags |= c;
            v |= ((c & 0x3F) as u64) << (6 * (KEY_LEN - 1 - i));
        }
        if flags & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

static IDX16: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Same result as `fast_key`, using one 16-byte vector load. Caller guarantees 16 readable bytes.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
#[inline(always)]
unsafe fn simd_key16(s: &[u8]) -> Option<u64> {
    use std::arch::x86_64::*;
    let n = s.len();
    if n.wrapping_sub(1) >= KEY_LEN {
        return None;
    }
    unsafe {
        let v = _mm_loadu_si128(s.as_ptr() as *const __m128i);
        let d = _mm_sub_epi8(v, _mm_set1_epi8(0x30));
        let is_digit = _mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d);
        let l = _mm_sub_epi8(v, _mm_set1_epi8(0x61));
        let is_lower = _mm_cmpeq_epi8(_mm_min_epu8(l, _mm_set1_epi8(25)), l);
        let is_us = _mm_cmpeq_epi8(v, _mm_set1_epi8(0x5F));
        let valid = _mm_or_si128(_mm_or_si128(is_digit, is_lower), is_us);
        let idx = _mm_loadu_si128(IDX16.as_ptr() as *const __m128i);
        let inlen = _mm_cmpgt_epi8(_mm_set1_epi8(n as i8), idx);
        if _mm_movemask_epi8(_mm_andnot_si128(valid, inlen)) != 0 {
            return None;
        }
        let code = _mm_or_si128(
            _mm_or_si128(
                _mm_and_si128(is_digit, _mm_add_epi8(d, _mm_set1_epi8(1))),
                _mm_and_si128(is_lower, _mm_add_epi8(l, _mm_set1_epi8(12))),
            ),
            _mm_and_si128(is_us, _mm_set1_epi8(11)),
        );
        let code = _mm_and_si128(code, inlen);
        // pairs -> 12-bit lanes (c0*64+c1), then -> 24-bit lanes (4 chars each)
        let w = _mm_maddubs_epi16(code, _mm_set1_epi16(0x0140));
        let x = _mm_madd_epi16(w, _mm_set1_epi32(0x0001_1000));
        let lo = _mm_cvtsi128_si64(x) as u64;
        let x2 = _mm_extract_epi32::<2>(x) as u32 as u64;
        Some(((lo & 0xFFFF_FFFF) << 36) | ((lo >> 32) << 12) | (x2 >> 12))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn simd_key16(s: &[u8]) -> Option<u64> {
    use std::arch::aarch64::*;
    let n = s.len();
    if n.wrapping_sub(1) >= KEY_LEN {
        return None;
    }
    unsafe {
        let v = vld1q_u8(s.as_ptr());
        let d = vsubq_u8(v, vdupq_n_u8(0x30));
        let is_digit = vcleq_u8(d, vdupq_n_u8(9));
        let l = vsubq_u8(v, vdupq_n_u8(0x61));
        let is_lower = vcleq_u8(l, vdupq_n_u8(25));
        let is_us = vceqq_u8(v, vdupq_n_u8(0x5F));
        let valid = vorrq_u8(vorrq_u8(is_digit, is_lower), is_us);
        let inlen = vcltq_u8(vld1q_u8(IDX16.as_ptr()), vdupq_n_u8(n as u8));
        if vmaxvq_u8(vbicq_u8(inlen, valid)) != 0 {
            return None;
        }
        let code = vorrq_u8(
            vorrq_u8(vandq_u8(is_digit, vaddq_u8(d, vdupq_n_u8(1))), vandq_u8(is_lower, vaddq_u8(l, vdupq_n_u8(12)))),
            vandq_u8(is_us, vdupq_n_u8(11)),
        );
        let code = vreinterpretq_u64_u8(vandq_u8(code, inlen));
        let w0 = vgetq_lane_u64(code, 0);
        let w1 = vgetq_lane_u64(code, 1);
        let t = ((w0 & 0x00FF_00FF_00FF_00FF) << 6) | ((w0 >> 8) & 0x00FF_00FF_00FF_00FF);
        let t2 = ((t & 0x0000_FFFF_0000_FFFF) << 12) | ((t >> 16) & 0x0000_FFFF_0000_FFFF);
        let t3 = ((t2 & 0xFFFF_FFFF) << 24) | (t2 >> 32);
        Some((t3 << 12) | ((w1 & 0xFF) << 6) | ((w1 >> 8) & 0xFF))
    }
}

#[cfg(not(any(all(target_arch = "x86_64", target_feature = "sse4.1"), target_arch = "aarch64")))]
#[inline(always)]
unsafe fn simd_key16(s: &[u8]) -> Option<u64> {
    unsafe { fast_key_win(s) }
}

#[inline(always)]
fn fast_key_at(s: &[u8], end: usize) -> Option<u64> {
    let p = s.as_ptr() as usize;
    if p + 16 <= end {
        unsafe { simd_key16(s) }
    } else if p + KEY_LEN <= end {
        unsafe { fast_key_win(s) }
    } else {
        fast_key(s)
    }
}

#[inline(always)]
fn eq_key_at(s: &[u8], end: usize) -> u64 {
    match fast_key_at(s, end) {
        Some(v) => v,
        None => SLOW | (hash_bytes(s) >> 1),
    }
}

#[inline(always)]
fn key_at(s: &[u8], end: usize, longs: &mut LongMap) -> u64 {
    match fast_key_at(s, end) {
        Some(v) => v,
        None => long_key(s, longs),
    }
}

/// Equality-only key: short IDs encode exactly, others become SLOW | 63-bit hash.
#[inline(always)]
fn eq_key(s: &[u8]) -> u64 {
    match fast_key(s) {
        Some(v) => v,
        None => SLOW | (hash_bytes(s) >> 1),
    }
}

/// Per-thread map from a long key to its bytes (no locking).
type LongMap = FastMap<u64, Box<[u8]>>;

/// Key that also remembers the bytes of long IDs in `longs`, so they can be ordered and printed.
#[inline(always)]
fn key(s: &[u8], longs: &mut LongMap) -> u64 {
    match fast_key(s) {
        Some(v) => v,
        None => long_key(s, longs),
    }
}

#[inline(never)]
fn long_key(s: &[u8], longs: &mut LongMap) -> u64 {
    let k = SLOW | (hash_bytes(s) >> 1);
    longs.entry(k).or_insert_with(|| s.into());
    k
}

/// Bytes of a click id key (fast, file location, or LongMap).
/// Bytes of a long click id (file location or LongMap). Short ids are decoded by the caller.
#[inline(always)]
fn cid_long<'a>(k: u64, files: &'a [Option<Buf>], longs: &'a Longs) -> &'a [u8] {
    let file = ((k >> 52) & 0x3FF) as usize;
    if file == 0x3FF {
        return longs.get(k);
    }
    let off = ((k >> 12) & ((1 << 40) - 1)) as usize;
    let len = (k & 0xFFF) as usize;
    &files[file].as_ref().unwrap()[off..off + len]
}

#[inline(always)]
fn cid_cmp(a: u64, b: u64, files: &[Option<Buf>], longs: &Longs) -> std::cmp::Ordering {
    if (a | b) & SLOW == 0 {
        a.cmp(&b)
    } else {
        let (mut ba, mut bb) = ([0u8; KEY_LEN], [0u8; KEY_LEN]);
        let sa: &[u8] = if a & SLOW != 0 { cid_long(a, files, longs) } else { let n = short_bytes(a, &mut ba); &ba[..n] };
        let sb: &[u8] = if b & SLOW != 0 { cid_long(b, files, longs) } else { let n = short_bytes(b, &mut bb); &bb[..n] };
        sa.cmp(sb)
    }
}

/// Read-only view over every thread's long-ID map.
struct Longs<'a>(Vec<&'a LongMap>);
impl Longs<'_> {
    fn get(&self, k: u64) -> &[u8] {
        for m in &self.0 {
            if let Some(v) = m.get(&k) {
                return v;
            }
        }
        panic!("long key without bytes");
    }
}

fn key_bytes(k: u64, longs: &Longs) -> Vec<u8> {
    if k & SLOW != 0 {
        longs.get(k).to_vec()
    } else {
        (0..KEY_LEN)
            .map(|i| ((k >> (6 * (KEY_LEN - 1 - i))) & 0x3F) as usize)
            .take_while(|&c| c != 0)
            .map(|c| DEC[c])
            .collect()
    }
}

/// Decodes a short key into `buf`, returning its length.
#[inline(always)]
fn short_bytes(k: u64, buf: &mut [u8; KEY_LEN]) -> usize {
    let mut n = 0;
    for i in 0..KEY_LEN {
        let c = ((k >> (6 * (KEY_LEN - 1 - i))) & 0x3F) as usize;
        if c == 0 {
            break;
        }
        buf[i] = DEC[c];
        n += 1;
    }
    n
}

#[inline(always)]
fn key_cmp(a: u64, b: u64, longs: &Longs) -> std::cmp::Ordering {
    if (a | b) & SLOW == 0 {
        a.cmp(&b)
    } else if a == b {
        std::cmp::Ordering::Equal
    } else {
        let (mut ba, mut bb) = ([0u8; KEY_LEN], [0u8; KEY_LEN]);
        let sa: &[u8] = if a & SLOW != 0 { longs.get(a) } else { let n = short_bytes(a, &mut ba); &ba[..n] };
        let sb: &[u8] = if b & SLOW != 0 { longs.get(b) } else { let n = short_bytes(b, &mut bb); &bb[..n] };
        sa.cmp(sb)
    }
}

// ---------------------------------------------------------------- CSV dialect

/// Masks over 64 bytes from p: (',' or '\n', '\n', '"').
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn masks(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    unsafe {
        let c = _mm256_set1_epi8(b',' as i8);
        let q = _mm256_set1_epi8(b'"' as i8);
        let n = _mm256_set1_epi8(b'\n' as i8);
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let m = |x: __m256i| _mm256_movemask_epi8(x) as u32 as u64;
        let na = _mm256_cmpeq_epi8(a, n);
        let nb = _mm256_cmpeq_epi8(b, n);
        let sa = _mm256_or_si256(_mm256_cmpeq_epi8(a, c), na);
        let sb = _mm256_or_si256(_mm256_cmpeq_epi8(b, c), nb);
        (
            m(sa) | (m(sb) << 32),
            m(na) | (m(nb) << 32),
            m(_mm256_cmpeq_epi8(a, q)) | (m(_mm256_cmpeq_epi8(b, q)) << 32),
        )
    }
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[inline(always)]
unsafe fn masks(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    unsafe {
        let c = _mm_set1_epi8(b',' as i8);
        let q = _mm_set1_epi8(b'"' as i8);
        let n = _mm_set1_epi8(b'\n' as i8);
        let (mut s, mut nl, mut qq) = (0u64, 0u64, 0u64);
        for k in 0..4 {
            let v = _mm_loadu_si128(p.add(16 * k) as *const __m128i);
            let nv = _mm_cmpeq_epi8(v, n);
            s |= (_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(v, c), nv)) as u16 as u64) << (16 * k);
            nl |= (_mm_movemask_epi8(nv) as u16 as u64) << (16 * k);
            qq |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, q)) as u16 as u64) << (16 * k);
        }
        (s, nl, qq)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn masks(p: *const u8) -> (u64, u64, u64) {
    use std::arch::aarch64::*;
    unsafe {
        // 4 x 16 bytes; narrow each compare to a 4-bit-per-byte mask, then compress to 1 bit per byte
        #[inline(always)]
        unsafe fn bits(v: [uint8x16_t; 4]) -> u64 {
            unsafe {
                let mut out = 0u64;
                for k in 0..4 {
                    let nib = vshrn_n_u16(vreinterpretq_u16_u8(v[k]), 4);
                    let x = vget_lane_u64(vreinterpret_u64_u8(nib), 0);
                    // x has 4 bits per byte (all ones or zeros); take one bit of each nibble
                    let mut y = x & 0x1111_1111_1111_1111;
                    y = (y | (y >> 3)) & 0x0303_0303_0303_0303;
                    y = (y | (y >> 6)) & 0x000f_000f_000f_000f;
                    y = (y | (y >> 12)) & 0x0000_00ff_0000_00ff;
                    y = (y | (y >> 24)) & 0xffff;
                    out |= y << (16 * k);
                }
                out
            }
        }
        let c = vdupq_n_u8(b',');
        let q = vdupq_n_u8(b'"');
        let n = vdupq_n_u8(b'\n');
        let mut vs = [vdupq_n_u8(0); 4];
        let mut vn = [vdupq_n_u8(0); 4];
        let mut vq = [vdupq_n_u8(0); 4];
        for k in 0..4 {
            let v = vld1q_u8(p.add(16 * k));
            vn[k] = vceqq_u8(v, n);
            vs[k] = vorrq_u8(vceqq_u8(v, c), vn[k]);
            vq[k] = vceqq_u8(v, q);
        }
        (bits(vs), bits(vn), bits(vq))
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn masks(p: *const u8) -> (u64, u64, u64) {
    let (mut s, mut nl, mut qq) = (0u64, 0u64, 0u64);
    for i in 0..64 {
        let c = unsafe { *p.add(i) };
        s |= ((c == b',' || c == b'\n') as u64) << i;
        nl |= ((c == b'\n') as u64) << i;
        qq |= ((c == b'"') as u64) << i;
    }
    (s, nl, qq)
}

/// Tokenizes `data` into lines of exactly N unquoted fields; malformed lines
/// (wrong field count, stray quotes, empty) are skipped. Only '\n' ends a line.
/// A line is well quoted iff its number of '"' equals twice the number of fields
/// that are wrapped in quotes (length >= 2, first and last byte '"').
#[inline(always)]
#[inline(always)]
fn scan_lines<'a, const N: usize>(data: &'a [u8], on_line: impl FnMut(&[&'a [u8]; N])) {
    scan_lines_ref::<N>(data, on_line)
}

#[inline(always)]
fn scan_lines_ref<'a, const N: usize>(data: &'a [u8], mut on_line: impl FnMut(&[&'a [u8]; N])) {
    let mut fields: [&'a [u8]; N] = [&[]; N];
    let mut nf = 0usize;
    let mut fstart = 0usize;
    let mut expect = 0u32;
    let mut qbefore = 0u32;
    let mut line_q0 = 0u32;
    let len = data.len();
    let mut base = 0usize;
    let mut tail = [0u8; 64];
    macro_rules! field {
        ($end:expr) => {{
            let end = $end;
            if nf < N {
                let fl = end - fstart;
                let w = if fl >= 2 {
                    unsafe { (*data.get_unchecked(fstart) == b'"') & (*data.get_unchecked(end - 1) == b'"') }
                } else {
                    false
                } as usize;
                fields[nf] = unsafe { data.get_unchecked(fstart + w..end - w) };
                expect += 2 * w as u32;
            }
            nf += 1;
        }};
    }
    while base < len {
        let (mut sep, nl, q) = if base + 64 <= len {
            unsafe { masks(data.as_ptr().add(base)) }
        } else {
            tail = [0u8; 64];
            tail[..len - base].copy_from_slice(&data[base..]);
            unsafe { masks(tail.as_ptr()) }
        };
        while sep != 0 {
            let off = sep.trailing_zeros();
            let bit = 1u64 << off;
            sep &= sep - 1;
            let p = base + off as usize;
            field!(p);
            if nl & bit != 0 {
                let qnow = qbefore + (q & (bit - 1)).count_ones();
                if nf == N && qnow - line_q0 == expect {
                    on_line(&fields);
                }
                line_q0 = qnow;
                nf = 0;
                expect = 0;
            }
            fstart = p + 1;
        }
        qbefore += q.count_ones();
        base += 64;
    }
    if fstart < len || (len > 0 && data[len - 1] == b',') {
        field!(len);
        if nf == N && qbefore - line_q0 == expect {
            on_line(&fields);
        }
    }
}

/// Two-stage tokenizer with the same semantics as `scan_lines_ref`.
/// Stage 1 records every separator (',' or '\n') with the running quote count;
/// stage 2 walks lines, expecting N-1 commas then a newline.
#[allow(dead_code)]
#[inline(always)]
fn scan_lines2<'a, const N: usize>(data: &'a [u8], mut on_line: impl FnMut(&[&'a [u8]; N])) {
    const SUB: usize = 1 << 16; // bytes per stage-1 window
    let len = data.len();
    let mut pos: Vec<u32> = Vec::with_capacity(SUB / 2 + 64);
    let mut qc: Vec<u32> = Vec::with_capacity(SUB / 2 + 64);
    let mut fields: [&'a [u8]; N] = [&[]; N];
    let mut line_start = 0usize; // absolute offset of current line start
    let mut line_q0 = 0u32; // quotes before current line
    let mut qbefore = 0u32; // quotes before current block
    let mut base = 0usize;
    let mut tail = [0u8; 64];
    while base < len {
        // ---- stage 1 over [base, win_end)
        pos.clear();
        qc.clear();
        let win_start = base;
        let win_end = (base + SUB).min(len);
        while base < win_end {
            let (mut sep, nl, q) = if base + 64 <= len {
                unsafe { masks(data.as_ptr().add(base)) }
            } else {
                tail = [0u8; 64];
                tail[..len - base].copy_from_slice(&data[base..]);
                unsafe { masks(tail.as_ptr()) }
            };
            while sep != 0 {
                let off = sep.trailing_zeros();
                sep &= sep - 1;
                let rel = (base - win_start) as u32 + off;
                pos.push(rel << 1 | ((nl >> off) & 1) as u32);
                qc.push(qbefore + (q & ((1u64 << off) - 1)).count_ones());
            }
            qbefore += q.count_ones();
            base += 64;
        }
        if win_end == len && (len == 0 || data[len - 1] != b'\n') {
            // virtual newline at end of data (only if there is an unfinished line)
            pos.push(((len - win_start) as u32) << 1 | 1);
            qc.push(qbefore);
        }
        // ---- stage 2
        let ne = pos.len();
        let mut i = 0usize;
        while i < ne {
            // fast path: N entries, first N-1 commas, last newline
            if i + N <= ne {
                let mut ok = (pos[i + N - 1] & 1) == 1;
                for k in 0..N - 1 {
                    ok &= (pos[i + k] & 1) == 0;
                }
                if ok {
                    let mut st = line_start;
                    let mut expect = 0u32;
                    for k in 0..N {
                        let end = win_start + (pos[i + k] >> 1) as usize;
                        let fl = end - st;
                        let w = if fl >= 2 {
                            unsafe { (*data.get_unchecked(st) == b'"') & (*data.get_unchecked(end - 1) == b'"') }
                        } else {
                            false
                        } as usize;
                        fields[k] = unsafe { data.get_unchecked(st + w..end - w) };
                        expect += 2 * w as u32;
                        st = end + 1;
                    }
                    let qn = qc[i + N - 1];
                    if qn - line_q0 == expect {
                        on_line(&fields);
                    }
                    line_q0 = qn;
                    line_start = st;
                    i += N;
                    continue;
                }
            }
            // slow path: skip to this line's newline (malformed or crossing the window)
            let mut j = i;
            while j < ne && pos[j] & 1 == 0 {
                j += 1;
            }
            if j == ne {
                // line continues into the next window: re-scan it there
                break;
            }
            // line [line_start, newline) has a wrong field count -> malformed
            line_q0 = qc[j];
            line_start = win_start + (pos[j] >> 1) as usize + 1;
            i = j + 1;
        }
        if i < ne {
            // unfinished line crosses the window: restart stage 1 at its start
            base = line_start;
            qbefore = line_q0;
            // align: masks need no alignment, so just continue from line_start
            if base >= len {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------- timestamps

const DAY_MS: i64 = 86_400_000;

#[inline(always)]
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[inline(always)]
fn d2(f: &[u8], i: usize) -> Option<i64> {
    let a = f[i].wrapping_sub(b'0');
    let b = f[i + 1].wrapping_sub(b'0');
    if a > 9 || b > 9 {
        return None;
    }
    Some(a as i64 * 10 + b as i64)
}

#[inline(always)]
fn civil_ms(f: &[u8]) -> Option<i64> {
    // YYYY-MM-DDTHH:MM:SS
    if f[4] != b'-' || f[7] != b'-' || f[10] != b'T' || f[13] != b':' || f[16] != b':' {
        return None;
    }
    let y = d2(f, 0)? * 100 + d2(f, 2)?;
    let mo = d2(f, 5)?;
    let d = d2(f, 8)?;
    let h = d2(f, 11)?;
    let mi = d2(f, 14)?;
    let s = d2(f, 17)?;
    if !(1..=12).contains(&mo) || d < 1 || h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let dim = match mo {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    };
    if d > dim {
        return None;
    }
    Some((days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s) * 1000)
}

/// Parses F1, F2 or F3 into epoch ms. None if bad (F3 beyond i64 is also None).
#[inline(always)]
fn parse_ts(f: &[u8]) -> Option<i64> {
    let n = f.len();
    if n == 20 && f[19] == b'Z' {
        return civil_ms(f);
    }
    if n == 25 && (f[19] == b'+' || f[19] == b'-') && f[22] == b':' {
        let base = civil_ms(f)?;
        let oh = d2(f, 20)?;
        let om = d2(f, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        let off = (oh * 60 + om) * 60_000;
        return Some(if f[19] == b'+' { base - off } else { base + off });
    }
    parse_f3(f)
}

#[inline(always)]
fn parse_f3(f: &[u8]) -> Option<i64> {
    if f.is_empty() {
        return None;
    }
    if f.len() <= 18 {
        let mut v: u64 = 0;
        let mut bad = 0u8;
        for &c in f {
            let d = c.wrapping_sub(b'0');
            bad |= (d > 9) as u8;
            v = v * 10 + d as u64;
        }
        return if bad != 0 { None } else { Some(v as i64) };
    }
    let mut v: u64 = 0;
    let mut overflow = false;
    for &c in f {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        match v.checked_mul(10).and_then(|x| x.checked_add(d as u64)) {
            Some(x) => v = x,
            None => overflow = true,
        }
    }
    if overflow || v > i64::MAX as u64 {
        return None;
    }
    Some(v as i64)
}

/// Per-thread cache of "YYYY-MM-DD" -> epoch ms of that midnight (or invalid).
struct DateCache {
    keys: [u64; 64],
    vals: [i64; 64],
}
const BAD_DATE: i64 = i64::MIN;
impl DateCache {
    fn new() -> Self {
        DateCache { keys: [u64::MAX; 64], vals: [0; 64] }
    }
    #[inline(always)]
    fn day_ms(&mut self, f: &[u8]) -> i64 {
        // f has at least 19 bytes; f[4] and f[7] are checked to be '-'
        let k1 = u64::from_le_bytes(f[0..8].try_into().unwrap());
        let k = (k1 & 0x00FF_FF00_FFFF_FFFF) | ((f[8] as u64) << 32) | ((f[9] as u64) << 56);
        let i = (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize;
        if self.keys[i] == k {
            return self.vals[i];
        }
        let v = date_ms_slow(f);
        self.keys[i] = k;
        self.vals[i] = v;
        v
    }
}

#[cold]
fn date_ms_slow(f: &[u8]) -> i64 {
    let r = (|| {
        let y = d2(f, 0)? * 100 + d2(f, 2)?;
        let mo = d2(f, 5)?;
        let d = d2(f, 8)?;
        if !(1..=12).contains(&mo) || d < 1 {
            return None;
        }
        let dim = match mo {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            _ => {
                if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                    29
                } else {
                    28
                }
            }
        };
        if d > dim {
            return None;
        }
        Some(days_from_civil(y, mo, d) * DAY_MS)
    })();
    r.unwrap_or(BAD_DATE)
}

#[inline(always)]
fn civil_ms_c(f: &[u8], dc: &mut DateCache) -> Option<i64> {
    // f.len() >= 20 (F1 or F2). Separators f[4]='-', f[7]='-', f[10]='T' in one word.
    let sep = u64::from_le_bytes(f[4..12].try_into().unwrap());
    if sep & 0x00FF_0000_FF00_00FF != 0x0054_0000_2D00_002D {
        return None;
    }
    // "HH:MM:SS" at 11..19: digits must be 0..=9 and ':' must match, checked per byte without branches
    let x = u64::from_le_bytes(f[11..19].try_into().unwrap()) ^ 0x3030_3A30_303A_3030;
    if (x | x.wrapping_add(0x7676_7F76_767F_7676)) & 0x8080_8080_8080_8080 != 0 {
        return None;
    }
    let h = (x & 0xFF) * 10 + ((x >> 8) & 0xFF);
    let mi = ((x >> 24) & 0xFF) * 10 + ((x >> 32) & 0xFF);
    let sec = ((x >> 48) & 0xFF) * 10 + (x >> 56);
    if h > 23 || mi > 59 || sec > 59 {
        return None;
    }
    let day = dc.day_ms(f);
    if day == BAD_DATE {
        return None;
    }
    Some(day + ((h * 3600 + mi * 60 + sec) * 1000) as i64)
}

/// parse_ts with a date cache.
#[inline(always)]
fn parse_ts_c(f: &[u8], dc: &mut DateCache) -> Option<i64> {
    let n = f.len();
    if n == 20 && f[19] == b'Z' {
        return civil_ms_c(f, dc);
    }
    if n == 25 && (f[19] == b'+' || f[19] == b'-') && f[22] == b':' {
        let base = civil_ms_c(f, dc)?;
        let oh = d2(f, 20)?;
        let om = d2(f, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        let off = (oh * 60 + om) * 60_000;
        return Some(if f[19] == b'+' { base - off } else { base + off });
    }
    parse_f3(f)
}

/// Digits only, value 1..=10,000,000.
#[inline(always)]
fn parse_price(f: &[u8]) -> Option<u32> {
    if f.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    for &c in f {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v * 10 + d as u64;
        if v > 10_000_000 {
            return None;
        }
    }
    if v == 0 { None } else { Some(v as u32) }
}

// ---------------------------------------------------------------- input files

struct Chunk {
    file: usize,
    start: usize,
    end: usize,
}

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false) && p.is_file())
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

fn map_files_raw(paths: &[PathBuf]) -> Vec<Option<Mmap>> {
    paths
        .iter()
        .map(|p| {
            let f = File::open(p).ok()?;
            let len = f.metadata().ok()?.len();
            if len == 0 {
                return None;
            }
            let m = unsafe { Mmap::map(&f).ok()? };
            let _ = m.advise(memmap2::Advice::Sequential);
            Some(m)
        })
        .collect()
}

enum Buf {
    M(Mmap),
    V(Vec<u8>),
}
impl std::ops::Deref for Buf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Buf::M(m) => m,
            Buf::V(v) => v,
        }
    }
}

/// Loads files either as mmaps or by reading them into memory, 8 MB pieces in parallel.
fn load_files(paths: &[PathBuf], read: bool, threads: usize) -> Vec<Option<Buf>> {
    if !read {
        return map_files_raw(paths).into_iter().map(|m| m.map(Buf::M)).collect();
    }
    use std::os::unix::fs::FileExt;
    let files: Vec<Option<(File, usize)>> = paths
        .iter()
        .map(|p| {
            let f = File::open(p).ok()?;
            let len = f.metadata().ok()?.len() as usize;
            if len == 0 { None } else { Some((f, len)) }
        })
        .collect();
    let mut bufs: Vec<Vec<u8>> = files
        .iter()
        .map(|f| match f {
            Some((_, len)) => {
                let mut v = Vec::with_capacity(*len);
                unsafe { v.set_len(*len) };
                v
            }
            None => Vec::new(),
        })
        .collect();
    const PIECE: usize = 8 << 20;
    struct Piece {
        file: usize,
        off: usize,
        ptr: usize,
        len: usize,
    }
    let mut pieces: Vec<Piece> = Vec::new();
    for (fi, b) in bufs.iter_mut().enumerate() {
        let mut off = 0;
        while off < b.len() {
            let len = PIECE.min(b.len() - off);
            pieces.push(Piece { file: fi, off, ptr: b.as_mut_ptr() as usize + off, len });
            off += len;
        }
    }
    parallel(threads, pieces.len(), || (), |_, i| {
        let pc = &pieces[i];
        let (f, _) = files[pc.file].as_ref().unwrap();
        let dst = unsafe { std::slice::from_raw_parts_mut(pc.ptr as *mut u8, pc.len) };
        f.read_exact_at(dst, pc.off as u64).expect("read");
    });
    bufs.into_iter().map(|v| if v.is_empty() { None } else { Some(Buf::V(v)) }).collect()
}

fn make_chunks(maps: &[Option<Buf>], target: usize) -> Vec<Chunk> {
    let mut out = Vec::new();
    for (fi, m) in maps.iter().enumerate() {
        let Some(m) = m else { continue };
        let data: &[u8] = m;
        let body = match memchr::memchr(b'\n', data) {
            Some(p) => p + 1,
            None => continue, // header only
        };
        let mut s = body;
        while s < data.len() {
            let mut e = (s + target).min(data.len());
            if e < data.len() {
                match memchr::memchr(b'\n', &data[e..]) {
                    Some(p) => e = e + p + 1,
                    None => e = data.len(),
                }
            }
            out.push(Chunk { file: fi, start: s, end: e });
            s = e;
        }
    }
    out.sort_by_key(|c| std::cmp::Reverse(c.end - c.start));
    out
}

fn chunk_data<'a>(maps: &'a [Option<Buf>], c: &Chunk) -> &'a [u8] {
    &maps[c.file].as_ref().unwrap()[c.start..c.end]
}

fn read_small_csv<const N: usize>(path: &Path) -> Vec<[Vec<u8>; N]> {
    let data = std::fs::read(path).unwrap_or_default();
    let mut out = Vec::new();
    let Some(p) = memchr::memchr(b'\n', &data) else { return out };
    scan_lines::<N>(&data[p + 1..], |f| out.push(std::array::from_fn(|i| f[i].to_vec())));
    out
}

/// Runs `work(state, task)` for every task on `threads` threads; returns the states.
fn parallel<S: Send>(threads: usize, tasks: usize, init: impl Fn() -> S + Sync, work: impl Fn(&mut S, usize) + Sync) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut st = init();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= tasks {
                            break;
                        }
                        work(&mut st, i);
                    }
                    st
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Like `parallel`, but runs `fin(state, worker_index)` on each worker after it runs out of tasks.
fn parallel_fin<S: Send>(
    threads: usize,
    tasks: usize,
    init: impl Fn() -> S + Sync,
    work: impl Fn(&mut S, usize) + Sync,
    fin: impl Fn(&mut S, usize) + Sync,
) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|wi| {
                let (next, init, work, fin) = (&next, &init, &work, &fin);
                sc.spawn(move || {
                    let mut st = init();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= tasks {
                            break;
                        }
                        work(&mut st, i);
                    }
                    fin(&mut st, wi);
                    st
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn buckets<T>(n: usize, cap: usize) -> Vec<Vec<T>> {
    (0..n).map(|_| Vec::with_capacity(cap)).collect()
}

// ---------------------------------------------------------------- records

const INVALID: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct ERec {
    eid: u64,
    user: u64,
    ingest: u32, // seconds, relative
    ts: u32,     // ms since window_start; INVALID if the copy fails validation
    pc: u32,     // purchase: price << 8 | category; 0 otherwise
}

#[derive(Clone, Copy)]
struct Slot {
    eid: u64,
    ingest: u32,
    idx: u32,
}

#[derive(Clone, Copy)]
struct SRec {
    person: u64,
    ts: u32,
    pc: u32,
}

#[derive(Clone, Copy)]
struct CRec {
    person: u64,
    cid: u64,
    ts: u32, // ms since window_start - 7 days
    camp: u32,
}

struct URow {
    person: u64,
    day: i32,
    sessions: u32,
    events: u32,
    revenue: i64,
    top: u16, // NO_CAT if none
}
const NO_CAT: u16 = u16::MAX;

/// Thread-time accumulators (ns), filled only in diag builds.
static ACC: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
const A_PARSE: usize = 0;
const A_INS: usize = 1;
const A_EMIT: usize = 2;
const A_CLK: usize = 3;
const A_SESS: usize = 4;
const A_SESS_MAX: usize = 5;
const A_LINKS: usize = 6;
const A_WRITE: usize = 7;
macro_rules! timed {
    ($slot:expr, $e:expr) => {{
        if cfg!(feature = "diag") {
            let t = Instant::now();
            let r = $e;
            let ns = t.elapsed().as_nanos() as u64;
            ACC[$slot].fetch_add(ns, Ordering::Relaxed);
            if $slot == A_SESS {
                ACC[A_SESS_MAX].fetch_max(ns, Ordering::Relaxed);
            }
            r
        } else {
            $e
        }
    }};
}
#[allow(dead_code)]
fn acc_report() -> String {
    let names = ["parse", "ins", "emit", "clk", "sess", "smax", "links", "write"];
    let mut out = String::from(" T");
    for (i, n) in names.iter().enumerate() {
        out.push_str(&format!(" {}={:.0}", n, ACC[i].swap(0, Ordering::Relaxed) as f64 / 1e6));
    }
    out
}

static COMPACT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LOGBUF: Mutex<String> = Mutex::new(String::new());

fn cpu_times() -> (f64, f64) {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
    (t(ru.ru_utime), t(ru.ru_stime))
}

fn log(start: &Instant, msg: &str) {
    if COMPACT.load(Ordering::Relaxed) {
        let w = msg.split(|c: char| c == ':' || c == '(' || c == ';').next().unwrap_or("");
        let word = match w.trim() {
            "setup" => "su",
            "links built" => "lk",
            "events parse threads done" => "ep",
            "events parsed" => "ej",
            "events deduped, clicks parsed" => "dd",
            "sessions built" => "ss",
            "done" => "wr",
            x => x,
        };
        let (u, sy) = cpu_times();
        LOGBUF.lock().unwrap().push_str(&format!(" {}={:.0}[u{:.0},s{:.0}]", word, start.elapsed().as_secs_f64() * 1000.0, u * 1000.0, sy * 1000.0));
    } else {
        eprintln!("[{:7.3}s] {}", start.elapsed().as_secs_f64(), msg);
    }
}

fn manifest_ts(text: &str, key: &str) -> i64 {
    let i = text.find(&format!("\"{}\"", key)).expect("manifest key");
    let rest = &text[i + key.len() + 2..];
    let rest = &rest[rest.find(':').unwrap() + 1..];
    let q1 = rest.find('"').unwrap();
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"').unwrap();
    parse_ts(rest[..q2].as_bytes()).expect("manifest ts")
}

struct EvCtx<'a> {
    ws: i64,
    we: i64,
    ingest_base: i64,
    products: &'a FastMap<u64, u32>,
    unknown_cat: u32,
}

#[inline(never)]
fn parse_events(data: &[u8], c: &EvCtx, out: &mut [Vec<ERec>], longs: &mut LongMap) {
    let mask = out.len() - 1;
    let mut dc = DateCache::new();
    let dend = data.as_ptr() as usize + data.len();
    scan_lines::<7>(data, |f| {
        if f[0].is_empty() {
            return;
        }
        let Some(ingest) = parse_ts_c(f[3], &mut dc) else { return };
        let eid = eq_key_at(f[0], dend);
        let ing_rel = (ingest.div_euclid(1000) - c.ingest_base).clamp(0, u32::MAX as i64) as u32;
        let mut rec = ERec { eid, user: 0, ingest: ing_rel, ts: INVALID, pc: 0 };
        'v: {
            if f[1].is_empty() || f[5].is_empty() {
                break 'v;
            }
            let Some(ts) = parse_ts_c(f[2], &mut dc) else { break 'v };
            if ts < c.ws || ts >= c.we || ingest - ts > DAY_MS {
                break 'v;
            }
            let ty = f[4];
            if ty == b"purchase" {
                let Some(p) = parse_price(f[6]) else { break 'v };
                let cat = *c.products.get(&eq_key_at(f[5], dend)).unwrap_or(&c.unknown_cat);
                rec.pc = p << 8 | cat;
            } else if ty != b"view" && ty != b"cart" {
                break 'v;
            }
            rec.user = key_at(f[1], dend, longs);
            rec.ts = (ts - c.ws) as u32;
        }
        out[part_of(eid, mask)].push(rec);
    });
}

/// Flat open-addressing map user key -> person key. Key 0 never occurs (empty slot).
struct PersonMap {
    slots: Vec<[u64; 2]>,
    mask: usize,
    len: usize,
}
impl PersonMap {
    fn with_capacity(n: usize) -> Self {
        let cap = (n * 2).next_power_of_two().max(16);
        PersonMap { slots: vec![[0, 0]; cap], mask: cap - 1, len: 0 }
    }
    #[inline(always)]
    fn slot(&self, k: u64) -> usize {
        (mix(k) as usize) & self.mask
    }
    fn insert(&mut self, k: u64, v: u64) {
        let mut i = self.slot(k);
        loop {
            let s = &mut self.slots[i];
            if s[0] == 0 {
                *s = [k, v];
                self.len += 1;
                return;
            }
            if s[0] == k {
                s[1] = v;
                return;
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn prefetch(&self, k: u64) {
        let p = unsafe { self.slots.as_ptr().add(self.slot(k)) } as *const i8;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
        }
    }
    /// Person of user `k` (itself when unlinked).
    #[inline(always)]
    fn person(&self, k: u64) -> u64 {
        let mut i = self.slot(k);
        loop {
            let s = unsafe { self.slots.get_unchecked(i) };
            if s[0] == k {
                return s[1];
            }
            if s[0] == 0 {
                return k;
            }
            i = (i + 1) & self.mask;
        }
    }
    fn len(&self) -> usize {
        self.len
    }
}

static DEDUP_DENSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Marks every copy except the one with the smallest ingest_ts (per event_id) as dead.
/// Records of bucket `b` are numbered across worker vectors in order.
#[inline(never)]
fn dedup_insert(ev_parts: &[Vec<Vec<ERec>>], b: usize, table: &mut Vec<Slot>, dead: &mut Vec<bool>) {
    let n: usize = ev_parts.iter().map(|t| t[b].len()).sum();
    // dense: 1.5n rounded up (load 0.33-0.67), else 2n (load 0.25-0.5)
    let want = if DEDUP_DENSE.load(Ordering::Relaxed) { n + n / 2 } else { n * 2 };
    let cap = want.next_power_of_two().max(64);
    let mask = cap - 1;
    table.clear();
    table.resize(cap, Slot { eid: 0, ingest: 0, idx: u32::MAX });
    dead.clear();
    dead.resize(n, false);
    let mut g = 0u32;
    for t in ev_parts.iter() {
        for r in t[b].iter() {
            let mut h = (mix(r.eid) as usize) & mask;
            loop {
                let s = unsafe { table.get_unchecked_mut(h) };
                if s.idx == u32::MAX {
                    *s = Slot { eid: r.eid, ingest: r.ingest, idx: g };
                    break;
                }
                if s.eid == r.eid {
                    if r.ingest < s.ingest {
                        dead[s.idx as usize] = true;
                        s.ingest = r.ingest;
                        s.idx = g;
                    } else {
                        dead[g as usize] = true;
                    }
                    break;
                }
                h = (h + 1) & mask;
            }
            g += 1;
        }
    }
}

/// Emits surviving (winning, valid) events in record order, mapped to persons and scattered.
#[inline(never)]
fn dedup_emit(ev_parts: &[Vec<Vec<ERec>>], b: usize, dead: &[bool], pm: &PersonMap, out: &mut [Vec<SRec>]) {
    let qm = out.len() - 1;
    const BATCH: usize = 16;
    let mut buf: [(u64, u32, u32); BATCH] = [(0, 0, 0); BATCH];
    let mut n = 0usize;
    let flush = |buf: &[(u64, u32, u32)], out: &mut [Vec<SRec>]| {
        for &(user, ts, pc) in buf {
            let person = pm.person(user);
            unsafe { out.get_unchecked_mut(part_of(person, qm)) }.push(SRec { person, ts, pc });
        }
    };
    let mut g = 0usize;
    for t in ev_parts.iter() {
        for r in t[b].iter() {
            let d = unsafe { *dead.get_unchecked(g) };
            g += 1;
            if d || r.ts == INVALID {
                continue;
            }
            pm.prefetch(r.user);
            buf[n] = (r.user, r.ts, r.pc);
            n += 1;
            if n == BATCH {
                flush(&buf, out);
                n = 0;
            }
        }
    }
    flush(&buf[..n], out);
}

#[inline(never)]
fn parse_clicks(
    data: &[u8],
    file: usize,
    base: *const u8,
    cs: i64,
    we: i64,
    campaigns: &FastMap<u64, u32>,
    pm: &PersonMap,
    out: &mut [Vec<CRec>],
    longs: &mut LongMap,
) {
    let qm = out.len() - 1;
    let mut dc = DateCache::new();
    const BATCH: usize = 16;
    let mut buf = [CRec { person: 0, cid: 0, ts: 0, camp: 0 }; BATCH];
    let mut n = 0usize;
    let dend = data.as_ptr() as usize + data.len();
    scan_lines::<4>(data, |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let Some(ts) = parse_ts_c(f[3], &mut dc) else { return };
        if ts < cs || ts >= we {
            return;
        }
        let Some(&camp) = campaigns.get(&eq_key_at(f[2], dend)) else { return };
        let u = eq_key_at(f[1], dend);
        pm.prefetch(u);
        let cid = match fast_key_at(f[0], dend) {
            Some(v) => v,
            None if f[0].len() < 4096 => {
                let off = f[0].as_ptr() as usize - base as usize;
                SLOW | ((file as u64) << 52) | ((off as u64) << 12) | f[0].len() as u64
            }
            None => {
                let k = SLOW | (0x3FF << 52) | (hash_bytes(f[0]) & ((1 << 52) - 1));
                longs.entry(k).or_insert_with(|| f[0].into());
                k
            }
        };
        buf[n] = CRec { person: u, cid, ts: (ts - cs) as u32, camp };
        n += 1;
        if n == BATCH {
            for mut r in buf {
                r.person = pm.person(r.person);
                unsafe { out.get_unchecked_mut(part_of(r.person, qm)) }.push(r);
            }
            n = 0;
        }
    });
    for mut r in buf.into_iter().take(n) {
        r.person = pm.person(r.person);
        out[part_of(r.person, qm)].push(r);
    }
}

fn build_person_map(maps: &[Option<Buf>], chunks: &[Chunk]) -> (PersonMap, LongMap) {
    let mut longs = LongMap::default();
    let mut ids: FastMap<u64, u32> = FastMap::default();
    let mut keys: Vec<u64> = Vec::new();
    let mut parent: Vec<u32> = Vec::new();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            let g = p[p[x as usize] as usize];
            p[x as usize] = g;
            x = g;
        }
        x
    }
    let total: usize = chunks.iter().map(|c| c.end - c.start).sum();
    ids.reserve(total / 12);
    for c in chunks {
        let d = chunk_data(maps, c);
        let dend = d.as_ptr() as usize + d.len();
        scan_lines::<2>(d, |f| {
            if f[0].is_empty() || f[1].is_empty() {
                return;
            }
            let a = key_at(f[0], dend, &mut longs);
            let b = key_at(f[1], dend, &mut longs);
            if a == b {
                return;
            }
            let mut id = |k: u64| -> u32 {
                *ids.entry(k).or_insert_with(|| {
                    let i = keys.len() as u32;
                    keys.push(k);
                    parent.push(i);
                    i
                })
            };
            let ia = id(a);
            let ib = id(b);
            let ra = find(&mut parent, ia);
            let rb = find(&mut parent, ib);
            if ra != rb {
                if ra < rb {
                    parent[rb as usize] = ra;
                } else {
                    parent[ra as usize] = rb;
                }
            }
        });
    }
    let lv = Longs(vec![&longs]);
    let n = keys.len();
    let mut name: Vec<u64> = vec![u64::MAX; n];
    let mut has = vec![false; n];
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        if !has[r] || key_cmp(keys[i], name[r], &lv) == std::cmp::Ordering::Less {
            name[r] = keys[i];
            has[r] = true;
        }
    }
    let mut m = PersonMap::with_capacity(n);
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        if name[r] != keys[i] {
            m.insert(keys[i], name[r]);
        }
    }
    drop(lv);
    (m, longs)
}

struct Cfg {
    threads: usize,
    read: bool,
    p: usize,
    q: usize,
    dense: bool,
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    let data_dir = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out_dir = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let threads: usize = std::env::var("NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
        .clamp(1, 64);
    let cfg = Cfg {
        threads,
        read: env_usize("SS_READ", 0) == 1,
        p: env_usize("SS_P", 1024).next_power_of_two(),
        q: env_usize("SS_Q", 1024).next_power_of_two(),
        dense: env_usize("SS_DENSE", 0) == 1,
    };
    #[cfg(feature = "diag")]
    {
        COMPACT.store(true, Ordering::Relaxed);
        let scratch = PathBuf::from(std::env::var("SCRATCH_DIR").unwrap_or_else(|_| "/scratch".into()));
        let gen_dir = scratch.join("gen20");
        let generator = std::env::var("SS_GENERATOR").unwrap_or_else(|_| "/app/generate".into());
        if Path::new(&generator).exists() {
            let t = Instant::now();
            let st = std::process::Command::new(&generator)
                .args(["--seed", "7", "--events", "20000000", "--clicks", "5000000", "--output"])
                .arg(&gen_dir)
                .stdout(std::process::Stdio::null())
                .status();
            eprintln!("generate 20M: {:?} in {:.2}s", st, t.elapsed().as_secs_f64());
            let big_out = scratch.join("out20");
            let big = [
                Cfg { threads, read: false, p: 1024, q: 1024, dense: false }, // baseline
                Cfg { threads, read: false, p: 4096, q: 1024, dense: false }, // smaller dedup tables (L2 fit)
                Cfg { threads, read: false, p: 1024, q: 4096, dense: false }, // smaller session tables
                Cfg { threads, read: false, p: 4096, q: 4096, dense: false }, // both small
                Cfg { threads, read: false, p: 4096, q: 4096, dense: true },  // both small + dense dedup
                Cfg { threads: 1, read: false, p: 1024, q: 1024, dense: false }, // single-thread CPU work
            ];
            for c in &big {
                let _ = std::fs::remove_dir_all(&big_out);
                std::fs::create_dir_all(&big_out).unwrap();
                let (u0, s0) = cpu_times();
                pipeline(c, &gen_dir, &big_out);
                let (u1, s1) = cpu_times();
                let line = std::mem::take(&mut *LOGBUF.lock().unwrap());
                eprintln!("B t{} p{} q{} d{} cpu{:.2}/{:.2} {}{}", c.threads, c.p, c.q, c.dense as u8, u1 - u0, s1 - s0, line, acc_report());
            }
            unsafe { std::env::set_var("BENCH_PARSE", "1") };
            for th in [1usize, 8] {
                unsafe { std::env::set_var("BENCH_THREADS", th.to_string()) };
                pipeline(&big[0], &gen_dir, &big_out);
            }
        } else {
            eprintln!("no generator at {}", generator);
        }
        let configs: [Cfg; 0] = [];
        let _unused = [
            Cfg { threads, read: false, p: 1024, q: 1024, dense: false },
            Cfg { threads, read: true, p: 1024, q: 1024, dense: false },
            Cfg { threads, read: true, p: 256, q: 256, dense: false },
            Cfg { threads, read: true, p: 64, q: 64, dense: false },
            Cfg { threads, read: false, p: 64, q: 64, dense: false },
            Cfg { threads: 4, read: true, p: 1024, q: 1024, dense: false },
            Cfg { threads: 1, read: true, p: 1024, q: 1024, dense: false },
            Cfg { threads: 1, read: true, p: 64, q: 64, dense: false },
            Cfg { threads, read: true, p: 1024, q: 1024, dense: false },
        ];
        for c in &configs {
            pipeline(c, &data_dir, &out_dir);
            let line = std::mem::take(&mut *LOGBUF.lock().unwrap());
            eprintln!("cfg t={} read={} p={}:{}", c.threads, c.read as u8, c.p, line);
        }
        let _ = diag::bench_parse;
        eprintln!("DIAG build: exiting 1 on purpose to get logs");
        std::process::exit(1);
    }
    #[allow(unreachable_code)]
    pipeline(&cfg, &data_dir, &out_dir);
    // skip freeing GBs of buffers and unmapping inputs; outputs are already closed
    std::process::exit(0);
}

fn pipeline(cfg: &Cfg, data_dir: &Path, out_dir: &Path) {
    let t0 = Instant::now();
    DEDUP_DENSE.store(cfg.dense, Ordering::Relaxed);
    let threads = cfg.threads;
    let manifest = std::fs::read_to_string(data_dir.join("manifest.json")).expect("manifest");
    let ws = manifest_ts(&manifest, "window_start");
    let we = manifest_ts(&manifest, "window_end");
    let week = 7 * DAY_MS;
    let cs = ws - week; // click window start
    assert!(we - cs < INVALID as i64, "window too long");
    assert!(csv_files(&data_dir.join("clicks")).len() < 0x3FF, "too many click files");
    let ingest_base = ws.div_euclid(1000) - (1i64 << 31);

    // ---- lookups
    let prod_rows = read_small_csv::<2>(&data_dir.join("products.csv"));
    let mut cat_names: Vec<Vec<u8>> = prod_rows.iter().map(|r| r[1].clone()).collect();
    cat_names.push(b"unknown".to_vec());
    cat_names.sort();
    cat_names.dedup();
    assert!(cat_names.len() <= 256, "too many categories");
    let cat_index: HashMap<Vec<u8>, u32> = cat_names.iter().enumerate().map(|(i, c)| (c.clone(), i as u32)).collect();
    let unknown_cat = cat_index[&b"unknown"[..]];
    let mut products: FastMap<u64, u32> = FastMap::default();
    for r in &prod_rows {
        products.entry(eq_key(&r[0])).or_insert(cat_index[&r[1]]);
    }
    let camp_rows = read_small_csv::<2>(&data_dir.join("campaigns.csv"));
    let mut campaigns: FastMap<u64, u32> = FastMap::default();
    let mut camp_ids: Vec<Vec<u8>> = Vec::new();
    let mut camp_channels: Vec<Vec<u8>> = Vec::new();
    for r in &camp_rows {
        let k = eq_key(&r[0]);
        if campaigns.contains_key(&k) {
            continue;
        }
        campaigns.insert(k, camp_ids.len() as u32);
        camp_ids.push(r[0].clone());
        camp_channels.push(r[1].clone());
    }
    let ncamp = camp_ids.len();
    let organic = ncamp as u32;

    // ---- map input files
    let ev_paths = csv_files(&data_dir.join("events"));
    let ck_paths = csv_files(&data_dir.join("clicks"));
    let lk_paths = csv_files(&data_dir.join("links"));
    let ev_maps = load_files(&ev_paths, cfg.read, threads);
    let ck_maps = load_files(&ck_paths, cfg.read, threads);
    let lk_maps = load_files(&lk_paths, cfg.read, threads);
    let chunk_target = 2 << 20;
    let ev_chunks = make_chunks(&ev_maps, chunk_target);
    let ck_chunks = make_chunks(&ck_maps, chunk_target);
    let lk_chunks = make_chunks(&lk_maps, chunk_target);
    let ev_bytes: usize = ev_chunks.iter().map(|c| c.end - c.start).sum();
    let ck_bytes: usize = ck_chunks.iter().map(|c| c.end - c.start).sum();
    log(&t0, &format!("setup: {} MB events, {} MB clicks", ev_bytes >> 20, ck_bytes >> 20));

    if std::env::var_os("BENCH_PARSE").is_some() {
        // Component costs of event parsing using the production helpers (cached timestamps,
        // windowed keys). Results are black_boxed; allocation is per variant; teardown is untimed.
        let n = ev_chunks.len().min(env_usize("BENCH_N", 100000));
        let bt = env_usize("BENCH_THREADS", 1).max(1);
        let lines: usize = (0..n).map(|i| memchr::memchr_iter(b'\n', chunk_data(&ev_maps, &ev_chunks[i])).count()).sum();
        let ctx = EvCtx { ws, we, ingest_base, products: &products, unknown_cat };
        let mut line = format!("PB t{}", bt);
        for variant in 0..6usize {
            let per = lines / bt + 1024;
            let t = Instant::now();
            let states = parallel(
                bt,
                n,
                || {
                    let bk: Vec<Vec<ERec>> = match variant {
                        4 => vec![Vec::with_capacity(per)],
                        5 => buckets::<ERec>(1024, per / 1024 * 5 / 4 + 16),
                        _ => Vec::new(),
                    };
                    (bk, 0u64, LongMap::default())
                },
                |st, i| {
                    let d = chunk_data(&ev_maps, &ev_chunks[i]);
                    let dend = d.as_ptr() as usize + d.len();
                    let mut dc = DateCache::new();
                    let sink = &mut st.1;
                    match variant {
                        0 => scan_lines::<7>(d, |f| *sink = sink.wrapping_add(f[0].len() as u64 ^ f[6].len() as u64)),
                        1 => scan_lines::<7>(d, |f| {
                            let a = parse_ts_c(f[2], &mut dc).unwrap_or(0) as u64;
                            let b = parse_ts_c(f[3], &mut dc).unwrap_or(0) as u64;
                            *sink = sink.wrapping_add(a ^ b.rotate_left(7));
                        }),
                        2 => scan_lines::<7>(d, |f| *sink = sink.wrapping_add(eq_key_at(f[0], dend) ^ eq_key_at(f[1], dend).rotate_left(9))),
                        3 => scan_lines::<7>(d, |f| {
                            let a = parse_ts_c(f[2], &mut dc).unwrap_or(0) as u64;
                            let b = parse_ts_c(f[3], &mut dc).unwrap_or(0) as u64;
                            *sink = sink.wrapping_add(eq_key_at(f[0], dend) ^ eq_key_at(f[1], dend).rotate_left(9) ^ a ^ b.rotate_left(7));
                        }),
                        _ => parse_events(d, &ctx, &mut st.0, &mut st.2),
                    }
                },
            );
            let el = t.elapsed().as_secs_f64();
            std::hint::black_box(&states);
            let recs: usize = states.iter().map(|s| s.0.iter().map(|v| v.len()).sum::<usize>()).sum();
            std::hint::black_box(recs);
            drop(states);
            line.push_str(&format!(" v{}={:.1}", variant, el * 1e9 * bt as f64 / lines as f64));
        }
        eprintln!("{} ns/line/thread: tok,+ts,+keys,+ts+keys,full-1vec,full-1024", line);
        return;
    }

    let P: usize = cfg.p;
    let Q: usize = cfg.q;
    let QM: usize = Q - 1;
    let est_events = ev_bytes / 80 + 1024;
    let est_clicks = ck_bytes / 45 + 1024;

    // ---- links (background) + events parse
    let ((person_map, link_longs), ev_states) = std::thread::scope(|sc| {
        let lk = sc.spawn(|| {
            let m = timed!(A_LINKS, build_person_map(&lk_maps, &lk_chunks));
            log(&t0, "links built");
            m
        });
        let cap = est_events / threads / P * 5 / 4 + 16;
        let ctx = EvCtx { ws, we, ingest_base, products: &products, unknown_cat };
        let ev_states: Vec<(Vec<Vec<ERec>>, LongMap)> = parallel(
            threads,
            ev_chunks.len(),
            || (buckets::<ERec>(P, cap), LongMap::default()),
            |st, i| timed!(A_PARSE, parse_events(chunk_data(&ev_maps, &ev_chunks[i]), &ctx, &mut st.0, &mut st.1)),
        );
        log(&t0, "events parse threads done");
        (lk.join().unwrap(), ev_states)
    });
    let mut long_maps: Vec<LongMap> = vec![link_longs];
    let mut ev_parts: Vec<Vec<Vec<ERec>>> = Vec::with_capacity(ev_states.len());
    for (parts, lm) in ev_states {
        ev_parts.push(parts);
        long_maps.push(lm);
    }
    log(&t0, &format!("events parsed ({} chunks); links: {} remapped ids", ev_chunks.len(), person_map.len()));

    // ---- dedup events + parse clicks, both into person buckets
    struct Mid {
        ev: Vec<Vec<SRec>>,
        ck: Vec<Vec<CRec>>,
        table: Vec<Slot>,
        dead: Vec<bool>,
        longs: LongMap,
    }
    let ev_cap = est_events / threads / Q * 5 / 4 + 16;
    let ck_cap = est_clicks / threads / Q * 5 / 4 + 16;
    let mids: Vec<Mid> = parallel(
        threads,
        P + ck_chunks.len(),
        || Mid { ev: buckets(Q, ev_cap), ck: buckets(Q, ck_cap), table: Vec::new(), dead: Vec::new(), longs: LongMap::default() },
        |st, task| {
            if task < P {
                let b = task;
                timed!(A_INS, dedup_insert(&ev_parts, b, &mut st.table, &mut st.dead));
                timed!(A_EMIT, dedup_emit(&ev_parts, b, &st.dead, &person_map, &mut st.ev));
            } else {
                let c = &ck_chunks[task - P];
                timed!(A_CLK, parse_clicks(chunk_data(&ck_maps, c), c.file, ck_maps[c.file].as_ref().unwrap().as_ptr(), cs, we, &campaigns, &person_map, &mut st.ck, &mut st.longs));
            }
        },
    );

    std::mem::forget(ev_parts);
    let mut mids = mids;
    for m in mids.iter_mut() {
        long_maps.push(std::mem::take(&mut m.longs));
    }
    let longs = Longs(long_maps.iter().filter(|m| !m.is_empty()).collect());
    let longs = &longs;
    log(&t0, "events deduped, clicks parsed");

    // ---- sessionize + attribute per person bucket
    let day0 = ws.div_euclid(DAY_MS);
    let ndays = ((we - 1).div_euclid(DAY_MS) - day0 + 1) as usize;
    let ncols = ncamp + 1;
    let ncat = cat_names.len();
    let mut order: Vec<(usize, usize)> = (0..Q).map(|q| (mids.iter().map(|m| m.ev[q].len()).sum::<usize>(), q)).collect();
    order.sort_unstable_by(|a, b| b.cmp(a));
    let ud_dir = out_dir.join("user_daily");
    let cd_dir = out_dir.join("campaign_daily");
    std::fs::create_dir_all(&ud_dir).unwrap();
    std::fs::create_dir_all(&cd_dir).unwrap();
    let states: Vec<St> = parallel_fin(
        threads,
        Q,
        || St {
            rows: Vec::with_capacity(est_events / 8 / threads + 16),
            camp: vec![[0; 4]; ncols * ndays],
            counts: vec![0; ncols],
            active: Vec::new(),
            pos: vec![0; ncols],
            cat_sum: vec![0; ncat],
            cat_touched: Vec::new(),
            evs: Vec::new(),
            cks: Vec::new(),
            lslots: Vec::new(),
            lkeys: Vec::new(),
            lcount: Vec::new(),
            lids: Vec::new(),
            loff: Vec::new(),
            radix_tmp: Vec::new(),
            radix_ctmp: Vec::new(),
        },
        |st, qi| timed!(A_SESS, {
            let q = order[qi].1;
            let mut evs = std::mem::take(&mut st.evs);
            let mut cks = std::mem::take(&mut st.cks);
            let n_ev: usize = mids.iter().map(|m| m.ev[q].len()).sum();
            let n_ck: usize = mids.iter().map(|m| m.ck[q].len()).sum();
            // local dense person ids
            // table sized for distinct people (grows at 50% load), not for events
            let mut cap = (n_ev / 8).next_power_of_two().max(256);
            let mut mask = cap - 1;
            st.lslots.clear();
            st.lslots.resize(cap, (0, 0));
            st.lkeys.clear();
            st.lcount.clear();
            st.lids.clear();
            st.lids.reserve(n_ev.max(n_ck));
            for m in &mids {
                for e in &m.ev[q] {
                    let mut h = (mix(e.person) as usize) & mask;
                    let lid = loop {
                        let sl = unsafe { st.lslots.get_unchecked_mut(h) };
                        if sl.0 == e.person {
                            break sl.1;
                        }
                        if sl.0 == 0 {
                            let l = st.lkeys.len() as u32;
                            *sl = (e.person, l);
                            st.lkeys.push(e.person);
                            st.lcount.push(0);
                            break l;
                        }
                        h = (h + 1) & mask;
                    };
                    st.lcount[lid as usize] += 1;
                    st.lids.push(lid);
                    if st.lkeys.len() * 2 > cap {
                        cap *= 2;
                        mask = cap - 1;
                        st.lslots.clear();
                        st.lslots.resize(cap, (0, 0));
                        for (l, &k) in st.lkeys.iter().enumerate() {
                            let mut h = (mix(k) as usize) & mask;
                            while st.lslots[h].0 != 0 {
                                h = (h + 1) & mask;
                            }
                            st.lslots[h] = (k, l as u32);
                        }
                    }
                }
            }
            let np = st.lkeys.len();
            // event offsets
            st.loff.clear();
            st.loff.reserve(np + 1);
            let mut acc = 0u32;
            for &c in &st.lcount {
                st.loff.push(acc);
                acc += c;
            }
            st.loff.push(acc);
            evs.clear();
            evs.reserve(n_ev);
            // SAFETY: SRec is plain data; the counting placement below writes every slot in 0..n_ev exactly once
            unsafe { evs.set_len(n_ev) };
            {
                let mut cur: Vec<u32> = st.loff[..np].to_vec();
                let mut k = 0usize;
                for m in &mids {
                    for e in &m.ev[q] {
                        let l = st.lids[k] as usize;
                        k += 1;
                        let at = cur[l] as usize;
                        cur[l] += 1;
                        unsafe { *evs.get_unchecked_mut(at) = *e };
                    }
                }
            }
            // clicks of persons that have events
            st.lids.clear();
            let mut ccount = vec![0u32; np + 1];
            for m in &mids {
                for c in &m.ck[q] {
                    let mut h = (mix(c.person) as usize) & mask;
                    let lid = loop {
                        let sl = unsafe { st.lslots.get_unchecked(h) };
                        if sl.0 == c.person {
                            break sl.1;
                        }
                        if sl.0 == 0 {
                            break u32::MAX;
                        }
                        h = (h + 1) & mask;
                    };
                    if lid != u32::MAX {
                        ccount[lid as usize] += 1;
                    }
                    st.lids.push(lid);
                }
            }
            let mut coff = vec![0u32; np + 1];
            let mut acc = 0u32;
            for l in 0..np {
                coff[l] = acc;
                acc += ccount[l];
            }
            coff[np] = acc;
            cks.clear();
            cks.reserve(acc as usize);
            // SAFETY: as above, every slot in 0..acc is written exactly once
            unsafe { cks.set_len(acc as usize) };
            {
                let mut cur: Vec<u32> = coff[..np].to_vec();
                let mut k = 0usize;
                for m in &mids {
                    for c in &m.ck[q] {
                        let l = st.lids[k];
                        k += 1;
                        if l == u32::MAX {
                            continue;
                        }
                        let at = cur[l as usize] as usize;
                        cur[l as usize] += 1;
                        cks[at] = *c;
                    }
                }
            }
            for l in 0..np {
                let es = &mut evs[st.loff[l] as usize..st.loff[l + 1] as usize];
                if es.len() > 1 {
                    sort_srec_by_ts(es, &mut st.radix_tmp);
                }
                let cs_ = &mut cks[coff[l] as usize..coff[l + 1] as usize];
                if cs_.len() > 1 {
                    sort_crec(cs_, &mut st.radix_ctmp, &ck_maps, longs);
                }
                let person = st.lkeys[l];
                let (e0, e1) = (st.loff[l] as usize, st.loff[l + 1] as usize);
                let (c0, c1) = (coff[l] as usize, coff[l + 1] as usize);
                process_person(st, person, &evs[e0..e1], &cks[c0..c1], ws, day0, ndays, organic);
            }
            st.evs = evs;
            st.cks = cks;
        }),
        |st, wi| {
            // write this worker's user_daily rows while other workers finish their buckets
            let path = ud_dir.join(format!("part-{:05}.parquet", wi));
            timed!(A_WRITE, write_user_daily(&path, &st.rows, &cat_names, longs));
        },
    );
    let mids = std::mem::ManuallyDrop::new(mids);
    let _ = &mids;
    log(&t0, "sessions built");

    // ---- campaign_daily: merge
    let mut camp = vec![[0i64; 4]; ncols * ndays];
    for s in &states {
        for (a, b) in camp.iter_mut().zip(s.camp.iter()) {
            for x in 0..4 {
                a[x] += b[x];
            }
        }
    }

    log(&t0, "campaigns merged");
    // ---- write campaign_daily (user_daily was written by the session workers)
    write_campaign_daily(&cd_dir.join("part-00000.parquet"), &camp, ndays, day0, &camp_ids, &camp_channels);
    log(&t0, "done");
    if !cfg!(feature = "diag") {
        std::mem::forget(states);
        std::mem::forget(long_maps);
        std::mem::forget(ev_maps);
        std::mem::forget(ck_maps);
        std::mem::forget(lk_maps);
        std::mem::forget(person_map);
    } else {
        drop(std::mem::ManuallyDrop::into_inner(mids));
    }
}

struct St {
    rows: Vec<URow>,
    camp: Vec<[i64; 4]>, // [camp * ndays + day] -> sessions, purchasing, revenue, assisted
    counts: Vec<u32>,
    active: Vec<u32>,
    pos: Vec<u32>,
    cat_sum: Vec<u64>,
    cat_touched: Vec<u16>,
    evs: Vec<SRec>,
    cks: Vec<CRec>,
    lslots: Vec<(u64, u32)>,
    lkeys: Vec<u64>,
    lcount: Vec<u32>,
    lids: Vec<u32>,
    loff: Vec<u32>,
    radix_tmp: Vec<SRec>,
    radix_ctmp: Vec<CRec>,
}

const RADIX_MIN: usize = 2048;

/// LSD radix sort of u32 keys, 2 passes of 16 bits, stable. `tmp` is scratch.
fn radix_sort_by<T: Copy>(v: &mut [T], tmp: &mut Vec<T>, key: impl Fn(&T) -> u32) {
    let n = v.len();
    tmp.clear();
    tmp.reserve(n);
    // SAFETY: T is Copy; every slot of tmp in 0..n is written by the scatter before it is read
    unsafe { tmp.set_len(n) };
    let mut c0 = vec![0u32; 1 << 16];
    let mut c1 = vec![0u32; 1 << 16];
    for x in v.iter() {
        let k = key(x);
        c0[(k & 0xFFFF) as usize] += 1;
        c1[(k >> 16) as usize] += 1;
    }
    let (mut a0, mut a1) = (0u32, 0u32);
    for i in 0..(1 << 16) {
        let t0 = c0[i];
        c0[i] = a0;
        a0 += t0;
        let t1 = c1[i];
        c1[i] = a1;
        a1 += t1;
    }
    for x in v.iter() {
        let d = (key(x) & 0xFFFF) as usize;
        tmp[c0[d] as usize] = *x;
        c0[d] += 1;
    }
    for x in tmp.iter() {
        let d = (key(x) >> 16) as usize;
        v[c1[d] as usize] = *x;
        c1[d] += 1;
    }
}

fn sort_srec_by_ts(v: &mut [SRec], tmp: &mut Vec<SRec>) {
    if v.len() >= RADIX_MIN {
        radix_sort_by(v, tmp, |e| e.ts);
    } else {
        v.sort_unstable_by_key(|e| e.ts);
    }
}

/// Sort clicks by (ts, click_id byte order).
fn sort_crec(v: &mut [CRec], tmp: &mut Vec<CRec>, files: &[Option<Buf>], longs: &Longs) {
    if v.len() >= RADIX_MIN {
        radix_sort_by(v, tmp, |c| c.ts);
        // resolve ties in ts by click id
        let mut i = 0;
        while i < v.len() {
            let mut j = i + 1;
            while j < v.len() && v[j].ts == v[i].ts {
                j += 1;
            }
            if j - i > 1 {
                v[i..j].sort_unstable_by(|a, b| cid_cmp(a.cid, b.cid, files, longs));
            }
            i = j;
        }
    } else {
        v.sort_unstable_by(|a, b| a.ts.cmp(&b.ts).then_with(|| cid_cmp(a.cid, b.cid, files, longs)));
    }
}

fn flush_day(st: &mut St, person: u64, day: i32, sessions: u32, events: u32, revenue: i64) {
    let mut top = NO_CAT;
    let mut best = 0u64;
    for &c in st.cat_touched.iter() {
        let s = st.cat_sum[c as usize];
        if s > best || (s == best && c < top) {
            best = s;
            top = c;
        }
        st.cat_sum[c as usize] = 0;
    }
    st.cat_touched.clear();
    st.rows.push(URow { person, day, sessions, events, revenue, top });
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn process_person(st: &mut St, person: u64, evs: &[SRec], pc: &[CRec], ws: i64, day0: i64, ndays: usize, organic: u32) {
    let week = 7 * DAY_MS;
    let mut lo = 0usize; // first click with ts >= start - 7d
    let mut hi = 0usize; // first click with ts > start
    let mut cur_day: i32 = i32::MIN;
    let (mut u_sess, mut u_ev, mut u_rev) = (0u32, 0u32, 0i64);
    let n = evs.len();
    let mut k = 0usize;
    while k < n {
        let start = evs[k].ts as i64;
        let day = (ws + start).div_euclid(DAY_MS) as i32;
        if day != cur_day {
            if cur_day != i32::MIN {
                flush_day(st, person, cur_day, u_sess, u_ev, u_rev);
            }
            cur_day = day;
            u_sess = 0;
            u_ev = 0;
            u_rev = 0;
        }
        let mut last = evs[k].ts;
        let mut m = k;
        let mut rev = 0i64;
        let mut purch = false;
        while m < n && evs[m].ts - last <= 1_800_000 {
            let e = &evs[m];
            last = e.ts;
            if e.pc != 0 {
                purch = true;
                let price = (e.pc >> 8) as i64;
                rev += price;
                let c = (e.pc & 0xFF) as usize;
                if st.cat_sum[c] == 0 {
                    st.cat_touched.push(c as u16);
                }
                st.cat_sum[c] += price as u64;
            }
            m += 1;
        }
        u_sess += 1;
        u_ev += (m - k) as u32;
        u_rev += rev;
        k = m;

        // attribution, in click coordinates (ms since window_start - 7 days)
        if !pc.is_empty() {
            let sc = start + week;
            while hi < pc.len() && (pc[hi].ts as i64) <= sc {
                let c = pc[hi].camp as usize;
                if st.counts[c] == 0 {
                    st.pos[c] = st.active.len() as u32;
                    st.active.push(c as u32);
                }
                st.counts[c] += 1;
                hi += 1;
            }
            while lo < hi && (pc[lo].ts as i64) < sc - week {
                remove_click(st, pc[lo].camp as usize);
                lo += 1;
            }
        }
        let di = (day as i64 - day0) as usize;
        let credit = if hi > lo {
            let t = pc[hi - 1].ts;
            let mut f = hi - 1;
            while f > lo && pc[f - 1].ts == t {
                f -= 1;
            }
            pc[f].camp
        } else {
            organic
        };
        let cell = &mut st.camp[credit as usize * ndays + di];
        cell[0] += 1;
        cell[1] += purch as i64;
        cell[2] += rev;
        for ai in 0..st.active.len() {
            let a = st.active[ai] as usize;
            st.camp[a * ndays + di][3] += 1;
        }
    }
    if cur_day != i32::MIN {
        flush_day(st, person, cur_day, u_sess, u_ev, u_rev);
    }
    for &a in &st.active {
        st.counts[a as usize] = 0;
    }
    st.active.clear();
}

#[inline(always)]
fn remove_click(st: &mut St, c: usize) {
    st.counts[c] -= 1;
    if st.counts[c] == 0 {
        let p = st.pos[c] as usize;
        let last_c = *st.active.last().unwrap();
        st.active.swap_remove(p);
        if p < st.active.len() {
            st.pos[last_c as usize] = p as u32;
        }
    }
}

// ---------------------------------------------------------------- parquet

use parquet::basic::Compression;
use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use std::sync::Arc;

fn props() -> Arc<WriterProperties> {
    Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::None)
            .set_writer_version(WriterVersion::PARQUET_1_0)
            .build(),
    )
}

fn write_user_daily(path: &Path, rows: &[URow], cats: &[Vec<u8>], longs: &Longs) {
    if std::env::var_os("SS_PQ_CRATE").is_none() {
        return write_user_daily_fast(path, rows, cats, longs);
    }
    let schema = Arc::new(
        parse_message_type(
            "message user_daily {
                required binary user_id (STRING);
                required int32 day (DATE);
                required int64 sessions;
                required int64 events;
                required int64 revenue_cents;
                optional binary top_category (STRING);
            }",
        )
        .unwrap(),
    );
    let cat_ba: Vec<ByteArray> = cats.iter().map(|c| ByteArray::from(c.clone())).collect();
    let file = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(std::io::BufWriter::with_capacity(1 << 20, file), schema, props()).unwrap();
    for rows in rows.chunks(1 << 20) {
        let mut rg = w.next_row_group().unwrap();
        let mut col = 0;
        while let Some(mut c) = rg.next_column().unwrap() {
            match col {
                0 => {
                    let mut buf: Vec<u8> = Vec::with_capacity(rows.len() * 12);
                    let mut ends: Vec<u32> = Vec::with_capacity(rows.len());
                    let mut last = u64::MAX;
                    let mut sb = [0u8; KEY_LEN];
                    for r in rows {
                        if r.person != last {
                            if r.person & SLOW != 0 {
                                buf.extend_from_slice(longs.get(r.person));
                            } else {
                                let n = short_bytes(r.person, &mut sb);
                                buf.extend_from_slice(&sb[..n]);
                            }
                            last = r.person;
                        }
                        ends.push(buf.len() as u32);
                    }
                    let all = bytes::Bytes::from(buf);
                    let mut v: Vec<ByteArray> = Vec::with_capacity(rows.len());
                    let mut start = 0usize;
                    let mut prev_end = u32::MAX;
                    for &e in &ends {
                        if e != prev_end {
                            // new person: its bytes end at e; they start where the previous person ended
                            start = if prev_end == u32::MAX { 0 } else { prev_end as usize };
                            prev_end = e;
                        }
                        v.push(ByteArray::from(all.slice(start..e as usize)));
                    }
                    c.typed::<ByteArrayType>().write_batch(&v, None, None).unwrap();
                }
                1 => {
                    let v: Vec<i32> = rows.iter().map(|r| r.day).collect();
                    c.typed::<Int32Type>().write_batch(&v, None, None).unwrap();
                }
                2 | 3 | 4 => {
                    let v: Vec<i64> = rows
                        .iter()
                        .map(|r| match col {
                            2 => r.sessions as i64,
                            3 => r.events as i64,
                            _ => r.revenue,
                        })
                        .collect();
                    c.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
                }
                _ => {
                    let defs: Vec<i16> = rows.iter().map(|r| (r.top != NO_CAT) as i16).collect();
                    let v: Vec<ByteArray> =
                        rows.iter().filter(|r| r.top != NO_CAT).map(|r| cat_ba[r.top as usize].clone()).collect();
                    c.typed::<ByteArrayType>().write_batch(&v, Some(&defs), None).unwrap();
                }
            }
            c.close().unwrap();
            col += 1;
        }
        rg.close().unwrap();
    }
    w.close().unwrap();
}

fn write_user_daily_fast(path: &Path, rows: &[URow], cats: &[Vec<u8>], longs: &Longs) {
    let n = rows.len();
    let mut ubytes: Vec<u8> = Vec::with_capacity(n * 12);
    let mut uends: Vec<u32> = Vec::with_capacity(n);
    let mut last = u64::MAX;
    let mut last_range = (0usize, 0usize);
    let mut sb = [0u8; KEY_LEN];
    for r in rows {
        if r.person != last {
            let start = ubytes.len();
            if r.person & SLOW != 0 {
                ubytes.extend_from_slice(longs.get(r.person));
            } else {
                let k = short_bytes(r.person, &mut sb);
                ubytes.extend_from_slice(&sb[..k]);
            }
            last = r.person;
            last_range = (start, ubytes.len());
        } else {
            // repeat the person's bytes (PLAIN encoding stores each value)
            let (a, b) = last_range;
            ubytes.extend_from_within(a..b);
        }
        uends.push(ubytes.len() as u32);
    }
    let day: Vec<i32> = rows.iter().map(|r| r.day).collect();
    let ses: Vec<i32> = rows.iter().map(|r| r.sessions as i32).collect();
    let evn: Vec<i32> = rows.iter().map(|r| r.events as i32).collect();
    let rev: Vec<i64> = rows.iter().map(|r| r.revenue).collect();
    let present: Vec<bool> = rows.iter().map(|r| r.top != NO_CAT).collect();
    let cidx: Vec<u32> = rows.iter().filter(|r| r.top != NO_CAT).map(|r| r.top as u32).collect();
    let mut dbytes: Vec<u8> = Vec::new();
    let mut dends: Vec<u32> = Vec::new();
    for c in cats {
        dbytes.extend_from_slice(c);
        dends.push(dbytes.len() as u32);
    }
    pq::write(
        path,
        &[
            pq::Col::Str { name: "user_id", bytes: &ubytes, ends: &uends },
            pq::Col::Date { name: "day", vals: &day },
            pq::Col::I32 { name: "sessions", vals: &ses },
            pq::Col::I32 { name: "events", vals: &evn },
            pq::Col::I64 { name: "revenue_cents", vals: &rev },
            pq::Col::OptDictStr { name: "top_category", present: &present, dbytes: &dbytes, dends: &dends, idx: &cidx },
        ],
    )
    .expect("write user_daily");
}

fn write_campaign_daily(path: &Path, camp: &[[i64; 4]], ndays: usize, day0: i64, ids: &[Vec<u8>], channels: &[Vec<u8>]) {
    if std::env::var_os("SS_PQ_CRATE").is_none() {
        let (mut ib, mut ie, mut cb, mut ce) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut day: Vec<i32> = Vec::new();
        let mut vals: [Vec<i64>; 4] = Default::default();
        let ncols = camp.len() / ndays.max(1);
        for c in 0..ncols {
            for d in 0..ndays {
                let cell = camp[c * ndays + d];
                if cell[0] == 0 && cell[3] == 0 {
                    continue;
                }
                let (id, ch): (&[u8], &[u8]) = if c < ids.len() { (&ids[c], &channels[c]) } else { (b"organic", b"organic") };
                ib.extend_from_slice(id);
                ie.push(ib.len() as u32);
                cb.extend_from_slice(ch);
                ce.push(cb.len() as u32);
                day.push((day0 + d as i64) as i32);
                for x in 0..4 {
                    vals[x].push(cell[x]);
                }
            }
        }
        pq::write(
            path,
            &[
                pq::Col::Str { name: "campaign_id", bytes: &ib, ends: &ie },
                pq::Col::Str { name: "channel", bytes: &cb, ends: &ce },
                pq::Col::Date { name: "day", vals: &day },
                pq::Col::I64 { name: "sessions", vals: &vals[0] },
                pq::Col::I64 { name: "purchasing_sessions", vals: &vals[1] },
                pq::Col::I64 { name: "revenue_cents", vals: &vals[2] },
                pq::Col::I64 { name: "assisted_sessions", vals: &vals[3] },
            ],
        )
        .expect("write campaign_daily");
        return;
    }
    let schema = Arc::new(
        parse_message_type(
            "message campaign_daily {
                required binary campaign_id (STRING);
                required binary channel (STRING);
                required int32 day (DATE);
                required int64 sessions;
                required int64 purchasing_sessions;
                required int64 revenue_cents;
                required int64 assisted_sessions;
            }",
        )
        .unwrap(),
    );
    let mut cid: Vec<ByteArray> = Vec::new();
    let mut ch: Vec<ByteArray> = Vec::new();
    let mut day: Vec<i32> = Vec::new();
    let mut vals: [Vec<i64>; 4] = Default::default();
    let ncols = camp.len() / ndays.max(1);
    for c in 0..ncols {
        for d in 0..ndays {
            let cell = camp[c * ndays + d];
            if cell[0] == 0 && cell[3] == 0 {
                continue;
            }
            if c < ids.len() {
                cid.push(ByteArray::from(ids[c].clone()));
                ch.push(ByteArray::from(channels[c].clone()));
            } else {
                cid.push(ByteArray::from(&b"organic"[..]));
                ch.push(ByteArray::from(&b"organic"[..]));
            }
            day.push((day0 + d as i64) as i32);
            for x in 0..4 {
                vals[x].push(cell[x]);
            }
        }
    }
    let file = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(std::io::BufWriter::with_capacity(1 << 20, file), schema, props()).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = 0;
    while let Some(mut c) = rg.next_column().unwrap() {
        match col {
            0 => c.typed::<ByteArrayType>().write_batch(&cid, None, None).unwrap(),
            1 => c.typed::<ByteArrayType>().write_batch(&ch, None, None).unwrap(),
            2 => c.typed::<Int32Type>().write_batch(&day, None, None).unwrap(),
            k => c.typed::<Int64Type>().write_batch(&vals[k - 3], None, None).unwrap(),
        };
        c.close().unwrap();
        col += 1;
    }
    rg.close().unwrap();
    w.close().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_matches_sort() {
        let mut x: u64 = 5;
        for n in [0usize, 1, 2047, 2048, 5000, 70000] {
            let mut v: Vec<SRec> = (0..n)
                .map(|i| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    SRec { person: i as u64, ts: (x >> 20) as u32 % if i % 3 == 0 { 1000 } else { u32::MAX }, pc: 0 }
                })
                .collect();
            let mut w = v.clone();
            let mut tmp = Vec::new();
            radix_sort_by(&mut v, &mut tmp, |e| e.ts);
            w.sort_by_key(|e| e.ts); // stable, like LSD radix
            assert!(v.iter().zip(w.iter()).all(|(a, b)| a.ts == b.ts && a.person == b.person));
        }
    }

    #[test]
    fn simd_key_matches() {
        let mut x: u64 = 0x1234_5678_9abc;
        let alpha: &[u8] = b"09_az/:`{^AZ \r\x00\xff\x7f\x80\x2f\x3a\x60\x7b\x5e\x5f\x61\x7a\x30\x39";
        let mut buf = [0u8; 40];
        for _ in 0..400000 {
            for b in buf.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = if x & 3 != 0 { b"abcdefghijklmnopqrstuvwxyz0123456789_"[(x >> 8) as usize % 37] } else { alpha[(x >> 8) as usize % alpha.len()] };
            }
            let off = (x >> 33) as usize % 8;
            let n = (x >> 40) as usize % 15;
            let f = &buf[off..off + n];
            assert_eq!(unsafe { simd_key16(f) }, fast_key(f), "{:?}", f);
            assert_eq!(fast_key_at(f, buf.as_ptr() as usize + buf.len()), fast_key(f));
        }
    }

    #[test]
    fn civil_swar_matches() {
        let mut dc = DateCache::new();
        let mut x: u64 = 99;
        let bases: [&[u8]; 4] = [b"2026-10-03T23:58:10Z", b"2028-02-29T00:00:00Z", b"2026-10-03T19:58:10-04:00", b"1999-12-31T23:59:59+23:59"];
        let subst = b"0123456789-:TZ+ t/;\x00\xff\x2f\x3a\x8a";
        for _ in 0..300000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let mut v = bases[(x >> 3) as usize % 4].to_vec();
            for k in 0..((x >> 6) % 3) {
                let pos = ((x >> (10 + 8 * k)) as usize) % v.len();
                v[pos] = subst[((x >> (20 + 7 * k)) as usize) % subst.len()];
            }
            assert_eq!(parse_ts_c(&v, &mut dc), parse_ts(&v), "{:?}", String::from_utf8_lossy(&v));
        }
    }

    #[test]
    fn fast_key_win_matches() {
        let mut x: u64 = 77;
        let alpha = b"abz09_AZ ,\"\r\xff\x00m";
        for _ in 0..200000 {
            let mut buf = [0u8; 24];
            for b in buf.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = if x & 1 == 0 { b"abcdefghijklmnopqrstuvwxyz0123456789_"[(x >> 8) as usize % 37] } else { alpha[(x >> 8) as usize % alpha.len()] };
            }
            let n = (x >> 30) as usize % 14;
            let f = &buf[..n];
            assert_eq!(unsafe { fast_key_win(f) }, fast_key(f), "{:?}", f);
        }
    }

    #[test]
    fn scan_unterminated() {
        // unterminated last line ending in ',': the old tokenizer dropped it (wrong per rules.md)
        let mut a: Vec<Vec<Vec<u8>>> = Vec::new();
        scan_lines2::<2>(b"a,b\n,", |f| a.push(f.iter().map(|x| x.to_vec()).collect()));
        assert_eq!(a, vec![vec![b"a".to_vec(), b"b".to_vec()], vec![vec![], vec![]]]);
        let cases: Vec<&[u8]> = vec![b"a,b\n,", b"a,", b",", b"a,b", b"a,b\nc,d", b"a\nb,c", b"\"a\",b", b"a,b,c\nd,e", b"a", b"a,b\nc"];
        for c in cases {
            let mut a: Vec<Vec<Vec<u8>>> = Vec::new();
            let mut b: Vec<Vec<Vec<u8>>> = Vec::new();
            scan_lines2::<2>(c, |f| a.push(f.iter().map(|x| x.to_vec()).collect()));
            scan_lines::<2>(c, |f| b.push(f.iter().map(|x| x.to_vec()).collect()));
            assert_eq!(a, b, "case {:?}", String::from_utf8_lossy(c));
        }
    }

    #[test]
    fn scan_lines_matches_ref() {
        let mut x: u64 = 0x9e37_79b9;
        let alphabet = b"ab,,,\"\"\n\n\r x1,";
        for round in 0..3000 {
            let len = (round * 7) % 700 + (round % 3) * 70_000 * ((round % 50 == 0) as usize);
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                v.push(alphabet[(x >> 20) as usize % alphabet.len()]);
            }
            for n in [2usize, 4, 7] {
                let mut a: Vec<Vec<Vec<u8>>> = Vec::new();
                let mut b: Vec<Vec<Vec<u8>>> = Vec::new();
                match n {
                    2 => {
                        scan_lines2::<2>(&v, |f| a.push(f.iter().map(|x| x.to_vec()).collect()));
                        scan_lines::<2>(&v, |f| b.push(f.iter().map(|x| x.to_vec()).collect()));
                    }
                    4 => {
                        scan_lines2::<4>(&v, |f| a.push(f.iter().map(|x| x.to_vec()).collect()));
                        scan_lines::<4>(&v, |f| b.push(f.iter().map(|x| x.to_vec()).collect()));
                    }
                    _ => {
                        scan_lines2::<7>(&v, |f| a.push(f.iter().map(|x| x.to_vec()).collect()));
                        scan_lines::<7>(&v, |f| b.push(f.iter().map(|x| x.to_vec()).collect()));
                    }
                }
                assert_eq!(a, b, "round {} n {}", round, n);
            }
        }
    }

    #[test]
    fn cached_ts_matches() {
        let mut dc = DateCache::new();
        let cases: Vec<&[u8]> = vec![
            b"2026-10-03T23:58:10Z", b"2026-10-03T19:58:10-04:00", b"1791071890000", b"0001791071890000",
            b"2028-02-29T00:00:00Z", b"2027-02-29T00:00:00Z", b"2026-13-01T00:00:00Z", b"2026-02-30T00:00:00Z",
            b"2026-10-03T24:00:00Z", b"2026-10-03T23:60:00Z", b"2026-10-03T23:00:60Z", b"2026-10-03 23:00:00Z",
            b"2026-10-03T23:00:00z", b"2026-10-03t23:00:00Z", b"2026-10-03T23:00:00+24:00", b"2026-10-03T23:00:00+05:60",
            b"2026-10-03T23:00:00+0200", b"2026-10-03T23:00:00-00:00", b"2026-10-03T23:00:00+23:59", b"1791071890000.0",
            b"1.79e12", b"-5", b"", b"86400000", b"123456789012345678901", b"9223372036854775807", b"9223372036854775808",
            b"2026-1a-03T23:58:10Z", b"2026-10-0xT23:58:10Z", b"2026x10-03T23:58:10Z", b"2026-10-31T23:58:10Z",
            b"2026-11-31T23:58:10Z", b"2000-02-29T00:00:00Z", b"1900-02-29T00:00:00Z", b"2026-00-10T00:00:00Z",
            b"2026-10-00T00:00:00Z", b"\xd9\xa2\xd9\xa5", b"2026-10-03T23:58:10Z\r",
        ];
        for _ in 0..3 {
            for c in &cases {
                assert_eq!(parse_ts_c(c, &mut dc), parse_ts(c), "{:?}", String::from_utf8_lossy(c));
            }
        }
    }

    #[test]
    fn mask64_matches_naive() {
        let mut x: u64 = 0x1234_5678;
        let alphabet = b",\"\na\rz0 \x00\xff\x2c\x8a\xa2";
        for _ in 0..20000 {
            let mut buf = [0u8; 64];
            for b in buf.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = if x & 3 == 0 { (x >> 8) as u8 } else { alphabet[(x >> 16) as usize % alphabet.len()] };
            }
            let m = |f: &dyn Fn(u8) -> bool| -> u64 { (0..64).filter(|&i| f(buf[i])).map(|i| 1u64 << i).sum() };
            let want = (m(&|c| c == b',' || c == b'\n'), m(&|c| c == b'\n'), m(&|c| c == b'"'));
            assert_eq!(unsafe { masks(buf.as_ptr()) }, want);
        }
    }

    #[test]
    fn keys_roundtrip_and_order() {
        let mut lm = LongMap::default();
        let ids: Vec<&[u8]> = vec![b"a", b"a0", b"a_", b"ab", b"u8w84o3t0", b"zzzzzzzzzz", b"0", b"c_1", b"c_10", b"c_2"];
        let odd: Vec<&[u8]> = vec![b"U1", b"u1 ", b" u1", b"abcdefghijk", b"e3\r", b"u1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4"];
        let keys: Vec<u64> = ids.iter().chain(odd.iter()).map(|a| key(a, &mut lm)).collect();
        let lv = Longs(vec![&lm]);
        let all: Vec<&[u8]> = ids.iter().chain(odd.iter()).cloned().collect();
        for (i, a) in all.iter().enumerate() {
            assert_eq!(key_bytes(keys[i], &lv), a.to_vec());
            assert_eq!(eq_key(a), keys[i]);
            for (j, b) in all.iter().enumerate() {
                assert_eq!(key_cmp(keys[i], keys[j], &lv), a.cmp(b), "{:?} {:?}", a, b);
            }
        }
    }
}
