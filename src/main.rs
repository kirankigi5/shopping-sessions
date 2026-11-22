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
fn cid_bytes(k: u64, files: &[Option<Buf>], longs: &Longs) -> Vec<u8> {
    if k & SLOW == 0 {
        return key_bytes(k, longs);
    }
    let file = ((k >> 52) & 0x3FF) as usize;
    if file == 0x3FF {
        return longs.get(k).to_vec();
    }
    let off = ((k >> 12) & ((1 << 40) - 1)) as usize;
    let len = (k & 0xFFF) as usize;
    files[file].as_ref().unwrap()[off..off + len].to_vec()
}

#[inline(always)]
fn cid_cmp(a: u64, b: u64, files: &[Option<Buf>], longs: &Longs) -> std::cmp::Ordering {
    if (a | b) & SLOW == 0 {
        a.cmp(&b)
    } else {
        cid_bytes(a, files, longs).cmp(&cid_bytes(b, files, longs))
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

#[inline(always)]
fn key_cmp(a: u64, b: u64, longs: &Longs) -> std::cmp::Ordering {
    if (a | b) & SLOW == 0 {
        a.cmp(&b)
    } else if a == b {
        std::cmp::Ordering::Equal
    } else {
        key_bytes(a, longs).cmp(&key_bytes(b, longs))
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
fn scan_lines<'a, const N: usize>(data: &'a [u8], mut on_line: impl FnMut(&[&'a [u8]; N])) {
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
    if fstart < len {
        field!(len);
        if nf == N && qbefore - line_q0 == expect {
            on_line(&fields);
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
    if f[4] != b'-' || f[7] != b'-' || f[10] != b'T' || f[13] != b':' || f[16] != b':' {
        return None;
    }
    let h = d2(f, 11)?;
    let mi = d2(f, 14)?;
    let s = d2(f, 17)?;
    if h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let day = dc.day_ms(f);
    if day == BAD_DATE {
        return None;
    }
    Some(day + (h * 3600 + mi * 60 + s) * 1000)
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
        let word: String = msg.split(|c: char| c == ':' || c == '(' || c == ';').next().unwrap_or("").replace(' ', "_");
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
    scan_lines::<7>(data, |f| {
        if f[0].is_empty() {
            return;
        }
        let Some(ingest) = parse_ts_c(f[3], &mut dc) else { return };
        let eid = eq_key(f[0]);
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
                let cat = *c.products.get(&eq_key(f[5])).unwrap_or(&c.unknown_cat);
                rec.pc = p << 8 | cat;
            } else if ty != b"view" && ty != b"cart" {
                break 'v;
            }
            rec.user = key(f[1], longs);
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

#[inline(never)]
fn dedup_insert(ev_parts: &[Vec<Vec<ERec>>], b: usize, table: &mut Vec<Slot>) {
    let n: usize = ev_parts.iter().map(|t| t[b].len()).sum();
    let cap = (n * 2).next_power_of_two().max(64);
    let mask = cap - 1;
    table.clear();
    table.resize(cap, Slot { eid: 0, ingest: 0, idx: u32::MAX });
    for (ti, t) in ev_parts.iter().enumerate() {
        let tag = (ti as u32) << 26;
        for (i, r) in t[b].iter().enumerate() {
            let mut h = (mix(r.eid) as usize) & mask;
            loop {
                let s = &mut table[h];
                if s.idx == u32::MAX {
                    *s = Slot { eid: r.eid, ingest: r.ingest, idx: tag | i as u32 };
                    break;
                }
                if s.eid == r.eid {
                    if r.ingest < s.ingest {
                        s.ingest = r.ingest;
                        s.idx = tag | i as u32;
                    }
                    break;
                }
                h = (h + 1) & mask;
            }
        }
    }
}

#[inline(never)]
fn dedup_emit(ev_parts: &[Vec<Vec<ERec>>], b: usize, table: &[Slot], pm: &PersonMap, out: &mut [Vec<SRec>]) {
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
    for s in table.iter() {
        if s.idx == u32::MAX {
            continue;
        }
        let r = unsafe { ev_parts.get_unchecked((s.idx >> 26) as usize).get_unchecked(b).get_unchecked((s.idx & ((1 << 26) - 1)) as usize) };
        if r.ts == INVALID {
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
    scan_lines::<4>(data, |f| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let Some(ts) = parse_ts_c(f[3], &mut dc) else { return };
        if ts < cs || ts >= we {
            return;
        }
        let Some(&camp) = campaigns.get(&eq_key(f[2])) else { return };
        let u = eq_key(f[1]);
        pm.prefetch(u);
        let cid = match fast_key(f[0]) {
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
        scan_lines::<2>(chunk_data(maps, c), |f| {
            if f[0].is_empty() || f[1].is_empty() {
                return;
            }
            let a = key(f[0], &mut longs);
            let b = key(f[1], &mut longs);
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
                Cfg { threads, read: false, p: 1024, q: 1024 },
                Cfg { threads, read: false, p: 1024, q: 1024 },
                Cfg { threads, read: false, p: 1024, q: 1024 },
                Cfg { threads, read: false, p: 256, q: 256 },
                Cfg { threads, read: false, p: 4096, q: 4096 },
                Cfg { threads: 4, read: false, p: 1024, q: 1024 },
                Cfg { threads: 12, read: false, p: 1024, q: 1024 },
                Cfg { threads: 1, read: false, p: 1024, q: 1024 },
                Cfg { threads, read: true, p: 1024, q: 1024 },
            ];
            for c in &big {
                let _ = std::fs::remove_dir_all(&big_out);
                std::fs::create_dir_all(&big_out).unwrap();
                let (u0, s0) = cpu_times();
                pipeline(c, &gen_dir, &big_out);
                let (u1, s1) = cpu_times();
                let line = std::mem::take(&mut *LOGBUF.lock().unwrap());
                eprintln!("BIG t={} read={} p={} cpu u={:.2} s={:.2}:{}", c.threads, c.read as u8, c.p, u1 - u0, s1 - s0, line);
            }
        } else {
            eprintln!("no generator at {}", generator);
        }
        let configs = [
            Cfg { threads, read: false, p: 1024, q: 1024 },
            Cfg { threads, read: true, p: 1024, q: 1024 },
            Cfg { threads, read: true, p: 256, q: 256 },
            Cfg { threads, read: true, p: 64, q: 64 },
            Cfg { threads, read: false, p: 64, q: 64 },
            Cfg { threads: 4, read: true, p: 1024, q: 1024 },
            Cfg { threads: 1, read: true, p: 1024, q: 1024 },
            Cfg { threads: 1, read: true, p: 64, q: 64 },
            Cfg { threads, read: true, p: 1024, q: 1024 },
        ];
        for c in &configs {
            pipeline(c, &data_dir, &out_dir);
            let line = std::mem::take(&mut *LOGBUF.lock().unwrap());
            eprintln!("cfg t={} read={} p={}:{}", c.threads, c.read as u8, c.p, line);
        }
        diag::bench_parse(&data_dir);
        eprintln!("DIAG build: exiting 1 on purpose to get logs");
        std::process::exit(1);
    }
    #[allow(unreachable_code)]
    pipeline(&cfg, &data_dir, &out_dir);
}

fn pipeline(cfg: &Cfg, data_dir: &Path, out_dir: &Path) {
    let t0 = Instant::now();
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
        let n = ev_chunks.len().min(env_usize("BENCH_N", 400));
        let lines: usize = (0..n).map(|i| memchr::memchr_iter(b'\n', chunk_data(&ev_maps, &ev_chunks[i])).count()).sum();
        let ctx = EvCtx { ws, we, ingest_base, products: &products, unknown_cat };
        for round in 0..3 {
            for variant in 0..5 {
                let mut out: Vec<Vec<ERec>> = buckets(1024, lines / 1024 * 5 / 4 + 16);
                let mut sink = 0u64;
                let t = Instant::now();
                for i in 0..n {
                    let d = chunk_data(&ev_maps, &ev_chunks[i]);
                    match variant {
                        0 => scan_lines::<7>(d, |f| sink += f[0].len() as u64),
                        1 => scan_lines::<7>(d, |f| {
                            sink = sink.wrapping_add(parse_ts(f[2]).unwrap_or(0) as u64 ^ parse_ts(f[3]).unwrap_or(0) as u64)
                        }),
                        2 => scan_lines::<7>(d, |f| sink = sink.wrapping_add(eq_key(f[0]) ^ eq_key(f[1]))),
                        3 => scan_lines::<7>(d, |f| {
                            sink = sink.wrapping_add(eq_key(f[0]) ^ eq_key(f[1]) ^ parse_ts(f[2]).unwrap_or(0) as u64 ^ parse_ts(f[3]).unwrap_or(0) as u64)
                        }),
                        _ => parse_events(d, &ctx, &mut out, &mut LongMap::default()),
                    }
                }
                let el = t.elapsed().as_secs_f64();
                std::hint::black_box(sink);
                let kept: usize = out.iter().map(|v| v.len()).sum();
                eprintln!("round {} variant {} : {:.1} ns/line ({} recs)", round, variant, el * 1e9 / lines as f64, kept);
            }
        }
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
            let m = build_person_map(&lk_maps, &lk_chunks);
            log(&t0, "links built");
            m
        });
        let cap = est_events / threads / P * 5 / 4 + 16;
        let ctx = EvCtx { ws, we, ingest_base, products: &products, unknown_cat };
        let ev_states: Vec<(Vec<Vec<ERec>>, LongMap)> = parallel(
            threads,
            ev_chunks.len(),
            || (buckets::<ERec>(P, cap), LongMap::default()),
            |st, i| parse_events(chunk_data(&ev_maps, &ev_chunks[i]), &ctx, &mut st.0, &mut st.1),
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
        longs: LongMap,
    }
    let ev_cap = est_events / threads / Q * 5 / 4 + 16;
    let ck_cap = est_clicks / threads / Q * 5 / 4 + 16;
    let mids: Vec<Mid> = parallel(
        threads,
        P + ck_chunks.len(),
        || Mid { ev: buckets(Q, ev_cap), ck: buckets(Q, ck_cap), table: Vec::new(), longs: LongMap::default() },
        |st, task| {
            if task < P {
                let b = task;
                dedup_insert(&ev_parts, b, &mut st.table);
                dedup_emit(&ev_parts, b, &st.table, &person_map, &mut st.ev);
            } else {
                let c = &ck_chunks[task - P];
                parse_clicks(chunk_data(&ck_maps, c), c.file, ck_maps[c.file].as_ref().unwrap().as_ptr(), cs, we, &campaigns, &person_map, &mut st.ck, &mut st.longs);
            }
        },
    );

    drop(ev_parts);
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
    let states: Vec<St> = parallel(
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
        },
        |st, q| {
            let mut evs = std::mem::take(&mut st.evs);
            let mut cks = std::mem::take(&mut st.cks);
            evs.clear();
            cks.clear();
            for m in &mids {
                evs.extend_from_slice(&m.ev[q]);
                cks.extend_from_slice(&m.ck[q]);
            }
            evs.sort_unstable_by_key(|e| ((e.person as u128) << 32) | e.ts as u128);
            cks.sort_unstable_by(|a, b| {
                (a.person, a.ts).cmp(&(b.person, b.ts)).then_with(|| cid_cmp(a.cid, b.cid, &ck_maps, longs))
            });
            let mut ci = 0usize;
            let mut i = 0usize;
            while i < evs.len() {
                let person = evs[i].person;
                let mut j = i + 1;
                while j < evs.len() && evs[j].person == person {
                    j += 1;
                }
                while ci < cks.len() && cks[ci].person < person {
                    ci += 1;
                }
                let c0 = ci;
                while ci < cks.len() && cks[ci].person == person {
                    ci += 1;
                }
                process_person(st, person, &evs[i..j], &cks[c0..ci], ws, day0, ndays, organic);
                i = j;
            }
            st.evs = evs;
            st.cks = cks;
        },
    );
    drop(mids);
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

    // ---- write outputs
    let ud_dir = out_dir.join("user_daily");
    let cd_dir = out_dir.join("campaign_daily");
    std::fs::create_dir_all(&ud_dir).unwrap();
    std::fs::create_dir_all(&cd_dir).unwrap();
    std::thread::scope(|sc| {
        for (ti, s) in states.iter().enumerate() {
            let path = ud_dir.join(format!("part-{:05}.parquet", ti));
            let cat_names = &cat_names;
            sc.spawn(move || write_user_daily(&path, &s.rows, cat_names, longs));
        }
        let path = cd_dir.join("part-00000.parquet");
        let (camp, camp_ids, camp_channels) = (&camp, &camp_ids, &camp_channels);
        sc.spawn(move || write_campaign_daily(&path, camp, ndays, day0, camp_ids, camp_channels));
    });
    log(&t0, "done");
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
                    let v: Vec<ByteArray> = rows.iter().map(|r| ByteArray::from(key_bytes(r.person, longs))).collect();
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

fn write_campaign_daily(path: &Path, camp: &[[i64; 4]], ndays: usize, day0: i64, ids: &[Vec<u8>], channels: &[Vec<u8>]) {
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
