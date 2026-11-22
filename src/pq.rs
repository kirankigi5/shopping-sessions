//! Minimal Parquet writer: one row group, one uncompressed PLAIN data page (v1) per column.
//! Columns are required except where a definition-level vector is given (max level 1).

use std::io::Write;

pub enum Col<'a> {
    /// UTF-8 strings: concatenated bytes + end offsets, one per value.
    Str { name: &'a str, bytes: &'a [u8], ends: &'a [u32] },
    /// Nullable UTF-8 strings: `present[i]` says whether row i has a value; values are for present rows only.
    OptStr { name: &'a str, present: &'a [bool], bytes: &'a [u8], ends: &'a [u32] },
    Date { name: &'a str, vals: &'a [i32] },
    I32 { name: &'a str, vals: &'a [i32] },
    I64 { name: &'a str, vals: &'a [i64] },
    /// Dictionary-encoded strings: dictionary (bytes + ends) and one index per row.
    DictStr { name: &'a str, dbytes: &'a [u8], dends: &'a [u32], idx: &'a [u32] },
    /// Nullable dictionary-encoded strings: indices for present rows only.
    OptDictStr { name: &'a str, present: &'a [bool], dbytes: &'a [u8], dends: &'a [u32], idx: &'a [u32] },
}

// ---- thrift compact protocol
struct Thrift {
    buf: Vec<u8>,
    last: Vec<i16>,
}
const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BIN: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;

impl Thrift {
    fn new() -> Self {
        Thrift { buf: Vec::new(), last: vec![0] }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }
    fn zz(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn field(&mut self, id: i16, ty: u8) {
        let last = *self.last.last().unwrap();
        let d = id - last;
        if d > 0 && d <= 15 {
            self.buf.push(((d as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            self.zz(id as i64);
        }
        *self.last.last_mut().unwrap() = id;
    }
    fn i32f(&mut self, id: i16, v: i32) {
        self.field(id, T_I32);
        self.zz(v as i64);
    }
    fn i64f(&mut self, id: i16, v: i64) {
        self.field(id, T_I64);
        self.zz(v);
    }
    fn strf(&mut self, id: i16, s: &str) {
        self.field(id, T_BIN);
        self.varint(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }
    fn begin_struct_field(&mut self, id: i16) {
        self.field(id, T_STRUCT);
        self.last.push(0);
    }
    fn begin_struct_elem(&mut self) {
        self.last.push(0);
    }
    fn end_struct(&mut self) {
        self.buf.push(0);
        self.last.pop();
    }
    fn list_header(&mut self, n: usize, elem: u8) {
        if n < 15 {
            self.buf.push(((n as u8) << 4) | elem);
        } else {
            self.buf.push(0xF0 | elem);
            self.varint(n as u64);
        }
    }
    fn listf(&mut self, id: i16, n: usize, elem: u8) {
        self.field(id, T_LIST);
        self.list_header(n, elem);
    }
}

fn page_header(uncompressed: usize, num_values: usize) -> Vec<u8> {
    data_page_header(uncompressed, num_values, 0)
}

fn data_page_header(uncompressed: usize, num_values: usize, encoding: i32) -> Vec<u8> {
    let mut t = Thrift::new();
    t.i32f(1, 0); // DATA_PAGE
    t.i32f(2, uncompressed as i32);
    t.i32f(3, uncompressed as i32);
    t.begin_struct_field(5); // DataPageHeader
    t.i32f(1, num_values as i32);
    t.i32f(2, encoding);
    t.i32f(3, 3); // RLE definition levels
    t.i32f(4, 3); // RLE repetition levels
    t.end_struct();
    t.buf.push(0); // end PageHeader
    t.buf
}

fn dict_page_header(uncompressed: usize, num_values: usize) -> Vec<u8> {
    let mut t = Thrift::new();
    t.i32f(1, 2); // DICTIONARY_PAGE
    t.i32f(2, uncompressed as i32);
    t.i32f(3, uncompressed as i32);
    t.begin_struct_field(7); // DictionaryPageHeader
    t.i32f(1, num_values as i32);
    t.i32f(2, 0); // PLAIN
    t.end_struct();
    t.buf.push(0);
    t.buf
}

/// Bit width byte + RLE/bit-packed hybrid (one bit-packed run) of `idx`, no length prefix.
fn put_dict_indices(out: &mut Vec<u8>, idx: &[u32], ndict: usize) {
    let w = if ndict <= 1 { 0 } else { 32 - ((ndict - 1) as u32).leading_zeros() } as usize;
    out.push(w as u8);
    if idx.is_empty() {
        return;
    }
    if w == 0 {
        // a single RLE run of value 0 (zero bytes of value)
        let mut h = (idx.len() as u64) << 1;
        while h >= 0x80 {
            out.push((h as u8) | 0x80);
            h >>= 7;
        }
        out.push(h as u8);
        return;
    }
    let groups = idx.len().div_ceil(8);
    let mut h = ((groups as u64) << 1) | 1;
    while h >= 0x80 {
        out.push((h as u8) | 0x80);
        h >>= 7;
    }
    out.push(h as u8);
    out.reserve(groups * w);
    let mut acc: u64 = 0;
    let mut nbits = 0usize;
    for g in 0..groups * 8 {
        let v = if g < idx.len() { idx[g] as u64 } else { 0 };
        acc |= v << nbits;
        nbits += w;
        while nbits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    if nbits > 0 {
        out.push(acc as u8);
    }
}

fn put_strings(out: &mut Vec<u8>, bytes: &[u8], ends: &[u32]) {
    out.reserve(ends.len() * 4 + bytes.len());
    let mut start = 0usize;
    for &e in ends {
        let e = e as usize;
        out.extend_from_slice(&((e - start) as u32).to_le_bytes());
        out.extend_from_slice(&bytes[start..e]);
        start = e;
    }
}

/// RLE/bit-packed hybrid, bit width 1, as one bit-packed run, with the 4-byte length prefix.
fn put_def_levels(out: &mut Vec<u8>, present: &[bool]) {
    let groups = present.len().div_ceil(8);
    let mut enc: Vec<u8> = Vec::with_capacity(groups + 10);
    let mut h = ((groups as u64) << 1) | 1;
    while h >= 0x80 {
        enc.push((h as u8) | 0x80);
        h >>= 7;
    }
    enc.push(h as u8);
    for g in 0..groups {
        let mut b = 0u8;
        for k in 0..8 {
            let i = g * 8 + k;
            if i < present.len() && present[i] {
                b |= 1 << k;
            }
        }
        enc.push(b);
    }
    out.extend_from_slice(&(enc.len() as u32).to_le_bytes());
    out.extend_from_slice(&enc);
}

struct ChunkMeta {
    name: String,
    ty: i32,
    num_values: usize,
    offset: usize,
    size: usize,
    dict_offset: Option<usize>,
    data_offset: usize,
}

pub fn write(path: &std::path::Path, cols: &[Col]) -> std::io::Result<()> {
    let nrows = match &cols[0] {
        Col::Str { ends, .. } => ends.len(),
        Col::OptStr { present, .. } => present.len(),
        Col::Date { vals, .. } => vals.len(),
        Col::I32 { vals, .. } => vals.len(),
        Col::I64 { vals, .. } => vals.len(),
        Col::DictStr { idx, .. } => idx.len(),
        Col::OptDictStr { present, .. } => present.len(),
    };
    let mut file: Vec<u8> = Vec::with_capacity(nrows * 48 + 1024);
    file.extend_from_slice(b"PAR1");
    let mut metas: Vec<ChunkMeta> = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    for c in cols {
        data.clear();
        let offset = file.len();
        let mut dict_offset = None;
        let (name, ty, enc) = match c {
            Col::Str { name, bytes, ends } => {
                put_strings(&mut data, bytes, ends);
                (*name, 6, 0)
            }
            Col::OptStr { name, present, bytes, ends } => {
                put_def_levels(&mut data, present);
                put_strings(&mut data, bytes, ends);
                (*name, 6, 0)
            }
            Col::Date { name, vals } | Col::I32 { name, vals } => {
                data.reserve(vals.len() * 4);
                for v in vals.iter() {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                (*name, 1, 0)
            }
            Col::I64 { name, vals } => {
                data.reserve(vals.len() * 8);
                for v in vals.iter() {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                (*name, 2, 0)
            }
            Col::DictStr { name, dbytes, dends, idx } => {
                let mut d = Vec::new();
                put_strings(&mut d, dbytes, dends);
                file.extend_from_slice(&dict_page_header(d.len(), dends.len()));
                file.extend_from_slice(&d);
                dict_offset = Some(offset);
                put_dict_indices(&mut data, idx, dends.len());
                (*name, 6, 8)
            }
            Col::OptDictStr { name, present, dbytes, dends, idx } => {
                let mut d = Vec::new();
                put_strings(&mut d, dbytes, dends);
                file.extend_from_slice(&dict_page_header(d.len(), dends.len()));
                file.extend_from_slice(&d);
                dict_offset = Some(offset);
                put_def_levels(&mut data, present);
                put_dict_indices(&mut data, idx, dends.len());
                (*name, 6, 8)
            }
        };
        let hdr = data_page_header(data.len(), nrows, enc);
        let data_offset = file.len();
        file.extend_from_slice(&hdr);
        file.extend_from_slice(&data);
        metas.push(ChunkMeta { name: name.to_string(), ty, num_values: nrows, offset, size: file.len() - offset, dict_offset, data_offset });
    }
    // ---- FileMetaData
    let mut t = Thrift::new();
    t.i32f(1, 1); // version
    t.listf(2, cols.len() + 1, T_STRUCT); // schema
    t.begin_struct_elem();
    t.strf(4, "schema");
    t.i32f(5, cols.len() as i32);
    t.end_struct();
    for c in cols {
        t.begin_struct_elem();
        let (name, ty, rep, conv, logical) = match c {
            Col::Str { name, .. } => (*name, 6, 0, Some(0), Some(1i16)),
            Col::OptStr { name, .. } => (*name, 6, 1, Some(0), Some(1)),
            Col::Date { name, .. } => (*name, 1, 0, Some(6), Some(6)),
            Col::I32 { name, .. } => (*name, 1, 0, None, None),
            Col::I64 { name, .. } => (*name, 2, 0, None, None),
            Col::DictStr { name, .. } => (*name, 6, 0, Some(0), Some(1)),
            Col::OptDictStr { name, .. } => (*name, 6, 1, Some(0), Some(1)),
        };
        t.i32f(1, ty);
        t.i32f(3, rep);
        t.strf(4, name);
        if let Some(cv) = conv {
            t.i32f(6, cv);
        }
        if let Some(lt) = logical {
            t.begin_struct_field(10); // LogicalType union
            t.begin_struct_field(lt); // StringType (1) or DateType (6): empty structs
            t.end_struct();
            t.end_struct();
        }
        t.end_struct();
    }
    t.i64f(3, nrows as i64);
    t.listf(4, 1, T_STRUCT); // row_groups
    t.begin_struct_elem();
    t.listf(1, metas.len(), T_STRUCT); // columns
    let mut total = 0usize;
    for m in &metas {
        total += m.size;
        t.begin_struct_elem();
        t.i64f(2, m.offset as i64); // file_offset
        t.begin_struct_field(3); // ColumnMetaData
        t.i32f(1, m.ty);
        if m.dict_offset.is_some() {
            t.listf(2, 3, T_I32);
            t.zz(0); // PLAIN (dictionary page)
            t.zz(3); // RLE (levels)
            t.zz(8); // RLE_DICTIONARY
        } else {
            t.listf(2, 2, T_I32);
            t.zz(0); // PLAIN
            t.zz(3); // RLE
        }
        t.listf(3, 1, T_BIN);
        t.varint(m.name.len() as u64);
        t.buf.extend_from_slice(m.name.as_bytes());
        t.i32f(4, 0); // UNCOMPRESSED
        t.i64f(5, m.num_values as i64);
        t.i64f(6, m.size as i64);
        t.i64f(7, m.size as i64);
        t.i64f(9, m.data_offset as i64); // data_page_offset
        if let Some(d) = m.dict_offset {
            t.i64f(11, d as i64); // dictionary_page_offset
        }
        t.end_struct();
        t.end_struct();
    }
    t.i64f(2, total as i64);
    t.i64f(3, nrows as i64);
    t.end_struct();
    t.strf(6, "shopping-sessions rust");
    t.buf.push(0); // end FileMetaData
    file.extend_from_slice(&t.buf);
    file.extend_from_slice(&(t.buf.len() as u32).to_le_bytes());
    file.extend_from_slice(b"PAR1");
    let mut f = std::fs::File::create(path)?;
    f.write_all(&file)?;
    Ok(())
}
