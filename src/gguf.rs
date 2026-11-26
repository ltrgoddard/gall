// gguf reader: mmap the file, index kv and tensors. nothing is copied.
//   let g = gguf::open(path);
//   g.u("inkling2.embedding_length"), g.f(..), g.nums(..),
//   g.strs("tokenizer.ggml.tokens") -> iterator of &[u8]
//   g.t["blk.0.attn_q.weight"] -> Tensor { dims, ty, off, len }
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub enum Val {
    U(u64),
    I(i64),
    F(f64),
    S(&'static [u8]),
    A(u32, usize, &'static [u8]), // item type, count, raw items
}

pub struct Tensor {
    pub name: &'static str,
    pub dims: Vec<usize>, // ggml order: dims[0] is the row width
    pub ty: u32,
    pub off: usize, // absolute file offset
    pub len: usize,
}

pub struct Gguf {
    pub map: &'static [u8],
    pub kv: HashMap<&'static str, Val>,
    pub t: HashMap<&'static str, Tensor>,
}

// ggml type id -> (elements per block, bytes per block)
pub fn blk(ty: u32) -> (usize, usize) {
    match ty {
        0 => (1, 4),      // f32
        1 | 30 => (1, 2), // f16, bf16
        8 => (32, 34),    // q8_0
        12 => (256, 144), // q4_k
        13 => (256, 176), // q5_k
        14 => (256, 210), // q6_k
        16 => (256, 66),  // iq2_xxs
        17 => (256, 74),  // iq2_xs
        18 => (256, 98),  // iq3_xxs
        20 => (32, 18),   // iq4_nl
        22 => (256, 82),  // iq2_s
        40 => (64, 36),   // nvfp4
        _ => panic!("gguf: tensor type {ty}"),
    }
}

impl Tensor {
    pub fn row(&self) -> usize {
        let (e, b) = blk(self.ty);
        self.dims[0] / e * b
    }
}

struct Cur(&'static [u8], usize);

impl Cur {
    fn n(&mut self, n: usize) -> &'static [u8] {
        self.1 += n;
        &self.0[self.1 - n..self.1]
    }
    fn u(&mut self, n: usize) -> u64 {
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(self.n(n));
        u64::from_le_bytes(b)
    }
    fn s(&mut self) -> &'static [u8] {
        let n = self.u(8) as usize;
        self.n(n)
    }
    fn val(&mut self, ty: u32) -> Val {
        let sx = |v: u64, n: u32| ((v << (64 - n)) as i64) >> (64 - n);
        match ty {
            0 | 7 => Val::U(self.u(1)),
            2 => Val::U(self.u(2)),
            4 => Val::U(self.u(4)),
            10 => Val::U(self.u(8)),
            1 => Val::I(sx(self.u(1), 8)),
            3 => Val::I(sx(self.u(2), 16)),
            5 => Val::I(sx(self.u(4), 32)),
            11 => Val::I(self.u(8) as i64),
            6 => Val::F(f32::from_bits(self.u(4) as u32) as f64),
            12 => Val::F(f64::from_bits(self.u(8))),
            8 => Val::S(self.s()),
            9 => {
                let (ty, n) = (self.u(4) as u32, self.u(8) as usize);
                let a = self.1;
                for _ in 0..n {
                    self.val(ty);
                }
                Val::A(ty, n, &self.0[a..self.1])
            }
            _ => panic!("gguf: kv type {ty}"),
        }
    }
}

pub fn open(path: &str) -> Gguf {
    let map = unsafe {
        let c = std::ffi::CString::new(path).unwrap();
        let fd = libc::open(c.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
        assert!(fd >= 0, "gguf: cannot open {path}");
        let n = libc::lseek(fd, 0, libc::SEEK_END);
        assert!(n > 0, "gguf: {path} is empty");
        let n = n as usize;
        let p = libc::mmap(
            std::ptr::null_mut(),
            n,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        );
        libc::close(fd); // the mapping outlives the descriptor
        assert!(p != libc::MAP_FAILED, "gguf: mmap failed");
        std::slice::from_raw_parts(p as *const u8, n)
    };
    let st = |b| unsafe { std::str::from_utf8_unchecked(b) };
    let mut c = Cur(map, 0);
    assert!(c.n(4) == b"GGUF" && c.u(4) == 3, "gguf: bad header");
    let (nt, nkv) = (c.u(8), c.u(8));
    let mut kv = HashMap::new();
    for _ in 0..nkv {
        let k = st(c.s());
        let ty = c.u(4) as u32;
        kv.insert(k, c.val(ty));
    }
    // mainline llama.cpp names the looped model inkling2_loop and prefixes its
    // keys so (unsloth/Inkling-2-Mini-GGUF since 23 November 2026); the reads
    // below use inkling2.*, so each key is also kept under that name
    if let Some(Val::S(b"inkling2_loop")) = kv.get("general.architecture") {
        let al: Vec<&'static str> = kv.keys()
            .filter_map(|k| k.strip_prefix("inkling2_loop."))
            .filter(|k| !kv.contains_key(format!("inkling2.{k}").as_str()))
            .map(|k| &*Box::leak(format!("inkling2.{k}").into_boxed_str()))
            .collect();
        for a in al {
            let v = kv[format!("inkling2_loop.{}", &a[9..]).as_str()];
            kv.insert(a, v);
        }
    }
    let mut t: Vec<Tensor> = (0..nt)
        .map(|_| {
            let name = st(c.s());
            let nd = c.u(4);
            let dims: Vec<_> = (0..nd).map(|_| c.u(8) as usize).collect();
            let (ty, off) = (c.u(4) as u32, c.u(8) as usize);
            let (e, b) = blk(ty);
            let len = dims.iter().product::<usize>() / e * b;
            Tensor { name, dims, ty, off, len }
        })
        .collect();
    let al = match kv.get("general.alignment") {
        Some(Val::U(a)) => *a as usize,
        _ => 32,
    };
    let base = c.1.div_ceil(al) * al;
    t.iter_mut().for_each(|t| t.off += base);
    Gguf { map, kv, t: t.into_iter().map(|t| (t.name, t)).collect() }
}

impl Gguf {
    pub fn f(&self, k: &str) -> f64 {
        let Some(v) = self.kv.get(k) else { panic!("gguf: no key {k}") };
        match *v {
            Val::U(v) => v as f64,
            Val::I(v) => v as f64,
            Val::F(v) => v,
            _ => panic!("gguf: {k} is not a number"),
        }
    }
    pub fn u(&self, k: &str) -> usize {
        self.f(k) as usize
    }
    // a numeric array (or a scalar, as one item)
    pub fn nums(&self, k: &str) -> Vec<f64> {
        let Val::A(ty, n, b) = self.kv[k] else {
            return vec![self.f(k)];
        };
        let mut c = Cur(b, 0);
        (0..n)
            .map(|_| match c.val(ty) {
                Val::U(v) => v as f64,
                Val::I(v) => v as f64,
                Val::F(v) => v,
                _ => panic!("gguf: {k} is not numeric"),
            })
            .collect()
    }
    // a string array, lazily
    pub fn strs(&self, k: &str) -> impl Iterator<Item = &'static [u8]> {
        let Val::A(8, n, b) = self.kv[k] else {
            panic!("gguf: {k} is not a string array")
        };
        let mut c = Cur(b, 0);
        (0..n).map(move |_| c.s())
    }
    // tensor bytes
    pub fn data(&self, t: &Tensor) -> &'static [u8] {
        &self.map[t.off..t.off + t.len]
    }
}
