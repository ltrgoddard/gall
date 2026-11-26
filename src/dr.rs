// dspark block drafter on the target's hidden states (DRAFT.md, dr.metal).
//   let d = Dr::open(&m, path, n_ctx);
//   let next = d.step(&mut m, tok);     // target step + context inject
//   let ds = d.draft(&m, anchor, 7); // (token, probability) after anchor
//   let next = d.pass(&m, &[anchor, ids..]); d.commit(&mut m, rows, kept);
//   let row = d.next(&mut m, &mut spec, tok);  // all of it: feed tok;
//   m.logits_row(row) and id[16 row + 8] are what follows it
//   d.generate(&mut m, next, &mut Spec::new(), |tok| more);  // greedy loop
// gall dr MODEL DRAFTER REFDIR [-n 64] [-c 4096] [--ub] [--tv 2,4,8
//   [--at N,..]] [--cmp R [-o O]] [--drift 1,2,8 [-o O]]:
// teacher-forced acceptance; --tv: ms of a draft and of a pass by rows
// (at N tokens of context); --cmp: a pass against stepping; --drift:
// histories by passes against steps and the reference
// gall run MODEL --tokens FILE --draft DRAFTER [--block B] [-n 64] [-c 4096]
use crate::gguf::{self, Gguf, Tensor};
use crate::gpu::*;
use crate::model::{Model, Op, FT};
use objc2_metal::*;
use std::time::Instant;
use crate::mv::{name, Mv};
use std::collections::HashMap;

const N: usize = 2304;
const SRC: &str = include_str!("dr.metal");
const MM: &str = concat!(
    "#include <metal_stdlib>\n",
    include_str!("grids.metal"),
    include_str!("mv.metal"),
    include_str!("drm.metal")
);
const KINDS: [&str; 15] = [
    "tap", "fuse", "norm", "rope", "put", "attn", "silu", "add", "joint", "pl",
    "pset", "amax", "uni", "dsum", "amrg",
];
// drm.metal x_*: (type, weight rows per threadgroup at 1 and 2 uses)
const XT: [(u32, usize, usize); 10] = [(8, 4, 4), (12, 2, 4), (13, 2, 4),
    (14, 2, 4), (16, 8, 4), (17, 8, 8), (18, 8, 4), (20, 2, 4), (22, 8, 8),
    (40, 2, 4)];
// target activations, as in model.rs: the normed output and its source
const TA: [usize; 2] = [3 * N, 15 * N];
// drafter activations, in floats; block regions hold 8 rows
const CX: usize = 0; // 3 layer contexts, f_rnn
const KC: usize = 4 * N; // context k, v per layer
const X: usize = KC + 3 * 2048; // residual
const HN: usize = X + 8 * N; // normed
const QQ: usize = HN + 8 * N;
const KK: usize = QQ + 8 * 3072; // k then v
const AT: usize = KK + 8 * 2048; // attention out
const OO: usize = AT + 8 * 3072; // wo and down out
const GG: usize = OO + 8 * N;
const UU: usize = GG + 8 * 9216;
const ST: usize = UU + 8 * 9216; // head state
const JJ: usize = ST + 256;
const LG: usize = JJ + 768;
const WB: usize = LG + 201024; // output hidden_out bias
const BL: usize = WB + 201024; // base logits, 8 rows
const NV: usize = 201024;
const FR: usize = BL + 8 * NV; // 4 target features per verify row
const CF: usize = FR + 32 * N; // the drafter's probability of each draft
const SW: usize = CF + 8; // wb of the shortlist
const PA: usize = SW + SL; // attention partials: 192 queries x chunks x 130
// the drafter's head scores a shortlist: the SF tokens of greatest wb and
// a ring of the last SL - SF new tokens fed
const SL: usize = 32768;
const SF: usize = 28672;

pub struct Dr {
    g: &'static Gguf,
    w: Gpu, // the drafter file's buffers
    mv: Mv,
    pipes: HashMap<&'static str, Pipe>,
    s: Buf,
    pub b: Buf,
    c: Buf,
    id: Buf,
    p: Buf, // output hidden_out, 201024 x 256 f16
    // the shortlist: ids, then rows of output and of p; the slot + 1 of an
    // id, and the next ring slot
    sl: [Buf; 3],
    sv: std::cell::RefCell<(Vec<u32>, usize)>,
    // slab kernels by type and class: weight rows per threadgroup, name
    xp: HashMap<(u32, usize), (Pipe, usize, &'static str)>,
    ut: Buf, // the expert slabs of a pass by class; ui: their grids
    ui: Buf,
    t0: Buf, // the same for a dense matrix: slab 0, used by every row
    o: HashMap<&'static str, usize>, // f32 tensors in s
    co: [usize; 3],                  // cache rows before each layer
    n_ctx: usize,
    // positions with context: the last run of inject() in a row. what a
    // snapshot restored or the model stepped alone is not in it
    span: std::cell::Cell<(usize, usize)>,
}

impl Dr {
    pub fn open(m: &mut Model, path: &str, n_ctx: usize) -> Dr {
        m.taps();
        drop(m.sched_z(&[0; 2], 0, true, 0)); // compiles the row kernels now
        // one map per process, kept across unloads, as the model's is
        static G: std::sync::OnceLock<Gguf> = std::sync::OnceLock::new();
        let g: &'static Gguf = G.get_or_init(|| gguf::open(path));
        let mut w = Gpu::new();
        w.map(g);
        let (mut s, mut o) = (vec![0f32], HashMap::new());
        let tn = &m.g.t["token_embd_norm.weight"];
        for (g, t) in g.t.values().map(|t| (g, t)).chain([(m.g, tn)]) {
            if t.ty == 0 {
                o.insert(t.name, s.len());
                s.extend(g.data(t).chunks(4).map(|b| {
                    f32::from_le_bytes(b.try_into().unwrap())
                }));
            }
        }
        let sb = m.gpu.buf(4 * s.len());
        slice::<f32>(&sb)[..s.len()].copy_from_slice(&s);
        let pipes = KINDS.map(|k| (k, m.gpu.pipe(SRC, "", k))).into();
        let xp = XT.iter().flat_map(|&(t, r, r2)| {
            [(1, r), (2, r2), (3, 16)].map(|(k, r)| {
                let n = format!("x_{}{}", name(t), ["_1", "_2", ""][k - 1]);
                ((t, k), (m.gpu.pipe(MM, "", &n), r, &*n.leak()))
            })
        });
        // a dense matrix: slab 0 used by t rows, as uni would place it
        let t0 = m.gpu.buf(4 * 10 * 136);
        for t in 1..9 {
            let e = &mut slice::<u32>(&t0)[10 * [0, 65, 130 + t - 3][t.min(3) - 1]..];
            e[1] = t as u32;
            (0..t).for_each(|i| e[2 + i] = 8 * i as u32);
        }
        let d = Dr {
            xp: xp.collect(),
            ut: m.gpu.buf(4 * 10 * 195),
            ui: m.gpu.buf(4 * 3 * 12),
            t0,
            mv: Mv::new(&m.gpu, ""),
            pipes,
            s: sb,
            b: m.gpu.buf(4 * (PA + 192 * 130 * (n_ctx + 8).div_ceil(256))),
            p: m.gpu.buf(2 * 256 * NV),
            c: m.gpu.buf(2 * 2048 * (2048 + n_ctx)),
            id: m.gpu.buf(4 * 8),
            sl: [4, m.g.t["output.weight"].row(), 512].map(|n| m.gpu.buf(n * SL)),
            sv: Default::default(),
            co: [0, 1024, 1024 + n_ctx],
            n_ctx,
            span: Default::default(),
            g,
            w,
            o,
        };
        // P and wb: the head applied to hidden_out's columns and bias. 0.6 s
        // of head passes, so they are kept in a file by the two models
        let md = std::fs::metadata(path).unwrap();
        let at = md.modified().unwrap().duration_since(std::time::UNIX_EPOCH).unwrap();
        let f = format!("p-{}-{}-{}.dr", m.g.map.len(), md.len(), at.as_secs());
        let f = crate::up::cache().join(f);
        let wb = || &mut slice::<u8>(&d.b)[4 * WB..][..4 * NV];
        let read = |mut r: std::fs::File| {
            use std::io::Read;
            r.read_exact(wb()).and_then(|_| r.read_exact(slice::<u8>(&d.p))).is_ok()
        };
        if md.len() > 0 && std::fs::File::open(&f).is_ok_and(read) {
            return d.short(m);
        }
        let (b, out) = (slice::<f32>(&d.b), &m.g.t["output.weight"]);
        let ho = d.g.data(&d.g.t["dspark_hidden_out.weight"]);
        let hb = d.o["dspark_hidden_out.bias"];
        b[X..X + N].copy_from_slice(&slice::<f32>(&d.s)[hb..hb + N]);
        m.gpu.run(|e| d.mm(out, m.gpu.w(out), X, WB, NV, 1).2(e));
        for k in (0..256).step_by(8) {
            for (i, x) in b[HN..HN + 8 * N].iter_mut().enumerate() {
                let j = 2 * (i % N * 256 + k + i / N);
                *x = f32::from_bits((ho[j] as u32) << 16 | (ho[j + 1] as u32) << 24);
            }
            m.gpu.run(|e| {
                d.mm(out, m.gpu.w(out), HN, BL, NV, 8).2(e);
                d.k("pset", [6252, 1, 1], &[BL, k], vec![(&d.p, 0)]).2(e);
            });
        }
        let tmp = f.with_extension("tmp");
        let put = |mut w: std::fs::File| {
            use std::io::Write;
            w.write_all(wb()).and_then(|_| w.write_all(slice::<u8>(&d.p))).is_ok()
        };
        if std::fs::File::create(&tmp).is_ok_and(put) {
            std::fs::rename(tmp, f).ok();
        }
        d.short(m)
    }

    // the shortlist's first SL tokens: those of greatest wb
    fn short(self, m: &Model) -> Dr {
        let nv = m.g.u("inkling2.vocab_size_unpadded");
        let wb = &slice::<f32>(&self.b)[WB..][..nv];
        let mut o: Vec<u32> = (0..wb.len() as u32).collect();
        o.sort_by(|&a, &b| wb[b as usize].total_cmp(&wb[a as usize]));
        *self.sv.borrow_mut() = (vec![0; wb.len()], 0);
        o[..SL].iter().enumerate().for_each(|(k, &t)| self.put(m, k, t));
        self.sv.borrow_mut().1 = SL - SF;
        self
    }

    // token t into slot k of the shortlist
    fn put(&self, m: &Model, k: usize, t: u32) {
        let (sv, out) = (&mut self.sv.borrow_mut().0, &m.g.t["output.weight"]);
        let o = &mut slice::<u32>(&self.sl[0])[k];
        if sv[*o as usize] == k as u32 + 1 {
            sv[*o as usize] = 0;
        }
        (*o, sv[t as usize]) = (t, k as u32 + 1);
        let (n, t, p) = (out.row(), t as usize, slice::<u16>(&self.p));
        let o = &m.g.data(out)[t * n..][..n];
        slice::<u8>(&self.sl[1])[k * n..][..n].copy_from_slice(o);
        slice::<u16>(&self.sl[2])[k * 256..][..256].copy_from_slice(&p[t * 256..][..256]);
        slice::<f32>(&self.b)[SW + k] = slice::<f32>(&self.b)[WB + t];
    }

    // toks, fed: the new ones take the ring's oldest slots (an image row
    // is no token of the vocabulary)
    fn see(&self, m: &Model, toks: &[u32]) {
        for &t in toks {
            let k = self.sv.borrow().1;
            if self.sv.borrow().0.get(t as usize) == Some(&0) {
                self.put(m, SF + k % (SL - SF), t);
                self.sv.borrow_mut().1 = k + 1;
            }
        }
    }

    // a slab kernel of type ty for t uses of slab 0 (t0) or, with ind, for
    // the slabs of uni's class t (1, 2, 3 or more uses)
    fn x(&self, e: &Enc, ty: u32, t: usize, p: &[u32; 6], b: [B; 3], ind: Option<&Buf>) {
        let (pipe, r, _) = &self.xp[&(ty, t.min(3))];
        let Some(i) = ind else {
            let b = [b[0], b[1], b[2], (&self.t0, 40 * t.max(3) - 120)];
            return call(e, pipe, p, &b, [p[1] as usize / r, 1, 1], 32);
        };
        let o = 12 * (4 * t + r.trailing_zeros() as usize - 5);
        calli(e, pipe, p, &[b[0], b[1], b[2], (&self.ut, 0)], (i, o), 32)
    }

    // rows of y = W x: nt inputs n_in apart, outputs ys apart
    fn mm<'a>(&'a self, t: &'static Tensor, w: B<'a>, x: usize, y: usize, ys: usize, nt: usize) -> Op<'a> {
        self.mo(t, w, t.dims[1], x, y, ys, nt)
    }

    // the first no rows of t at w
    #[allow(clippy::too_many_arguments)]
    fn mo<'a>(&'a self, t: &'static Tensor, w: B<'a>, no: usize, x: usize,
              y: usize, ys: usize, nt: usize) -> Op<'a> {
        let p = [t.dims[0], no, t.row(), t.dims[0], ys, 0].map(|v| v as u32);
        (self.xp[&(t.ty, nt.min(3))].2, 0, Box::new(move |e: &Enc| {
            self.x(e, t.ty, nt, &p, [w, (&self.b, 4 * x), (&self.b, 4 * y)], None)
        }), None)
    }

    fn t(&self, l: usize, n: &str) -> &'static Tensor {
        &self.g.t[format!("blk.{l}.{n}.weight").as_str()]
    }

    fn o(&self, l: usize, n: &str) -> usize {
        self.o[self.t(l, n).name]
    }

    // a dr.metal kernel; x: buffers after b, s, c, id
    fn k<'a>(&'a self, n: &'static str, g: [usize; 3], p: &[usize], x: Vec<B<'a>>) -> Op<'a> {
        let mut a = [0u32; 8];
        p.iter().enumerate().for_each(|(i, &v)| a[i] = v as u32);
        (n, 0, Box::new(move |e: &Enc| {
            let mut b = [&self.b, &self.s, &self.c, &self.id].map(|b| (b, 0)).to_vec();
            b.extend(&x);
            let th = if g[0] == 1 { 768 } else if n == "attn" { 512 } else { 32 };
            call(e, &self.pipes[n], &a, &b, g, th)
        }), None)
    }

    // rows of y = W x; input rows are n_in apart, output rows ys (0: n_out)
    fn m<'a>(&'a self, l: usize, n: &str, x: usize, y: usize, ys: usize, rows: usize) -> impl Iterator<Item = Op<'a>> {
        let t = self.t(l, n);
        let (ni, no) = (t.dims[0], t.dims[1]);
        let one = (rows == 1).then(|| -> Op {
            (name(t.ty), 0, Box::new(move |e: &Enc| {
                self.mv.dense(e, t.ty, self.w.w(t), (&self.b, 4 * x), (&self.b, 4 * y), ni, no)
            }), None)
        });
        one.into_iter().chain((rows > 1).then(|| self.mm(t, self.w.w(t), x, y, ys.max(no), rows)))
    }

    // context k, v and f_rnn of the token at pos, from row r's features
    fn inject(&self, pos: usize, r: usize) -> Vec<Op<'_>> {
        let (lo, hi) = self.span.get();
        self.span.set((if pos < lo || pos > hi { pos } else { lo }, pos + 1));
        let nw = [0, 1, 2].map(|l| self.o(l, "dflare_norm"));
        let p = [FR + 4 * r * N, CX, self.o["dspark_fuse.weight"], nw[0], nw[1], nw[2],
            self.o["dspark_norm.weight"]];
        let mut v = vec![self.k("fuse", [72, 4, 1], &p, vec![])];
        for l in 0..3 {
            let (kc, r) = (KC + l * 2048, if l == 1 { pos } else { pos % 1024 });
            v.extend(self.m(l, "attn_k_ctx", CX + l * N, kc, 0, 1));
            v.extend(self.m(l, "attn_v_ctx", CX + l * N, kc + 1024, 0, 1));
            let p = [kc, 2048, self.o(l, "attn_k_ctx_norm"), pos];
            v.push(self.k("rope", [8, 1, 1], &p, vec![]));
            v.push(self.k("put", [64, 1, 1], &[kc, (self.co[l] + r) * 2048], vec![]));
        }
        v
    }

    // the target on toks at m.pos.. (1 to 8 rows): a weight is read once
    // for all rows, an expert slab once for the rows that use it, a small
    // op is one dispatch for all rows. taps each row's features. the
    // greedy id after each row; m.pos stays
    pub fn pass(&self, m: &Model, toks: &[u32]) -> Vec<u32> {
        let (t, rs) = (toks.len(), m.rs());
        // row 0 is kept whatever the others do: its context goes along
        let (ops, inj) = (m.sched_z(toks, m.pos, true, 0), self.inject(m.pos, 0));
        let mut q: Vec<(&str, Box<dyn Fn(&Enc) + '_>)> = vec![];
        let tap = |r: usize, i: usize, src: usize, vr: usize| -> (&str, Box<dyn Fn(&Enc) + '_>) {
            let p = [src, FR + (4 * r + i) * N, vr].map(|v| v as u32);
            let b = [(&m.a, 4 * r * rs), (&self.b, 0)];
            ("tap", Box::new(move |e: &Enc| call(e, &self.pipes["tap"], &p, &b, [72, 1, 1], 32)))
        };
        // for PROF: q index, bytes of the matrix or of a slab, use class
        let (mut gu, mut fl) = (0, vec![]);
        for o in &ops {
            let Some((mode, w, x, y)) = o.3.filter(|_| t > 1) else {
                q.push((o.0, Box::new(move |e: &Enc| o.2(e))));
                continue;
            };
            let (ni, no) = (w.dims[0], w.dims[1]);
            if mode == "up" {
                gu = y;
                q.push(("uni", Box::new(move |e: &Enc| {
                    let b = [(&m.id, 0), (&self.ut, 0), (&self.ui, 0)];
                    call(e, &self.pipes["uni"], &[t as u32], &b, [1, 1, 1], 64)
                })));
            }
            // experts go to the 8 gate_up rows, down's are then summed
            let (kx, yy) = if mode == "down" { (ni, gu) } else { (0, y) };
            let p = [ni, no, w.row(), rs, rs, kx].map(|v| v as u32);
            let b = [m.gpu.w(w), (&m.a, 4 * x), (&m.a, 4 * yy)];
            if mode == "mv" {
                let n = self.xp[&(w.ty, t.min(3))].2;
                fl.push((q.len(), w.len, 0));
                q.push((n, Box::new(move |e: &Enc| self.x(e, w.ty, t, &p, b, None))));
            }
            for k in (1..4).filter(|_| mode != "mv") {
                let n = self.xp[&(w.ty, k)].2;
                fl.push((q.len(), w.len / 128, k));
                q.push((n, Box::new(move |e: &Enc| self.x(e, w.ty, k, &p, b, Some(&self.ui)))));
            }
            if mode == "down" {
                let p = [gu, y, rs].map(|v| v as u32);
                q.push(("dsum", Box::new(move |e: &Enc| {
                    call(e, &self.pipes["dsum"], &p, &[(&m.a, 0)], [72, t, 1], 32)
                })));
            }
        }
        for r in 0..t {
            (0..3).for_each(|i| q.push(tap(r, i, FT + i * N, 0)));
            q.push(tap(r, 3, TA[0], TA[1]));
        }
        inj.iter().for_each(|o| q.push((o.0, Box::new(move |e: &Enc| o.2(e)))));
        if std::env::var("PROF").is_ok_and(|_| t > 1) {
            // every dispatch in its own command buffer: read as shares. the
            // floor: its bytes at 87 GB/s, slabs as uni counted them
            let mut by = std::collections::BTreeMap::<_, [f64; 3]>::new();
            let (mut u, mut f) = ([1; 4], fl.iter().peekable());
            for (i, o) in q.iter().enumerate() {
                let t = m.gpu.run(|e| o.1(e));
                if o.0 == "uni" {
                    let s = slice::<u32>(&self.ut);
                    for k in 1..4 {
                        let n = |j| s[(65 * k - 65 + j) * 10 + 1] > 0;
                        u[k] = (0..65).take_while(|&j| n(j)).count();
                    }
                }
                let b = f.next_if(|x| x.0 == i).map_or(0, |x| x.1 * u[x.2]);
                let e = by.entry(o.0).or_default();
                *e = [e[0] + 1., e[1] + 1e3 * t, e[2] + b as f64 / 87e6];
            }
            for (k, [c, t, b]) in by {
                eprintln!("{k:12} {c:5} {t:7.2} ms {b:7.2} floor");
            }
        }
        // several command buffers: the gpu starts while the rest encodes
        objc2::rc::autoreleasepool(|_| {
            let cb = q.chunks(128).map(|c| {
                let cb = m.gpu.q.commandBuffer().unwrap();
                let e = cb.computeCommandEncoder().unwrap();
                c.iter().for_each(|o| o.1(&e));
                e.endEncoding();
                cb.commit();
                cb
            });
            let l = cb.last().unwrap();
            l.waitUntilCompleted();
            assert!(l.error().is_none(), "metal: {:?}", l.error());
        });
        (0..t).map(|r| slice::<u32>(&m.id)[16 * r + 8]).collect()
    }

    // keep rows 0..=k of a pass of t rows: position, rollback of the rest,
    // drafter context of the rows after the first
    pub fn commit(&self, m: &mut Model, t: usize, k: usize) {
        let pos = m.pos;
        m.pos += t;
        assert!(m.truncate(pos + k + 1));
        m.top = pos + t;
        let ops: Vec<Op> = (1..=k).flat_map(|r| self.inject(pos + r, r)).collect();
        if k > 0 {
            m.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
        }
    }

    // a prompt: batched prefill with the context of each token, the tail
    // by step(). greedy next id
    pub fn feed(&self, m: &mut Model, toks: &[u32]) -> u32 {
        self.see(m, toks);
        let rest = m.prefill_tap(toks, 512, Some(&mut |m: &Model, ft: &[f32]| {
            for (i, f) in ft.chunks(32 * N).enumerate() {
                slice::<f32>(&self.b)[FR..][..f.len()].copy_from_slice(f);
                let n = (0..f.len() / (4 * N)).flat_map(|r| self.inject(m.pos + 8 * i + r, r));
                let ops: Vec<Op> = n.collect();
                m.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
            }
        }));
        // the tail may hold image rows: the general path
        m.stage(rest);
        rest.iter().fold(0, |_, &t| {
            let r = self.pass(m, &[t])[0];
            self.commit(m, 1, 0);
            r
        })
    }

    // one target token with its context; greedy next id
    pub fn step(&self, m: &mut Model, tok: u32) -> u32 {
        self.see(m, &[tok]);
        let r = m.step(tok);
        // the next token's command buffer comes after this one
        let ops = self.inject(m.pos - 1, 0);
        m.gpu.go(|e| {
            let p = |i: usize| if i < 3 { [FT + i * N, 0] } else { TA };
            for i in 0..4 {
                let p = [p(i)[0], FR + i * N, p(i)[1]].map(|v| v as u32);
                call(e, &self.pipes["tap"], &p, &[(&m.a, 0), (&self.b, 0)], [72, 1, 1], 32);
            }
            ops.iter().for_each(|o| o.2(e))
        });
        r
    }

    // feed tok, the token after those the model holds; the activation row
    // whose logits (`Model::logits_row`) and greedy id follow it. tok is
    // any token, sampled or not: a row verified ahead is used only while
    // the tokens fed are its drafts, so the logits are those of a serial
    // step. with drafting on, a new pass is over tok and the drafts that
    // st.plan() takes, from a new draft or from what is left of the last
    // one if that held; else a serial step, and a draft now and then to
    // keep score. m.pos counts the tokens fed; rows ahead lie past it
    pub fn next(&self, m: &mut Model, st: &mut Spec, tok: u32) -> usize {
        self.see(m, &[tok]);
        if st.pos != m.pos {
            st.drop();
        }
        let hit = st.d.get(st.r - 1).copied().is_some_and(|d| st.see(d.1, d.0 == tok));
        st.s = (st.s + 1) * hit as usize;
        if hit && st.r <= st.j {
            let ops = self.inject(m.pos, st.r);
            m.gpu.go(|e| ops.iter().for_each(|o| o.2(e)));
            (m.pos, st.r, st.n, st.rows[1]) = (m.pos + 1, st.r + 1, st.n + 1, st.rows[1] + 1);
            st.pos = m.pos;
            return st.r - 1;
        }
        // the draft past the verified rows was scored by tok as well: if
        // it holds, so does the rest
        st.kept[st.r - 1] += (st.j > 0) as usize;
        let mut d = if hit { st.d.split_off(st.r) } else { vec![] };
        let on = st.block.map_or(st.on(), |b| b > 0);
        let nd = st.block.unwrap_or(if on || st.n >= st.due[0] { st.nb() } else { 0 });
        // a longer draft from elsewhere (a copy, serve.rs) costs nothing
        let cp = st.cp.len() > d.len();
        if cp {
            d = std::mem::take(&mut st.cp);
        } else if d.is_empty() && nd > 0 {
            d = self.draft(m, tok, nd);
            st.fresh(&d, on);
        }
        let room = self.n_ctx - 1 - m.pos;
        let j = (st.plan(&d, st.s).0 * (on || cp) as usize).min(room);
        if j == 0 {
            self.step(m, tok);
        } else {
            let toks: Vec<u32> = [tok].into_iter().chain(d[..j].iter().map(|x| x.0)).collect();
            self.pass(m, &toks);
            (m.pos, m.top) = (m.pos + 1, m.pos + 1 + j);
        }
        (st.d, st.j, st.r, st.pos) = (d, j, 1, m.pos);
        (st.n, st.rows[0]) = (st.n + 1, st.rows[0] + j);
        0
    }

    // greedy decode from `next` until emit returns false
    pub fn generate(&self, m: &mut Model, mut next: u32, st: &mut Spec, mut emit: impl FnMut(u32) -> bool) {
        while emit(next) {
            let r = self.next(m, st, next);
            next = slice::<u32>(&m.id)[16 * r + 8];
        }
    }

    // the model moved on without the drafter (a snapshot, a prefix fed by
    // other code): no context before pos
    pub fn forget(&self, pos: usize) {
        self.span.set((pos, pos));
    }

    // nb tokens after anchor a at position m.pos, each with the drafter's
    // probability of it
    pub fn draft(&self, m: &Model, a: u32, nb: usize) -> Vec<(u32, f32)> {
        let (n, (lo, hi)) = (m.pos, self.span.get());
        let lo = if hi < n { n } else { lo.min(n) };
        slice::<f32>(&self.b)[ST..ST + 256].fill(0.);
        slice::<u32>(&self.id)[0] = a;
        let (emb, out) = (&m.g.t["token_embd.weight"], &m.g.t["output.weight"]);
        let cap = (m.g.f("inkling2.final_logit_softcapping") as f32).to_bits() as usize;
        let mut v: Vec<Op> = vec![("row", 0, Box::new(move |e: &Enc| {
            self.mv.row(e, emb.ty, m.gpu.w(emb), a as usize, (&self.b, 4 * OO), N)
        }), None)];
        let g8 = [72, 8, 1];
        let p = [OO, X, self.o["token_embd_norm.weight"], self.o["mask_embd.weight"]];
        v.push(self.k("norm", g8, &p, vec![]));
        for l in 0..3 {
            v.push(self.k("norm", g8, &[X, HN, self.o(l, "attn_norm")], vec![]));
            v.extend(self.m(l, "attn_q", HN, QQ, 0, 8));
            v.extend(self.m(l, "attn_k", HN, KK, 2048, 8));
            v.extend(self.m(l, "attn_v", HN, KK + 1024, 2048, 8));
            v.push(self.k("rope", [24, 8, 1], &[QQ, 3072, self.o(l, "attn_q_norm"), n], vec![]));
            v.push(self.k("rope", [8, 8, 1], &[KK, 2048, self.o(l, "attn_k_norm"), n], vec![]));
            let (nk, ring) = if l == 1 { (n - lo, 0) } else { ((n - lo).min(1024), 1024) };
            let nc = (nk + 8).div_ceil(256);
            let p = [QQ, KK, self.co[l] * 2048, nk, n - nk, ring, PA, nc];
            v.push(self.k("attn", [8, 8, nc], &p, vec![]));
            v.push(self.k("amrg", [24, 8, 1], &[PA, nc, AT], vec![]));
            v.extend(self.m(l, "attn_output", AT, OO, 0, 8));
            v.push(self.k("add", g8, &[X, OO], vec![]));
            v.push(self.k("norm", g8, &[X, HN, self.o(l, "ffn_norm")], vec![]));
            v.extend(self.m(l, "ffn_gate", HN, GG, 0, 8));
            v.extend(self.m(l, "ffn_up", HN, UU, 0, 8));
            v.push(self.k("silu", [288, 8, 1], &[GG, UU], vec![]));
            v.extend(self.m(l, "ffn_down", GG, OO, 0, 8));
            v.push(self.k("add", g8, &[X, OO], vec![]));
        }
        v.push(self.k("norm", g8, &[X, HN, self.o["output_norm.weight"]], vec![]));
        v.push(self.mo(out, (&self.sl[1], 0), SL, HN + N, BL, NV, nb));
        let w = |n: &str| self.w.w(&self.g.t[n]);
        for i in 1..=nb {
            let p = [ST, HN + i * N, CX + 3 * N, JJ, self.o["joint_proj.bias"], i - 1];
            let x = vec![w("joint_proj.weight"), w("markov_w1.weight")];
            v.push(self.k("joint", [768, 1, 1], &p, x));
            let p = [ST, JJ, BL + (i - 1) * NV, LG, SW];
            v.push(self.k("pl", [SL / 32, 1, 1], &p, vec![(&self.sl[2], 0)]));
            let p = [LG, SL, i, CF + i, cap];
            v.push(self.k("amax", [1, 1, 1], &p, vec![(&self.sl[0], 0)]));
        }
        if std::env::var("PROF").is_ok() {
            // every dispatch in its own command buffer: read as shares
            let mut by = std::collections::BTreeMap::<_, (usize, f64)>::new();
            for o in &v {
                let e = by.entry(o.0).or_default();
                *e = (e.0 + 1, e.1 + m.gpu.run(|e| o.2(e)));
            }
            by.iter().for_each(|(k, (c, t))| eprintln!("{k:8} {c:4} {:7.2} ms", 1e3 * t));
        }
        m.gpu.run(|e| v.iter().for_each(|o| o.2(e)));
        let (id, p) = (slice::<u32>(&self.id), slice::<f32>(&self.b));
        (1..=nb).map(|i| (id[i], p[CF + i])).collect()
    }
}

// running estimates for plan(). the costs are fixed, seconds of a step, a
// pass and a draft in the loop of next() as measured here (m2 air), not
// timed: a pass's rows follow from the tokens alone, so
// the output does not change from run to run
pub struct Spec {
    pub block: Option<usize>, // a fixed count of drafts per round; 0: none
    pub q: [[f64; 12]; 2],    // acceptance of a draft by its probability,
                              // after < 3 or >= 3 tokens foretold in a row
    pub e: [f64; 2],          // tokens and seconds to expect of a draft
    pub tv: [f64; 9],         // a pass by rows
    pub td: [f64; 8],         // a draft by length
    pub kept: [usize; 8],     // passes by drafts kept
    pub rows: [usize; 2],     // drafts verified, and kept
    pub drafts: usize,
    pub n: usize,
    pub cp: Vec<(u32, f32)>, // a draft for the next round, from elsewhere
    d: Vec<(u32, f32)>, // the draft in hand, after the last pass's token
    j: usize,           // its first j are verified: rows 1..=j of the pass
    r: usize,           // d[r - 1] is the draft of the token to come
    pos: usize,         // the model's position then
    s: usize,           // tokens in a row the drafts in hand foretold
    due: [usize; 2],       // off: the next draft is at n = due[0]; the gap
    hits: [f64; 2],        // of this draft, and their mean
}

impl Spec {
    pub fn new() -> Spec {
        let ms = |v: [f64; 8]| v.map(|x| 1e-3 * x);
        let tv = [40.2, 63., 85., 97., 107., 117., 126., 136.];
        Spec {
            block: None,
            // as measured over 13 workloads of 400 tokens
            q: [[5., 11., 13., 26., 29., 41., 47., 55., 62., 68., 72., 91.],
                [5., 19., 22., 29., 35., 45., 54., 65., 69., 76., 80., 96.]]
                .map(|q| q.map(|v| v / 100.)),
            e: [1., 0.055],
            tv: std::array::from_fn(|i| 1e-3 * tv[i.max(1) - 1]),
            td: ms([0., 6.3, 6.8, 7.4, 7.7, 8.1, 8.4, 8.8]),
            kept: [0; 8],
            rows: [0; 2],
            drafts: 0,
            n: 0,
            cp: vec![],
            d: vec![],
            j: 0,
            r: 1,
            pos: 0,
            s: 0,
            due: [0; 2],
            hits: [0., 2.],
        }
    }

    // a probability's place in q: half steps of its logit, 0.12 to 0.97
    fn bin(p: f32) -> usize {
        (4. + 2. * (p / (1. - p)).ln()).clamp(0., 11.) as usize
    }

    // forget the draft in hand: the model was moved by other hands
    pub fn drop(&mut self) {
        (self.j, self.r) = (0, 1);
        self.d.clear();
    }

    // the chance that a draft of probability p holds, s held before it
    fn q(&self, s: usize, p: f32) -> f64 {
        self.q[(s >= 3) as usize][Spec::bin(p)]
    }

    fn see(&mut self, p: f32, hit: bool) -> bool {
        let q = &mut self.q[(self.s >= 3) as usize][Spec::bin(p)];
        *q += 0.05 * (hit as u8 as f64 - *q);
        self.hits[0] += hit as u8 as f64;
        hit
    }

    // drafting beats serial decode
    pub fn on(&self) -> bool {
        self.e[0] * self.tv[1] > self.e[1]
    }

    // drafts to ask for: 2 more than held of late
    fn nb(&self) -> usize {
        (self.hits[1] + 2.5).clamp(3., 7.) as usize
    }

    // the drafts to verify, and the tokens to expect of the round: a draft
    // is taken while its chance to be kept pays for its row at the rate
    // in hand
    fn plan(&self, d: &[(u32, f32)], s: usize) -> (usize, f64) {
        let (mut p, mut e) = (1., 1.);
        let rate = (self.e[0] / self.e[1]).max(1. / self.tv[1]);
        for (j, x) in d.iter().take(7).enumerate() {
            p *= self.q(s + j, x.1);
            if self.block.is_none() && p < rate * (self.tv[j + 2] - self.tv[j + 1]) {
                return (j, e);
            }
            e += p;
        }
        (d.len().min(7), e)
    }

    // a new draft: the tokens and seconds to expect of it over its
    // rounds, each reached if the drafts before it hold, go into e. off,
    // the next draft is 8 tokens on if this one would have paid, else 16,
    // then 32
    fn fresh(&mut self, d: &[(u32, f32)], on: bool) {
        let (mut i, mut s, mut x) = (0, 1., [0., self.td[d.len()]]);
        while i == 0 || i < d.len() {
            let (j, e) = self.plan(&d[i..], self.s + i);
            x = [x[0] + s * e, x[1] + s * self.tv[j + 1]];
            let held = i..d.len().min(i + j + 1);
            s *= held.map(|k| self.q(self.s + k, d[k].1)).product::<f64>();
            i += j + 1;
        }
        let w = if on { 0.1 } else { 0.3 };
        (0..2).for_each(|i| self.e[i] += w * (x[i] - self.e[i]));
        self.hits = [0., self.hits[1] + 0.2 * (self.hits[0] - self.hits[1])];
        let off = (2 * self.due[1]).clamp(16, 32);
        let gap = if x[0] * self.tv[1] > x[1] { 8 } else { off };
        self.due = [self.n + gap, gap];
        self.drafts += 1;
    }
}

pub fn main(a: &[String]) {
    let opt = |k: &str, d: usize| {
        a.iter().position(|x| x == k).map_or(d, |i| a[i + 1].parse().unwrap())
    };
    let (n, d) = (opt("-n", 64), &a[2]);
    let am = |l: &[f32]| (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b });
    let mut m = Model::open(&a[0], opt("-c", 4096));
    m.reset();
    let dr = Dr::open(&mut m, &a[1], opt("-c", 4096));
    let mut all = crate::ids(&format!("{d}/tokens.txt"));
    let np = all.len();
    all.extend(crate::ids(&format!("{d}/gen.txt")));
    // run[k]: drafts whose first k tokens are the target's; hit: per slot
    let (mut run, mut hit, mut td) = ([0usize; 8], [0usize; 7], 0.);
    let mut cal = [[0usize; 2]; 10]; // by probability: drafts, hits
    if let Some(i) = a.iter().position(|x| x == "--tv") {
        // ms of a draft of 7 and of a pass by rows, medians of 7, at each
        // context length of --at N,.. (the prompt's first N tokens; all of
        // it by default), with the same rows of gen.txt at each
        let at = a.iter().position(|x| x == "--at").map_or(vec![np], |j| {
            a[j + 1].split(',').map(|v| v.parse().unwrap()).collect()
        });
        let md = |f: &mut dyn FnMut()| {
            let mut v: Vec<f64> = (0..7).map(|_| {
                let t0 = Instant::now();
                f();
                1e3 * t0.elapsed().as_secs_f64()
            }).collect();
            v.sort_by(f64::total_cmp);
            v[3]
        };
        let mut k = 0;
        for n in at {
            dr.feed(&mut m, &all[k..n.clamp(k + 1, np)]);
            k = n.clamp(k + 1, np);
            let mut i0 = np;
            let d = md(&mut || drop(dr.draft(&m, all[np], 7)));
            print!("at {}: d(7) {d:.1}", m.pos);
            for r in a[i + 1].split(',').map(|v| v.parse::<usize>().unwrap()) {
                print!(" t({r}) {:.1}", md(&mut || {
                    dr.pass(&m, &all[i0..i0 + r]);
                    dr.commit(&mut m, r, r - 1);
                    i0 += r;
                }));
            }
            println!();
        }
        return;
    }
    if let Some(i) = a.iter().position(|x| x == "--cmp") {
        // a pass of r rows on the reference text against stepping them:
        // the greatest logit difference, the argmax of each row and the
        // margin of the stepped one. -o O: O tokens into gen.txt, by
        // passes of r rows
        let (r, p) = (a[i + 1].parse::<usize>().unwrap(), np - 1 + opt("-o", 0));
        dr.feed(&mut m, &all[..np - 1]);
        while m.pos < p {
            let k = r.min(p - m.pos);
            dr.pass(&m, &all[m.pos..m.pos + k]);
            dr.commit(&mut m, k, k - 1);
        }
        let gap = |l: &[f32]| {
            let b = am(l);
            l[b] - (0..l.len()).filter(|&i| i != b).fold(f32::MIN, |a, i| a.max(l[i]))
        };
        dr.pass(&m, &all[p..p + r]);
        let v: Vec<_> = (0..r).map(|j| m.logits_row(j)).collect();
        (m.top, m.pos) = (p + r, p + r);
        assert!(m.truncate(p));
        for (j, v) in v.iter().enumerate() {
            m.step(all[p + j]);
            let s = m.logits();
            let d = s.iter().zip(v).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            println!("row {j}: max diff {d:.2e}, argmax {} {}, margin {:.2e}", am(&s), am(v),
                gap(&s));
        }
        return;
    }
    if let Some(i) = a.iter().position(|x| x == "--drift") {
        // -o O tokens of gen.txt teacher-forced by steps (r 1) or by
        // passes of r rows, r in R. by 100 rows: the median and greatest
        // difference from the steps over their 32 best logits, kl from the
        // reference logits.f32 where it has rows (mean, max), and argmax
        // other than gen.txt
        let (o, nv) = (opt("-o", 256), 200058);
        let lg = crate::f32s(&format!("{d}/logits.f32"));
        let mut run = |r: usize| -> Vec<Vec<f32>> {
            m.reset();
            dr.forget(0);
            dr.feed(&mut m, &all[..np - 1]);
            let mut v = vec![];
            while v.len() < o {
                let (q, k) = (v.len(), r.min(o - v.len()));
                if r == 1 {
                    m.step(all[np - 1 + q]);
                    v.push(m.logits());
                    continue;
                }
                dr.pass(&m, &all[np - 1 + q..][..k]);
                v.extend((0..k).map(|j| m.logits_row(j)));
                dr.commit(&mut m, k, k - 1);
            }
            v
        };
        let s = run(1);
        // the greatest difference over the 32 best of s; median, greatest
        let td = |s: &[f32], v: &[f32]| {
            let mut ix: Vec<usize> = (0..s.len()).collect();
            ix.select_nth_unstable_by(32, |a, b| s[*b].total_cmp(&s[*a]));
            ix[..32].iter().fold(0f32, |a, &i| a.max((s[i] - v[i]).abs()))
        };
        let mm = |mut t: Vec<f32>| {
            t.sort_by(f32::total_cmp);
            format!("{:.1e}/{:.1e}", t[t.len() / 2], t[t.len() - 1])
        };
        for r in a[i + 1].split(',').map(|v| v.parse::<usize>().unwrap()) {
            let v = if r == 1 { s.clone() } else { run(r) };
            let w: Vec<String> = v.chunks(100).zip(s.chunks(100)).enumerate().map(|(c, (v, s))| {
                let t = v.iter().zip(s).map(|(v, s)| td(s, v)).collect();
                let q = |j: usize| 100 * c + j;
                let j: Vec<usize> = (0..v.len()).filter(|&j| (q(j) + 1) * nv <= lg.len()).collect();
                let rf = |j: usize| &lg[q(j) * nv..][..nv];
                let k: Vec<f64> = j.iter().map(|&j| crate::kl(rf(j), &v[j])).collect();
                let km = k.iter().fold(0f64, |a, &b| a.max(b)) as f32;
                let x = (0..v.len()).filter(|&j| am(&v[j]) as u32 != all[np + q(j)]).count();
                let rd = if j.is_empty() { "-".into() } else {
                    mm(j.iter().map(|&j| td(rf(j), &v[j])).collect())
                };
                format!("{} ref {rd} kl {:.1e}/{:.1e} ({}) x {x}", mm(t),
                    k.iter().sum::<f64>() / k.len().max(1) as f64, km, k.len())
            }).collect();
            // fnv-1a of the logit bits, to see a kernel change keep them
            let h = v.iter().flatten().fold(0xcbf29ce484222325u64, |h, x| {
                (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3)
            });
            println!("r {r}: {w:?} bits {h:016x}");
        }
        return;
    }
    // --ub: the prompt by feed() (batched), not step by step
    let ub = a.iter().any(|x| x == "--ub");
    if ub {
        dr.feed(&mut m, &all[..np - 1]);
    }
    for i in if ub { np - 1 } else { 0 }..np + n {
        dr.step(&mut m, all[i]);
        if i == np {
            // the step that fed gen[0] is ref step s0000; step() does not
            // wait for its taps
            m.gpu.run(|_| ());
            for (j, f) in ["l_out-4", "l_out-28", "l_out-46", "result_norm"].iter().enumerate() {
                let r = crate::f32s(&format!("{d}/s0000/{f}.f32"));
                let x = &slice::<f32>(&dr.b)[FR + j * N..][..N];
                println!("feature {j} ({f}): rel {:.2e}", crate::rel(x, &r));
            }
        }
        if i + 1 < np {
            continue;
        }
        let t0 = Instant::now();
        let ds = dr.draft(&m, all[i + 1], 7);
        td += t0.elapsed().as_secs_f64();
        let ok: Vec<bool> = (0..7).map(|k| ds[k].0 == all[i + 2 + k]).collect();
        let r = ok.iter().take_while(|&&x| x).count();
        // a draft counts while those before it hold
        for k in 0..(r + 1).min(7) {
            let c = &mut cal[(ds[k].1 * 10.).min(9.) as usize];
            *c = [c[0] + 1, c[1] + ok[k] as usize];
        }
        run[r] += 1;
        (0..7).for_each(|k| hit[k] += ok[k] as usize);
    }
    let c = (n + 1) as f64;
    println!("draft of 7 {:.1} ms", 1e3 * td / c);
    println!("slot hit {:.2?}", hit.map(|h| h as f64 / c));
    println!("drafts, hits by probability in tenths: {cal:?}");
    for b in 1..8 {
        let e: usize = run.iter().enumerate().map(|(k, r)| r * (1 + k.min(b))).sum();
        println!("block {b}: {:.2} tokens per pass", e as f64 / c);
    }
}

// whether the drafter beside the model has the dspark layout dr reads. the
// mainline llama.cpp drafter (architecture dflash, unsloth/Inkling-2-Mini-GGUF
// since 23 November 2026) does not; its dspark-Inkling-2-Mini-Q8_0_Old.gguf does
pub fn usable(model: &str) -> bool {
    let d = std::path::Path::new(model).with_file_name(crate::up::DRAFT);
    if !d.exists() {
        return true;
    }
    let g = gguf::open(d.to_str().unwrap());
    let ok = g.t.contains_key("dspark_hidden_out.weight");
    unsafe { libc::munmap(g.map.as_ptr() as *mut _, g.map.len()) };
    if !ok {
        eprintln!("gall: {} is not a dspark drafter (mainline layout?); \
                   serving without drafts", d.display());
    }
    ok
}

// greedy decode with drafts; --block B fixes the drafts per round (0: none)
pub fn run(a: &[String]) {
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].as_str());
    let (n, ctx) = (opt("-n").map_or(64, |v| v.parse().unwrap()), opt("-c").map_or(4096, |v| v.parse().unwrap()));
    let mut m = Model::open(&a[0], ctx);
    m.reset();
    let t0 = Instant::now();
    let dr = Dr::open(&mut m, opt("--draft").unwrap(), ctx);
    eprintln!("drafter: open {:.2} s", t0.elapsed().as_secs_f64());
    let next = dr.feed(&mut m, &crate::ids(opt("--tokens").expect("--tokens FILE")));
    let (mut st, mut out, t0) = (Spec::new(), 0, Instant::now());
    st.block = opt("--block").map(|v| v.parse::<usize>().unwrap().min(7));
    dr.generate(&mut m, next, &mut st, |t| {
        print!("{t} ");
        out += 1;
        out < n
    });
    println!();
    let ms = |v: &[f64]| v.iter().map(|x| (1e3 * x).round()).collect::<Vec<_>>();
    eprintln!(
        "{:.1} t/s, kept {:?}, {} drafts, on {}, a draft {:.2} tokens {:.0} ms, \
         pass ms {:?}, draft ms {:?}, acceptance {:.2?}",
        out as f64 / t0.elapsed().as_secs_f64(), st.kept, st.drafts, st.on(), st.e[0],
        1e3 * st.e[1],
        ms(&st.tv[1..]), ms(&st.td[1..]), st.q
    );
}
