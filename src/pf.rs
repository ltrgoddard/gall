// batched prefill (pf.metal, pfm.metal): up to ub tokens per pass.
//   let rest = m.prefill(&toks, 256);   // then step the rest; feed() does
// the math is sched() of model.rs with a token per threadgroup and a
// mat-mat for every weight, the small matrices included. kv, conv ring and
// pos end as after stepping. one cpu sync per visit sorts the (token, k)
// expert pairs by expert; their rows live in pe, in that order.
use super::{visits, Model, CS, N};
use crate::gguf::{blk, Tensor};
use crate::gpu::*;
use crate::mv::name;
use std::collections::HashMap;

pub const MM: &str = concat!(
    "#include <metal_stdlib>\n",
    include_str!("grids.metal"),
    include_str!("mv.metal"),
    include_str!("pfm.metal")
);
const OPS: &str = include_str!("pf.metal");
// regions of a token, in floats. 0 means "none" in params.
const PRE: usize = N; // prelude output
const H: usize = 2 * N; // residual stream
const X: usize = 3 * N; // q, k, v inputs
const XF: usize = 6 * N; // ffn input
const F: usize = 7 * N; // ffn_inp
const AO: usize = 8 * N; // attention output after wo
const D: usize = 9 * N; // expert output
const BK: usize = 10 * N; // normalized block stack, 5 deep
const VT: usize = 15 * N; // mixed inputs, 3
const CAT: usize = 18 * N; // control inputs, 3 x 2N
const GF: usize = 24 * N; // control gain and forget, 3 x 2N
const Q: usize = 30 * N; // q, gate, attention out: 3072 each
const G: usize = Q + 3072;
const O: usize = G + 3072;
const K: usize = O + 3072;
const V: usize = K + 1024;
const R: usize = V + 1024;
const SK: usize = R + 384; // the small mat-mats write 64 rows
const DK: usize = SK + 64;
const DV: usize = DK + 64;
const GC: usize = DV + 64; // control low rank, 3 x 64
const RT: usize = GC + 192; // router logits
const WT: usize = RT + 128;
const MSG: usize = WT + 8; // 12 layers x 4 messages
const STR: usize = MSG + 48 * N;

pub struct Pf {
    tb: usize, // tokens per pass, a multiple of 32; token tb is padding
    np: usize, // rows of pe: 8 per token, an expert padded to 8
    a: Buf,
    pe: Buf,  // per pair: gate_up N, situ 1152, down N
    id: Buf,  // 8 expert ids per token, then the pair of each
    col: Buf, // token | k << 16 of each pair
    tl: Buf,  // tiles of the mat-mats: (expert, first column, columns)
    rt: Buf,  // relative bias by token, q head, distance
    k: HashMap<&'static str, Pipe>,
    mm: HashMap<u32, Pipe>,
    // PROF=1: by kind, gpu seconds, weight bytes, tile and real macs
    pt: std::cell::RefCell<std::collections::BTreeMap<String, [f64; 4]>>,
}

impl Pf {
    fn new(g: &Gpu, tb: usize) -> Pf {
        let d = format!("#define STR {STR}\n");
        let ks = ["prep", "ctl", "post", "relb", "attn", "mid", "top", "act", "dsum",
            "fin", "end", "msg"];
        let ty = [1, 8, 12, 13, 14, 16, 17, 18, 20, 22, 40];
        let f = |t| if t == 1 { "mm_f16".into() } else { format!("mm_{}", name(t)) };
        let np = 8 * tb + 128 * 7;
        Pf {
            tb,
            np,
            a: g.buf(4 * (tb + 1) * STR),
            pe: g.buf(4 * np * (2 * N + 1152)),
            id: g.buf(4 * (16 * tb + 8)),
            col: g.buf(4 * np),
            tl: g.buf(12 * (tb / 32 + np / 8)),
            rt: g.buf(4 * tb * 24 * 1024),
            k: ks.map(|k| (k, g.pipe(OPS, &d, k))).into(),
            mm: ty.map(|t| (t, g.pipe(MM, &d, &f(t)))).into(),
            pt: Default::default(),
        }
    }
}

impl Model {
    // evaluate toks but for a tail for step(): passes of ub tokens, then
    // one of the rest less the last token, if that is 16 or more.
    // self.t is the gpu seconds
    pub fn prefill<'a>(&mut self, toks: &'a [u32], ub: usize) -> &'a [u32] {
        self.prefill_tap(toks, ub, None)
    }

    // prefill(); tap(model, f) after each pass, before pos moves: f is the
    // drafter's 4 features of each token of the pass (DRAFT.md), 4 N each
    pub fn prefill_tap<'a>(&mut self, mut toks: &'a [u32], ub: usize,
                           mut tap: Option<&mut dyn FnMut(&Model, &[f32])>) -> &'a [u32] {
        self.t = 0.;
        self.stale(self.pos);
        while toks.len() > 16 {
            let (n, tb) = (ub.min(toks.len() - 1), ub.div_ceil(32) * 32);
            let pf = self.pf.take().filter(|p| p.tb >= tb);
            let pf = pf.unwrap_or_else(|| Pf::new(&self.gpu, tb));
            let mut ft = vec![0.; if tap.is_some() { 4 * N * n } else { 0 }];
            self.stage(&toks[..n]);
            self.t += self.pass(&pf, &toks[..n], &mut ft);
            if let Some(f) = &mut tap {
                f(self, &ft);
            }
            (self.pf, self.pos, toks) = (Some(pf), self.pos + n, &toks[n..]);
        }
        if let Some(p) = self.pf.as_ref().filter(|_| prof()) {
            for (k, [s, b, m, r]) in p.pt.take() {
                let [s, b, m, r] = [1e3 * s, b / 1e9, m / 1e9, r / 1e9];
                eprintln!("{k:8} {s:8.1} ms {b:6.2} GB {m:7.0} {r:7.0} GMAC");
            }
        }
        toks
    }

    // one pass over 16 <= t <= tb tokens; gpu seconds. ft, if not empty,
    // takes the residual entering visits 5, 23, 41 and the normed output
    fn pass(&self, pf: &Pf, toks: &[u32], ft: &mut [f32]) -> f64 {
        let (t, pos, tb) = (toks.len(), self.pos, pf.tb);
        assert!(pos + t <= self.n_ctx, "context full");
        let (col, tl) = (slice::<u32>(&pf.col), slice::<u32>(&pf.tl));
        // tiles of the dense mat-mats: all tokens, to a multiple of 8
        let nd = t.div_ceil(32);
        for i in 0..nd {
            let n = (t.div_ceil(8) * 8 - 32 * i).min(32);
            tl[3 * i..3 * i + 3].copy_from_slice(&[0, 32 * i as u32, n as u32]);
        }
        let bufs =
            [&pf.a, &self.s, &self.kv, &self.cs, &pf.id, &pf.pe, &pf.col, &pf.rt];
        // PROF: each dispatch in a command buffer of its own
        let go = |e: &Enc, k: String, w: [usize; 3], f: &dyn Fn(&Enc)| {
            if !prof() {
                return f(e);
            }
            let s = self.gpu.run(f);
            let mut pt = pf.pt.borrow_mut();
            let a = pt.entry(k).or_default();
            a[0] += s;
            (0..3).for_each(|i| a[i + 1] += w[i] as f64);
        };
        macro_rules! k {
            ($e:expr, $n:expr, $g:expr, $th:expr, $($p:expr),*) => {{
                let mut p = [0u32; 24];
                let v = [$($p as u32),*];
                p[..v.len()].copy_from_slice(&v);
                go($e, $n.into(), [0; 3],
                    &|e| call(e, &pf.k[$n], &p, &bufs.map(|b| (b, 0)), $g, $th))
            }};
        }
        // y = W x over the tiles tl[t0..][..nt]; o = [ls, xo, xs, yo, ys]
        #[allow(clippy::too_many_arguments)]
        let mm = |e: &Enc, ty: u32, w: B, x: &Buf, y: &Buf, (t0, nt): (usize, usize),
                  o: [usize; 5], ni: usize, no: usize| {
            let rb = if ty == 1 { 2 * ni } else { ni / blk(ty).0 * blk(ty).1 };
            let p = [ni, no, rb, o[0], o[1], o[2], o[3], o[4]].map(|v| v as u32);
            let b = [w, (x, 0), (y, 0), (&pf.col, 0), (&pf.tl, 12 * t0)];
            let (g, xp) = ([no.div_ceil(64), nt, 1], t0 >= nd);
            // PROF: bytes of the distinct slabs, tile and real macs
            let mut wk = [0; 3];
            if prof() {
                let tl = &slice::<u32>(&pf.tl)[3 * t0..][..3 * nt];
                let (col, tl) = (slice::<u32>(&pf.col), tl.chunks(3));
                let tb = tb as u32;
                let mut ex: Vec<_> = tl.clone().map(|c| c[0]).collect();
                ex.dedup();
                let cs = tl.flat_map(|c| c[1] as usize..(c[1] + c[2]) as usize);
                let real = |&i: &usize| [i < t, col[i] != tb][xp as usize];
                let rc = cs.clone().filter(real).count();
                wk = [ex.len() * no * rb, g[0] * 64 * ni * cs.count(),
                    no * ni * rc];
            }
            let f = if ty == 1 { "f16" } else { name(ty) };
            let k = [["", "x "][xp as usize], f].concat();
            go(e, k, wk, &|e| call(e, &pf.mm[&ty], &p, &b, g, 128))
        };
        let dense = |e: &Enc, w: &Tensor, x: usize, y: usize, ni, no| {
            let o = [0, x, STR, y, STR];
            mm(e, w.ty, self.gpu.w(w), &pf.a, &pf.a, (0, nd), o, ni, no)
        };
        // a small f16 matrix at sh[w..]
        let small = |e: &Enc, w: usize, x: usize, y: usize, ni, no| {
            let o = [0, x, STR, y, STR];
            mm(e, 1, (&self.sh, 2 * w), &pf.a, &pf.a, (0, nd), o, ni, no)
        };
        // controls c0.. of the nx inputs in VT of a looped layer, to x
        let ctl = |e: &Enc, w: u32, c0: usize, nx: usize, msg: usize, x: usize,
                   nw: [u32; 3]| {
            for j in 0..nx {
                let c = w as usize + (c0 + j) * 256 * N;
                small(e, c, CAT + j * 2 * N, GC + j * 64, 2 * N, 64);
                small(e, c + 128 * N, GC + j * 64, GF + j * 2 * N, 64, 2 * N);
            }
            k!(e, "ctl", [t, 1, 1], 32, 0, 0, 0, 0, 0, x, nx, msg, 0, c0, nw[0],
                nw[1], nw[2], VT, 0, GF);
        };
        let g1 = [t, 1, 1];
        let vs = visits();
        // blocks on the stack at prep and at mid of each visit
        let mut nb = 0;
        let nbs: Vec<_> = vs
            .iter()
            .map(|&(i, ..)| {
                nb = if i == 6 { 1 } else { nb };
                let a = nb;
                nb += (i % 6 == 0) as usize;
                (a, nb)
            })
            .collect();
        // a visit up to the router
        let head = |e: &Enc, v: usize| {
            let ((i, slot, pass), l) = (vs[v], &self.l[vs[v].0]);
            let h = if i == 6 { PRE } else { H };
            let msg = if pass > 0 { MSG + (i - 6) * 4 * N } else { 0 };
            let nx = if pass > 0 || !l.swa { 3 } else { 1 };
            let x = |j: usize| X + (nx / 3) * j * N;
            k!(e, "prep", g1, 32, h, BK, nbs[v].0, l.nq, l.aq, X, nx, msg, 0, 0,
                l.nw[0], l.nw[1], l.nw[2], VT, CAT);
            if msg > 0 {
                ctl(e, l.ctl, 0, 3, msg, X, l.nw);
            }
            dense(e, l.q, x(0), Q, N, 3072);
            dense(e, l.k, x(1), K, N, 1024);
            dense(e, l.v, x(2), V, N, 1024);
            dense(e, l.rel, x(0), R, N, 384);
            dense(e, l.gate, x(0), G, N, 3072);
            small(e, l.sink as usize, x(0), SK, N, 24);
            if !l.swa {
                small(e, l.kd as usize, x(1), DK, N, 32);
                small(e, l.vd as usize, x(2), DV, N, 32);
            }
            let (cs, kv, dk) = (slot * CS, self.kvo[slot], if l.swa { 0 } else { DK });
            k!(e, "post", g1, 32, Q, K, V, R, l.qc, l.rc, l.kc, l.vc, dk, DV, cs, pos,
                l.swa, self.hp[0].to_bits(), self.hp[1].to_bits(),
                self.hp[2].to_bits(), self.hp[3].to_bits(), kv);
            k!(e, "relb", [32, 24, t], 32, R, l.rb);
            // bound the keys of one dispatch
            let tq = ((1 << 22) / if l.swa { 1024 } else { pos + t }).max(8) / 8 * 8;
            for t0 in (0..t).step_by(tq) {
                let n = (t - t0).min(tq);
                k!(e, "attn", [8, n.div_ceil(8), 1], 96, Q, 0, SK, kv, G, pos, l.swa,
                    0, O, t0, t);
            }
            dense(e, l.o, O, AO, 3072, N);
            k!(e, "mid", g1, 32, F, BK, nbs[v].1, 1, l.fq, XF, 1, msg, 0, 3, l.fnw,
                0, 0, VT, CAT, 0, AO, if i % 6 == 0 { 0 } else { h }, cs + 2048,
                l.apc, pos);
            if msg > 0 {
                ctl(e, l.ctl, 3, 1, msg, XF, [l.fnw, 0, 0]);
            }
            small(e, l.gi as usize, XF, RT, N, 128);
            k!(e, "top", g1, 32, RT, l.pb, WT);
        };
        // the rest: np pairs in the tiles tl[nd..][..ne]
        let (gu, hh, dn) = (0, pf.np * N, pf.np * (N + 1152));
        let tail = |e: &Enc, v: usize, np: usize, ne: usize| {
            let ((i, slot, pass), l) = (vs[v], &self.l[vs[v].0]);
            let (w, o) = (self.gpu.w(l.gu), [1, XF, STR, gu, N]);
            mm(e, l.gu.ty, w, &pf.a, &pf.pe, (nd, ne), o, N, N);
            k!(e, "act", [36, np, 1], 32, gu, hh, WT, l.us, l.ds);
            let (w, o) = (self.gpu.w(l.dn), [0, hh, 1152, dn, N]);
            mm(e, l.dn.ty, w, &pf.pe, &pf.pe, (nd, ne), o, 1152, N);
            k!(e, "dsum", g1, 32, dn, D, 8 * (tb + 1));
            let cs = slot * CS;
            k!(e, "fin", g1, 32, D, F, if i == 5 { PRE } else { H }, cs + 4352,
                l.fpc, pos);
            k!(e, "end", g1, 32, pos, t, K, V, AO, D, cs);
            if i == 17 && pass < 2 {
                k!(e, "msg", g1, 32, BK + 2 * N, H, self.mq, MSG);
            }
        };
        let em = |e: &Enc, x, y| go(e, "embd".into(), [0; 3], &|e| {
            self.embd(e, x, y)
        });
        let (mut np, mut ne, mut gpu, mut oc) = (0, 0, 0., [0; 4]);
        for v in 0..=vs.len() {
            gpu += self.gpu.run(|e| {
                if v == 0 {
                    for (i, &x) in toks.iter().enumerate() {
                        em(e, x, (&pf.a, 4 * (i * STR + H)));
                    }
                    k!(e, "prep", g1, 32, H, BK, 0, 0, 0, H, 1, 0, 0, 0, self.tn, 0,
                        0, VT);
                    // an image row is the stream as it is: no norm
                    for (i, &x) in toks.iter().enumerate() {
                        if x >= super::IMG {
                            em(e, x, (&pf.a, 4 * (i * STR + H)));
                        }
                    }
                } else {
                    tail(e, v - 1, np, ne);
                }
                if v < vs.len() {
                    head(e, v);
                } else if !ft.is_empty() {
                    k!(e, "prep", g1, 32, H, BK, nbs[v - 1].1, 1, self.oq, X, 1, 0, 0,
                        0, self.on, 0, 0, VT);
                }
            });
            let i = [5, 23, 41, 48].iter().position(|&x| x == v);
            if let Some(i) = i.filter(|_| !ft.is_empty()) {
                let o = if i == 3 { X } else { H };
                for (k, f) in ft.chunks_mut(4 * N).enumerate() {
                    let a = &slice::<f32>(&pf.a)[k * STR + o..][..N];
                    f[i * N..][..N].copy_from_slice(a);
                }
            }
            // sort the (token, k) pairs by expert; an expert is padded to 8
            let id = slice::<u32>(&pf.id);
            let mut by = vec![vec![]; 128];
            for (j, &x) in id[..8 * t].iter().enumerate() {
                by[x as usize].push(j);
            }
            (np, ne) = (0, 0);
            for (x, c) in by.iter().enumerate().filter(|c| !c.1.is_empty()) {
                let n = c.len().div_ceil(8) * 8;
                col[np..np + n].fill(tb as u32);
                for (i, &j) in c.iter().enumerate() {
                    col[np + i] = (j / 8 | j % 8 << 16) as u32;
                    id[8 * (tb + 1) + j] = (np + i) as u32;
                }
                for c0 in (0..n).step_by(32) {
                    let d = [x, np + c0, (n - c0).min(32)].map(|v| v as u32);
                    tl[3 * (nd + ne)..][..3].copy_from_slice(&d);
                    ne += 1;
                }
                np += n;
            }
            if v < vs.len() {
                let nx = by.iter().filter(|c| !c.is_empty()).count();
                oc = [oc[0] + nx, oc[1] + np, oc[2] + 32 * ne, oc[3] + 1];
            }
        }
        if prof() {
            let [x, p, q, n] = oc.map(|v| v as f64);
            let r = 8. * t as f64 * n;
            eprintln!("{t} rows: {:.1} experts a visit, {:.1} rows an expert, \
                {:.2} of 8-wide, {:.2} of 32-wide", x / n, r / x, r / p, r / q);
        }
        gpu
    }
}

fn prof() -> bool {
    std::env::var("PROF").is_ok()
}

// time the mat-mat on random bytes: no model
pub fn bench() {
    let g = Gpu::new();
    let d = format!("#define STR {STR}\n");
    let (ni, no, t) = (N, 3072, 256);
    let (w, a, col, tl) = (g.buf(no * ni * 2), g.buf(4 * 257 * STR), g.buf(4096), g.buf(4096));
    let mut s = 1u32;
    for b in slice::<u8>(&w) {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 24) as u8 & 0x3f;
    }
    slice::<f32>(&a).fill(0.01);
    for i in 0..t / 32 {
        slice::<u32>(&tl)[3 * i..3 * i + 3].copy_from_slice(&[0, 32 * i as u32, 32]);
    }
    for ty in [8u32, 14, 18, 40] {
        let p = g.pipe(MM, &d, &format!("mm_{}", name(ty)));
        let rb = ni / blk(ty).0 * blk(ty).1;
        let pm = [ni, no, rb, 0, X, STR, Q, STR].map(|v| v as u32);
        for _ in 0..2 {
            let s = g.run(|e| {
                for _ in 0..10 {
                    let b = [(&w, 0), (&a, 0), (&a, 0), (&col, 0), (&tl, 0)];
                    call(e, &p, &pm, &b, [no / 64, t / 32, 1], 128);
                }
            });
            let mac = (10 * ni * no * t) as f64;
            println!("{}: {:.2} ms, {:.0} GMAC/s", name(ty), 100. * s, mac / s / 1e9);
        }
    }
}
