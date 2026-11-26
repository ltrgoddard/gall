// inkling-2-mini decode, one token per command buffer (ops.metal, mv.rs).
//   let mut m = Model::open(path, n_ctx);
//   let next = m.step(token);     // greedy next id; m.logits() for all
//   m.reset();
use crate::gguf::{self, Gguf, Tensor};
use crate::gpu::*;
use crate::mv::Mv;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBarrierScope, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLDispatchType};
use objc2_metal::MTLSharedEvent;
use std::cell::Cell;
use std::collections::HashMap;

#[path = "pf.rs"]
pub mod pf;
#[path = "snap.rs"]
pub mod snap;

const N: usize = 2304;
const OPS: &str = include_str!("ops.metal");
pub const KINDS: [&str; 20] = [
    "dots", "elem", "low", "ctl", "rows", "post", "score", "smax", "gather",
    "merge", "top8", "act", "msg", "amax", "best", "pack", "scorew", "gatherw",
    "fix", "keep",
];
// regions of the activation buffer, in floats. 0 means "none" in params.
const PRE: usize = N; // prelude output
const H: usize = 2 * N; // residual stream
const X: usize = 3 * N; // q, k, v inputs
const XF: usize = 6 * N; // ffn input
const F: usize = 7 * N; // ffn_inp
const AO: usize = 8 * N; // attention output after wo
const D: usize = 9 * N; // expert output
const BK: usize = 10 * N; // normalized block stack, 5 deep
const VR: usize = 15 * N; // mixed inputs before the norm, 3
const Q: usize = 24 * N; // q, gate, attention out: 3072 each
const G: usize = Q + 3072;
const O: usize = G + 3072;
const K: usize = O + 3072; // k then v
const R: usize = K + 2048;
const SK: usize = R + 384; // sinks
const DK: usize = SK + 24; // dynamic conv taps, k then v
const GC: usize = DK + 64; // control low rank, 3 x 130
const SC: usize = GC + 390; // mix scalars, 3 x 8
const IS: usize = SC + 24; // rms scales: q input, ffn input
const LGT: usize = IS + 2; // router logits, then the ffn input v . v
const WT: usize = LGT + 132; // router weights, before the norm
const GU: usize = WT + 8; // 8 expert gate_up rows
const HH: usize = GU + 8 * N;
const MSG: usize = HH + 4 * N; // 12 layers x 4 messages
const LG: usize = MSG + 48 * N; // logits
pub const FT: usize = LG + 201024; // the residual entering visits 5, 23, 41
const PT: usize = FT + 3 * N; // attention scratch, to the end: scores
// (24 per key), their maxima per 8 keys, the 24 maxima, partials per 256
const CH: usize = 6656; // conv channels per slot
const W8: usize = 4096; // full attention reads the int8 copy from this depth
const CS: usize = 16 * CH; // conv state floats per slot, a ring of 16
const SW: usize = 2048; // sliding window ring

type T = &'static Tensor;

struct L {
    swa: bool,
    q: T,
    k: T,
    v: T,
    o: T,
    gate: T,
    rel: T,
    gu: T,
    dn: T,
    nq: u32, // attention mix queries
    aq: u32,
    nw: [u32; 3],
    sink: u32,
    qc: u32,
    rc: u32,
    rb: u32,
    kc: u32,
    vc: u32,
    kd: u32,
    vd: u32,
    apc: u32,
    fpc: u32,
    fq: u32,
    fnw: u32,
    gi: u32,
    pb: u32,
    us: u32,
    ds: u32,
    ctl: u32,
}

// (kind, visit, encode, a weight op's (mode, tensor, x, y) in floats)
pub type Op<'a> =
    (&'static str, usize, Box<dyn Fn(&Enc) + 'a>, Option<(&'static str, T, usize, usize)>);
// an image row's token id: IMG, a 19-bit key of the image, the row
// (12 bits). the same image has the same ids in every process
pub const IMG: u32 = 1 << 31;
const IMR: usize = 8192; // image rows kept
pub const ROWS: usize = 8; // activation rows, for dr.rs's batched verify
const TID: u32 = !0; // the token of id[9]
type Cb = objc2::rc::Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Ev = objc2::rc::Retained<ProtocolObject<dyn MTLSharedEvent>>;

pub struct Model {
    pub g: &'static Gguf,
    pub gpu: Gpu,
    mv: Mv,
    pipes: HashMap<&'static str, Pipe>,
    zp: std::cell::OnceCell<HashMap<&'static str, Pipe>>, // ops.metal ZR
    pub a: Buf,
    s: Buf,
    sh: Buf,
    kv: Buf,
    cs: Buf,
    pub id: Buf,
    l: Vec<L>,
    emb: T,
    im: Buf, // image embeddings, f32 rows
    pub ni: usize,
    pub ib: HashMap<u32, usize>, // first row of an image by its key
    // every image the prompt may feed, by key, until stage() copies the
    // ones a pass reads into im: the store holds a pass's images, not a chat's
    pub pend: HashMap<u32, Vec<f32>>,
    out: T,
    tn: u32,
    oq: u32,
    on: u32,
    mq: u32,
    kvo: Vec<usize>, // kv slot offsets, in halfs
    hp: [f32; 4],    // qk clip, rel clip, log floor, log alpha
    n_ctx: usize,
    k8: Buf, // int8 copy of the full-attention slots, 2304 bytes per key
    hi: Cell<usize>, // its rows [0, hi) are current
    // from W8 deep, full attention reads the copy, and f16 for the keys
    // within vt of the greatest weight. 0: f16 only
    pub vt: f32,
    pub pos: usize,
    pub top: usize, // a batched pass (dr.rs) wrote the conv rings up to here
    taps: bool,     // keep the drafter's features at FT
    pub t: f64, // gpu seconds of the last step
    ck: Option<(usize, Vec<f32>, Vec<u16>)>, // checkpoint: pos, conv, swa
    pf: Option<pf::Pf>,
    pre: Option<(usize, Cb, bool)>, // the encoded token of a position; run
    // greedy: the caller feeds step()'s result, so the next token is
    // committed before the wait, on the gpu's greedy id; its head waits
    // for ev >= k, which the next step() sets
    pub greedy: bool,
    ev: Ev,
    k: u64,
}

// f32 to f16, round to nearest; tiny values to 0
fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let (s, e, m) = (b >> 16 & 0x8000, (b >> 23 & 255) as i32 - 112, b & 0x7fffff);
    match e {
        ..=0 => s as u16,
        31.. => (s | 0x7c00) as u16,
        _ => (s + ((e as u32) << 10) + (m >> 13) + (m >> 12 & 1)) as u16,
    }
}

// (physical layer, cache slot, pass) in execution order
pub fn visits() -> Vec<(usize, usize, usize)> {
    let core = |p| (6..18).map(move |i| (i, if p > 0 { 12 * p + 6 + i } else { i }, p));
    let lin = |r: std::ops::Range<usize>| r.map(|i| (i, i, 0));
    lin(0..6).chain((0..3).flat_map(core)).chain(lin(18..24)).collect()
}

impl Drop for Model {
    fn drop(&mut self) {
        self.idle()
    }
}

impl Model {
    pub fn open(path: &str, n_ctx: usize) -> Model {
        // one map per process, kept across unloads (one model per process)
        static G: std::sync::OnceLock<Gguf> = std::sync::OnceLock::new();
        let g: &'static Gguf = G.get_or_init(|| gguf::open(path));
        let mut gpu = Gpu::new();
        gpu.map(g);
        // small tensors, as f32 in one buffer; offset 0 is N ones
        let mut s = vec![1f32; N];
        let mut sm = |n: String| -> u32 {
            let Some(t) = g.t.get(n.as_str()) else { return 0 };
            let (o, d) = (s.len() as u32, g.data(t));
            match t.ty {
                0 => s.extend(d.chunks(4).map(|b| {
                    f32::from_le_bytes(b.try_into().unwrap())
                })),
                30 => s.extend(d.chunks(2).map(|b| {
                    f32::from_bits((b[0] as u32) << 16 | (b[1] as u32) << 24)
                })),
                _ => panic!("{n}: not a small tensor"),
            }
            o
        };
        // the larger ones as f16, offsets in halfs
        let mut sh = vec![0u16; 4];
        let mut hm = |n: String| -> u32 {
            let Some(t) = g.t.get(n.as_str()) else { return 0 };
            let (o, d) = (sh.len() as u32, g.data(t));
            match t.ty {
                0 => sh.extend(d.chunks(4).map(|b| {
                    f16(f32::from_le_bytes(b.try_into().unwrap()))
                })),
                30 => sh.extend(d.chunks(2).map(|b| {
                    f16(f32::from_bits((b[0] as u32) << 16 | (b[1] as u32) << 24))
                })),
                _ => panic!("{n}: not a small tensor"),
            }
            o
        };
        let l = (0..24)
            .map(|i| {
                let t = |n: &str| &g.t[format!("blk.{i}.{n}.weight").as_str()];
                let mut w = |n: &str| sm(format!("blk.{i}.{n}"));
                let mut wh = |n: &str| hm(format!("blk.{i}.{n}"));
                let swa = i % 6 != 5;
                let aq = w("attn_res_query.weight");
                let nw = if swa {
                    [w("attn_norm.weight"); 3]
                } else {
                    ["q", "k", "v"].map(|c| w(&format!("attn_{c}_inp_norm.weight")))
                };
                // controllers, as unsloth named them (attn k v mlp) or as
                // mainline llama.cpp does (q k v ffn, loop_{c}_ctrl_{k})
                let ctl = [("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "ffn")]
                    .map(|(c, m)| {
                        ["down", "gain", "forget"].map(|k| {
                            let n = format!("loop_{c}_{k}.weight");
                            if g.t.contains_key(format!("blk.{i}.{n}").as_str()) {
                                wh(&n)
                            } else {
                                wh(&format!("loop_{m}_ctrl_{k}.weight"))
                            }
                        })[0]
                    })[0];
                L {
                    swa,
                    q: t("attn_q"),
                    k: t("attn_k"),
                    v: t("attn_v"),
                    o: t("attn_output"),
                    gate: t("attn_gate"),
                    rel: t("attn_rel"),
                    gu: t("ffn_gate_up_exps"),
                    dn: t("ffn_down_exps"),
                    nq: if aq == 0 { 0 } else if swa { 1 } else { 3 },
                    aq,
                    nw,
                    ctl,
                    sink: wh("attn_sink_proj.weight"),
                    qc: w("attn_q_clip.weight"),
                    rc: w("attn_rel_clip.weight"),
                    rb: w("attn_rel_bias.weight"),
                    kc: w("attn_k_conv.weight"),
                    vc: w("attn_v_conv.weight"),
                    kd: wh("attn_k_conv_proj.weight"),
                    vd: wh("attn_v_conv_proj.weight"),
                    apc: w("attn_post_conv.weight"),
                    fpc: w("ffn_post_conv.weight"),
                    fq: w("ffn_res_query.weight"),
                    fnw: w("ffn_norm.weight"),
                    gi: wh("ffn_gate_inp.weight"),
                    pb: w("exp_probs_b.bias"),
                    us: w("ffn_gate_up_exps.scale"),
                    ds: w("ffn_down_exps.scale"),
                }
            })
            .collect();
        let tn = sm("token_embd_norm.weight".into());
        let oq = sm("output_res_query.weight".into());
        let on = sm("output_norm.weight".into());
        // message queries of core layer i: res[2i], kv[2i], kv[2i+1], res[2i+1]
        // (unsloth layout), which are the q, k, v and ffn columns of mainline
        // llama.cpp's blk.{6+i}.loop_msg_query, [N, 4]
        let mq = if g.t.contains_key("loop_res_query.weight") {
            let (res, kvq) = (
                sm("loop_res_query.weight".into()) as usize,
                sm("loop_kv_res_query.weight".into()) as usize,
            );
            let mq = s.len() as u32;
            for i in 0..12 {
                for o in [res + 2 * i * N, kvq + 2 * i * N, kvq + (2 * i + 1) * N, res + (2 * i + 1) * N] {
                    s.extend_from_within(o..o + N);
                }
            }
            mq
        } else {
            let o: Vec<u32> = (6..18)
                .map(|l| sm(format!("blk.{l}.loop_msg_query.weight")))
                .collect();
            for (i, &x) in o.iter().enumerate() {
                assert!(x != 0 && x as usize == o[0] as usize + 4 * N * i,
                        "blk.{}.loop_msg_query: missing or not [N, 4]", 6 + i);
            }
            o[0]
        };
        let (sb, hb) = (gpu.buf(4 * s.len()), gpu.buf(2 * sh.len()));
        slice::<f32>(&sb)[..s.len()].copy_from_slice(&s);
        slice::<u16>(&hb)[..sh.len()].copy_from_slice(&sh);
        let mut kvo = vec![0];
        let mut by_slot = visits();
        by_slot.sort_by_key(|v| v.1);
        for (i, _, _) in by_slot {
            let n = if i % 6 != 5 { SW * 2048 } else { n_ctx * 2048 };
            kvo.push(kvo.last().unwrap() + n);
        }
        let nk = n_ctx.max(1024) + 256;
        let pipes = KINDS.map(|k| (k, gpu.pipe(OPS, "", k))).into();
        Model {
            mv: Mv::new(&gpu, ""),
            pipes,
            zp: Default::default(),
            a: gpu.buf(4 * ROWS * (PT + nk * 27 + 48 + (nk.div_ceil(256).max(65) + 64) * 24 * 132)),
            s: sb,
            sh: hb,
            kv: gpu.buf(2 * kvo[48]),
            cs: gpu.buf(4 * 48 * CS),
            id: gpu.buf(4 * 16 * ROWS),
            l,
            emb: &g.t["token_embd.weight"],
            im: gpu.buf(4 * N * IMR),
            ni: 0,
            ib: HashMap::new(),
            pend: HashMap::new(),
            out: &g.t["output.weight"],
            tn,
            oq,
            on,
            mq,
            kvo,
            hp: [
                g.f("inkling2.attention.qk_clip") as f32,
                g.f("inkling2.attention.relative_clip") as f32,
                g.f("inkling2.attention.log_scale_floor") as f32,
                g.f("inkling2.attention.log_scale_alpha") as f32,
            ],
            n_ctx,
            k8: gpu.buf(n_ctx * 8 * 2304),
            hi: Cell::new(0),
            vt: std::env::var("VT").map_or(0.01, |v| v.parse().unwrap()),
            pos: 0,
            top: 0,
            taps: false,
            t: 0.,
            ck: None,
            pf: None,
            pre: None,
            greedy: false,
            ev: gpu.dev.newSharedEvent().unwrap(),
            k: 0,
            g,
            gpu,
        }
    }

    // no token runs ahead: one that does is let through and waited for
    pub fn idle(&mut self) {
        if let Some((_, cb, true)) = &self.pre {
            self.ev.setSignaledValue(self.k);
            cb.waitUntilCompleted();
            self.pre = None;
        }
    }

    // the raw cache and conv state
    pub fn raw(&mut self) -> [&mut [u8]; 2] {
        self.idle();
        [slice(&self.kv), slice(&self.cs)]
    }

    // a token's embedding: a row of the table, or an image row past IMG
    pub fn embd(&self, e: &Enc, tok: u32, y: B) {
        match tok.checked_sub(IMG) {
            Some(i) => {
                let i = self.ib[&(i >> 12)] + (i & 0xfff) as usize;
                self.mv.row(e, 0, (&self.im, 0), i, y, N)
            }
            None => self.mv.row(e, self.emb.ty, self.gpu.w(self.emb), tok as usize, y, N),
        }
    }

    // the token ids of an image's n rows
    pub fn ids(key: u32, n: usize) -> std::ops::Range<u32> {
        let a = IMG | (key & 0x7ffff) << 12;
        a..a + n as u32
    }

    // the rows of image key, for stage() to copy in when a pass reads them;
    // false when one image is more than the store holds
    pub fn image(&mut self, key: u32, e: &[f32]) -> bool {
        if e.len() / N > IMR {
            return false;
        }
        self.ib.remove(&(key & 0x7ffff)); // what the store held may be another's
        self.pend.insert(key & 0x7ffff, e.to_vec());
        true
    }

    // the images toks read, in the store: one not there is copied in, and
    // when they do not fit the store starts again with only these. rows
    // already fed are in the cache, so an image is only needed while fed
    pub fn stage(&mut self, toks: &[u32]) {
        let mut keys: Vec<u32> = vec![];
        for &t in toks.iter().filter(|&&t| t >= IMG && t != TID) {
            let k = (t - IMG) >> 12;
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
        let miss: Vec<u32> = keys.iter().copied().filter(|k| !self.ib.contains_key(k)).collect();
        if miss.is_empty() {
            return;
        }
        self.idle();
        let rows = |k: &u32| self.pend.get(k).unwrap_or_else(|| panic!("image {k:x}: no rows")).len() / N;
        let fresh = self.ni + miss.iter().map(rows).sum::<usize>() > IMR;
        if fresh {
            (self.ni, self.ib) = (0, HashMap::new());
        }
        for k in if fresh { keys } else { miss } {
            let (e, a) = (&self.pend[&k], self.ni);
            assert!(a + e.len() / N <= IMR, "the images of one pass are more than the store");
            slice::<f32>(&self.im)[a * N..][..e.len()].copy_from_slice(e);
            self.ib.insert(k, a);
            self.ni = a + e.len() / N;
        }
    }

    // keep the drafter's features of every token from now on
    pub fn taps(&mut self) {
        self.idle();
        (self.taps, self.pre) = (true, None);
    }

    // the f16 rows from n on change without their int8 copy, so a token
    // encoded ahead is void too: it holds no copy of the rows before it
    fn stale(&mut self, n: usize) {
        self.idle();
        self.hi.set(self.hi.get().min(n));
        self.pre = None;
    }

    pub fn reset(&mut self) {
        (self.pos, self.ck) = (0, None);
        self.stale(0);
        slice::<f32>(&self.cs).fill(0.);
    }

    // the sliding window cache rows of the n positions before pos
    fn swa(&self, pos: usize) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let v = visits().into_iter().filter(|v| v.0 % 6 != 5);
        v.flat_map(move |(_, slot, _)| {
            (pos.saturating_sub(1024)..pos).map(move |p| {
                let o = self.kvo[slot] + p % SW * 2048;
                o..o + 2048
            })
        })
    }

    // checkpoint the state at pos, for truncate()
    pub fn save(&mut self) {
        self.idle();
        let kv = slice::<u16>(&self.kv);
        let mut w = Vec::with_capacity(self.swa(self.pos).count() * 2048);
        self.swa(self.pos).for_each(|r| w.extend_from_slice(&kv[r]));
        self.ck = Some((self.pos, slice::<f32>(&self.cs).to_vec(), w));
    }

    // where truncate() can go back to beyond the rings' few steps
    pub fn mark(&self) -> Option<usize> {
        self.ck.as_ref().map(|c| c.0)
    }

    // forget all but the first n tokens: a few steps back by the rings,
    // or back to the checkpoint. false: not possible, reset
    pub fn truncate(&mut self, n: usize) -> bool {
        // rows of a batched pass that were not kept took ring rows too
        let top = if self.top > self.pos + 8 { self.pos } else { self.top.max(self.pos) };
        if n <= self.pos && top - n <= 12 {
            self.ck.take_if(|c| c.0 > n);
        } else if let Some((_, cs, w)) = self.ck.as_ref().filter(|c| c.0 == n) {
            slice::<f32>(&self.cs).copy_from_slice(cs);
            let kv = slice::<u16>(&self.kv);
            for (r, w) in self.swa(n).zip(w.chunks(2048)) {
                kv[r].copy_from_slice(w);
            }
        } else {
            return false;
        }
        self.pos = n;
        self.stale(n);
        true
    }

    // the dispatches of one token: (kind, visit, encode)
    pub fn sched(&self, tok: u32, pos: usize, head: bool) -> Vec<Op<'_>> {
        self.sched_row(tok, pos, head, 0)
    }

    // floats of an activation row
    pub fn rs(&self) -> usize {
        slice::<f32>(&self.a).len() / ROWS
    }

    // sched() on activation row `row` (and its 16 ids)
    pub fn sched_row(&self, tok: u32, pos: usize, head: bool, row: usize) -> Vec<Op<'_>> {
        self.sched_z(&[tok], pos, head, row)
    }

    // sched() of toks at pos.. on the activation rows from `row`: every
    // small op is one dispatch for all rows (ops.metal ZR), a weight op
    // is that of the first row, for dr.rs to run on all
    pub fn sched_z(&self, toks: &[u32], pos: usize, head: bool, row: usize) -> Vec<Op<'_>> {
        let (ro, io, nz) = (4 * row * self.rs(), 64 * row, toks.len());
        assert!(pos + nz <= self.n_ctx, "context full");
        let mut ops: Vec<Op> = vec![];
        let (mut vi, mut tz) = (0, [0u32; 8]); // tz: tau of each row
        let pz = if nz == 1 { &self.pipes } else {
            self.zp.get_or_init(|| KINDS.iter().chain(&["stor"]).map(|&k| {
                (k, self.gpu.pipe(OPS, "#define ZR\n", k))
            }).collect())
        };
        // an ops.metal kernel
        macro_rules! k {
            ($n:expr, $g:expr, $th:expr, $($p:expr),*) => {{
                let mut p = [0u32; 33];
                let v = [$($p as u32),*];
                p[..v.len()].copy_from_slice(&v);
                (p[23], p[32]) = (self.rs() as u32, nz as u32);
                p[24..32].copy_from_slice(&tz);
                let (pipe, mut g): (_, [usize; 3]) = (&pz[$n], $g);
                // these take every row in one threadgroup (ops.metal ZR)
                let one = ["low", "ctl", "rows", "post"].contains(&$n);
                g[2] = if one { 1 } else { nz };
                ops.push(($n, vi, Box::new(move |e: &Enc| {
                    let b = [&self.a, &self.s, &self.kv, &self.cs, &self.id, &self.sh,
                        &self.k8];
                    let mut b = b.map(|b| (b, 0));
                    (b[0].1, b[4].1) = (ro, io);
                    call(e, pipe, &p, &b, g, $th)
                }), None));
            }};
        }
        // a weight mat-vec from a[x] to a[y]
        macro_rules! mv {
            ($n:expr, $t:expr, $x:expr, $y:expr, $ni:expr, $no:expr) => {{
                let (t, x, y): (T, usize, usize) = ($t, $x, $y);
                let (ty, w) = (t.ty, self.gpu.w(t));
                ops.push((crate::mv::name(ty), vi, Box::new(move |e: &Enc| {
                    let id = (&self.id, io);
                    let (x, y) = ((&self.a, ro + 4 * x), (&self.a, ro + 4 * y));
                    match $n {
                        "up" => self.mv.experts_up(e, t.ty, w, id, x, y, $ni, $no),
                        "down" => self.mv.experts_down(e, t.ty, w, id, x, y, $ni, $no),
                        _ => self.mv.dense(e, ty, w, x, y, $ni, $no),
                    }
                }), Some(($n, t, x, y))));
            }};
        }
        let t32 = [18, 1, 1];
        for (z, &tok) in toks.iter().enumerate() {
            let y = (&self.a, ro + 4 * (z * self.rs() + VR));
            ops.push(("row", 0, Box::new(move |e: &Enc| match tok {
                TID => {
                    let w = self.gpu.w(self.emb);
                    self.mv.row_id(e, self.emb.ty, w, (&self.id, io + 36), y, N)
                }
                _ => self.embd(e, tok, y),
            }), None));
        }
        let tok = toks[0]; // an image row comes alone
        // ZR: the conv input of every row, before a mix reads the ring
        macro_rules! stor {
            ($x:expr, $st:expr) => {
                if nz > 1 && $x > 0 {
                    k!("stor", t32, 32, $x, 0, 0, $st, 0, pos);
                }
            };
        }
        k!("dots", [2, 1, 1], 576, 0, VR, 0, 0, 0, pos, BK, 0, 0, 0, SC);
        // an image row is the stream as it is: no norm (ones, no scale)
        let (tn, sc) = if tok < IMG || tok == TID { (self.tn, 1) } else { (0, 0) };
        k!("elem", t32, 32, 0, VR, 0, 0, 0, pos, BK, 0, 0, 0, SC, 1, VR, H, tn,
            0, 0, sc);
        let (mut nb, mut fin) = (0, [0, H, 0, 0, 0]); // pending ffn conv
        // the int8 copy of the rows that prefill or a snapshot wrote
        let (lt, full) = (self.vt.ln().to_bits(), visits().into_iter().filter(|v| v.0 % 6 == 5));
        let full: Vec<_> = full.map(|v| v.1).collect();
        let k8 = |slot| full.iter().position(|&s| s == slot).unwrap() * self.n_ctx * 1152;
        let (hi, w8) = (self.hi.get().min(pos), self.vt > 0. && pos >= W8);
        if w8 {
            if hi < pos {
                self.gpu.run(|e| for &s in &full {
                    let p = [self.kvo[s] + hi * 2048, k8(s) + hi * 1152].map(|x| x as u32);
                    let mut b = [&self.kv; 7].map(|b| (b, 0));
                    b[6].0 = &self.k8;
                    call(e, &self.pipes["pack"], &p, &b, [pos - hi, 4, 1], 32)
                });
            }
            self.hi.set(pos + 1);
        }
        for (i, slot, pass) in visits() {
            let l = &self.l[i];
            let (first, lp) = (i % 6 == 0, pass > 0);
            let msg = if lp { MSG + (i - 6) * 4 * N } else { 0 };
            let nx = if lp || !l.swa { 3 } else { 1 };
            let x = |j: usize| X + (nx / 3) * j * N;
            if i == 6 {
                nb = 1;
                fin[2] = PRE;
            }
            if i == 6 && lp {
                fin = [0, PRE, 0, 0, 0];
            }
            let ([f0, f1, f2, f3, f4], [w0, w1, w2]) = (fin, l.nw);
            stor!(f0, f3);
            k!("dots", [nb + 2, l.nq.max(1) as usize, 1], 576, f0, f1, f2, f3, f4, pos,
                BK, nb, l.nq, l.aq, SC);
            k!("elem", t32, 32, f0, f1, f2, f3, f4, pos, BK, nb, l.nq, l.aq, SC, nx,
                VR, X, w0, w1, w2);
            if let Some(i) = [5, 23, 41].iter().position(|&v| v == vi && self.taps) {
                k!("keep", t32, 32, H, FT + i * N);
            }
            if lp {
                k!("low", [65, 3, 1], 576, VR, msg, l.ctl, GC);
                k!("ctl", [N / 2, 3, 1], 32, VR, msg, l.ctl, GC, X, w0, w1, w2);
            }
            mv!("mv", l.q, x(0), Q, N, 3072);
            mv!("mv", l.k, x(1), K, N, 1024);
            mv!("mv", l.v, x(2), K + 1024, N, 1024);
            mv!("mv", l.rel, x(0), R, N, 384);
            mv!("mv", l.gate, x(0), G, N, 3072);
            // (cache row, keys, tau) of the token at p
            let at = |p: usize| if l.swa {
                (p % SW, (p + 1).min(1024), 1.)
            } else {
                let t = ((p + 1) as f32 / self.hp[2]).max(1.).ln();
                (p, p + 1, 1. + self.hp[3] * t)
            };
            let ((idx, nk, tau), nkm) = (at(pos), at(pos + nz - 1).1);
            (0..nz).for_each(|z| tz[z] = at(pos + z).2.to_bits());
            let (cs, kv) = (slot * CS, self.kvo[slot]);
            // ZR: the cache row follows from the ring size
            let (kw, ring) = match nz {
                1 => (kv + idx * 2048, 0),
                _ => (kv, if l.swa { SW } else { 0 }),
            };
            k!("post", [24, 1, 1], 576, Q, K, R, l.qc, l.rc, l.kc, l.vc, l.kd, l.vd,
                cs, kw, f32::to_bits(tau), self.hp[0].to_bits(), self.hp[1].to_bits(),
                pos, VR, nx, X, IS, l.sink, SK, ring);
            let ng = nkm.div_ceil(128);
            let nm = self.n_ctx.max(1024) + 256;
            let (mx, gm, pa) = (PT + 24 * nm, PT + 27 * nm, PT + 27 * nm + 48);
            let th = (ng * 16).next_multiple_of(32).min(1024);
            let nc = if l.swa || !w8 {
                k!("score", [ng, 8, 1], 512, Q, R, PT, kv, nk, pos, l.swa, l.rb, mx);
                // up to 512 group maxima, gather finds the head maxima
                let ns = if ng <= 32 { ng * 16 } else {
                    k!("smax", [24, 1, 1], th, mx, ng * 16, SK, gm, th);
                    0
                };
                k!("gather", [nkm.div_ceil(256), 8, 1], 512, gm, 0, PT, kv, nk, pos, l.swa,
                    0, pa, mx, ns, SK);
                nkm.div_ceil(256)
            } else {
                // int8 first, f16 for the keys with weight
                let (k8, cw) = (k8(slot), (nk / 64).min(256));
                k!("pack", [nz, 4, 1], 32, kv + pos * 2048, k8 + pos * 1152);
                k!("scorew", [ng * 16, 2, 1], 32, Q, R, PT, k8, nk, pos, 0, l.rb, mx);
                k!("smax", [24, 1, 1], th, mx, ng * 16, 0, gm, th, lt);
                let nc = nkm.div_ceil(cw);
                k!("gatherw", [nc, 2, 1], 32, gm, 0, PT, 0, nk, 0, 0, 0, pa, 0, 0, k8, cw);
                k!("fix", [64, 8, 1], 32, Q, R, PT, kv, nk, pos, 0, l.rb, pa, gm, 0, 0, 64, nc);
                nc + 64
            };
            let ns = nc.min(16);
            k!("merge", [24, 1, 1], 32 * ns, pa, nc, SK, G, O, IS, gm, ns);
            mv!("mv", l.o, O, AO, 3072, N);
            nb += first as usize;
            let (r, st) = (if first { 0 } else { H }, cs + 2048);
            stor!(AO, st);
            k!("dots", [nb + 2, 1, 1], 576, AO, r, F, st, l.apc, pos, BK, nb, 1, l.fq, SC);
            k!("elem", t32, 32, AO, r, F, st, l.apc, pos, BK, nb, 1, l.fq, SC, 1, VR,
                XF, l.fnw);
            if lp {
                let (m, c) = (msg + 3 * N, l.ctl as usize + 768 * N);
                k!("low", [65, 1, 1], 576, VR, m, c, GC);
                k!("ctl", [N / 2, 1, 1], 32, VR, m, c, GC, XF, l.fnw);
            }
            k!("rows", [129, 1, 1], 576, XF, l.gi, LGT, VR);
            k!("top8", [128, 1, 1], 32, LGT, l.pb, WT);
            mv!("up", l.gu, XF, GU, N, N);
            k!("act", [9, 1, 1], 256, GU, HH, WT, l.us, l.ds, LGT + 128);
            mv!("down", l.dn, HH, D, 1152, N);
            fin = [D, F, H, cs + 4352, l.fpc as usize];
            if i == 17 && pass < 2 {
                stor!(D, fin[3]);
                k!("elem", t32, 32, D, F, H, fin[3], fin[4], pos, BK, nb, 0, 0, SC);
                let (b, q) = (BK + 2 * N, self.mq);
                k!("dots", [3, 48, 1], 576, 0, H, 0, 0, 0, pos, b, 1, 0, q, GC);
                k!("msg", [18, 48, 1], 32, b, H, GC, MSG);
            }
            vi += 1;
        }
        let [f0, f1, _, f3, f4] = fin;
        stor!(f0, f3);
        if !head {
            // only the conv state of the last ffn
            k!("elem", t32, 32, f0, f1, 0, f3, f4, pos, BK, nb, 0, 0, SC);
            return ops;
        }
        k!("dots", [nb + 2, 1, 1], 576, f0, f1, 0, f3, f4, pos, BK, nb, 1, self.oq, SC);
        k!("elem", t32, 32, f0, f1, 0, f3, f4, pos, BK, nb, 1, self.oq, SC, 1, VR, X,
            self.on);
        // v . v to LGT + 1, for the scale of the logits
        k!("dots", [2, 1, 1], 576, 0, VR, 0, 0, 0, pos, BK, 0, 0, 0, LGT);
        mv!("mv", self.out, X, LG, N, 201024);
        let c = self.g.f("inkling2.final_logit_softcapping") as f32;
        k!("amax", [1563, 1, 1], 32, LG, 200058, PT, LGT + 1, c.to_bits());
        k!("best", [1, 1, 1], 1024, PT, 1563, 8);
        ops
    }

    // the command buffer of the token at pos; the token is read from id[9].
    // gate k: the head waits for ev >= k
    fn enc(&self, pos: usize, mut gate: Option<u64>) -> Cb {
        objc2::rc::autoreleasepool(|_| {
            let cb = self.gpu.q.commandBuffer().unwrap();
            // concurrent: a barrier before each op, except between mat-vecs
            // that read none of the earlier ones' outputs (q, k, v, rel, gate)
            let ne = || cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent);
            let (mut e, mut ys) = (ne().unwrap(), vec![]);
            for o in self.sched(TID, pos, true) {
                if let Some(k) = gate.take_if(|_| o.1 == 48) {
                    e.endEncoding();
                    cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*self.ev), k);
                    e = ne().unwrap();
                }
                match o.3 {
                    Some(("mv", _, x, y)) if !ys.is_empty() && !ys.contains(&x) => ys.push(y),
                    m => {
                        e.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                        ys = m.iter().filter(|m| m.0 == "mv").map(|m| m.3).collect();
                    }
                }
                o.2(&e);
            }
            e.endEncoding();
            cb
        })
    }

    // decode one token; returns the greedy next id. the next position is
    // encoded while this one runs, so a step starts with a commit (greedy:
    // it runs too, and its head waits for the next step)
    pub fn step(&mut self, tok: u32) -> u32 {
        let p = self.pos;
        let cb = match self.pre.take() {
            Some((q, cb, true)) if q == p && tok == self.next() => {
                self.ev.setSignaledValue(self.k);
                cb
            }
            c => {
                self.pre = c;
                self.idle();
                slice::<u32>(&self.id)[9] = tok;
                let c = self.pre.take().filter(|c| c.0 == p).map(|c| c.1);
                let cb = c.unwrap_or_else(|| self.enc(p, None));
                cb.commit();
                cb
            }
        };
        self.pos += 1;
        self.pre = (p + 1 < self.n_ctx).then(|| {
            let g = self.greedy && !self.taps;
            self.k += g as u64;
            let c = self.enc(p + 1, g.then_some(self.k));
            if g {
                c.commit();
            }
            (p + 1, c, g)
        });
        cb.waitUntilCompleted();
        assert!(cb.error().is_none(), "metal: {:?}", cb.error());
        self.t = cb.GPUEndTime() - cb.GPUStartTime();
        self.next()
    }

    // decode toks (not empty), logits only for the last: no output head
    // and no wait between the others
    pub fn feed(&mut self, toks: &[u32]) -> u32 {
        self.idle();
        let toks = self.prefill(toks, 512);
        let (l, h) = toks.split_last().unwrap();
        for c in h.chunks(4) {
            self.stage(c);
            let ops = c.iter().enumerate();
            let ops: Vec<_> = ops.flat_map(|(i, &t)| self.sched(t, self.pos + i, false)).collect();
            self.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
            drop(ops);
            self.pos += c.len();
        }
        if *l < IMG {
            return self.step(*l);
        }
        // step() reads the token table: an image row goes the long way
        self.stage(&[*l]);
        let ops = self.sched(*l, self.pos, true);
        self.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
        drop(ops);
        self.pos += 1;
        self.next()
    }

    // the experts of the last visit
    pub fn experts(&self) -> Vec<u32> {
        slice::<u32>(&self.id)[..8].to_vec()
    }

    // greedy id of the last step
    pub fn next(&self) -> u32 {
        slice::<u32>(&self.id)[8]
    }

    // logits of the last step, unpadded vocabulary, after the softcap
    pub fn logits(&self) -> Vec<f32> {
        self.logits_row(0)
    }

    // those of activation row r of a batched pass (dr.rs)
    pub fn logits_row(&self, r: usize) -> Vec<f32> {
        let n = self.g.u("inkling2.vocab_size_unpadded");
        slice::<f32>(&self.a)[r * self.rs() + LG..][..n].to_vec()
    }

    // a region of the activations by ref dump name
    pub fn peek(&self, name: &str, phys: usize) -> Option<&[f32]> {
        let (o, n) = match name {
            "Qcur" => (Q, 3072),
            "Kcur" => (K, 1024),
            "Vcur" => (K + 1024, 1024),
            "Rcur" => (R, 384),
            "kqv_out" => (O, 3072),
            "gate" => (G, 3072),
            "is" => (IS, 1),
            "ffn_inp" => (F, N),
            "ffn_moe_out" => (D, N),
            "ffn_moe_situ" => (HH, 8 * 1152),
            "ffn_moe_weights_scaled" => (WT, 8), // before the norm
            "l_out" => (if phys == 5 { PRE } else { H }, N),
            "result_output" => (LG, 200058),
            _ => return None,
        };
        Some(&slice::<f32>(&self.a)[o..o + n])
    }
}
