// prefix snapshots on disk, so a known prompt prefix is not fed again.
//   let mut s = Snaps::new(path, &m);      // index of ~/.cache/gall/*.snap
//   let (k, r) = s.prime(&mut m, &p, k, sys); // m holds p[..k], r restored
// a file: tag line, n, the ring rows past n that a batched pass took
// (Model::top), grow (a session's: write() extends it in place), the conv
// rings, the last ROWS rows of each sliding window ring, the rows of the
// full-attention slots token by token, the n token ids. the tag is zeros
// while a write is under way
use super::*;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::{fs, ops::Range, time::SystemTime};

const ROWS: usize = 1024 + 12; // the window, and truncate()'s 12 steps
const MIN: usize = 512; // shortest prefix worth a file
const KEEP: usize = 8; // files, least recently used goes first

pub struct Snaps {
    dir: PathBuf,
    tag: String,
    at: [u64; 2], // bytes before the full-attention rows, and per token
    ix: Vec<(PathBuf, Toks, bool)>,
    pub of: usize, // the length of the file the last prime() read, or 0
}
type Toks = Vec<u32>;

fn u32s(r: &mut impl Read, n: usize) -> Option<Vec<u32>> {
    let mut b = vec![0; 4 * n];
    r.read_exact(&mut b).ok()?;
    let le = |c: &[u8]| u32::from_le_bytes(c.try_into().unwrap());
    Some(b.chunks(4).map(le).collect())
}

fn common(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

fn le(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|t| t.to_le_bytes()).collect()
}

impl Model {
    // the cache rows of a snapshot at n, in halfs: the last ROWS of each
    // window ring, and token by token the full-attention rows of [a, n)
    fn snap_rows(&self, n: usize, a: usize) -> [Vec<Range<usize>>; 2] {
        let v = visits().into_iter().map(|v| (v.0 % 6 == 5, self.kvo[v.1]));
        let (f, w): (Vec<_>, Vec<_>) = v.partition(|v| v.0);
        let row = |o: usize, p: usize| o + p * 2048..o + (p + 1) * 2048;
        let ring = |&(_, o)| {
            (n + SW - ROWS..n + SW).map(move |p| row(o, p % SW))
        };
        let full = |p| f.iter().map(move |&(_, o)| row(o, p));
        [w.iter().flat_map(ring).collect(), (a..n).flat_map(full).collect()]
    }
}

impl Snaps {
    // path: the model file, part of the tag with the cache layout
    pub fn new(path: &str, m: &Model) -> Snaps {
        let (md, home) = (fs::metadata(path).unwrap(), std::env::var("HOME"));
        let at = md.modified().unwrap().duration_since(SystemTime::UNIX_EPOCH);
        let tag = format!("gall snap 3 sw {SW} cs {CS} model {} {}\n",
            md.len(), at.unwrap().as_secs());
        let b = |n| m.snap_rows(n, 0).concat().into_iter().map(|r| 2 * r.len());
        let (w, f): (usize, usize) = (b(0).sum(), b(1).sum());
        let c = tag.len() + 12 + slice::<u8>(&m.cs).len();
        let at = [c + w, f - w].map(|x| x as u64);
        let dir = PathBuf::from(home.unwrap()).join(".cache/gall");
        fs::create_dir_all(&dir).unwrap();
        let mut s = Snaps { dir, tag, at, ix: vec![], of: 0 };
        let ls = fs::read_dir(&s.dir).unwrap().flatten().map(|e| e.path());
        let ls = ls.filter(|p| p.extension().is_some_and(|x| x == "snap"));
        let ix = ls.filter_map(|p| s.head(&p).map(|h| (p, h.2, h.1[2] > 0)));
        s.ix = ix.collect();
        eprintln!("gall: {} snapshots in {}", s.ix.len(), s.dir.display());
        s
    }

    // the head ([n, top, grow]) and tokens of a file, the reader at the
    // conv rings
    fn head(&self, p: &PathBuf) -> Option<(BufReader<fs::File>, Toks, Toks)> {
        let f = fs::File::open(p).ok()?;
        let len = f.metadata().ok()?.len();
        let mut r = BufReader::with_capacity(1 << 20, f);
        let mut t = vec![0; self.tag.len()];
        r.read_exact(&mut t).ok().filter(|_| t == self.tag.as_bytes())?;
        let h = u32s(&mut r, 3)?;
        let end = self.at[0] + h[0] as u64 * self.at[1];
        (len == end + 4 * h[0] as u64).then_some(())?;
        r.seek(SeekFrom::Start(end)).ok()?;
        let toks = u32s(&mut r, h[0] as usize)?;
        r.seek(SeekFrom::Start(self.tag.len() as u64 + 12)).ok()?;
        Some((r, h, toks))
    }

    // write the state of m, which holds toks, as a session's
    pub fn write(&mut self, m: &Model, toks: &[u32]) -> io::Result<()> {
        self.put(m, toks, true)
    }

    // a session's file that toks extend takes only the rows past its end
    fn put(&mut self, m: &Model, toks: &[u32], grow: bool) -> io::Result<()> {
        let mut h = DefaultHasher::new();
        (&self.tag, toks).hash(&mut h);
        let p = self.dir.join(format!("{:016x}.snap", h.finish()));
        let ex = |e: &&(_, Toks, bool)| grow && e.2 && toks.starts_with(&e.1);
        let old = self.ix.iter().filter(ex).max_by_key(|e| e.1.len());
        let (o, a) = old.map_or((p.clone(), 0), |e| (e.0.clone(), e.1.len()));
        if o != p {
            fs::rename(&o, &p)?;
        }
        let f = fs::OpenOptions::new().create(true).write(true).open(&p)?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        w.write_all(&vec![0; self.tag.len()])?;
        let top = Some(m.top.wrapping_sub(m.pos) as u32).filter(|a| *a <= 8);
        w.write_all(&le(&[toks.len() as u32, top.unwrap_or(0), grow as u32]))?;
        w.write_all(slice::<u8>(&m.cs))?;
        let (kv, n) = (slice::<u8>(&m.kv), toks.len());
        let [ring, full] = m.snap_rows(n, a);
        for r in ring {
            w.write_all(&kv[2 * r.start..2 * r.end])?;
        }
        w.seek(SeekFrom::Start(self.at[0] + a as u64 * self.at[1]))?;
        for r in full {
            w.write_all(&kv[2 * r.start..2 * r.end])?;
        }
        w.write_all(&le(toks))?;
        let f = w.into_inner()?;
        f.set_len(self.at[0] + n as u64 * (self.at[1] + 4))?;
        f.sync_data()?;
        f.write_all_at(self.tag.as_bytes(), 0)?;
        self.ix.retain(|e| e.0 != p && e.0 != o);
        self.ix.push((p, toks.to_vec(), grow));
        // drop the least recently used
        let age = |e: &(PathBuf, _, _)| fs::metadata(&e.0)?.modified();
        self.ix.sort_by_key(|e| std::cmp::Reverse(age(e).ok()));
        for e in self.ix.drain(KEEP.min(self.ix.len())..) {
            fs::remove_file(e.0)?;
        }
        Ok(())
    }

    // read snapshot i into m
    fn read(&self, m: &mut Model, i: usize) -> Option<()> {
        let (p, toks, _) = &self.ix[i];
        let (mut r, h, _) = self.head(p).filter(|h| h.2 == *toks)?;
        m.top = toks.len() + h[1] as usize;
        r.read_exact(slice::<u8>(&m.cs)).ok()?;
        let kv = slice::<u8>(&m.kv);
        for x in m.snap_rows(toks.len(), 0).concat() {
            r.read_exact(&mut kv[2 * x.start..2 * x.end]).ok()?;
        }
        let f = fs::OpenOptions::new().append(true).open(p).ok()?;
        f.set_modified(SystemTime::now()).ok();
        (m.pos, m.ck) = (toks.len(), None);
        m.stale(0);
        Some(())
    }
    // m holds p[..k]. restore the longest stored prefix of p past k (or a
    // file that parts from p in its last 12 tokens, truncated), then feed
    // up to and store: the longest prefix p shares with a stored
    // snapshot, and p[..sys] (the system region). returns the new k and
    // the length restored. p[k..] is never empty
    pub fn prime(&mut self, m: &mut Model, p: &[u32], mut k: usize, sys: usize) -> (usize, usize) {
        let c: Vec<_> = self.ix.iter().map(|e| common(&e.1, p)).collect();
        let n = |i: usize| self.ix[i].1.len();
        let pre = |i: &usize| c[*i] + 12 >= n(*i) && c[*i] < p.len();
        let best = (0..c.len()).filter(pre).max_by_key(|&i| c[i]);
        let mut got = 0;
        self.of = 0;
        if let Some(i) = best.filter(|&i| c[i] > k) {
            let t = std::time::Instant::now();
            if self.read(m, i).is_some() && m.truncate(c[i]) {
                (k, got, self.of) = (c[i], c[i], n(i));
                eprintln!("gall: snap {k} of {} read in {:.2} s", n(i), t.elapsed().as_secs_f64());
            } else {
                // a short or foreign file: start again
                m.reset();
                k = 0;
            }
        }
        let have: Vec<_> = (0..c.len()).filter(pre).map(|i| c[i]).collect();
        let mut cuts = vec![c.iter().copied().max().unwrap_or(0), sys];
        cuts.sort();
        for n in cuts {
            // not for under half a second of feeding
            if n < MIN || n < k + 64 || n >= p.len() || have.contains(&n) {
                continue;
            }
            m.feed(&p[k..n]);
            k = n;
            let t = std::time::Instant::now();
            match self.put(m, &p[..n], false) {
                Ok(()) => eprintln!("gall: snap {n} written in {:.2} s",
                    t.elapsed().as_secs_f64()),
                Err(e) => eprintln!("gall: snap {n}: {e}"),
            }
        }
        (k, got)
    }
}
