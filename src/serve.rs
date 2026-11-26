// openai- and anthropic-compatible chat server over one model: chats
// queue, one at a time.
//   gall serve <gguf> [--port 9940] [--host 127.0.0.1] [--ctx 102400]
//             [--effort none] [--alias inkling-2-mini] [--api-key-file F]
//             [--idle-unload SECONDS] [--no-draft] [--dummy [answer]]
// POST /v1/chat/completions or /v1/messages (anthropic, see anth()), stream
// or not, GET /v1/models, POST /park, GET /health ({status, loaded, busy,
// last: the timings of the last chat}). with a key, all but /health need
// "authorization: bearer KEY" or "x-api-key: KEY"; a host other than
// loopback needs a key. a park puts the model to disk (a
// snapshot of the last conversation) and out of memory, until the next
// chat request: after --idle-unload seconds idle (longer after tool_calls),
// on POST /park, at memory pressure (warn) and, before an exit, on SIGTERM;
// it waits for the chat that runs. with the dspark drafter beside the
// model, tokens are verified ahead in batches where that pays (dr.rs); the
// logits of every step stay those of the model.
// the chat template of the gguf is written out in prompt(); the model
// answers in blocks, <|message_model|> then <|content_thinking|>,
// <|content_text|> or NAME<|content_invoke_tool_json|>, each closed by
// <|end_message|>. a new prompt that extends the last one is fed from
// where they part.
use crate::{dr::{Dr, Spec}, model::Model, tok::Tok, vis::Vis};
use std::{collections::HashMap, ops::Range};
use std::fmt::{self, Write as _};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{mpsc::{channel, Sender}, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub trait Lm {
    fn reset(&mut self);
    // feed one token; logits of the next one over the real vocabulary
    fn step(&mut self, tok: u32) -> &[f32];
    // tokens fed since reset
    fn len(&self) -> usize;
    // forget all but the first n tokens; false if that needs a reset
    fn truncate(&mut self, n: usize) -> bool;
    // a length truncate() can always go back to (the checkpoint), if any
    fn mark(&self) -> Option<usize> {
        None
    }
    // step() over toks (not empty); override to batch
    fn feed(&mut self, toks: &[u32]) -> &[f32] {
        let (l, h) = toks.split_last().unwrap();
        for &t in h {
            self.step(t);
        }
        self.step(*l)
    }
    // holds p[..k]: restore or feed and store prefixes of p up to the
    // system region p[..sys]; the new k, the length restored and the
    // length of the snapshot it came from
    fn prime(&mut self, _p: &[u32], k: usize, _sys: usize) -> (usize, usize, usize) {
        (k, 0, 0)
    }
    // forget the images not among these keys (those of the prompt)
    fn keep(&mut self, _keys: &[u32]) {}
    // store toks, which it holds (or nothing, if empty), and free memory
    fn park(&mut self, _toks: &[u32]) {}
    fn loaded(&self) -> bool {
        true
    }
    // the token ids that stand for the image at url (data: or http)
    fn image(&mut self, _url: &str) -> Result<Range<u32>, String> {
        Err("this server has no vision model".into())
    }
    // drafting since the last call, for the log
    fn draft(&mut self) -> String {
        String::new()
    }
    // a draft of the tokens after the next one fed, each with its chance
    fn copy(&mut self, _d: Vec<(u32, f32)>) {}
    // the caller feeds the first maximum of step()'s logits
    fn greedy(&mut self, _on: bool) {}
}

// an Lm that load() makes again on first use after park()
type Load = Box<dyn Fn() -> Box<dyn Lm>>;
struct Lazy(Option<Box<dyn Lm>>, Load);

impl Lazy {
    fn get(&mut self) -> &mut dyn Lm {
        &mut **self.0.get_or_insert_with(|| {
            let t = Instant::now();
            let m = objc2::rc::autoreleasepool(|_| (self.1)());
            UP.store(true, Relaxed);
            eprintln!("gall: loaded in {:.2} s", t.elapsed().as_secs_f64());
            m
        })
    }
}

impl Lm for Lazy {
    fn reset(&mut self) {
        self.get().reset()
    }
    fn step(&mut self, t: u32) -> &[f32] {
        self.get().step(t)
    }
    fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |m| m.len())
    }
    fn truncate(&mut self, n: usize) -> bool {
        self.get().truncate(n)
    }
    fn mark(&self) -> Option<usize> {
        self.0.as_ref().and_then(|m| m.mark())
    }
    fn feed(&mut self, toks: &[u32]) -> &[f32] {
        self.get().feed(toks)
    }
    fn prime(&mut self, p: &[u32], k: usize, sys: usize) -> (usize, usize, usize) {
        self.get().prime(p, k, sys)
    }
    fn keep(&mut self, keys: &[u32]) {
        if let Some(m) = &mut self.0 {
            m.keep(keys)
        }
    }
    fn park(&mut self, toks: &[u32]) {
        if let Some(mut m) = self.0.take() {
            // drop m in the pool
            objc2::rc::autoreleasepool(move |_| m.park(toks));
            crate::alloc::trim();
            crate::up::lock(false);
            eprintln!("gall: unloaded");
        }
    }
    fn loaded(&self) -> bool {
        self.0.is_some()
    }
    fn image(&mut self, u: &str) -> Result<Range<u32>, String> {
        self.get().image(u)
    }
    fn draft(&mut self) -> String {
        self.get().draft()
    }
    fn copy(&mut self, d: Vec<(u32, f32)>) {
        self.get().copy(d)
    }
    fn greedy(&mut self, on: bool) {
        self.get().greedy(on)
    }
}

// ---- json

#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

static NULL: J = J::Null;

macro_rules! from {
    ($($t:ty => $f:expr),*) => {$(
        impl From<$t> for J {
            fn from(v: $t) -> J { $f(v) }
        }
    )*};
}
from!(&str => |v: &str| J::Str(v.into()), String => J::Str, f64 => J::Num,
    usize => |v| J::Num(v as f64), bool => J::Bool, Vec<J> => J::Arr);

macro_rules! obj {
    ($($k:expr => $v:expr),* $(,)?) => {
        J::Obj(vec![$(($k.to_string(), J::from($v))),*])
    };
}

impl std::ops::Index<&str> for J {
    type Output = J;
    fn index(&self, k: &str) -> &J {
        let J::Obj(v) = self else { return &NULL };
        v.iter().find(|x| x.0 == k).map_or(&NULL, |x| &x.1)
    }
}

impl fmt::Display for J {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.dump(false))
    }
}

struct P<'a>(&'a [u8], usize);

impl P<'_> {
    fn eat(&mut self, s: &str) -> bool {
        while self.0.get(self.1).is_some_and(u8::is_ascii_whitespace) {
            self.1 += 1;
        }
        let ok = self.0[self.1..].starts_with(s.as_bytes());
        self.1 += if ok { s.len() } else { 0 };
        ok
    }
    fn hex(&mut self) -> Option<u32> {
        let s = std::str::from_utf8(self.0.get(self.1..self.1 + 4)?).ok()?;
        self.1 += 4;
        u32::from_str_radix(s, 16).ok()
    }
    fn string(&mut self) -> Option<String> {
        let mut o = vec![];
        loop {
            let mut c = *self.0.get(self.1)?;
            self.1 += 1;
            match c {
                b'"' => return String::from_utf8(o).ok(),
                b'\\' => {
                    c = *self.0.get(self.1)?;
                    self.1 += 1;
                    let c = match c {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'b' => '\x08',
                        b'f' => '\x0c',
                        b'u' => {
                            let mut u = self.hex()?;
                            let hi = (0xd800..0xdc00).contains(&u);
                            if hi && self.0[self.1..].starts_with(b"\\u") {
                                self.1 += 2;
                                let lo = self.hex()?.wrapping_sub(0xdc00);
                                u = 0x10000 + ((u - 0xd800) << 10) + lo;
                            }
                            char::from_u32(u).unwrap_or('\u{fffd}')
                        }
                        c => c as char,
                    };
                    o.extend(c.encode_utf8(&mut [0; 4]).bytes());
                }
                c => o.push(c),
            }
        }
    }
    fn val(&mut self) -> Option<J> {
        Some(if self.eat("null") {
            J::Null
        } else if self.eat("true") {
            J::Bool(true)
        } else if self.eat("false") {
            J::Bool(false)
        } else if self.eat("\"") {
            J::Str(self.string()?)
        } else if self.eat("[") {
            let mut v = vec![];
            while !self.eat("]") {
                if !v.is_empty() && !self.eat(",") {
                    return None;
                }
                v.push(self.val()?);
            }
            J::Arr(v)
        } else if self.eat("{") {
            let mut v = vec![];
            while !self.eat("}") {
                if !v.is_empty() && !self.eat(",") || !self.eat("\"") {
                    return None;
                }
                let k = self.string()?;
                if !self.eat(":") {
                    return None;
                }
                v.push((k, self.val()?));
            }
            J::Obj(v)
        } else {
            let a = self.1;
            while self.0.get(self.1).is_some_and(|c| b"+-.eE".contains(c)
                || c.is_ascii_digit())
            {
                self.1 += 1;
            }
            let s = std::str::from_utf8(&self.0[a..self.1]).ok()?;
            J::Num(s.parse().ok()?)
        })
    }
}

impl J {
    pub fn parse(b: &[u8]) -> Option<J> {
        let mut p = P(b, 0);
        let v = p.val()?;
        (!p.eat("\0") && p.1 == b.len()).then_some(v)
    }
    fn s(&self) -> Option<&str> {
        if let J::Str(s) = self { Some(s) } else { None }
    }
    fn f(&self) -> Option<f64> {
        if let J::Num(n) = self { Some(*n) } else { None }
    }
    fn a(&self) -> &[J] {
        if let J::Arr(v) = self { v } else { &[] }
    }
    // jinja truth
    fn t(&self) -> bool {
        match self {
            J::Null => false,
            J::Bool(b) => *b,
            J::Num(n) => *n != 0.0,
            J::Str(s) => !s.is_empty(),
            J::Arr(v) => !v.is_empty(),
            J::Obj(v) => !v.is_empty(),
        }
    }
    fn or(&self, d: J) -> J {
        if self.t() { self.clone() } else { d }
    }
    fn put(&self, o: &mut String, sort: bool) {
        match self {
            J::Null => o.push_str("null"),
            J::Bool(b) => write!(o, "{b}").unwrap(),
            J::Num(n) => write!(o, "{n}").unwrap(),
            J::Str(s) => {
                o.push('"');
                for c in s.chars() {
                    match c {
                        '"' => o.push_str("\\\""),
                        '\\' => o.push_str("\\\\"),
                        '\n' => o.push_str("\\n"),
                        '\r' => o.push_str("\\r"),
                        '\t' => o.push_str("\\t"),
                        '\x08' => o.push_str("\\b"),
                        '\x0c' => o.push_str("\\f"),
                        c if c < ' ' => {
                            write!(o, "\\u{:04x}", c as u32).unwrap()
                        }
                        c => o.push(c),
                    }
                }
                o.push('"');
            }
            J::Arr(v) => {
                o.push('[');
                for (i, x) in v.iter().enumerate() {
                    o.push_str(if i > 0 { "," } else { "" });
                    x.put(o, sort);
                }
                o.push(']');
            }
            J::Obj(v) => {
                let mut v: Vec<_> = v.iter().collect();
                if sort {
                    v.sort_by(|a, b| a.0.cmp(&b.0));
                }
                o.push('{');
                for (i, (k, x)) in v.into_iter().enumerate() {
                    o.push_str(if i > 0 { "," } else { "" });
                    J::from(k.as_str()).put(o, sort);
                    o.push(':');
                    x.put(o, sort);
                }
                o.push('}');
            }
        }
    }
    fn dump(&self, sort: bool) -> String {
        let mut o = String::new();
        self.put(&mut o, sort);
        o
    }
}

// ---- chat template (tokenizer.chat_template of the gguf, by hand)

const END: &str = "<|end_message|>";

fn effort(e: &J) -> Result<String, String> {
    let map = [("none", 0.0), ("minimal", 0.1), ("low", 0.2), ("medium", 0.7),
        ("high", 0.9), ("xhigh", 0.99), ("max", 0.99)];
    let n = match e {
        J::Str(s) => map
            .iter()
            .find(|m| m.0 == s.trim())
            .ok_or(format!("Unknown reasoning_effort: {s}"))?
            .1,
        J::Num(n) => *n,
        _ => 0.9,
    };
    if !(0.0..=0.99).contains(&n) {
        return Err("reasoning_effort must be in [0.0, 0.99]".into());
    }
    Ok(format!(
        "<|message_system|><|content_text|>Thinking effort level: {n}{END}"
    ))
}

// r: the request; eff: the reasoning effort if the request has none
// the prompt, and the length of its system region
fn b64(s: &str) -> Vec<u8> {
    let (mut o, mut a, mut n) = (vec![], 0u32, 0);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - 65,
            b'a'..=b'z' => c - 71,
            b'0'..=b'9' => c + 4,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => continue,
        };
        (a, n) = (a << 6 | v as u32, n + 6);
        if n >= 8 {
            n -= 8;
            o.push((a >> n) as u8);
        }
    }
    o
}

// the prompt, the length of its system region, and the images: where
// in the prompt each one's rows go, and its url
type Prompt = (String, usize, Vec<(usize, String)>);
fn prompt(r: &J, eff: &J) -> Result<Prompt, String> {
    let mut urls = vec![];
    let (mut o, mut sys) = (String::new(), None);
    let tools = r["tools"].a();
    if !tools.is_empty() {
        let spec = |t: &J| {
            let f = if t["function"] == NULL { t } else { &t["function"] };
            obj!["description" => f["description"].or("".into()),
                "name" => f["name"].clone(),
                "parameters" => f["parameters"].or(obj![]),
                "type" => t["type"].or("function".into())]
        };
        let specs = J::Arr(tools.iter().map(spec).collect());
        o += "<|message_system|>tool_declare<|content_xml|>";
        o += &(specs.dump(true) + END);
    }
    let e = [&r["reasoning_effort"], &r["chat_template_kwargs"]["reasoning_effort"]];
    let e = e.into_iter().find(|e| **e != NULL).unwrap_or(eff);
    let mut eff = Some(effort(e)?);
    let msgs = r["messages"].a();
    for m in msgs {
        let role = m["role"].s().unwrap_or("");
        let mut h = match role {
            "user" => "<|message_user|>",
            "assistant" => "<|message_model|>",
            "system" => "<|message_system|>",
            "tool" => "<|message_tool|>",
            _ => return Err(format!("Unknown message role: {role}")),
        }
        .to_string();
        if role != "system" {
            o += &eff.take().unwrap_or_default();
            sys.get_or_insert(o.len());
        }
        let asst = role == "assistant";
        let calls = if asst { m["tool_calls"].a() } else { &[] };
        if role == "tool" && !m["name"].t() && m["tool_call_id"].t() {
            for tc in msgs.iter().flat_map(|p| p["tool_calls"].a()) {
                if tc["id"] == m["tool_call_id"] {
                    h += tc["function"]["name"].s().unwrap_or("");
                }
            }
        } else if asst || role == "tool" {
            h += m["name"].s().unwrap_or("");
        }
        let rc = m["reasoning_content"].s().filter(|_| asst).unwrap_or("");
        if !rc.is_empty() {
            o += &format!("{h}<|content_thinking|>{rc}{END}");
        }
        macro_rules! part {
            ($k:expr, $t:expr) => {
                o += &format!("{h}<|content_{}|>{}{END}", $k, $t)
            };
        }
        match &m["content"] {
            J::Str(c) => {
                if !c.is_empty() || calls.is_empty() && rc.is_empty() {
                    part!("text", c)
                }
            }
            c => {
                for p in c.a() {
                    match (p, p["type"].s()) {
                        (J::Str(s), _) => part!("text", s),
                        (_, None | Some("text" | "input_text")) => {
                            part!("text", p["text"].s().unwrap_or(""))
                        }
                        (_, Some("image" | "input_image" | "image_url")) => {
                            if !["user", "tool"].contains(&role) {
                                return Err("Image content is only supported \
                                    for user and tool messages".into());
                            }
                            let u = [&p["image_url"]["url"], &p["image_url"],
                                &p["url"], &p["image"]];
                            let u = u.iter().find_map(|u| u.s());
                            let u = u.ok_or("image part without a url")?;
                            part!("image", "");
                            urls.push((o.len() - END.len(), u.into()));
                        }
                        (_, Some(t)) => {
                            return Err(format!(
                                "Unsupported content part type: {t}"
                            ))
                        }
                    }
                }
            }
        }
        for tc in calls {
            let f = &tc["function"];
            let n = f["name"].s().ok_or("tool call needs a function name")?;
            let args = match &f["arguments"] {
                J::Str(s) => J::parse(s.as_bytes()),
                a => Some(a.clone()),
            };
            let body = match (args.map(|a| a.or(obj![])), &f["arguments"]) {
                (Some(J::Obj(mut kv)), _) => {
                    kv.sort_by(|a, b| a.0.cmp(&b.0));
                    let kv: Vec<_> = kv
                        .iter()
                        .map(|(k, v)| format!("{}:{v}", J::from(k.as_str())))
                        .collect();
                    format!("{{\"name\":{},\"args\":{{{}}}}}", J::from(n), kv.join(","))
                }
                // a call the model wrote badly comes back as it wrote it
                (_, J::Str(s)) => s.clone(),
                (_, a) => a.to_string(),
            };
            o += &format!("<|message_model|>{n}<|content_invoke_tool_json|>{body}{END}");
        }
        if asst {
            o += "<|content_model_end_sampling|>";
        }
    }
    let o = o + &eff.unwrap_or_default();
    let n = sys.unwrap_or(o.len());
    Ok((o, n, urls))
}

// an anthropic messages request as a chat completions one: tool_result
// blocks are tool messages, the other blocks of a user turn follow them;
// tools without an input_schema (server tools) are left out
fn anth(a: &J) -> J {
    let txt = |c: &J| match c {
        J::Str(s) => s.clone(),
        c => c.a().iter().filter(|b| b["type"].s() == Some("text"))
            .filter_map(|b| b["text"].s()).collect::<Vec<_>>().join("\n"),
    };
    let part = |b: &J| {
        let (s, ty) = (&b["source"], b["type"].s());
        let u = match s["type"].s() {
            Some("url") => s["url"].clone(),
            _ => format!("data:{};base64,{}", s["media_type"].s()
                .unwrap_or(""), s["data"].s().unwrap_or("")).into(),
        };
        let i = obj!["type" => "image_url", "image_url" => u];
        if ty == Some("image") { i } else { b.clone() }
    };
    let fun = |n: &J, k: &str, v: &J, d: Option<&J>| {
        let mut f = vec![("name".into(), n.clone()), (k.into(), v.clone())];
        f.extend(d.map(|d| ("description".into(), d.or("".into()))));
        obj!["type" => "function", "function" => J::Obj(f)]
    };
    let mut m = vec![];
    if a["system"] != NULL {
        m.push(obj!["role" => "system", "content" => txt(&a["system"])]);
    }
    for x in a["messages"].a() {
        let (role, c) = (x["role"].s().unwrap_or(""), &x["content"]);
        let ty = |t| c.a().iter().filter(move |b| b["type"].s() == Some(t));
        if let J::Str(_) = c {
            m.push(obj!["role" => role, "content" => c.clone()]);
        } else if role == "assistant" {
            let calls = ty("tool_use").map(|b| {
                let J::Obj(mut f) = fun(&b["name"], "arguments", &b["input"],
                    None) else { unreachable!() };
                f.push(("id".into(), b["id"].clone()));
                J::Obj(f)
            });
            let rc = ty("thinking").filter_map(|b| b["thinking"].s());
            m.push(obj!["role" => role, "content" => txt(c),
                "reasoning_content" => rc.collect::<String>(),
                "tool_calls" => calls.collect::<Vec<_>>()]);
        } else {
            for b in ty("tool_result") {
                let r = match &b["content"] {
                    J::Str(s) => vec![obj!["type" => "text", "text" => &**s]],
                    r => r.a().iter().map(part).collect(),
                };
                let e = vec![obj!["type" => "text", "text" => ""]];
                m.push(obj!["role" => "tool", "tool_call_id" =>
                    b["tool_use_id"].clone(),
                    "content" => if r.is_empty() { e } else { r }]);
            }
            let p: Vec<_> = c.a().iter()
                .filter(|b| b["type"].s() != Some("tool_result"))
                .map(part).collect();
            if !p.is_empty() {
                m.push(obj!["role" => role, "content" => p]);
            }
        }
    }
    let tools = a["tools"].a().iter().filter(|t| t["input_schema"] != NULL)
        .map(|t| fun(&t["name"], "parameters", &t["input_schema"],
            Some(&t["description"])));
    let (oc, off) = (&a["output_config"], a["thinking"]["type"].s());
    let f = [&oc["format"], &a["output_format"]].into_iter()
        .find(|f| f["type"].s() == Some("json_schema"));
    let rf = f.map_or(NULL.clone(), |f| obj!["type" => "json_schema",
        "json_schema" => obj!["schema" => f["schema"].clone()]]);
    let e = if off == Some("disabled") { "none".into() }
        else { oc["effort"].clone() };
    let mut o = obj!["messages" => m, "tools" => tools.collect::<Vec<_>>(),
        "stop" => a["stop_sequences"].clone(), "reasoning_effort" => e,
        "response_format" => rf];
    let J::Obj(kv) = &mut o else { unreachable!() };
    for k in ["max_tokens", "temperature", "top_p", "top_k", "stream"] {
        kv.push((k.into(), a[k].clone()));
    }
    o
}

// anthropic events. a block opens at the first delta of its type and
// closes at another type, at a tool call or at the end (b: null)
fn ev(w: &mut impl Write, e: &str, j: J) -> io::Result<()> {
    write!(w, "event: {e}\ndata: {j}\n\n")?;
    w.flush()
}
struct Ev(usize, Option<String>); // blocks opened, the type of the open one
impl Ev {
    fn to(&mut self, w: &mut impl Write, b: J) -> io::Result<()> {
        let k = b["type"].s().map(String::from);
        let tool = k.as_deref() == Some("tool_use");
        if self.1.is_some() && (self.1 != k || tool) {
            ev(w, "content_block_stop", obj!["type" => "content_block_stop",
                "index" => self.0 - 1])?;
            self.1 = None;
        }
        if k.is_some() && self.1.is_none() {
            ev(w, "content_block_start", obj!["type" => "content_block_start",
                "index" => self.0, "content_block" => b])?;
            (self.0, self.1) = (self.0 + 1, k);
        }
        Ok(())
    }
    fn delta(&mut self, w: &mut impl Write, b: J, d: J) -> io::Result<()> {
        self.to(w, b)?;
        ev(w, "content_block_delta", obj!["type" => "content_block_delta",
            "index" => self.0 - 1, "delta" => d])
    }
}
// a block of thinking (0) or text (1) that holds s, and its delta
fn blk(i: usize, s: &str) -> (J, J) {
    let k = ["thinking", "text"][i];
    let mut b = vec![("type".into(), k.into()), (k.into(), s.into())];
    if i == 0 {
        b.push(("signature".into(), "".into()));
    }
    let d = format!("{k}_delta");
    (J::Obj(b), J::Obj(vec![("type".into(), d.into()), (k.into(), s.into())]))
}
fn tool_use(c: &J) -> J {
    let (f, id) = (&c["function"], c["id"].clone());
    let a = J::parse(f["arguments"].s().unwrap_or("").as_bytes());
    let a = a.filter(|a| matches!(a, J::Obj(_))).unwrap_or(obj![]);
    obj!["type" => "tool_use", "id" => id, "name" => f["name"].clone(),
        "input" => a]
}

// ---- sampling: top_k, top_p, min_p, temperature, in llama.cpp's order

struct Sp {
    temp: f64,
    top_k: f64,
    top_p: f64,
    min_p: f64,
}

fn probs(c: &[(f32, u32)], t: f32) -> Vec<f32> {
    let m = c.iter().fold(f32::MIN, |m, x| m.max(x.0));
    let p: Vec<f32> = c.iter().map(|x| ((x.0 - m) / t).exp()).collect();
    let z: f32 = p.iter().sum();
    p.iter().map(|x| x / z).collect()
}

// the token at u of the softmax of l at temperature t, as sample() finds
// it with no filter, to the bit, without most of its 200k exps: a term
// under half an ulp of a float32 sum or of u leaves it as it is, so it is
// not computed. such a term has (x - m) / t < ln(sum) - 17.4 (e^-17.4 <
// 2^-25); `under` is that bound on x, with room for its own rounding
fn pick(l: &[f32], t: f32, mut u: f32) -> u32 {
    let m = l.iter().fold(f32::MIN, |m, &x| m.max(x));
    let under = |s: f32| m + (s.ln() - 17.5) * t - 1e-3;
    let (mut e, mut z, mut z0, mut cut) = (vec![-1f32; l.len()], 0f32, 0f32, f32::MIN);
    for (i, &x) in l.iter().enumerate() {
        if x >= cut {
            e[i] = ((x - m) / t).exp();
            z += e[i];
            if z >= 2. * z0 {
                (z0, cut) = (z, under(z));
            }
        }
    }
    let (mut u0, mut cut) = (u / 2., under(u / 2.));
    for (i, &x) in l.iter().enumerate() {
        if e[i] < 0. && x < cut {
            continue;
        }
        u -= if e[i] < 0. { ((x - m) / t).exp() } else { e[i] } / z;
        if u < 0. {
            return i as u32;
        } else if u < u0 {
            (u0, cut) = (u / 2., under(u / 2.));
        }
    }
    l.len() as u32 - 1
}

fn sample(l: &[f32], p: &Sp, rng: &mut u64) -> u32 {
    let draw = |rng: &mut u64| {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        (*rng >> 40) as f32 / (1u64 << 24) as f32
    };
    if p.top_k as usize == 0 && p.temp <= 0.0 {
        // the first maximum
        return (0..l.len()).fold(0, |m, i| if l[i] > l[m] { i } else { m }) as u32;
    } else if p.top_k as usize == 0 && p.top_p >= 1.0 && p.min_p <= 0.0 {
        return pick(l, p.temp as f32, draw(rng));
    }
    let mut c: Vec<(f32, u32)> = l.iter().copied().zip(0..).collect();
    let k = p.top_k as usize;
    if k > 0 && k < c.len() {
        c.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        c.truncate(k);
    }
    if p.temp <= 0.0 {
        // the first maximum
        return c.iter().fold(c[0], |m, &x| if x.0 > m.0 { x } else { m }).1;
    }
    if p.top_p < 1.0 {
        let pr = probs(&c, 1.0);
        let by = |a: &usize, b: &usize| pr[*b].total_cmp(&pr[*a]);
        let mut ix: Vec<usize> = (0..c.len()).collect();
        // the head nearly always holds top_p: sort it alone first
        let h = ix.len().min(1024);
        if h < ix.len() {
            ix.select_nth_unstable_by(h - 1, by);
        }
        ix[..h].sort_unstable_by(by);
        if (ix[..h].iter().map(|&i| pr[i]).sum::<f32>() as f64) < p.top_p {
            ix.sort_unstable_by(by);
        }
        let mut s = 0.0;
        let n = ix.iter().position(|&i| {
            s += pr[i] as f64;
            s >= p.top_p
        });
        ix.truncate(n.map_or(ix.len(), |n| n + 1));
        c = ix.iter().map(|&i| c[i]).collect();
    }
    if p.min_p > 0.0 {
        let m = c.iter().fold(f32::MIN, |m, x| m.max(x.0));
        c.retain(|x| x.0 >= m + p.min_p.ln() as f32);
    }
    let mut u = draw(rng);
    let pr = probs(&c, p.temp as f32);
    let i = pr.iter().position(|&x| {
        u -= x;
        u < 0.0
    });
    c[i.unwrap_or(c.len() - 1)].1
}

// ---- response_format: text that is json and fits a schema (type,
// properties, required, additionalProperties: false, items, enum, const,
// anyOf, oneOf; other keywords are not checked)

fn ws(b: &[u8], i: &mut usize) {
    while b.get(*i).is_some_and(|c| b" \n\r\t".contains(c)) {
        *i += 1;
    }
}

// a value of schema s at b[*i..]. Some(true): read whole; Some(false): b
// ends inside a value that still fits; None: no fit
fn fits(b: &[u8], i: &mut usize, s: &J) -> Option<bool> {
    for k in ["anyOf", "oneOf"] {
        if s[k] != NULL {
            let f = |s| {
                let mut j = *i;
                fits(b, &mut j, s).map(|d| (d, j))
            };
            let (d, j) = s[k].a().iter().filter_map(f).max()?;
            *i = j;
            return Some(d);
        }
    }
    ws(b, i);
    let Some(&c) = b.get(*i) else { return Some(false) };
    let ty = match c {
        b'{' => "object",
        b'[' => "array",
        b'"' => "string",
        b't' | b'f' => "boolean",
        b'n' => "null",
        b'-' | b'0'..=b'9' => "number",
        _ => return None,
    };
    let int = |t: &J| t.s() == Some("integer");
    let ok = |t: &J| t.s() == Some(ty) || ty == "number" && int(t);
    let (want, a) = (&s["type"], *i);
    if !(*want == NULL || ok(want) || want.a().iter().any(ok)) {
        return None;
    }
    let done = raw(b, i, s, int(want))?;
    let en = s["enum"].a().iter().chain((s["const"] != NULL).then_some(&s["const"]));
    let (mut en, t) = (en.peekable(), &b[a..*i]);
    let is = |e: &J| match done {
        true => J::parse(t).as_ref() == Some(e),
        _ => e.to_string().as_bytes().starts_with(t),
    };
    (en.peek().is_none() || en.any(is)).then_some(done)
}

fn raw(b: &[u8], i: &mut usize, s: &J, int: bool) -> Option<bool> {
    let c = b[*i];
    if c == b'{' || c == b'[' {
        let (obj, mut seen) = (c == b'{', vec![]);
        *i += 1;
        ws(b, i);
        if b.get(*i) == Some(&(c + 2)) {
            *i += 1;
        } else {
            loop {
                let mut sub = &s["items"];
                if obj {
                    ws(b, i);
                    let a = *i;
                    match b.get(a) {
                        Some(b'"') if raw(b, i, &NULL, false)? => {}
                        c => return c.is_none_or(|c| *c == b'"').then_some(false),
                    }
                    let J::Str(k) = J::parse(&b[a..*i])? else { return None };
                    ws(b, i);
                    match b.get(*i) {
                        Some(b':') => *i += 1,
                        c => return c.is_none().then_some(false),
                    }
                    sub = &s["properties"][k.as_str()];
                    if *sub == NULL && s["additionalProperties"] == J::Bool(false) {
                        return None;
                    }
                    seen.push(k);
                }
                if !fits(b, i, sub)? {
                    return Some(false);
                }
                ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(&e) if e == c + 2 => break *i += 1,
                    e => return e.is_none().then_some(false),
                }
            }
        }
        let has = |r: &J| r.s().is_some_and(|r| seen.iter().any(|k| k == r));
        return s["required"].a().iter().all(has).then_some(true);
    } else if c == b'"' {
        *i += 1;
        loop {
            match b.get(*i) {
                None => return Some(false),
                Some(b'"') => break *i += 1,
                Some(b'\\') => *i += 2,
                Some(0..32) => return None,
                _ => *i += 1,
            }
        }
        return (*i <= b.len()).then_some(true).or(Some(false));
    } else if let Some(w) = ["true", "false", "null"].iter().find(|w| w.as_bytes()[0] == c) {
        let n = (b.len() - *i).min(w.len());
        *i += n;
        return (b[*i - n..*i] == w.as_bytes()[..n]).then_some(n == w.len());
    }
    let a = *i;
    while b.get(*i).is_some_and(|c| b"-+.eE0123456789".contains(c)) {
        *i += 1;
    }
    let t = std::str::from_utf8(&b[a..*i]).unwrap();
    let num = |t: &str| t.parse::<f64>().is_ok() && !t.starts_with('+');
    if int && t.contains(['.', 'e', 'E']) {
        None
    } else if *i == b.len() {
        (num(t) || num(&format!("{t}0"))).then_some(false)
    } else {
        num(t).then_some(true)
    }
}

// ---- copy drafts: what followed the last place of the longest of the
// last 12 or 3 tokens before, in the prompt or the answer

struct Cp(HashMap<u64, usize>); // an n-gram's hash: the end of its last place

impl Cp {
    // s grew by a token: the length of the match, what followed it
    fn next<'a>(&mut self, s: &'a [u32]) -> (usize, &'a [u32]) {
        let (mut r, l) = ((0, &[][..]), s.len());
        let f = |h: u64, &x: &u32| (h ^ x as u64).wrapping_mul(0x100000001b3);
        for n in [12, 3].into_iter().filter(|&n| n <= l) {
            let h = s[l - n..].iter().fold(n as u64, f);
            let e = self.0.insert(h, l).filter(|&e| s[e - n..e] == s[l - n..]);
            if let Some(e) = e.filter(|_| r.0 == 0) {
                let m = (1..=e.min(64)).take_while(|i| s[e - i] == s[l - i]);
                r = (m.count(), &s[e..]);
            }
        }
        r
    }
}

#[test]
fn copy_follows_the_longest_match() {
    // 10..20 1 2 3 4 1 2 3 9 10..20 1 2 3: the 12 tokens, not the last 3
    let a: Vec<u32> = (10..20).collect();
    let s = [&a[..], &[1, 2, 3, 4, 1, 2, 3, 9], &a, &[1, 2, 3]].concat();
    let mut c = Cp(HashMap::new());
    let r: Vec<_> = (1..=s.len()).map(|i| c.next(&s[..i])).collect();
    assert_eq!(r[16], (3, &s[13..17]));
    assert_eq!(r[s.len() - 1], (13, &s[13..]));
}

// ---- http

fn head(w: &mut impl Write, code: u16, kind: &str, more: &str) -> io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {code} OK\r\nContent-Type: {kind}\r\nConnection: close\r\n\
         Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n\
         {more}\r\n"
    )
}

fn reply(w: &mut impl Write, code: u16, j: &J) -> io::Result<()> {
    let b = j.to_string();
    head(w, code, "application/json", &format!("Content-Length: {}\r\n", b.len()))?;
    w.write_all(b.as_bytes())
}

fn sse(w: &mut impl Write, j: &J) -> io::Result<()> {
    write!(w, "data: {j}\n\n")?;
    w.flush()
}

fn err(m: &str) -> J {
    obj!["type" => "error",
        "error" => obj!["message" => m, "type" => "invalid_request_error"]]
}

// method, path, body, authorization
fn request(c: &TcpStream) -> Option<(String, String, Vec<u8>, String)> {
    let (mut r, mut l, mut n) = (BufReader::new(c), String::new(), 0);
    let mut auth = String::new();
    r.read_line(&mut l).ok()?;
    let mut it = l.split_whitespace().map(str::to_string);
    let mp = (it.next()?, it.next()?);
    loop {
        l.clear();
        r.read_line(&mut l).ok()?;
        let h = l.trim().to_ascii_lowercase();
        if h.is_empty() {
            break;
        } else if let Some(v) = h.strip_prefix("content-length:") {
            n = v.trim().parse().ok()?;
        } else if h.starts_with("x-api-key:") {
            auth = format!("bearer {}", l.trim()[10..].trim());
        } else if h.starts_with("authorization:") {
            // the scheme is case-insensitive, the key is not
            let (s, k) = l.trim()[14..].trim().split_once(' ').unwrap_or_default();
            auth = format!("{} {}", s.to_ascii_lowercase(), k.trim());
        } else if h.starts_with("expect:") {
            (&*c).write_all(b"HTTP/1.1 100 Continue\r\n\r\n").ok()?;
        }
    }
    let mut b = vec![0; n];
    r.read_exact(&mut b).ok()?;
    Some((mp.0, mp.1, b, auth))
}

// the whole utf-8 of b past *sent, less the last hold bytes
fn part(b: &[u8], sent: &mut usize, hold: usize) -> Option<String> {
    let e = b.len().saturating_sub(hold).max(*sent);
    let e = match std::str::from_utf8(&b[*sent..e]) {
        Err(x) if x.error_len().is_none() => *sent + x.valid_up_to(),
        _ => e,
    };
    let s = String::from_utf8_lossy(&b[*sent..e]).into_owned();
    *sent = e;
    (!s.is_empty()).then_some(s)
}

struct Srv {
    lm: Box<dyn Lm>,
    tok: Tok,
    fed: Vec<u32>, // the tokens in lm
    ctx: usize,
    effort: J,
    name: String,
    auth: Option<String>, // "bearer KEY", lowercase as request() reads it
    sp: Sp,
    t: [u32; 5], // model, thinking, text, tool json, end
}

impl Srv {
    // ant: r is an anthropic messages request, and so is the answer
    fn chat(&mut self, r: &J, w: &mut TcpStream, ant: bool)
            -> io::Result<()> {
        let a;
        let r = if ant { a = anth(r); &a } else { r };
        let (mut o, n, urls) = match prompt(r, &self.effort) {
            Ok(p) => p,
            Err(e) => return reply(w, 400, &err(&e)),
        };
        // response_format: the schema, and a line that tells the model
        let rf = &r["response_format"];
        let js = match rf["type"].s() {
            Some("json_schema") => Some(rf["json_schema"]["schema"].clone()),
            Some("json_object") => Some(obj!["type" => "object"]),
            _ => None,
        };
        if let Some(s) = &js {
            o += &format!("<|message_system|><|content_text|>Answer with one \
                JSON value and nothing else. It must fit this JSON schema: \
                {s}{END}");
        }
        // the rows of each image, as token ids, go in at its place. a full
        // image store empties itself; the next try starts again, and then
        // the oldest images are left out (an empty part) until the rest fit
        type R = Result<Vec<Range<u32>>, String>;
        let (t, mut skip, mut again) = (Instant::now(), 0, false);
        let ids = loop {
            let r = urls[skip..].iter().map(|u| self.lm.image(&u.1));
            match r.collect::<R>() {
                Ok(i) => break (0..skip).map(|_| 0..0).chain(i),
                Err(e) if e == FULL && skip < urls.len() => {
                    skip += again as usize;
                    again = true;
                }
                Err(e) => return reply(w, 400, &err(&e)),
            }
        };
        if !urls.is_empty() {
            eprintln!("gall: {} images in {:.2} s{}", urls.len(),
                t.elapsed().as_secs_f64(), match skip {
                    0 => String::new(),
                    k => format!(", the oldest {k} left out (store full)"),
                });
        }
        let ids: Vec<Range<u32>> = ids.collect();
        let keys: Vec<u32> = ids.iter().filter(|r| !r.is_empty())
            .map(|r| (r.start - crate::model::IMG) >> 12).collect();
        self.lm.keep(&keys);
        let (enc, mut p, mut at) = (|s: &str| self.tok.enc(s, true), vec![], 0);
        for ((to, _), ids) in urls.iter().zip(ids) {
            p.extend(enc(&o[at..*to]));
            p.extend(ids);
            at = *to;
        }
        p.extend(enc(&o[at..]));
        let sys = enc(&o[..n]);
        let sys = if p.starts_with(&sys) { sys.len() } else { 0 };
        if p.is_empty() || p.len() >= self.ctx {
            let e = format!("{} prompt tokens, context {}", p.len(), self.ctx);
            return reply(w, 400, &err(&e));
        }
        let num = |k: &str, d: f64| r[k].f().unwrap_or(d);
        let sp = Sp {
            temp: num("temperature", self.sp.temp),
            top_k: num("top_k", self.sp.top_k),
            top_p: num("top_p", self.sp.top_p),
            min_p: num("min_p", self.sp.min_p),
        };
        let max = r["max_completion_tokens"].f().or(r["max_tokens"].f());
        let max = max.filter(|&n| n >= 0.0).map_or(usize::MAX, |n| n as usize);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let mut rng = num("seed", now.as_nanos() as f64) as u64 | 1 << 40;
        let stops: Vec<&[u8]> = match &r["stop"] {
            J::Str(s) => vec![s.as_bytes()],
            a => a.a().iter().filter_map(J::s).map(str::as_bytes).collect(),
        };
        let stops: Vec<_> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let hold = stops.iter().map(|s| s.len()).max().unwrap_or(1) - 1;
        let stream = r["stream"].t();

        let id = if ant { "msg_" } else { "chatcmpl-" };
        let id = format!("{id}{}", now.as_nanos());
        let name = &self.name;
        let chunk = |delta: J, fin: J| {
            obj!["id" => id.as_str(), "object" => "chat.completion.chunk",
                "created" => now.as_secs() as usize, "model" => name.as_str(),
                "choices" => vec![obj!["index" => 0, "delta" => delta,
                    "finish_reason" => fin]]]
        };
        let mut e = Ev(0, None);
        let msg = |c: J, fin: J, sq: J, u: J| obj!["id" => id.as_str(),
            "type" => "message", "role" => "assistant", "model" => &**name,
            "content" => c, "stop_reason" => fin, "stop_sequence" => sq,
            "usage" => u];
        if stream {
            head(w, 200, "text/event-stream", "Cache-Control: no-cache\r\n")?;
            match ant {
                true => ev(w, "message_start", obj!["type" => "message_start",
                    "message" => msg(J::Arr(vec![]), J::Null, J::Null,
                        obj!["input_tokens" => p.len(),
                            "output_tokens" => 0])])?,
                _ => sse(w, &chunk(obj!["role" => "assistant"], J::Null))?,
            }
        }

        let t0 = Instant::now();
        let prev = self.fed.len(); // the state of the last chat, if in memory
        if self.lm.len() != self.fed.len() {
            self.lm.reset();
            self.fed.clear();
        }
        let same = self.fed.iter().zip(&p).take_while(|(a, b)| a == b).count();
        let mut k = same.min(p.len() - 1);
        if k < self.fed.len() && !self.lm.truncate(k) {
            // a client that drops the thinking parts from where the last
            // prompt ended: back to the checkpoint just before that end
            match self.lm.mark().filter(|&c| c <= k) {
                Some(c) if self.lm.truncate(c) => k = c,
                _ => {
                    self.lm.reset();
                    k = 0;
                }
            }
        }
        let (at, snap, of) = self.lm.prime(&p, k, sys);
        // hit: all of the last state (in memory, or the snapshot read) is a
        // prefix of p; partial: a shorter prefix is, the rest is fed (the
        // client changed earlier messages, or dropped thinking); miss: none
        let (mem, used) = (if snap > 0 { 0 } else { k }, k.max(snap));
        let kind = match () {
            _ if used == 0 => "miss",
            _ if (snap > 0 && snap == of) || (snap == 0 && mem == prev) => "hit",
            _ => "partial",
        };
        let src = match () {
            _ if snap > 0 => format!("snapshot of {of}"),
            _ if mem > 0 => "memory".to_string(),
            _ => "nothing".to_string(),
        };
        eprintln!("gall: cache {kind}: {used} of {} prompt tokens reused from {src}, {} fed",
            p.len(), p.len() - used);
        let k = used; // not fed for this request
        self.fed.clear();
        self.fed.extend(&p);
        self.lm.draft();
        self.lm.greedy(js.is_none() && sp.temp <= 0.0);
        let mut lg = self.lm.feed(&p[at..]);
        let t1 = Instant::now();

        let [model, think, text, tool, end] = self.t;
        let (mut st, mut n, mut sent, mut sq) = (0, 0, [0; 2], J::Null);
        let mut b = [vec![], vec![], vec![], vec![]]; // thinking, text, name, tool
        let mut calls = vec![];
        let mut cp = Cp(HashMap::new());
        (1..=p.len()).for_each(|i| _ = cp.next(&p[..i]));
        let fin = loop {
            if n == max || self.fed.len() >= self.ctx {
                break "length";
            }
            // the client hung up (a read of 0 bytes): stop
            w.set_nonblocking(true)?;
            let gone = matches!(w.peek(&mut [0]), Ok(0));
            w.set_nonblocking(false)?;
            if gone {
                return Err(io::ErrorKind::BrokenPipe.into());
            }
            // response_format: of the text tokens, take one that keeps the
            // text a fit; the message ends when the value is whole
            let fit = |x: &[u8], t: u32| {
                let (mut x, stop) = (x.to_vec(), t == end);
                let y = if stop { b" " as &[u8] } else { &self.tok.txt[t as usize] };
                // no space at the start, nor after a space
                let sp = |c: &u8| c.is_ascii_whitespace();
                if !stop && y.iter().all(sp) && x.last().is_none_or(sp) {
                    return false;
                }
                x.extend(y);
                let mut i = 0;
                match js.as_ref().and_then(|s| fits(&x, &mut i, s)) {
                    Some(d) => (d || !stop) && x[i..].trim_ascii().is_empty(),
                    None => false,
                }
            };
            let fit = |x: &[u8], t: u32| {
                fit(x, t) && (t == end || !self.tok.ctl[t as usize])
            };
            let mut t = sample(lg, &sp, &mut rng);
            if js.is_some() && st == text && !fit(&b[1], t) {
                // the likeliest token that fits
                let mut ix: Vec<u32> = (0..lg.len() as u32).collect();
                ix.sort_unstable_by(|a, b| lg[*b as usize].total_cmp(&lg[*a as usize]));
                t = ix.into_iter().find(|&t| fit(&b[1], t)).unwrap_or(end);
            }
            n += 1;
            if t == self.tok.eos {
                break "stop";
            }
            if t == end && st == tool {
                let j = J::parse(&b[3]).unwrap_or(J::Null);
                let s = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
                let (name, raw) = (s(&b[2]), s(&b[3]));
                let args = &j["args"];
                calls.push(obj!["index" => calls.len(),
                    "id" => format!("call_{}{}", now.as_micros(), calls.len()),
                    "type" => "function",
                    "function" => obj![
                        "name" => j["name"].s().unwrap_or(name.trim()),
                        "arguments" => if *args == NULL { raw }
                            else { args.to_string() }]]);
                b[3].clear();
                let c = &calls[calls.len() - 1];
                if stream && ant {
                    let a = c["function"]["arguments"].clone();
                    let mut u = tool_use(c);
                    let J::Obj(kv) = &mut u else { unreachable!() };
                    kv[3].1 = obj![];
                    e.delta(w, u, obj!["type" => "input_json_delta",
                        "partial_json" => a])?;
                } else if stream {
                    let c = obj!["tool_calls" => vec![c.clone()]];
                    sse(w, &chunk(c, J::Null))?;
                }
            }
            if self.tok.ctl[t as usize] {
                st = if [model, think, text, tool].contains(&t) { t } else { 0 };
                if t == model {
                    b[2].clear();
                }
            } else {
                let i = [think, text, model, tool].iter().position(|&x| x == st);
                b[i.unwrap_or(1)].extend(&self.tok.txt[t as usize]);
            }
            let find = |s: &&[u8]| b[1][sent[1]..].windows(s.len()).position(|w| w == *s);
            let hit = stops.iter().filter_map(|s| Some((find(s)?, *s)));
            let hit = hit.min();
            if let Some((i, s)) = hit {
                b[1].truncate(sent[1] + i);
                sq = String::from_utf8_lossy(s).into_owned().into();
                break "stop";
            }
            for (i, key) in ["reasoning_content", "content"].iter().enumerate() {
                if !stream {
                    break;
                }
                if let Some(s) = part(&b[i], &mut sent[i], hold * i) {
                    match ant {
                        true => e.delta(w, blk(i, "").0, blk(i, &s).1)?,
                        _ => sse(w, &chunk(obj![key => s], J::Null))?,
                    }
                }
            }
            self.fed.push(t);
            // a copy of 12 or more tokens goes ahead, as far as the schema
            // takes it
            let (m, d) = cp.next(&self.fed);
            let cut = js.is_some() && st == text;
            let (mut x, p) = (b[1].clone(), if m < 24 { 0.9 } else { 0.99 });
            let d = d.iter().take(if m < 12 { 0 } else { 7 });
            let d = d.take_while(|&&t| !cut || fit(&x, t) && {
                x.extend(&self.tok.txt[t as usize]);
                true
            });
            self.lm.copy(d.map(|&t| (t, p)).collect());
            lg = self.lm.step(t);
        };
        self.lm.greedy(false);
        let fin = if calls.is_empty() { fin } else { "tool_calls" };
        TC.store(fin == "tool_calls", Relaxed);

        let (dp, dg) = (t1 - t0, t1.elapsed());
        let (np, ms) = (p.len() - k, |d: Duration| d.as_secs_f64() * 1e3);
        let rate = |n: usize, d: Duration| n as f64 / d.as_secs_f64().max(1e-9);
        eprintln!(
            "gall: prompt {} ({k} cached, snap {snap}) {:.1} t/s, first token \
             {:.2} s, predicted {n} {:.1} t/s{}",
            p.len(),
            rate(np, dp),
            dp.as_secs_f64(),
            rate(n, dg),
            self.lm.draft()
        );
        let usage = obj!["prompt_tokens" => p.len(), "completion_tokens" => n,
            "total_tokens" => p.len() + n];
        let timings = obj!["cache_n" => k, "prompt_n" => np,
            "prompt_ms" => ms(dp), "prompt_per_second" => rate(np, dp),
            "predicted_n" => n, "predicted_ms" => ms(dg),
            "predicted_per_second" => rate(n, dg)];
        *LAST.lock().unwrap() = timings.clone();
        let why = match fin {
            "length" => "max_tokens",
            "tool_calls" => "tool_use",
            _ if sq != NULL => "stop_sequence",
            _ => "end_turn",
        };
        let au = || obj!["input_tokens" => p.len(), "output_tokens" => n];
        if stream && ant {
            for i in 0..2 {
                if let Some(s) = part(&b[i], &mut sent[i], 0) {
                    e.delta(w, blk(i, "").0, blk(i, &s).1)?;
                }
            }
            e.to(w, J::Null)?;
            ev(w, "message_delta", obj!["type" => "message_delta",
                "delta" => obj!["stop_reason" => why, "stop_sequence" => sq],
                "usage" => au()])?;
            return ev(w, "message_stop", obj!["type" => "message_stop"]);
        }
        let s = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
        if ant {
            let mut c = vec![];
            for i in 0..2 {
                if !b[i].is_empty() {
                    c.push(blk(i, &s(&b[i])).0);
                }
            }
            c.extend(calls.iter().map(tool_use));
            return reply(w, 200, &msg(c.into(), why.into(), sq, au()));
        }
        if stream {
            for (i, key) in ["reasoning_content", "content"].iter().enumerate() {
                if let Some(s) = part(&b[i], &mut sent[i], 0) {
                    sse(w, &chunk(obj![key => s], J::Null))?;
                }
            }
            let J::Obj(mut c) = chunk(obj![], fin.into()) else { unreachable!() };
            c.push(("timings".into(), timings));
            sse(w, &J::Obj(c))?;
            if r["stream_options"]["include_usage"].t() {
                let J::Obj(mut c) = chunk(obj![], J::Null) else { unreachable!() };
                c[4].1 = vec![].into();
                c.push(("usage".into(), usage));
                sse(w, &J::Obj(c))?;
            }
            return w.write_all(b"data: [DONE]\n\n");
        }
        let mut m = vec![("role".to_string(), "assistant".into())];
        m.push(("content".into(), s(&b[1]).into()));
        if !b[0].is_empty() {
            m.push(("reasoning_content".into(), s(&b[0]).into()));
        }
        if !calls.is_empty() {
            m.push(("tool_calls".into(), calls.into()));
        }
        let c = obj!["index" => 0, "message" => J::Obj(m), "finish_reason" => fin];
        reply(w, 200, &obj!["id" => id.as_str(), "object" => "chat.completion",
            "created" => now.as_secs() as usize, "model" => name.as_str(),
            "choices" => vec![c], "usage" => usage, "timings" => timings])
    }
}

// a connection, on its own thread: answers all but a chat, which waits its
// turn on the queue q for the one model, as does a park (then an exit if
// the bool is set; an answer to the stream if there is one)
enum Q {
    Chat(J, TcpStream, bool), // anthropic
    Park(Option<TcpStream>, bool),
}
// loaded, a chat running, a park queued, the timings of the last chat
static UP: AtomicBool = AtomicBool::new(false);
static BUSY: AtomicBool = AtomicBool::new(false);
static PEND: AtomicBool = AtomicBool::new(false);
static TC: AtomicBool = AtomicBool::new(false); // the last finish: tool_calls
static LAST: Mutex<J> = Mutex::new(J::Null);
fn conn(mut c: TcpStream, name: &str, auth: &Option<String>, q: &Sender<Q>)
        -> io::Result<()> {
    c.set_nodelay(true)?;
    c.set_read_timeout(Some(Duration::from_secs(30)))?;
    let Some((m, p, body, a)) = request(&c) else { return Ok(()) };
    let p = p.trim_start_matches("/v1");
    match (m.as_str(), p) {
        ("OPTIONS", _) => head(&mut c, 204, "text/plain", ""),
        ("GET", "/health") => reply(&mut c, 200, &obj!["status" => "ok",
            "loaded" => UP.load(Relaxed), "busy" => BUSY.load(Relaxed),
            "last" => LAST.lock().unwrap().clone()]),
        _ if auth.as_ref().is_some_and(|k| *k != a) => {
            reply(&mut c, 401, &err("invalid api key"))
        }
        ("GET", "/models") => {
            let m = obj!["id" => name, "object" => "model", "owned_by" => "gall"];
            reply(&mut c, 200, &obj!["object" => "list", "data" => vec![m]])
        }
        ("POST", "/chat/completions" | "/messages") => match J::parse(&body) {
            Some(r) => Ok(q.send(Q::Chat(r, c, p == "/messages")).unwrap()),
            None => reply(&mut c, 400, &err("the body is not json")),
        },
        ("POST", "/park") => Ok(q.send(Q::Park(Some(c), false)).unwrap()),
        _ => reply(&mut c, 404, &err("not found")),
    }
}

// answers every prompt with the same line: for tests without a model
struct Dummy(Vec<u32>, usize, Vec<f32>, usize);

impl Lm for Dummy {
    fn reset(&mut self) {
        self.1 = 0;
    }
    fn step(&mut self, t: u32) -> &[f32] {
        self.1 += 1;
        let hit = t == self.0[self.3];
        self.3 = if hit { (self.3 + 1) % self.0.len() } else { 0 };
        self.2.fill(0.0);
        self.2[self.0[self.3] as usize] = 30.0;
        &self.2
    }
    fn len(&self) -> usize {
        self.1
    }
    fn truncate(&mut self, n: usize) -> bool {
        self.1 = n;
        true
    }
}

const FULL: &str = "the image store is full";

// the engine as an Lm; a checkpoint at each feed (the prompt). the
// vision encoder, and the row count of each image held, by its url's hash;
// the drafter, its estimates and the counts at the last report
type Draft = (Dr, Spec, [usize; 3]);
struct Ink(crate::model::Model, Vec<f32>, crate::model::snap::Snaps,
    Option<Vis>, HashMap<u64, usize>, Option<Draft>);

impl Lm for Ink {
    fn reset(&mut self) {
        self.draft();
        self.0.reset()
    }
    // a row verified ahead answers without a pass (Dr::next)
    fn step(&mut self, t: u32) -> &[f32] {
        let r = match &mut self.5 {
            Some((d, s, _)) => d.next(&mut self.0, s, t),
            None => {
                self.0.step(t);
                0
            }
        };
        self.1 = self.0.logits_row(r);
        &self.1
    }
    fn len(&self) -> usize {
        self.0.pos
    }
    fn truncate(&mut self, n: usize) -> bool {
        self.draft();
        self.0.truncate(n)
    }
    fn mark(&self) -> Option<usize> {
        self.0.mark()
    }
    // the checkpoint is before the last token: the same prompt again
    // (regenerate) goes back to it
    fn feed(&mut self, toks: &[u32]) -> &[f32] {
        let (l, h) = toks.split_last().unwrap();
        match &self.5 {
            _ if h.is_empty() => 0,
            Some((d, ..)) => d.feed(&mut self.0, h),
            None => self.0.feed(h),
        };
        self.0.save();
        self.step(*l)
    }
    // what prime restores or feeds has no drafter context
    fn prime(&mut self, p: &[u32], k: usize, sys: usize) -> (usize, usize, usize) {
        let r = self.2.prime(&mut self.0, p, k, sys);
        if let Some((d, ..)) = self.5.as_ref().filter(|_| r.0 > k) {
            d.forget(r.0);
        }
        (r.0, r.1, self.2.of)
    }
    fn keep(&mut self, keys: &[u32]) {
        let id = |k: u64| k as u32 & 0x7ffff;
        self.4.retain(|&k, _| keys.contains(&id(k)));
        self.0.pend.retain(|k, _| keys.contains(k));
    }
    // also drops the rows verified ahead: the model is about to move
    fn draft(&mut self) -> String {
        let Some((_, s, c)) = &mut self.5 else { return String::new() };
        (s.drop(), s.cp.clear());
        let n = [s.drafts, s.rows[0], s.rows[1]];
        let d = [0, 1, 2].map(|i| n[i] - c[i]);
        *c = n;
        format!(", {} drafts, {} of {} rows ahead kept", d[0], d[2], d[1])
    }
    fn copy(&mut self, d: Vec<(u32, f32)>) {
        self.5.as_mut().map(|x| x.1.cp = d);
    }
    // without drafts the next token runs ahead (Model.greedy); every
    // other use of the gpu waits for it (Model::idle)
    fn greedy(&mut self, on: bool) {
        self.0.greedy = on && self.5.is_none();
    }
    fn image(&mut self, url: &str) -> Result<Range<u32>, String> {
        let v = self.3.as_ref().ok_or("no mmproj file beside the model")?;
        let mut h = std::hash::DefaultHasher::new();
        std::hash::Hash::hash(url, &mut h);
        let key = std::hash::Hasher::finish(&h);
        if let Some(&n) = self.4.get(&key) {
            return Ok(Model::ids(key as u32, n));
        }
        // rows are kept on disk too (the 32 newest), for after an unload
        let dir = crate::up::cache().join("img");
        let f = dir.join(format!("{key:016x}.{}", v.max));
        let e = crate::f32s(f.to_str().unwrap());
        let e = if !e.is_empty() { e } else {
            let b = match url.split_once(";base64,") {
                Some((_, b)) if url.starts_with("data:") => b64(b),
                _ if url.starts_with("http") => {
                    let c = std::process::Command::new("curl")
                        .args(["-sfL", "--max-time", "30", "--", url]).output();
                    c.map(|o| o.stdout).unwrap_or_default()
                }
                _ => vec![],
            };
            let (w, h, rgb) = crate::vis::decode(&b, v.max)
                .ok_or("the image could not be read")?;
            let e = v.embed(w, h, &rgb);
            let b: Vec<u8> = e.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::create_dir_all(&dir).ok();
            std::fs::write(&f, b).ok();
            let age = |p: &std::path::PathBuf| p.metadata().and_then(|m| m.modified()).ok();
            let mut old: Vec<_> = std::fs::read_dir(&dir).into_iter().flatten()
                .flatten().map(|d| d.path()).collect();
            old.sort_by_key(|p| std::cmp::Reverse(age(p)));
            old.iter().skip(32).for_each(|p| { std::fs::remove_file(p).ok(); });
            e
        };
        // two images with one 19-bit key: forget what was built on the first
        let (id, m) = (|k: u64| k as u32 & 0x7ffff, &mut self.0);
        if self.4.keys().any(|&k| id(k) == id(key)) {
            (m.ni, self.4) = (0, HashMap::new());
            (m.ib, m.pend) = (HashMap::new(), HashMap::new());
            m.reset();
            return Err(FULL.into());
        }
        // the store needs room only for the images of one pass (Model::stage)
        if !m.image(key as u32, &e) {
            return Err("the image is larger than the image store".into());
        }
        self.4.insert(key, e.len() / 2304);
        Ok(Model::ids(key as u32, e.len() / 2304))
    }
    fn park(&mut self, toks: &[u32]) {
        let t = Instant::now();
        if toks.len() < 512 {
            return;
        } else if let Err(e) = self.2.write(&self.0, toks) {
            return eprintln!("gall: snap {}: {e}", toks.len());
        }
        let s = t.elapsed().as_secs_f64();
        eprintln!("gall: snap {} written in {s:.2} s", toks.len());
    }
}

pub fn main(a: &[String]) {
    let opt = |k: &str, d: &str| {
        let i = a.iter().position(|x| x == k);
        let v = i.and_then(|i| a.get(i + 1)).filter(|v| !v.starts_with("--"));
        v.map_or(d, |s| s).to_string()
    };
    let g = crate::gguf::open(&a[0]);
    let tok = Tok::new(&g);
    if a.get(1).is_some_and(|x| x == "--prompt") {
        // gall serve <gguf> --prompt <request.json>: the rendered prompt
        let r = J::parse(&std::fs::read(&a[2]).unwrap()).unwrap();
        return print!("{}", prompt(&r, &"none".into()).unwrap().0);
    }
    // SIGTERM, blocked in every thread, is a park then an exit
    let (tx, rx) = channel();
    let (t, mut m) = (tx.clone(), 0);
    unsafe {
        libc::sigaddset(&mut m, libc::SIGTERM);
        libc::pthread_sigmask(libc::SIG_BLOCK, &m, 0 as _);
    }
    std::thread::spawn(move || {
        unsafe { libc::sigwait(&m, &mut 0) };
        t.send(Q::Park(None, true)).unwrap()
    });
    let ctx: usize = opt("--ctx", "102400").parse().unwrap();
    let say = "<|message_model|><|content_text|>hello from gall\
               <|end_message|><|content_model_end_sampling|>";
    let load: Load = if a.iter().any(|x| x == "--dummy") {
        let (say, n) = (tok.enc(&opt("--dummy", say), true), tok.txt.len());
        Box::new(move || Box::new(Dummy(say.clone(), 0, vec![0.0; n], 0)))
    } else {
        let t = tx.clone();
        crate::up::guard(move || {
            if UP.load(Relaxed) && !PEND.swap(true, Relaxed) {
                t.send(Q::Park(None, false)).unwrap()
            }
        });
        let (p, draft) = (a[0].clone(), !a.iter().any(|x| x == "--no-draft"));
        let draft = draft && crate::dr::usable(&p);
        Box::new(move || {
            crate::up::lock(true);
            let mut m = crate::model::Model::open(&p, ctx);
            let s = crate::model::snap::Snaps::new(&p, &m);
            let v = std::path::Path::new(&p).with_file_name(crate::up::MMPROJ);
            let v = v.exists().then(|| Vis::open(v.to_str().unwrap()));
            let d = std::path::Path::new(&p).with_file_name(crate::up::DRAFT);
            let d = (d.exists() && draft).then(|| {
                (Dr::open(&mut m, d.to_str().unwrap(), ctx), Spec::new(), [0; 3])
            });
            Box::new(Ink(m, vec![], s, v, HashMap::new(), d))
        })
    };
    let lm = Box::new(Lazy(None, load));
    let kv = |k: &str, d: f64| {
        let k = format!("general.sampling.{k}");
        if g.kv.contains_key(k.as_str()) { g.f(&k) } else { d }
    };
    let sp = Sp {
        temp: kv("temp", 1.0),
        top_k: kv("top_k", 0.0),
        top_p: kv("top_p", 1.0),
        min_p: kv("min_p", 0.0),
    };
    let t = ["message_model", "content_thinking", "content_text",
        "content_invoke_tool_json", "end_message"]
        .map(|s| tok.id(&format!("<|{s}|>")));
    let port: u16 = opt("--port", "9940").parse().unwrap();
    let host = opt("--host", "127.0.0.1");
    let l = TcpListener::bind((host.as_str(), port)).unwrap();
    let mut s = Srv {
        lm,
        tok,
        fed: vec![],
        ctx,
        effort: opt("--effort", "none").into(),
        name: opt("--alias", "inkling-2-mini"),
        auth: a.iter().position(|x| x == "--api-key-file").map(|_| {
            let k = std::fs::read_to_string(opt("--api-key-file", "")).unwrap();
            assert!(!k.trim().is_empty(), "gall: the key file is empty");
            format!("bearer {}", k.trim())
        }),
        sp,
        t,
    };
    if s.auth.is_none() && !l.local_addr().unwrap().ip().is_loopback() {
        eprintln!("gall: --host {host} needs --api-key-file");
        std::process::exit(1)
    }
    eprintln!("gall: http://{host}:{port}/v1, context {ctx}");
    let idle = Duration::from_secs(opt("--idle-unload", "0").parse().unwrap());
    let (cap, mut t) = (Duration::from_secs(30), Instant::now());
    let mut gaps = vec![];
    let (name, auth) = (s.name.clone(), s.auth.clone());
    std::thread::spawn(move || for c in l.incoming().flatten() {
        let (name, auth, tx) = (name.clone(), auth.clone(), tx.clone());
        std::thread::spawn(move || conn(c, &name, &auth, &tx));
    });
    loop {
        UP.store(s.lm.loaded(), Relaxed);
        // a client back from its tool: hold for 4 x the median of the last 5
        // idle gaps after tool_calls, 30 s at most (and with none yet)
        let mut g = gaps[gaps.len().saturating_sub(5)..].to_vec();
        g.sort();
        let m = g.get(g.len() / 2).map_or(cap, |&m: &Duration| cap.min(m * 4));
        let w = if TC.load(Relaxed) { idle.max(m) } else { idle };
        let r = match !idle.is_zero() && s.lm.loaded() {
            true => rx.recv_timeout(w).ok(),
            _ => rx.recv().ok(),
        };
        match r.unwrap_or(Q::Park(None, false)) {
            Q::Chat(r, mut c, ant) => {
                if TC.swap(false, Relaxed) {
                    gaps.push(t.elapsed());
                }
                BUSY.store(true, Relaxed);
                let e = objc2::rc::autoreleasepool(|_| s.chat(&r, &mut c, ant));
                if let Err(e) = e {
                    eprintln!("gall: {e}");
                }
                BUSY.store(false, Relaxed);
                t = Instant::now();
            }
            Q::Park(c, exit) => {
                let n = if s.lm.len() == s.fed.len() { s.fed.len() } else { 0 };
                s.lm.park(&s.fed[..n]);
                s.fed.clear();
                UP.store(false, Relaxed);
                PEND.store(false, Relaxed);
                if exit {
                    std::process::exit(0)
                }
                c.map(|mut c| reply(&mut c, 200, &obj!["parked" => true]));
            }
        }
    }
}

#[test]
fn anthropic_requests_render_as_chat_completions() {
    let p = |s: &str| prompt(&J::parse(s.as_bytes()).unwrap(), &NULL).unwrap();
    let a = r#"{"system":[{"type":"text","text":"s"}],"thinking":{"type":
        "disabled"},"tools":[{"name":"sh","description":"d","input_schema":
        {"type":"object"}},{"type":"web_search_20250305","name":"w"}],
        "messages":[{"role":"user","content":"hi"},{"role":"assistant",
        "content":[{"type":"thinking","thinking":"t"},{"type":"tool_use",
        "id":"t1","name":"sh","input":{"c":"ls"}}]},{"role":"user","content":
        [{"type":"tool_result","tool_use_id":"t1","content":"a"},
        {"type":"text","text":"go"}]}]}"#;
    let o = r#"{"reasoning_effort":"none","tools":[{"type":"function",
        "function":{"name":"sh","description":"d","parameters":{"type":
        "object"}}}],"messages":[{"role":"system","content":"s"},{"role":
        "user","content":"hi"},{"role":"assistant","content":"",
        "reasoning_content":"t","tool_calls":[{"id":"t1","function":{"name":
        "sh","arguments":"{\"c\":\"ls\"}"}}]},{"role":"tool","tool_call_id":
        "t1","content":"a"},{"role":"user","content":"go"}]}"#;
    let a = anth(&J::parse(a.as_bytes()).unwrap()).to_string();
    assert_eq!(p(&a), p(o));
}
