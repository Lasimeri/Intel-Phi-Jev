//! ARTICHOKE: the in-process interrogation engine. It never lets the model
//! answer; it reads the involuntary response (the next-token distribution
//! after the answer cue) for every fingerprint of a session at once.
//!
//! The session (the shared prefix of every rendered prompt) is prefilled
//! once on sequence 0. Each fingerprint gets its own sequence, copied from
//! sequence 0 (`llama_memory_seq_cp`, which on a hybrid model copies the
//! recurrent state as well as sharing the attention cells), and all the
//! fingerprints' suffixes go through the model in one batch. Sequence 0 is
//! kept between requests, so a session asked again costs only its suffixes.
//! See mod.md.

mod sys;

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, Once};
use std::time::Instant;

use crate::backend::{BackendError, ScoreCost, Scored, Scorer};
use crate::prompt::Segs;

/// How to open the engine.
#[derive(Debug, Clone)]
pub struct Options {
    /// The GGUF file.
    pub gguf: PathBuf,
    /// Cells of the unified cache: the session plus every suffix of one
    /// round of forks must fit.
    pub n_ctx: u32,
    /// Fingerprints evaluated together (sequences 1..=forks).
    pub forks: u32,
    /// Tokens handed to one `llama_decode`.
    pub n_batch: u32,
    /// Tokens in one physical step of a decode.
    pub n_ubatch: u32,
    /// Threads for llama.cpp's own work (not the cards' daemons).
    pub threads: i32,
    /// Where the CPU variants are loaded from; the build's own by default.
    pub backend_dir: Option<PathBuf>,
    /// Show llama.cpp's informational log, not only its warnings.
    pub verbose: bool,
    /// Let llama.cpp repack weights for this host's CPU. Off by default:
    /// a repacked copy sits beside the mapped file, which for a subject
    /// near this host's memory (the 21.7 GB 35B on 31 GB) thrashed a run
    /// from 44 to 1.5 tokens a second; and with the Phi payload loaded a
    /// repacked weight lives in a CPU-only buffer the payload is never
    /// offered, so it could never reach a card.
    pub repack: bool,
    /// Let llama.cpp use flash attention where it would (off on the avx512
    /// site: its tiled loop is one the card cannot yet run, site.md).
    pub flash_attn: bool,
    /// Map the subject without `MAP_POPULATE`, its pages read in as they
    /// are used (`LAZY_PAGES`). For the offloaded sites: the rows the cards
    /// keep are then never all resident at once, and pages no request
    /// touches are never read in. See mod.md, "Pages read in as used".
    pub lazy_pages: bool,
}

/// While set, a file mapping made in this process loses `MAP_POPULATE`:
/// the `mmap` the xks binary exports ahead of libc's (main.rs) reads it.
/// Set only while the subject loads (`Options::lazy_pages`), so nothing
/// else this process maps is changed.
pub static LAZY_PAGES: AtomicBool = AtomicBool::new(false);

/// Clears `LAZY_PAGES` however the load it was set for ends.
struct Lazy;

impl Drop for Lazy {
    fn drop(&mut self) {
        LAZY_PAGES.store(false, Ordering::SeqCst);
    }
}

impl Options {
    /// Defaults for a file: 12 threads when the Phi backend is loaded (the
    /// card daemons need the rest; more costs 5x at one token), else 16.
    pub fn new(gguf: impl Into<PathBuf>) -> Self {
        let cards = std::env::var_os("GGML_BACKEND_PATH").is_some();
        Self {
            gguf: gguf.into(),
            n_ctx: 16384,
            forks: 15,
            n_batch: 2048,
            n_ubatch: 512,
            threads: if cards { 12 } else { 16 },
            backend_dir: None,
            verbose: false,
            repack: false,
            flash_attn: true,
            lazy_pages: false,
        }
    }
}

static INIT: Once = Once::new();
static VERBOSE: AtomicBool = AtomicBool::new(false);
static LAST_LEVEL: AtomicU32 = AtomicU32::new(0);

unsafe extern "C" fn log_cb(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    let level = if level == sys::ggml_log_level_GGML_LOG_LEVEL_CONT {
        LAST_LEVEL.load(Ordering::Relaxed)
    } else {
        LAST_LEVEL.store(level, Ordering::Relaxed);
        level
    };
    if level >= sys::ggml_log_level_GGML_LOG_LEVEL_WARN || VERBOSE.load(Ordering::Relaxed) {
        if text.is_null() {
            return;
        }
        // SAFETY: llama.cpp passes a NUL-terminated string valid for the call.
        let s = unsafe { CStr::from_ptr(text) };
        eprint!("{}", s.to_string_lossy());
    }
}

/// Register the backends. A build with dynamic backends loads the best
/// CPU variant from `dir` and then `GGML_BACKEND_PATH` (the payload, when
/// the site installed it). A build without them (the AVX-512 one) has its
/// CPU backend linked in, so only the payload is loaded.
///
/// # Safety
/// Calls into ggml's registry; once, before any model is loaded.
unsafe fn load_backends(dir: &CStr) {
    #[cfg(not(xks_static_cpu))]
    unsafe {
        sys::ggml_backend_load_all_from_path(dir.as_ptr());
    }
    #[cfg(xks_static_cpu)]
    {
        let _ = dir;
        if let Some(p) = std::env::var_os("GGML_BACKEND_PATH") {
            if let Ok(p) = CString::new(p.as_bytes()) {
                unsafe { sys::ggml_backend_load(p.as_ptr()) };
            }
        }
    }
}

/// State that only one request may touch at a time.
struct Inner {
    ctx: *mut sys::llama_context,
    /// Tokens sequence 0 holds (the last session prefilled).
    cached: Vec<sys::llama_token>,
    /// Single-token candidate labels already looked up.
    labels: HashMap<String, Vec<sys::llama_token>>,
}

pub struct Artichoke {
    model: *mut sys::llama_model,
    vocab: *const sys::llama_vocab,
    n_vocab: usize,
    inner: Mutex<Inner>,
    name: String,
    forks: usize,
    n_ctx: usize,
    n_batch: usize,
    /// The model keeps recurrent state (pure or hybrid).
    pub recurrent: bool,
}

// SAFETY: the context is only touched under `inner`'s lock; the model and
// vocabulary are read-only after load and llama.cpp allows sharing them.
unsafe impl Send for Artichoke {}
unsafe impl Sync for Artichoke {}

/// One label read out of a decode.
struct Reading {
    logprobs: Vec<f64>,
}

impl Artichoke {
    pub fn open(opts: &Options) -> Result<Self, String> {
        VERBOSE.store(opts.verbose, Ordering::Relaxed);
        let dir = opts
            .backend_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(env!("XKS_LLAMA_BUILD_DIR")));
        let dir_c = CString::new(dir.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
        INIT.call_once(|| unsafe {
            sys::llama_log_set(Some(log_cb), ptr::null_mut());
            sys::llama_backend_init();
            load_backends(&dir_c);
            let n = sys::ggml_backend_dev_count();
            let names: Vec<String> = (0..n)
                .map(|i| {
                    let d = sys::ggml_backend_dev_get(i);
                    CStr::from_ptr(sys::ggml_backend_dev_name(d))
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            eprintln!("xks: devices: {}", names.join(", "));
        });
        let path = CString::new(opts.gguf.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
        // The long step, said before it starts: a detached server's log
        // (which Mechanical Jev's TUI shows as a start's progress) is
        // otherwise silent from the devices line to the listening line.
        let size = std::fs::metadata(&opts.gguf).map_or(0, |m| m.len());
        eprintln!(
            "xks: loading {} ({:.1} GB)",
            opts.gguf.file_name().map_or_else(
                || opts.gguf.display().to_string(),
                |n| n.to_string_lossy().into_owned()
            ),
            size as f64 / 1e9
        );
        let lazy = opts.lazy_pages.then(|| {
            LAZY_PAGES.store(true, Ordering::SeqCst);
            Lazy
        });
        let model = unsafe {
            let mut mp = sys::llama_model_default_params();
            mp.use_extra_bufts = opts.repack;
            sys::llama_model_load_from_file(path.as_ptr(), mp)
        };
        drop(lazy);
        if model.is_null() {
            return Err(format!("could not load {}", opts.gguf.display()));
        }
        let ctx = unsafe {
            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = opts.n_ctx;
            cp.n_batch = opts.n_batch;
            cp.n_ubatch = opts.n_ubatch.min(opts.n_batch);
            cp.n_seq_max = opts.forks + 1;
            // At most one output per fork in any step (the last token of
            // each suffix); the default sizes the output buffer for n_batch
            // rows of the whole vocabulary.
            cp.n_outputs_max = opts.forks + 1;
            cp.n_threads = opts.threads;
            cp.n_threads_batch = opts.threads;
            // Forks share the session's attention cells instead of copying
            // them into per-sequence streams.
            cp.kv_unified = true;
            cp.no_perf = true;
            if !opts.flash_attn {
                cp.flash_attn_type = sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
            }
            sys::llama_init_from_model(model, cp)
        };
        if ctx.is_null() {
            unsafe { sys::llama_model_free(model) };
            return Err("could not create a context (memory for the forks?)".into());
        }
        let (vocab, n_vocab, recurrent) = unsafe {
            let vocab = sys::llama_model_get_vocab(model);
            (
                vocab,
                sys::llama_vocab_n_tokens(vocab) as usize,
                sys::llama_model_is_recurrent(model) || sys::llama_model_is_hybrid(model),
            )
        };
        let name = format!(
            "artichoke/{}",
            opts.gguf
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        );
        Ok(Self {
            model,
            vocab,
            n_vocab,
            inner: Mutex::new(Inner {
                ctx,
                cached: Vec::new(),
                labels: HashMap::new(),
            }),
            name,
            forks: opts.forks as usize,
            n_ctx: opts.n_ctx as usize,
            n_batch: opts.n_batch as usize,
            recurrent,
        })
    }

    /// Plain text: no start token, no special tokens.
    fn tokenize_plain(&self, text: &str) -> Result<Vec<sys::llama_token>, BackendError> {
        self.tokenize_with(text, false, false)
    }

    /// A rendered prompt, segment by segment: template text with special
    /// tokens parsed, user text without, the model's start token before the
    /// first. Nothing a caller sent can become a special token, and the
    /// caller's text is tokenized as it was sent.
    fn tokenize_segs(&self, segs: &Segs) -> Result<Vec<sys::llama_token>, BackendError> {
        let mut out = Vec::new();
        for (i, s) in segs.0.iter().enumerate() {
            out.extend(self.tokenize_with(&s.text, i == 0, !s.user)?);
        }
        // A template that writes its BOS itself (Llama 3's <|begin_of_text|>)
        // after the tokenizer has added one: keep one.
        let bos = unsafe { sys::llama_vocab_bos(self.vocab) };
        if out.len() >= 2 && out[0] == bos && out[1] == bos {
            out.remove(0);
        }
        Ok(out)
    }

    fn tokenize_with(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<sys::llama_token>, BackendError> {
        let mut buf: Vec<sys::llama_token> = vec![0; text.len() + 8];
        let mut n = unsafe {
            sys::llama_tokenize(
                self.vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
                add_special,
                parse_special,
            )
        };
        if n < 0 {
            buf.resize((-n) as usize, 0);
            n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    add_special,
                    parse_special,
                )
            };
        }
        if n < 0 {
            return Err(BackendError::Malformed("tokenizer failed".into()));
        }
        buf.truncate(n as usize);
        Ok(buf)
    }

    /// A label's tokens as it follows the answer cue (no start token, no
    /// special-token parsing), cached.
    fn label_tokens(
        &self,
        inner: &mut Inner,
        label: &str,
    ) -> Result<Vec<sys::llama_token>, BackendError> {
        if let Some(t) = inner.labels.get(label) {
            return Ok(t.clone());
        }
        let toks = self.tokenize_plain(label)?;
        if toks.is_empty() {
            return Err(BackendError::Rejected(format!(
                "label {label:?} has no tokens"
            )));
        }
        inner.labels.insert(label.to_string(), toks.clone());
        Ok(toks)
    }

    /// A one-token label, for the polygraph's single-sequence readings.
    fn label_id(&self, inner: &mut Inner, label: &str) -> Result<sys::llama_token, BackendError> {
        let t = self.label_tokens(inner, label)?;
        if t.len() != 1 {
            return Err(BackendError::Rejected(format!(
                "label {label:?} is {} tokens; the polygraph reads one-token labels",
                t.len()
            )));
        }
        Ok(t[0])
    }

    /// Full-vocabulary log-probabilities of `ids` from the logits of output
    /// row `i` of the last decode.
    fn read(&self, ctx: *mut sys::llama_context, i: i32, ids: &[sys::llama_token]) -> Reading {
        // SAFETY: row `i` was requested as an output of the last decode;
        // llama.cpp returns n_vocab floats valid until the next decode.
        let row =
            unsafe { std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, i), self.n_vocab) };
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max
            + row
                .iter()
                .map(|&x| (x as f64 - max).exp())
                .sum::<f64>()
                .ln();
        Reading {
            logprobs: ids.iter().map(|&t| row[t as usize] as f64 - lse).collect(),
        }
    }

    /// Decode `toks` (token, position, sequence, output slot) in windows of
    /// n_batch, calling `out` with (slot, batch row) for every output.
    fn decode(
        &self,
        ctx: *mut sys::llama_context,
        toks: &[(sys::llama_token, i32, i32, Option<usize>)],
        mut out: impl FnMut(usize, i32),
    ) -> Result<(), BackendError> {
        for window in toks.chunks(self.n_batch) {
            unsafe {
                let mut b = sys::llama_batch_init(window.len() as i32, 0, 1);
                for (w, &(t, pos, seq, slot)) in window.iter().enumerate() {
                    *b.token.add(w) = t;
                    *b.pos.add(w) = pos;
                    *b.n_seq_id.add(w) = 1;
                    *(*b.seq_id.add(w)) = seq;
                    *b.logits.add(w) = slot.is_some() as i8;
                }
                b.n_tokens = window.len() as i32;
                let rc = sys::llama_decode(ctx, b);
                if rc != 0 {
                    sys::llama_batch_free(b);
                    return Err(BackendError::Http(format!("llama_decode returned {rc}")));
                }
                for (w, &(.., slot)) in window.iter().enumerate() {
                    if let Some(s) = slot {
                        out(s, w as i32);
                    }
                }
                sys::llama_batch_free(b);
            }
        }
        Ok(())
    }

    /// The control reading, for the polygraph: the whole prompt on sequence
    /// 0 from an empty cache, no fork, nothing kept.
    pub fn read_control(
        &self,
        prompt: &Segs,
        candidates: &[String],
    ) -> Result<Scored, BackendError> {
        let t0 = Instant::now();
        let mut g = self.inner.lock().expect("artichoke lock");
        let seqs = candidates
            .iter()
            .map(|c| self.label_tokens(&mut g, c))
            .collect::<Result<Vec<_>, _>>()?;
        let toks = self.tokenize_segs(prompt)?;
        let ctx = g.ctx;
        let mem = unsafe { sys::llama_get_memory(ctx) };
        g.cached.clear();
        let last = toks.len() - 1;
        let plan: Vec<_> = toks
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, i as i32, 0, (i == last).then_some(0)))
            .collect();
        let mut evaluated = 0usize;
        let logprobs = if seqs.iter().all(|s| s.len() == 1) {
            let ids: Vec<_> = seqs.iter().map(|s| s[0]).collect();
            unsafe { sys::llama_memory_clear(mem, true) };
            let mut reading = None;
            self.decode(ctx, &plan, |_, row| {
                reading = Some(self.read(ctx, row, &ids))
            })?;
            evaluated += toks.len();
            reading
                .ok_or_else(|| BackendError::Malformed("no output row".into()))?
                .logprobs
        } else {
            // Several-token labels, brute force: every label from an empty
            // cache, the prompt and then the label one token at a time, the
            // log-probabilities of its tokens summed. No fork anywhere.
            let mut out = Vec::with_capacity(seqs.len());
            for s in &seqs {
                unsafe { sys::llama_memory_clear(mem, true) };
                let mut row_lp = None;
                self.decode(ctx, &plan, |_, row| {
                    row_lp = Some(self.read(ctx, row, &s[..1]).logprobs[0])
                })?;
                let mut sum =
                    row_lp.ok_or_else(|| BackendError::Malformed("no output row".into()))?;
                evaluated += toks.len();
                for d in 1..s.len() {
                    let step = [(s[d - 1], (toks.len() + d - 1) as i32, 0, Some(0))];
                    let mut lp = None;
                    self.decode(ctx, &step, |_, row| {
                        lp = Some(self.read(ctx, row, &s[d..=d]).logprobs[0])
                    })?;
                    sum += lp.ok_or_else(|| BackendError::Malformed("no output row".into()))?;
                    evaluated += 1;
                }
                out.push(sum);
            }
            out
        };
        unsafe { sys::llama_memory_clear(mem, true) };
        Ok(Scored {
            logprobs,
            cost: ScoreCost {
                prompt_evaluated: evaluated as u64,
                prompt_cached: 0,
                latency_ms: t0.elapsed().as_secs_f64() * 1e3,
            },
        })
    }

    /// The split reading, for the polygraph: the first `at` tokens of the
    /// prompt on sequence 0, then the rest on sequence 0 as a second
    /// decode. The same two decodes a fork makes, without the copy.
    pub fn read_split(
        &self,
        prompt: &Segs,
        candidates: &[String],
        at: usize,
    ) -> Result<Scored, BackendError> {
        let t0 = Instant::now();
        let mut g = self.inner.lock().expect("artichoke lock");
        let ids = candidates
            .iter()
            .map(|c| self.label_id(&mut g, c))
            .collect::<Result<Vec<_>, _>>()?;
        let toks = self.tokenize_segs(prompt)?;
        let at = at.min(toks.len() - 1);
        let ctx = g.ctx;
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(ctx), true) };
        g.cached.clear();
        let last = toks.len() - 1;
        let head: Vec<_> = toks[..at]
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, i as i32, 0, None))
            .collect();
        let tail: Vec<_> = toks[at..]
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, (at + i) as i32, 0, (at + i == last).then_some(0)))
            .collect();
        self.decode(ctx, &head, |_, _| {})?;
        let mut reading = None;
        self.decode(ctx, &tail, |_, row| {
            reading = Some(self.read(ctx, row, &ids))
        })?;
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(ctx), true) };
        let r = reading.ok_or_else(|| BackendError::Malformed("no output row".into()))?;
        Ok(Scored {
            logprobs: r.logprobs,
            cost: ScoreCost {
                prompt_evaluated: toks.len() as u64,
                prompt_cached: 0,
                latency_ms: t0.elapsed().as_secs_f64() * 1e3,
            },
        })
    }

    fn fork_and_read(
        &self,
        prompts: &[Vec<sys::llama_token>],
        cands: &[Vec<Vec<sys::llama_token>>],
        shared: usize,
    ) -> Result<Vec<Scored>, BackendError> {
        let t0 = Instant::now();
        check_limits(prompts, cands, shared, self.n_ctx)?;
        let mut g = self.inner.lock().expect("artichoke lock");
        let ctx = g.ctx;
        let mem = unsafe { sys::llama_get_memory(ctx) };

        let session = &prompts[0][..shared];

        // Reuse what sequence 0 already holds. A recurrent state cannot be
        // cut back, so a partial match there means starting over.
        let reuse = g
            .cached
            .iter()
            .zip(session)
            .take_while(|(a, b)| a == b)
            .count();
        if reuse < g.cached.len() {
            let cut = unsafe { sys::llama_memory_seq_rm(mem, 0, reuse as i32, -1) };
            if cut {
                g.cached.truncate(reuse);
            } else {
                unsafe { sys::llama_memory_clear(mem, true) };
                g.cached.clear();
            }
        }
        let start = g.cached.len();
        if shared > start {
            let plan: Vec<_> = session[start..]
                .iter()
                .enumerate()
                .map(|(i, &t)| (t, (start + i) as i32, 0, None))
                .collect();
            if let Err(e) = self.decode(ctx, &plan, |_, _| {}) {
                unsafe { sys::llama_memory_clear(mem, true) };
                g.cached.clear();
                return Err(e);
            }
        }
        g.cached = session.to_vec();

        // One-token labels are read in rounds of forks, as many as there are
        // sequences and cells for; multi-token labels one fingerprint at a
        // time as a trie (`read_trie`).
        let single: Vec<usize> = (0..prompts.len())
            .filter(|&n| cands[n].iter().all(|c| c.len() == 1))
            .collect();
        let firsts: Vec<Vec<sys::llama_token>> = cands
            .iter()
            .map(|cs| cs.iter().map(|c| c[0]).collect())
            .collect();
        let mut readings: Vec<Option<(Reading, usize)>> =
            (0..prompts.len()).map(|_| None).collect();
        let mut i = 0;
        while i < single.len() {
            let mut j = i;
            let mut cells = shared;
            while j < single.len()
                && j - i < self.forks
                && cells + prompts[single[j]].len() - shared <= self.n_ctx
            {
                cells += prompts[single[j]].len() - shared;
                j += 1;
            }
            if j == i {
                return Err(BackendError::Rejected(format!(
                    "a prompt of {} tokens does not fit a context of {}",
                    prompts[single[i]].len(),
                    self.n_ctx
                )));
            }
            let mut plan = Vec::new();
            for (k, &n) in single[i..j].iter().enumerate() {
                let seq = (k + 1) as i32;
                // The Hologram: every fork holds the whole session.
                unsafe { sys::llama_memory_seq_cp(mem, 0, seq, -1, -1) };
                let suffix = &prompts[n][shared..];
                for (m, &t) in suffix.iter().enumerate() {
                    let slot = (m + 1 == suffix.len()).then_some(n);
                    plan.push((t, (shared + m) as i32, seq, slot));
                }
            }
            let result = self.decode(ctx, &plan, |slot, row| {
                let r = self.read(ctx, row, &firsts[slot]);
                readings[slot] = Some((r, prompts[slot].len() - shared));
            });
            for k in 0..(j - i) {
                unsafe { sys::llama_memory_seq_rm(mem, (k + 1) as i32, -1, -1) };
            }
            result?;
            i = j;
        }
        for n in 0..prompts.len() {
            if readings[n].is_none() {
                readings[n] = Some(self.read_trie(ctx, mem, &prompts[n], shared, &cands[n])?);
            }
        }
        let each = t0.elapsed().as_secs_f64() * 1e3 / prompts.len() as f64;
        readings
            .into_iter()
            .enumerate()
            .map(|(n, r)| {
                let (r, evaluated) =
                    r.ok_or_else(|| BackendError::Malformed(format!("no output for prompt {n}")))?;
                Ok(Scored {
                    logprobs: r.logprobs,
                    cost: ScoreCost {
                        // The session is charged to the first fingerprint.
                        prompt_evaluated: (evaluated + if n == 0 { shared - start } else { 0 })
                            as u64,
                        prompt_cached: if n == 0 { start } else { shared } as u64,
                        latency_ms: each,
                    },
                })
            })
            .collect()
    }

    /// One fingerprint whose labels are several tokens: its suffix forked
    /// from the session onto sequence 1 and read at its end (the root),
    /// then every distinct proper prefix of the labels forked from
    /// sequence 1 and read one step further, the probes in rounds on the
    /// other sequences. A label's log-probability is the sum of its tokens'
    /// along the trie. Returns the reading and the tokens it evaluated.
    fn read_trie(
        &self,
        ctx: *mut sys::llama_context,
        mem: sys::llama_memory_t,
        prompt: &[sys::llama_token],
        shared: usize,
        cands: &[Vec<sys::llama_token>],
    ) -> Result<(Reading, usize), BackendError> {
        use std::collections::BTreeMap;
        // What each node (a label prefix) needs read: the next tokens of
        // the labels through it.
        let mut needs: BTreeMap<Vec<sys::llama_token>, Vec<sys::llama_token>> = BTreeMap::new();
        for c in cands {
            for d in 0..c.len() {
                let e = needs.entry(c[..d].to_vec()).or_default();
                if !e.contains(&c[d]) {
                    e.push(c[d]);
                }
            }
        }
        let mut read: BTreeMap<Vec<sys::llama_token>, Vec<f64>> = BTreeMap::new();
        let len = prompt.len();
        let mut evaluated = len - shared;

        // The root on sequence 1.
        unsafe { sys::llama_memory_seq_cp(mem, 0, 1, -1, -1) };
        let suffix = &prompt[shared..];
        let plan: Vec<_> = suffix
            .iter()
            .enumerate()
            .map(|(m, &t)| {
                (
                    t,
                    (shared + m) as i32,
                    1,
                    (m + 1 == suffix.len()).then_some(0),
                )
            })
            .collect();
        let root_ids = needs.get(&Vec::new()).cloned().unwrap_or_default();
        let mut root = None;
        let r = self.decode(ctx, &plan, |_, row| {
            root = Some(self.read(ctx, row, &root_ids).logprobs)
        });
        if let Err(e) = r {
            unsafe { sys::llama_memory_seq_rm(mem, 1, -1, -1) };
            return Err(e);
        }
        read.insert(
            Vec::new(),
            root.ok_or_else(|| BackendError::Malformed("no root row".into()))?,
        );

        // The probes, forked from sequence 1 in rounds of the free sequences.
        let probes: Vec<Vec<sys::llama_token>> =
            needs.keys().filter(|p| !p.is_empty()).cloned().collect();
        if self.forks < 2 && !probes.is_empty() {
            unsafe { sys::llama_memory_seq_rm(mem, 1, -1, -1) };
            return Err(BackendError::Rejected(
                "labels of several tokens need --forks 2 or more (a sequence for the probes)"
                    .into(),
            ));
        }
        let free = self.forks - 1;
        let mut result = Ok(());
        for round in probes.chunks(free) {
            let mut plan = Vec::new();
            for (k, p) in round.iter().enumerate() {
                let seq = (k + 2) as i32;
                unsafe { sys::llama_memory_seq_cp(mem, 1, seq, -1, -1) };
                for (m, &t) in p.iter().enumerate() {
                    let slot = (m + 1 == p.len()).then_some(k);
                    plan.push((t, (len + m) as i32, seq, slot));
                }
                evaluated += p.len();
            }
            let mut rows = vec![Vec::new(); round.len()];
            result = self.decode(ctx, &plan, |slot, row| {
                rows[slot] = self.read(ctx, row, &needs[&round[slot]]).logprobs;
            });
            for k in 0..round.len() {
                unsafe { sys::llama_memory_seq_rm(mem, (k + 2) as i32, -1, -1) };
            }
            if result.is_err() {
                break;
            }
            for (p, r) in round.iter().zip(rows) {
                read.insert(p.clone(), r);
            }
        }
        unsafe { sys::llama_memory_seq_rm(mem, 1, -1, -1) };
        result?;
        let logprobs = cands
            .iter()
            .map(|c| {
                (0..c.len())
                    .map(|d| {
                        let node = &c[..d];
                        let at = needs[node].iter().position(|&t| t == c[d]).expect("need");
                        read[node][at]
                    })
                    .sum()
            })
            .collect();
        Ok((Reading { logprobs }, evaluated))
    }
}

impl Drop for Artichoke {
    fn drop(&mut self) {
        let g = self.inner.get_mut().expect("artichoke lock");
        unsafe {
            sys::llama_free(g.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

impl Scorer for Artichoke {
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError> {
        // A flattened prompt: its user text is already escaped, so the
        // whole string may be read as template.
        let mut whole = Segs::default();
        whole.t(prompt);
        let mut v = self.score_many(&Segs::default(), &[(whole, candidates.to_vec())])?;
        Ok(v.remove(0))
    }

    fn score_many(
        &self,
        prefix: &Segs,
        items: &[(Segs, Vec<String>)],
    ) -> Result<Vec<Scored>, BackendError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let cands = {
            let mut g = self.inner.lock().expect("artichoke lock");
            items
                .iter()
                .map(|(_, c)| {
                    c.iter()
                        .map(|l| self.label_tokens(&mut g, l))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        // Whole prompts, tokenized as one string each, so the fork point is
        // wherever their tokens part and never a tokenizer seam.
        let (prompts, shared) = self.prepare(prefix, items)?;
        self.fork_and_read(&prompts, &cands, shared)
    }

    fn multi_token_labels(&self) -> bool {
        true
    }

    fn model_name(&self) -> String {
        self.name.clone()
    }
}

/// TypeSafe's limits: the state and the longest question within 32k tokens,
/// the whole request within 64k (docs.typesafe.ai, models).
pub const BRANCH_LIMIT: usize = 32_000;
pub const REQUEST_LIMIT: usize = 64_000;

/// REBAL, the Resonant Energy Balloon: the field around the engine.
/// Refuse, before any prefill, a request that cannot be read: past
/// TypeSafe's limits (counted with this subject's tokenizer, template
/// included), or a fingerprint that with its longest label does not fit
/// the context. A refusal is the caller's request (422), not a failure of
/// the engine (502), and costs nothing.
fn check_limits(
    prompts: &[Vec<sys::llama_token>],
    cands: &[Vec<Vec<sys::llama_token>>],
    shared: usize,
    n_ctx: usize,
) -> Result<(), BackendError> {
    let longest = prompts
        .iter()
        .zip(cands)
        .map(|(p, cs)| {
            p.len()
                + cs.iter()
                    .map(|c| c.len().saturating_sub(1))
                    .max()
                    .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);
    let request = shared + prompts.iter().map(|p| p.len() - shared).sum::<usize>();
    if longest > BRANCH_LIMIT {
        return Err(BackendError::Rejected(format!(
            "the state and a question come to {longest} tokens, past the {BRANCH_LIMIT} limit"
        )));
    }
    if request > REQUEST_LIMIT {
        return Err(BackendError::Rejected(format!(
            "the request comes to {request} tokens, past the {REQUEST_LIMIT} limit"
        )));
    }
    if longest > n_ctx {
        return Err(BackendError::Rejected(format!(
            "the state and a question come to {longest} tokens, past this server's context \
             of {n_ctx} (xks --ctx)"
        )));
    }
    Ok(())
}

/// The session's length in tokens: the longest prefix every prompt shares,
/// short of each prompt's last token (every fork decodes at least one).
fn fork_point(prompts: &[Vec<sys::llama_token>]) -> usize {
    let mut shared = prompts[0].len() - 1;
    for p in &prompts[1..] {
        let lcp = prompts[0].iter().zip(p).take_while(|(a, b)| a == b).count();
        shared = shared.min(lcp).min(p.len() - 1);
    }
    shared
}

impl Artichoke {
    /// Where `score_many` would cut these fingerprints from their session.
    pub fn session_tokens(
        &self,
        prefix: &Segs,
        items: &[(Segs, Vec<String>)],
    ) -> Result<usize, BackendError> {
        Ok(self.prepare(prefix, items)?.1)
    }

    /// Every fingerprint's whole prompt, and where they fork: where their
    /// tokens part, but never past the end of the session itself (the
    /// prefix's own tokens). Cut inside the questions, the kept session
    /// would run into one request's question text, and the same state asked
    /// with other questions could not reuse it: a recurrent state cannot be
    /// cut back, so it would be prefilled again from the start.
    fn prepare(
        &self,
        prefix: &Segs,
        items: &[(Segs, Vec<String>)],
    ) -> Result<(Vec<Vec<sys::llama_token>>, usize), BackendError> {
        let prompts = items
            .iter()
            .map(|(s, _)| self.tokenize_segs(&prefix.concat(s)))
            .collect::<Result<Vec<_>, _>>()?;
        if prompts.iter().any(|p| p.is_empty()) {
            return Err(BackendError::Rejected("empty prompt".into()));
        }
        let own = self.tokenize_segs(prefix)?;
        let session_end = own
            .iter()
            .zip(&prompts[0])
            .take_while(|(a, b)| a == b)
            .count();
        Ok((prompts.clone(), fork_point(&prompts).min(session_end)))
    }
}

/// The directory the engine loads CPU variants from by default.
pub fn default_backend_dir() -> &'static Path {
    Path::new(env!("XKS_LLAMA_BUILD_DIR"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_requests_are_refused_before_any_prefill() {
        let p = |n: usize| vec![1 as sys::llama_token; n];
        let one = vec![vec![1 as sys::llama_token]];
        // Fits.
        assert!(check_limits(&[p(100), p(120)], &[one.clone(), one.clone()], 90, 16384).is_ok());
        // Past this server's context, before TypeSafe's limits.
        let e = check_limits(&[p(20_000)], std::slice::from_ref(&one), 19_990, 16384).unwrap_err();
        assert!(matches!(e, BackendError::Rejected(ref m) if m.contains("--ctx")));
        // Past the 32k branch limit.
        assert!(check_limits(&[p(33_000)], std::slice::from_ref(&one), 32_990, 65_536).is_err());
        // Past the 64k request limit with every branch inside 32k.
        let many: Vec<_> = (0..5).map(|_| p(31_000)).collect();
        let cands: Vec<_> = (0..5).map(|_| one.clone()).collect();
        assert!(matches!(
            check_limits(&many, &cands, 10_000, 131_072),
            Err(BackendError::Rejected(ref m)) if m.contains("64000")
        ));
        // A multi-token label counts past the prompt.
        let long = vec![vec![1 as sys::llama_token; 10]];
        assert!(check_limits(&[p(16_380)], &[long], 16_000, 16_384).is_err());
    }
}
