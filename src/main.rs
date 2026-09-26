//! The `xks` command line. See main.md.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use serde_json::{json, Map, Value};

use xks::backend::bluebird::Bluebird;
use xks::backend::openai::OpenAiChat;
use xks::backend::typesafe::TypeSafe;
use xks::backend::Scorer;
use xks::eval;
use xks::judge::{Judge, JudgeConfig};
use xks::prompt::{Layout, Template};
use xks::protocol::Request;
use xks::score::Calibration;
use xks::server::{serve, ServerConfig};

#[derive(Parser)]
#[command(
    name = "xks",
    version,
    about = "XKEYSCORE for Jev: typed System One judgments, one prefill per session",
    after_help = "Start here: `xks serve --detach`, then `xks query --file examples/query.json` \
                  (a plain query goes to the running server), then `xks stop`."
)]
struct Cli {
    /// The subject (a GGUF) ARTICHOKE interrogates.
    #[arg(long, alias = "gguf", env = "XKS_SUBJECT")]
    subject: Option<PathBuf>,
    /// artichoke: cells of the unified cache (session plus one round of suffixes).
    #[arg(long, env = "XKS_CTX", default_value_t = 16384)]
    ctx: u32,
    /// artichoke: fingerprints forked from the session and decoded together.
    #[arg(long, env = "XKS_FORKS", default_value_t = 15)]
    forks: u32,
    /// artichoke: tokens per llama_decode.
    #[arg(long, env = "XKS_BATCH", default_value_t = 2048)]
    batch: u32,
    /// artichoke: tokens per physical step.
    #[arg(long, env = "XKS_UBATCH", default_value_t = 512)]
    ubatch: u32,
    /// artichoke: llama.cpp threads (default 12 with the Phi payload loaded, else 16).
    #[arg(long, env = "XKS_THREADS")]
    threads: Option<i32>,
    /// artichoke: show llama.cpp's informational log.
    #[arg(long)]
    verbose: bool,
    /// artichoke: let llama.cpp repack weights for the host CPU (default: only
    /// without the Phi payload; with it, repacked weights never reach a card).
    #[arg(long, env = "XKS_REPACK")]
    repack: Option<bool>,
    /// Where the subject runs: auto (cards when one is up) | x86 (this host
    /// alone, the reference) | cards (weight multiplies on the Phi VPUs) |
    /// avx512 (the AVX-512 build under phi512, with the cards' payload).
    #[arg(long, env = "XKS_SITE", default_value = "auto")]
    site: String,
    /// cards / avx512: let the payload keep a tensor on the host when the
    /// host is faster (default: every row the cards hold is theirs).
    #[arg(long)]
    no_offload: bool,
    /// artichoke (in-process, default) | bluebird (a llama-server at --backend)
    /// | openai (chat/completions with logprobs at --backend)
    #[arg(long, env = "XKS_BACKEND_KIND", default_value = "artichoke")]
    backend_kind: String,
    /// bluebird: a llama-server root; openai: an OpenAI-compatible `/v1` root.
    #[arg(long, env = "XKS_BACKEND_URL", default_value = "http://127.0.0.1:8089")]
    backend: String,
    /// Model id for --backend-kind openai (e.g. deepseek-chat).
    #[arg(long, env = "XKS_MODEL")]
    model: Option<String>,
    /// Environment variable holding the API key for --backend-kind openai.
    #[arg(long, default_value = "XKS_API_KEY")]
    api_key_env: String,
    /// Extra JSON merged into openai requests, e.g. '{"thinking":{"type":"disabled"}}'.
    #[arg(long, env = "XKS_EXTRA")]
    extra: Option<String>,
    /// How fingerprints are laid out: letters (lettered options) | jev (the
    /// layout reconstructed from Jev's documentation; needs artichoke).
    #[arg(long, env = "XKS_LAYOUT", default_value = "letters")]
    layout: String,
    /// Chat template of the subject: chatml | gemma | llama3 | raw
    #[arg(long, env = "XKS_TEMPLATE", default_value = "chatml")]
    template: String,
    /// Conditioning: per-bucket temperatures (from `xks condition`).
    #[arg(long, alias = "calibration", env = "XKS_CONDITIONING")]
    conditioning: Option<PathBuf>,
    /// Cyclic option rotations to average for `choice` (position-bias control).
    #[arg(long, env = "XKS_PERMUTATIONS", default_value_t = 1)]
    permutations: usize,
    /// Include raw logprobs and per-question latency in responses.
    #[arg(long)]
    debug: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve POST /v1/systemone (Jev wire format).
    Serve {
        #[arg(long, env = "XKS_BIND", default_value = "127.0.0.1:8090")]
        bind: String,
        /// Comma-separated bearer tokens; default none.
        #[arg(long, env = "XKS_API_KEYS", default_value = "")]
        api_keys: String,
        /// Kill date: exit after this many seconds with no request, so the
        /// cards go back to whoever needs them. 0 = never.
        #[arg(long, env = "XKS_KILL_DATE", default_value_t = 0)]
        kill_date: u64,
        /// Run in the background: log and pid file under
        /// $XDG_RUNTIME_DIR/xks, returns once the server answers /health.
        #[arg(long)]
        detach: bool,
        /// Record every question set and its answers here, one JSON line
        /// each (src/decisions.md): by default a hash of the state and of
        /// each question, not their text.
        #[arg(long, env = "XKS_DECISION_LOG")]
        decision_log: Option<PathBuf>,
        /// Keep the state and questions as sent in the decision log.
        #[arg(long, env = "XKS_DECISION_LOG_TEXT")]
        log_text: bool,
    },
    /// One query. Reads a JSON request from --file or stdin, or builds one
    /// from --state and --noul/--choice/--score flags. With a server
    /// running at XKS_BIND and no engine option on the command line, the
    /// request goes to that server; otherwise the subject loads here.
    #[command(alias = "ask")]
    Query {
        /// Load the subject in this process even when a server is running.
        #[arg(long)]
        local: bool,
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long)]
        state: Option<String>,
        /// id=instructions
        #[arg(long)]
        noul: Vec<String>,
        /// id=instructions|key1:desc,key2:desc
        #[arg(long)]
        choice: Vec<String>,
        /// id=instructions|level0,level1,level2
        #[arg(long)]
        score: Vec<String>,
        /// Send the same request to the hosted TypeSafe API too (needs TYPESAFE_API_KEY).
        #[arg(long)]
        compare: bool,
    },
    /// Run a labelled JSONL case file: accuracy, Brier, ECE, latency.
    Eval {
        file: PathBuf,
        /// Record the raw readings here (for `replay`).
        #[arg(long)]
        rows: Option<PathBuf>,
        /// Only the first N cases.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Every experiment as one command: `list`, `run NN`, `run all`.
    Subproject {
        /// `list` (default) or `run`.
        #[arg(default_value = "list")]
        action: String,
        /// A subproject id or name, or `all`.
        which: Option<String>,
    },
    /// Stop a server started with `serve --detach`, and release the cards.
    Stop,
    /// Replay recorded readings (from `eval --rows`) through a conditioning,
    /// or fit a new one, without the subject.
    Replay {
        rows: PathBuf,
        /// Fit a conditioning to the recorded readings and write it here.
        #[arg(long)]
        fit: Option<PathBuf>,
    },
    /// Run as an MCP server over stdio (Claude Code, Codex, OpenCode).
    Mcp,
    /// Run a labelled case file and fit a conditioning (per-bucket temperatures).
    #[command(alias = "calibrate")]
    Condition {
        file: PathBuf,
        #[arg(long, default_value = "conditioning.json")]
        out: PathBuf,
    },
    /// The payload's accounting: total a `PHI_GGML_VERBOSE=1` log (a file,
    /// else stdin) into what each card and the host computed.
    Ledger { log: Option<PathBuf> },
    /// Compare two recorded runs (`eval --rows`) fingerprint by fingerprint,
    /// e.g. the x86 site against the cards.
    Corroborate {
        a: PathBuf,
        b: PathBuf,
        /// Also list every pair.
        #[arg(long)]
        pairs: bool,
    },
    /// Stop the card workers and release their huge pages.
    #[cfg(feature = "artichoke")]
    Release,
    /// What this machine has of what xks needs (the llama.cpp build, the
    /// subject, Intel-Phi-AVX512 and its payload, the cards, a server),
    /// and the fix for what is missing. Starts nothing. Exit 0 when a
    /// question can be answered, 1 when not.
    #[cfg(feature = "artichoke")]
    Doctor {
        /// Do the fixes that are a build or a link: the payload built,
        /// xks linked into PREFIX/bin. Never a card, a download or sudo.
        #[arg(long)]
        fix: bool,
        /// Where --fix links xks (PREFIX/bin/xks).
        #[arg(long, default_value = "~/.local")]
        prefix: String,
    },
    /// ARTICHOKE only: read every fingerprint of every case three ways
    /// (forked, split, control) and compare the readings.
    #[cfg(feature = "artichoke")]
    Polygraph {
        file: PathBuf,
        /// Also print every row.
        #[arg(long)]
        rows: bool,
        /// Only the first N cases.
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[cfg(feature = "artichoke")]
fn artichoke(cli: &Cli) -> Result<(xks::artichoke::Artichoke, xks::site::Placed), String> {
    let subject = cli.subject.clone().ok_or(
        "no subject: set XKS_SUBJECT=/path/to/model.gguf in xks.local.conf, or pass --subject",
    )?;
    check_subject(&subject)?;
    // The site first: it sets the payload's environment, which must be in
    // place before llama.cpp registers its backends.
    let site = xks::site::resolve(&cli.site)?;
    let placed = xks::site::prepare(site, !cli.no_offload)?;
    eprintln!(
        "xks: site {}{}",
        placed.site.name(),
        if placed.cards.is_empty() {
            String::new()
        } else {
            format!(" (cards {:?})", placed.cards)
        }
    );
    let mut o = xks::artichoke::Options::new(subject);
    o.n_ctx = cli.ctx;
    o.forks = cli.forks.max(1);
    o.n_batch = cli.batch;
    o.n_ubatch = cli.ubatch;
    o.threads = cli.threads.unwrap_or(placed.threads);
    o.repack = cli.repack.unwrap_or(placed.repack);
    o.flash_attn = placed.flash_attn;
    o.backend_dir = std::env::var_os("XKS_BACKEND_DIR").map(PathBuf::from);
    o.verbose = cli.verbose;
    // Offloaded, the cards' rows leave the host after their upload: read
    // them in only then, a tensor at a time, not the whole file at load.
    o.lazy_pages = placed.site != xks::site::Site::X86 && !cli.no_offload;
    Ok((xks::artichoke::Artichoke::open(&o)?, placed))
}

/// libc's `mmap`, found once past this binary's (`RTLD_NEXT`: under
/// phi512 a preloaded library could stand between, and must not be
/// skipped).
#[cfg(feature = "artichoke")]
fn next_mmap() -> MmapFn {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    extern "C" {
        fn dlsym(
            handle: *mut std::ffi::c_void,
            name: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void;
    }
    let mut f = NEXT.load(Ordering::Relaxed);
    if f == 0 {
        // RTLD_NEXT is ((void *) -1) in glibc's dlfcn.h.
        // SAFETY: a NUL-terminated name; dlsym is thread-safe.
        f = unsafe { dlsym(usize::MAX as *mut std::ffi::c_void, c"mmap".as_ptr()) } as usize;
        if f == 0 {
            std::process::abort();
        }
        NEXT.store(f, Ordering::Relaxed);
    }
    // SAFETY: dlsym returned libc's mmap, which has this signature.
    unsafe { std::mem::transmute::<usize, MmapFn>(f) }
}

#[cfg(feature = "artichoke")]
type MmapFn =
    unsafe extern "C" fn(*mut std::ffi::c_void, usize, i32, i32, i32, i64) -> *mut std::ffi::c_void;

/// `mmap`, exported from this binary (build.rs) so that libllama's call
/// resolves here before libc: llama.cpp maps a model with `MAP_POPULATE`
/// (its `llama_mmap`, prefetch on), which reads the whole file into this
/// process at load. While the subject loads on an offloaded site
/// (`artichoke::LAZY_PAGES`), a file mapping loses that flag and its pages
/// come in as they are used; every other call passes through unchanged.
/// llama.cpp itself is not modified. See main.md.
///
/// # Safety
/// As libc's `mmap`.
#[cfg(feature = "artichoke")]
#[no_mangle]
pub unsafe extern "C" fn mmap(
    addr: *mut std::ffi::c_void,
    len: usize,
    prot: i32,
    flags: i32,
    fd: i32,
    off: i64,
) -> *mut std::ffi::c_void {
    const MAP_POPULATE: i32 = 0x8000;
    let lazy = fd >= 0 && xks::artichoke::LAZY_PAGES.load(std::sync::atomic::Ordering::SeqCst);
    let flags = if lazy { flags & !MAP_POPULATE } else { flags };
    // SAFETY: the caller's arguments, to libc's mmap.
    unsafe { next_mmap()(addr, len, prot, flags, fd, off) }
}

fn main() {
    // Defaults from xks.conf and friends, before clap reads the environment
    // and before any thread exists.
    xks::config::load();
    // Bare `xks`: the help, not a missing-subcommand error. By hand, since
    // clap counts the variables xks.conf sets as arguments given.
    if std::env::args_os().len() == 1 {
        let _ = Cli::command().print_help();
        std::process::exit(2);
    }
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `--backend-kind jev`: the real Jev, through TypeSafe's API. Needs
/// `TYPESAFE_API_KEY` (the environment, or `xks.local.conf`). The request
/// and the cases come already read and checked.
fn jev(cmd: &Cmd, request: Option<Request>, cases: Option<Vec<eval::Case>>) -> Result<(), String> {
    let ts = TypeSafe::from_env().ok_or(
        "the real Jev needs TYPESAFE_API_KEY: create a key at console.typesafe.ai and put \
         TYPESAFE_API_KEY=... in xks.local.conf (not tracked) or the environment",
    )?;
    match (cmd, request, cases) {
        (Cmd::Query { .. }, Some(req), _) => {
            let (ev, ms) = ts.evaluate(&req).map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(&ev).unwrap());
            eprintln!("jev ({}): {ms:.1} ms end-to-end", ev.model);
            Ok(())
        }
        (Cmd::Eval { rows, .. }, _, Some(cases)) => {
            let t0 = Instant::now();
            let (r, failed) = eval::run_jev(&ts, &cases).map_err(|e| e.to_string())?;
            let wall = t0.elapsed().as_secs_f64();
            let m = eval::metrics(&r, failed, &Calibration::default());
            let mut v = serde_json::to_value(&m).unwrap();
            v["subject"] = json!(format!("typesafe/{}", ts.model));
            v["site"] = json!("typesafe (hosted)");
            v["cases"] = json!(cases.len());
            v["wall_s"] = json!((wall * 100.0).round() / 100.0);
            println!("{}", serde_json::to_string_pretty(&v).unwrap());
            if let Some(p) = rows {
                let text: String = r
                    .iter()
                    .map(|x| serde_json::to_string(x).unwrap() + "\n")
                    .collect();
                std::fs::write(p, text).map_err(|e| e.to_string())?;
            }
            Ok(())
        }
        _ => Err("--backend-kind jev runs query and eval (the hosted Jev serves itself)".into()),
    }
}

/// Where a detached server keeps its pid file and log.
fn run_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("xks")
}

/// `serve --detach`: the same command line without `--detach`, in its own
/// process group, its output in a log; returns once it answers /health.
fn detach(bind: &str) -> Result<(), String> {
    use std::os::unix::process::CommandExt;
    let dir = run_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let pidfile = dir.join("serve.pid");
    if let Some(pid) = detached_server(&pidfile) {
        return Err(format!("a server is already running (pid {pid}); xks stop"));
    }
    let log = dir.join("serve.log");
    let out = std::fs::File::create(&log).map_err(|e| e.to_string())?;
    let err = out.try_clone().map_err(|e| e.to_string())?;
    let args: Vec<_> = std::env::args_os()
        .skip(1)
        .filter(|a| a != "--detach")
        .collect();
    let url = format!("http://{bind}");
    let answers = || {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(2))
            .build()
            .get(&format!("{url}/health"))
            .call()
            .is_ok()
    };
    if answers() {
        return Err(format!(
            "something already answers on {bind} (a server not started with --detach?)"
        ));
    }
    let mut child = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
        .args(args)
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .process_group(0)
        .spawn()
        .map_err(|e| e.to_string())?;
    std::fs::write(&pidfile, child.id().to_string()).map_err(|e| e.to_string())?;
    eprintln!(
        "xks: server pid {} starting, log {}",
        child.id(),
        log.display()
    );
    // The child's own lines (site, workers, loading, its error) relayed as
    // they come: a 35B start is minutes, and silence reads as a hang.
    let mut relay = Relay::default();
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(900) {
        relay.more(&log);
        // Reaped here, so a child that dies at startup is seen at once
        // (unreaped, its /proc entry would outlive it until the timeout).
        if let Ok(Some(status)) = child.try_wait() {
            relay.rest(&log);
            let _ = std::fs::remove_file(&pidfile);
            return Err(format!(
                "the server exited ({status}); log {}",
                log.display()
            ));
        }
        if answers() {
            relay.rest(&log);
            println!("{url}/v1/systemone");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!(
        "the server did not answer in 900 s (still running, pid {}); log {}",
        child.id(),
        log.display()
    ))
}

/// The lines of a detached server's log shown while `serve --detach`
/// waits: `xks`'s own (`xks: ...`, `xks listening ...`) and its
/// `error: ...`, on stderr; llama.cpp's are left in the log. Stdout
/// carries only the URL.
#[derive(Default)]
struct Relay {
    /// Bytes of the log already looked at.
    seen: usize,
}

impl Relay {
    /// The complete lines written since the last call.
    fn more(&mut self, log: &Path) {
        self.take(log, false);
    }

    /// Everything left, a last unterminated line included.
    fn rest(&mut self, log: &Path) {
        self.take(log, true);
    }

    fn take(&mut self, log: &Path, all: bool) {
        let Ok(bytes) = std::fs::read(log) else {
            return;
        };
        let new = bytes.get(self.seen..).unwrap_or_default();
        let end = if all {
            new.len()
        } else {
            new.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1)
        };
        for line in String::from_utf8_lossy(&new[..end]).lines() {
            if relayed(line) {
                eprintln!("{line}");
            }
        }
        self.seen += end;
    }
}

/// Whether a server log line is one `serve --detach` shows.
fn relayed(line: &str) -> bool {
    line.starts_with("xks:") || line.starts_with("xks listening") || line.starts_with("error:")
}

/// The pid in `pidfile` when that process is still a detached `xks serve`.
/// A pid alone is not enough: after a crash or a reboot it can name any
/// other process, which `stop` would then kill.
fn detached_server(pidfile: &Path) -> Option<u32> {
    let pid: u32 = std::fs::read_to_string(pidfile).ok()?.trim().parse().ok()?;
    let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    is_server_cmdline(&cmdline).then_some(pid)
}

/// A NUL-separated command line that runs `xks ... serve` (directly, or
/// the avx512 site's `phi512.sh ... xks serve`).
fn is_server_cmdline(cmdline: &[u8]) -> bool {
    let args: Vec<&[u8]> = cmdline.split(|&b| b == 0).collect();
    let xks = args.iter().position(|a| a.ends_with(b"xks"));
    xks.is_some_and(|i| args[i + 1..].iter().any(|a| *a == b"serve"))
}

/// `stop`: end a detached server (its whole process group, SIGKILL after
/// 30 s) and give back the cards whose workers `xks` started. Workers
/// something else started on the other cards are left alone.
fn stop() -> Result<(), String> {
    let pidfile = run_dir().join("serve.pid");
    match detached_server(&pidfile) {
        Some(pid) => {
            let group = format!("-{pid}");
            let _ = Command::new("kill").args(["-TERM", "--", &group]).status();
            let alive = || Path::new(&format!("/proc/{pid}")).exists();
            let t0 = Instant::now();
            while alive() && t0.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_millis(200));
            }
            if alive() {
                let _ = Command::new("kill").args(["-KILL", "--", &group]).status();
                std::thread::sleep(Duration::from_millis(500));
            }
            if alive() {
                return Err(format!("server {pid} did not stop"));
            }
            eprintln!("xks: server {pid} stopped");
        }
        None => eprintln!("xks: no detached server"),
    }
    let _ = std::fs::remove_file(&pidfile);
    #[cfg(feature = "artichoke")]
    {
        let cards = xks::site::xks_workers();
        // Another xks (a foreground server, an eval) is using the cards:
        // stopping its workers would break it.
        if let Some(who) = xks::site::cards_holder() {
            if !cards.is_empty() {
                eprintln!("xks: cards {cards:?} left running: {who} is using them");
            }
            return Ok(());
        }
        if !cards.is_empty() {
            xks::site::release(&cards)?;
            eprintln!("xks: card workers stopped, huge pages released on {cards:?}");
        }
    }
    Ok(())
}

fn subproject(action: &str, which: Option<&str>) -> Result<(), String> {
    use xks::subproject::{find, run, Ctx, ALL};
    match action {
        "list" => {
            eprintln!(
                "(* = run only by name, not by `all`: it cannot finish inside the budget yet)"
            );
            for s in ALL {
                let mark = if s.in_all { " " } else { "*" };
                println!("{}{mark} {:<20} {}", s.id, s.name, s.what);
            }
            Ok(())
        }
        "run" => {
            let which = which.ok_or("which subproject? an id, a name, or all")?;
            let chosen = find(which);
            if chosen.is_empty() {
                return Err(format!("no subproject `{which}` (xks subproject list)"));
            }
            // Before the first one starts: 03 and 07 would otherwise run
            // their x86 half for minutes and then be refused the cards.
            let need: Vec<&str> = chosen.iter().filter(|s| s.cards).map(|s| s.id).collect();
            if let (false, Some(who)) = (need.is_empty(), xks::site::cards_holder()) {
                return Err(format!(
                    "subprojects {need:?} need the cards, which {who} holds; \
                     stop it first (`xks stop` for a detached server)"
                ));
            }
            let ctx = Ctx::new()?;
            let mut failed = Vec::new();
            for s in chosen {
                let (path, finished) = run(&ctx, s)?;
                println!("{}", path.display());
                if !finished {
                    failed.push(s.id);
                }
            }
            if failed.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "subprojects {failed:?} did not finish (their records say why)"
                ))
            }
        }
        other => Err(format!("unknown action `{other}` (list|run)")),
    }
}

fn read_rows(path: &PathBuf) -> Result<Vec<eval::Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("{}: {e}", path.display())))
        .collect()
}

fn run() -> Result<(), String> {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());
    // Before anything is parsed from the configuration: a bad value there
    // is one of the doctor's findings, not an error that stops it.
    #[cfg(feature = "artichoke")]
    if let Cmd::Doctor { fix, prefix } = &cli.cmd {
        let prefix = match prefix.strip_prefix("~/") {
            Some(rest) => Path::new(&std::env::var("HOME").unwrap_or_default()).join(rest),
            None => PathBuf::from(prefix),
        };
        let repo = xks::config::repo_root()
            .map_or_else(|| "(not in a checkout)".into(), |p| p.display().to_string());
        println!("xks doctor: Intel Phi Jev at {repo}");
        let report = xks::doctor::run(*fix, &prefix);
        print!("{}", report.text());
        std::process::exit(if report.ready() { 0 } else { 1 });
    }
    let template: Template = cli.template.parse()?;
    let layout: Layout = cli.layout.parse()?;
    let conditioning: Calibration = match &cli.conditioning {
        Some(p) => {
            let at = |e: &dyn std::fmt::Display| format!("conditioning {}: {e}", p.display());
            serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| at(&e))?)
                .map_err(|e| at(&e))?
        }
        None => Calibration::default(),
    };
    // Commands that need no backend.
    if let Cmd::Replay { rows, fit } = &cli.cmd {
        let r = read_rows(rows)?;
        let replayed = eval::metrics(&r, 0, &conditioning);
        match fit {
            None => println!("{}", serde_json::to_string_pretty(&replayed).unwrap()),
            Some(out) => {
                let fitted = eval::fit(&r);
                let after = eval::metrics(&r, 0, &fitted);
                std::fs::write(out, serde_json::to_string_pretty(&fitted).unwrap())
                    .map_err(|e| e.to_string())?;
                println!(
                    "{}",
                    json!({"conditioning": fitted, "before": replayed, "after": after, "written": out})
                );
            }
        }
        return Ok(());
    }
    match &cli.cmd {
        Cmd::Serve {
            detach: true, bind, ..
        } => return detach(bind),
        Cmd::Stop => return stop(),
        Cmd::Subproject { action, which } => return subproject(action, which.as_deref()),
        Cmd::Ledger { log } => {
            let l = match log {
                Some(p) => {
                    let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
                    xks::ledger::read(std::io::BufReader::new(f))
                }
                None => xks::ledger::read(std::io::stdin().lock()),
            };
            println!("{}", serde_json::to_string_pretty(&l).unwrap());
            return Ok(());
        }
        Cmd::Corroborate { a, b, pairs } => {
            let mut r = xks::corroborate::compare(&read_rows(a)?, &read_rows(b)?);
            if !pairs {
                r.pairs.clear();
            }
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
            return Ok(());
        }
        #[cfg(feature = "artichoke")]
        Cmd::Release => {
            if let Some(who) = xks::site::cards_holder() {
                return Err(format!(
                    "{who} is using the cards; stop it first \
                     (`xks stop` for a detached server)"
                ));
            }
            let cards = xks::site::card_windows();
            xks::site::release(&cards)?;
            eprintln!("xks: workers stopped and huge pages released on cards {cards:?}");
            return Ok(());
        }
        _ => {}
    }
    // Everything a command reads is read and checked here, before a site
    // is prepared (a worker restart) or the subject loads (minutes on the
    // 35B): a mistake in it is said in a moment.
    //
    // First, whether a plain query goes to a running server, which already
    // has the subject loaded (and on the cards site, the cards): whether
    // stdin is read here depends on it.
    let server = match &cli.cmd {
        Cmd::Query { local: false, .. }
            if !engine_on_command_line(&matches)
                && !matches!(cli.backend_kind.as_str(), "jev" | "typesafe") =>
        {
            running_server()
        }
        _ => None,
    };
    let request = match &cli.cmd {
        Cmd::Query {
            file,
            state,
            noul,
            choice,
            score,
            compare,
            ..
        } => {
            if *compare && TypeSafe::from_env().is_none() {
                return Err(
                    "--compare asks the hosted Jev, which needs TYPESAFE_API_KEY \
                            (in xks.local.conf or the environment)"
                        .into(),
                );
            }
            let stdin = file.is_none()
                && state.is_none()
                && noul.is_empty()
                && choice.is_empty()
                && score.is_empty();
            let req = build_request(
                file.clone(),
                state.clone(),
                noul.clone(),
                choice.clone(),
                score.clone(),
            )?;
            if stdin && reexecs(&cli) && server.is_none() {
                hand_over(&req)?;
            }
            Some(req)
        }
        _ => None,
    };
    let cases = match &cli.cmd {
        Cmd::Eval { file, limit, .. } => {
            let mut c = eval::load_cases(file)?;
            if let Some(n) = limit {
                c.truncate(*n);
            }
            Some(c)
        }
        Cmd::Condition { file, .. } => Some(eval::load_cases(file)?),
        #[cfg(feature = "artichoke")]
        Cmd::Polygraph { file, .. } => Some(eval::load_cases(file)?),
        _ => None,
    };
    if let Cmd::Serve { bind, .. } = &cli.cmd {
        free_to_bind(bind)?;
    }
    // Opened now, so a path that cannot be written is said before the load.
    let decision_log = match &cli.cmd {
        Cmd::Serve {
            decision_log: Some(p),
            log_text,
            ..
        } => Some(xks::decisions::Log::open(p, *log_text)?),
        _ => None,
    };
    #[cfg(feature = "artichoke")]
    if let Cmd::Polygraph { rows, limit, .. } = &cli.cmd {
        let (engine, _) = artichoke(&cli)?;
        let cases = cases.unwrap_or_default();
        let report =
            xks::polygraph::run(&engine, template, &cases, *limit).map_err(|e| e.to_string())?;
        if *rows {
            for r in &report.rows {
                println!("{}", serde_json::to_string(r).unwrap());
            }
        }
        println!("{}", serde_json::to_string_pretty(&report.summary).unwrap());
        return Ok(());
    }
    // The real Jev: TypeSafe's hosted model. It returns typed answers, not
    // label log-probabilities, so it is not a Scorer; query and eval talk to
    // it directly.
    if matches!(cli.backend_kind.as_str(), "jev" | "typesafe") {
        return jev(&cli.cmd, request, cases);
    }
    if let (Some((url, subject)), Some(req), Cmd::Query { compare, .. }) =
        (&server, &request, &cli.cmd)
    {
        return ask_server(url, subject, req, *compare);
    }
    let mut site_name = "remote";
    let mut site_cards = Vec::new();
    let (scorer, template): (Box<dyn Scorer>, Template) = match cli.backend_kind.as_str() {
        #[cfg(feature = "artichoke")]
        "artichoke" => {
            let (engine, placed) = artichoke(&cli)?;
            site_name = placed.site.name();
            site_cards = placed.cards;
            (Box::new(engine), template)
        }
        "bluebird" | "llamacpp" => (Box::new(Bluebird::new(cli.backend.clone())), template),
        "openai" | "chat" => {
            let model = cli
                .model
                .clone()
                .ok_or("--model is required with --backend-kind openai")?;
            let mut s = OpenAiChat::new(
                cli.backend.clone(),
                model,
                std::env::var(&cli.api_key_env).ok(),
            );
            if let Some(x) = &cli.extra {
                s.extra = serde_json::from_str(x).map_err(|e| format!("--extra: {e}"))?;
            }
            // The chat server applies its own template; render plain text.
            (Box::new(s), Template::Raw)
        }
        other => {
            return Err(format!(
                "unknown --backend-kind `{other}` (artichoke|bluebird|openai)"
            ))
        }
    };
    let cfg = JudgeConfig {
        template,
        layout,
        calibration: conditioning,
        permutations: cli.permutations,
        debug: cli.debug,
    };
    let judge = Judge::new(scorer, cfg);

    match cli.cmd {
        Cmd::Serve {
            bind,
            api_keys,
            kill_date,
            ..
        } => {
            let api_keys = api_keys
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            // Holding the cards: say where to ask instead (site.md).
            #[cfg(feature = "artichoke")]
            xks::site::note_serving(&bind);
            let used = site_cards.clone();
            serve(
                judge,
                ServerConfig {
                    bind,
                    api_keys,
                    kill_date: (kill_date > 0).then(|| Duration::from_secs(kill_date)),
                    site: site_name.into(),
                    cards: site_cards,
                    decision_log,
                },
            )?;
            // Only the kill date ends `serve` without an error.
            retire(&used);
            Ok(())
        }
        Cmd::Query { compare, .. } => {
            // Read before the site was prepared; in the avx512 site's
            // outer process (the one that left it for stdin) this point
            // is never reached.
            let req = request.ok_or("no request was read")?;
            let t0 = std::time::Instant::now();
            let ev = judge.evaluate(&req).map_err(|e| e.to_string())?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!("{}", serde_json::to_string_pretty(&ev).unwrap());
            eprintln!("local: {:.1} ms end-to-end", ms);
            if compare {
                compare_with_jev(&req)?;
            }
            Ok(())
        }
        Cmd::Eval { rows, .. } => {
            let cases = cases.unwrap_or_default();
            let t0 = std::time::Instant::now();
            let (r, failed) = eval::run(&judge, &cases).map_err(|e| e.to_string())?;
            let wall = t0.elapsed().as_secs_f64();
            let m = eval::metrics(&r, failed, &judge.cfg.calibration);
            let mut v = serde_json::to_value(&m).unwrap();
            v["subject"] = json!(judge.scorer.model_name());
            v["site"] = json!(site_name);
            v["cases"] = json!(cases.len());
            v["wall_s"] = json!((wall * 100.0).round() / 100.0);
            v["per_case_s"] = json!((wall / cases.len().max(1) as f64 * 100.0).round() / 100.0);
            if let Some(g) = std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| peak_gib(&s))
            {
                v["host_peak_gib"] = json!((g * 100.0).round() / 100.0);
            }
            println!("{}", serde_json::to_string_pretty(&v).unwrap());
            if let Some(p) = rows {
                let text: String = r
                    .iter()
                    .map(|x| serde_json::to_string(x).unwrap() + "\n")
                    .collect();
                std::fs::write(&p, text).map_err(|e| e.to_string())?;
            }
            Ok(())
        }
        Cmd::Mcp => xks::mcp::serve(&judge).map_err(|e| e.to_string()),
        Cmd::Condition { out, .. } => {
            let cases = cases.unwrap_or_default();
            let (r, failed) = eval::run(&judge, &cases).map_err(|e| e.to_string())?;
            let before = eval::metrics(&r, failed, &Calibration::default());
            let fitted = eval::fit(&r);
            let after = eval::metrics(&r, failed, &fitted);
            std::fs::write(&out, serde_json::to_string_pretty(&fitted).unwrap())
                .map_err(|e| e.to_string())?;
            println!(
                "{}",
                json!({"conditioning": fitted, "before": before, "after": after, "written": out})
            );
            Ok(())
        }
        // Handled before any backend is built.
        Cmd::Replay { .. }
        | Cmd::Ledger { .. }
        | Cmd::Corroborate { .. }
        | Cmd::Subproject { .. }
        | Cmd::Stop => Ok(()),
        #[cfg(feature = "artichoke")]
        Cmd::Release => Ok(()),
        #[cfg(feature = "artichoke")]
        Cmd::Polygraph { .. } => Ok(()),
        // Handled first thing in run().
        #[cfg(feature = "artichoke")]
        Cmd::Doctor { .. } => Ok(()),
    }
}

/// A server past its kill date gives back what `xks stop` would: the
/// workers of the cards it used (`used`, empty on the x86 site, which
/// leaves the cards to whoever has them) and its pid file when it is this
/// process's. The engine is already dropped: `serve` owned it.
fn retire(used: &[u32]) {
    #[cfg(feature = "artichoke")]
    {
        let cards: Vec<u32> = xks::site::xks_workers()
            .into_iter()
            .filter(|c| used.contains(c))
            .collect();
        if !cards.is_empty() {
            match xks::site::release(&cards) {
                Ok(()) => eprintln!("xks: card workers stopped, huge pages released on {cards:?}"),
                Err(e) => eprintln!("xks: the cards were not released: {e} (xks release)"),
            }
        }
    }
    #[cfg(not(feature = "artichoke"))]
    let _ = used;
    let pidfile = run_dir().join("serve.pid");
    let ours = std::fs::read_to_string(&pidfile)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        == Some(std::process::id());
    if ours {
        let _ = std::fs::remove_file(pidfile);
    }
}

/// The peak resident memory in a `/proc/PID/status` text (`VmHWM`, the
/// kernel's high-water mark: the subject's mapped file pages count), GiB.
fn peak_gib(status: &str) -> Option<f64> {
    let kb: f64 = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kb / 1048576.0)
}

/// Whether this process is the avx512 site's outer one, which `prepare`
/// replaces (exec) with the AVX-512 build under phi512.
fn reexecs(cli: &Cli) -> bool {
    cfg!(feature = "artichoke")
        && cli.backend_kind == "artichoke"
        && cli.site == "avx512"
        && std::env::var_os("XKS_SITE_INNER").is_none()
}

/// Where the avx512 site's outer process leaves a request it read from
/// stdin, for the process under phi512 (the variable holds the path).
const STDIN_REQUEST: &str = "XKS_STDIN_REQUEST";

/// Hand a request read from stdin to the process that replaces this one
/// under phi512: stdin is read here already, and phi512's ssh to card 0
/// would consume what was left of it on the way (measured: the inner one
/// saw it empty). A file under the run directory, removed by the reader.
fn hand_over(req: &Request) -> Result<(), String> {
    let dir = run_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let p = dir.join(format!("request-{}.json", std::process::id()));
    let text = serde_json::to_string(req).map_err(|e| e.to_string())?;
    std::fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))?;
    std::env::set_var(STDIN_REQUEST, &p);
    Ok(())
}

/// The global options that shape the engine (clap ids). One of them on
/// the command line means the caller wants this process's engine, so
/// `query` does not hand its request to a running server. A value from
/// the environment or a config file does not count: `xks.conf` sets most.
const ENGINE_ARGS: &[&str] = &[
    "subject",
    "ctx",
    "forks",
    "batch",
    "ubatch",
    "threads",
    "verbose",
    "repack",
    "site",
    "no_offload",
    "backend_kind",
    "backend",
    "model",
    "api_key_env",
    "extra",
    "layout",
    "template",
    "conditioning",
    "permutations",
    "debug",
];

fn engine_on_command_line(m: &clap::ArgMatches) -> bool {
    ENGINE_ARGS
        .iter()
        .any(|id| m.value_source(id) == Some(clap::parser::ValueSource::CommandLine))
}

/// The server answering at `XKS_BIND`, else the one holding the cards
/// (it wrote its address into the lock; Mechanical Jev may have started
/// it elsewhere), if one does: its URL and subject.
fn running_server() -> Option<(String, String)> {
    let bind = std::env::var("XKS_BIND").unwrap_or_else(|_| "127.0.0.1:8090".into());
    let first = format!("http://{bind}");
    let mut urls = vec![first.clone()];
    #[cfg(feature = "artichoke")]
    urls.extend(xks::site::cards_server().filter(|u| *u != first));
    urls.into_iter().find_map(|url| {
        let health: Value = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(2))
            .build()
            .get(&format!("{url}/health"))
            .call()
            .ok()?
            .into_json()
            .ok()?;
        let subject = health.get("subject")?.as_str()?.to_string();
        Some((url, subject))
    })
}

/// `query` through the running server: the same request, the answer
/// printed as a local one is.
fn ask_server(url: &str, subject: &str, req: &Request, compare: bool) -> Result<(), String> {
    eprintln!(
        "xks: asking the server at {url} ({subject}); --local loads the subject here instead"
    );
    let mut call = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        // A long session on the 35B is a minute and more.
        .timeout_read(Duration::from_secs(900))
        .build()
        .post(&format!("{url}/v1/systemone"));
    let key = std::env::var("XKS_API_KEYS").ok().and_then(|k| {
        k.split(',')
            .map(str::trim)
            .find(|k| !k.is_empty())
            .map(String::from)
    });
    if let Some(k) = key {
        call = call.set("Authorization", &format!("Bearer {k}"));
    }
    let t0 = Instant::now();
    let ev: Value = match call.send_json(serde_json::to_value(req).map_err(|e| e.to_string())?) {
        Ok(r) => r.into_json().map_err(|e| format!("{url}: {e}"))?,
        Err(ureq::Error::Status(code, r)) => {
            let body: Value = r.into_json().unwrap_or(Value::Null);
            let msg = body
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("(no message)")
                .to_string();
            return Err(format!("the server answered {code}: {msg}"));
        }
        Err(e) => return Err(format!("{url}: {e}")),
    };
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("{}", serde_json::to_string_pretty(&ev).unwrap());
    eprintln!("server: {ms:.1} ms end-to-end");
    if compare {
        compare_with_jev(req)?;
    }
    Ok(())
}

/// `query --compare`: the same request to the hosted Jev, after the local
/// answer.
fn compare_with_jev(req: &Request) -> Result<(), String> {
    let ts = TypeSafe::from_env().ok_or("TYPESAFE_API_KEY not set")?;
    let (remote, rms) = ts.evaluate(req).map_err(|e| e.to_string())?;
    println!("{}", serde_json::to_string_pretty(&remote).unwrap());
    eprintln!("typesafe: {rms:.1} ms end-to-end");
    Ok(())
}

/// The subject, checked before the site is prepared: a path that is not
/// there is said with where to set it, not after a worker restart and
/// llama.cpp's loader.
#[cfg(feature = "artichoke")]
fn check_subject(p: &Path) -> Result<(), String> {
    if p.is_file() {
        return Ok(());
    }
    Err(format!(
        "the subject {} {}: set XKS_SUBJECT=/path/to/model.gguf in xks.local.conf, or pass --subject",
        p.display(),
        if p.exists() {
            "is not a file"
        } else {
            "does not exist"
        }
    ))
}

/// A foreground `serve`'s address, tried before the subject loads and let
/// go at once; the server binds it for real when the engine is up.
fn free_to_bind(bind: &str) -> Result<(), String> {
    std::net::TcpListener::bind(bind).map(drop).map_err(|e| {
        format!(
            "cannot listen on {bind}: {e} (a server already there? xks stop, or --bind another)"
        )
    })
}

/// A request read from `from` (a file's path, or stdin), and its
/// questions checked: the same checks the server makes.
fn parse_request(text: &str, from: &str) -> Result<Request, String> {
    if text.trim().is_empty() {
        return Err(format!(
            "{from}: empty (a JSON request: state and questions)"
        ));
    }
    let r: Request = serde_json::from_str(text).map_err(|e| format!("{from}: {e}"))?;
    xks::protocol::parse_questions(&r.questions).map_err(|e| format!("{from}: {e}"))?;
    Ok(r)
}

fn build_request(
    file: Option<PathBuf>,
    state: Option<String>,
    noul: Vec<String>,
    choice: Vec<String>,
    score: Vec<String>,
) -> Result<Request, String> {
    if let Some(p) = file {
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        return parse_request(&text, &p.display().to_string());
    }
    if state.is_none() && noul.is_empty() && choice.is_empty() && score.is_empty() {
        use std::io::IsTerminal;
        // Under phi512: the outer process read stdin and left it here.
        if let Some(p) = std::env::var_os(STDIN_REQUEST) {
            let p = PathBuf::from(p);
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let _ = std::fs::remove_file(&p);
            return parse_request(&text, "the request on stdin");
        }
        if std::io::stdin().is_terminal() {
            return Err(
                "no request: --file req.json, --state with --noul/--choice/--score, \
                        or a JSON request piped on stdin"
                    .into(),
            );
        }
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)
            .map_err(|e| format!("stdin: {e}"))?;
        return parse_request(&text, "the request on stdin");
    }
    let state = state.ok_or("--state is required with --noul/--choice/--score")?;
    let mut questions = Map::new();
    for s in noul {
        let (id, instr) = split_once(&s, '=')?;
        questions.insert(id.into(), json!({"type": "noul", "instructions": instr}));
    }
    for s in choice {
        let (id, rest) = split_once(&s, '=')?;
        let (instr, opts) = split_once(rest, '|')?;
        let mut criteria = Map::new();
        // An empty entry (`a,,b`, a trailing comma) is no option.
        for o in opts.split(',').filter(|o| !o.trim().is_empty()) {
            let (k, d) = o.split_once(':').unwrap_or((o, ""));
            criteria.insert(
                k.trim().into(),
                if d.trim().is_empty() {
                    Value::Null
                } else {
                    json!(d.trim())
                },
            );
        }
        questions.insert(
            id.into(),
            json!({"type": "choice", "instructions": instr, "criteria": criteria}),
        );
    }
    for s in score {
        let (id, rest) = split_once(&s, '=')?;
        let (instr, levels) = split_once(rest, '|')?;
        let levels: Vec<&str> = levels
            .split(',')
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        questions.insert(
            id.into(),
            json!({"type": "score", "instructions": instr, "criteria": levels}),
        );
    }
    let r = Request {
        model: None,
        state: Value::String(state),
        questions,
    };
    xks::protocol::parse_questions(&r.questions)?;
    Ok(r)
}

fn split_once(s: &str, c: char) -> Result<(&str, &str), String> {
    s.split_once(c)
        .map(|(a, b)| (a.trim(), b.trim()))
        .ok_or_else(|| format!("expected `{c}` in `{s}`"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_a_detached_xks_server_is_recognised() {
        let is = |args: &[&str]| super::is_server_cmdline(args.join("\0").as_bytes());
        assert!(is(&[
            "/r/target/release/xks",
            "serve",
            "--bind",
            "127.0.0.1:8090"
        ]));
        assert!(is(&["/r/target/release/xks", "--site", "cards", "serve"]));
        assert!(is(&[
            "/bin/bash",
            "scripts/phi512.sh",
            "--card",
            "0",
            "/r/target/avx512/release/xks",
            "serve"
        ]));
        // A recycled pid: some other program, or xks doing something else.
        assert!(!is(&["/usr/bin/firefox", "serve"]));
        assert!(!is(&["/r/target/release/xks", "eval", "cases.jsonl"]));
        assert!(!is(&[]));
    }

    #[test]
    fn every_engine_option_is_a_real_argument() {
        use clap::CommandFactory;
        let cmd = super::Cli::command();
        for id in super::ENGINE_ARGS {
            assert!(
                cmd.get_arguments().any(|a| a.get_id() == *id),
                "{id} is not an argument of xks"
            );
        }
    }

    #[test]
    fn a_query_goes_to_a_server_only_without_engine_options() {
        use clap::CommandFactory;
        let on_line = |args: &[&str]| {
            let m = super::Cli::command().try_get_matches_from(args).unwrap();
            super::engine_on_command_line(&m)
        };
        assert!(!on_line(&["xks", "query", "--file", "q.json"]));
        assert!(!on_line(&["xks", "query", "--compare", "--state", "s"]));
        assert!(on_line(&[
            "xks", "--site", "x86", "query", "--file", "q.json"
        ]));
        assert!(on_line(&["xks", "--debug", "query", "--file", "q.json"]));
        assert!(on_line(&["xks", "--subject", "m.gguf", "query"]));
    }

    #[test]
    fn a_request_error_names_where_it_came_from() {
        let dir = std::env::temp_dir().join(format!("xks-req-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = |name: &str, text: &str| {
            let p = dir.join(name);
            std::fs::write(&p, text).unwrap();
            Some(p)
        };
        let from = |f| super::build_request(f, None, vec![], vec![], vec![]);
        let missing = from(Some(dir.join("absent.json"))).unwrap_err();
        assert!(missing.contains("absent.json"), "{missing}");
        let bad = from(file("bad.json", "{\"state\":")).unwrap_err();
        assert!(bad.contains("bad.json"), "{bad}");
        let empty = from(file("empty.json", "\n")).unwrap_err();
        assert!(empty.contains("empty"), "{empty}");
        // The server's own checks, before any subject loads.
        let one_level = from(file(
            "score.json",
            r#"{"state":"s","questions":{"a":{"type":"score","instructions":"i","criteria":["x"]}}}"#,
        ))
        .unwrap_err();
        assert!(one_level.contains("2 to 10 levels"), "{one_level}");
        let none = from(file("none.json", r#"{"state":"s","questions":{}}"#)).unwrap_err();
        assert!(none.contains("no questions"), "{none}");
        assert!(from(file(
            "ok.json",
            r#"{"state":"s","questions":{"a":{"type":"noul","instructions":"i"}}}"#
        ))
        .is_ok());
        let no_options = super::build_request(
            None,
            Some("s".into()),
            vec![],
            vec!["c=pick|".into()],
            vec![],
        );
        let no_options = no_options.unwrap_err();
        assert!(no_options.contains("at least one option"), "{no_options}");
        let trailing = super::build_request(
            None,
            Some("s".into()),
            vec![],
            vec![],
            vec!["r=how much|low,high,".into()],
        )
        .unwrap();
        assert_eq!(
            trailing.questions["r"]["criteria"],
            serde_json::json!(["low", "high"])
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn detach_shows_xks_lines_and_leaves_llama_cpp_in_the_log() {
        assert!(super::relayed("xks: loading m.gguf (21.7 GB)"));
        assert!(super::relayed(
            "xks listening on http://127.0.0.1:8090  (subject s)"
        ));
        assert!(super::relayed("error: the subject /m.gguf does not exist"));
        assert!(!super::relayed(
            "llama_context: n_ctx_seq (16384) > n_ctx_train"
        ));
        assert!(!super::relayed("load: control-looking token"));
    }

    #[test]
    fn detach_relays_whole_lines_once_and_the_last_at_the_end() {
        let log = std::env::temp_dir().join(format!("xks-relay-{}.log", std::process::id()));
        std::fs::write(&log, "xks: site x86\nxks: loa").unwrap();
        let mut r = super::Relay::default();
        r.more(&log);
        assert_eq!(r.seen, "xks: site x86\n".len());
        std::fs::write(&log, "xks: site x86\nxks: loading\nerror: x").unwrap();
        r.more(&log);
        assert_eq!(r.seen, "xks: site x86\nxks: loading\n".len());
        r.rest(&log);
        assert_eq!(r.seen, "xks: site x86\nxks: loading\nerror: x".len());
        std::fs::remove_file(&log).unwrap();
    }

    /// The resident kilobytes of the mapping at `addr`, from smaps.
    #[cfg(feature = "artichoke")]
    fn rss_kb(addr: usize) -> u64 {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        let mut here = false;
        for line in smaps.lines() {
            if let Some((range, _)) = line.split_once(' ') {
                if let Some((lo, _)) = range.split_once('-') {
                    if let Ok(lo) = usize::from_str_radix(lo, 16) {
                        here = lo == addr;
                        continue;
                    }
                }
            }
            if here {
                if let Some(v) = line.strip_prefix("Rss:") {
                    return v.trim().trim_end_matches(" kB").trim().parse().unwrap();
                }
            }
        }
        panic!("no mapping at {addr:#x}");
    }

    #[test]
    #[cfg(feature = "artichoke")]
    fn a_populated_file_mapping_is_left_lazy_only_while_the_subject_loads() {
        use std::os::fd::AsRawFd;
        use std::sync::atomic::Ordering;
        extern "C" {
            fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
        }
        const PROT_READ: i32 = 1;
        const MAP_PRIVATE: i32 = 2;
        const MAP_POPULATE: i32 = 0x8000;
        let len = 4 << 20;
        let path = std::env::temp_dir().join(format!("xks-lazy-{}", std::process::id()));
        std::fs::write(&path, vec![7u8; len]).unwrap();
        let f = std::fs::File::open(&path).unwrap();
        let map = |lazy: bool| {
            xks::artichoke::LAZY_PAGES.store(lazy, Ordering::SeqCst);
            // SAFETY: a read-only private mapping of a file this test owns.
            let p = unsafe {
                super::mmap(
                    std::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE | MAP_POPULATE,
                    f.as_raw_fd(),
                    0,
                )
            };
            xks::artichoke::LAZY_PAGES.store(false, Ordering::SeqCst);
            assert_ne!(p as isize, -1, "mmap failed");
            let kb = rss_kb(p as usize);
            // SAFETY: the mapping made just above.
            unsafe { munmap(p, len) };
            kb
        };
        // As llama.cpp asks, all 4 MiB are read in at once; while the
        // subject loads lazily, none until touched.
        assert_eq!(map(false), 4096);
        assert_eq!(map(true), 0);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn the_peak_is_read_from_vmhwm() {
        let status = "Name:\txks\nVmPeak:\t 9999999 kB\nVmHWM:\t 2097152 kB\nVmRSS:\t 1048576 kB\n";
        assert_eq!(super::peak_gib(status), Some(2.0));
        assert_eq!(super::peak_gib("Name:\txks\n"), None);
    }
}
