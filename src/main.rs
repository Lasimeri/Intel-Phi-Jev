//! The `xks` command line. See main.md.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use serde_json::{json, Map, Value};

use xks::backend::bluebird::Bluebird;
use xks::backend::openai::OpenAiChat;
use xks::backend::typesafe::TypeSafe;
use xks::backend::Scorer;
use xks::eval;
use xks::judge::{Judge, JudgeConfig};
use xks::prompt::Template;
use xks::protocol::Request;
use xks::score::Calibration;
use xks::server::{serve, ServerConfig};

#[derive(Parser)]
#[command(
    name = "xks",
    version,
    about = "XKEYSCORE for Jev: typed System One judgments, one prefill per session"
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
    },
    /// One query. Reads a JSON request from --file or stdin, or builds one
    /// from --state and --noul/--choice/--score flags.
    #[command(alias = "ask")]
    Query {
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
    let subject = cli
        .subject
        .clone()
        .ok_or("--subject (or XKS_SUBJECT) is required with --backend-kind artichoke")?;
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
    Ok((xks::artichoke::Artichoke::open(&o)?, placed))
}

fn main() {
    // Defaults from xks.conf and friends, before clap reads the environment
    // and before any thread exists.
    xks::config::load();
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `--backend-kind jev`: the real Jev, through TypeSafe's API. Needs
/// `TYPESAFE_API_KEY` (the environment, or `xks.local.conf`).
fn jev(cmd: &Cmd) -> Result<(), String> {
    let ts = TypeSafe::from_env().ok_or(
        "the real Jev needs TYPESAFE_API_KEY: create a key at console.typesafe.ai and put \
         TYPESAFE_API_KEY=... in xks.local.conf (not tracked) or the environment",
    )?;
    match cmd {
        Cmd::Query {
            file,
            state,
            noul,
            choice,
            score,
            ..
        } => {
            let req = build_request(
                file.clone(),
                state.clone(),
                noul.clone(),
                choice.clone(),
                score.clone(),
            )?;
            let (ev, ms) = ts.evaluate(&req).map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(&ev).unwrap());
            eprintln!("jev ({}): {ms:.1} ms end-to-end", ev.model);
            Ok(())
        }
        Cmd::Eval { file, rows, limit } => {
            let mut cases = eval::load_cases(file)?;
            if let Some(n) = limit {
                cases.truncate(*n);
            }
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
    if let Ok(pid) = std::fs::read_to_string(&pidfile) {
        if Path::new(&format!("/proc/{}", pid.trim())).exists() {
            return Err(format!(
                "a server is already running (pid {}); xks stop",
                pid.trim()
            ));
        }
    }
    let log = dir.join("serve.log");
    let out = std::fs::File::create(&log).map_err(|e| e.to_string())?;
    let err = out.try_clone().map_err(|e| e.to_string())?;
    let args: Vec<_> = std::env::args_os()
        .skip(1)
        .filter(|a| a != "--detach")
        .collect();
    let child = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
        .args(args)
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .process_group(0)
        .spawn()
        .map_err(|e| e.to_string())?;
    std::fs::write(&pidfile, child.id().to_string()).map_err(|e| e.to_string())?;
    let url = format!("http://{bind}");
    eprintln!(
        "xks: server pid {} starting, log {}",
        child.id(),
        log.display()
    );
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(900) {
        if !Path::new(&format!("/proc/{}", child.id())).exists() {
            return Err(format!("the server exited; see {}", log.display()));
        }
        if ureq::get(&format!("{url}/health")).call().is_ok() {
            println!("{url}/v1/systemone");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!(
        "the server did not answer in 900 s; see {}",
        log.display()
    ))
}

/// `stop`: end a detached server and give the cards their memory back.
fn stop() -> Result<(), String> {
    let pidfile = run_dir().join("serve.pid");
    if let Ok(pid) = std::fs::read_to_string(&pidfile) {
        let pid = pid.trim().to_string();
        let proc_dir = format!("/proc/{pid}");
        if Path::new(&proc_dir).exists() {
            let _ = Command::new("kill").arg(&pid).status();
            let t0 = Instant::now();
            while Path::new(&proc_dir).exists() && t0.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_millis(200));
            }
            eprintln!("xks: server {pid} stopped");
        }
        let _ = std::fs::remove_file(&pidfile);
    } else {
        eprintln!("xks: no detached server");
    }
    #[cfg(feature = "artichoke")]
    {
        let cards = xks::site::card_windows();
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
    let cli = Cli::parse();
    let template: Template = cli.template.parse()?;
    let conditioning: Calibration = match &cli.conditioning {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?)
            .map_err(|e| format!("conditioning: {e}"))?,
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
            let cards = xks::site::card_windows();
            xks::site::release(&cards)?;
            eprintln!("xks: workers stopped and huge pages released on cards {cards:?}");
            return Ok(());
        }
        _ => {}
    }
    #[cfg(feature = "artichoke")]
    if let Cmd::Polygraph { file, rows, limit } = &cli.cmd {
        let (engine, _) = artichoke(&cli)?;
        let cases = eval::load_cases(file)?;
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
        return jev(&cli.cmd);
    }
    let mut site_name = "remote";
    let (scorer, template): (Box<dyn Scorer>, Template) = match cli.backend_kind.as_str() {
        #[cfg(feature = "artichoke")]
        "artichoke" => {
            let (engine, placed) = artichoke(&cli)?;
            site_name = placed.site.name();
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
        layout: cli.layout.parse()?,
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
            detach: _,
        } => {
            let api_keys = api_keys
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            serve(
                judge,
                ServerConfig {
                    bind,
                    api_keys,
                    kill_date: (kill_date > 0).then(|| Duration::from_secs(kill_date)),
                },
            )
        }
        Cmd::Query {
            file,
            state,
            noul,
            choice,
            score,
            compare,
        } => {
            let req = build_request(file, state, noul, choice, score)?;
            let t0 = std::time::Instant::now();
            let ev = judge.evaluate(&req).map_err(|e| e.to_string())?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!("{}", serde_json::to_string_pretty(&ev).unwrap());
            eprintln!("local: {:.1} ms end-to-end", ms);
            if compare {
                let ts = TypeSafe::from_env().ok_or("TYPESAFE_API_KEY not set")?;
                let (remote, rms) = ts.evaluate(&req).map_err(|e| e.to_string())?;
                println!("{}", serde_json::to_string_pretty(&remote).unwrap());
                eprintln!("typesafe: {:.1} ms end-to-end", rms);
            }
            Ok(())
        }
        Cmd::Eval { file, rows, limit } => {
            let mut cases = eval::load_cases(&file)?;
            if let Some(n) = limit {
                cases.truncate(n);
            }
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
        Cmd::Condition { file, out } => {
            let cases = eval::load_cases(&file)?;
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
    }
}

fn build_request(
    file: Option<PathBuf>,
    state: Option<String>,
    noul: Vec<String>,
    choice: Vec<String>,
    score: Vec<String>,
) -> Result<Request, String> {
    if let Some(p) = file {
        let text = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
        return serde_json::from_str(&text).map_err(|e| e.to_string());
    }
    if state.is_none() && noul.is_empty() && choice.is_empty() && score.is_empty() {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)
            .map_err(|e| e.to_string())?;
        return serde_json::from_str(&text).map_err(|e| e.to_string());
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
        for o in opts.split(',') {
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
        let levels: Vec<&str> = levels.split(',').map(str::trim).collect();
        questions.insert(
            id.into(),
            json!({"type": "score", "instructions": instr, "criteria": levels}),
        );
    }
    Ok(Request {
        model: None,
        state: Value::String(state),
        questions,
    })
}

fn split_once(s: &str, c: char) -> Result<(&str, &str), String> {
    s.split_once(c)
        .map(|(a, b)| (a.trim(), b.trim()))
        .ok_or_else(|| format!("expected `{c}` in `{s}`"))
}
