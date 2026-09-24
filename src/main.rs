//! The `xks` command line. See main.md.

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use serde_json::{json, Map, Value};

use xks::backend::llamacpp::LlamaServer;
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
    /// Chat template of the subject: chatml | gemma | llama3 | raw
    #[arg(long, env = "XKS_TEMPLATE", default_value = "chatml")]
    template: String,
    /// Conditioning: per-bucket temperatures (from `xks condition`).
    #[arg(long, alias = "calibration", env = "XKS_CONDITIONING")]
    conditioning: Option<PathBuf>,
    /// Cyclic option rotations to average for `choice` (position-bias control).
    #[arg(long, default_value_t = 1)]
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
    },
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
fn artichoke(cli: &Cli) -> Result<xks::artichoke::Artichoke, String> {
    let subject = cli
        .subject
        .clone()
        .ok_or("--subject (or XKS_SUBJECT) is required with --backend-kind artichoke")?;
    let mut o = xks::artichoke::Options::new(subject);
    o.n_ctx = cli.ctx;
    o.forks = cli.forks.max(1);
    o.n_batch = cli.batch;
    o.n_ubatch = cli.ubatch;
    if let Some(t) = cli.threads {
        o.threads = t;
    }
    o.backend_dir = std::env::var_os("XKS_BACKEND_DIR").map(PathBuf::from);
    o.verbose = cli.verbose;
    if let Some(r) = cli.repack {
        o.repack = r;
    }
    xks::artichoke::Artichoke::open(&o)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
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
    #[cfg(feature = "artichoke")]
    if let Cmd::Polygraph { file, rows, limit } = &cli.cmd {
        let engine = artichoke(&cli)?;
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
    let (scorer, template): (Box<dyn Scorer>, Template) = match cli.backend_kind.as_str() {
        #[cfg(feature = "artichoke")]
        "artichoke" => (Box::new(artichoke(&cli)?), template),
        "bluebird" | "llamacpp" => (Box::new(LlamaServer::new(cli.backend.clone())), template),
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
        Cmd::Eval { file, rows } => {
            let cases = eval::load_cases(&file)?;
            let t0 = std::time::Instant::now();
            let (r, failed) = eval::run(&judge, &cases).map_err(|e| e.to_string())?;
            let wall = t0.elapsed().as_secs_f64();
            let m = eval::metrics(&r, failed, &judge.cfg.calibration);
            let mut v = serde_json::to_value(&m).unwrap();
            v["subject"] = json!(judge.scorer.model_name());
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
        Cmd::Replay { .. } => Ok(()),
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
