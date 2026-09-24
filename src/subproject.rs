//! Subprojects: every experiment in this repository as one command,
//! `xks subproject run NN` (or `all`), after MKULTRA's numbered subprojects,
//! each with its own report. A subproject runs `xks` itself as child
//! processes (one process at a time may hold the cards, and each site
//! wants its own), starts and stops llama-server where it compares against
//! it, and writes one JSON record to `docs/subprojects/results/`: the git
//! revision, the time, the subject, the configuration and what was
//! measured. The prose that reads a record is `docs/subprojects/NN-*.md`.
//! See subproject.md.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

pub struct Subproject {
    pub id: &'static str,
    pub name: &'static str,
    pub what: &'static str,
    run: fn(&Ctx) -> Result<Value, String>,
}

pub const ALL: &[Subproject] = &[
    Subproject {
        id: "01",
        name: "bluebird-baseline",
        what: "BLUEBIRD: stock llama-server (x86) re-prefilling every question, dev_tasks",
        run: s01_bluebird,
    },
    Subproject {
        id: "02",
        name: "polygraph",
        what: "ARTICHOKE's fork read three ways (forked, split, control) on the 35B, the copy isolated with one fork, and the dense 0.5B as the noise floor",
        run: s02_polygraph,
    },
    Subproject {
        id: "03",
        name: "dev-eval-sites",
        what: "ARTICHOKE on dev_tasks at the x86 site and the cards site, corroborated",
        run: s03_dev_sites,
    },
    Subproject {
        id: "04",
        name: "long-sessions",
        what: "Long real sessions: BLUEBIRD against ARTICHOKE (x86), ARTICHOKE on the cards with the payload's ledger, corroborated",
        run: s04_long,
    },
    Subproject {
        id: "05",
        name: "wide-choice-trie",
        what: "A 30-option Choice (multi-token labels) read as a trie of forks against brute force",
        run: s05_trie,
    },
    Subproject {
        id: "06",
        name: "avx512-parity",
        what: "The AVX-512 build under phi512 (with the payload) against the x86 reference, dense 0.5B",
        run: s06_avx512,
    },
];

pub struct Ctx {
    exe: PathBuf,
    repo: PathBuf,
    logs: PathBuf,
    subject: String,
    small: String,
}

impl Ctx {
    pub fn new() -> Result<Self, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let repo = crate::config::repo_root()
            .ok_or("run from the repository's build (target/release/xks)")?;
        let logs = repo.join("target/subprojects");
        std::fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
        let var = |k: &str| std::env::var(k).map_err(|_| format!("{k} is not set (xks.conf)"));
        Ok(Self {
            exe,
            repo,
            logs,
            subject: var("XKS_SUBJECT")?,
            small: var("XKS_SUBJECT_SMALL")?,
        })
    }

    fn example(&self, name: &str) -> String {
        self.repo.join("examples").join(name).display().to_string()
    }

    fn log(&self, name: &str) -> PathBuf {
        self.logs.join(name)
    }

    /// Run `xks` with `args`; stdout is its JSON result, stderr goes to
    /// `target/subprojects/<log>`.
    fn xks(&self, log: &str, args: &[&str], env: &[(&str, &str)]) -> Result<Value, String> {
        let log = self.log(log);
        let err = std::fs::File::create(&log).map_err(|e| e.to_string())?;
        eprintln!("  xks {}", args.join(" "));
        let t0 = Instant::now();
        let out = Command::new(&self.exe)
            .args(args)
            .envs(env.iter().copied())
            .stdout(Stdio::piped())
            .stderr(err)
            .output()
            .map_err(|e| e.to_string())?;
        eprintln!(
            "    {:.1} s, log {}",
            t0.elapsed().as_secs_f64(),
            log.display()
        );
        if !out.status.success() {
            return Err(format!(
                "xks {} failed ({}); see {}",
                args.join(" "),
                out.status,
                log.display()
            ));
        }
        serde_json::from_slice(&out.stdout)
            .map_err(|e| format!("xks {}: output is not JSON: {e}", args.join(" ")))
    }
}

/// A llama-server child, stopped when dropped.
struct Bluebird(Child);

impl Drop for Bluebird {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn bluebird(ctx: &Ctx, log: &str, subject: &str) -> Result<(Bluebird, String), String> {
    let server =
        std::env::var("XKS_LLAMA_SERVER").map_err(|_| "XKS_LLAMA_SERVER is not set (xks.conf)")?;
    let port = std::env::var("XKS_BLUEBIRD_PORT").unwrap_or_else(|_| "8089".into());
    let url = format!("http://127.0.0.1:{port}");
    let log = ctx.log(log);
    let err = std::fs::File::create(&log).map_err(|e| e.to_string())?;
    eprintln!("  llama-server {subject} on {url}");
    let child = Command::new(&server)
        .args(["-m", subject, "--host", "127.0.0.1", "--port", &port])
        .args(["-t", "16", "-np", "1", "-c", "16384"])
        .stdout(Stdio::null())
        .stderr(err)
        .spawn()
        .map_err(|e| format!("{server}: {e}"))?;
    let bb = Bluebird(child);
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(900) {
        if let Ok(r) = ureq::get(&format!("{url}/health")).call() {
            if r.status() == 200 {
                return Ok((bb, url));
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err(format!(
        "llama-server did not come up; see {}",
        log.display()
    ))
}

fn rows(ctx: &Ctx, name: &str) -> String {
    ctx.log(name).display().to_string()
}

fn s01_bluebird(ctx: &Ctx) -> Result<Value, String> {
    let (_bb, url) = bluebird(ctx, "01-llama-server.log", &ctx.subject)?;
    let r = rows(ctx, "01-bluebird-rows.jsonl");
    let eval = ctx.xks(
        "01-bluebird.log",
        &[
            "--backend-kind",
            "bluebird",
            "--backend",
            &url,
            "eval",
            &ctx.example("dev_tasks.jsonl"),
            "--rows",
            &r,
        ],
        &[],
    )?;
    Ok(json!({"subject": ctx.subject, "site": "x86 (llama-server)", "eval": eval}))
}

fn s02_polygraph(ctx: &Ctx) -> Result<Value, String> {
    let dev = ctx.example("dev_tasks.jsonl");
    let batched = ctx.xks(
        "02-batched.log",
        &["--site", "x86", "polygraph", &dev, "--limit", "6"],
        &[],
    )?;
    let one = ctx.xks(
        "02-one-fork.log",
        &[
            "--site",
            "x86",
            "--forks",
            "1",
            "polygraph",
            &dev,
            "--limit",
            "2",
        ],
        &[],
    )?;
    let floor = ctx.xks(
        "02-small.log",
        &[
            "--site",
            "x86",
            "--subject",
            &ctx.small,
            "polygraph",
            &dev,
            "--limit",
            "6",
        ],
        &[],
    )?;
    Ok(
        json!({"subject": ctx.subject, "small": ctx.small, "site": "x86",
        "batched_forks": batched, "one_fork": one, "dense_small": floor}),
    )
}

fn s03_dev_sites(ctx: &Ctx) -> Result<Value, String> {
    let dev = ctx.example("dev_tasks.jsonl");
    let (a, b) = (
        rows(ctx, "03-x86-rows.jsonl"),
        rows(ctx, "03-cards-rows.jsonl"),
    );
    let x86 = ctx.xks(
        "03-x86.log",
        &["--site", "x86", "eval", &dev, "--rows", &a],
        &[],
    )?;
    let cards = ctx.xks(
        "03-cards.log",
        &["--site", "cards", "eval", &dev, "--rows", &b],
        &[],
    )?;
    let cor = ctx.xks("03-corroborate.log", &["corroborate", &a, &b], &[])?;
    Ok(json!({"subject": ctx.subject, "x86": x86, "cards": cards, "corroborate": cor}))
}

fn s04_long(ctx: &Ctx) -> Result<Value, String> {
    let long = ctx.example("long_sessions.jsonl");
    let bb_rows = rows(ctx, "04-bluebird-rows.jsonl");
    let bb = {
        let (_bb, url) = bluebird(ctx, "04-llama-server.log", &ctx.subject)?;
        ctx.xks(
            "04-bluebird.log",
            &[
                "--backend-kind",
                "bluebird",
                "--backend",
                &url,
                "eval",
                &long,
                "--rows",
                &bb_rows,
            ],
            &[],
        )?
    };
    let (a, b) = (
        rows(ctx, "04-x86-rows.jsonl"),
        rows(ctx, "04-cards-rows.jsonl"),
    );
    let x86 = ctx.xks(
        "04-x86.log",
        &["--site", "x86", "eval", &long, "--rows", &a],
        &[],
    )?;
    let cards = ctx.xks(
        "04-cards.log",
        &["--site", "cards", "eval", &long, "--rows", &b],
        &[("PHI_GGML_VERBOSE", "1")],
    )?;
    let ledger = ctx.xks(
        "04-ledger.log",
        &["ledger", &ctx.log("04-cards.log").display().to_string()],
        &[],
    )?;
    let cor = ctx.xks("04-corroborate.log", &["corroborate", &a, &b], &[])?;
    Ok(
        json!({"subject": ctx.subject, "bluebird": bb, "artichoke_x86": x86,
        "artichoke_cards": cards, "ledger": ledger, "corroborate_x86_cards": cor}),
    )
}

fn s05_trie(ctx: &Ctx) -> Result<Value, String> {
    let wide = ctx.example("wide_choice.jsonl");
    let p = ctx.xks(
        "05-trie.log",
        &["--site", "x86", "--subject", &ctx.small, "polygraph", &wide],
        &[],
    )?;
    Ok(json!({"subject": ctx.small, "site": "x86", "polygraph": p}))
}

fn s06_avx512(ctx: &Ctx) -> Result<Value, String> {
    let dev = ctx.example("dev_tasks.jsonl");
    let (a, b) = (
        rows(ctx, "06-x86-rows.jsonl"),
        rows(ctx, "06-avx512-rows.jsonl"),
    );
    let x86 = ctx.xks(
        "06-x86.log",
        &[
            "--site",
            "x86",
            "--subject",
            &ctx.small,
            "eval",
            &dev,
            "--limit",
            "2",
            "--rows",
            &a,
        ],
        &[],
    )?;
    let avx = ctx.xks(
        "06-avx512.log",
        &[
            "--site",
            "avx512",
            "--subject",
            &ctx.small,
            "eval",
            &dev,
            "--limit",
            "2",
            "--rows",
            &b,
        ],
        &[],
    )?;
    let cor = ctx.xks("06-corroborate.log", &["corroborate", &a, &b], &[])?;
    Ok(json!({"subject": ctx.small, "x86": x86, "avx512": avx, "corroborate": cor}))
}

/// `YYYY-MM-DDTHH:MM:SSZ` for a UNIX time (civil-from-days, no calendar crate).
pub fn utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

fn git(repo: &Path, args: &[&str]) -> String {
    Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn hostname() -> String {
    let mut s = String::new();
    let _ =
        std::fs::File::open("/proc/sys/kernel/hostname").and_then(|mut f| f.read_to_string(&mut s));
    s.trim().to_string()
}

/// Run one subproject and write its record.
pub fn run(ctx: &Ctx, sp: &Subproject) -> Result<PathBuf, String> {
    eprintln!("subproject {} {}: {}", sp.id, sp.name, sp.what);
    let t0 = Instant::now();
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let results = (sp.run)(ctx)?;
    let record = json!({
        "subproject": sp.id,
        "name": sp.name,
        "what": sp.what,
        "utc": utc(started),
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "git": git(&ctx.repo, &["rev-parse", "--short", "HEAD"]),
        "dirty": !git(&ctx.repo, &["status", "--porcelain", "--untracked-files=no"]).is_empty(),
        "host": hostname(),
        "results": results,
    });
    let dir = ctx.repo.join("docs/subprojects/results");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}-{}.json", sp.id, sp.name));
    std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap() + "\n")
        .map_err(|e| e.to_string())?;
    eprintln!(
        "subproject {} done in {:.0} s: {}",
        sp.id,
        t0.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(path)
}

pub fn find(which: &str) -> Vec<&'static Subproject> {
    if which == "all" {
        return ALL.iter().collect();
    }
    ALL.iter()
        .filter(|s| s.id == which || s.name == which || format!("{}-{}", s.id, s.name) == which)
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn utc_formats() {
        assert_eq!(super::utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(super::utc(1_790_000_000), "2026-09-21T14:13:20Z");
    }
}
