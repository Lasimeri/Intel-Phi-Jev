//! `xks doctor`: what this machine has of what `xks` needs, from the
//! llama.cpp build down to the cards, and what is missing, with the fix
//! for each. It reads files, sockets and one `/health`; it starts nothing,
//! prepares no site and reaches no card over ssh. `--fix` does the fixes
//! that are a build or a link (the payload built, `xks` linked onto PATH),
//! never a card, a download or the system. See doctor.md.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::site;

/// How a finding bears on answering a question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// In place.
    Ok,
    /// Worth knowing; nothing waits on it.
    Note,
    /// A question cannot be answered until it is fixed.
    Missing,
}

/// One line of the report, and its fix when there is one.
#[derive(Debug)]
pub struct Finding {
    pub level: Level,
    pub what: &'static str,
    pub detail: String,
    pub fix: Option<String>,
}

/// Everything `doctor` found, in the order it looked.
#[derive(Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    /// The site `auto` (or the configured one) resolves to.
    pub site: Option<&'static str>,
}

impl Report {
    fn add(&mut self, level: Level, what: &'static str, detail: String, fix: Option<String>) {
        self.findings.push(Finding {
            level,
            what,
            detail,
            fix,
        });
    }

    fn ok(&mut self, what: &'static str, detail: impl Into<String>) {
        self.add(Level::Ok, what, detail.into(), None);
    }

    fn note(&mut self, what: &'static str, detail: impl Into<String>, fix: Option<String>) {
        self.add(Level::Note, what, detail.into(), fix);
    }

    fn missing(&mut self, what: &'static str, detail: impl Into<String>, fix: impl Into<String>) {
        self.add(Level::Missing, what, detail.into(), Some(fix.into()));
    }

    /// Whether a question can be answered: nothing missing.
    pub fn ready(&self) -> bool {
        !self.findings.iter().any(|f| f.level == Level::Missing)
    }

    /// The report as printed: one line per finding, its fix under it.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for f in &self.findings {
            let tag = match f.level {
                Level::Ok => "ok  ",
                Level::Note => "note",
                Level::Missing => "MISS",
            };
            out.push_str(&format!("  {tag}  {:<12} {}\n", f.what, f.detail));
            if let Some(fix) = &f.fix {
                out.push_str(&format!("        {:<12} fix: {fix}\n", ""));
            }
        }
        let missing = self
            .findings
            .iter()
            .filter(|f| f.level == Level::Missing)
            .count();
        if missing == 0 {
            out.push_str(&format!(
                "ready: xks can answer (site {})\n",
                self.site.unwrap_or("?")
            ));
        } else {
            out.push_str(&format!("not ready: {missing} to fix (MISS above)\n"));
        }
        out
    }
}

/// What is at a link's place, against the file it should point at.
#[derive(Debug, PartialEq, Eq)]
pub enum LinkState {
    /// Nothing there.
    Absent,
    /// A link to `target` (through any links in between).
    Ours,
    /// A link to something else that exists: another checkout's build.
    Other(PathBuf),
    /// A link to nothing.
    Dangling(PathBuf),
    /// A file or directory, not a link: never replaced.
    File,
}

pub fn link_state(link: &Path, target: &Path) -> LinkState {
    let Ok(meta) = std::fs::symlink_metadata(link) else {
        return LinkState::Absent;
    };
    if !meta.file_type().is_symlink() {
        return LinkState::File;
    }
    let to = std::fs::read_link(link).unwrap_or_default();
    match (link.canonicalize(), target.canonicalize()) {
        (Ok(a), Ok(b)) if a == b => LinkState::Ours,
        (Err(_), _) => LinkState::Dangling(to),
        _ => LinkState::Other(to),
    }
}

/// Point `link` at `target`: created when absent, replaced when it is a
/// dangling link or a link to another build, left alone (an error) when a
/// file is there. What was done, in words.
pub fn place_link(link: &Path, target: &Path) -> Result<String, String> {
    let at = |e: std::io::Error| format!("{}: {e}", link.display());
    let before = link_state(link, target);
    match &before {
        LinkState::Ours => return Ok(format!("{} already links here", link.display())),
        LinkState::File => {
            return Err(format!(
                "{} is a file, not a link: left alone (move it, then run this again)",
                link.display()
            ))
        }
        LinkState::Absent => {
            if let Some(d) = link.parent() {
                std::fs::create_dir_all(d).map_err(at)?;
            }
        }
        LinkState::Other(_) | LinkState::Dangling(_) => std::fs::remove_file(link).map_err(at)?,
    }
    std::os::unix::fs::symlink(target, link).map_err(at)?;
    Ok(match before {
        LinkState::Other(p) => format!(
            "{} now links here (it linked {})",
            link.display(),
            p.display()
        ),
        LinkState::Dangling(p) => format!(
            "{} now links here (it linked {}, which is gone)",
            link.display(),
            p.display()
        ),
        _ => format!("{} links here", link.display()),
    })
}

/// The first executable called `name` in a `PATH` directory.
pub fn on_path(name: &str) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|d| d.join(name))
        .find(|p| {
            p.metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        })
}

/// Whether `dir` is one of `PATH`'s directories.
fn path_has(dir: &Path) -> bool {
    std::env::var_os("PATH").is_some_and(|p| std::env::split_paths(&p).any(|d| d == dir))
}

/// `MemTotal` and `MemAvailable` in bytes.
fn memory() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb = |key: &str| -> Option<u64> {
        text.lines()
            .find_map(|l| l.strip_prefix(key))?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse::<u64>()
            .ok()
            .map(|k| k * 1024)
    };
    Some((kb("MemTotal:")?, kb("MemAvailable:")?))
}

fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

/// The last `n` non-empty lines of a build's output.
fn tail(bytes: &[u8], n: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n        ")
}

/// Everything, in the order a question needs it. `prefix` is where
/// `--fix` links `xks` (`PREFIX/bin/xks`).
pub fn run(fix: bool, prefix: &Path) -> Report {
    let mut r = Report::default();
    let exe = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .unwrap_or_default();

    // xks itself, and whether a shell finds it.
    let bin = prefix.join("bin");
    let found = on_path("xks").and_then(|p| p.canonicalize().ok());
    if found.as_deref() == Some(exe.as_path()) {
        r.ok("xks", format!("{} (on PATH)", exe.display()));
    } else if fix {
        match place_link(&bin.join("xks"), &exe) {
            Ok(done) => r.ok("xks", done),
            Err(e) => r.note("xks", e, None),
        }
    } else {
        let also = match found {
            Some(other) => format!(" (PATH finds {} first)", other.display()),
            None => String::new(),
        };
        r.note(
            "xks",
            format!("{} is not on PATH{also}", exe.display()),
            Some("make install, or xks doctor --fix".into()),
        );
    }
    if !path_has(&bin) {
        r.note(
            "PATH",
            format!(
                "{} is not on PATH, so a link there is not found",
                bin.display()
            ),
            Some(format!(
                "fish: fish_add_path {0}; bash: export PATH=\"{0}:$PATH\" in ~/.bashrc",
                bin.display()
            )),
        );
    }

    // Configuration: which files were read, and the values that are parsed.
    let read: Vec<String> = crate::config::files()
        .into_iter()
        .filter(|f| f.is_file())
        .map(|f| f.display().to_string())
        .collect();
    r.ok("config", format!("read {}", read.join(", ")));
    let var = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    for (key, value, parsed) in [
        (
            "XKS_TEMPLATE",
            var("XKS_TEMPLATE", "chatml"),
            var("XKS_TEMPLATE", "chatml")
                .parse::<crate::prompt::Template>()
                .err(),
        ),
        (
            "XKS_LAYOUT",
            var("XKS_LAYOUT", "letters"),
            var("XKS_LAYOUT", "letters")
                .parse::<crate::prompt::Layout>()
                .err(),
        ),
    ] {
        if let Some(e) = parsed {
            r.missing(
                "config",
                format!("{key}={value}: {e}"),
                "correct it in xks.local.conf",
            );
        }
    }

    // The engine's llama.cpp build (the binary would not have started
    // without its libraries; this names where they are).
    let llama = PathBuf::from(env!("XKS_LLAMA_BUILD_DIR"));
    if llama.join("libllama.so").exists() {
        r.ok("llama.cpp", llama.display().to_string());
    } else {
        r.note(
            "llama.cpp",
            format!("{} has no libllama.so (the libraries come from XKS_BACKEND_DIR or the loader's path)", llama.display()),
            None,
        );
    }

    // The subject.
    let subject = std::env::var_os("XKS_SUBJECT").map(PathBuf::from);
    let subject_bytes = match &subject {
        None => {
            r.missing(
                "subject",
                "XKS_SUBJECT is not set",
                "XKS_SUBJECT=/path/to/model.gguf in xks.local.conf",
            );
            None
        }
        Some(p) if !p.is_file() => {
            r.missing(
                "subject",
                format!("{} does not exist", p.display()),
                "XKS_SUBJECT=/path/to/model.gguf in xks.local.conf (xks.conf names the 35B)",
            );
            None
        }
        Some(p) => {
            let bytes = std::fs::metadata(p).map_or(0, |m| m.len());
            r.ok(
                "subject",
                format!(
                    "{} ({})",
                    p.file_name().map_or_else(
                        || p.display().to_string(),
                        |n| n.to_string_lossy().into_owned()
                    ),
                    gib(bytes)
                ),
            );
            Some(bytes)
        }
    };

    // The site, and what it needs.
    let requested = var("XKS_SITE", "auto");
    let cards = site::card_windows();
    // Asked for by name, or `auto` with a card up (it then resolves to the
    // cards, and a missing payload stops the load).
    let needs_cards = matches!(requested.as_str(), "cards" | "phi" | "avx512")
        || (requested == "auto" && !cards.is_empty());
    match site::resolve(&requested) {
        Ok(s) => r.site = Some(s.name()),
        Err(e) => r.missing(
            "site",
            e,
            "XKS_SITE=auto (or x86, cards, avx512) in xks.local.conf",
        ),
    }
    let need_level = if needs_cards {
        Level::Missing
    } else {
        Level::Note
    };

    // The cards' side: the sibling repository, its payload, the stack.
    let root = site::sibling_root();
    if root.join("scripts/phi-vpu.sh").exists() {
        r.ok("AVX-512", root.display().to_string());
        match site::payload(&root) {
            Ok(lib) => r.ok(
                "payload",
                lib.strip_prefix(&root)
                    .unwrap_or(&lib)
                    .display()
                    .to_string(),
            ),
            Err(_) if fix => {
                let out = Command::new("cargo")
                    .args(["build", "--release", "-p", "phi-ggml"])
                    .current_dir(root.join("host"))
                    .output();
                match out {
                    Ok(o) if o.status.success() => {
                        r.ok("payload", "built (host/target/release/libggml_phi.so)")
                    }
                    Ok(o) => r.add(
                        need_level,
                        "payload",
                        format!("the build failed:\n        {}", tail(&o.stderr, 6)),
                        Some(format!(
                            "cd \"{}/host\" && cargo build --release -p phi-ggml",
                            root.display()
                        )),
                    ),
                    Err(e) => r.add(need_level, "payload", format!("cargo: {e}"), None),
                }
            }
            Err(e) => r.add(
                need_level,
                "payload",
                "not built".into(),
                Some(format!(
                    "{} (or xks doctor --fix)",
                    e.trim_start_matches("the payload is not built: ")
                )),
            ),
        }
    } else {
        r.add(
            need_level,
            "AVX-512",
            format!("no Intel-Phi-AVX512 checkout (looked for {})", root.display()),
            Some("clone github.com/Lasimeri/Intel-Phi-AVX512 next to this checkout, or set PHI_AVX512_ROOT".into()),
        );
    }
    match on_path("phi") {
        Some(p) => r.ok("stack", format!("phi at {}", p.display())),
        None => r.add(
            need_level,
            "stack",
            "no `phi` on PATH (Intel-Phi-3120A's command that starts the cards)".into(),
            Some("its installer, or link its scripts/phi.sh as ~/.local/bin/phi".into()),
        ),
    }
    let debug: Vec<&str> = ["phictl", "phitop"]
        .into_iter()
        .filter(|n| {
            std::fs::read_link(bin.join(n))
                .is_ok_and(|t| t.to_string_lossy().contains("/target/debug/"))
        })
        .collect();
    if !debug.is_empty() {
        r.note(
            "stack",
            format!(
                "{} on PATH link debug builds (the stack's own choice; left alone)",
                debug.join(" and ")
            ),
            None,
        );
    }

    // The cards themselves.
    if cards.is_empty() {
        let detail = if needs_cards {
            format!("no card is up, and XKS_SITE={requested} needs them")
        } else {
            "no card is up: `auto` runs on this host alone (x86)".into()
        };
        r.add(need_level, "cards", detail, Some("phi up all".into()));
    } else {
        let list: Vec<String> = cards.iter().map(u32::to_string).collect();
        r.ok(
            "cards",
            format!(
                "{} up; site {} resolves to {}",
                list.join(", "),
                requested,
                r.site.unwrap_or("?")
            ),
        );
    }
    if let Some(who) = site::cards_holder() {
        r.note("cards", format!("in use by {who}"), None);
    }
    let held = site::xks_workers();
    if !held.is_empty() && site::cards_holder().is_none() {
        r.note(
            "cards",
            format!("xks's workers still hold huge pages on {held:?} (4.7 GiB a card) with nothing using them"),
            Some("xks release".into()),
        );
    }

    // The avx512 site's pieces: only needed when it is asked for.
    let avx_bin = std::env::var_os("XKS_AVX512_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            exe.parent()
                .and_then(Path::parent)
                .map_or_else(PathBuf::new, |t| t.join("avx512/release/xks"))
        });
    if avx_bin.is_file() {
        r.ok(
            "avx512 site",
            format!("its build is at {}", avx_bin.display()),
        );
    } else if requested == "avx512" {
        r.missing(
            "avx512 site",
            "XKS_SITE=avx512 but its build is missing",
            "make build-avx512",
        );
    } else {
        r.note(
            "avx512 site",
            "not built (only XKS_SITE=avx512 needs it)",
            Some("make build-avx512".into()),
        );
    }

    // Memory for the subject.
    if let (Some(bytes), Some((total, avail))) = (subject_bytes, memory()) {
        let on = r.site.unwrap_or("x86");
        let detail = format!(
            "this host has {} ({} available); the subject is {}: the x86 site maps all of it, the cards site held 12.5 to 13.3 GiB of the 35B at its peak (subprojects 03, 07)",
            gib(total),
            gib(avail),
            gib(bytes)
        );
        if on == "x86" && bytes > avail {
            r.note(
                "memory",
                detail,
                Some("free memory, or bring the cards up (phi up all)".into()),
            );
        } else {
            r.ok("memory", detail);
        }
    }

    // A server already running.
    let bind = var("XKS_BIND", "127.0.0.1:8090");
    let health: Option<serde_json::Value> = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(2))
        .build()
        .get(&format!("http://{bind}/health"))
        .call()
        .ok()
        .and_then(|resp| resp.into_json().ok());
    match health {
        Some(h) => r.ok(
            "server",
            format!(
                "up at http://{bind}: {} on {}",
                h["subject"].as_str().unwrap_or("?"),
                h["site"].as_str().unwrap_or("?")
            ),
        ),
        None => r.note(
            "server",
            format!("none at http://{bind} (Mechanical Jev starts one when asked)"),
            Some("xks serve --detach".into()),
        ),
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_link_is_placed_only_where_no_file_would_be_lost() {
        let dir = std::env::temp_dir().join(format!("xks-link-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("xks-build");
        std::fs::write(&target, "").unwrap();
        let other = dir.join("other-build");
        std::fs::write(&other, "").unwrap();
        let link = dir.join("bin/xks");

        assert_eq!(link_state(&link, &target), LinkState::Absent);
        place_link(&link, &target).unwrap();
        assert_eq!(link_state(&link, &target), LinkState::Ours);
        assert!(place_link(&link, &target).unwrap().contains("already"));

        // Another build's link is replaced, and said.
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(&other, &link).unwrap();
        assert_eq!(link_state(&link, &target), LinkState::Other(other.clone()));
        assert!(place_link(&link, &target).unwrap().contains("other-build"));
        assert_eq!(link_state(&link, &target), LinkState::Ours);

        // A dangling link is replaced.
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(dir.join("gone"), &link).unwrap();
        assert!(matches!(link_state(&link, &target), LinkState::Dangling(_)));
        place_link(&link, &target).unwrap();
        assert_eq!(link_state(&link, &target), LinkState::Ours);

        // A file is never replaced.
        std::fs::remove_file(&link).unwrap();
        std::fs::write(&link, "a user's own script").unwrap();
        assert_eq!(link_state(&link, &target), LinkState::File);
        assert!(place_link(&link, &target).is_err());
        assert_eq!(
            std::fs::read_to_string(&link).unwrap(),
            "a user's own script"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_is_what_makes_a_report_not_ready() {
        let mut r = Report::default();
        r.ok("a", "fine");
        r.note("b", "worth knowing", Some("do this".into()));
        assert!(r.ready());
        assert!(r.text().contains("ready: xks can answer"));
        r.missing("c", "absent", "fix it");
        assert!(!r.ready());
        let t = r.text();
        assert!(t.contains("MISS  c"), "{t}");
        assert!(t.contains("fix: fix it"), "{t}");
        assert!(t.contains("not ready: 1 to fix"), "{t}");
    }
}
