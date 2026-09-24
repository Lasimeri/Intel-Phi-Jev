//! Where the subject runs, and the dropper that puts the payload in place.
//!
//! Three sites:
//!
//! - `x86`: this host alone, llama.cpp's standard x86-64 build (AVX2 and
//!   its repacked kernels). The reference for accuracy and for comparison.
//! - `cards`: the same build with the payload (`libggml_phi.so`, from the
//!   sibling Intel Phi AVX-512 repository) installed, so every weight
//!   matrix the cards can hold is multiplied by their vector units, 57
//!   threads per card, one per core. Offloaded by default: the cards'
//!   rows are theirs alone and no judge keeps a tensor on the host.
//! - `avx512`: llama.cpp's AVX-512 build (`build-avx512`) under the
//!   sibling's phi512 wrapper, which catches every AVX-512 instruction the
//!   host refuses and runs the region on card 0; the payload is installed
//!   too, so the multiplies go to the cards through it and the rest of
//!   llama.cpp's vector code through phi512. The outer `xks` re-executes
//!   the AVX-512 build of itself under the wrapper.
//!
//! `auto` is `cards` when a card's host window exists, else `x86`.
//! The card workers are the sibling's (`scripts/phi-vpu.sh`, its
//! interface); a worker is restarted when it was started for another site,
//! since the payload wants its memory in huge pages for weights and the
//! seamless path wants a page pool. See site.md.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    X86,
    Cards,
    Avx512,
}

impl Site {
    pub fn name(self) -> &'static str {
        match self {
            Site::X86 => "x86",
            Site::Cards => "cards",
            Site::Avx512 => "avx512",
        }
    }
}

/// The environment variable the outer `xks` sets for the one it runs
/// under phi512, so that one does not re-execute again.
pub const INNER: &str = "XKS_SITE_INNER";

/// The sibling's directory names: its GitHub clone's and the spaced one.
pub const SIBLING_NAMES: [&str; 2] = ["Intel-Phi-AVX512", "Intel Phi AVX-512"];

/// The sibling repository holding the payload, the workers and phi512:
/// `PHI_AVX512_ROOT`, else a directory next to this checkout under either
/// name, else one in `$HOME`. The first that has `scripts/phi-vpu.sh`;
/// the spaced name next to this checkout when none has (so an error
/// names a path).
pub fn sibling_root() -> PathBuf {
    if let Some(p) = std::env::var_os("PHI_AVX512_ROOT") {
        return PathBuf::from(p);
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bases: Vec<PathBuf> = here.parent().map(Path::to_path_buf).into_iter().collect();
    if let Some(home) = std::env::var_os("HOME") {
        bases.push(PathBuf::from(home));
    }
    find_sibling(&bases, &SIBLING_NAMES, "scripts/phi-vpu.sh")
        .unwrap_or_else(|| here.with_file_name(SIBLING_NAMES[1]))
}

/// The first `base/name` holding `probe`, bases in order, names in order.
pub fn find_sibling(bases: &[PathBuf], names: &[&str], probe: &str) -> Option<PathBuf> {
    bases
        .iter()
        .flat_map(|b| names.iter().map(move |n| b.join(n)))
        .find(|d| d.join(probe).exists())
}

/// The cards with a host window (`/dev/shm/phi-hostmem` is card 0,
/// `phi-hostmem-N` card N): the cards that are up.
pub fn card_windows() -> Vec<u32> {
    let mut cards = Vec::new();
    if Path::new("/dev/shm/phi-hostmem").exists() {
        cards.push(0);
    }
    if let Ok(dir) = std::fs::read_dir("/dev/shm") {
        for e in dir.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if let Some(n) = name.strip_prefix("phi-hostmem-") {
                if let Ok(n) = n.parse() {
                    cards.push(n);
                }
            }
        }
    }
    cards.sort_unstable();
    cards.dedup();
    cards
}

pub fn resolve(requested: &str) -> Result<Site, String> {
    match requested {
        "x86" | "host" => Ok(Site::X86),
        "cards" | "phi" => Ok(Site::Cards),
        "avx512" => Ok(Site::Avx512),
        "auto" => Ok(if card_windows().is_empty() {
            Site::X86
        } else {
            Site::Cards
        }),
        other => Err(format!("unknown site `{other}` (auto|x86|cards|avx512)")),
    }
}

/// The payload library: the sibling's release build, else its debug one.
pub fn payload(root: &Path) -> Result<PathBuf, String> {
    for p in [
        "host/target/release/libggml_phi.so",
        "host/target/debug/libggml_phi.so",
    ] {
        let lib = root.join(p);
        if lib.is_file() {
            return Ok(lib);
        }
    }
    Err(format!(
        "the payload is not built: (cd \"{}/host\" && cargo build --release -p phi-ggml)",
        root.display()
    ))
}

/// How a worker is started for a site: huge pages reserved on the card
/// and the worker's own options.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkerConfig {
    hugepages: u32,
    args: String,
}

impl WorkerConfig {
    fn tag(&self) -> String {
        format!("hugepages={} args={}", self.hugepages, self.args)
    }
}

fn marker(card: u32) -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("xks").join(format!("worker-{card}"))
}

/// The sibling's worker script. Its stdout goes to our stderr: `xks`'s own
/// stdout carries only its result (a subproject parses it as JSON, and a
/// "worker started" line in it broke subproject 03).
fn phi_vpu(root: &Path, card: u32, args: &[&str]) -> Command {
    let mut c = Command::new(root.join("scripts/phi-vpu.sh"));
    c.arg("-c")
        .arg(card.to_string())
        .args(args)
        .stdout(std::process::Stdio::from(std::io::stderr()));
    c
}

fn polling(root: &Path, card: u32) -> bool {
    phi_vpu(root, card, &["status"])
        .stdout(std::process::Stdio::piped())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("worker: polling"))
        .unwrap_or(false)
}

/// A worker polling on `card` with `cfg`, restarted when it was started
/// with anything else (or by something other than xks).
fn ensure_worker(root: &Path, card: u32, cfg: &WorkerConfig) -> Result<(), String> {
    let mark = marker(card);
    let same = std::fs::read_to_string(&mark).ok().as_deref() == Some(cfg.tag().as_str());
    if same && polling(root, card) {
        return Ok(());
    }
    eprintln!("xks: starting the worker on card {card} ({})", cfg.tag());
    let status = phi_vpu(root, card, &["start"])
        .env("PHI_VPU_HUGEPAGES", cfg.hugepages.to_string())
        .env("PHI_VPU_ARGS", &cfg.args)
        .status()
        .map_err(|e| format!("phi-vpu.sh: {e}"))?;
    if !status.success() || !polling(root, card) {
        return Err(format!("the worker on card {card} did not start polling"));
    }
    if let Some(dir) = mark.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&mark, cfg.tag());
    Ok(())
}

/// The cards whose workers `xks` started (a marker records each one's
/// configuration): what `stop` gives back. `xks release` takes every card.
pub fn xks_workers() -> Vec<u32> {
    card_windows()
        .into_iter()
        .filter(|&c| marker(c).is_file())
        .collect()
}

/// Stop the workers and give the cards their huge pages back.
pub fn release(cards: &[u32]) -> Result<(), String> {
    let root = sibling_root();
    for &c in cards {
        let status = phi_vpu(&root, c, &["stop"])
            .status()
            .map_err(|e| format!("phi-vpu.sh: {e}"))?;
        if !status.success() {
            return Err(format!("could not stop the worker on card {c}"));
        }
        let _ = std::fs::remove_file(marker(c));
    }
    Ok(())
}

/// What the rest of `xks` needs to know about the site it is running on.
#[derive(Debug, Clone)]
pub struct Placed {
    pub site: Site,
    pub cards: Vec<u32>,
    /// llama.cpp threads for this site.
    pub threads: i32,
    /// Let llama.cpp repack weights for the host CPU.
    pub repack: bool,
    /// Flash attention allowed on this site.
    pub flash_attn: bool,
}

/// Put the subject's site in place: environment for the payload, workers
/// polling with this site's configuration. For `avx512` in the outer
/// process this re-executes the AVX-512 build under phi512 and does not
/// return. Call before any thread starts (it sets the environment).
pub fn prepare(site: Site, offload: bool) -> Result<Placed, String> {
    let inner = std::env::var_os(INNER).is_some();
    match site {
        Site::X86 => {
            // The reference must be the host alone: a GGML_BACKEND_PATH
            // inherited from the shell or a config file would load the
            // payload into it, and x86 against cards would compare the
            // payload with itself.
            if std::env::var_os("GGML_BACKEND_PATH").is_some() {
                eprintln!("xks: x86 site: ignoring GGML_BACKEND_PATH");
                std::env::remove_var("GGML_BACKEND_PATH");
            }
            Ok(Placed {
                site,
                cards: Vec::new(),
                threads: 16,
                repack: false,
                flash_attn: true,
            })
        }
        Site::Cards => {
            let root = sibling_root();
            let cards = card_windows();
            if cards.is_empty() {
                return Err("no card is up (no /dev/shm/phi-hostmem*); phi status".into());
            }
            let lib = payload(&root)?;
            let cfg = WorkerConfig {
                hugepages: 2400,
                args: "-e 0".into(),
            };
            for &c in &cards {
                ensure_worker(&root, c, &cfg)?;
            }
            install(&lib, &cards, offload, None);
            Ok(Placed {
                site,
                cards,
                threads: 12,
                repack: false,
                flash_attn: true,
            })
        }
        Site::Avx512 if !inner => reexec_avx512(),
        Site::Avx512 => {
            // Inside phi512. The cards split by role: card 0 runs phi512's
            // regions and nothing else; the payload's multiplies go to the
            // others. One worker serving both deadlocked: the single host
            // thread waited on a multiply while its own region queued
            // behind it on the same card (site.md).
            let root = sibling_root();
            let cards = card_windows();
            if !cards.contains(&0) {
                return Err("phi512 runs on card 0, which is not up; phi status".into());
            }
            ensure_worker(
                &root,
                0,
                &WorkerConfig {
                    hugepages: 768,
                    args: String::new(),
                },
            )?;
            let payload_cards: Vec<u32> = cards.iter().copied().filter(|&c| c != 0).collect();
            if payload_cards.is_empty() {
                eprintln!("xks: one card up: phi512 only, no payload");
            } else {
                let lib = payload(&root)?;
                for &c in &payload_cards {
                    ensure_worker(
                        &root,
                        c,
                        &WorkerConfig {
                            hugepages: 2400,
                            args: "-e 0".into(),
                        },
                    )?;
                }
                install(&lib, &payload_cards, offload, None);
            }
            // One host thread: the AVX-512 build's OpenMP barrier spins
            // while one thread's region runs on the card.
            std::env::set_var("PHI_GGML_HOST_THREADS", "1");
            Ok(Placed {
                site,
                cards,
                threads: 1,
                repack: false,
                flash_attn: false,
            })
        }
    }
}

fn install(lib: &Path, cards: &[u32], offload: bool, card_bytes: Option<f64>) {
    let list: Vec<String> = cards.iter().map(|c| c.to_string()).collect();
    std::env::set_var("GGML_BACKEND_PATH", lib);
    std::env::set_var("PHI_GGML_CARDS", list.join(","));
    if offload {
        std::env::set_var("PHI_GGML_OFFLOAD", "1");
    } else {
        std::env::remove_var("PHI_GGML_OFFLOAD");
    }
    if let Some(b) = card_bytes {
        std::env::set_var("PHI_GGML_CARD_BYTES", format!("{b}"));
    }
}

/// The AVX-512 build of this binary: `XKS_AVX512_BIN`, else
/// `target/avx512/release/xks` beside `target/release/xks`.
fn avx512_binary() -> Result<PathBuf, String> {
    if let Some(p) = std::env::var_os("XKS_AVX512_BIN") {
        return Ok(PathBuf::from(p));
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let target = exe
        .parent()
        .and_then(Path::parent)
        .ok_or("cannot place the AVX-512 build next to this binary")?;
    let bin = target.join("avx512/release/xks");
    if bin.is_file() {
        Ok(bin)
    } else {
        Err(format!(
            "the AVX-512 build is missing ({}); make build-avx512",
            bin.display()
        ))
    }
}

fn reexec_avx512() -> Result<Placed, String> {
    use std::os::unix::process::CommandExt;
    let root = sibling_root();
    let wrapper = root.join("scripts/phi512.sh");
    if !wrapper.is_file() {
        return Err(format!("phi512 wrapper not found: {}", wrapper.display()));
    }
    let bin = avx512_binary()?;
    let err = Command::new(wrapper)
        .arg("--card")
        .arg("0")
        .arg(bin)
        .args(std::env::args_os().skip(1))
        .env(INNER, "1")
        .exec();
    Err(format!("could not run phi512: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sibling_is_found_under_either_name_nearest_base_first() {
        let tmp = std::env::temp_dir().join(format!("xks-sibling-{}", std::process::id()));
        let (near, home) = (tmp.join("near"), tmp.join("home"));
        let probe = "scripts/phi-vpu.sh";
        let plant = |base: &Path, name: &str| {
            let d = base.join(name).join("scripts");
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("phi-vpu.sh"), "").unwrap();
        };
        let bases = [near.clone(), home.clone()];
        assert_eq!(find_sibling(&bases, &SIBLING_NAMES, probe), None);
        plant(&home, SIBLING_NAMES[1]);
        assert_eq!(
            find_sibling(&bases, &SIBLING_NAMES, probe),
            Some(home.join(SIBLING_NAMES[1]))
        );
        plant(&near, SIBLING_NAMES[0]);
        assert_eq!(
            find_sibling(&bases, &SIBLING_NAMES, probe),
            Some(near.join(SIBLING_NAMES[0]))
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
