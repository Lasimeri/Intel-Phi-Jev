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

/// The cards that are up: a host window (`/dev/shm/phi-hostmem` is card 0,
/// `phi-hostmem-N` card N) whose card's daemon answers on its control
/// socket. The stack never unlinks a window, so a card that is down keeps
/// one, and `auto` used to pick the cards site for it and fail at the
/// worker; a daemon that is not running refuses the connection at once.
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
    cards.retain(|&c| daemon_answers(c));
    cards
}

/// The stack's control socket for `card` (its `phi-env.sh`: the runtime
/// directory's `phictl/control.sock` for card 0, `phictl/N/control.sock`
/// for card N).
pub fn control_socket(card: u32) -> PathBuf {
    let run = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("phictl");
    if card == 0 {
        run.join("control.sock")
    } else {
        run.join(card.to_string()).join("control.sock")
    }
}

/// Whether `card`'s daemon accepts a connection on its control socket.
fn daemon_answers(card: u32) -> bool {
    std::os::unix::net::UnixStream::connect(control_socket(card)).is_ok()
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

/// The lock that gives the cards to one `xks` process at a time: the
/// payload frees every card's uploads when it opens and a worker restart
/// kills the one running, so a second process (an `eval` beside a
/// server) would break the first without a word. `flock`, so the lock
/// goes when its holder does, crash or not.
pub fn lock_path() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("xks")
        .join("cards.lock")
}

/// Held for the life of the process once taken.
static HOLD: std::sync::OnceLock<std::fs::File> = std::sync::OnceLock::new();

/// Take the cards for this process, or say which process has them.
pub fn take_cards() -> Result<(), String> {
    if HOLD.get().is_none() {
        let f = take(&lock_path())?;
        let _ = HOLD.set(f);
    }
    Ok(())
}

fn open_lock(path: &Path) -> Result<std::fs::File, String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// The lock at `path`, taken, with this process written into it.
fn take(path: &Path) -> Result<std::fs::File, String> {
    use std::io::Write;
    let mut f = open_lock(path)?;
    match f.try_lock() {
        Ok(()) => {
            let cmd: Vec<String> = std::env::args().skip(1).collect();
            let _ = f.set_len(0);
            let _ = write!(f, "{} xks {}", std::process::id(), cmd.join(" "));
            Ok(f)
        }
        Err(std::fs::TryLockError::WouldBlock) => {
            let who = holder(path).unwrap_or_else(|| "another process".into());
            let ask = match serving(path) {
                Some(url) => format!("ask that server at {url} (a plain `xks query` does)"),
                None => "wait for it".into(),
            };
            Err(format!(
                "the cards are in use by {who}; one process at a time may hold them: \
                 {ask}, or stop it first (`xks stop` for a detached server)"
            ))
        }
        Err(std::fs::TryLockError::Error(e)) => Err(format!("{}: {e}", path.display())),
    }
}

/// Who holds the lock at `path`, when something does: `pid N (command)`
/// as the holder wrote it while that pid is alive, else a pointer to the
/// tool that names it. Held is held: unreadable contents still say so.
fn holder(path: &Path) -> Option<String> {
    let f = open_lock(path).ok()?;
    if f.try_lock().is_ok() {
        return None;
    }
    let named = std::fs::read_to_string(path).ok().and_then(|t| {
        let (pid, cmd) = t.lines().next()?.trim().split_once(' ')?;
        let pid: u32 = pid.parse().ok()?;
        Path::new(&format!("/proc/{pid}"))
            .exists()
            .then(|| format!("pid {pid} (`{cmd}`)"))
    });
    Some(
        named
            .unwrap_or_else(|| format!("another process (`fuser -v {}` names it)", path.display())),
    )
}

/// The URL the holder of the lock at `path` serves on, when it is held by
/// a server that said so (`note_serving`).
fn serving(path: &Path) -> Option<String> {
    holder(path)?;
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix("serves "))
        .map(|u| u.trim().to_string())
}

/// Who holds the cards, when a process does (`release` and `stop` leave
/// its workers alone).
pub fn cards_holder() -> Option<String> {
    holder(&lock_path())
}

/// Where the server holding the cards answers, when one does: a plain
/// `xks query` looks there after `XKS_BIND` (Mechanical Jev can start a
/// server on another address).
pub fn cards_server() -> Option<String> {
    serving(&lock_path())
}

/// Write the address this process serves on into the lock it holds, for
/// the refusal message and `cards_server`. Nothing when it holds none
/// (the x86 site).
pub fn note_serving(bind: &str) {
    use std::io::Write;
    if let Some(mut f) = HOLD.get() {
        let _ = write!(f, "\nserves http://{bind}\n");
    }
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

/// Whether the card holds what `cfg` asks for: `phi-vpu.sh config` (the
/// sibling's; "hugepages N" and "worker ARGS") read back and compared. A
/// sibling without the verb reads as not holding it, and the worker is
/// restarted, which is always safe.
fn holds(root: &Path, card: u32, cfg: &WorkerConfig) -> bool {
    phi_vpu(root, card, &["config"])
        .stdout(std::process::Stdio::piped())
        .output()
        .map(|o| config_matches(&String::from_utf8_lossy(&o.stdout), cfg))
        .unwrap_or(false)
}

/// `phi-vpu.sh config`'s output against `cfg`: the reservation equal, and
/// the worker's arguments (after its `-v`, before its thread count) the
/// ones xks passes.
fn config_matches(out: &str, cfg: &WorkerConfig) -> bool {
    let field = |k: &str| {
        out.lines()
            .find_map(|l| l.strip_prefix(k).map(str::trim))
            .unwrap_or("")
    };
    let hugepages_ok = field("hugepages ").parse::<u32>().ok() == Some(cfg.hugepages);
    let worker: Vec<&str> = field("worker ").split_whitespace().collect();
    let args: Vec<&str> = match worker.as_slice() {
        ["-v", rest @ .., _threads] => rest.to_vec(),
        _ => return false,
    };
    hugepages_ok && args == cfg.args.split_whitespace().collect::<Vec<_>>()
}

/// A worker polling on `card` with `cfg`, restarted when it was started
/// with anything else (or by something other than xks).
fn ensure_worker(root: &Path, card: u32, cfg: &WorkerConfig) -> Result<(), String> {
    let mark = marker(card);
    let same = std::fs::read_to_string(&mark).ok().as_deref() == Some(cfg.tag().as_str());
    // The marker says what xks started; the card says what is there now.
    // Something else (phi512.sh, phi-ggml.sh) may have restarted the worker
    // since, with another reservation, and the marker would not know.
    if same && polling(root, card) && holds(root, card, cfg) {
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
            // Before any worker is touched: a restart kills the holder's.
            take_cards()?;
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
            // Taken here, in the process under phi512, not in the outer
            // one: the outer is replaced by exec before it could use it.
            take_cards()?;
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
    fn a_stale_control_socket_does_not_count_as_a_card() {
        // A socket file with nothing listening, as a stopped daemon leaves.
        let dir = std::env::temp_dir().join(format!("xks-sock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("control.sock");
        drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
        assert!(path.exists());
        assert!(std::os::unix::net::UnixStream::connect(&path).is_err());
        // And one that is listening does.
        let _live = std::os::unix::net::UnixListener::bind(dir.join("live.sock")).unwrap();
        assert!(std::os::unix::net::UnixStream::connect(dir.join("live.sock")).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_cards_lock_is_one_holder_at_a_time_and_names_it() {
        let dir = std::env::temp_dir().join(format!("xks-lock-{}", std::process::id()));
        let path = dir.join("cards.lock");
        assert_eq!(holder(&path), None);
        let held = take(&path).unwrap();
        // A second open file description conflicts under flock, as a
        // second process would.
        let err = take(&path).unwrap_err();
        assert!(
            err.contains(&format!("pid {}", std::process::id())),
            "{err}"
        );
        let who = holder(&path).unwrap();
        assert!(
            who.starts_with(&format!("pid {} (`xks", std::process::id())),
            "{who}"
        );
        assert_eq!(serving(&path), None);
        // A server says where it answers, as note_serving writes it.
        {
            use std::io::Write;
            write!(&held, "\nserves http://127.0.0.1:8095\n").unwrap();
        }
        assert_eq!(serving(&path).as_deref(), Some("http://127.0.0.1:8095"));
        let err = take(&path).unwrap_err();
        assert!(err.contains("at http://127.0.0.1:8095"), "{err}");
        assert!(holder(&path).unwrap().ends_with("`)"), "one line only");
        drop(held);
        assert_eq!(holder(&path), None);
        // The line outlives its writer in the file; a free lock has no server.
        assert_eq!(serving(&path), None);
        // Held by something that wrote nothing (flock(1), say, or a
        // holder between its lock and its write): still held, unnamed.
        let other = open_lock(&path).unwrap();
        other.set_len(0).unwrap();
        other.try_lock().unwrap();
        let who = holder(&path).unwrap();
        assert!(who.starts_with("another process"), "{who}");
        drop(other);
        drop(take(&path).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_worker_is_kept_only_when_the_card_holds_what_xks_asked_for() {
        let cards = WorkerConfig {
            hugepages: 2400,
            args: "-e 0".into(),
        };
        assert!(config_matches(
            "hugepages 2400\nworker -v -e 0 57\n",
            &cards
        ));
        // Restarted by phi512.sh with its defaults, or not running at all.
        assert!(!config_matches("hugepages 768\nworker -v 57\n", &cards));
        assert!(!config_matches("hugepages 2400\nworker none\n", &cards));
        assert!(!config_matches("", &cards));
        let plain = WorkerConfig {
            hugepages: 768,
            args: String::new(),
        };
        assert!(config_matches("hugepages 768\nworker -v 57\n", &plain));
    }

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
