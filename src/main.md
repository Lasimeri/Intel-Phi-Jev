# main.rs: the `xks` command line

Loads defaults ([`config.rs`](config.md)) before clap reads the environment,
then dispatches. Commands that need no subject (`replay`, `ledger`,
`corroborate`, `release`, `stop`, `subproject`, `serve --detach`) return
before any model is loaded. For the rest, the backend is built once:
ARTICHOKE after the site is prepared ([`site.rs`](site.md)), or BLUEBIRD /
OpenAI over HTTP.

`serve --detach` re-runs the same command line without `--detach` in its
own process group, logs to `$XDG_RUNTIME_DIR/xks/serve.log`, writes
`serve.pid`, and returns once `/health` answers. `stop` acts on the pid
only when `/proc/<pid>/cmdline` is still an `xks ... serve` (directly, or
under `phi512.sh` on the avx512 site): a pid left over from a crash or a
reboot can name any other process. It sends SIGTERM to the whole process
group, SIGKILL after 30 s, and fails if the server is still there. It then
releases only the cards whose workers `xks` started (the markers
`$XDG_RUNTIME_DIR/xks/worker-N`), so workers another program started on
the cards, a `phi-ggml.sh` llama-server say, keep running; `xks release`
takes every card. Before 2026-09-24 `stop` released every card that was
up, even after a server that ran on the x86 site.

`eval` adds `subject`, `site`, `cases`, `wall_s` and `per_case_s` to the
metrics, so every record says what produced it.

The global flags are documented in `xks --help`; every one has an `XKS_*`
environment variable, which is how `xks.conf` sets them.

`serve --detach` refuses when something already answers on the address,
and reaps its child while it waits: a child that dies at startup is
reported at once (unreaped, its `/proc` entry outlived it and the wait ran
its full 900 s).

## Environment

Every global flag and `serve`'s have a variable (clap's `env`), which is
how `xks.conf` and `xks.local.conf` set them; a flag on the command line
wins over both.

| variable | flag | default |
| --- | --- | --- |
| `XKS_SUBJECT` | `--subject` (`--gguf`) | none; `xks.conf` names one |
| `XKS_CTX` | `--ctx` | 16384 |
| `XKS_FORKS` | `--forks` | 15 |
| `XKS_BATCH` | `--batch` | 2048 |
| `XKS_UBATCH` | `--ubatch` | 512 (2048 measured, no difference: [`artichoke/mod.md`](artichoke/mod.md)) |
| `XKS_THREADS` | `--threads` | 12 with the payload loaded, else 16 |
| `XKS_REPACK` | `--repack` | only without the payload |
| `XKS_SITE` | `--site` | `auto` |
| `XKS_BACKEND_KIND` | `--backend-kind` | `artichoke` |
| `XKS_BACKEND_URL` | `--backend` | `http://127.0.0.1:8089` |
| `XKS_MODEL` | `--model` (openai) | none |
| `XKS_API_KEY` | named by `--api-key-env` (openai) | |
| `XKS_EXTRA` | `--extra` (openai) | none |
| `XKS_LAYOUT` | `--layout` | `letters` |
| `XKS_TEMPLATE` | `--template` | `chatml` |
| `XKS_CONDITIONING` | `--conditioning` (`--calibration`) | none |
| `XKS_PERMUTATIONS` | `--permutations` | 1 (`xks.conf` sets 3) |
| `XKS_BIND` | `serve --bind` | `127.0.0.1:8090` |
| `XKS_API_KEYS` | `serve --api-keys` | none |
| `XKS_KILL_DATE` | `serve --kill-date` | 0 (never) |

Read directly, not through a flag: `XKS_CONFIG` ([`config.md`](config.md)),
`XKS_BACKEND_DIR` (another llama.cpp build's libraries, else the one
`build.rs` found, [`../build.md`](../build.md)), `XKS_AVX512_BIN` and
`XKS_SITE_INNER` ([`site.md`](site.md)), `PHI_AVX512_ROOT` (the sibling),
`XKS_SUBJECT_SMALL`, `XKS_LLAMA_SERVER` and `XKS_BLUEBIRD_PORT` (the
subprojects, [`subproject.md`](subproject.md)), and for `--backend-kind jev`
`TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL`
([`backend/typesafe.md`](backend/typesafe.md)).
