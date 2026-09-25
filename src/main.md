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
up, even after a server that ran on the x86 site. Neither touches a
worker while another process holds the cards lock ([`site.md`](site.md)):
`stop` leaves them running and says who is using them, `release` refuses.

`eval` adds `subject`, `site`, `cases`, `wall_s` and `per_case_s` to the
metrics, so every record says what produced it.

The global flags are documented in `xks --help`; every one has an `XKS_*`
environment variable, which is how `xks.conf` sets them.

`serve --detach` refuses when something already answers on the address,
and reaps its child while it waits: a child that dies at startup is
reported at once (unreaped, its `/proc` entry outlived it and the wait ran
its full 900 s). While it waits it relays the child's own log lines
(`xks: ...`, `xks listening ...`, `error: ...`) to stderr as they are
written, llama.cpp's staying in the log: a 35B start is minutes, and a
silent wait read as a hang. A child that dies has its `error:` line shown
before the parent's own, which stays the last line (Mechanical Jev shows
the last lines of this output when a start fails). Stdout still carries
only the URL.

## Checked before anything slow

Since 2026-09-25 everything a command reads is read and checked before a
site is prepared (a worker restart, tens of seconds) or the subject loads
(the 35B is 21.7 GB). Before, a missing `--file`, a request that was not
JSON, a `--choice` without `|`, a missing case file or a bad `--layout`
was reported after the load. Now, in this order:

- `--template`, `--layout`, and the `--conditioning` file (its path in the
  error);
- `query`'s request: `--file` (its path in the error), the flags, or stdin
  (refused at once when stdin is a terminal, instead of waiting on it;
  "empty" when nothing came), then its questions with the server's own
  checks (`parse_questions`: a type, a Choice option, 2 to 10 Score
  levels). Empty entries in `--choice a,,b|` or a trailing comma in
  `--score` are dropped, not made options. `--compare` without
  `TYPESAFE_API_KEY` is refused before the local answer is computed;
- `eval`, `condition` and `polygraph`'s case file
  ([`eval.md`](eval.md): its path and the editor's line number in an
  error, and a file with no case refused);
- a foreground `serve`'s address, bound and let go (`cannot listen on
  ...` instead of a load and then `bind ...: Address in use`);
- the subject: a missing `XKS_SUBJECT` or a path that is not a file says
  to set it in `xks.local.conf` or pass `--subject`, before the site.

On the avx512 site the outer process is replaced by exec with the
AVX-512 build under phi512, so a request it read from stdin is handed to
that one in a file (`XKS_STDIN_REQUEST`, [`site.md`](site.md)).

Bare `xks` prints the help (exit 2) rather than clap's missing-subcommand
error; by hand, since clap counts the `XKS_*` variables `xks.conf` sets as
arguments given, and `arg_required_else_help` never fired.

## A query goes to the running server

`query` with no engine option on its command line (`ENGINE_ARGS`: the
subject, site, context, layout, template, conditioning, `--debug` and the
rest of the global options that shape the engine) first asks
`http://$XKS_BIND/health`, then the address the server holding the cards
wrote into the lock ([`site.md`](site.md); Mechanical Jev may have started
it on another port). When an xks server answers, the request goes to
it: the subject is already loaded there (and on the cards site, the
cards are its), where a second in-process load would take minutes and,
on the cards, collide with it ([`site.md`](site.md), the lock). One stderr
line says so and how to opt out (`query --local`); the answer is printed
as a local one is, with `server: N ms end-to-end`; an error status
becomes `error: the server answered N: MESSAGE`. The first of
`XKS_API_KEYS`, when set, is sent as the bearer key. An engine option on
the command line (a value from the environment or `xks.conf` does not
count: they set most of them) keeps the query in this process, as does
`--backend-kind jev`, which asks the hosted Jev. Nothing else in the
family runs `xks query` (grepped 2026-09-25 across Mechanical-Jev and
Intel-Phi-AVX512: the subprojects run `eval` with `--site`, `mjev query`
is its own HTTP client), so no consumer changes behaviour.

Measured 2026-09-25, the 0.5B served on the cards at 127.0.0.1:8095:
`XKS_BIND=127.0.0.1:8095 xks query --file examples/query.json` answered
through the server in 6.4 s (its first request).

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
| `XKS_BIND` | `serve --bind`; where `query` looks for a server | `127.0.0.1:8090` |
| `XKS_API_KEYS` | `serve --api-keys` | none |
| `XKS_KILL_DATE` | `serve --kill-date` | 0 (never) |

Read directly, not through a flag: `XKS_CONFIG` ([`config.md`](config.md)),
`XKS_BACKEND_DIR` (another llama.cpp build's libraries, else the one
`build.rs` found, [`../build.md`](../build.md)), `XKS_AVX512_BIN` and
`XKS_SITE_INNER` and `XKS_STDIN_REQUEST` ([`site.md`](site.md)),
`PHI_AVX512_ROOT` (the sibling),
`XKS_SUBJECT_SMALL`, `XKS_LLAMA_SERVER` and `XKS_BLUEBIRD_PORT` (the
subprojects, [`subproject.md`](subproject.md)), and for `--backend-kind jev`
`TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL`
([`backend/typesafe.md`](backend/typesafe.md)).
