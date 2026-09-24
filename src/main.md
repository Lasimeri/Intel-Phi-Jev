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
