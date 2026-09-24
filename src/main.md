# main.rs: the `xks` command line

Loads defaults ([`config.rs`](config.md)) before clap reads the environment,
then dispatches. Commands that need no subject (`replay`, `ledger`,
`corroborate`, `release`, `stop`, `subproject`, `serve --detach`) return
before any model is loaded. For the rest, the backend is built once:
ARTICHOKE after the site is prepared ([`site.rs`](site.md)), or BLUEBIRD /
OpenAI over HTTP.

`serve --detach` re-runs the same command line without `--detach` in its
own process group, logs to `$XDG_RUNTIME_DIR/xks/serve.log`, writes
`serve.pid`, and returns once `/health` answers; `stop` signals it, waits,
and releases the card workers' huge pages.

`eval` adds `subject`, `site`, `cases`, `wall_s` and `per_case_s` to the
metrics, so every record says what produced it.

The global flags are documented in `xks --help`; every one has an `XKS_*`
environment variable, which is how `xks.conf` sets them.
