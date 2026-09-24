# config.rs: defaults from files

So that every command runs without paths: `xks query --file req.json` finds
the subject in `xks.conf`.

Order, first to set a key winning: the environment, `$XKS_CONFIG`,
`xks.local.conf` (untracked, this machine's own), `~/.config/xks/xks.conf`,
`xks.conf` (tracked defaults). The repository is found from the binary's
path (an ancestor holding both `Cargo.toml` and `xks.conf`), which covers
`target/release/xks` and `target/avx512/release/xks`.

The format is what a shell can source: `KEY=VALUE`, `#` comments, optional
`export`, optional quotes, `~/`, `$HOME` and `${HOME}` expanded. Nothing
else is expanded; a value needing more belongs in the environment.

`load` runs first thing in `main`, before any thread exists, since it sets
environment variables.

Values are read as a shell reads them, for what these files use:
unquoted, `"double"` (with `$HOME` expanded and `\"`, `\\`, `\$` escaped)
or `'single'` (literal) segments, joined; a leading `~/`; an unquoted `#`
at the start or after a blank starts a comment, and unquoted blanks end
the value. `$HOME` and `${HOME}` expand only as that whole name. Before
2026-09-24 `XKS_PERMUTATIONS=3  # rotations` read as `3  # rotations`
(which the command line then refused), `'$HOME'` was expanded, and
`$HOME_DIR` became the home directory plus `_DIR`. The same parser is
Mechanical Jev's.
