# Contributing

These rules exist so that every number here can be re-run with one
command and every change reviewed against its reason. The sections are
the same in every repository of the family (see [The family](#the-family));
what differs is said where it applies.

## Languages

- **Rust** for everything `xks` does.
- **C** only through llama.cpp's own C API (bindgen over its headers, in
  `build.rs`); nothing here is written in C.
- **Shell and make** only for `scripts/check-docs.sh` and the `Makefile`,
  which check the tree or call into `xks`.
- **Never Python or JavaScript** for anything here.

## Documentation

- Every code file has a sibling `.md` with the same stem: what it does,
  why, what was measured. A change to behaviour changes its `.md` in the
  same commit.
- No em or en dash characters anywhere, commit messages included.
- Relative links between Markdown files must resolve. A file that lives in
  a sibling repository is linked on GitHub, never named as if it were here.
- `scripts/check-docs.sh` enforces the sibling, dash and link rules
  (`make docs-check`, the first step of `make check`).

## Measurements

- A claim about speed or accuracy cites a subproject record in
  `docs/subprojects/results/`; add or re-run a subproject
  (`xks subproject run NN`, ten minutes at most each) rather than quote a
  one-off run.
- Timings closer than this host's drift (a quarter over tens of minutes)
  need interleaved runs.
- Test data is real text (this project's and its siblings' own documents,
  real commands, real operation names) or obviously artificial. Never
  invented people, companies, tickets or accounts.

## The family

| repository | what | finds its dependency by |
| --- | --- | --- |
| [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A) | the cards' software stack: daemon, kernel, boot, storage, the `phi` CLI | (none) |
| [Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512) | the cards as an AVX-512 co-processor: phi512, the card worker, the `libggml_phi.so` backend | `PHI_STACK_ROOT`, `phi` on PATH, a checkout next to it, `$HOME` |
| [Intel-Phi-Jev](https://github.com/Lasimeri/Intel-Phi-Jev) (this one) | `xks`, a local Jev (System One) whose subject runs on the host and the cards | `PHI_AVX512_ROOT`, a checkout next to this one, `$HOME` |
| [Mechanical-Jev](https://github.com/Lasimeri/Mechanical-Jev) | `mjev`, the asking side of Jev, and Jev reverse engineered from its docs | `MJEV_XKS`, `xks` on PATH, a checkout next to it, `$HOME` |

- A dependency is found in that order, as a checkout under its GitHub
  clone's name (`Intel-Phi-AVX512`) or the spaced one (`Intel Phi AVX-512`)
  ([`src/site.md`](src/site.md)). Nothing of a sibling is copied into
  another (the one exception is between Intel-Phi-3120A and
  Intel-Phi-AVX512: the `knc-mvex` library, kept identical by the latter's
  `make check`).
- The interfaces Mechanical-Jev consumes keep working across changes:
  `xks serve --detach --bind`, `xks stop`, `target/release/xks`, `/health`,
  `/v1/models`, the System One wire format, and `xks doctor [--fix]
  [--prefix P]` (its text relayed as it is, exit 0 ready, 1 not). Add, do
  not rename; when
  one must change, change its consumer in the same session.

## Git

- One subject line that says what changed (a leading `Area:` is fine), then
  the why. `make check` before every commit (docs, format, lint with
  warnings as errors, both builds, tests), push after.
- Never commit a key: `TYPESAFE_API_KEY` lives in `xks.local.conf`, which
  git ignores.
- MIT or Apache-2.0 ([`LICENSE-MIT`](LICENSE-MIT),
  [`LICENSE-APACHE`](LICENSE-APACHE)); the imported jev-rs portions keep
  their authors' notice ([`NOTICE`](NOTICE)).
