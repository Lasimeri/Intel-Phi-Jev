# Contributing

- Rust for everything `xks` does. No Python, no JavaScript. The Makefile
  and `scripts/check-docs.sh` are the only other code, and they only call
  into `xks` or check the tree.
- Every code file has a sibling `.md` (what it does, why, what was
  measured). A change to behaviour changes its `.md` in the same commit.
- No em or en dashes anywhere.
- A claim about speed or accuracy cites a subproject record in
  `docs/subprojects/results/`; add or re-run a subproject rather than quote
  a one-off run. Timings closer than this host's drift (a quarter over tens
  of minutes) need interleaved runs.
- Test data is real text (this project's and its siblings' own documents,
  real commands, real operation names) or obviously artificial. Never
  invented people, companies, tickets or accounts.
- `make check` before a commit: docs, format, lint (warnings are errors),
  both builds, tests.
