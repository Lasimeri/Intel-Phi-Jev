# Intel-Phi-Jev: notes for an agent working here

Read `CONTRIBUTING.md` first; it is the authority. The non-obvious rules:

- What this is: `xks`, a local Jev (TypeSafe System One): `/v1/systemone`,
  Noul, Choice (up to 255 options), Score (2 to 10 levels). The spec is
  docs.typesafe.ai and TypeSafe's MIT `system-one-adapter`; unofficial Jev
  sites are not. It needs Intel-Phi-AVX512 for the cards
  (`PHI_AVX512_ROOT`, else a checkout next to this one or in `$HOME`,
  under either name); Mechanical-Jev consumes `xks serve --detach --bind`,
  `xks stop`, `target/release/xks`, the wire format, `/health`'s
  `status`, `subject`, `kill_date_s` and `idle_s`, and `xks doctor [--fix] [--prefix P]` (its
  text relayed, exit 0 ready, 1 not): add, do not rename.
- Rust only. No Python or JavaScript, ever.
- Names (README, Naming): the Revelation to John is the foundation;
  XKEYSCORE, MKULTRA, Stuxnet and the Gateway Process are lenses built on
  it. A name is placed by what its original was, as close to the part's
  job as it can be; new names go in the README table and doc comments,
  and no identifier, flag or wire name is renamed for a theme.
- Sibling `.md` per code file, same change. No em or en dashes anywhere.
- Sites: `x86` (reference), `cards` (the payload), `avx512` (AVX-512 build
  under phi512). One process at a time may hold the cards, enforced by
  `$XDG_RUNTIME_DIR/xks/cards.lock` (flock; a second process is refused
  with the holder's pid). `xks release` frees their huge pages (4.7 GiB
  each). Inputs are checked before a site or subject loads: keep new
  checks there. A plain `xks query` goes to a running server.
- Measure through `xks subproject run NN` (600 s budget each); records in
  docs/subprojects/results/. The 35B's label log-probabilities move up to
  0.99 with how a prompt is cut into decodes (subproject 02): compare
  against that floor.
- Test data is real text or obviously artificial, never invented people,
  companies or tickets. `make check` before committing, push after.
- Traps: perl substitutions with `{}` or `|` delimiters over Rust code break
  (use the Edit tool); `pkill -f` from a tool shell can kill the shell
  itself (kill by pid); the payload needs repacking off or it is offered
  almost nothing; `--forks 1` is the exactness test for the copy; the
  binary exports its own `mmap` (it strips `MAP_POPULATE` while the
  subject loads on an offloaded site, main.md), so a host-memory figure
  from before 2026-09-25 counts the whole model populated at load.
