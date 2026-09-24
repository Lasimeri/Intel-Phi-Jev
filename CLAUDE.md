# Intel Phi Jev: notes for an agent working here

- `xks` is a local Jev (TypeSafe System One): `/v1/systemone`, Noul, Choice
  (up to 255 options), Score (2 to 10 levels). The spec is docs.typesafe.ai
  and TypeSafe's MIT `system-one-adapter`; unofficial Jev sites are not.
- Rust only (see CONTRIBUTING.md). Sibling `.md` per code file, no dashes,
  `make check` green before committing, push to origin after.
- Sites: `x86` (reference), `cards` (payload from ~/Intel Phi AVX-512),
  `avx512` (AVX-512 build under phi512). One process at a time may hold the
  cards. `xks release` frees their huge pages (4.7 GiB each).
- Measure through `xks subproject run NN`; records in
  docs/subprojects/results/. The 35B's label log-probabilities move up to
  0.8 with how a prompt is cut into decodes: compare against that floor.
- Traps: perl substitutions with `{}` or `|` delimiters over Rust code break
  (use the Edit tool); `pkill -f` from a tool shell can kill the shell
  itself (kill by pid); the phi-ggml payload needs repacking off or it is
  offered almost nothing; `--forks 1` is the exactness test for the copy.
