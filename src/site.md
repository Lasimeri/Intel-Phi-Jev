# site.rs: where the subject runs, and the dropper

| site | llama.cpp build | payload | card workers | threads | flash attention |
| --- | --- | --- | --- | --- | --- |
| `x86` | x86-64 (`build-native`) | no | untouched | 16 | auto |
| `cards` | x86-64 | yes, offloaded | 2400 huge pages, `-e 0` (no seamless pool) | 12 | auto |
| `avx512` | AVX-512 (`build-avx512`) under phi512 | yes, 3.4 GB per card | card 0: 2100 huge pages with the seamless pool; others as `cards` | 1 | off |

`auto` is `cards` when `/dev/shm/phi-hostmem*` exists (a card is up), else
`x86`.

## The dropper

`prepare` installs the payload into this process: `GGML_BACKEND_PATH` names
the sibling's `libggml_phi.so`, `PHI_GGML_CARDS` the cards,
`PHI_GGML_OFFLOAD=1` unless `--no-offload`. It runs before llama.cpp
registers its backends and before any thread exists. This replaced a shell
script: the payload's environment is now set by the process that uses it.

Offload is the default because the user's direction is that the cards do
the work: every row a card holds is computed there, with no per-tensor judge
keeping it on a faster host. `--no-offload` restores the payload's own
judgement (it times host against cards per tensor).

## Workers

The workers are the sibling's (`scripts/phi-vpu.sh`, its interface). The
payload wants card memory as huge pages for weight rows and no seamless
pool; phi512 wants the pool. A worker started for one site is wrong for the
other, and `phi-vpu.sh` does not restart a polling worker, so `xks` records
the configuration it started each worker with
(`$XDG_RUNTIME_DIR/xks/worker-N`) and restarts a worker whose record differs
or is missing. `release` stops the workers and returns the huge pages
(`nr_hugepages` to 0): 4.7 GiB per card.

One process at a time may hold the cards: the payload frees every card's
uploads when it opens, so a second process would pull the first one's rows
out from under it.

## The avx512 site

The outer `xks` (x86-64 build) re-executes `target/avx512/release/xks`
(`XKS_AVX512_BIN` overrides) under `scripts/phi512.sh --card 0` with
`XKS_SITE_INNER=1`, which is how the inner one knows not to re-execute. The
wrapper preloads libphi512 through `LD_PRELOAD` (never `/etc/ld.so.preload`,
which also catches `sudo`).

## What stops the avx512 site today

Both found running one fingerprint of a 0.5B subject; both are the sibling
co-processor's to fix, and both were refused cleanly (the process stops,
the host is untouched):

- q4_0 weights: llama.cpp's AVX-512 q4_0 path uses `vpaddb ymm, ymm, ymm31`
  (AVX-512BW/VL byte lanes), which is not in the card's instruction table;
  KNC's vector unit has 32-bit lanes only, so byte forms need translating
  through wider lanes.
- flash attention, fp16 subject: a phase in
  `ggml_compute_forward_flash_attn_ext_tiled` (a `vbroadcastss (%rdx)`,
  `vmovups (%rax)` loop, `rax` advancing 256 bytes an iteration) touched
  memory the process never mapped, after five minutes of regions that ran.
  The planner's reach for that loop overshoots; the site runs with flash
  attention off to route around it.
