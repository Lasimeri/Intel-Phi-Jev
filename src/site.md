# site.rs: where the subject runs, and the dropper

| site | llama.cpp build | payload | card workers | threads | flash attention |
| --- | --- | --- | --- | --- | --- |
| `x86` | x86-64 (`build-native`) | no | untouched | 16 | auto |
| `cards` | x86-64 | yes, offloaded | 2400 huge pages, `-e 0` (no seamless pool) | 12 | auto |
| `avx512` | AVX-512 (`build-avx512`) under phi512 | yes, on every card but 0 | card 0: phi512 only (768 huge pages with the seamless pool); others as `cards` | 1 | off |

`auto` is `cards` when a card is up (`card_windows`: its
`/dev/shm/phi-hostmem*` window exists and its daemon answers, below), else
`x86`.

## Finding the sibling

The payload, the workers and phi512 are Intel-Phi-AVX512's. In order, the
first that exists (has `scripts/phi-vpu.sh`):

1. `PHI_AVX512_ROOT`.
2. A checkout next to this one (`CARGO_MANIFEST_DIR`'s parent), named
   `Intel-Phi-AVX512` (a `git clone`) or `Intel Phi AVX-512`.
3. The same two names in `$HOME`.

The family's lookup order is environment, `PATH`, next to this checkout,
`$HOME` (Mechanical Jev finds `xks` with all four); the sibling has no
command on `PATH`, so that step is skipped here.
`find_sibling` is the search, tested against a temporary tree.

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
(`nr_hugepages` to 0): 4.7 GiB per card at the cards site's 2400. `xks
stop` releases only the cards `xks_workers` names (a marker says xks
started their worker); `xks release` takes every card that is up.

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
  attention off to route around it. (2026-09-25: the sibling found a
  candidate cause, an unreserved thunk area inside a chunk the card maps
  without fetching, and fixed it; not re-run, since the test takes longer
  than the ten-minute budget. [Subproject 06](../docs/subprojects/06-avx512-parity.md).)
- card 0 serving phi512's regions and the payload's multiplies at once:
  after 32 minutes a multiply got no answer within 60 s. The one host
  thread waits on a card's multiply while its own AVX-512 region queues
  behind it on the same worker. The site now splits the cards by role:
  card 0 runs regions only, the payload uses the others.

The x86 site removes an inherited `GGML_BACKEND_PATH` (with a line saying
so): set in the shell or a config file, it loaded the payload into the
reference, and x86 against cards compared the payload with itself.

A worker is kept only when its marker, its polling and the card agree:
`phi-vpu.sh -c N config` (the sibling's) reports the card's huge-page
reservation and the running worker's arguments, and `ensure_worker`
restarts a worker whose reservation or arguments are not the ones xks
asked for. The marker alone could not see a worker something else had
restarted since (`phi512.sh` starts one with 768 pages and the seamless
pool; the cards site wants 2400 and none), and xks would have used it.

A card counts as up when its host window exists and its daemon accepts a
connection on the stack's control socket (`$XDG_RUNTIME_DIR/phictl/control.sock`
for card 0, `phictl/N/control.sock` for card N). The stack never unlinks
a window, so a card that is down used to count as up: `auto` picked the
cards site and failed at the worker. A socket file can outlive its daemon
too, but connecting to it is refused at once, so the check costs nothing
and needs no ssh.
