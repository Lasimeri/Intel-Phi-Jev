# artichoke/mod.rs: the interrogation engine

ARTICHOKE reads a subject's involuntary response: the next-token
distribution after the answer cue. It never lets the model generate. It
links llama.cpp in process through [`sys.rs`](sys.rs) (bindgen), so one
process holds the model, the session state and, on the `cards` site, the
payload.

## One request

1. Every fingerprint's full prompt is tokenized segment by segment
   (`tokenize_segs`): template text with special tokens parsed, user text
   without, the model's start token first. The session is the longest token
   prefix all prompts share, short of each prompt's last token
   (`fork_point`), so the fork falls wherever the tokens part and never at a
   tokenizer seam.
2. Sequence 0 is the rolling buffer: if it already holds a prefix of the
   session, only the rest is decoded. A recurrent state cannot be cut back,
   so a partial match on a hybrid model clears and starts over
   (`llama_memory_seq_rm` returns false and `llama_memory_clear` follows).
3. Fingerprints with one-token labels go in rounds of up to `forks`
   sequences: `llama_memory_seq_cp(0, k)` for each, then one batch holding
   every suffix, an output row only at each suffix's last token.
4. Fingerprints with several-token labels (past 26 options) go through
   `read_trie`: the suffix on sequence 1, read at its end (the root), then
   every distinct proper prefix of the labels forked from sequence 1 onto
   the other sequences and read one step further. A label's log-probability
   is the sum along the trie. Needs `--forks 2` or more.
5. `read` turns an output row into full-vocabulary log-probabilities of the
   label tokens (log-sum-exp over the row in f64).

## Context parameters, and why

| parameter | value | reason |
| --- | --- | --- |
| `kv_unified` | true | forks share the session's attention cells instead of copying them into per-sequence streams |
| `n_seq_max` | forks + 1 | sequence 0 plus one per fork; on a hybrid model each sequence also has its own recurrent state |
| `n_outputs_max` | forks + 1 | at most one output row per fork in a step; the default sizes the buffer for `n_batch` rows of the whole vocabulary (over a GB) |
| `use_extra_bufts` | off with the payload | a repacked weight lives in a CPU-only buffer the payload is never offered: 5.1 GB offered to the cards with repacking, 20.9 GB without (35B-A3B Q4_K_M) |
| threads | 12 with the payload, 16 without, 1 on `avx512` | the card daemons need host cores (16 costs 5x at one token); the AVX-512 build's OpenMP barrier spins while a region runs on the card |

## Backends

`load_backends`: a llama.cpp build with dynamic backends loads its best CPU
variant from the build directory and then `GGML_BACKEND_PATH` (the payload,
which [`../site.rs`](../site.rs) sets). A build without them
(`cfg(xks_static_cpu)`, the AVX-512 one) has its CPU backend linked in and
loads only the payload. The device list is printed at open (`xks: devices:
CPU` or `Phi, CPU`), then `xks: loading NAME (N GB)` before the model
load, the long step: a detached server's log says what it is doing while
it loads, and Mechanical Jev's TUI shows that line as a start's progress.

## Pages read in as used (2026-09-25)

llama.cpp maps a model with `MAP_POPULATE` (`llama_mmap`,
`src/llama-mmap.cpp`, prefetch on from `llama_model::load_tensors`), so a
load reads the whole file into the process. On the offloaded sites that
is memory the host never needed: the rows the cards keep are dropped from
the host after their upload (the payload's `drop_pages`), and some pages
no request touches (tokens never embedded, experts never routed to) are
held all the same. `Options::lazy_pages` (set by `main.rs` for the `cards`
and `avx512` sites unless `--no-offload`) raises `LAZY_PAGES` around
`llama_model_load_from_file`, and the `mmap` the xks binary exports
([`../main.md`](../main.md)) maps the subject without the flag: its pages
come in as the first request uses them, and the cards' rows pass through
the host one tensor at a time on their way to the cards. llama.cpp is not
modified. The `x86` site is left as it was.

Measured 2026-09-25, the 35B-A3B Q4_K_M at the cards site on 8095, xks's
resident memory from `/proc/PID/status`, `examples/query.json` three
times (the steady figure depends on how much of the model the requests
reach: another request can route to experts these three did not):

| | populated (before) | read in as used |
| --- | --- | --- |
| right after the load, a server nobody has asked yet | 21.3 GiB (20.2 of it the file) | 1.42 GiB (11 MB of the file) |
| the first request | 21 s | 30.7 s (the pages fault in as used) |
| after three requests | 14.1 GiB, peak 21.6 | 11.2 GiB, peak 11.2 |

On the avx512 site (the 0.5B, the inner xks under phi512 with libphi512
preloaded) the load left 13 MB of the file resident. A test maps a
file with `MAP_POPULATE` through the exported `mmap` and reads its
resident size from `/proc/self/smaps`: all of it with the flag clear,
none with it set.

## Measured

- The copy is exact: with one fork per round (so forked and split make the
  same decodes), forked against split is 0.0 on every fingerprint of the
  hybrid 35B-A3B (subproject 02).
- Cutting a prompt into two decodes moves the 35B's label log-probabilities
  by up to 0.8 (mean 0.3); a dense 0.5B moves 0.045. That is the subject's
  sensitivity to batch composition, not the engine.
- The trie against a brute-force control on 30 options: 0.032 max
  difference, same argmax, 0.54 s against 17.4 s (subproject 05).

## Readings for the polygraph

`read_control` (the whole prompt from an empty cache, no fork; for
several-token labels every label decoded token by token) and `read_split`
(the same two decodes a fork makes, without the copy) exist only to be
compared against the forked reading.

## Where the session ends, and what is refused first (2026-09-24)

The kept session (sequence 0) ends where the state does: the fork point is
where the fingerprints' tokens part, but never past the prefix's own
tokens. It used to be the longest common prefix of the rendered prompts,
which runs into question text (with one question, the whole prompt but one
token), so the same state asked with other questions matched only part of
the kept session; a recurrent state cannot be cut back (`seq_rm` refuses a
partial removal on the hybrid 35B), so the whole state was prefilled again.
Now a repeated state costs only its questions, as the top of this file says.

Before any prefill, `check_limits` refuses (422) a request past TypeSafe's
limits (the state and the longest question within 32,000 tokens, the whole
request within 64,000, counted with this subject's tokenizer, template
included) or one whose longest fingerprint with its longest label does not
fit the context (`--ctx`). An oversized state used to be prefilled first
and then fail as a 502, which the official SDKs retry.

A template that writes its own BOS (Llama 3's `<|begin_of_text|>`) after
the tokenizer has added one keeps one.

`n_ubatch` stays 512 (`--ubatch`, `XKS_UBATCH`): on 2026-09-25 the long
sessions (two cases, 16 questions, the 35B, x86 site) took 87.6 and 77.3 s
at 512 and 78.6 and 82.5 s at 2048, interleaved; the difference is inside
this host's drift.
