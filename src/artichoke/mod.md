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
CPU` or `Phi, CPU`).

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
