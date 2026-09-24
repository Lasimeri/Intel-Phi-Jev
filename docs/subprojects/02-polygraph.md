# Subproject 02: polygraph

`xks subproject run 02`. Record:
[`results/02-polygraph.json`](results/02-polygraph.json) (git `6f7b372`,
2026-09-24T12:25:44Z, clean tree).

**Question:** does forking a session (sequence 0 copied per fingerprint,
suffixes in one batch) read the same thing as not forking? On a hybrid
subject the copy must carry the recurrent state as well as the attention
cells, and a wrong copy would read plausibly and wrongly.

**Method:** every fingerprint of the first dev_tasks cases read three ways
([`polygraph.rs`](../../src/polygraph.md)): forked (what `serve` does), split
(the same cut on sequence 0, no copy), control (the whole prompt in one
decode). Site `x86`. Three runs: the 35B-A3B with the default 15 forks, the
35B with one fork per round (so forked and split make identical decodes),
and a dense 0.5B (no recurrent state, no experts) as the floor.

**Result** (largest absolute difference in label log-probability, mean in
brackets):

| subject, forks | fingerprints | forked vs split | split vs control | forked vs control | same argmax |
| --- | --- | --- | --- | --- | --- |
| 35B-A3B, 15 | 18 | 1.342 (0.263) | 0.485 (0.154) | 1.342 (0.258) | 17 of 18 |
| 35B-A3B, 1 | 6 | **0.000 (0.000)** | 0.303 (0.113) | | 6 of 6 |
| 0.5B dense, 15 | 18 | 0.009 (0.003) | 0.028 (0.007) | 0.028 (0.008) | 18 of 18 |

**Reading it:**

- With one fork the forked and split readings are identical to the bit on
  a hybrid MoE subject: the copy, recurrent state included, is exact.
- Everything else is the subject's arithmetic, not the engine. The same
  prompt read with a different decode shape (cut in two, or batched beside
  other sequences) moves the 35B-A3B by up to 1.3 in label log-probability,
  and the dense 0.5B by 0.03 at most. A mixture of experts picks 8 of 256
  experts per token by a near tie often enough that a last-bit difference
  in a batch changes a choice, and the difference compounds through 40
  layers.
- That sets a floor: the 35B's probabilities cannot be read more finely
  than this, by any engine that batches. One of 18 argmaxes flipped.
- The labels carry 0.96 to 0.97 of the probability on both subjects (mean
  label mass), so the prompt is on the subjects' distribution.

Timing is not this subproject's claim (the forked readings were not faster
on states of a few tokens, where there is almost no session to share, and
another run shared the host); [subproject 04](04-long-sessions.md) measures
it on long sessions.
