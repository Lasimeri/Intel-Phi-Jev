# Subproject 02: polygraph

`xks subproject run 02`. Record:
[`results/02-polygraph.json`](results/02-polygraph.json) (git `6f944a0`,
2026-09-24T15:41:44Z, clean tree, 117 s).

**Question:** does forking a session read the same thing as not forking?
On a hybrid subject the copy must carry the recurrent state as well as the
attention cells.

**Method:** every fingerprint of the first dev_tasks cases read forked,
split (the same cut, no copy) and control (the whole prompt in one decode)
([`polygraph.rs`](../../src/polygraph.md)), site `x86`.

| subject, forks | fingerprints | forked vs split | split vs control | forked vs control | same argmax |
| --- | --- | --- | --- | --- | --- |
| 35B-A3B, 15 | 12 | 0.672 (mean 0.299) | 0.781 (0.099) | 0.672 (0.286) | 12 of 12 |
| 35B-A3B, 1 | 3 | **0.000** | 0.034 (0.011) | | 3 of 3 |
| 0.5B dense, 15 | 18 | 0.009 (0.003) | 0.028 (0.007) | 0.028 (0.008) | 18 of 18 |

With one fork the forked and split readings are identical to the bit: the
copy is exact, recurrent state included. The rest is the mixture of
experts' sensitivity to decode shape (a batch beside other sequences, or a
prompt cut in two, moves label log-probabilities by up to 0.8; the dense
model by 0.03): a floor on how finely the 35B's probabilities can be read.
Timing is [subproject 04](04-long-sessions.md)'s claim, not this one's.
