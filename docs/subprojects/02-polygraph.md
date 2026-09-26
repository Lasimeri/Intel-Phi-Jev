# Subproject 02: polygraph

`xks subproject run 02`. Record:
[`results/02-polygraph.json`](results/02-polygraph.json) (git `9ce2915`,
2026-09-26T15:39:00Z, clean tree, 178 s).

**Question:** does forking a session read the same thing as not forking?
On a hybrid subject the copy must carry the recurrent state as well as the
attention cells.

**Method:** every fingerprint of the first dev_tasks cases read forked,
split (the same cut, no copy) and control (the whole prompt in one decode)
([`polygraph.rs`](../../src/polygraph.md)), site `x86`.

| subject, forks | fingerprints | forked vs split | split vs control | forked vs control | same argmax |
| --- | --- | --- | --- | --- | --- |
| 35B-A3B, 15 | 12 | 0.728 (mean 0.279) | 0.703 (0.229) | 0.989 (0.281) | 12 of 12 |
| 35B-A3B, 1 | 3 | **0.000** | 0.703 (0.300) | 0.703 (0.300) | 3 of 3 |
| Qwen3.8 2B Q8_0 (the small subject), 15 | 18 | 0.142 (0.049) | 0.122 (0.043) | 0.213 (0.057) | 18 of 18 |

With one fork the forked and split readings are identical to the bit: the
copy is exact, recurrent state included. The rest is the mixture of
experts' sensitivity to decode shape (a batch beside other sequences, or a
prompt cut in two, moves label log-probabilities by up to 0.99; the small
subject by 0.21): a floor on how finely the 35B's probabilities can be read.
Every answer (the argmax) is the same all three ways.

The small subject is Qwen3.8 2B Q8_0 since 2026-09-26 (the user's choice,
`xks.conf`), a hybrid like the 35B, so the second floor is no longer a
dense model's: the dense Qwen2.5 0.5B f16 it replaced moved 0.016, 0.049
and 0.049 (means 0.004, 0.011, 0.011) over the same 18 fingerprints on
87a89d8, 18 of 18 the same answer. The record's key for it is still
`dense_small` (a field name, kept). The 35B's rows are the same as
87a89d8's to the last digit.

The previous record (6f944a0, 2026-09-24) had split against control at
0.034 with one fork, and the floor at 0.8. The difference is where the
prompt is cut, not the copy: since 82439bb the session ends at the state
("the session ends at the state", [`prompt.rs`](../../src/prompt.md)), so
the second decode carries the whole question where it used to carry only
the answer cue, and the 35B's reading moves with that cut (the 0.5B's by
0.05 against 0.03 before). The copy's exactness, the claim this
subproject exists for, is unchanged.
Timing is [subproject 04](04-long-sessions.md)'s claim, not this one's.
