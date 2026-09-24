# Subproject 01: BLUEBIRD baseline

`xks subproject run 01`. Record:
[`results/01-bluebird-baseline.json`](results/01-bluebird-baseline.json)
(git `6f944a0`, 2026-09-24T15:38:22Z, clean tree, 202 s).

**What:** stock llama-server (x86, repacking off) with the 35B-A3B, asked
the first 10 dev_tasks cases (30 questions) through BLUEBIRD, one request
per fingerprint, Choice options averaged over 3 rotations.

| | value |
| --- | --- |
| accuracy | 0.833 |
| Brier | 0.243 |
| ECE | 0.155 |
| coverage at 5 percent error | 0.767 |
| prompt tokens evaluated | 7,820 |
| wall time | 156.6 s |

The hybrid subject cannot keep a recurrent state at the point where two
prompts part, so llama-server evaluates every fingerprint's whole prompt;
compare [subproject 04](04-long-sessions.md), where the same work forked
from one session costs a third of the tokens. Repacking is off: a repacked
copy beside the mapped 21.7 GB file thrashed an earlier run from 44 to 1.5
tokens a second on this 31 GB host.
