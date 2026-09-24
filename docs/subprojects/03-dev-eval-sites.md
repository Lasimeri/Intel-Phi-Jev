# Subproject 03: x86 against the cards

`xks subproject run 03`. Record:
[`results/03-dev-eval-sites.json`](results/03-dev-eval-sites.json) (git
`6f944a0`, 2026-09-24T15:43:46Z, clean tree, 184 s).

**Question:** does moving the subject's matrix multiplies onto the Phi
cards change what it reads?

| site | accuracy | ECE | p50 latency per question | wall |
| --- | --- | --- | --- | --- |
| x86 | 0.800 | 0.165 | 1.56 s | 77.3 s |
| cards | 0.833 | 0.203 | 1.65 s | 95.1 s |

Corroborated question by question: **29 of 30** answers agree, mean
probability difference 0.028, largest 0.109, inside the subject's own
noise floor ([subproject 02](02-polygraph.md)). The cards change the
arithmetic's order and precision (float16 activations), not the reading.
They are slower here: the 35B's prompt work at Q4_K_M is faster on this
host than split with the cards (see [subproject 07](07-long-sessions-cards.md)
for what the cards compute).
