# Subproject 03: x86 against the cards

`xks subproject run 03`. Record:
[`results/03-dev-eval-sites.json`](results/03-dev-eval-sites.json) (git
`3ce3124`, 2026-09-26T00:40:13Z, clean tree, 203 s).

**Question:** does moving the subject's matrix multiplies onto the Phi
cards change what it reads, and what does it cost the host?

| site | accuracy | ECE | p50 latency per question | wall | host memory at peak |
| --- | --- | --- | --- | --- | --- |
| x86 | 0.833 | 0.200 | 1.58 s | 83.3 s | 19.95 GiB |
| cards | 0.800 | 0.163 | 1.66 s | 95.2 s | **12.50 GiB** |

Corroborated question by question: **29 of 30** answers agree, mean
probability difference 0.024, largest 0.085, inside the subject's own
noise floor ([subproject 02](02-polygraph.md)). The cards change the
arithmetic's order and precision (float16 activations), not the reading.
They are slower here: the 35B's prompt work at Q4_K_M is faster on this
host than split with the cards (see [subproject 07](07-long-sessions-cards.md)
for what the cards compute).

What the cards buy on this subject is host memory, not time: at its peak
the cards site held 12.5 GiB against 19.95 on x86 (`host_peak_gib`, the
eval process's `VmHWM`), the cards keeping 41 percent of the weights and
the subject's pages read in only as they are used
([`artichoke/mod.md`](../../src/artichoke/mod.md), "Pages read in as
used"). The one answer the sites disagree on flipped the other way from
the previous record (6f944a0, 2026-09-24: x86 0.800, cards 0.833, mean
difference 0.028), which is the noise floor at work, not a change.
