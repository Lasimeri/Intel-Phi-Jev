# Subproject 07: long sessions on the cards

`xks subproject run 07`. Record:
[`results/07-long-sessions-cards.json`](results/07-long-sessions-cards.json)
(git `6f944a0`, 2026-09-24T15:53:22Z, clean tree, 360 s).

**Question:** all four long sessions (32 questions) on the x86 site and on
the cards: do they agree, and what did the cards compute?

| site | accuracy | wall |
| --- | --- | --- |
| x86 | 32 of 32 | 160.1 s |
| cards | 32 of 32 | 192.8 s |

All 32 answers agree, mean probability difference 0.0075.

The payload's ledger ([`ledger.rs`](../../src/ledger.md)) for the cards run:

| | value |
| --- | --- |
| multiplies through the payload | 25,522 |
| of them with rows on the cards | 6,552 |
| each card's vector compute | 63.2 s of 192.8 s |
| rows computed per card | 2,535,936 |
| the host's own rows | 98.1 s |
| the host waiting for the cards | 44.6 s |

Each card's 57 cores computed for a third of the run; the cards hold about
41 percent of the 35B's weights (4.4 GB each of 21.7 GB), which bounds
their share. The 45 s the host spent waiting is why the cards site is
slower than x86 for this subject at Q4_K_M.
