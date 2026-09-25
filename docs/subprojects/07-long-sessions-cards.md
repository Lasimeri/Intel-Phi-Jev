# Subproject 07: long sessions on the cards

`xks subproject run 07`. Record:
[`results/07-long-sessions-cards.json`](results/07-long-sessions-cards.json)
(git `98980ec`, 2026-09-25T10:38:30Z, clean tree, 382 s).

**Question:** all four long sessions (32 questions) on the x86 site and on
the cards: do they agree, and what did the cards compute?

| site | accuracy | wall |
| --- | --- | --- |
| x86 | 32 of 32 | 168.7 s |
| cards | 32 of 32 | 188.4 s |

All 32 answers agree, mean probability difference 0.006 (largest 0.060).

The payload's ledger ([`ledger.rs`](../../src/ledger.md)) for the cards run:

| | value |
| --- | --- |
| multiplies through the payload | 25,522 |
| of them with rows on the cards | 6,240 |
| each card's vector compute | 61.2 s and 61.5 s of 188.4 s |
| the host's own rows | 98.3 s |
| the host waiting for the cards | 44.2 s |

Each card's 57 cores computed for a third of the run; the cards hold about
41 percent of the 35B's weights (4.36 GB each of 21.7 GB, the budget now
counted in the whole 2 MiB pages the card allocates), which bounds their
share. The 44 s the host spent waiting is why the cards site is slower
than x86 for this subject at Q4_K_M.

The previous record (6f944a0, 2026-09-24) gave the same picture: x86 160.1
s, cards 192.8 s, 32 of 32 agreeing (mean 0.0075), 6,552 multiplies with
rows on the cards, 63.2 s of card compute. The re-run is on the backend
after the review's fixes (a device only when the cards open, the budget
in pages, a mixture's batch share whole; the cards site runs offloaded,
where the mixture fix changes nothing) and xks's (a card up when its daemon
answers; a worker kept only when the card holds what xks asked for). A
run of it the day before, on the first version of the device count, had
reported "devices: CPU" and zero multiplies on the cards: the regression
it caught was fixed (Intel-Phi-AVX512 b603fc9) and that record discarded.
