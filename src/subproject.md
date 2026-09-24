# subproject.rs: every experiment as one command

`xks subproject list`, `xks subproject run 04`, `xks subproject run all`
(`make subprojects`). Each subproject runs `xks` itself as child processes:
one process at a time may hold the cards, each site wants its own process,
and a child's stderr becomes a log under `target/subprojects/`. BLUEBIRD
subprojects start `llama-server` (`XKS_LLAMA_SERVER`, port
`XKS_BLUEBIRD_PORT`) and stop it when done, even on failure (its guard kills
it on drop).

Each run writes `docs/subprojects/results/NN-name.json`: the subproject, the
UTC time it started, how long it took, the git revision and whether the
tree was dirty, the host, and every child's JSON output. The report that
reads a record is `docs/subprojects/NN-name.md`.

| id | name | compares |
| --- | --- | --- |
| 01 | bluebird-baseline | stock llama-server on dev_tasks |
| 02 | polygraph | forked / split / control on the 35B, one fork (the copy alone), and the dense 0.5B floor |
| 03 | dev-eval-sites | `x86` against `cards` on dev_tasks, corroborated |
| 04 | long-sessions | BLUEBIRD against ARTICHOKE on real long documents, the cards with the payload's ledger |
| 05 | wide-choice-trie | 30 options, trie against brute force |
| 06 | avx512-parity | `avx512` against `x86`, dense 0.5B, one question (the site is about a thousand times slower) |

A subproject's timings are one run each. This host's throughput drifts by up
to a quarter over tens of minutes, so a timing claim closer than that needs
interleaved runs; the subprojects state accuracy and token counts, which do
not drift, and timings only where the gap is several times the drift.

`utc` formats a UNIX time with the civil-from-days algorithm rather than a
calendar crate.
