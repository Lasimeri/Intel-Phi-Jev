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
reads a record is `docs/subprojects/NN-*.md` (04's record is
`04-long-sessions-x86.json`, its page `04-long-sessions.md`). The subjects
are `XKS_SUBJECT` and `XKS_SUBJECT_SMALL` (the dense 0.5B), both from
`xks.conf`.

| id | name | compares |
| --- | --- | --- |
| 01 | bluebird-baseline | stock llama-server (x86, no repack) on the first 10 dev_tasks cases |
| 02 | polygraph | forked / split / control on the 35B, one fork (the copy alone), and the dense 0.5B floor |
| 03 | dev-eval-sites | `x86` against `cards` on the first 10 dev_tasks cases, corroborated |
| 04 | long-sessions-x86 | BLUEBIRD against ARTICHOKE on two long real sessions, eight questions each, this host alone |
| 05 | wide-choice-trie | 30 options, trie against brute force |
| 06 | avx512-parity | `avx512` (card 0 regions, card 1 the payload) against `x86`, dense 0.5B, one question; run by name only (`in_all` false: it does not finish inside the budget) |
| 07 | long-sessions-cards | all four long sessions on `x86` and on `cards`, the payload's ledger, corroborated |

Each subproject has `BUDGET`, 600 s: its children share one deadline, and
a child still running at it is killed and the record says "over budget"
with the log to read. `run all` skips a subproject whose `in_all` is false.

A subproject's timings are one run each. This host's throughput drifts by up
to a quarter over tens of minutes, so a timing claim closer than that needs
interleaved runs; the subprojects state accuracy and token counts, which do
not drift, and timings only where the gap is several times the drift.

`utc` formats a UNIX time with the civil-from-days algorithm rather than a
calendar crate.
