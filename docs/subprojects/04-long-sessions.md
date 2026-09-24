# Subproject 04: long sessions, BLUEBIRD against ARTICHOKE

`xks subproject run 04`. Record:
[`results/04-long-sessions-x86.json`](results/04-long-sessions-x86.json)
(git `6f944a0`, 2026-09-24T15:46:58Z, clean tree, 356 s).

**Question:** what does forking the session buy on real long documents?
Two sessions from [`examples/long_sessions.jsonl`](../../examples/long_sessions.jsonl)
(3.6 to 5 KB of the sibling's result documents), eight questions each, the
35B-A3B on this host alone.

| | BLUEBIRD (llama-server) | ARTICHOKE (fork) |
| --- | --- | --- |
| accuracy | 16 of 16 | 16 of 16 |
| prompt tokens evaluated | 17,110 | **5,065** |
| p50 latency per question | 19.6 s | **2.8 s** |
| wall | 233.3 s | **76.4 s** |

The two agree on all 16 answers, mean probability difference 0.008: the
fork reads what the stock server reads, 3.1 times faster, a third of the
tokens. The difference grows with the session: BLUEBIRD re-reads the whole
document for every question on a hybrid subject.
