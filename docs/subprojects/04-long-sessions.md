# Subproject 04: long sessions, BLUEBIRD against ARTICHOKE

`xks subproject run 04`. Record:
[`results/04-long-sessions-x86.json`](results/04-long-sessions-x86.json)
(git `4bfa67b`, 2026-09-26T03:41:45Z, clean tree, 315 s).

**Question:** what does forking the session buy on real long documents?
Two sessions from [`examples/long_sessions.jsonl`](../../examples/long_sessions.jsonl)
(3.6 to 5 KB of the sibling's result documents), eight questions each, the
35B-A3B on this host alone.

| | BLUEBIRD (llama-server) | ARTICHOKE (fork) |
| --- | --- | --- |
| accuracy | 16 of 16 | 16 of 16 |
| prompt tokens evaluated | 17,110 | **5,149** |
| p50 latency per question | 19.7 s | **2.7 s** |
| wall | 233.7 s | **75.6 s** |

The two agree on all 16 answers, mean probability difference 0.004: the
fork reads what the stock server reads, 3.1 times faster, a third of the
tokens. The difference grows with the session: BLUEBIRD re-reads the whole
document for every question on a hybrid subject.

Re-run on 4bfa67b (2026-09-26 UTC); on 6f944a0 (2026-09-24) it was 233.3
against 76.4 s, 17,110 against 5,065 tokens, mean difference 0.008.
ARTICHOKE now evaluates 84 more tokens because the session ends at the
state (82439bb) and each fingerprint's suffix carries the whole question.
ARTICHOKE's eval held 21.77 GiB at its peak (`host_peak_gib`: the x86
site maps the whole subject); BLUEBIRD's record shows only its client,
the subject living in the llama-server process.
