# eval.rs: labelled cases

A case file is JSONL (`#` lines are comments): `state`, `questions` and
`gold` (question id to the expected option key, level index, or `"yes"` /
`"no"`). `run` asks every case and keeps, per question, the raw label
log-probabilities and the gold index (`Row`, which `--rows` writes and
`replay` and `corroborate` read back). `metrics` gives accuracy, Brier,
top-label ECE over 10 bins, mean confidence, coverage at 5 percent error
(the largest share of questions answerable, most confident first, keeping
the error at or under 5 percent), latency percentiles, token counts, and
accuracy per question kind.

Latency per question is the request's time shared out among its
fingerprints when the backend reads them together.
