# eval.rs: labelled cases

A case file is JSONL (`#` lines and blank lines are skipped; an error
names the file and the line number an editor shows, and a file with no
case is an error): `state`, `questions` and
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

## Failures and labels (2026-09-24)

- A case the engine refuses for its size (past the context, or TypeSafe's
  32k and 64k limits) is a failed case like any other; only a case whose
  questions do not parse aborts the run. One long case used to throw away
  every row before it.
- A gold label names an option: a key (a Score's level number is its key;
  a number that is a key is that key), `true`/`false` for a Noul, else a
  number read as an index, and only an index inside the options. A 1-based
  level used to pass as an index past the end, and `xks condition` then
  panicked after the whole eval.
- Coverage at 5 percent error is cut only where the confidence changes:
  rows of equal confidence are accepted or refused together (and a NaN
  confidence no longer panics the sort).
