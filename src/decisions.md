# decisions.rs: the decision log

`xks serve --decision-log PATH` (or `XKS_DECISION_LOG`) appends one JSON
line per question set the server answers or refuses, so a decision can
be traced afterwards from what was asked to what was answered: the
record the write-ups on running Jev in production ask for first
(Mechanical Jev's `docs/uses.md`: every automated decision traceable from
the state and the probabilities to the outcome). Off unless asked for.

## A line

```json
{"utc": "2026-09-26T20:21:16Z", "status": 200, "ms": 602.5,
 "state": {"fnv64": "df474b2a93da15cf", "chars": 46},
 "questions": {"is_urgent": {"type": "noul", "fnv64": "bc6efa42c880c499"}},
 "model": "artichoke/Qwen3.8-2B-Q8_0",
 "answers": {"is_urgent": {"type": "noul", "noul": 0.8519}}}
```

By default the state and each question are kept as marks, not text: an
FNV-1a 64 hash of their JSON (`fnv64`, the published algorithm, stable
across runs and builds; the same request marks the same) and the state's
length. That says "the same state as before" and "a different question"
without the log holding what callers sent. A mark identifies; it does
not hide: FNV-1a is not a cryptographic hash and has no salt, so a short
or guessable state (a shell command, a stock phrase) can be found again
by hashing guesses. Treat the log as sensitive as the traffic whenever
the states are. `--log-text` (or
`XKS_DECISION_LOG_TEXT`) keeps the state and questions as sent instead,
which makes each line replayable. The answers are always kept: they are
the decision. A refused request (422, 502) keeps its `message` in place
of answers; a body that is not JSON has no `state`. Lines are appended
under a lock, so concurrent requests do not interleave; the file (and
its directory) is opened before the subject loads, so a path that cannot
be written is said at once. `xks mcp` does not log.

`utc` (moved here from `subproject.rs`, which re-exports it) writes the
time.

## Tests

`cargo test decisions`: the time format, FNV-1a against its published
vectors (the empty string, "a"), a line keeping marks and not text (and
text when asked), a refusal's message, lines appended.

Driven 2026-09-26: the 2B at 127.0.0.1:8095 with the log on; `mjev gate`
on TypeSafe's payouts example and a request without questions gave the
two lines of the example above and a 422, neither holding the state's
text.
