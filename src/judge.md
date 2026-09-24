# judge.rs: a request in, typed answers out

The layout (`--layout letters|jev`, [`prompt.rs`](prompt.md)) decides how
each fingerprint is rendered; the jev layout needs a backend that reads
labels of several tokens. `--debug` reports, per question, `label_mass`:
the probability the subject put on the labels at all (mean over
rotations), well under one when a layout is off its distribution.

`raw` parses the questions (validating TypeSafe's limits: a Choice has at
most 255 options, a Score 2 to 10 levels), renders the session once and
every fingerprint (each question, and each rotation of its options when
`--permutations` asks for position-bias averaging), and hands them all to
the backend in one `score_many` call, so a backend that can fork the session
reads them together. A Choice past 26 options needs a backend with
multi-token labels (ARTICHOKE) and is refused by the others with a message
saying so.

`evaluate` conditions each question's distribution at its bucket's
temperature and builds the answer: Noul `P(yes)`, Choice the argmax with
every option's probability and confidence, Score the expected level with the
legend, per-level probabilities and confidence. `usage.input_tokens` counts
tokens the backend evaluated; a session served from the rolling buffer costs
nothing and shows under `debug`.
