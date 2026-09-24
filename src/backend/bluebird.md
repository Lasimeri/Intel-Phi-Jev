# backend/bluebird.rs: the llama-server baseline

BLUEBIRD, the first program: one `POST /completion` per fingerprint with
`n_predict: 1` and `n_probs`, reading the top log-probabilities of the first
generated position and matching them against the labels (a label outside
the top N reads as minus infinity). From upstream (`llamacpp.rs`), renamed.

It is the baseline because it is what a llama-server user would do, and it
shows the cost ARTICHOKE removes: on a hybrid subject (recurrent state, the
35B-A3B) llama-server re-evaluates the whole prompt for every fingerprint,
since the state cannot be cut back to where two prompts part; the log shows
each question's full prompt length as evaluated (subproject 01).

Labels must be single tokens; a Choice past 26 options is refused by the
judge before it gets here.
