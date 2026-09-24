# backend/mod.rs: the Scorer trait

A backend answers one thing: the log-probabilities of candidate labels as
the continuation of a prompt.

- `score(prompt, candidates)`: one flattened prompt (user text escaped).
- `score_many(prefix, items)`: every fingerprint of one session, as
  template and user segments. The default flattens each and calls `score`
  one after another; ARTICHOKE overrides it to prefill the session once,
  fork it, and read every fingerprint in one batch.
- `multi_token_labels`: whether labels may be several tokens (Choice past
  26 options). Only ARTICHOKE (a trie of forks) says yes.

`impl Scorer for Box<dyn Scorer>` forwards every method, `score_many` and
`multi_token_labels` included; a forgotten forward silently falls back to
the default (sequential, one-token) behaviour.

| backend | file | what it is |
| --- | --- | --- |
| ARTICHOKE | [`../artichoke/mod.rs`](../artichoke/mod.md) | in process, llama.cpp, forks |
| BLUEBIRD | [`bluebird.rs`](bluebird.md) | a llama-server over HTTP |
| OpenAI | [`openai.rs`](openai.md) | chat/completions with logprobs |
| TypeSafe | [`typesafe.rs`](typesafe.md) | the hosted Jev, for `query --compare` |
