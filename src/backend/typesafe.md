# backend/typesafe.rs: the real Jev

Jev is TypeSafe's model and runs only on their servers: no weights are
published, so it cannot run on this host, the Phi cards, or llama.cpp. This
is the client for it: `POST https://api.typesafe.ai/v1/systemone` (or
`TYPESAFE_BASE_URL`) with `TYPESAFE_API_KEY`, model `jev-latest` unless
`TYPESAFE_DEFAULT_MODEL` or the request names another.

Used three ways:

- `xks --backend-kind jev query ...` (`make jev`): a request to the real Jev.
- `xks --backend-kind jev eval cases.jsonl --rows R` (`make jev-eval`): a
  labelled case file scored by the real Jev; `eval::run_jev` turns its
  answers' probabilities into the same rows a local run records, so
  `corroborate` sets Jev against the local imitation fingerprint by
  fingerprint.
- `xks query --compare`: the local answer and Jev's for the same request.

The key comes from console.typesafe.ai (the API has been public since 21
September 2026) and belongs in `xks.local.conf`, which is not tracked, or
the environment. Jev returns typed answers, not label log-probabilities,
so it is not a `Scorer`.
