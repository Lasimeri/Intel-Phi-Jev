# backend/typesafe.rs: the hosted Jev

A client for `https://api.typesafe.ai/v1/systemone` (or
`TYPESAFE_BASE_URL`) with `TYPESAFE_API_KEY`, used by `xks query --compare`
to send the same request to the real Jev and print both answers. It is the
reference `xks` imitates; with a key, the same case file can be scored by
both and set side by side with `corroborate`-style reading. Not a `Scorer`:
Jev returns typed answers, not label log-probabilities.
