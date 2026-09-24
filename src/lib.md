# lib.rs: the crate

The modules and the path a request takes through them:

[`prompt`](prompt.md) renders the session and each fingerprint as template
and user segments; a [`backend`](backend/mod.md) (`Scorer`) reads the label
log-probabilities, ARTICHOKE ([`artichoke`](artichoke/mod.md)) by forking
the session; [`score`](score.md) normalises, conditions and derives
confidence; [`judge`](judge.md) ties them into typed answers;
[`server`](server.md) and [`mcp`](mcp.md) expose it; [`site`](site.md)
decides where the arithmetic runs. [`eval`](eval.md),
[`polygraph`](polygraph.md), [`corroborate`](corroborate.md),
[`ledger`](ledger.md) and [`subproject`](subproject.md) measure it;
[`config`](config.md) supplies defaults.

`artichoke`, `polygraph`, `site` and `subproject` need the `artichoke`
feature (llama.cpp linked); without it the crate still builds the HTTP
backends and the measurement tools.
