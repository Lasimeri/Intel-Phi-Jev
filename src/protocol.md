# protocol.rs: the wire format

TypeSafe's `POST /v1/systemone` (docs.typesafe.ai/api), field for field:
request `state` (string, object or array), optional `model`, `questions`
(id to question); questions `noul` (optional `criteria` with `true` and
`false`), `choice` (`criteria` as option to description or null), `score`
(`criteria` as an ordered array); answers keyed by the same ids with `type`,
and `usage`. Question ids are never shown to the subject, as TypeSafe
specifies.

`instructions` and criteria may be objects or arrays (TypeSafe's structured
form); they reach the prompt as compact JSON. `debug` is an `xks` extension,
present only with `--debug`.
