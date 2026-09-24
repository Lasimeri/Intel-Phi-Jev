# server.rs: the HTTP endpoint

`tiny_http`, one thread per request (the engine serialises on its own lock).
Routes: `POST /v1/systemone`, `GET /v1/models`, `GET /health` (with the
subject's name). Optional bearer keys (`--api-keys`). Status codes follow
TypeSafe: 401 for a missing or wrong key, 422 for a request that fails
validation; a backend failure is 502.

The kill date: with `--kill-date S`, the accept loop wakes every second and
exits once no request has arrived or been in flight for S seconds, which
drops the engine and leaves the cards free for the next process. The card
workers themselves keep their huge pages until `xks release` or `xks stop`.
