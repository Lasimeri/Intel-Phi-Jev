# server.rs: the HTTP endpoint

`tiny_http`, one thread per request (the engine serialises on its own lock).
Status codes follow TypeSafe: 401 for a missing or wrong key, 422 for a
request that fails validation; a backend failure is 502.

| route | answers |
| --- | --- |
| `POST /v1/systemone` | the Jev wire format ([`protocol.md`](protocol.md)) |
| `GET /health` | `status`, `subject`, `site` (`x86`, `cards`, `avx512`, `remote`) and `cards` (the card indices the subject uses) |
| `GET /v1/models` | the subject's id and `jev-latest` as its alias |
| `GET /` | what this is: the service, subject, site, the routes, the repository |

`site` and `cards` in `/health` and the index were added 2026-09-25;
`status` and `subject` are as they were (Mechanical Jev reads them).

Matching is a pure function (`route`, tested): a trailing slash is the
same path; a known path asked with the other method is a 405 with an
`Allow` header and a message naming the method it takes (a `GET
/v1/systemone` from a browser used to be a bare 404); an unknown path is
a 404 that points at `POST /v1/systemone` and `GET /`. An empty body is a
422 that says what to send, not a JSON parser's "EOF while parsing"; the
401 says the key goes in `Authorization: Bearer KEY`.

Optional bearer keys (`--api-keys`); `xks query` sends the first of
`XKS_API_KEYS` when it hands a request to a running server
([`main.md`](main.md)).

The kill date: with `--kill-date S`, the accept loop wakes every second and
exits once no request has arrived or been in flight for S seconds, which
drops the engine and leaves the cards free for the next process. The card
workers themselves keep their huge pages until `xks release` or `xks stop`.
