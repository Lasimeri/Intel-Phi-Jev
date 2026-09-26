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

The kill date: with `--kill-date S` (`XKS_KILL_DATE`, 1800 in `xks.conf`
since 2026-09-25: 30 minutes), the accept loop wakes every second and
exits once no question has been asked or been in flight for S seconds.
Only `POST /v1/systemone` counts: `/health` and the rest do not, because
Mechanical Jev's TUI polls `/health` every 5 s, and before 2026-09-25 an
open TUI kept an idle server (the 35B's 11 to 13 GiB of host memory, and
the cards) alive for as long as it stayed open. Past the kill date the
engine is dropped and `main.rs` (`retire`) gives back what `xks stop`
would: the workers and huge pages of the cards this server used (none on
the x86 site), and its pid file. The listening line says when:
`stops itself after 30 min without a question (kill date)`.

`/health` carries `kill_date_s` (0: never) and `idle_s` (seconds since
the last question, or since the start) beside `status`, `subject`,
`site` and `cards`, so a client can say when the server will stop itself
(Mechanical Jev's server screen and its line at quit do).

Measured 2026-09-25, the 0.5B at the cards site on 8095, `XKS_KILL_DATE=40`,
one question, then `/health` polled every 2 s: the server exited 44 s
after the question, both cards with no worker and 0 huge pages, the pid
file gone. And from Mechanical Jev's TUI with `XKS_KILL_DATE=120`: the
server it started stopped itself after 2 min and released both cards.

With `--decision-log`, each question set is recorded after it is answered
or refused, with its time ([`decisions.md`](decisions.md)); the request
is parsed as JSON first, so the log keeps it (as marks or text) even when
it is not a valid request, and a syntax error keeps its line and column.
