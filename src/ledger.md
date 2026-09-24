# ledger.rs: what the cards actually computed

`xks ledger LOG` (or stdin) totals a log written with `PHI_GGML_VERBOSE=1`:
per card, the multiplies it took part in, the rows, and its compute, pull
and push time; for the host, its own rows' time and its wait for the cards.

It exists because a card monitor cannot tell arithmetic from a worker
spinning between requests: `phi top` showed 57 threads per card busy while
the ledger of the same run said each card's vector units computed for 50.6
of 186 seconds (subproject 04). The ledger is the number to quote.

Only multiplies the payload handled appear; ones it declined at the
scheduler (unsupported types) run on llama.cpp's CPU backend and never reach
the log.
