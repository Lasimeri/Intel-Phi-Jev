# Intel Phi Jev: `xks`

**The real Jev** is TypeSafe's hosted model: it runs only on their servers
(no weights exist outside them), and `xks --backend-kind jev` (`make jev`)
calls it with your `TYPESAFE_API_KEY` from console.typesafe.ai, set in
`xks.local.conf`. Everything else here is a local imitation of its interface.

A local implementation of TypeSafe's Jev contract, the first "System One"
model: send a **state** and typed **questions** (Noul, Choice, Score), get back
typed answers with probability distributions and confidence, never
generated text. `xks` serves the same wire format as TypeSafe
(`POST /v1/systemone`, [docs.typesafe.ai/api](https://docs.typesafe.ai/api.md)),
so the official SDKs work against it by pointing `TYPESAFE_BASE_URL` at it.

What is different from the hosted Jev: the model is an open-weight LLM on
this machine (default Qwen3.8-35B-A3B), read by its next-token distribution,
and its arithmetic runs on this host together with its two Xeon Phi 3120
cards through the sibling
[Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512) repository.
TypeSafe trains Jev with RLCD; `xks` cannot, and conditions (calibrates) the
readings after the fact instead.

## One command each

```sh
make build          # both binaries: x86-64 and AVX-512 builds of llama.cpp
make serve          # background server, subject and site from xks.conf
curl -s localhost:8090/v1/systemone -d @examples/query.json | jq .
make stop           # stop it and release the cards' huge pages
make subprojects    # re-run every experiment, records in docs/subprojects/results/
make check          # docs, format, lint, build, tests
```

Defaults live in [`xks.conf`](xks.conf); put a machine's own values in
`xks.local.conf` (not tracked). Anything in the environment wins.

## How a request is answered

1. The state is rendered once as a document between template text, the
   **session**, and each question as a suffix whose next token is an option
   label, a **fingerprint** ([`src/prompt.rs`](src/prompt.rs)). User text is
   its own segment and is tokenized with special tokens off, so nothing a
   caller sends can close the document or open a chat turn.
2. **ARTICHOKE** ([`src/artichoke/mod.rs`](src/artichoke/mod.rs)) prefills
   the session once on sequence 0, copies it to one sequence per
   fingerprint (`llama_memory_seq_cp`; on a hybrid model this copies the
   recurrent state too, bit-exact), and decodes every suffix in one batch:
   the local form of Jev's "every question in parallel, in one pass".
   Sequence 0 keeps the session (the rolling buffer), so the same state
   asked again costs only the suffixes. Labels past 26 options (TypeSafe
   allows 255) are read as a trie of forks.
3. The label log-probabilities are normalised, conditioned, and turned into
   typed answers with TypeSafe's own confidence formulas
   ([`src/score.rs`](src/score.rs), from their `system-one-adapter`).

## Sites: where the arithmetic runs

| site | what runs where |
| --- | --- |
| `x86` | this host alone, llama.cpp's standard x86-64 build: the reference for accuracy and comparison |
| `cards` | the payload (`libggml_phi.so`) installed into `xks`: every weight matrix row the cards can hold is multiplied on their vector units, 57 threads per card, one per core; the rest on the host |
| `avx512` | llama.cpp's AVX-512 build under the sibling's phi512 wrapper, which runs each AVX-512 region the host refuses on card 0, with the payload for the multiplies |
| `auto` | `cards` when a card is up, else `x86` |

Capacity sets the share: the cards hold about 4.4 GB of weights each, so a
21.7 GB subject keeps most of its rows on the host. `xks ledger` totals what
each card actually computed; a busy-looking card monitor is not evidence,
since a waiting worker spins.

## Commands

| command | does |
| --- | --- |
| `xks serve [--detach] [--kill-date S]` | the Jev endpoint; `xks stop` ends a detached one and releases the cards |
| `xks query --file req.json` | one request; `--compare` also asks the hosted Jev (needs `TYPESAFE_API_KEY`) |
| `xks mcp` | an MCP server over stdio, one tool, `judge` |
| `xks eval cases.jsonl [--rows R]` | accuracy, Brier, ECE, coverage, latency on labelled cases |
| `xks condition cases.jsonl` | fit a conditioning (per-bucket temperatures) |
| `xks replay R [--fit out.json]` | recorded readings through a conditioning, without the subject |
| `xks polygraph cases.jsonl` | every fingerprint read forked, split and control, compared |
| `xks corroborate A B` | two recorded runs compared fingerprint by fingerprint (sites) |
| `xks ledger LOG` | what the payload's verbose log says each card computed |
| `xks subproject list / run NN / run all` | the experiments |
| `xks release` | stop the card workers, free their huge pages |

## Naming

Three lenses, each name chosen for what its original did.

| name | from | what it was | what it is here |
| --- | --- | --- | --- |
| `xks` | XKEYSCORE | NSA system that runs classification rules over captured sessions where they were collected | the engine and binary: typed questions run over a state |
| session | XKEYSCORE | a reconstructed network session, the unit the rules run over | a request's state, the shared prefix every fingerprint forks from |
| fingerprint | XKEYSCORE | a named rule that tags the sessions it matches | one typed question, rendered as a prompt suffix |
| rolling buffer | XKEYSCORE | the collection site's short-lived store of recent sessions | sequence 0, the last session kept prefilled |
| site | XKEYSCORE | where the data stays and the queries go to it | where the subject's arithmetic runs; the cards keep weight rows and the activations travel to them |
| corroborate | intelligence practice | confirming a reading with a second source | two recorded runs compared |
| ARTICHOKE | CIA, 1951 to 1953 | interrogation program: getting what a subject knows without its cooperation | the engine: reads the subject's involuntary next-token distribution and never lets it answer |
| BLUEBIRD | CIA, 1950 | the first program, ARTICHOKE's predecessor | the llama-server baseline |
| subject | MKULTRA | the person under study | the model |
| conditioning | MKULTRA | behaviour modification | calibration temperatures |
| Subproject NN | MKULTRA | 149 numbered subprojects, each with its own report | each experiment, its record and its report |
| polygraph | CIA practice | relevant questions read against control questions | forked readings against split and control ones |
| dropper | Stuxnet | the component that installs the payload into its target | [`src/site.rs`](src/site.rs): installs `libggml_phi.so` into `xks` itself |
| payload | Stuxnet | the code that acted on the target controllers | `libggml_phi.so`, the sibling's ggml backend for the cards |
| replay | Stuxnet | recorded normal readings played back to the operators | recorded readings played back through a conditioning |
| kill date | Stuxnet | the date the worm stopped itself | `serve --kill-date`: idle seconds after which the server exits and the cards are free |

The wire names (`/v1/systemone`, `state`, `questions`, `noul`, `choice`,
`score`) are TypeSafe's and stay as they are.

## Measured

Every number comes from a subproject record under
[`docs/subprojects/results/`](docs/subprojects/results/), each with its
git revision, time and configuration, read in
[`docs/subprojects/`](docs/subprojects/README.md).

## Limits

- Not Jev: no RLCD training, an open-weight subject, and the speed of this
  hardware (seconds per request on the 35B, not 70 to 500 ms).
- The cards hold about 8.8 GB of weights between them; a larger subject
  keeps most of its arithmetic on the host.
- The `avx512` site is correct but slow (each AVX-512 region is a round
  trip to the card), and some AVX-512 forms are not yet in the card's
  instruction table (byte lanes, `vpaddb`, stop a q4_0 subject).
- The 35B's readings move by up to 0.8 in label log-probability with how
  the prompt is cut into decodes (a dense 0.5B moves 0.045): a noise floor
  on how finely its probabilities can be read.

## Provenance

`xks` began as an import of [jev-rs](https://github.com/yijunyu/jev-rs)
(pinned in [`UPSTREAM`](UPSTREAM)); the semantics follow
[docs.typesafe.ai](https://docs.typesafe.ai/) and TypeSafe's MIT-licensed
`system-one-adapter`. See [`NOTICE`](NOTICE). MIT or Apache-2.0.
