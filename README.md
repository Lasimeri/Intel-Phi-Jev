# Intel Phi Jev: `xks`

A local implementation of TypeSafe's Jev contract, the first "System One"
model: send a **state** and typed **questions** (Noul, Choice, Score), get back
typed answers with probability distributions and confidence, never
generated text. `xks` serves the same wire format as TypeSafe
(`POST /v1/systemone`, [docs.typesafe.ai/api](https://docs.typesafe.ai/api.md)),
so [Mechanical Jev](https://github.com/Lasimeri/Mechanical-Jev) (`mjev`)
and the official SDKs work against it by pointing `TYPESAFE_BASE_URL` at it.

What is different from the hosted Jev: the model is an open-weight LLM on
this machine (default Qwen3.8-35B-A3B), read by its next-token distribution,
and its arithmetic runs on this host together with its two Xeon Phi 3120
cards through the sibling
[Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512) repository.
TypeSafe trains Jev with RLCD; `xks` cannot, and conditions (calibrates) the
readings after the fact instead.

## One command each

The whole family is set up from Mechanical Jev, the asking side: `make
setup` there builds and links both commands and checks this repository
and the cards through `xks doctor`. Here alone:

```sh
make install        # build, then link xks into ~/.local/bin (PREFIX=...; never over a file)
xks doctor          # what is missing for xks to answer, and the fix for each (--fix: build, link)
make build          # xks against llama.cpp's x86-64 build, and its AVX-512 build when present
make serve          # background server, subject and site from xks.conf (shows its progress)
make query          # examples/query.json, answered by that server
curl -s localhost:8090/v1/systemone -d @examples/query.json | jq .   # the same over HTTP
make stop           # stop it and release the huge pages of the cards xks started
make subprojects    # re-run every experiment, records in docs/subprojects/results/
make check          # docs, format, lint, build, tests
```

Defaults live in [`xks.conf`](xks.conf); put a machine's own values in
`xks.local.conf` (not tracked). Anything in the environment wins.

A plain `xks query` goes to the server running at `XKS_BIND` when there
is one, instead of loading the subject a second time; `--local`, or any
engine option on its command line, keeps it in its own process. A request
is checked before anything slow starts, and one process at a time may
hold the cards: a second one (an `eval` beside a server on the cards) is
refused at once with the first one's pid, not left to break it
([`src/main.md`](src/main.md), [`src/site.md`](src/site.md)).

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
| `xks serve [--detach] [--kill-date S] [--decision-log F]` | the Jev endpoint; it stops itself after 30 min without a question (`XKS_KILL_DATE`), giving the cards back, and `xks stop` ends a detached one at once |
| `xks query --file req.json` | one request (a running server answers it; `--local` loads the subject here); `--compare` also asks the hosted Jev (needs `TYPESAFE_API_KEY`) |
| `xks mcp` | an MCP server over stdio, one tool, `judge` |
| `xks eval cases.jsonl [--rows R]` | accuracy, Brier, ECE, coverage, latency on labelled cases |
| `xks condition cases.jsonl` | fit a conditioning (per-bucket temperatures) |
| `xks replay R [--fit out.json]` | recorded readings through a conditioning, without the subject |
| `xks polygraph cases.jsonl` | every fingerprint read forked, split and control, compared |
| `xks corroborate A B` | two recorded runs compared fingerprint by fingerprint (sites) |
| `xks ledger LOG` | what the payload's verbose log says each card computed |
| `xks subproject list / run NN / run all` | the experiments |
| `xks release` | stop the card workers, free their huge pages (refused while a process holds the cards) |
| `xks doctor [--fix]` | what this machine has of what xks needs and the fix for what is missing; exit 0 ready, 1 not ([`src/doctor.md`](src/doctor.md)) |

## Naming

### The foundation: Revelation

"Therefore whosoever heareth these sayings of mine, and doeth them, I will
liken him unto a wise man, which built his house upon a rock" (Matthew
7:24); the one who built on the sand saw it fall (7:26 to 27). The names
of this project stand on one book, the Revelation to John, and the lenses
below are built on it: they name parts, it names what the parts are for.
Quotations are the King James Version.

| name | Revelation | what it was | what it is here |
| --- | --- | --- | --- |
| Apocalypse | 1:1 | the book's own name, *apokalypsis*, an unveiling: what was hidden, shown | what `xks` is for: the subject's hidden judgment shown as its whole distribution, never as words it chose to say |
| the sealed book | 5:1 | "a book written within and on the backside, sealed with seven seals", the seals opened one by one | the session: the state sealed in its own segment and read once; each fingerprint opens it once, and opening one leaves the others as they were (each fork is a copy) |
| Alpha and Omega | 1:8, 22:13 | "the beginning and the end, the first and the last" | the lettered options, A onward (the `letters` layout): an answer is a letter |
| the pair of balances | 6:5 | the third seal's rider, "a pair of balances in his hand" | the confidence formulas ([`src/score.rs`](src/score.rs)): an answer's distribution weighed |
| the sea of glass | 4:6 | "a sea of glass like unto crystal" before the throne | every answer's whole distribution returned, clear to the bottom: never a verdict alone |
| the measuring reed | 11:1, 21:15 | the reed given to "measure the temple of God", the golden reed that measured the city | the subprojects: every claim measured, the measure recorded beside it |
| the books opened | 20:12 | "judged out of those things which were written in the books, according to their works" | the records ([`docs/subprojects/results`](docs/subprojects/results/)) and the ledger: a claim is judged by what was written when it was measured |
| time no longer | 10:6 | "that there should be time no longer" | the end of a server's time (`serve --kill-date`, which the Stuxnet lens also names) |
| the foundations of the wall | 21:14, 21:19 | the city wall's twelve foundations, "garnished with all manner of precious stones" | `make check`: what every change stands on (the documentation rules, format, lint, both builds, the tests) |

### The lenses, built on it

Four lenses, each name chosen for what its original did. The fourth is the
Gateway Process, from "Analysis and Assessment of Gateway Process" (Lt.
Col. Wayne M. McDonnell, US Army, 1983; released through the CIA's
reading room in 2003), on the Monroe Institute's Gateway Experience. Its
names are in doc comments and here, not identifiers, which stay as they
are.

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
| kill date | Stuxnet | the date the worm stopped itself | `serve --kill-date`: seconds without a question after which the server exits and the cards are free (30 min by default) |
| Gateway Affirmation | Gateway Process | the statement recited to open every session | the fixed text every session opens with (`SYSTEM`, `JEV_PREAMBLE` in [`src/prompt.rs`](src/prompt.rs)) |
| Energy Conversion Box | Gateway Process | a container the practitioner sets distracting concerns in before a session, so nothing in them intrudes | the caller's text as its own segment, tokenized with special tokens off: nothing a caller sends can act on the session |
| Resonant Energy Balloon (REBAL) | Gateway Process | a protective field set up around the practitioner | `check_limits`: a request past the context or TypeSafe's limits is refused before it reaches the engine |
| Patterning | Gateway Process | fixing an intended outcome before it happens | the answer cue: the answer's form opened, so the subject's next token can only be the decision |
| Hologram | Gateway Process | the model of the universe in which every part holds the whole | every fork holds the whole session (`llama_memory_seq_cp`) |
| Hemi-Sync | Gateway Process | sound that brings the brain's two hemispheres into balance | option rotations averaged (`XKS_PERMUTATIONS`): the pull toward one end of the list balanced out |

The wire names (`/v1/systemone`, `state`, `questions`, `noul`, `choice`,
`score`) are TypeSafe's and stay as they are.

## Measured

Every number comes from a subproject record under
[`docs/subprojects/results/`](docs/subprojects/results/), each with its
git revision, time and configuration, read in
[`docs/subprojects/`](docs/subprojects/README.md). The 35B-A3B at Q4_K_M:

| what | result |
| --- | --- |
| the fork's copy, recurrent state included (02) | bit-exact (0.000) |
| two long sessions, BLUEBIRD against ARTICHOKE (04) | 234 s against **76 s**, 17,110 against 5,149 tokens, same 16 answers |
| x86 against the cards (03, 07) | 29 of 30 (mean difference 0.024) and 32 of 32 (0.006) answers agree, on `3ce3124`; each card computes for a third of the run |
| host memory at peak, x86 against the cards (03, 07, `host_peak_gib`) | 20.0 and 21.8 GiB against **12.5 and 13.3 GiB**: the cards hold 41 % of the weights, and the subject's pages are read in as used |
| wall, x86 against the cards (03, 07) | 83 and 153 s against 95 and 189 s: on this subject the cards save host memory, not time |
| a 30-option Choice as a trie (05) | brute force to 0.020, 22 times faster, 29 times fewer tokens |
| Jev's own 28 published questions (Mechanical Jev `make closeness`) | **28 of 28** of Jev's decisions, mean probability difference 0.113; 28 of 28 and 0.120 re-run on 82439bb, and again on 4bfa67b at the cards site |

## Limits

- Not Jev: no RLCD training, an open-weight subject, and the speed of this
  hardware (seconds per request on the 35B, not 70 to 500 ms).
- The cards hold about 8.8 GB of weights between them; a larger subject
  keeps most of its arithmetic on the host.
- The `avx512` site is correct but slow (each AVX-512 region is a round
  trip to the card), and some AVX-512 forms are not yet in the card's
  instruction table (byte lanes, `vpaddb`, stop a q4_0 subject); it runs
  with flash attention off, whose fault has a candidate fix on the
  sibling's side not yet re-run
  ([subproject 06](docs/subprojects/06-avx512-parity.md)).
- The 35B's readings move by up to 0.99 in label log-probability with how
  the prompt is cut into decodes (the small subject, Qwen3.8 2B Q8_0,
  up to 0.21; the dense Qwen2.5 0.5B it replaced, 0.05; subproject 02,
  2026-09-26 UTC), every answer the same: a noise floor on how finely
  their probabilities can be read.

## The hosted Jev

TypeSafe's own Jev runs only on their servers (no weights exist outside
them). `xks --backend-kind jev` (`make jev`, `make jev-eval`) sends the
same requests to it for comparison, with a `TYPESAFE_API_KEY` from
console.typesafe.ai in `xks.local.conf` (not tracked). Nothing else here
needs it: Mechanical Jev's `make closeness` measures `xks` against Jev's
published answers instead.

## The repositories

| repository | what | how it is found |
| --- | --- | --- |
| [Intel-Phi-3120A](https://github.com/Lasimeri/Intel-Phi-3120A) | the cards' software stack: boots them, serves their memory, the `phi` command | by Intel-Phi-AVX512 |
| [Intel-Phi-AVX512](https://github.com/Lasimeri/Intel-Phi-AVX512) | the cards as an AVX-512 co-processor: the payload, the workers, phi512 | `PHI_AVX512_ROOT`, else a checkout next to this one, else in `$HOME` |
| Intel-Phi-Jev (this one) | `xks`, the local Jev | `MJEV_XKS`, else `xks` on PATH, else a checkout next to Mechanical-Jev, else in `$HOME` |
| [Mechanical-Jev](https://github.com/Lasimeri/Mechanical-Jev) | `mjev`, the asking side, and Jev reverse engineered from its docs | |

Cloned side by side, the repositories find each other without
configuration, under each one's clone name (`Intel-Phi-AVX512`) or the
spaced one (`Intel Phi AVX-512`). What does need setting: the model and
llama.cpp paths in [`xks.conf`](xks.conf), and for the cards, the stack's
`phi` command with a card up. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the
rules they share.

## Provenance

`xks` began as an import of [jev-rs](https://github.com/yijunyu/jev-rs)
(pinned in [`UPSTREAM`](UPSTREAM)); the semantics follow
[docs.typesafe.ai](https://docs.typesafe.ai/) and TypeSafe's MIT-licensed
`system-one-adapter`. See [`NOTICE`](NOTICE). MIT or Apache-2.0.
