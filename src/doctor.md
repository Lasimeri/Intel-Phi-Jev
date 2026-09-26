# doctor.rs: what this machine has of what xks needs

`xks doctor` (and `make doctor`) walks the chain a question needs, from
this binary down to the cards, and prints one line per finding:

| tag | meaning |
| --- | --- |
| `ok` | in place |
| `note` | worth knowing; nothing waits on it (an optional piece, a fix that is a convenience) |
| `MISS` | a question cannot be answered until it is fixed; the fix is printed under it |

It ends with `ready: xks can answer (site S)` or `not ready: N to fix`,
and exits 0 or 1 accordingly. That exit code and the text are what
Mechanical Jev's `mjev doctor` relays (the family's one setup command,
`make setup` there): add to this, do not rename it.

What it looks at, in order:

| what | how | blocks when |
| --- | --- | --- |
| xks | `PATH` resolved and canonicalised against this binary | never (Mechanical Jev finds a checkout without it) |
| PATH | whether `PREFIX/bin` is one of `PATH`'s directories | never |
| config | the files read ([`config.rs`](config.md)), `XKS_TEMPLATE` and `XKS_LAYOUT` parsed | a value does not parse |
| llama.cpp | the build directory baked in at build time (`XKS_LLAMA_BUILD_DIR`) | never (the binary would not have started without it) |
| subject | `XKS_SUBJECT` set and a file | unset or missing |
| site | `XKS_SITE` resolved (`auto` consults the cards) | an unknown site |
| AVX-512, payload | the sibling found as [`site.rs`](site.md) finds it, the payload built | the site needs the cards: asked for by name, or `auto` with a card up |
| stack | `phi` on `PATH`; `phictl` and `phitop` linking debug builds is a note | as above |
| cards | `site::card_windows` (a window whose daemon answers), the lock's holder, xks's worker markers | none up and the site asked for them by name |
| avx512 site | its build (`XKS_AVX512_BIN` or `target/avx512/release/xks`) | `XKS_SITE=avx512` without it |
| memory | `/proc/meminfo` against the subject's size | never: a note on the x86 site when the subject is larger than what is available |
| server | one `GET /health` at `XKS_BIND`, 2 s | never |

It starts nothing and changes nothing: no site is prepared, no worker
started, no card reached over ssh (`phi-vpu.sh status` would, slowly), no
server started, and the configuration's own parse errors are findings
rather than the error that stops every other command (it runs before
`run()` parses them).

`--fix` does what is a build or a link: the payload built (`cargo build
--release -p phi-ggml` in the sibling's `host`, the last lines of cargo's
error shown when it fails), and `xks` linked into `PREFIX/bin` (`--prefix`,
default `~/.local`, the same `PREFIX` as `make install`). A link is
created when absent and replaced when it is dangling or links another
build; a file in its place is never replaced (`place_link`, tested with
each case). It never boots a card, downloads a model, runs `sudo` or
touches the stack's daemons; for those it prints the command.

Measured 2026-09-25 on this machine, each state set through the
environment with a scratch prefix: a missing subject, `XKS_SITE=cards`
with no card up (a scratch `XDG_RUNTIME_DIR`), no Intel-Phi-AVX512
checkout with the cards up (`PHI_AVX512_ROOT` empty) and an unknown
template exit 1 with a `MISS` naming the fix; no card up with `auto`
exits 0 with a note (the x86 site); `--fix` linked `xks` and a second run
with that directory on `PATH` reported it found; a file in the link's
place was left as it was. No xks process, worker or listener appeared.
