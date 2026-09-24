# scripts/check-docs.sh

`make docs-check`. Three rules, the same as the sibling repositories':

1. Every code file under `src/`, `scripts/` and `build.rs` has a sibling
   Markdown file with the same stem.
2. No em dash (U+2014) or en dash (U+2013) anywhere tracked.
3. Every relative Markdown link points at a file that exists.

It stays a shell script on purpose: it is repository hygiene run by `make`,
not part of `xks`, and it matches the siblings' copy line for line apart
from the directories it scans.
