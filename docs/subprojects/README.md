# Subprojects

Each experiment is one command, `xks subproject run NN` (all of them:
`make subprojects`), inside a ten-minute budget, and leaves a record in
[`results/`](results/): the git revision, the UTC time, the host, the
configuration and every measured value, or why it did not finish. The pages
here read those records.

| id | page | latest record |
| --- | --- | --- |
| 01 | [BLUEBIRD baseline](01-bluebird-baseline.md) | `6f944a0`, 2026-09-24 |
| 02 | [Polygraph](02-polygraph.md) | `6f944a0`, 2026-09-24 |
| 03 | [x86 against the cards](03-dev-eval-sites.md) | `6f944a0`, 2026-09-24 |
| 04 | [Long sessions, BLUEBIRD against ARTICHOKE](04-long-sessions.md) | `6f944a0`, 2026-09-24 |
| 05 | [The 30-option trie](05-wide-choice-trie.md) | `6f944a0`, 2026-09-24 |
| 06 | [The avx512 site](06-avx512-parity.md) (run by name only) | none: it does not finish inside the budget |
| 07 | [Long sessions on the cards](07-long-sessions-cards.md) | `98980ec`, 2026-09-25 |

## Why every record is kept

MKULTRA, which gives these their numbers, is known in its particulars
because records outlived the order to destroy them: its files were
ordered destroyed in 1973, and about 20,000 documents that had been filed
among financial records were found in 1977 through a Freedom of
Information request, which led to the Senate's hearings that year. So a
record here is written for every run, a failure included, and a newer run
replaces the file in a new commit: every earlier record stays in the
history.

With a nod to the admin and mod team of the Discord server of
[The Eye](https://the-eye.eu/), the community-run open data archive
("We are digital librarians"), who hold records to the same standard. When
a disk failed under the archive in November 2025, the notice on its front
page said what a record keeper should be able to say: "All previously
hosted data is safe. Preserve, Prolong, Persist."
