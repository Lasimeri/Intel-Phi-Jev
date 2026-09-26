# Subproject 05: the 30-option trie

`xks subproject run 05`. Record:
[`results/05-wide-choice-trie.json`](results/05-wide-choice-trie.json) (git
`4bfa67b`, 2026-09-26T03:46:59Z, clean tree, 19 s).

**Question:** TypeSafe allows 255 options per Choice; one-letter labels stop
at 26. Past that, labels are three digits read as a trie of forks
([`artichoke/mod.rs`](../../src/artichoke/mod.md), `read_trie`). Is that
the same as decoding every label from scratch?

**Method:** one Choice of 30 real ggml operation names
([`examples/wide_choice.jsonl`](../../examples/wide_choice.jsonl)), the
dense 0.5B, the polygraph's control decoding each label token by token from
an empty cache.

| | trie of forks | brute force |
| --- | --- | --- |
| largest label log-probability difference | 0.020 | |
| argmax | MUL_MAT_ID (gold) | MUL_MAT_ID |
| tokens evaluated | 358 | 10,380 |
| time | 0.80 s | 17.9 s |

Inside the dense subject's floor (0.05, [subproject 02](02-polygraph.md)):
exact to its arithmetic, 29 times fewer tokens, 22 times less time.
Re-run on 4bfa67b (2026-09-26 UTC); on 6f944a0 (2026-09-24) the trie took
0.55 s and differed by 0.023, the same tokens.
