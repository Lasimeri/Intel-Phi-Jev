# Subproject 05: the 30-option trie

`xks subproject run 05`. Record:
[`results/05-wide-choice-trie.json`](results/05-wide-choice-trie.json) (git
`6f944a0`, 2026-09-24T15:53:03Z, clean tree, 19 s).

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
| largest label log-probability difference | 0.023 | |
| argmax | MUL_MAT_ID (gold) | MUL_MAT_ID |
| tokens evaluated | 358 | 10,380 |
| time | 0.55 s | 17.9 s |

Inside the dense subject's floor (0.028): exact to its arithmetic, 29 times
fewer tokens, 33 times less time.
