# Subproject 05: the 30-option trie

`xks subproject run 05`. Record:
[`results/05-wide-choice-trie.json`](results/05-wide-choice-trie.json) (git
`9ce2915`, 2026-09-26T15:37:53Z, clean tree, 67 s).

**Question:** TypeSafe allows 255 options per Choice; one-letter labels stop
at 26. Past that, labels are three digits read as a trie of forks
([`artichoke/mod.rs`](../../src/artichoke/mod.md), `read_trie`). Is that
the same as decoding every label from scratch?

**Method:** one Choice of 30 real ggml operation names
([`examples/wide_choice.jsonl`](../../examples/wide_choice.jsonl)), the
small subject (`XKS_SUBJECT_SMALL`: Qwen3.8 2B Q8_0 since 2026-09-26, a
hybrid), the polygraph's control decoding each label token by token from
an empty cache.

| | trie of forks | brute force |
| --- | --- | --- |
| largest label log-probability difference | 0.077 | |
| argmax | MUL | MUL |
| tokens evaluated | 356 | 10,320 |
| time | 2.2 s | 64.2 s |

Inside the small subject's floor (0.21, [subproject 02](02-polygraph.md)):
the trie reads what brute force reads, with 29 times fewer tokens in 29
times less time. The answer itself is wrong: the gold is MUL_MAT_ID, and
the 2B puts its most on MUL both ways (`xks --subject ... eval
examples/wide_choice.jsonl`: accuracy 0 of 1; 41 percent of its
probability on the 30 labels at all). That is the subject's reading, not
the trie's: the trie exists to read the same thing cheaper, and it does.

Before the small subject changed, with the dense Qwen2.5 0.5B f16: on
4bfa67b (2026-09-26) the trie differed by 0.020 in 0.80 s against 17.9 s,
358 against 10,380 tokens, and chose the gold MUL_MAT_ID; on 6f944a0
(2026-09-24) 0.023 in 0.55 s.
