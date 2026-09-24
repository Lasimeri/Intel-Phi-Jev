# Subproject 05: the 30-option trie

`xks subproject run 05`. Record:
[`results/05-wide-choice-trie.json`](results/05-wide-choice-trie.json) (git
`6f7b372`, clean tree).

**Question:** TypeSafe allows 255 options per Choice; one-letter labels
stop at 26. Past that, labels are three digits (` 001` to ` 255`), several
tokens each, and ARTICHOKE reads them as a trie of forks: the suffix read
once, then one fork per distinct label prefix, one token further each
([`artichoke/mod.rs`](../../src/artichoke/mod.md), `read_trie`). Is that the
same as decoding every label from scratch?

**Method:** [`examples/wide_choice.jsonl`](../../examples/wide_choice.jsonl):
one Choice of 30 real ggml operation names against a sentence from the
sibling's MoE result document (gold `MUL_MAT_ID`). The polygraph's control
reading decodes the prompt and then each of the 30 labels token by token
from an empty cache, no fork anywhere, and sums the log-probabilities.
Dense 0.5B, site `x86`.

**Result:**

| | trie of forks | brute force |
| --- | --- | --- |
| largest label log-probability difference | 0.023 | |
| argmax | MUL_MAT_ID (gold) | MUL_MAT_ID |
| tokens evaluated | 358 | 10,380 |
| time | 3.5 s | 120.3 s |

The difference is inside the dense subject's own floor (0.028,
[subproject 02](02-polygraph.md)); the trie is exact to the subject's
arithmetic, 29 times fewer tokens, 34 times less time. The labels carried
0.75 of the probability: lower than with letters (0.96), since three-digit
labels are a less familiar answer shape for this subject.
