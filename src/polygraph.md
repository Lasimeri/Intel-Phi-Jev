# polygraph.rs: forked, split and control readings

A fork that copied the wrong state would read plausibly and wrongly, so the
engine's fast path is checked against two slower readings of the same
fingerprint:

| reading | decodes |
| --- | --- |
| forked | the session on sequence 0, copied per fingerprint, suffixes in one batch (what `serve` does) |
| split | the session on sequence 0, then the suffix on sequence 0: the same cut, no copy |
| control | the whole prompt in one decode from an empty cache |

forked against split isolates the copy; split against control isolates the
cut. With `--forks 1` forked and split make identical decodes, and on the
hybrid 35B-A3B they agree exactly (0.0): the copy, recurrent state included,
is bit-exact. Split against control is the subject's own sensitivity to how
a prompt is cut into decodes: up to 0.99 in label log-probability on the
35B, 0.05 on a dense 0.5B (subproject 02 on 87a89d8; 0.8 and 0.045 before
the session ended at the state, 82439bb).

Several-token labels (trie questions) have no split reading; they are
checked forked against control, where control decodes every label token by
token from an empty cache.

`label_mass` is the probability the control reading puts on the labels at
all: well under one means the prompt is off the subject's distribution and
the restricted softmax is hiding it.
