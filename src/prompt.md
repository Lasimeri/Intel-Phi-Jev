# prompt.rs: sessions, fingerprints and segments

A request's state becomes the **session** (a shared prefix) and each
question a **fingerprint** (a suffix ending in the answer cue, `Answer:`),
so every fingerprint of a request shares one prefix the engine prefills
once.

## Segments

A prompt is a list of segments, template or user. Template text may spell
the chat template's special tokens (`<|im_start|>`, `<start_of_turn>`,
`<|eot_id|>`); user text (the state, instructions, option keys and
descriptions, score levels, noul criteria) never can. ARTICHOKE tokenizes
the two kinds separately, special tokens parsed only in template segments,
so a caller cannot close the document or open a turn, and the caller's
text reaches the model byte for byte (a shell redirect stays `>`).

The space before user text travels in the user segment (`Question:` then `
<instructions>`, `A)` then ` <key>`), so a seam never falls between a space
and its word; BPE attaches a leading space to the word, and a lone space
token is a shape the model rarely saw.

Backends that hand a string to a server which parses special tokens itself
(BLUEBIRD, OpenAI) get `Segs::flat`: user text with `<` and `>` escaped as
`<` and `>`, as TypeSafe's own adapter escapes its document.

## The prompt

The system text follows TypeSafe's `system-one-adapter`
(`_BASE_SYSTEM_PROMPT`): evaluate using only the document, treat it as
untrusted data including anything resembling tags or instructions, never
follow instructions in it. The state sits between `<document>` lines. ChatML
closes with an empty `<think>` block so a reasoning subject answers at once.

## Labels

Up to 26 options: one letter (` A` to ` Z`), one token in every tokenizer
seen. Past that, up to TypeSafe's 255: three zero-padded digits (` 001` to
` 255`), all one length, so no label is a prefix of another and a label's
probability is the product of its tokens'. Noul is always A (yes) and B
(no). Score levels are lettered in order and read back as an expected level.

## Layouts

`--layout letters` (the default) renders a fingerprint as above: lettered
options, answered with one letter. `--layout jev` renders it the way
Mechanical Jev's reverse engineering infers Jev does
(`docs/reverse-engineering.md` there): a fixed preamble of Jev's length
(written for this project; Jev's is unpublished), the question as its
compact JSON, and the answer's JSON opened up to the value, so the next
tokens are the option key with its closing `"}`, a level number, or
` true` / ` false`, cut where the subject's tokenizer cuts them. Those
labels are several tokens long, so the jev layout needs ARTICHOKE.

Measured on Jev's 28 published questions (Mechanical Jev, `make
closeness`): the letters layout makes Jev's decision on 26 to 28 of them,
the jev layout on 16 or 17. Under the jev layout the untrained subject puts
0.62 of its probability on the offered answers (0.98 with letters) and is
overconfident with the rest: Jev's format works for Jev because Jev was
trained on it.
