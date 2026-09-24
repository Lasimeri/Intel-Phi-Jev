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
