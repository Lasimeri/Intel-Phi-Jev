# Subproject 06: the avx512 site

`xks subproject run 06`, run only by name (`all` skips it): one
fingerprint takes longer than the ten-minute budget on this site today, so
there is no finished record yet. What was found running it by hand, with
the dense 0.5B:

- **q4_0 weights:** llama.cpp's AVX-512 q4_0 path uses
  `vpaddb ymm, ymm, ymm31` (AVX-512BW/VL byte lanes); the card's vector unit
  has 32-bit lanes only and the translator has no byte forms yet. phi512
  refuses cleanly (the process stops, the host is untouched).
- **flash attention, fp16:** after five minutes of regions that ran, a phase
  in `ggml_compute_forward_flash_attn_ext_tiled` touched memory the process
  never mapped (the planner's reach for a loop advancing 256 bytes an
  iteration overshoots). The site now runs with flash attention off.
- **one card for two jobs:** card 0 serving phi512's regions and the
  payload's multiplies deadlocked after 32 minutes (the one host thread
  waited on a multiply while its own region queued behind it). The site now
  splits the cards by role: card 0 regions, the others the payload.
- **speed:** one 0.5B fingerprint took over 30 minutes and 2.4 million
  card requests: each AVX-512 region is a round trip to the card.

All four are the sibling co-processor's (Intel-Phi-AVX512) to move; the
site in [`site.rs`](../../src/site.md) is ready when they do.

## Since then, on the sibling's side (2026-09-25)

Intel-Phi-AVX512's review of the transparent path
([its record](https://github.com/Lasimeri/Intel-Phi-AVX512/blob/main/docs/results/2026-09-25-review-transparent-path.md))
found a candidate cause for the flash attention fault: the thunk area was
unreserved address space inside a 2 MiB chunk the card maps without
fetching. It is now a whole reserved chunk. Whether that was the fault is
unproven: the test that failed takes longer than the ten-minute budget and
has not been re-run. The same day the path's fetch and write-back were
staged through a huge page (about 1.7 to 2.4 times faster at a million
elements), which shortens each region but not the round trip per region
that the speed item above is about. `vpaddb` and the other byte-lane forms
are still not in the translator.
