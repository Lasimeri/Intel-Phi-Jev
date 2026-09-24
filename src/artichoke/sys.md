# artichoke/sys.rs: the generated bindings

`include!`s the Rust bindgen writes at build time from `llama.h` and
`ggml-backend.h` ([`../../build.rs`](../../build.rs)). Nothing here is written
by hand: every struct passed by value (`llama_model_params`,
`llama_context_params`, `llama_batch`) has the layout the headers of the
linked build declare. A hand-copied layout was the alternative; llama.h
gained `n_rs_seq` and a sampler configuration in 2026, and a stale copy
would have been silent memory corruption.

The lints a generated file trips (naming, unused items, transmutes,
function-pointer comparisons) are allowed here and only here.
