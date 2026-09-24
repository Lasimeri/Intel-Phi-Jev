# build.rs

With the `artichoke` feature (default), binds llama.cpp and links one of its
builds.

| variable | default | meaning |
| --- | --- | --- |
| `LLAMA_CPP_DIR` | `~/llama.cpp` | the source tree: headers for bindgen |
| `LLAMA_BUILD_DIR` | `$LLAMA_CPP_DIR/build-native/bin` | the libraries linked, and the rpath |

`make build-x86` uses the default (the x86-64 build, dynamic backends, all
CPU variants). `make build-avx512` points `LLAMA_BUILD_DIR` at
`build-avx512/bin` and `CARGO_TARGET_DIR` at `target/avx512`, giving the
second binary the `avx512` site runs under phi512.

The script reads `CMakeCache.txt` next to the build's `bin/`: without
`GGML_BACKEND_DL:BOOL=ON` it sets `cfg(xks_static_cpu)`, and the engine then
loads only the payload instead of searching for CPU variants that were never
built (the AVX-512 build links its CPU backend in).

Both builds come from one source tree, so one set of headers fits both;
`llama.h` has not changed since both were built. If a build is ever older
than the headers, rebuild it before linking against it.

`XKS_LLAMA_BUILD_DIR` is baked into the binary as the default directory for
CPU variants; `XKS_BACKEND_DIR` overrides it at run time.
