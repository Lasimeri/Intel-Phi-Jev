//! Build script. With the `artichoke` feature it binds llama.cpp's C API
//! with bindgen, so every struct passed by value (the model and context
//! parameters, the batch) has the layout the headers declare rather than
//! one copied by hand, and links the shared libraries of one llama.cpp
//! build: `LLAMA_BUILD_DIR` (default `~/llama.cpp/build-native/bin`, the
//! x86-64 build; `make build-avx512` points it at `build-avx512/bin`).
//! A build without dynamic backends (`GGML_BACKEND_DL=OFF`, as the AVX-512
//! one is) gets `cfg(xks_static_cpu)`: its CPU backend is linked in and
//! only the payload is loaded at run time. See build.md.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_DIR");
    println!("cargo::rustc-check-cfg=cfg(xks_static_cpu)");
    #[cfg(feature = "artichoke")]
    artichoke();
}

#[cfg(feature = "artichoke")]
fn artichoke() {
    use std::env;
    use std::path::{Path, PathBuf};

    let home = env::var("HOME").unwrap_or_default();
    let src = env::var("LLAMA_CPP_DIR").unwrap_or(format!("{home}/llama.cpp"));
    let lib = env::var("LLAMA_BUILD_DIR").unwrap_or(format!("{src}/build-native/bin"));
    let bindings = bindgen::Builder::default()
        .header_contents(
            "wrapper.h",
            "#include <llama.h>\n#include <ggml-backend.h>\n",
        )
        .clang_arg(format!("-I{src}/include"))
        .clang_arg(format!("-I{src}/ggml/include"))
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_backend_load")
        .allowlist_function("ggml_backend_load_all_from_path")
        .allowlist_function("ggml_backend_dev_(count|get|name|description)")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .derive_default(true)
        .generate()
        .unwrap_or_else(|e| panic!("bindgen over {src}/include/llama.h: {e}"));
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("llama.rs");
    bindings.write_to_file(&out).expect("write llama.rs");
    println!("cargo:rerun-if-changed={src}/include/llama.h");
    println!("cargo:rerun-if-changed={src}/ggml/include/ggml-backend.h");
    println!("cargo:rustc-link-search=native={lib}");
    for l in ["llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    // The directory the CPU variants (libggml-cpu-*.so) are loaded from at
    // run time, unless XKS_BACKEND_DIR names another.
    println!("cargo:rustc-env=XKS_LLAMA_BUILD_DIR={lib}");
    // CMakeCache.txt sits one level above bin/.
    let cache = Path::new(&lib).parent().map(|p| p.join("CMakeCache.txt"));
    println!(
        "cargo:rerun-if-changed={}",
        cache
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    );
    let dynamic = cache
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().any(|l| l.trim() == "GGML_BACKEND_DL:BOOL=ON"))
        .unwrap_or(true);
    if !dynamic {
        println!("cargo:rustc-cfg=xks_static_cpu");
    }
}
