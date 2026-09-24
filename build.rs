//! Build script. With the `artichoke` feature it binds llama.cpp's C API
//! with bindgen, so every struct passed by value (the model and context
//! parameters, the batch) has the layout the headers declare rather than
//! one copied by hand, and links the shared libraries of one llama.cpp
//! build. See build.md.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_DIR");
    #[cfg(feature = "artichoke")]
    artichoke();
}

#[cfg(feature = "artichoke")]
fn artichoke() {
    use std::env;
    use std::path::PathBuf;

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
        .allowlist_function("ggml_backend_load_all_from_path")
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
}
