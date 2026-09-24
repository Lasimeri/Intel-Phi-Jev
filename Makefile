# Top-level entry points. Run `make help` for the list. Every target is a
# thin call into the `xks` binary; the logic lives in Rust.

.DEFAULT_GOAL := help
XKS := target/release/xks
AVX512_ENV := LLAMA_BUILD_DIR=$(HOME)/llama.cpp/build-avx512/bin CARGO_TARGET_DIR=target/avx512

.PHONY: help build build-x86 build-avx512 serve stop query subprojects subproject test fmt clippy docs-check check clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

build: build-x86 build-avx512 ## Build both xks binaries (x86-64 and AVX-512)

build-x86: ## xks against llama.cpp's x86-64 build (build-native)
	cargo build --release

build-avx512: ## xks against llama.cpp's AVX-512 build, for the avx512 site
	$(AVX512_ENV) cargo build --release

serve: build-x86 ## Start the server in the background (site and subject from xks.conf)
	$(XKS) serve --detach

stop: ## Stop the server and release the cards' huge pages
	$(XKS) stop

query: build-x86 ## One example query (examples/query.json)
	$(XKS) query --file examples/query.json

subprojects: build ## Run every subproject; records in docs/subprojects/results/
	$(XKS) subproject run all

subproject: build ## One subproject: make subproject N=04
	$(XKS) subproject run $(N)

test: ## Unit tests (no subject, no cards)
	cargo test --release

fmt: ## Check formatting
	cargo fmt --all -- --check

clippy: ## Lint
	cargo clippy --release --all-targets -- -D warnings

docs-check: ## Sibling .md files, the no-dash rule, relative links
	scripts/check-docs.sh

check: docs-check fmt clippy build test ## Everything before a commit

clean: ## Remove build outputs
	cargo clean
