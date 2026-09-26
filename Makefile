# Top-level entry points. Run `make help` for the list. Every target is a
# thin call into the `xks` binary; the logic lives in Rust.

.DEFAULT_GOAL := help
XKS := target/release/xks
PREFIX ?= $(HOME)/.local
LLAMA_CPP_DIR ?= $(HOME)/llama.cpp
export LLAMA_CPP_DIR
AVX512_LLAMA := $(LLAMA_CPP_DIR)/build-avx512/bin
AVX512_ENV := LLAMA_BUILD_DIR=$(AVX512_LLAMA) CARGO_TARGET_DIR=target/avx512

.PHONY: help build build-x86 build-avx512 install uninstall doctor serve stop query jev jev-eval subprojects subproject test fmt clippy docs-check check clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

build: build-x86 build-avx512 ## Build both xks binaries (x86-64 and AVX-512)

build-x86: ## xks against llama.cpp's x86-64 build (build-native)
	cargo build --release

build-avx512: ## xks against llama.cpp's AVX-512 build, for the avx512 site (skipped when that build is absent)
	@if [ -d "$(AVX512_LLAMA)" ]; then $(AVX512_ENV) cargo build --release; \
	else echo "build-avx512: skipped, no $(AVX512_LLAMA) (llama.cpp built with GGML_AVX512=ON; only the avx512 site needs it)"; fi

install: build-x86 ## Link xks into $(PREFIX)/bin (a link, so every rebuild is what runs; never over a file)
	@mkdir -p "$(PREFIX)/bin"
	@if [ -e "$(PREFIX)/bin/xks" ] && [ ! -L "$(PREFIX)/bin/xks" ]; then echo "$(PREFIX)/bin/xks is a file, not a link: left alone"; exit 1; fi
	@ln -sfn "$(CURDIR)/$(XKS)" "$(PREFIX)/bin/xks" && echo "$(PREFIX)/bin/xks -> $(CURDIR)/$(XKS)"

uninstall: ## Remove that link (only a link, never a file)
	@if [ -L "$(PREFIX)/bin/xks" ]; then rm "$(PREFIX)/bin/xks" && echo "removed $(PREFIX)/bin/xks"; else echo "no link at $(PREFIX)/bin/xks"; fi

doctor: build-x86 ## What is missing for xks to answer, and the fix for each (FIX=1 builds and links what it can)
	$(XKS) doctor --prefix "$(PREFIX)" $(if $(FIX),--fix)

serve: build-x86 ## Start the server in the background (site and subject from xks.conf)
	$(XKS) serve --detach

stop: ## Stop the server and release the cards' huge pages
	$(XKS) stop

query: build-x86 ## One example query (examples/query.json; a running server answers it)
	$(XKS) query --file examples/query.json

jev: build-x86 ## The same query to the real Jev (TypeSafe hosted; TYPESAFE_API_KEY in xks.local.conf)
	$(XKS) --backend-kind jev query --file examples/query.json

jev-eval: build-x86 ## Score the long sessions with the real Jev; rows in target/jev-long-rows.jsonl
	$(XKS) --backend-kind jev eval examples/long_sessions.jsonl --rows target/jev-long-rows.jsonl

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

check: docs-check fmt clippy build test ## Everything before a commit (the foundations of the wall, Revelation 21:14)

clean: ## Remove build outputs
	cargo clean
