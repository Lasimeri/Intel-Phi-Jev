# Top-level entry points. Run `make help` for the list.

.DEFAULT_GOAL := help

.PHONY: help build test fmt clippy docs-check check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

build: ## Build xks (release) with the ARTICHOKE engine
	cargo build --release

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
