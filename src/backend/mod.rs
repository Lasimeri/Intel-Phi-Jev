//! Backends. A [`Scorer`] answers one question: given a prompt, what are the
//! log-probabilities of each candidate next token? Everything else
//! (rendering, softmax, calibration, the wire format) is backend-independent.
//!
//! Implemented: [`llamacpp::LlamaServer`] (any GGUF via `llama-server`).
//! Planned: an in-process ds4-rs-metal session (`AttnStepState` fork per
//! question), an in-process llama.cpp context (`llama_kv_self_seq_cp`).

pub mod llamacpp;
pub mod openai;
pub mod typesafe;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend http: {0}")]
    Http(String),
    #[error("backend returned malformed output: {0}")]
    Malformed(String),
    #[error("request rejected: {0}")]
    Rejected(String),
}

/// What one scoring call cost.
#[derive(Debug, Clone, Default)]
pub struct ScoreCost {
    /// Prompt tokens the backend actually evaluated (cache misses).
    pub prompt_evaluated: u64,
    /// Prompt tokens served from the backend's cache.
    pub prompt_cached: u64,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct Scored {
    /// Natural-log probability of each candidate, `-inf` when the backend did
    /// not report the candidate (it fell outside its top-N).
    pub logprobs: Vec<f64>,
    pub cost: ScoreCost,
}

pub trait Scorer: Send + Sync {
    /// Log-probabilities of `candidates` as the *next* token after `prompt`.
    /// Candidate strings are exact token texts (e.g. `" A"`).
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError>;

    /// Every fingerprint of one session: `items` are (suffix, candidates)
    /// pairs whose prompts are `prefix` followed by the suffix. The default
    /// scores them one after another; a backend that can hold the session
    /// once and fork it (ARTICHOKE) overrides this.
    fn score_many(
        &self,
        prefix: &str,
        items: &[(String, Vec<String>)],
    ) -> Result<Vec<Scored>, BackendError> {
        items
            .iter()
            .map(|(suffix, c)| self.score(&format!("{prefix}{suffix}"), c))
            .collect()
    }

    /// Whether candidates may be several tokens long (labels past 26
    /// options). Only a backend that can read a label token by token (a
    /// trie of forks) can; the others read one next token.
    fn multi_token_labels(&self) -> bool {
        false
    }

    /// Human-readable identity for the `model` field.
    fn model_name(&self) -> String;
}

impl Scorer for Box<dyn Scorer> {
    fn multi_token_labels(&self) -> bool {
        (**self).multi_token_labels()
    }
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError> {
        (**self).score(prompt, candidates)
    }
    fn score_many(
        &self,
        prefix: &str,
        items: &[(String, Vec<String>)],
    ) -> Result<Vec<Scored>, BackendError> {
        (**self).score_many(prefix, items)
    }
    fn model_name(&self) -> String {
        (**self).model_name()
    }
}
