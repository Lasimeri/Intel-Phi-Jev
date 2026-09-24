//! Backends. A [`Scorer`] answers one question: given a prompt, what are the
//! log-probabilities of each candidate next token? Everything else
//! (rendering, softmax, calibration, the wire format) is backend-independent.
//!
//! Implemented: [`bluebird::Bluebird`] (a `llama-server` over HTTP, the
//! baseline), [`openai::OpenAiChat`] (chat/completions with logprobs),
//! [`typesafe::TypeSafe`] (the hosted Jev, for `query --compare`), and the
//! in-process ARTICHOKE engine in `crate::artichoke`, which forks the session
//! with `llama_memory_seq_cp` (what was planned upstream).

pub mod bluebird;
pub mod openai;
pub mod typesafe;

use thiserror::Error;

use crate::prompt::Segs;

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
    /// Log-probabilities of `candidates` as the continuation of `prompt`, a
    /// flattened string (user text escaped, `Segs::flat`). Candidate strings
    /// are exact label texts (e.g. `" A"`).
    fn score(&self, prompt: &str, candidates: &[String]) -> Result<Scored, BackendError>;

    /// Every fingerprint of one session: `items` are (suffix, candidates)
    /// pairs whose prompts are `prefix` followed by the suffix. The default
    /// flattens each prompt and scores them one after another; a backend
    /// that tokenizes segments itself and can hold the session once and
    /// fork it (ARTICHOKE) overrides this.
    fn score_many(
        &self,
        prefix: &Segs,
        items: &[(Segs, Vec<String>)],
    ) -> Result<Vec<Scored>, BackendError> {
        items
            .iter()
            .map(|(suffix, c)| self.score(&prefix.concat(suffix).flat(), c))
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
        prefix: &Segs,
        items: &[(Segs, Vec<String>)],
    ) -> Result<Vec<Scored>, BackendError> {
        (**self).score_many(prefix, items)
    }
    fn model_name(&self) -> String {
        (**self).model_name()
    }
}
