//! The polygraph: every fingerprint of every case read three ways by
//! ARTICHOKE and the readings compared, the way a polygraph examiner reads
//! the relevant questions against control questions.
//!
//! - forked: what `serve` and `eval` do (the session on sequence 0, copied
//!   to one sequence per fingerprint, the suffixes decoded in one batch);
//! - split: the same two decodes on sequence 0, with no copy;
//! - control: the whole prompt in one decode from an empty cache.
//!
//! forked against split isolates the copy; split against control isolates
//! cutting the prompt into two decodes. A fork that copied the wrong state
//! reads plausibly and wrongly, which is what this is here to catch. See
//! polygraph.md.

use std::time::Instant;

use serde::Serialize;

use crate::artichoke::Artichoke;
use crate::backend::{BackendError, Scorer};
use crate::eval::Case;
use crate::prompt::{self, Template};
use crate::protocol::{parse_questions, render_state};
use crate::score::{argmax, softmax};

#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub case_index: usize,
    pub id: String,
    /// Label log-probabilities over the whole vocabulary, per reading.
    pub forked: Vec<f64>,
    pub split: Vec<f64>,
    pub control: Vec<f64>,
    /// Largest absolute label log-probability difference, per pair.
    pub forked_vs_split: f64,
    pub split_vs_control: f64,
    pub forked_vs_control: f64,
    /// The forked and control readings pick the same label.
    pub same_argmax: bool,
    /// Probability the control reading puts on the labels at all; well
    /// under one means the prompt is off the subject's distribution and
    /// the restricted softmax hides it.
    pub label_mass: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Pair {
    pub max: f64,
    pub mean: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub subject: String,
    pub recurrent: bool,
    pub fingerprints: usize,
    pub forked_vs_split: Pair,
    pub split_vs_control: Pair,
    pub forked_vs_control: Pair,
    pub same_argmax: usize,
    pub min_label_mass: f64,
    pub mean_label_mass: f64,
    pub forked_ms: f64,
    pub control_ms: f64,
    pub forked_tokens: u64,
    pub control_tokens: u64,
}

pub struct Report {
    pub rows: Vec<Row>,
    pub summary: Summary,
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn pair(rows: &[Row], f: impl Fn(&Row) -> f64) -> Pair {
    let n = rows.len().max(1) as f64;
    Pair {
        max: rows.iter().map(&f).fold(0.0, f64::max),
        mean: rows.iter().map(&f).sum::<f64>() / n,
    }
}

pub fn run(
    engine: &Artichoke,
    template: Template,
    cases: &[Case],
    limit: Option<usize>,
) -> Result<Report, BackendError> {
    let mut rows = Vec::new();
    let (mut forked_ms, mut control_ms) = (0.0, 0.0);
    let (mut forked_tokens, mut control_tokens) = (0u64, 0u64);
    for (ci, c) in cases.iter().enumerate().take(limit.unwrap_or(usize::MAX)) {
        let questions = parse_questions(&c.questions).map_err(BackendError::Rejected)?;
        let prefix = prompt::prefix(template, &render_state(&c.state));
        let rendered: Vec<_> = questions
            .iter()
            .map(|(id, q)| (id.clone(), prompt::render(template, &prefix, q, None)))
            .collect();
        let items: Vec<(prompt::Segs, Vec<String>)> = rendered
            .iter()
            .map(|(_, r)| (r.suffix.clone(), r.labels.clone()))
            .collect();
        let t0 = Instant::now();
        let forked = engine.score_many(&prefix, &items)?;
        forked_ms += t0.elapsed().as_secs_f64() * 1e3;
        forked_tokens += forked.iter().map(|s| s.cost.prompt_evaluated).sum::<u64>();
        let at = engine.session_tokens(&prefix, &items)?;
        for ((id, r), f) in rendered.iter().zip(&forked) {
            let t1 = Instant::now();
            let p = engine.read_control(&r.prompt(), &r.labels)?;
            control_ms += t1.elapsed().as_secs_f64() * 1e3;
            control_tokens += p.cost.prompt_evaluated;
            // The split reading exists for one-token labels; a trie question
            // is checked forked against control only.
            let s = match engine.read_split(&r.prompt(), &r.labels, at) {
                Err(BackendError::Rejected(_)) => p.clone(),
                other => other?,
            };
            rows.push(Row {
                case_index: ci,
                id: id.clone(),
                forked_vs_split: max_abs(&f.logprobs, &s.logprobs),
                split_vs_control: max_abs(&s.logprobs, &p.logprobs),
                forked_vs_control: max_abs(&f.logprobs, &p.logprobs),
                same_argmax: argmax(&softmax(&f.logprobs, 1.0))
                    == argmax(&softmax(&p.logprobs, 1.0)),
                label_mass: p.logprobs.iter().map(|l| l.exp()).sum(),
                forked: f.logprobs.clone(),
                split: s.logprobs,
                control: p.logprobs,
            });
        }
    }
    let n = rows.len().max(1) as f64;
    let summary = Summary {
        subject: engine.model_name(),
        recurrent: engine.recurrent,
        fingerprints: rows.len(),
        forked_vs_split: pair(&rows, |r| r.forked_vs_split),
        split_vs_control: pair(&rows, |r| r.split_vs_control),
        forked_vs_control: pair(&rows, |r| r.forked_vs_control),
        same_argmax: rows.iter().filter(|r| r.same_argmax).count(),
        min_label_mass: rows
            .iter()
            .map(|r| r.label_mass)
            .fold(f64::INFINITY, f64::min),
        mean_label_mass: rows.iter().map(|r| r.label_mass).sum::<f64>() / n,
        forked_ms,
        control_ms,
        forked_tokens,
        control_tokens,
    };
    Ok(Report { rows, summary })
}
