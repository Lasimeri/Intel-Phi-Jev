//! The judge: render each question against the shared state prefix, score
//! the option labels with a [`Scorer`], calibrate, and build typed answers.

use serde_json::{json, Map, Value};

use crate::backend::{BackendError, Scorer};
use crate::prompt::{self, Template, LETTERS, MAX_OPTIONS};
use crate::protocol::{
    parse_questions, render_state, Answer, Evaluation, Question, Request, Usage,
};
use crate::score::{argmax, confidence, expected_level, softmax, Calibration};

#[derive(Debug, Clone)]
pub struct JudgeConfig {
    pub template: Template,
    /// How fingerprints are laid out: lettered options, or the layout
    /// reconstructed from Jev's documentation.
    pub layout: prompt::Layout,
    pub calibration: Calibration,
    /// For `choice`: score this many cyclic rotations of the option order and
    /// average the per-key probabilities. 1 = no position-bias averaging.
    /// Hemi-Sync: the pull toward one side of the list is balanced out.
    pub permutations: usize,
    pub debug: bool,
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            template: Template::ChatMl,
            layout: prompt::Layout::Letters,
            calibration: Calibration::default(),
            permutations: 1,
            debug: false,
        }
    }
}

pub struct Judge<S: Scorer> {
    pub scorer: S,
    pub cfg: JudgeConfig,
}

/// Raw, uncalibrated result for one question (used by `eval`/`calibrate`).
#[derive(Debug, Clone)]
pub struct RawQuestion {
    pub id: String,
    pub kind: &'static str,
    pub keys: Vec<String>,
    /// Averaged over permutations, still uncalibrated (log of mean prob).
    pub logprobs: Vec<f64>,
    pub prompt_evaluated: u64,
    pub prompt_cached: u64,
    /// Probability the subject put on the labels at all (mean over
    /// rotations): well under one means the layout is off its distribution.
    pub label_mass: f64,
    pub latency_ms: f64,
}

impl<S: Scorer> Judge<S> {
    pub fn new(scorer: S, cfg: JudgeConfig) -> Self {
        Self { scorer, cfg }
    }

    /// Score every question and return uncalibrated per-option logprobs.
    pub fn raw(&self, req: &Request) -> Result<Vec<RawQuestion>, BackendError> {
        let questions = parse_questions(&req.questions).map_err(BackendError::Rejected)?;
        let prefix = prompt::prefix_for(
            self.cfg.layout,
            self.cfg.template,
            &render_state(&req.state),
        );
        // Render every fingerprint (each question, each rotation) first, so
        // the backend sees the whole session at once and can fork it.
        struct Plan {
            id: String,
            kind: &'static str,
            keys: Vec<String>,
            orders: Vec<Vec<usize>>,
        }
        let mut plans = Vec::with_capacity(questions.len());
        let mut items: Vec<(prompt::Segs, Vec<String>)> = Vec::new();
        for (id, q) in &questions {
            let (kind, n) = match q {
                Question::Noul { .. } => ("noul", 2),
                Question::Choice { criteria, .. } => ("choice", criteria.len()),
                Question::Score { criteria, .. } => ("score", criteria.len()),
            };
            if n > MAX_OPTIONS {
                return Err(BackendError::Rejected(format!(
                    "question `{id}`: {n} options; a Choice takes at most {MAX_OPTIONS}"
                )));
            }
            if (n > LETTERS || self.cfg.layout == prompt::Layout::Jev)
                && !self.scorer.multi_token_labels()
            {
                return Err(BackendError::Rejected(format!(
                    "question `{id}`: this needs --backend-kind artichoke (the jev layout, or more than {LETTERS} options, answers with labels of several tokens)"
                )));
            }
            let perms = if kind == "choice" {
                self.cfg.permutations.clamp(1, n)
            } else {
                1
            };
            let mut plan = Plan {
                id: id.clone(),
                kind,
                keys: Vec::new(),
                orders: Vec::with_capacity(perms),
            };
            for r in 0..perms {
                let order: Vec<usize> = (0..n).map(|i| (i + r) % n).collect();
                let rendered = match self.cfg.layout {
                    prompt::Layout::Letters => {
                        prompt::render(self.cfg.template, &prefix, q, Some(&order))
                    }
                    prompt::Layout::Jev => {
                        prompt::render_jev(self.cfg.template, &prefix, q, Some(&order))
                    }
                };
                if r == 0 {
                    // rotation 0 is the identity: keys are in request order.
                    plan.keys = rendered.keys.clone();
                }
                items.push((rendered.suffix, rendered.labels));
                plan.orders.push(order);
            }
            plans.push(plan);
        }
        let scored = self.scorer.score_many(&prefix, &items)?;
        let mut scored = scored.into_iter();
        let mut out = Vec::with_capacity(plans.len());
        for plan in plans {
            let n = plan.keys.len();
            let perms = plan.orders.len();
            let mut mean = vec![0.0f64; n];
            let mut evaluated = 0;
            let mut cached = 0;
            let mut latency_ms = 0.0;
            let mut mass = 0.0;
            for order in &plan.orders {
                let s = scored.next().ok_or_else(|| {
                    BackendError::Malformed("backend returned too few readings".into())
                })?;
                mass += s.logprobs.iter().map(|l| l.exp()).sum::<f64>() / perms as f64;
                let p = softmax(&s.logprobs, 1.0);
                // rendered position `pos` holds original option `order[pos]`.
                for (pos, &orig) in order.iter().enumerate() {
                    mean[orig] += p[pos] / perms as f64;
                }
                evaluated += s.cost.prompt_evaluated;
                cached += s.cost.prompt_cached;
                latency_ms += s.cost.latency_ms;
            }
            out.push(RawQuestion {
                id: plan.id,
                kind: plan.kind,
                keys: plan.keys,
                logprobs: mean.iter().map(|p| p.max(1e-300).ln()).collect(),
                prompt_evaluated: evaluated,
                prompt_cached: cached,
                label_mass: mass,
                latency_ms,
            });
        }
        Ok(out)
    }

    /// Every answer with its whole distribution: the sea of glass
    /// (Revelation 4:6), clear to the bottom, never a verdict alone.
    pub fn evaluate(&self, req: &Request) -> Result<Evaluation, BackendError> {
        let raws = self.raw(req)?;
        let mut answers = Map::new();
        let mut usage = Usage::default();
        let mut debug = Vec::new();
        for r in &raws {
            let t = self.cfg.calibration.temperature(r.kind, r.keys.len());
            let probs = softmax(&r.logprobs, t);
            let answer = match r.kind {
                "noul" => Answer::Noul {
                    noul: round(probs[0]),
                },
                "choice" => {
                    let best = argmax(&probs);
                    Answer::Choice {
                        choice: r.keys[best].clone(),
                        probabilities: prob_map(&r.keys, &probs),
                        confidence: round(confidence("choice", &probs)),
                    }
                }
                _ => {
                    let legend: Map<String, Value> = match req.questions.get(&r.id) {
                        Some(Value::Object(o)) => o
                            .get("criteria")
                            .and_then(Value::as_array)
                            .map(|a| {
                                a.iter()
                                    .enumerate()
                                    .map(|(i, v)| (i.to_string(), v.clone()))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        _ => Map::new(),
                    };
                    Answer::Score {
                        score: round(expected_level(&probs)),
                        legend,
                        probabilities: prob_map(&r.keys, &probs),
                        confidence: round(confidence("score", &probs)),
                    }
                }
            };
            answers.insert(r.id.clone(), serde_json::to_value(answer).unwrap());
            // Only tokens the backend actually evaluated; cached prefix
            // tokens cost nothing and are reported under `debug`.
            usage.input_tokens += r.prompt_evaluated;
            if self.cfg.debug {
                debug.push(json!({
                    "id": r.id, "kind": r.kind, "temperature": t,
                    "raw_logprobs": r.logprobs,
                    "prompt_evaluated": r.prompt_evaluated,
                    "prompt_cached": r.prompt_cached,
                    "label_mass": round(r.label_mass),
                    "latency_ms": round(r.latency_ms),
                }));
            }
        }
        Ok(Evaluation {
            model: self.scorer.model_name(),
            answers,
            usage,
            debug: if self.cfg.debug {
                Some(Value::Array(debug))
            } else {
                None
            },
        })
    }
}

fn prob_map(keys: &[String], probs: &[f64]) -> Map<String, Value> {
    keys.iter()
        .zip(probs)
        .map(|(k, p)| (k.clone(), json!(round(*p))))
        .collect()
}

fn round(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
