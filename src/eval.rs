//! Labelled evaluation: accuracy, Brier, ECE, latency, and temperature fitting.
//!
//! Case file: JSON Lines, one object per line:
//! `{"state": ..., "questions": {...}, "gold": {"<id>": "<key or level index>"}}`

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::backend::{BackendError, Scorer};
use crate::judge::{Judge, RawQuestion};
use crate::protocol::Request;
use crate::score::{argmax, confidence, softmax, Calibration};

#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    #[serde(default)]
    pub model: Option<String>,
    pub state: Value,
    pub questions: Map<String, Value>,
    pub gold: Map<String, Value>,
}

pub fn load_cases(path: &Path) -> Result<Vec<Case>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .enumerate()
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("line {}: {e}", i + 1)))
        .collect()
}

/// One scored question with its gold index, kept for fitting/reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub case_index: usize,
    pub id: String,
    pub kind: String,
    pub n: usize,
    pub raw_logprobs: Vec<f64>,
    pub gold: usize,
    pub latency_ms: f64,
    pub prompt_evaluated: u64,
    pub prompt_cached: u64,
}

/// Scored rows plus the number of cases whose backend call failed. A failed
/// case contributes no rows; it is reported next to accuracy, as the public
/// Jev benchmarks do. A malformed case (its questions do not parse) aborts;
/// one the engine refuses for its size (past the context or TypeSafe's
/// limits) is a failed case like any other, so one long case does not
/// throw away the rest of a run.
pub fn run<S: Scorer>(judge: &Judge<S>, cases: &[Case]) -> Result<(Vec<Row>, usize), BackendError> {
    let mut rows = Vec::new();
    let mut failed = 0usize;
    for (ci, c) in cases.iter().enumerate() {
        crate::protocol::parse_questions(&c.questions)
            .map_err(|m| BackendError::Rejected(format!("case {ci}: {m}")))?;
        let req = Request {
            model: c.model.clone(),
            state: c.state.clone(),
            questions: c.questions.clone(),
        };
        let raws: Vec<RawQuestion> = match judge.raw(&req) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("case {ci}: {e} (counted as failed)");
                failed += 1;
                continue;
            }
        };
        for r in raws {
            let Some(g) = c.gold.get(&r.id) else { continue };
            let Some(gold) = gold_index(g, &r.keys) else {
                return Err(BackendError::Rejected(format!(
                    "case {ci} question `{}`: gold {g} is not an option",
                    r.id
                )));
            };
            rows.push(Row {
                case_index: ci,
                id: r.id.clone(),
                kind: r.kind.to_string(),
                n: r.keys.len(),
                raw_logprobs: r.logprobs.clone(),
                gold,
                latency_ms: r.latency_ms,
                prompt_evaluated: r.prompt_evaluated,
                prompt_cached: r.prompt_cached,
            });
        }
    }
    Ok((rows, failed))
}

/// The option a gold label names: a key (a Score's level number is its
/// key), `true`/`false` or `"true"`/`"false"` for a Noul, else a number
/// read as an option index. `None` when it names no option, so a 1-based
/// level or a stray key is an error, never an index past the end.
fn gold_index(g: &Value, keys: &[String]) -> Option<usize> {
    let noul = keys.len() == 2 && keys[0] == "yes" && keys[1] == "no";
    let key = |s: &str| keys.iter().position(|k| k == s);
    let idx = match g {
        Value::Number(n) => key(&n.to_string()).or_else(|| n.as_u64().map(|x| x as usize)),
        Value::String(s) => key(s)
            .or(match s.as_str() {
                "true" if noul => Some(0),
                "false" if noul => Some(1),
                _ => None,
            })
            .or_else(|| s.parse().ok()),
        Value::Bool(b) if noul => Some(if *b { 0 } else { 1 }),
        _ => None,
    }?;
    (idx < keys.len()).then_some(idx)
}

/// The same cases asked of the real Jev (TypeSafe's hosted API). Its
/// answers carry probabilities, not log-probabilities: each becomes a row
/// with `ln p` per option in the question's own order (Noul as yes, no),
/// so `metrics`, `replay` and `corroborate` read Jev's run exactly as they
/// read a local one.
pub fn run_jev(
    ts: &crate::backend::typesafe::TypeSafe,
    cases: &[Case],
) -> Result<(Vec<Row>, usize), BackendError> {
    use crate::protocol::{parse_questions, Question};
    let lnp = |p: f64| p.max(1e-12).ln();
    let prob = |m: Option<&Value>, k: &str| -> f64 {
        m.and_then(|m| m.get(k))
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    };
    let mut rows = Vec::new();
    let mut failed = 0usize;
    for (ci, c) in cases.iter().enumerate() {
        let req = Request {
            model: c.model.clone(),
            state: c.state.clone(),
            questions: c.questions.clone(),
        };
        let (ev, ms) = match ts.evaluate(&req) {
            Ok(r) => r,
            Err(BackendError::Rejected(m)) => return Err(BackendError::Rejected(m)),
            Err(e) => {
                eprintln!("case {ci}: {e} (counted as failed)");
                failed += 1;
                continue;
            }
        };
        let questions = parse_questions(&c.questions).map_err(BackendError::Rejected)?;
        let each = ms / questions.len().max(1) as f64;
        for (qi, (id, q)) in questions.iter().enumerate() {
            let Some(a) = ev.answers.get(id) else {
                return Err(BackendError::Malformed(format!(
                    "Jev did not answer `{id}`"
                )));
            };
            let probs = a.get("probabilities");
            let (kind, keys, ps): (&str, Vec<String>, Vec<f64>) = match q {
                Question::Noul { .. } => {
                    let p = a.get("noul").and_then(Value::as_f64).unwrap_or(0.5);
                    ("noul", vec!["yes".into(), "no".into()], vec![p, 1.0 - p])
                }
                Question::Choice { criteria, .. } => {
                    let keys: Vec<String> = criteria.keys().cloned().collect();
                    let ps = keys.iter().map(|k| prob(probs, k)).collect();
                    ("choice", keys, ps)
                }
                Question::Score { criteria, .. } => {
                    let keys: Vec<String> = (0..criteria.len()).map(|i| i.to_string()).collect();
                    let ps = keys.iter().map(|k| prob(probs, k)).collect();
                    ("score", keys, ps)
                }
            };
            let Some(g) = c.gold.get(id) else { continue };
            let Some(gold) = gold_index(g, &keys) else {
                return Err(BackendError::Rejected(format!(
                    "case {ci} question `{id}`: gold {g} is not an option"
                )));
            };
            rows.push(Row {
                case_index: ci,
                id: id.clone(),
                kind: kind.to_string(),
                n: keys.len(),
                raw_logprobs: ps.into_iter().map(lnp).collect(),
                gold,
                latency_ms: each,
                prompt_evaluated: if qi == 0 { ev.usage.input_tokens } else { 0 },
                prompt_cached: 0,
            });
        }
    }
    Ok((rows, failed))
}

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub questions: usize,
    /// Cases whose backend call failed (no rows; not counted in accuracy).
    pub failed_cases: usize,
    pub accuracy: f64,
    pub brier: f64,
    /// Top-label expected calibration error, 10 equal-width bins.
    pub ece: f64,
    pub mean_confidence: f64,
    /// Coverage at <=5% empirical error when gating on confidence.
    pub coverage_at_5pct_error: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub prompt_tokens_evaluated: u64,
    pub prompt_tokens_cached: u64,
    pub by_kind: BTreeMap<String, f64>,
}

pub fn metrics(rows: &[Row], failed_cases: usize, cal: &Calibration) -> Metrics {
    let mut correct = 0usize;
    let mut brier = 0.0;
    let mut conf_sum = 0.0;
    let mut bins = vec![(0usize, 0usize, 0.0f64); 10]; // (n, correct, conf_sum)
    let mut gated: Vec<(f64, bool)> = Vec::new();
    let mut kind_tot: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut lat: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for r in rows {
        let t = cal.temperature(&r.kind, r.n);
        let p = softmax(&r.raw_logprobs, t);
        let pred = argmax(&p);
        let ok = pred == r.gold;
        correct += ok as usize;
        let pmax = p[pred];
        conf_sum += confidence(&r.kind, &p);
        brier += p
            .iter()
            .enumerate()
            .map(|(i, &pi)| {
                let y = if i == r.gold { 1.0 } else { 0.0 };
                (pi - y).powi(2)
            })
            .sum::<f64>();
        let b = ((pmax * 10.0).floor() as usize).min(9);
        bins[b].0 += 1;
        bins[b].1 += ok as usize;
        bins[b].2 += pmax;
        gated.push((confidence(&r.kind, &p), ok));
        let e = kind_tot.entry(r.kind.clone()).or_default();
        e.0 += 1;
        e.1 += ok as usize;
    }
    let n = rows.len().max(1) as f64;
    let ece = bins
        .iter()
        .filter(|(c, _, _)| *c > 0)
        .map(|(c, k, s)| (*c as f64 / n) * ((*k as f64 / *c as f64) - (s / *c as f64)).abs())
        .sum();
    // coverage: sort by confidence desc, take the longest prefix with error <= 5%
    // Coverage is what a confidence threshold accepts, and a threshold
    // cannot split rows of equal confidence: cut only where it changes.
    gated.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut best = 0usize;
    let mut wrong = 0usize;
    for (i, (c, ok)) in gated.iter().enumerate() {
        wrong += (!ok) as usize;
        let boundary = gated.get(i + 1).is_none_or(|next| next.0 != *c);
        if boundary && wrong as f64 / (i + 1) as f64 <= 0.05 {
            best = i + 1;
        }
    }
    let pct = |q: f64| -> f64 {
        if lat.is_empty() {
            0.0
        } else {
            lat[((lat.len() - 1) as f64 * q).round() as usize]
        }
    };
    Metrics {
        questions: rows.len(),
        failed_cases,
        accuracy: correct as f64 / n,
        brier: brier / n,
        ece,
        mean_confidence: conf_sum / n,
        coverage_at_5pct_error: best as f64 / n,
        latency_p50_ms: pct(0.5),
        latency_p95_ms: pct(0.95),
        prompt_tokens_evaluated: rows.iter().map(|r| r.prompt_evaluated).sum(),
        prompt_tokens_cached: rows.iter().map(|r| r.prompt_cached).sum(),
        by_kind: kind_tot
            .into_iter()
            .map(|(k, (t, c))| (k, c as f64 / t.max(1) as f64))
            .collect(),
    }
}

pub fn fit(rows: &[Row]) -> Calibration {
    let samples: Vec<(String, Vec<f64>, usize)> = rows
        .iter()
        .map(|r| {
            (
                Calibration::bucket(&r.kind, r.n),
                r.raw_logprobs.clone(),
                r.gold,
            )
        })
        .collect();
    Calibration::fit(&samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn keys(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_gold_label_names_an_option_or_nothing() {
        let yn = keys(&["yes", "no"]);
        let lv = keys(&["0", "1", "2"]);
        let num = keys(&["16", "32", "64"]);
        assert_eq!(gold_index(&json!(true), &yn), Some(0));
        assert_eq!(gold_index(&json!("false"), &yn), Some(1));
        assert_eq!(gold_index(&json!("no"), &yn), Some(1));
        assert_eq!(gold_index(&json!(2), &lv), Some(2));
        assert_eq!(gold_index(&json!("2"), &lv), Some(2));
        // 1-based or stray labels are not options, never an index past the end.
        assert_eq!(gold_index(&json!(3), &lv), None);
        assert_eq!(gold_index(&json!("5"), &lv), None);
        // A number that is a key is that key, not an index.
        assert_eq!(gold_index(&json!(32), &num), Some(1));
        assert_eq!(gold_index(&json!(true), &lv), None);
    }

    #[test]
    fn coverage_does_not_split_tied_confidences() {
        // Twenty rows of equal confidence, ten right and ten wrong: no
        // threshold takes some of them, so nothing is covered.
        let row = |gold: usize| Row {
            case_index: 0,
            id: String::new(),
            kind: "noul".into(),
            n: 2,
            raw_logprobs: vec![0.9f64.ln(), 0.1f64.ln()],
            gold,
            latency_ms: 0.0,
            prompt_evaluated: 0,
            prompt_cached: 0,
        };
        let rows: Vec<Row> = (0..20).map(|i| row(usize::from(i >= 10))).collect();
        let m = metrics(&rows, 0, &Calibration::default());
        assert_eq!(m.coverage_at_5pct_error, 0.0);
    }
}
