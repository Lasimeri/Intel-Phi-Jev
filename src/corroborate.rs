//! Corroborate: two recorded runs (`eval --rows`) of the same case file
//! compared fingerprint by fingerprint, the way a second source confirms a
//! reading. Its use here is the sites: the same subject on the x86-64
//! reference and on the cards (or through phi512), where every difference
//! is the site's arithmetic and not the model. See corroborate.md.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::eval::Row;
use crate::score::{argmax, softmax};

#[derive(Debug, Clone, Serialize)]
pub struct Pair {
    pub case_index: usize,
    pub id: String,
    /// Largest absolute difference of the label probabilities.
    pub max_prob_diff: f64,
    pub same_argmax: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub matched: usize,
    pub only_in_a: usize,
    pub only_in_b: usize,
    pub same_argmax: usize,
    pub max_prob_diff: f64,
    pub mean_prob_diff: f64,
    pub accuracy_a: f64,
    pub accuracy_b: f64,
    pub pairs: Vec<Pair>,
}

pub fn compare(a: &[Row], b: &[Row]) -> Report {
    let key = |r: &Row| (r.case_index, r.id.clone());
    let bm: BTreeMap<_, &Row> = b.iter().map(|r| (key(r), r)).collect();
    let mut pairs = Vec::new();
    let (mut ok_a, mut ok_b) = (0usize, 0usize);
    for ra in a {
        let Some(rb) = bm.get(&key(ra)) else {
            continue;
        };
        let pa = softmax(&ra.raw_logprobs, 1.0);
        let pb = softmax(&rb.raw_logprobs, 1.0);
        let max_prob_diff = pa
            .iter()
            .zip(&pb)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        let (aa, ab) = (argmax(&pa), argmax(&pb));
        ok_a += usize::from(aa == ra.gold);
        ok_b += usize::from(ab == rb.gold);
        pairs.push(Pair {
            case_index: ra.case_index,
            id: ra.id.clone(),
            max_prob_diff,
            same_argmax: aa == ab,
        });
    }
    let n = pairs.len().max(1) as f64;
    Report {
        matched: pairs.len(),
        only_in_a: a.len() - pairs.len(),
        only_in_b: b.len() - pairs.len(),
        same_argmax: pairs.iter().filter(|p| p.same_argmax).count(),
        max_prob_diff: pairs.iter().map(|p| p.max_prob_diff).fold(0.0, f64::max),
        mean_prob_diff: pairs.iter().map(|p| p.max_prob_diff).sum::<f64>() / n,
        accuracy_a: ok_a as f64 / n,
        accuracy_b: ok_b as f64 / n,
        pairs,
    }
}
