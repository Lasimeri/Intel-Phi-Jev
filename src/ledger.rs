//! The books opened (Revelation 20:12): judged by what was written.
//!
//! The ledger: what the payload's verbose log (`PHI_GGML_VERBOSE=1`) says
//! each card and the host did, totalled, so "the cards did the work" is a
//! number and not a picture of busy threads (a waiting worker spins, and a
//! process monitor counts that as busy). See ledger.md.
//!
//! Reads lines of the form
//! `ggml-phi: multiply N: host part H ms, waited W ms more; card C rows R:
//! T ms (pull P, compute K, push U); ...`.

use std::collections::BTreeMap;
use std::io::BufRead;

use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Card {
    pub multiplies: u64,
    pub rows: u64,
    pub total_ms: f64,
    pub compute_ms: f64,
    pub pull_ms: f64,
    pub push_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Ledger {
    /// Multiplies the payload handled (declined ones never reach it).
    pub multiplies: u64,
    /// Of those, the ones at least one card took rows of.
    pub with_cards: u64,
    /// The host's own rows, summed.
    pub host_part_ms: f64,
    /// The host waiting for the cards after its own rows.
    pub host_wait_ms: f64,
    pub cards: BTreeMap<u32, Card>,
}

/// The number after `key` up to the next space or delimiter.
fn number_after(s: &str, key: &str) -> Option<f64> {
    let at = s.find(key)? + key.len();
    let rest = &s[at..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn parse_card(seg: &str) -> Option<(u32, Card)> {
    // "card C rows R: T ms (pull P, compute K, push U)"
    let seg = seg.trim();
    let id = number_after(seg, "card ")? as u32;
    let rows = number_after(seg, "rows ")? as u64;
    let total = number_after(seg, ": ")?;
    Some((
        id,
        Card {
            multiplies: 1,
            rows,
            total_ms: total,
            compute_ms: number_after(seg, "compute ")?,
            pull_ms: number_after(seg, "pull ")?,
            push_ms: number_after(seg, "push ")?,
        },
    ))
}

pub fn read(input: impl BufRead) -> Ledger {
    let mut l = Ledger::default();
    for line in input.lines().map_while(Result::ok) {
        let Some(at) = line.find("ggml-phi: multiply ") else {
            continue;
        };
        let line = &line[at..];
        let (Some(h), Some(w)) = (
            number_after(line, "host part "),
            number_after(line, "waited "),
        ) else {
            continue;
        };
        l.multiplies += 1;
        l.host_part_ms += h;
        l.host_wait_ms += w;
        let mut any = false;
        for seg in line
            .split("; ")
            .filter(|s| s.trim_start().starts_with("card "))
        {
            if let Some((id, c)) = parse_card(seg) {
                let e = l.cards.entry(id).or_default();
                e.multiplies += 1;
                e.rows += c.rows;
                e.total_ms += c.total_ms;
                e.compute_ms += c.compute_ms;
                e.pull_ms += c.pull_ms;
                e.push_ms += c.push_ms;
                any = true;
            }
        }
        l.with_cards += u64::from(any);
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_payload_format() {
        let log = "ggml-phi: multiply 1: host part 82.632 ms, waited 1.532 ms more; card 0 rows 1728: 19.698 ms (pull 1.226, compute 16.938, push 1.350); card 1 rows 1728: 19.855 ms (pull 1.617, compute 15.550, push 2.492); the cards' rows read by the host so far 0.0 MB\n\
ggml-phi: multiply 2: host part 1.027 ms, waited 0.001 ms more; ; the cards' rows read by the host so far 0.0 MB\n\
something else\n";
        let l = read(log.as_bytes());
        assert_eq!(l.multiplies, 2);
        assert_eq!(l.with_cards, 1);
        assert!((l.host_part_ms - 83.659).abs() < 1e-9);
        let c0 = &l.cards[&0];
        assert_eq!(c0.rows, 1728);
        assert!((c0.compute_ms - 16.938).abs() < 1e-9);
        assert!((l.cards[&1].push_ms - 2.492).abs() < 1e-9);
    }
}
