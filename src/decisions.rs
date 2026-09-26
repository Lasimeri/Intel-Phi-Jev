//! The decision log: one JSON line per question set the server answers,
//! so every decision can be traced from what was asked to what was
//! answered ("the books opened", Revelation 20:12: judged by what was
//! written). Off unless asked for; by default it keeps a hash of the
//! state and of each question, not their text. See decisions.md.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

/// A UTC time as `YYYY-MM-DDTHH:MM:SSZ` (days to a civil date, Howard
/// Hinnant's algorithm).
pub fn utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// FNV-1a, 64 bits, as 16 hex digits: the same text always gives the same
/// mark, across runs and builds, which is all a log needs to say "the same
/// state as before" without keeping it.
pub fn fnv64(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// A JSON value's mark: its serialization hashed (key order is kept, so
/// the same request marks the same).
fn mark(v: &Value) -> String {
    fnv64(v.to_string().as_bytes())
}

/// The log, open for appending; lines from concurrent requests do not
/// interleave.
pub struct Log {
    file: Mutex<File>,
    /// Keep the state and the questions as sent, not their marks.
    pub with_text: bool,
}

impl Log {
    /// Open (or create, with its directory) the log at `path`.
    pub fn open(path: &Path, with_text: bool) -> Result<Self, String> {
        if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("decision log {}: {e}", path.display()))?;
        Ok(Self {
            file: Mutex::new(file),
            with_text,
        })
    }

    /// One question set: the request (when it parsed), the HTTP status,
    /// and the response body (answers, or the message of a refusal).
    pub fn record(&self, request: Option<&Value>, status: u16, body: &Value, ms: f64) {
        let line = entry(request, status, body, ms, self.with_text, now());
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The line for one question set.
pub fn entry(
    request: Option<&Value>,
    status: u16,
    body: &Value,
    ms: f64,
    with_text: bool,
    secs: u64,
) -> Value {
    let mut e = Map::new();
    e.insert("utc".into(), json!(utc(secs)));
    e.insert("status".into(), json!(status));
    e.insert("ms".into(), json!((ms * 10.0).round() / 10.0));
    if let Some(r) = request {
        let state = r.get("state").cloned().unwrap_or(Value::Null);
        let questions = r.get("questions").cloned().unwrap_or(Value::Null);
        if with_text {
            e.insert("state".into(), state);
            e.insert("questions".into(), questions);
        } else {
            let chars = match &state {
                Value::String(s) => s.chars().count(),
                other => other.to_string().chars().count(),
            };
            e.insert(
                "state".into(),
                json!({"fnv64": mark(&state), "chars": chars}),
            );
            let marks: Map<String, Value> = questions
                .as_object()
                .map(|q| {
                    q.iter()
                        .map(|(id, v)| {
                            (id.clone(), json!({"type": v.get("type"), "fnv64": mark(v)}))
                        })
                        .collect()
                })
                .unwrap_or_default();
            e.insert("questions".into(), Value::Object(marks));
        }
    }
    match body.get("answers") {
        Some(a) => {
            e.insert(
                "model".into(),
                body.get("model").cloned().unwrap_or(Value::Null),
            );
            e.insert("answers".into(), a.clone());
        }
        None => {
            e.insert(
                "message".into(),
                body.get("message").cloned().unwrap_or(Value::Null),
            );
        }
    }
    Value::Object(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_formats() {
        assert_eq!(utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc(1_790_000_000), "2026-09-21T14:13:20Z");
    }

    #[test]
    fn fnv64_is_the_published_fnv1a() {
        // FNV-1a 64 test vectors (the empty string and "a").
        assert_eq!(fnv64(b""), "cbf29ce484222325");
        assert_eq!(fnv64(b"a"), "af63dc4c8601ec8c");
    }

    #[test]
    fn a_line_keeps_marks_not_text_unless_asked() {
        let req = json!({"state": "Help! My payouts have been failing for 3 days.",
            "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}});
        let body = json!({"model": "artichoke/m", "answers": {"is_urgent": {"type": "noul", "noul": 0.9}}});
        let e = entry(Some(&req), 200, &body, 12.34, false, 0);
        assert_eq!(e["utc"], "1970-01-01T00:00:00Z");
        assert_eq!(e["ms"], json!(12.3));
        assert_eq!(e["state"]["chars"], json!(46));
        assert_eq!(e["state"]["fnv64"].as_str().unwrap().len(), 16);
        assert_eq!(e["questions"]["is_urgent"]["type"], "noul");
        assert!(
            !e.to_string().contains("payouts"),
            "the state's text is not kept"
        );
        assert!(!e.to_string().contains("urgency?"), "nor the question's");
        assert_eq!(e["answers"]["is_urgent"]["noul"], json!(0.9));
        // The same request marks the same.
        let again = entry(Some(&req), 200, &body, 1.0, false, 0);
        assert_eq!(again["state"]["fnv64"], e["state"]["fnv64"]);
        // With text, as sent.
        let t = entry(Some(&req), 200, &body, 1.0, true, 0);
        assert_eq!(t["state"], req["state"]);
        // A refusal keeps its message, and a body that did not parse no request.
        let r = entry(
            None,
            422,
            &json!({"message": "invalid request"}),
            0.5,
            false,
            0,
        );
        assert_eq!(r["message"], "invalid request");
        assert!(r.get("state").is_none());
    }

    #[test]
    fn lines_are_appended_to_the_file() {
        let p =
            std::env::temp_dir().join(format!("xks-decisions-{}/log.jsonl", std::process::id()));
        let log = Log::open(&p, false).unwrap();
        log.record(None, 422, &json!({"message": "a"}), 1.0);
        log.record(None, 422, &json!({"message": "b"}), 1.0);
        let text = std::fs::read_to_string(&p).unwrap();
        assert_eq!(text.lines().count(), 2);
        std::fs::remove_dir_all(p.parent().unwrap()).unwrap();
    }
}
