//! A tiny synchronous HTTP server exposing `POST /v1/systemone` (the Jev
//! wire format, so the TypeSafe SDKs work against it through
//! `TYPESAFE_BASE_URL`), with an optional kill date: an idle period after
//! which it exits and gives the cards back ("that there should be time no
//! longer", Revelation 10:6). See server.md.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tiny_http::{Header, Method, Request as HttpRequest, Response, Server};

use crate::backend::{BackendError, Scorer};
use crate::judge::Judge;
use crate::protocol::Request;

pub struct ServerConfig {
    pub bind: String,
    /// Accepted bearer tokens; empty = no auth.
    pub api_keys: Vec<String>,
    /// Exit after this long with no request in flight or arriving.
    pub kill_date: Option<Duration>,
    /// Where the subject runs (`x86`, `cards`, `avx512`, `remote`), for
    /// `/health` and the index.
    pub site: String,
    /// The cards the subject uses, for `/health`.
    pub cards: Vec<u32>,
    /// Where each question set and its answers are recorded, if anywhere
    /// ([`crate::decisions`]).
    pub decision_log: Option<crate::decisions::Log>,
}

/// What a request asks for, from its method and path alone.
#[derive(Debug, PartialEq)]
enum Route {
    Index,
    Health,
    Models,
    SystemOne,
    /// A known path asked with the wrong method: the one it takes.
    Method(&'static str),
    NotFound,
}

/// A trailing slash is the same path; a known path with another method
/// is a 405 naming the right one, not a 404.
fn route(method: &Method, path: &str) -> Route {
    let path = match path.trim_end_matches('/') {
        "" => "/",
        p => p,
    };
    let (want, r) = match path {
        "/" => ("GET", Route::Index),
        "/health" => ("GET", Route::Health),
        "/v1/models" => ("GET", Route::Models),
        "/v1/systemone" => ("POST", Route::SystemOne),
        _ => return Route::NotFound,
    };
    if method.as_str() == want {
        r
    } else {
        Route::Method(want)
    }
}

pub fn serve<S: Scorer + 'static>(judge: Judge<S>, cfg: ServerConfig) -> Result<(), String> {
    let server = Server::http(&cfg.bind).map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    let judge = Arc::new(judge);
    let keys = Arc::new(cfg.api_keys.clone());
    let about = Arc::new(About {
        site: cfg.site.clone(),
        cards: cfg.cards.clone(),
        kill_date: cfg.kill_date,
        last: Mutex::new(Instant::now()),
        log: cfg.decision_log,
    });
    let busy = Arc::new(AtomicUsize::new(0));
    eprintln!(
        "xks listening on http://{}  (subject {}){}",
        cfg.bind,
        judge.scorer.model_name(),
        match cfg.kill_date {
            Some(d) => format!(
                "; stops itself after {} without a question (kill date)",
                span(d)
            ),
            None => String::new(),
        }
    );
    loop {
        let req = match server.recv_timeout(Duration::from_secs(1)) {
            Ok(Some(r)) => r,
            Ok(None) => {
                if let Some(d) = cfg.kill_date {
                    let idle = about.last.lock().expect("last").elapsed();
                    if busy.load(Ordering::SeqCst) == 0 && idle >= d {
                        eprintln!(
                            "xks: kill date reached: {} without a question, exiting",
                            span(idle)
                        );
                        return Ok(());
                    }
                }
                continue;
            }
            Err(e) => return Err(format!("accept: {e}")),
        };
        let judge = judge.clone();
        let keys = keys.clone();
        let about = about.clone();
        let busy = busy.clone();
        busy.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
            if handle(req, &judge, &keys, &about) {
                *about.last.lock().expect("last") = Instant::now();
            }
            busy.fetch_sub(1, Ordering::SeqCst);
        });
    }
}

/// What `/health` and the index say beside the subject, and the clock the
/// kill date runs on.
struct About {
    site: String,
    cards: Vec<u32>,
    kill_date: Option<Duration>,
    /// When the last question ended, or the server started. Only a
    /// question counts: a monitor polling `/health` (Mechanical Jev's TUI
    /// does, every 5 s) must not keep alive a server nobody is asking.
    last: Mutex<Instant>,
    /// The decision log, when the server keeps one.
    log: Option<crate::decisions::Log>,
}

/// A duration as a person reads it: whole minutes when it is some, else
/// seconds.
fn span(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 60 && s.is_multiple_of(60) {
        format!("{} min", s / 60)
    } else {
        format!("{s} s")
    }
}

/// Answer one request; whether it was a question (what the kill date
/// counts).
fn handle<S: Scorer>(
    mut req: HttpRequest,
    judge: &Judge<S>,
    keys: &[String],
    about: &About,
) -> bool {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    let subject = judge.scorer.model_name();
    let mut allow = None;
    let route = route(&method, &path);
    let question = route == Route::SystemOne;
    let result: (u16, Value) = match route {
        Route::Index => (
            200,
            json!({
                "service": "xks, a local Jev (TypeSafe System One)",
                "subject": subject,
                "site": about.site,
                "routes": {
                    "POST /v1/systemone": "a request: {\"state\": ..., \"questions\": {...}}",
                    "GET /health": "the subject and where it runs",
                    "GET /v1/models": "the model ids this server answers to",
                },
                "docs": "https://github.com/Lasimeri/Intel-Phi-Jev",
            }),
        ),
        Route::Health => (
            200,
            json!({
                "status": "ok",
                "subject": subject,
                "site": about.site,
                "cards": about.cards,
                // The kill date in seconds (0: never), and how long since
                // the last question: when the server will stop itself.
                "kill_date_s": about.kill_date.map_or(0, |d| d.as_secs()),
                "idle_s": about.last.lock().expect("last").elapsed().as_secs(),
            }),
        ),
        Route::Models => (
            200,
            json!({"object": "list", "data": [
                {"id": subject, "object": "model", "owned_by": "xks"},
                {"id": "jev-latest", "object": "model", "owned_by": "xks", "alias_of": subject}
            ]}),
        ),
        Route::SystemOne => {
            if !keys.is_empty() && !authorized(&req, keys) {
                (
                    401,
                    json!({"message": "invalid or missing API key (Authorization: Bearer KEY)"}),
                )
            } else {
                let mut body = String::new();
                if req.as_reader().read_to_string(&mut body).is_err() {
                    (400, json!({"message": "unreadable body"}))
                } else if body.trim().is_empty() {
                    (
                        422,
                        json!({"message": "empty body: POST a JSON request, {\"state\": ..., \"questions\": {...}}"}),
                    )
                } else {
                    let t0 = Instant::now();
                    // As JSON first (a syntax error keeps its line and
                    // column), then as a request; the log keeps the JSON.
                    let parsed: Result<Value, String> =
                        serde_json::from_str(&body).map_err(|e| e.to_string());
                    let asked = parsed.clone().and_then(|v| {
                        serde_json::from_value::<Request>(v).map_err(|e| e.to_string())
                    });
                    let out = match asked {
                        Err(e) => (422, json!({"message": format!("invalid request: {e}")})),
                        Ok(r) => match judge.evaluate(&r) {
                            Ok(ev) => (200, serde_json::to_value(ev).unwrap()),
                            Err(BackendError::Rejected(m)) => (422, json!({"message": m})),
                            Err(e) => (502, json!({"message": e.to_string()})),
                        },
                    };
                    if let Some(log) = &about.log {
                        let ms = t0.elapsed().as_secs_f64() * 1e3;
                        log.record(parsed.as_ref().ok(), out.0, &out.1, ms);
                    }
                    out
                }
            }
        }
        Route::Method(want) => {
            allow = Some(want);
            (
                405,
                json!({"message": format!("{path} takes {want}, not {method}")}),
            )
        }
        Route::NotFound => (
            404,
            json!({"message": format!("no route {method} {path} (POST /v1/systemone; GET / lists the routes)")}),
        ),
    };
    let json_hdr = Header::from_bytes("Content-Type", "application/json").unwrap();
    let mut resp = Response::from_string(result.1.to_string())
        .with_status_code(result.0)
        .with_header(json_hdr);
    if let Some(m) = allow {
        resp = resp.with_header(Header::from_bytes("Allow", m).unwrap());
    }
    let _ = req.respond(resp);
    question
}

fn authorized(req: &HttpRequest, keys: &[String]) -> bool {
    req.headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| {
            h.value
                .as_str()
                .trim_start_matches("Bearer ")
                .trim()
                .to_string()
        })
        .map(|k| keys.iter().any(|x| x == &k))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_by_method_and_path() {
        assert_eq!(route(&Method::Post, "/v1/systemone"), Route::SystemOne);
        assert_eq!(route(&Method::Post, "/v1/systemone/"), Route::SystemOne);
        assert_eq!(route(&Method::Get, "/health"), Route::Health);
        assert_eq!(route(&Method::Get, "/v1/models"), Route::Models);
        assert_eq!(route(&Method::Get, "/"), Route::Index);
        assert_eq!(route(&Method::Get, ""), Route::Index);
        // The path is right, the method is not: say which one it takes.
        assert_eq!(route(&Method::Get, "/v1/systemone"), Route::Method("POST"));
        assert_eq!(route(&Method::Post, "/health"), Route::Method("GET"));
        assert_eq!(route(&Method::Post, "/v1/systemon"), Route::NotFound);
        assert_eq!(route(&Method::Get, "/v2/systemone"), Route::NotFound);
    }

    #[test]
    fn a_span_reads_as_minutes_when_whole() {
        assert_eq!(span(Duration::from_secs(1800)), "30 min");
        assert_eq!(span(Duration::from_secs(90)), "90 s");
        assert_eq!(span(Duration::from_secs(5)), "5 s");
    }
}
