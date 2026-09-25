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
}

pub fn serve<S: Scorer + 'static>(judge: Judge<S>, cfg: ServerConfig) -> Result<(), String> {
    let server = Server::http(&cfg.bind).map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    let judge = Arc::new(judge);
    let keys = Arc::new(cfg.api_keys);
    let busy = Arc::new(AtomicUsize::new(0));
    let last = Arc::new(Mutex::new(Instant::now()));
    eprintln!(
        "xks listening on http://{}  (subject {}){}",
        cfg.bind,
        judge.scorer.model_name(),
        match cfg.kill_date {
            Some(d) => format!(", kill date {} s idle", d.as_secs()),
            None => String::new(),
        }
    );
    loop {
        let req = match server.recv_timeout(Duration::from_secs(1)) {
            Ok(Some(r)) => r,
            Ok(None) => {
                if let Some(d) = cfg.kill_date {
                    let idle = last.lock().expect("last").elapsed();
                    if busy.load(Ordering::SeqCst) == 0 && idle >= d {
                        eprintln!(
                            "xks: kill date reached ({} s idle), exiting",
                            idle.as_secs()
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
        let busy = busy.clone();
        let last = last.clone();
        busy.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
            handle(req, &judge, &keys);
            *last.lock().expect("last") = Instant::now();
            busy.fetch_sub(1, Ordering::SeqCst);
        });
    }
}

fn handle<S: Scorer>(mut req: HttpRequest, judge: &Judge<S>, keys: &[String]) {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    let result: (u16, Value) = match (&method, path.as_str()) {
        (Method::Get, "/health") => (
            200,
            json!({"status": "ok", "subject": judge.scorer.model_name()}),
        ),
        (Method::Get, "/v1/models") => (
            200,
            json!({"object": "list", "data": [
                {"id": judge.scorer.model_name(), "object": "model", "owned_by": "xks"},
                {"id": "jev-latest", "object": "model", "owned_by": "xks", "alias_of": judge.scorer.model_name()}
            ]}),
        ),
        (Method::Post, "/v1/systemone") => {
            if !keys.is_empty() && !authorized(&req, keys) {
                (401, json!({"message": "invalid or missing API key"}))
            } else {
                let mut body = String::new();
                if req.as_reader().read_to_string(&mut body).is_err() {
                    (400, json!({"message": "unreadable body"}))
                } else {
                    match serde_json::from_str::<Request>(&body) {
                        Err(e) => (422, json!({"message": format!("invalid request: {e}")})),
                        Ok(r) => match judge.evaluate(&r) {
                            Ok(ev) => (200, serde_json::to_value(ev).unwrap()),
                            Err(BackendError::Rejected(m)) => (422, json!({"message": m})),
                            Err(e) => (502, json!({"message": e.to_string()})),
                        },
                    }
                }
            }
        }
        _ => (
            404,
            json!({"message": format!("no route {} {}", method, path)}),
        ),
    };
    let json_hdr = Header::from_bytes("Content-Type", "application/json").unwrap();
    let resp = Response::from_string(result.1.to_string())
        .with_status_code(result.0)
        .with_header(json_hdr);
    let _ = req.respond(resp);
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
