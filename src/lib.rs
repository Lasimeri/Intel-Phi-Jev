//! xks: XKEYSCORE for Jev. Typed System One judgments (`noul` / `choice` /
//! `score`) served on a Jev-compatible `POST /v1/systemone`, from a model
//! running on this host with its matrix multiplies shared with the Xeon Phi
//! cards.
//!
//! The foundation of its names is the Revelation to John (README, Naming):
//! an apocalypse, an unveiling, the subject's hidden judgment shown whole.
//!
//! Pipeline: [`prompt`] renders the session (the state) once and each
//! fingerprint (a question) as a suffix whose next token is an option
//! label; a [`backend::Scorer`] returns the label log-probabilities (the
//! in-process [`artichoke`] engine forks the session once per fingerprint
//! and reads them all in one batch); [`score`] normalises and conditions
//! (calibrates) them and derives the typed answer; [`judge::Judge`] ties
//! it together; [`server`] speaks HTTP and [`mcp`] speaks MCP.

#[cfg(feature = "artichoke")]
pub mod artichoke;
pub mod backend;
pub mod config;
pub mod corroborate;
pub mod decisions;
#[cfg(feature = "artichoke")]
pub mod doctor;
pub mod eval;
pub mod judge;
pub mod ledger;
pub mod mcp;
#[cfg(feature = "artichoke")]
pub mod polygraph;
pub mod prompt;
pub mod protocol;
pub mod score;
pub mod server;
#[cfg(feature = "artichoke")]
pub mod site;
#[cfg(feature = "artichoke")]
pub mod subproject;
