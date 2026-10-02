//! convmat's pliron dialects.
//!
//! - [`matlab`] — MATLAB-level semantics (what `mir_to_mlir` emits).
//! - [`emitc`] — C-level statements/expressions (what the C emitter prints).

pub mod emitc;
pub mod matlab;
