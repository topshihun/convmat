//! convmat's pliron dialects.
//!
//! - [`matlab`] — MATLAB-level semantics (what `hir_to_mlir` emits).
//! - [`emitc`] — C-level statements/expressions (what the C emitter prints).

pub mod emitc;
pub mod matlab;
