//! convmat — a MATLAB/Octave-to-C compiler that aims to be a better
//! MATLAB Coder.
//!
//! Pipeline:
//!
//! ```text
//! MATLAB source
//!   -> runmat frontend (lexer / parser / HIR)          [src/frontend]
//!   -> triage (static vs dynamic classification)       [src/triage]
//!   -> HIR -> matlab dialect (pliron)                  [src/hir_to_mlir]
//!   -> matlab -> emitc dialect (pliron)                [src/lowering]
//!   -> emitc -> C                                      [src/emit_c]
//! ```

pub mod backend;
pub mod builtins;
pub mod dialects;
pub mod emit_c;
pub mod error;
pub mod frontend;
pub mod hir_to_mlir;
pub mod lowering;
pub mod passes;
pub mod pipeline;
pub mod runtime;
pub mod triage;

pub use error::{Error, Result};
