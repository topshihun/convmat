//! convmat — a MATLAB/Octave-to-C compiler that aims to be a better
//! MATLAB Coder.
//!
//! Pipeline:
//!
//! ```text
//! MATLAB source
//!   -> runmat frontend (lexer / parser / HIR / MIR)   [src/frontend]
//!   -> triage (static vs dynamic classification)       [src/triage]
//!   -> MIR -> matlab dialect (pliron)                  [src/mir_to_mlir]
//!   -> matlab -> emitc dialect (pliron)                [src/lowering]
//!   -> emitc -> C                                      [src/emit_c]
//! ```

pub mod backend;
pub mod builtins;
pub mod dialects;
pub mod emit_c;
pub mod error;
pub mod frontend;
pub mod lowering;
pub mod mir_to_mlir;
pub mod pipeline;
pub mod runtime;
pub mod triage;

pub use error::{Error, Result};
