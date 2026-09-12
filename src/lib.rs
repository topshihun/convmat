//! convmat — a MATLAB/Octave-to-MLIR compiler that aims to be a better
//! MATLAB Coder.
//!
//! Pipeline:
//!
//! ```text
//! MATLAB source
//!   -> runmat frontend (lexer / parser / HIR / MIR)   [src/frontend]
//!   -> triage (static vs dynamic classification)       [src/triage]
//!   -> MIR -> MLIR (melior, core dialects)             [src/mir_to_mlir]
//!   -> MLIR passes (-> emitc)                          [src/passes]
//!   -> backend (emitc -> C today; LLVM/GPU reserved)   [src/backend]
//!   -> runtime fallback (deferred code)                [src/runtime]
//! ```

pub mod backend;
pub mod builtins;
pub mod error;
pub mod frontend;
pub mod mir_to_mlir;
pub mod passes;
pub mod pipeline;
pub mod runtime;
pub(crate) mod tool;
pub mod triage;

pub use error::{Error, Result};
