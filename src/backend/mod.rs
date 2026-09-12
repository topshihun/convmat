//! Layer 5: code-generation backends.
//!
//! The backend consumes `emitc`-dialect MLIR and produces target code. Today
//! only the `C` backend is implemented (via `mlir-translate --mlir-to-cpp`);
//! `Llvm` and `Gpu` are reserved without changing the frontend or lowering.

use crate::error::{Error, Result};

/// Available backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    C,
    Llvm,
    Gpu,
}

impl BackendKind {
    fn name(self) -> &'static str {
        match self {
            BackendKind::C => "c",
            BackendKind::Llvm => "llvm",
            BackendKind::Gpu => "gpu",
        }
    }

    /// Emit target code from `emitc`-dialect MLIR.
    pub fn emit(self, emitc_mlir: &str) -> Result<String> {
        match self {
            BackendKind::C => translate_to_c(emitc_mlir),
            BackendKind::Llvm | BackendKind::Gpu => {
                Err(Error::NotImplemented(format!("{} backend", self.name())))
            }
        }
    }
}

/// Translate `emitc`-dialect MLIR to C source via `mlir-translate`.
///
/// TODO: melior exposes no "translate to C" API, so this still shells out to
/// `mlir-translate --mlir-to-cpp`. Replace it with an in-process translator (or
/// emit C directly from the `emitc` ops) once one is available.
pub fn translate_to_c(emitc_mlir: &str) -> Result<String> {
    crate::tool::run_tool("mlir-translate", &["--mlir-to-cpp"], emitc_mlir)
}
