//! Layer 5: code-generation backends.
//!
//! The C backend is the only one implemented today: the pipeline lowers to the
//! `emitc` dialect and prints C directly (no external tool needed). `Llvm` and
//! `Gpu` are reserved without changing the frontend or lowering.

/// Available backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    C,
    Llvm,
    Gpu,
}

impl BackendKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            BackendKind::C => "c",
            BackendKind::Llvm => "llvm",
            BackendKind::Gpu => "gpu",
        }
    }
}
