//! Error types shared across the pipeline.

use thiserror::Error;

/// Errors produced by convmat.
#[derive(Debug, Error)]
pub enum Error {
    /// An underlying I/O failure (e.g. reading a source file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The frontend failed (runmat parse/HIR/MIR lowering).
    #[error("frontend error: {0}")]
    Frontend(String),

    /// A function was classified as not statically lowerable (deferred to the
    /// runtime).
    #[error("not lowerable: {0}")]
    NotLowerable(String),

    /// A backend failed to emit code (MLIR/emitc tools or melior).
    #[error("backend error: {0}")]
    Backend(String),

    /// A backend is planned but not implemented yet.
    #[error("not implemented yet: {0}")]
    NotImplemented(String),
}

/// Convenience result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
