//! Layer 6: runtime support for code the static boundary cannot lower.
//!
//! When triage defers a function (dynamic typing, unknown shape, unsupported
//! construct), the architecture lowers it to a `func.call` into the
//! runmat/convmat runtime instead of failing the whole compile. This module is
//! the seam for that path: it will own the runtime symbol registry, dynamic
//! type boxes, memory management, and built-ins.
//!
//! The MVP has no runtime shim yet, so the fallback is currently a hard error.

use crate::error::{Error, Result};

/// Lower a function that failed static triage into a runtime call.
///
/// This is the single choke point where deferred code turns into a runtime
/// `func.call`. Until the runtime shim exists it reports the reason as
/// [`Error::NotLowerable`], so the pipeline still compiles the statically
/// lowerable subset. When the shim lands, this signature will grow to take the
/// deferred body and emit the runtime call in place of the error.
pub fn defer_to_runtime(reason: &str) -> Result<()> {
    Err(Error::NotLowerable(reason.to_string()))
}
