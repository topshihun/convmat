//! Layer 6: runtime support for code the static boundary cannot lower.
//!
//! When triage defers a function (dynamic typing, unknown shape, unsupported
//! construct), the architecture lowers it to a `func.call` into the
//! runmat/convmat runtime instead of failing the whole compile. This module is
//! the seam for that path: it will own the runtime symbol registry, dynamic
//! type boxes, memory management, and built-ins.
//!
//! The MVP has no runtime shim yet, so the fallback is currently a hard error.
//!
//! This module also owns the "封装" (wrapped) operator helpers: the small C
//! library of `convmat_*` functions that implement operations which change
//! memory layout or carry a non-trivial algorithm (transpose, matrix multiply,
//! integer matrix power). Each helper is a named [`helper_source`] entry that
//! [`crate::lowering`] emits only when a wrapped operator actually references
//! it (the MVP inlines them instead of linking a separate object).

use crate::error::{Error, Result};

/// The runtime helper names, used by `mir_to_mlir` to reference the wrapped
/// operators. Kept as constants so a typo fails at compile time rather than
/// producing an unknown-helper error deep in lowering.
pub const TRANSPOSE: &str = "convmat_transpose";
pub const MATMUL: &str = "convmat_matmul";
pub const MPOWER: &str = "convmat_mpower";

/// The C source of a wrapped-operator helper by name, or `None` for an unknown
/// name. Dims are passed as `double` (convmat's single numeric type) and cast to
/// `int` internally; they are small integers.
pub fn helper_source(name: &str) -> Option<&'static str> {
    Some(match name {
        TRANSPOSE => TRANSPOSE_C,
        MATMUL => MATMUL_C,
        MPOWER => MPOWER_C,
        _ => return None,
    })
}

const TRANSPOSE_C: &str = "\
// ---- convmat_transpose: dst (cols x rows) = src (rows x cols)^T ----
void convmat_transpose(double* dst, const double* src, double rows, double cols) {
    int r = (int)rows, c = (int)cols;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            dst[j * r + i] = src[i * c + j];
        }
    }
}
";

const MATMUL_C: &str = "\
// ---- convmat_matmul: dst (m x n) = a (m x k) * b (k x n) ----
void convmat_matmul(double* dst, const double* a, const double* b,
                    double m, double k, double n) {
    int mm = (int)m, kk = (int)k, nn = (int)n;
    for (int i = 0; i < mm; i++) {
        for (int j = 0; j < nn; j++) {
            double acc = 0.0;
            for (int p = 0; p < kk; p++) {
                acc += a[i + p * mm] * b[p + j * kk];
            }
            dst[i + j * mm] = acc;
        }
    }
}
";

const MPOWER_C: &str = "\
// ---- convmat_mpower: dst (n x n) = a (n x n) ^ k, integer k >= 0 ----
void convmat_mpower(double* dst, const double* a, double m, double k) {
    int n = (int)m;
    long long e = (long long)k;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i + j * n] = (i == j) ? 1.0 : 0.0;
        }
    }
    if (e == 0) return;  // A^0 = I
    if (e == 1) {        // A^1 = A (no heap allocation)
        for (int i = 0; i < n * n; i++) dst[i] = a[i];
        return;
    }
    double* base = new double[n * n];
    double* tmp = new double[n * n];
    for (int i = 0; i < n * n; i++) base[i] = a[i];
    while (e > 0) {
        if (e & 1) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double acc = 0.0;
                    for (int p = 0; p < n; p++) acc += dst[i + p * n] * base[p + j * n];
                    tmp[i + j * n] = acc;
                }
            }
            for (int i = 0; i < n * n; i++) dst[i] = tmp[i];
        }
        e >>= 1;
        if (e > 0) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double acc = 0.0;
                    for (int p = 0; p < n; p++) acc += base[i + p * n] * base[p + j * n];
                    tmp[i + j * n] = acc;
                }
            }
            for (int i = 0; i < n * n; i++) base[i] = tmp[i];
        }
    }
    delete[] base;
    delete[] tmp;
}
";

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
