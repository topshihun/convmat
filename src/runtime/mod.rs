//! Layer 6: runtime support for code the static boundary cannot lower.
//!
//! When triage defers a function (dynamic typing, unknown shape, unsupported
//! construct), the architecture lowers it to a call into the convmat runtime
//! instead of failing the whole compile. This module is the seam for that path:
//! it owns the runtime symbol registry, the dynamic value model (boxed
//! `convmat_value`), memory management, and built-ins.
//!
//! The MVP has no dynamic-tier shim yet, so the fallback is currently a hard
//! error ([`defer_to_runtime`]). The consolidated design of that dynamic tier is
//! in `docs/runtime.md`; this module already carries its C header
//! ([`DYNAMIC_RUNTIME_H`]) and the ABI/function symbol constants.
//!
//! This module also owns the "封装" (wrapped) operator helpers: the small C
//! library of `convmat_*` functions that implement operations which change
//! memory layout or carry a non-trivial algorithm (transpose, matrix multiply,
//! integer matrix power). Each helper is a named [`helper_source`] entry that
//! [`crate::lowering`] emits only when a wrapped operator actually references
//! it (the MVP inlines them instead of linking a separate object).

use crate::error::{Error, Result};

/// The runtime helper names, used by `hir_to_mlir` to reference the wrapped
/// operators. Kept as constants so a typo fails at compile time rather than
/// producing an unknown-helper error deep in lowering.
pub const TRANSPOSE: &str = "convmat_transpose";
pub const MATMUL: &str = "convmat_matmul";
pub const MPOWER: &str = "convmat_mpower";

// --- Dynamic tier (see docs/runtime.md) -------------------------------------
//
// The static tier (`matlab` dialect → `double[N]` / fixed `struct`) can only
// express values whose type and shape are known at compile time. The dynamic
// tier — array parameters, cell arrays, strings, open-world varargin/varargout,
// `try`/`catch`, `feval`/handles — needs boxed values whose type/shape are only
// known at run time. The consolidated design lives in `docs/runtime.md`; this
// module owns its runtime symbol registry and the C header that the future
// lowering emits (via `VerbatimOp`, the same mechanism as [`helper_source`]).

/// The runtime ABI typedef name for a dynamic-tier function.
pub const RUNTIME_FN: &str = "convmat_runtime_fn";

/// The C header (types + prototypes) for the dynamic value model. Emitted as a
/// `VerbatimOp` once, before any function that bridges into the dynamic tier.
/// The `convmat_*` implementation bodies follow the same inline-emission path as
/// [`helper_source`] when the dynamic tier lands (see `docs/runtime.md` §8).
pub const DYNAMIC_RUNTIME_H: &str = "\
// ---- convmat runtime: dynamic tier value model (docs/runtime.md) ----\n\
// Emitted once, at file scope, before functions that bridge into the dynamic\n\
// tier. `int64_t` comes from <cstdint> (always in the generated preamble); the\n\
// try/catch record also needs <setjmp.h> once error propagation lands.\n\
typedef enum convmat_dtype {\n\
    CONVMAT_DOUBLE = 0,\n\
    CONVMAT_LOGICAL,\n\
    CONVMAT_INT32,\n\
    CONVMAT_CHAR,\n\
} convmat_dtype;\n\
\n\
typedef enum convmat_kind {\n\
    CONVMAT_EMPTY = 0,\n\
    CONVMAT_SCALAR,\n\
    CONVMAT_ARRAY,\n\
    CONVMAT_CELL,\n\
    CONVMAT_STRUCT,\n\
    CONVMAT_FUNCTION,\n\
} convmat_kind;\n\
\n\
typedef struct convmat_dims {\n\
    int64_t ndims;\n\
    int64_t *dims;\n\
} convmat_dims;\n\
\n\
typedef struct convmat_value {\n\
    int32_t refcount;\n\
    convmat_kind kind;\n\
    convmat_dtype dtype;\n\
    union {\n\
        struct { double d; } scalar;\n\
        struct {\n\
            convmat_dims shape;\n\
            void *data;\n\
            int64_t capacity;\n\
            int owns_data;\n\
        } array;\n\
        struct {\n\
            convmat_dims shape;\n\
            convmat_value **elems;\n\
        } cell;\n\
        struct {\n\
            convmat_dims shape;\n\
            int64_t nfields;\n\
            const char **field_names;\n\
            convmat_value **fields;\n\
        } strct;\n\
        struct {\n\
            int64_t handle_id;\n\
        } func;\n\
    } u;\n\
} convmat_value;\n\
\n\
// --- lifecycle / memory (docs/runtime.md §4) ---\n\
convmat_value *convmat_value_new(convmat_kind kind, convmat_dtype dtype);\n\
convmat_value *convmat_value_retain(convmat_value *v);\n\
void convmat_value_release(convmat_value *v);\n\
convmat_value *convmat_value_copy(const convmat_value *v);\n\
convmat_value *convmat_array_create(convmat_dtype dtype, int64_t ndims, const int64_t *dims);\n\
void *convmat_array_data(convmat_value *v);\n\
void convmat_array_resize(convmat_value *v, int64_t ndims, const int64_t *dims);\n\
convmat_value *convmat_cell_create(int64_t ndims, const int64_t *dims);\n\
convmat_value *convmat_cell_get(const convmat_value *c, int64_t lin);\n\
void convmat_cell_set(convmat_value *c, int64_t lin, convmat_value *v);\n\
convmat_value *convmat_struct_create(int64_t nfields, const char *const *names,\n\
                                     int64_t ndims, const int64_t *dims);\n\
int64_t convmat_struct_field_index(const convmat_value *s, const char *name);\n\
convmat_value *convmat_struct_get(const convmat_value *s, int64_t field, int64_t lin);\n\
void convmat_struct_set(convmat_value *s, int64_t field, int64_t lin, convmat_value *v);\n\
int64_t convmat_numel(const convmat_value *v);\n\
int64_t convmat_linear_index(const convmat_value *v, const int64_t *subs);\n\
\n\
// --- dynamic ABI / error (docs/runtime.md §5, §7) ---\n\
typedef void (*convmat_runtime_fn)(int64_t nargs, convmat_value **args,\n\
                                   int64_t nouts, convmat_value **outs);\n\
\n\
typedef struct convmat_error {\n\
    jmp_buf jmp;\n\
    char message[256];\n\
    int armed;\n\
} convmat_error;\n\
\n\
void convmat_error_throw(const char *msg);\n\
";

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
/// This is the single choke point where deferred code turns into a dynamic-tier
/// call (the static↔dynamic seam in `docs/runtime.md` §6). Until the dynamic
/// tier exists it reports the reason as [`Error::NotLowerable`], so the pipeline
/// still compiles the statically lowerable subset.
///
/// When the dynamic tier lands, this seam will (1) box any static operands into
/// [`DYNAMIC_RUNTIME_H`]'s `convmat_value` (borrowed `array` with `owns_data=0`),
/// (2) emit a `convmat_runtime_fn`-style call, and (3) unbox any result whose
/// `kind`/`dtype`/`dims` match a statically known layout back into `double[N]`.
pub fn defer_to_runtime(reason: &str) -> Result<()> {
    Err(Error::NotLowerable(reason.to_string()))
}
