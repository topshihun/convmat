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
//! ([`DYNAMIC_RUNTIME_H`]), the value-model kernel implementation
//! ([`DYNAMIC_RUNTIME_C`]), and the ABI/function symbol constants.
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
pub const SORT: &str = "convmat_sort";
pub const SORT_COLS: &str = "convmat_sort_cols";
pub const SUM: &str = "convmat_sum";
pub const PROD: &str = "convmat_prod";
pub const MIN: &str = "convmat_min";
pub const MAX: &str = "convmat_max";
pub const COPY: &str = "convmat_copy";
pub const SCALE: &str = "convmat_scale";
pub const ADD: &str = "convmat_add";
pub const SUB: &str = "convmat_sub";
pub const EWMUL: &str = "convmat_ewmul";
pub const NEG: &str = "convmat_neg";
pub const ADD_SCALAR: &str = "convmat_add_scalar";
pub const SUB_SCALAR: &str = "convmat_sub_scalar";
pub const RSUB_SCALAR: &str = "convmat_rsub_scalar";
pub const DIV_SCALAR: &str = "convmat_div_scalar";
pub const RDIV_SCALAR: &str = "convmat_rdiv_scalar";
pub const EWDIV: &str = "convmat_ewdiv";
pub const INV: &str = "convmat_inv";
pub const DET: &str = "convmat_det";
pub const NORM: &str = "convmat_norm";
pub const SOLVE: &str = "convmat_solve";
pub const RAND: &str = "convmat_rand";

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
/// The `convmat_*` implementation bodies are in [`DYNAMIC_RUNTIME_C`]; both are
/// emitted together (header then impl) when the dynamic tier is used (see
/// `docs/runtime.md` §8).
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
            char **field_names;\n\
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

/// The C implementation of the dynamic value-model kernel: lifecycle, deep
/// copy, and the array/cell/struct helpers (docs/runtime.md §2–§4). Storage is
/// `double` (convmat's single numeric type) for `CONVMAT_DOUBLE`/`CONVMAT_LOGICAL`;
/// other `convmat_dtype` variants are reserved. Emitted together with
/// [`DYNAMIC_RUNTIME_H`] when a function bridges into the dynamic tier.
pub const DYNAMIC_RUNTIME_C: &str = r#"// ---- convmat runtime: value-model kernel (docs/runtime.md) ----
// Lifecycle, deep copy, and the array/cell/struct helpers. The static tier
// (matlab dialect) stays in plain `double[N]` / `struct`; only values that
// cross into the dynamic tier are boxed here. Storage is `double` (convmat's
// single numeric type) for CONVMAT_DOUBLE/CONVMAT_LOGICAL; other dtypes are
// reserved.

static bool convmat_streq(const char *a, const char *b) {
    while (*a != 0 && *b != 0 && *a == *b) { ++a; ++b; }
    return *a == *b;
}

static int64_t *convmat_dims_clone(int64_t ndims, const int64_t *dims) {
    if (ndims <= 0 || dims == nullptr) return nullptr;
    int64_t *out = new int64_t[ndims];
    for (int64_t i = 0; i < ndims; i++) out[i] = dims[i];
    return out;
}

static int64_t convmat_dims_numel(int64_t ndims, const int64_t *dims) {
    int64_t n = 1;
    for (int64_t i = 0; i < ndims; i++) n *= dims[i];
    return n;
}

convmat_value *convmat_value_new(convmat_kind kind, convmat_dtype dtype) {
    convmat_value *v = new convmat_value();
    v->refcount = 1;
    v->kind = kind;
    v->dtype = dtype;
    switch (kind) {
        case CONVMAT_SCALAR: v->u.scalar.d = 0.0; break;
        case CONVMAT_ARRAY:
            v->u.array.shape.ndims = 0;
            v->u.array.shape.dims = nullptr;
            v->u.array.data = nullptr;
            v->u.array.capacity = 0;
            v->u.array.owns_data = 0;
            break;
        case CONVMAT_CELL:
            v->u.cell.shape.ndims = 0;
            v->u.cell.shape.dims = nullptr;
            v->u.cell.elems = nullptr;
            break;
        case CONVMAT_STRUCT:
            v->u.strct.shape.ndims = 0;
            v->u.strct.shape.dims = nullptr;
            v->u.strct.nfields = 0;
            v->u.strct.field_names = nullptr;
            v->u.strct.fields = nullptr;
            break;
        case CONVMAT_FUNCTION: v->u.func.handle_id = -1; break;
        case CONVMAT_EMPTY: break;
    }
    return v;
}

convmat_value *convmat_value_retain(convmat_value *v) {
    if (v != nullptr) v->refcount++;
    return v;
}

void convmat_value_release(convmat_value *v) {
    if (v == nullptr) return;
    v->refcount--;
    if (v->refcount > 0) return;
    switch (v->kind) {
        case CONVMAT_ARRAY:
            delete[] v->u.array.shape.dims;
            if (v->u.array.owns_data) delete[] static_cast<double *>(v->u.array.data);
            break;
        case CONVMAT_CELL: {
            int64_t n = convmat_dims_numel(v->u.cell.shape.ndims, v->u.cell.shape.dims);
            for (int64_t i = 0; i < n; i++) convmat_value_release(v->u.cell.elems[i]);
            delete[] v->u.cell.elems;
            delete[] v->u.cell.shape.dims;
            break;
        }
        case CONVMAT_STRUCT: {
            int64_t n = convmat_dims_numel(v->u.strct.shape.ndims, v->u.strct.shape.dims);
            int64_t total = n * v->u.strct.nfields;
            for (int64_t i = 0; i < total; i++) convmat_value_release(v->u.strct.fields[i]);
            delete[] v->u.strct.fields;
            for (int64_t f = 0; f < v->u.strct.nfields; f++) delete[] v->u.strct.field_names[f];
            delete[] v->u.strct.field_names;
            delete[] v->u.strct.shape.dims;
            break;
        }
        default: break;
    }
    delete v;
}

int64_t convmat_numel(const convmat_value *v) {
    switch (v->kind) {
        case CONVMAT_ARRAY: return convmat_dims_numel(v->u.array.shape.ndims, v->u.array.shape.dims);
        case CONVMAT_CELL: return convmat_dims_numel(v->u.cell.shape.ndims, v->u.cell.shape.dims);
        case CONVMAT_STRUCT: return convmat_dims_numel(v->u.strct.shape.ndims, v->u.strct.shape.dims);
        default: return 1;
    }
}

int64_t convmat_linear_index(const convmat_value *v, const int64_t *subs) {
    int64_t ndims = 0;
    const int64_t *dims = nullptr;
    switch (v->kind) {
        case CONVMAT_ARRAY: ndims = v->u.array.shape.ndims; dims = v->u.array.shape.dims; break;
        case CONVMAT_CELL: ndims = v->u.cell.shape.ndims; dims = v->u.cell.shape.dims; break;
        case CONVMAT_STRUCT: ndims = v->u.strct.shape.ndims; dims = v->u.strct.shape.dims; break;
        default: return 0;
    }
    int64_t offset = 0;
    int64_t stride = 1;
    for (int64_t i = 0; i < ndims; i++) {
        offset += subs[i] * stride;
        stride *= dims[i];
    }
    return offset;
}

convmat_value *convmat_array_create(convmat_dtype dtype, int64_t ndims, const int64_t *dims) {
    convmat_value *v = convmat_value_new(CONVMAT_ARRAY, dtype);
    v->u.array.shape.ndims = ndims;
    v->u.array.shape.dims = convmat_dims_clone(ndims, dims);
    int64_t n = convmat_dims_numel(ndims, dims);
    v->u.array.data = new double[n]();
    v->u.array.capacity = n;
    v->u.array.owns_data = 1;
    return v;
}

void *convmat_array_data(convmat_value *v) {
    return v->u.array.data;
}

void convmat_array_resize(convmat_value *v, int64_t ndims, const int64_t *dims) {
    int64_t newn = convmat_dims_numel(ndims, dims);
    if (newn > v->u.array.capacity) {
        double *oldbuf = static_cast<double *>(v->u.array.data);
        double *newbuf = new double[newn]();
        if (oldbuf != nullptr) {
            int64_t copy_n = v->u.array.capacity < newn ? v->u.array.capacity : newn;
            for (int64_t i = 0; i < copy_n; i++) newbuf[i] = oldbuf[i];
            if (v->u.array.owns_data) delete[] oldbuf;
        }
        v->u.array.data = newbuf;
        v->u.array.capacity = newn;
        v->u.array.owns_data = 1;
    }
    delete[] v->u.array.shape.dims;
    v->u.array.shape.ndims = ndims;
    v->u.array.shape.dims = convmat_dims_clone(ndims, dims);
}

convmat_value *convmat_cell_create(int64_t ndims, const int64_t *dims) {
    convmat_value *v = convmat_value_new(CONVMAT_CELL, CONVMAT_DOUBLE);
    v->u.cell.shape.ndims = ndims;
    v->u.cell.shape.dims = convmat_dims_clone(ndims, dims);
    int64_t n = convmat_dims_numel(ndims, dims);
    v->u.cell.elems = new convmat_value *[n];
    for (int64_t i = 0; i < n; i++) v->u.cell.elems[i] = nullptr;
    return v;
}

convmat_value *convmat_cell_get(const convmat_value *c, int64_t lin) {
    return convmat_value_retain(c->u.cell.elems[lin]);
}

void convmat_cell_set(convmat_value *c, int64_t lin, convmat_value *v) {
    convmat_value *old = c->u.cell.elems[lin];
    c->u.cell.elems[lin] = convmat_value_retain(v);
    convmat_value_release(old);
}

convmat_value *convmat_struct_create(int64_t nfields, const char *const *names,
                                     int64_t ndims, const int64_t *dims) {
    convmat_value *v = convmat_value_new(CONVMAT_STRUCT, CONVMAT_DOUBLE);
    v->u.strct.shape.ndims = ndims;
    v->u.strct.shape.dims = convmat_dims_clone(ndims, dims);
    v->u.strct.nfields = nfields;
    v->u.strct.field_names = new char *[nfields];
    for (int64_t f = 0; f < nfields; f++) {
        const char *src = names[f];
        int64_t len = 0;
        while (src[len] != 0) len++;
        char *dst = new char[len + 1];
        for (int64_t i = 0; i <= len; i++) dst[i] = src[i];
        v->u.strct.field_names[f] = dst;
    }
    int64_t n = convmat_dims_numel(ndims, dims);
    int64_t total = n * nfields;
    v->u.strct.fields = new convmat_value *[total];
    for (int64_t i = 0; i < total; i++) v->u.strct.fields[i] = nullptr;
    return v;
}

int64_t convmat_struct_field_index(const convmat_value *s, const char *name) {
    for (int64_t f = 0; f < s->u.strct.nfields; f++) {
        if (convmat_streq(s->u.strct.field_names[f], name)) return f;
    }
    return -1;
}

convmat_value *convmat_struct_get(const convmat_value *s, int64_t field, int64_t lin) {
    int64_t n = convmat_numel(s);
    return convmat_value_retain(s->u.strct.fields[field * n + lin]);
}

void convmat_struct_set(convmat_value *s, int64_t field, int64_t lin, convmat_value *v) {
    int64_t n = convmat_numel(s);
    convmat_value *old = s->u.strct.fields[field * n + lin];
    s->u.strct.fields[field * n + lin] = convmat_value_retain(v);
    convmat_value_release(old);
}

convmat_value *convmat_value_copy(const convmat_value *src) {
    if (src == nullptr) return nullptr;
    switch (src->kind) {
        case CONVMAT_EMPTY:
            return convmat_value_new(CONVMAT_EMPTY, src->dtype);
        case CONVMAT_SCALAR: {
            convmat_value *v = convmat_value_new(CONVMAT_SCALAR, src->dtype);
            v->u.scalar.d = src->u.scalar.d;
            return v;
        }
        case CONVMAT_ARRAY: {
            convmat_value *v = convmat_array_create(src->dtype, src->u.array.shape.ndims, src->u.array.shape.dims);
            int64_t n = convmat_numel(src);
            double *dst = static_cast<double *>(v->u.array.data);
            const double *sdata = static_cast<const double *>(src->u.array.data);
            for (int64_t i = 0; i < n; i++) dst[i] = sdata[i];
            return v;
        }
        case CONVMAT_CELL: {
            convmat_value *v = convmat_cell_create(src->u.cell.shape.ndims, src->u.cell.shape.dims);
            int64_t n = convmat_numel(src);
            for (int64_t i = 0; i < n; i++) v->u.cell.elems[i] = convmat_value_copy(src->u.cell.elems[i]);
            return v;
        }
        case CONVMAT_STRUCT: {
            convmat_value *v = convmat_struct_create(src->u.strct.nfields, src->u.strct.field_names, src->u.strct.shape.ndims, src->u.strct.shape.dims);
            int64_t n = convmat_numel(src);
            for (int64_t i = 0; i < n * src->u.strct.nfields; i++) v->u.strct.fields[i] = convmat_value_copy(src->u.strct.fields[i]);
            return v;
        }
        case CONVMAT_FUNCTION: {
            convmat_value *v = convmat_value_new(CONVMAT_FUNCTION, src->dtype);
            v->u.func.handle_id = src->u.func.handle_id;
            return v;
        }
    }
    return nullptr;
}
"#;

/// The C source of a wrapped-operator helper by name, or `None` for an unknown
/// name. Dims are passed as `double` (convmat's single numeric type) and cast to
/// `int` internally; they are small integers.
pub fn helper_source(name: &str) -> Option<&'static str> {
    Some(match name {
        TRANSPOSE => TRANSPOSE_C,
        MATMUL => MATMUL_C,
        MPOWER => MPOWER_C,
        SORT => SORT_C,
        SORT_COLS => SORT_COLS_C,
        SUM => SUM_C,
        PROD => PROD_C,
        MIN => MIN_C,
        MAX => MAX_C,
        COPY => COPY_C,
        SCALE => SCALE_C,
        ADD => ADD_C,
        SUB => SUB_C,
        EWMUL => EWMUL_C,
        NEG => NEG_C,
        ADD_SCALAR => ADD_SCALAR_C,
        SUB_SCALAR => SUB_SCALAR_C,
        RSUB_SCALAR => RSUB_SCALAR_C,
        DIV_SCALAR => DIV_SCALAR_C,
        RDIV_SCALAR => RDIV_SCALAR_C,
        EWDIV => EWDIV_C,
        INV => INV_C,
        DET => DET_C,
        NORM => NORM_C,
        SOLVE => SOLVE_C,
        RAND => RAND_C,
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

const SORT_C: &str = "\
// ---- convmat_sort: dst = src sorted ascending (flattened, vector) ----\n\
void convmat_sort(double* dst, const double* src, double n) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = src[i];\n\
    for (int i = 1; i < m; i++) {\n\
        double key = dst[i];\n\
        int j = i - 1;\n\
        while (j >= 0 && dst[j] > key) { dst[j + 1] = dst[j]; j--; }\n\
        dst[j + 1] = key;\n\
    }\n\
}\n\
";

const SORT_COLS_C: &str = "\
// ---- convmat_sort_cols: sort each column of a (rows x cols) matrix ascending ----\n\
void convmat_sort_cols(double* dst, const double* src, double rows, double cols) {\n\
    int r = (int)rows, c = (int)cols;\n\
    for (int j = 0; j < c; j++) {\n\
        for (int i = 0; i < r; i++) dst[i + j * r] = src[i + j * r];\n\
        for (int i = 1; i < r; i++) {\n\
            double key = dst[i + j * r];\n\
            int k = i - 1;\n\
            while (k >= 0 && dst[k + j * r] > key) {\n\
                dst[(k + 1) + j * r] = dst[k + j * r];\n\
                k--;\n\
            }\n\
            dst[(k + 1) + j * r] = key;\n\
        }\n\
    }\n\
}\n\
";

const SUM_C: &str = "\
// ---- convmat_sum: sum(src[0..n)) ----\n\
double convmat_sum(const double* src, double n) {\n\
    int m = (int)n;\n\
    double acc = 0.0;\n\
    for (int i = 0; i < m; i++) acc += src[i];\n\
    return acc;\n\
}\n\
";

const PROD_C: &str = "\
// ---- convmat_prod: prod(src[0..n)) ----\n\
double convmat_prod(const double* src, double n) {\n\
    int m = (int)n;\n\
    double acc = 1.0;\n\
    for (int i = 0; i < m; i++) acc *= src[i];\n\
    return acc;\n\
}\n\
";

const MIN_C: &str = "\
// ---- convmat_min: min(src[0..n)) ----\n\
double convmat_min(const double* src, double n) {\n\
    int m = (int)n;\n\
    double acc = m > 0 ? src[0] : 0.0;\n\
    for (int i = 1; i < m; i++) if (src[i] < acc) acc = src[i];\n\
    return acc;\n\
}\n\
";

const MAX_C: &str = "\
// ---- convmat_max: max(src[0..n)) ----\n\
double convmat_max(const double* src, double n) {\n\
    int m = (int)n;\n\
    double acc = m > 0 ? src[0] : 0.0;\n\
    for (int i = 1; i < m; i++) if (src[i] > acc) acc = src[i];\n\
    return acc;\n\
}\n\
";

const COPY_C: &str = "\
// ---- convmat_copy: dst[0..n) = src[0..n) ----\n\
void convmat_copy(double* dst, const double* src, double n) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = src[i];\n\
}\n\
";

const SCALE_C: &str = "\
// ---- convmat_scale: dst[0..n) = src[0..n) * k (scalar broadcast) ----\n\
void convmat_scale(double* dst, const double* src, double n, double k) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = src[i] * k;\n\
}\n\
";

const ADD_C: &str = "\
// ---- convmat_add: dst[0..n) = a[0..n) + b[0..n) (elementwise) ----\n\
void convmat_add(double* dst, const double* a, const double* b, double n) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = a[i] + b[i];\n\
}\n\
";

const SUB_C: &str = "\
// ---- convmat_sub: dst[0..n) = a[0..n) - b[0..n) (elementwise) ----\n\
void convmat_sub(double* dst, const double* a, const double* b, double n) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = a[i] - b[i];\n\
}\n\
";

const EWMUL_C: &str = "\
// ---- convmat_ewmul: dst[0..n) = a[0..n) .* b[0..n) (elementwise) ----\n\
void convmat_ewmul(double* dst, const double* a, const double* b, double n) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = a[i] * b[i];\n\
}\n\
";

const NEG_C: &str = "\
// ---- convmat_neg: dst[0..n) = -src[0..n) ----\n\
void convmat_neg(double* dst, const double* src, double n) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = -src[i];\n\
}\n\
";

const ADD_SCALAR_C: &str = "\
// ---- convmat_add_scalar: dst[0..n) = src[0..n) + k (scalar broadcast) ----\n\
void convmat_add_scalar(double* dst, const double* src, double n, double k) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = src[i] + k;\n\
}\n\
";

const SUB_SCALAR_C: &str = "\
// ---- convmat_sub_scalar: dst[0..n) = src[0..n) - k (scalar broadcast) ----\n\
void convmat_sub_scalar(double* dst, const double* src, double n, double k) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = src[i] - k;\n\
}\n\
";

const RSUB_SCALAR_C: &str = "\
// ---- convmat_rsub_scalar: dst[0..n) = k - src[0..n) (scalar broadcast) ----\n\
void convmat_rsub_scalar(double* dst, const double* src, double n, double k) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = k - src[i];\n\
}\n\
";

const DIV_SCALAR_C: &str = "\
// ---- convmat_div_scalar: dst[0..n) = src[0..n) / k (scalar broadcast) ----\n\
void convmat_div_scalar(double* dst, const double* src, double n, double k) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = src[i] / k;\n\
}\n\
";

const RDIV_SCALAR_C: &str = "\
// ---- convmat_rdiv_scalar: dst[0..n) = k / src[0..n) (scalar broadcast) ----\n\
void convmat_rdiv_scalar(double* dst, const double* src, double n, double k) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = k / src[i];\n\
}\n\
";

const EWDIV_C: &str = "\
// ---- convmat_ewdiv: dst[0..n) = a[0..n) ./ b[0..n) (elementwise) ----\n\
void convmat_ewdiv(double* dst, const double* a, const double* b, double n) {\n\
    int m = (int)n;\n\
    for (int i = 0; i < m; i++) dst[i] = a[i] / b[i];\n\
}\n\
";

const INV_C: &str = "\
// ---- convmat_inv: dst (n x n) = inverse of a (n x n), Gauss-Jordan with\n\
// partial pivoting. A singular input yields a zero matrix. ----\n\
void convmat_inv(double* dst, const double* a, double n) {\n\
    int m = (int)n;\n\
    double* aug = new double[m * 2 * m];\n\
    for (int i = 0; i < m; i++) {\n\
        for (int j = 0; j < m; j++) {\n\
            aug[i + j * m] = a[i + j * m];\n\
            aug[i + (j + m) * m] = (i == j) ? 1.0 : 0.0;\n\
        }\n\
    }\n\
    for (int k = 0; k < m; k++) {\n\
        int piv = k;\n\
        double best = std::fabs(aug[k + k * m]);\n\
        for (int i = k + 1; i < m; i++) {\n\
            double v = std::fabs(aug[i + k * m]);\n\
            if (v > best) { best = v; piv = i; }\n\
        }\n\
        if (best == 0.0) {\n\
            for (int i = 0; i < m * m; i++) dst[i] = 0.0;\n\
            delete[] aug;\n\
            return;\n\
        }\n\
        if (piv != k) {\n\
            for (int j = 0; j < 2 * m; j++) {\n\
                double t = aug[k + j * m];\n\
                aug[k + j * m] = aug[piv + j * m];\n\
                aug[piv + j * m] = t;\n\
            }\n\
        }\n\
        double d = aug[k + k * m];\n\
        for (int j = 0; j < 2 * m; j++) aug[k + j * m] /= d;\n\
        for (int i = 0; i < m; i++) {\n\
            if (i == k) continue;\n\
            double f = aug[i + k * m];\n\
            for (int j = 0; j < 2 * m; j++) aug[i + j * m] -= f * aug[k + j * m];\n\
        }\n\
    }\n\
    for (int i = 0; i < m; i++)\n\
        for (int j = 0; j < m; j++) dst[i + j * m] = aug[i + (j + m) * m];\n\
    delete[] aug;\n\
}\n\
";

const DET_C: &str = "\
// ---- convmat_det: determinant of a (n x n) via LU with partial pivoting ----\n\
double convmat_det(const double* a, double n) {\n\
    int m = (int)n;\n\
    double* lu = new double[m * m];\n\
    for (int i = 0; i < m * m; i++) lu[i] = a[i];\n\
    double det = 1.0;\n\
    for (int k = 0; k < m; k++) {\n\
        int piv = k;\n\
        double best = std::fabs(lu[k + k * m]);\n\
        for (int i = k + 1; i < m; i++) {\n\
            double v = std::fabs(lu[i + k * m]);\n\
            if (v > best) { best = v; piv = i; }\n\
        }\n\
        if (best == 0.0) { det = 0.0; break; }\n\
        if (piv != k) {\n\
            for (int j = 0; j < m; j++) {\n\
                double t = lu[k + j * m];\n\
                lu[k + j * m] = lu[piv + j * m];\n\
                lu[piv + j * m] = t;\n\
            }\n\
            det = -det;\n\
        }\n\
        det *= lu[k + k * m];\n\
        for (int i = k + 1; i < m; i++) {\n\
            double f = lu[i + k * m] / lu[k + k * m];\n\
            for (int j = k; j < m; j++) lu[i + j * m] -= f * lu[k + j * m];\n\
        }\n\
    }\n\
    delete[] lu;\n\
    return det;\n\
}\n\
";

const NORM_C: &str = "\
// ---- convmat_norm: 2-norm of a vector a[0..n) ----\n\
double convmat_norm(const double* a, double n) {\n\
    int m = (int)n;\n\
    double s = 0.0;\n\
    for (int i = 0; i < m; i++) s += a[i] * a[i];\n\
    return std::sqrt(s);\n\
}\n\
";

const SOLVE_C: &str = "\
// ---- convmat_solve: X (n x k) solves A (n x n) X = B (n x k), Gaussian\n\
// elimination with partial pivoting. A singular system yields a zero X. ----\n\
void convmat_solve(double* dst, const double* a, const double* b, double n, double k) {\n\
    int m = (int)n, kk = (int)k;\n\
    double* aug = new double[m * (m + kk)];\n\
    for (int i = 0; i < m; i++) {\n\
        for (int j = 0; j < m; j++) aug[i + j * m] = a[i + j * m];\n\
        for (int j = 0; j < kk; j++) aug[i + (m + j) * m] = b[i + j * m];\n\
    }\n\
    for (int c = 0; c < m; c++) {\n\
        int piv = c;\n\
        double best = std::fabs(aug[c + c * m]);\n\
        for (int i = c + 1; i < m; i++) {\n\
            double v = std::fabs(aug[i + c * m]);\n\
            if (v > best) { best = v; piv = i; }\n\
        }\n\
        if (best == 0.0) {\n\
            for (int i = 0; i < m * kk; i++) dst[i] = 0.0;\n\
            delete[] aug;\n\
            return;\n\
        }\n\
        if (piv != c) {\n\
            for (int j = 0; j < m + kk; j++) {\n\
                double t = aug[c + j * m];\n\
                aug[c + j * m] = aug[piv + j * m];\n\
                aug[piv + j * m] = t;\n\
            }\n\
        }\n\
        for (int i = c + 1; i < m; i++) {\n\
            double f = aug[i + c * m] / aug[c + c * m];\n\
            for (int j = c; j < m + kk; j++) aug[i + j * m] -= f * aug[c + j * m];\n\
        }\n\
    }\n\
    for (int j = 0; j < kk; j++) {\n\
        for (int i = m - 1; i >= 0; i--) {\n\
            double s = aug[i + (m + j) * m];\n\
            for (int p = i + 1; p < m; p++) s -= aug[i + p * m] * dst[p + j * m];\n\
            dst[i + j * m] = s / aug[i + i * m];\n\
        }\n\
    }\n\
    delete[] aug;\n\
}\n\
";

const RAND_C: &str = "\
// ---- convmat_rand: a pseudo-random double in [0, 1), xorshift64* ----\n\
double convmat_rand(void) {\n\
    static uint64_t state = 88172645463325252ULL;\n\
    state ^= state << 13;\n\
    state ^= state >> 7;\n\
    state ^= state << 17;\n\
    return (double)(state >> 11) * (1.0 / 9007199254740992.0);\n\
}\n\
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The wrapped-operator helper registry. Kept next to `helper_source` so the
    /// supported runtime surface is explicit and machine-checked (see
    /// `docs/runtime.md` §9).
    const WRAPPED_HELPERS: &[&str] = &[
        TRANSPOSE,
        MATMUL,
        MPOWER,
        SORT,
        SORT_COLS,
        SUM,
        PROD,
        MIN,
        MAX,
        COPY,
        SCALE,
        ADD,
        SUB,
        EWMUL,
        NEG,
        ADD_SCALAR,
        SUB_SCALAR,
        RSUB_SCALAR,
        DIV_SCALAR,
        RDIV_SCALAR,
        EWDIV,
        INV,
        DET,
        NORM,
        SOLVE,
        RAND,
    ];

    /// The dynamic-tier kernel symbols that `DYNAMIC_RUNTIME_H` declares and
    /// `DYNAMIC_RUNTIME_C` must implement (see `docs/runtime.md` §9.2).
    const DYNAMIC_KERNEL_SYMBOLS: &[&str] = &[
        "convmat_value_new",
        "convmat_value_retain",
        "convmat_value_release",
        "convmat_value_copy",
        "convmat_array_create",
        "convmat_array_data",
        "convmat_array_resize",
        "convmat_cell_create",
        "convmat_cell_get",
        "convmat_cell_set",
        "convmat_struct_create",
        "convmat_struct_field_index",
        "convmat_struct_get",
        "convmat_struct_set",
        "convmat_numel",
        "convmat_linear_index",
    ];

    #[test]
    fn every_registered_helper_has_source() {
        for name in WRAPPED_HELPERS {
            assert!(helper_source(name).is_some(), "no C source for `{name}`");
        }
    }

    #[test]
    fn helper_sources_define_their_registered_symbol() {
        // A copy/paste error (source under the wrong key) fails here instead of
        // producing an undefined symbol at C link time.
        for name in WRAPPED_HELPERS {
            let source = helper_source(name).unwrap();
            assert!(
                source.contains(name),
                "`{name}` source does not define `{name}`"
            );
        }
    }

    #[test]
    fn unknown_helper_has_no_source() {
        assert!(helper_source("convmat_nope").is_none());
    }

    #[test]
    fn dynamic_kernel_implements_its_declared_symbols() {
        for symbol in DYNAMIC_KERNEL_SYMBOLS {
            assert!(
                DYNAMIC_RUNTIME_H.contains(symbol),
                "header is missing `{symbol}`"
            );
            assert!(
                DYNAMIC_RUNTIME_C.contains(symbol),
                "kernel is missing an implementation for `{symbol}`"
            );
        }
    }
}
