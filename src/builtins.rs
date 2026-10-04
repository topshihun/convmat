//! Lowering recipes for MATLAB built-in functions.
//!
//! Maps a built-in name to a small description of how it should be lowered by
//! the C backend. The set is deliberately limited to the pure numeric
//! elementwise/reduction subset that can be expressed as a C `libm` call or a
//! short inline pattern; everything else (impure, dynamic, shape-dependent) is
//! left for the runtime fallback (see [`crate::triage`]).

/// A supported MATLAB built-in and its lowering recipe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Builtin {
    /// Unary elementwise function (scalar or array): `libm` symbol.
    Unary(&'static str),
    /// Binary elementwise function (scalar only): `libm` symbol.
    Binary(&'static str),
    /// `mod(x, y)`: MATLAB remainder with the sign of `y` (floor semantics).
    /// Unlike [`Builtin::Binary`], it cannot be a single `libm` call and is
    /// lowered inline as `fmod(fmod(x, y) + y, y)`.
    Mod,
    /// `sign(x)` lowered inline as `-1`/`0`/`1`.
    Sign,
    /// `min`/`max`: elementwise `fmin`/`fmax` over two scalars, or a reduction
    /// over one array.
    MinMax(MinMax),
    /// A reduction over an array (`sum`, `prod`).
    Reduce(ReduceOp),
    /// `numel(x)`: total element count.
    Numel,
    /// `length(x)`: the largest dimension.
    Length,
    /// `size(x, dim)`: the extent of one dimension (or `[rows cols]` for
    /// `size(x)`).
    Size,
    /// `zeros`/`ones`: a constant-filled array whose dims come from scalar
    /// arguments.
    Fill(f64),
    /// `eye`: the identity matrix.
    Eye,
    /// `reshape(A, dims...)`: relayout `A` to the given static dims.
    Reshape,
    /// `sort(A)`: sort a vector ascending (a wrapped `convmat_sort` runtime
    /// helper; only vectors are supported in the MVP).
    Sort,
    /// `mean(A)`: arithmetic mean of a vector (a reduction to a scalar).
    Mean,
    /// `std(A)`: sample standard deviation of a vector (a reduction to a
    /// scalar).
    Std,
    /// `median(A)`: middle value of a sorted vector (a reduction to a scalar).
    Median,
    /// `cumsum(A)`: cumulative sum (an array of the same shape as `A`).
    CumSum,
    /// `diff(A)`: adjacent differences (an array one element shorter than `A`).
    Diff,
    /// `isempty(x)`: whether the value has zero elements (`0`/`1`).
    IsEmpty,
    /// `logical(x)`: elementwise `x != 0` (`0`/`1`), preserving shape.
    Logical,
    /// `var(A)`: sample variance of a vector (a reduction to a scalar).
    Var,
    /// `linspace(a, b, n)`: `n` evenly spaced points (a 1xN row).
    LinSpace,
    /// `repmat(A, m, n)`: tile `A` `m`x`n` times (an array).
    Repmat,
    /// `permute(A, order)`: reorder the dimensions of `A` (an array).
    Permute,
    /// `inv(A)`: the inverse of a square matrix (a runtime helper).
    Inv,
    /// `det(A)`: the determinant of a square matrix (a scalar).
    Det,
    /// `norm(v)`: the 2-norm of a vector (a scalar).
    Norm,
    /// `rand()`: a pseudo-random scalar in `[0, 1)`.
    Rand,
    /// `strcmp(a, b)`: whether two char arrays are equal (a scalar `0`/`1`).
    StrCmp,
    /// `int32(x)`: convert to a 32-bit signed integer scalar.
    Int32,
    /// `fft(x)`: the discrete Fourier transform (a runtime helper; complex).
    Fft,
    /// `eig(A)`: eigenvalues of a small square matrix (a runtime helper; complex).
    Eig,
    /// A graphics/output builtin with no C representation (`plot`, ...). Its
    /// arguments are evaluated for side effects and it lowers to the no-op
    /// runtime helper [`crate::runtime::PLOT`]; see `docs/unsupported.md`. It is
    /// meaningful only as an expression statement (using its result defers).
    Noop,
    /// `fopen(name, mode)`: open a file (a runtime helper); returns a scalar id.
    /// `name` is a char array; `mode` is a single-char code point (`'r'`/`'w'`).
    Fopen,
    /// `fread(fid)`: read a binary file of `double`s (a runtime helper); returns
    /// a dynamic-shape array.
    Fread,
    /// `fclose(fid)`: close a file (a runtime helper); returns a scalar status.
    Fclose,
}

/// Which of `min`/`max` a [`Builtin::MinMax`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinMax {
    Min,
    Max,
}

/// Which reduction a [`Builtin::Reduce`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
}

impl Builtin {
    /// Whether `n` positional arguments are valid for this built-in.
    pub fn valid_arity(self, n: usize) -> bool {
        match self {
            Builtin::Unary(_)
            | Builtin::Sign
            | Builtin::Numel
            | Builtin::Length
            | Builtin::Mean
            | Builtin::Std
            | Builtin::Median
            | Builtin::CumSum
            | Builtin::Diff
            | Builtin::IsEmpty
            | Builtin::Logical
            | Builtin::Var
            | Builtin::Inv
            | Builtin::Det
            | Builtin::Norm
            | Builtin::Int32
            | Builtin::Fft
            | Builtin::Eig => n == 1,
            Builtin::Rand => n == 0,
            Builtin::Binary(_) | Builtin::Mod | Builtin::StrCmp => n == 2,
            Builtin::MinMax(_) | Builtin::Reduce(_) | Builtin::Size | Builtin::Eye => {
                n == 1 || n == 2
            }
            Builtin::LinSpace | Builtin::Repmat => n == 2 || n == 3,
            Builtin::Permute => n == 2,
            // `zeros`/`ones` accept any number of dimensions (N-D).
            Builtin::Fill(_) => n >= 1,
            Builtin::Reshape => n == 3,
            Builtin::Sort => n == 1,
            // Graphics no-ops take one or more data arguments / format strings.
            Builtin::Noop => n >= 1,
            // File I/O: `fopen(name, mode)`, `fread(fid)`, `fclose(fid)`.
            Builtin::Fopen => n == 2,
            Builtin::Fread | Builtin::Fclose => n == 1,
        }
    }
}

/// Look up a supported built-in by name, returning `None` for anything the
/// codegen boundary does not yet lower (impure or dynamic built-ins).
pub fn lookup(name: &str) -> Option<Builtin> {
    Some(match name {
        "sin" => Builtin::Unary("sin"),
        "cos" => Builtin::Unary("cos"),
        "tan" => Builtin::Unary("tan"),
        "asin" => Builtin::Unary("asin"),
        "acos" => Builtin::Unary("acos"),
        "atan" => Builtin::Unary("atan"),
        "sinh" => Builtin::Unary("sinh"),
        "cosh" => Builtin::Unary("cosh"),
        "tanh" => Builtin::Unary("tanh"),
        "exp" => Builtin::Unary("exp"),
        "log" => Builtin::Unary("log"),
        "log10" => Builtin::Unary("log10"),
        "log2" => Builtin::Unary("log2"),
        "sqrt" => Builtin::Unary("sqrt"),
        "abs" => Builtin::Unary("fabs"),
        "floor" => Builtin::Unary("floor"),
        "ceil" => Builtin::Unary("ceil"),
        "round" => Builtin::Unary("round"),
        "sign" => Builtin::Sign,
        "pow" => Builtin::Binary("pow"),
        "atan2" => Builtin::Binary("atan2"),
        "hypot" => Builtin::Binary("hypot"),
        "mod" => Builtin::Mod,
        "rem" => Builtin::Binary("fmod"),
        "min" => Builtin::MinMax(MinMax::Min),
        "max" => Builtin::MinMax(MinMax::Max),
        "sum" => Builtin::Reduce(ReduceOp::Sum),
        "prod" => Builtin::Reduce(ReduceOp::Prod),
        "numel" => Builtin::Numel,
        "length" => Builtin::Length,
        "size" => Builtin::Size,
        "zeros" => Builtin::Fill(0.0),
        "ones" => Builtin::Fill(1.0),
        // `Inf`/`NaN` as constructors (`Inf(1, n)`): a constant-filled array.
        "Inf" | "Infinity" => Builtin::Fill(f64::INFINITY),
        "NaN" => Builtin::Fill(f64::NAN),
        "eye" => Builtin::Eye,
        "reshape" => Builtin::Reshape,
        "sort" => Builtin::Sort,
        "mean" => Builtin::Mean,
        "std" => Builtin::Std,
        "median" => Builtin::Median,
        "cumsum" => Builtin::CumSum,
        "diff" => Builtin::Diff,
        "isempty" => Builtin::IsEmpty,
        "logical" => Builtin::Logical,
        "var" => Builtin::Var,
        "linspace" => Builtin::LinSpace,
        "repmat" => Builtin::Repmat,
        "permute" => Builtin::Permute,
        "inv" => Builtin::Inv,
        "det" => Builtin::Det,
        "norm" => Builtin::Norm,
        "rand" => Builtin::Rand,
        "strcmp" => Builtin::StrCmp,
        "int32" => Builtin::Int32,
        "fft" => Builtin::Fft,
        "eig" => Builtin::Eig,
        "isnan" => Builtin::Unary("std::isnan"),
        "isinf" => Builtin::Unary("std::isinf"),
        // Graphics: accepted for codegen readiness but lowered to a no-op (there
        // is no graphics backend in the generated C; see `docs/unsupported.md`).
        "plot" | "plot3" => Builtin::Noop,
        // File I/O: stdio-backed runtime helpers (see `docs/unsupported.md`).
        "fopen" => Builtin::Fopen,
        "fread" => Builtin::Fread,
        "fclose" => Builtin::Fclose,
        _ => return None,
    })
}

/// The `libm` symbol for an elementwise builtin, when it lowers to a direct C
/// call. `None` for inline patterns ([`Builtin::Sign`]) and reductions.
pub fn libm_symbol(builtin: Builtin) -> Option<&'static str> {
    match builtin {
        Builtin::Unary(sym) | Builtin::Binary(sym) => Some(sym),
        Builtin::Mod => Some("fmod"),
        Builtin::Sign
        | Builtin::Reduce(_)
        | Builtin::Mean
        | Builtin::Std
        | Builtin::Median
        | Builtin::CumSum
        | Builtin::Diff
        | Builtin::IsEmpty
        | Builtin::Logical
        | Builtin::Var
        | Builtin::LinSpace
        | Builtin::Repmat
        | Builtin::Permute
        | Builtin::Numel
        | Builtin::Length
        | Builtin::Size
        | Builtin::Fill(_)
        | Builtin::Eye
        | Builtin::Reshape
        | Builtin::Inv
        | Builtin::Det
        | Builtin::Norm
        | Builtin::Rand
        | Builtin::StrCmp
        | Builtin::Int32
        | Builtin::Fft
        | Builtin::Eig
        | Builtin::Noop
        | Builtin::Fopen
        | Builtin::Fread
        | Builtin::Fclose
        | Builtin::Sort => None,
        Builtin::MinMax(MinMax::Min) => Some("fmin"),
        Builtin::MinMax(MinMax::Max) => Some("fmax"),
    }
}
