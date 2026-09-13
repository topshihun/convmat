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
            Builtin::Unary(_) | Builtin::Sign | Builtin::Numel | Builtin::Length => n == 1,
            Builtin::Binary(_) => n == 2,
            Builtin::MinMax(_)
            | Builtin::Reduce(_)
            | Builtin::Size
            | Builtin::Fill(_)
            | Builtin::Eye => n == 1 || n == 2,
            Builtin::Reshape => n == 3,
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
        "mod" => Builtin::Binary("fmod"),
        "rem" => Builtin::Binary("remainder"),
        "min" => Builtin::MinMax(MinMax::Min),
        "max" => Builtin::MinMax(MinMax::Max),
        "sum" => Builtin::Reduce(ReduceOp::Sum),
        "prod" => Builtin::Reduce(ReduceOp::Prod),
        "numel" => Builtin::Numel,
        "length" => Builtin::Length,
        "size" => Builtin::Size,
        "zeros" => Builtin::Fill(0.0),
        "ones" => Builtin::Fill(1.0),
        "eye" => Builtin::Eye,
        "reshape" => Builtin::Reshape,
        _ => return None,
    })
}

/// The `libm` symbol for an elementwise builtin, when it lowers to a direct C
/// call. `None` for inline patterns ([`Builtin::Sign`]) and reductions.
pub fn libm_symbol(builtin: Builtin) -> Option<&'static str> {
    match builtin {
        Builtin::Unary(sym) | Builtin::Binary(sym) => Some(sym),
        Builtin::Sign
        | Builtin::Reduce(_)
        | Builtin::Numel
        | Builtin::Length
        | Builtin::Size
        | Builtin::Fill(_)
        | Builtin::Eye
        | Builtin::Reshape => None,
        Builtin::MinMax(MinMax::Min) => Some("fmin"),
        Builtin::MinMax(MinMax::Max) => Some("fmax"),
    }
}
