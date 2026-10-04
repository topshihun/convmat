//! Layer 2: the codegen boundary.
//!
//! Classifies each HIR function as statically lowerable (`Verdict::Static`) or
//! deferred to the runtime (`Verdict::Deferred`). This is a function-level
//! decision, so a program can mix compiled and runtime code.
//!
//! [`dispatch`] is the compile-time dispatch phase built on this boundary: it
//! resolves each binding's [`ValueClass`] (scalar / static matrix / runtime
//! matrix / struct) and each function's [`Route`] into a [`FunctionPlan`] that
//! the pipeline and `hir_to_mlir` consume without re-deriving the choice.
//!
//! The current implementation is a hand-written whitelist of the scalar-double
//! subset (plus statically-shaped array literals and pure numeric built-ins).
//! It is exposed behind [`Classifier`] so a `runmat-static-analysis`-driven
//! classifier (type/shape inference, definite assignment) can replace it
//! without changing callers.

use std::collections::{HashMap, HashSet};

use runmat_hir::{
    BindingId, FunctionId, HirCall, HirCallableRef, HirExpr, HirExprKind, HirFunction, HirPlace,
    HirStmt, HirStmtKind, IndexComponent, IndexKind, IndexResultContext, IndexingSemantics,
    OperatorKind,
};

use crate::builtins::{self, Builtin};

pub mod dispatch;

pub use dispatch::{dispatch, FunctionPlan, Route, ValueClass};

/// A per-function classification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Statically lowerable: type-determined, closed-world, supported subset.
    Static,
    /// Deferred to the runtime with a human-readable reason.
    Deferred { reason: String },
}

/// Maximum tensor rank tracked by the lightweight shape analysis.
pub const MAX_RANK: usize = 4;

/// The shape of a numeric array, in MATLAB column-major logical order
/// (`dims[0]` is rows, `dims[1]` is columns, and so on).
///
/// [`Shape::Static`] is resolved at compile time and lowered to a static
/// `memref<nxf64>` (stack or caller buffer). [`Shape::Dynamic`] is only known at
/// runtime and is deferred to a `memref<?×…×f64>` plus a shape descriptor — that
/// tier is not implemented yet (P7, see `docs/architecture.md` §11.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Compile-time known dimensions.
    Static {
        rank: usize,
        dims: [usize; MAX_RANK],
    },
    /// Runtime-determined dimensions.
    Dynamic,
}

impl Shape {
    /// A static 2-D `rows x cols` shape.
    pub fn matrix(rows: usize, cols: usize) -> Self {
        let mut dims = [1; MAX_RANK];
        dims[0] = rows;
        dims[1] = cols;
        Shape::Static { rank: 2, dims }
    }

    /// A static N-D shape from explicit dimensions (rank = `dims.len()`), or
    /// `None` for an empty or over-rank (`> MAX_RANK`) dimension list.
    pub fn from_dims(dims: &[usize]) -> Option<Self> {
        if dims.is_empty() || dims.len() > MAX_RANK {
            return None;
        }
        let mut out = [1; MAX_RANK];
        out[..dims.len()].copy_from_slice(dims);
        Some(Shape::Static {
            rank: dims.len(),
            dims: out,
        })
    }

    /// Whether this shape is runtime-determined.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Shape::Dynamic)
    }

    /// Number of meaningful dimensions, for a static shape.
    pub fn rank(&self) -> usize {
        match self {
            Shape::Static { rank, .. } => *rank,
            Shape::Dynamic => panic!("dynamic shape has no static rank"),
        }
    }

    /// The meaningful dimensions, for a static shape.
    pub fn dims(&self) -> &[usize] {
        match self {
            Shape::Static { rank, dims } => &dims[..*rank],
            Shape::Dynamic => panic!("dynamic shape has no static dims"),
        }
    }

    /// Total element count (`numel`), for a static shape.
    pub fn numel(&self) -> usize {
        self.dims().iter().product()
    }

    /// Column-major linear offset of the given logical index, for a static shape.
    pub fn linear(&self, index: &[usize]) -> usize {
        let dims = self.dims();
        let mut offset = 0;
        let mut stride = 1;
        for (k, &coord) in index.iter().enumerate() {
            offset += coord * stride;
            stride *= dims[k];
        }
        offset
    }
}

/// The static type of a HIR binding, as inferred by the boundary's lightweight
/// shape analysis (a stand-in for `runmat-static-analysis` in the MVP).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalTy {
    /// A scalar `f64` value.
    Scalar,
    /// A `f64` array, with either a static or dynamic shape.
    Array { shape: Shape },
    /// A `struct` with an ordered field list (name + type). Field types may be
    /// scalar, array, or (recursively) struct.
    Struct { fields: Vec<(String, LocalTy)> },
    /// Not a numeric value / not statically resolvable; deferred to the runtime.
    Dynamic,
}

/// The variadic-argument shape of a function body: which bindings back
/// `varargin`/`varargout`/`nargin`/`nargout`, and how many extra scalar
/// arguments/outputs the body actually uses.
///
/// In the closed-world, fixed-arity specialization (see `docs/architecture.md`
/// §12), `nargin`/`nargout` fold to `named_* + *_count`, and `varargin{k}` /
/// `varargout{k}` (constant `k`) resolve directly to the `k`-th extra argument
/// or output — no cell is materialized.
#[derive(Debug, Clone, Default)]
pub struct Variadics {
    pub varargin_local: Option<BindingId>,
    pub varargin_count: usize,
    pub varargout_local: Option<BindingId>,
    pub varargout_count: usize,
    pub nargin_local: Option<BindingId>,
    pub nargout_local: Option<BindingId>,
    /// Number of named (non-variadic) inputs: `fixed_inputs` minus `varargin`.
    pub named_inputs: usize,
    /// Number of named (non-variadic) outputs: `fixed_outputs` minus `varargout`.
    pub named_outputs: usize,
}

impl Variadics {
    /// Compute the variadic shape of `function` by scanning its body for
    /// constant `varargin{k}` / `varargout{k}` indices.
    pub fn compute(function: &HirFunction) -> Self {
        let varargin_local = function.abi.varargin;
        let varargout_local = function.abi.varargout;

        let mut varargin_count = 0;
        let mut varargout_count = 0;
        scan_block(
            &function.body.statements,
            varargin_local,
            varargout_local,
            &mut varargin_count,
            &mut varargout_count,
        );

        Variadics {
            varargin_local,
            varargin_count,
            varargout_local,
            varargout_count,
            nargin_local: function.abi.implicit_nargin,
            nargout_local: function.abi.implicit_nargout,
            named_inputs: function
                .abi
                .fixed_inputs
                .len()
                .saturating_sub(usize::from(function.abi.varargin.is_some())),
            named_outputs: function
                .abi
                .fixed_outputs
                .len()
                .saturating_sub(usize::from(function.abi.varargout.is_some())),
        }
    }
}

/// Recursively scan a statement list for `varargin{k}` reads and `varargout{k}`
/// writes, updating the maximum constant index of each.
fn scan_block(
    stmts: &[HirStmt],
    varargin_local: Option<BindingId>,
    varargout_local: Option<BindingId>,
    varargin_count: &mut usize,
    varargout_count: &mut usize,
) {
    for stmt in stmts {
        match &stmt.kind {
            HirStmtKind::Assign(place, value, _) => {
                if let Some(k) = varargout_index(place, varargout_local) {
                    *varargout_count = (*varargout_count).max(k);
                }
                scan_varargin(value, varargin_local, varargin_count);
            }
            HirStmtKind::ExprStmt(value, _) => {
                scan_varargin(value, varargin_local, varargin_count);
            }
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                scan_varargin(cond, varargin_local, varargin_count);
                scan_block(
                    &then_body.statements,
                    varargin_local,
                    varargout_local,
                    varargin_count,
                    varargout_count,
                );
                for (cond, block) in elseif_blocks {
                    scan_varargin(cond, varargin_local, varargin_count);
                    scan_block(
                        &block.statements,
                        varargin_local,
                        varargout_local,
                        varargin_count,
                        varargout_count,
                    );
                }
                if let Some(block) = else_body {
                    scan_block(
                        &block.statements,
                        varargin_local,
                        varargout_local,
                        varargin_count,
                        varargout_count,
                    );
                }
            }
            HirStmtKind::While { cond, body } => {
                scan_varargin(cond, varargin_local, varargin_count);
                scan_block(
                    &body.statements,
                    varargin_local,
                    varargout_local,
                    varargin_count,
                    varargout_count,
                );
            }
            HirStmtKind::For { range, body, .. } => {
                scan_varargin(range, varargin_local, varargin_count);
                scan_block(
                    &body.statements,
                    varargin_local,
                    varargout_local,
                    varargin_count,
                    varargout_count,
                );
            }
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => {
                scan_varargin(expr, varargin_local, varargin_count);
                for (case, block) in cases {
                    scan_varargin(case, varargin_local, varargin_count);
                    scan_block(
                        &block.statements,
                        varargin_local,
                        varargout_local,
                        varargin_count,
                        varargout_count,
                    );
                }
                if let Some(block) = otherwise {
                    scan_block(
                        &block.statements,
                        varargin_local,
                        varargout_local,
                        varargin_count,
                        varargout_count,
                    );
                }
            }
            _ => {}
        }
    }
}

/// Accumulate the largest constant `varargin{k}` index referenced anywhere in
/// `expr`.
fn scan_varargin(expr: &HirExpr, varargin_local: Option<BindingId>, count: &mut usize) {
    let Some(varargin_local) = varargin_local else {
        return;
    };
    match &expr.kind {
        HirExprKind::Index(base, indexing) => {
            if indexing.kind == IndexKind::Brace {
                if let HirExprKind::Binding(id) = base.kind {
                    if id == varargin_local {
                        if let Some(k) = brace_index(indexing) {
                            *count = (*count).max(k);
                        }
                    }
                }
            }
            scan_varargin(base, Some(varargin_local), count);
            for component in &indexing.components {
                if let IndexComponent::Expr(e) = component {
                    scan_varargin(e, Some(varargin_local), count);
                }
            }
        }
        HirExprKind::Unary(_, operand) => scan_varargin(operand, Some(varargin_local), count),
        HirExprKind::Binary(lhs, _, rhs) => {
            scan_varargin(lhs, Some(varargin_local), count);
            scan_varargin(rhs, Some(varargin_local), count);
        }
        HirExprKind::Range(start, step, end) => {
            scan_varargin(start, Some(varargin_local), count);
            if let Some(step) = step {
                scan_varargin(step, Some(varargin_local), count);
            }
            scan_varargin(end, Some(varargin_local), count);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for element in row {
                    scan_varargin(element, Some(varargin_local), count);
                }
            }
        }
        HirExprKind::Call(call) => {
            for arg in &call.args {
                scan_varargin(arg, Some(varargin_local), count);
            }
        }
        _ => {}
    }
}

/// The 1-based constant index of a single-component `{}` (cell) index, if known.
pub(crate) fn brace_index(indexing: &IndexingSemantics) -> Option<usize> {
    match indexing.components.as_slice() {
        [IndexComponent::Expr(expr)] => constant_expr_value(expr),
        _ => None,
    }
}

/// The constant `usize` value of an expression, if it is a numeric/integer
/// literal.
fn constant_expr_value(expr: &HirExpr) -> Option<usize> {
    match &expr.kind {
        HirExprKind::Number(text) => text.trim().parse::<usize>().ok(),
        HirExprKind::IntegerLiteral(literal) => Some(literal.bits() as usize),
        _ => None,
    }
}

/// Strip one level of single/double quotes from a string literal (`'a'` -> `a`).
pub(crate) fn unquote_str(s: &str) -> &str {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && matches!(bytes[0], b'\'' | b'"') && bytes[bytes.len() - 1] == bytes[0] {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Whether an index component is the colon operator `:`.
///
/// runmat represents `a(:)` (identifier base) as
/// `IndexComponent::Expr(HirExprKind::Colon)` and `expr(:)` (expression base)
/// as `IndexComponent::Colon`; both must be treated uniformly.
pub(crate) fn component_is_colon(component: &IndexComponent) -> bool {
    match component {
        IndexComponent::Colon => true,
        IndexComponent::Expr(expr) => matches!(expr.kind, HirExprKind::Colon),
        _ => false,
    }
}

/// The relative offset of an `end` index component (`end`, `end+k`, `end-k`),
/// or `None` if the component is not an `end` expression.
pub(crate) fn component_end_offset(component: &IndexComponent) -> Option<isize> {
    match component {
        IndexComponent::End { offset, .. } => Some(*offset),
        IndexComponent::Expr(expr) => expr_end_offset(expr),
        _ => None,
    }
}

/// The relative offset of an `end` expression (`end`, `end+k`, `end-k`).
fn expr_end_offset(expr: &HirExpr) -> Option<isize> {
    match &expr.kind {
        HirExprKind::End => Some(0),
        HirExprKind::Binary(lhs, op, rhs) => match op {
            OperatorKind::Add => {
                if matches!(lhs.kind, HirExprKind::End) {
                    int_literal(rhs)
                } else if matches!(rhs.kind, HirExprKind::End) {
                    int_literal(lhs)
                } else {
                    None
                }
            }
            OperatorKind::Subtract if matches!(lhs.kind, HirExprKind::End) => {
                int_literal(rhs).and_then(|offset| offset.checked_neg())
            }
            _ => None,
        },
        _ => None,
    }
}

/// The integer value of a numeric/integer literal expression, as an `isize`.
fn int_literal(expr: &HirExpr) -> Option<isize> {
    match &expr.kind {
        HirExprKind::Number(text) => text.trim().parse::<isize>().ok(),
        HirExprKind::IntegerLiteral(literal) => Some(literal.bits() as isize),
        _ => None,
    }
}

/// The 1-based constant index of a `varargout{k}` write target, if `place` is
/// `varargout{k}`.
pub(crate) fn varargout_index(
    place: &HirPlace,
    varargout_local: Option<BindingId>,
) -> Option<usize> {
    let varargout_local = varargout_local?;
    match place {
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            if indexing.kind == IndexKind::Brace {
                if let HirExprKind::Binding(id) = base.kind {
                    if id == varargout_local {
                        return brace_index(indexing);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Operators supported by the current scalar lowering.
fn supported_unary(op: &OperatorKind) -> bool {
    matches!(
        op,
        OperatorKind::UnaryPlus
            | OperatorKind::UnaryMinus
            | OperatorKind::Not
            | OperatorKind::Transpose
            | OperatorKind::ConjugateTranspose
    )
}

/// Binary operators supported by the current scalar lowering.
fn supported_binary(op: &OperatorKind) -> bool {
    matches!(
        op,
        OperatorKind::Add
            | OperatorKind::Subtract
            | OperatorKind::MatrixMultiply
            | OperatorKind::ElementwiseMultiply
            | OperatorKind::MatrixPower
            | OperatorKind::ElementwisePower
            | OperatorKind::Mrdivide
            | OperatorKind::ElementwiseDivide
            | OperatorKind::Mldivide
            | OperatorKind::ElementwiseLeftDivide
            | OperatorKind::Equal
            | OperatorKind::NotEqual
            | OperatorKind::Less
            | OperatorKind::LessEqual
            | OperatorKind::Greater
            | OperatorKind::GreaterEqual
            | OperatorKind::ShortCircuitAnd
            | OperatorKind::ShortCircuitOr
            | OperatorKind::ElementwiseAnd
            | OperatorKind::ElementwiseOr
    )
}

/// Resolve a callable to its source-level name, when one exists.
pub fn call_name(callee: &HirCallableRef) -> Option<String> {
    callee
        .identity()
        .and_then(|identity| identity.display_name())
}

/// Whether a call targets a user-defined function (a closed-world, same-program
/// call), as opposed to a builtin or a dynamic/unresolved call.
fn is_user_call(call: &HirCall) -> bool {
    matches!(
        call.callee,
        HirCallableRef::Function(_) | HirCallableRef::ExternalFunction { .. }
    )
}

fn expr_reason(expr: &HirExpr) -> Option<String> {
    match &expr.kind {
        HirExprKind::Binding(_) => None,
        HirExprKind::Number(_) | HirExprKind::IntegerLiteral(_) => None,
        HirExprKind::End | HirExprKind::Colon => None,
        HirExprKind::Constant(symbol) => match symbol.0.as_str() {
            "true" | "false" | "Inf" | "Infinity" | "NaN" => None,
            other => Some(format!("unsupported constant `{other}`")),
        },
        HirExprKind::Unary(op, operand) => {
            if supported_unary(op) {
                expr_reason(operand)
            } else {
                Some(format!("unsupported unary operator {op:?}"))
            }
        }
        HirExprKind::Binary(lhs, op, rhs) => {
            expr_reason(lhs).or_else(|| expr_reason(rhs)).or_else(|| {
                if supported_binary(op) {
                    None
                } else {
                    Some(format!("unsupported operator {op:?}"))
                }
            })
        }
        HirExprKind::Tensor(rows) => rows.iter().flatten().find_map(expr_reason),
        HirExprKind::Cell(_) => Some("cell array literals are not supported yet".to_string()),
        HirExprKind::Range(start, step, end) => expr_reason(start)
            .or_else(|| step.as_deref().and_then(expr_reason))
            .or_else(|| expr_reason(end)),
        HirExprKind::Index(base, indexing) => {
            if indexing.result_context == IndexResultContext::FunctionArgumentExpansion {
                return Some("argument expansion (`{:}`/`varargin`) is not supported".to_string());
            }
            expr_reason(base).or_else(|| {
                indexing
                    .components
                    .iter()
                    .find_map(|component| match component {
                        IndexComponent::Expr(expr) => expr_reason(expr),
                        IndexComponent::Logical(expr) => expr_reason(expr)
                            .or(Some("logical indexing is not supported yet".to_string())),
                        IndexComponent::Colon | IndexComponent::End { .. } => None,
                    })
            })
        }
        HirExprKind::Member(base, _) => expr_reason(base),
        HirExprKind::StructLiteral(fields) => {
            fields.iter().find_map(|(_, value)| expr_reason(value))
        }
        HirExprKind::Call(call) => call_reason(call),
        other => Some(format!("unsupported expression {other:?}")),
    }
}

fn call_reason(call: &HirCall) -> Option<String> {
    // A user-defined function call (same file, closed world). The scalar-ABI
    // subset is validated by the lowerer; non-scalar calls defer.
    if is_user_call(call) {
        return call.args.iter().find_map(expr_reason);
    }
    let Some(name) = call_name(&call.callee) else {
        return Some("dynamic or non-static function call is not supported".to_string());
    };
    // `struct('a', 1, ...)` is a special constructor: alternating string field
    // names and values.
    if name == "struct" {
        return struct_reason(call);
    }
    let Some(builtin) = builtins::lookup(&name) else {
        return Some(format!(
            "unsupported builtin `{name}` (deferred to runtime)"
        ));
    };
    if !builtin.valid_arity(call.args.len()) {
        return Some(format!(
            "builtin `{name}` called with {} argument(s)",
            call.args.len()
        ));
    }
    call.args.iter().find_map(expr_reason)
}

/// Whether a `struct(...)` construction is lowerable: alternating string field
/// names and lowerable values.
fn struct_reason(call: &HirCall) -> Option<String> {
    let mut args = call.args.iter();
    while let Some(name_arg) = args.next() {
        if !matches!(name_arg.kind, HirExprKind::String(_)) {
            return Some("struct field names must be string literals".to_string());
        }
        let Some(value_arg) = args.next() else {
            return Some("struct requires a value for every field".to_string());
        };
        if let Some(reason) = expr_reason(value_arg) {
            return Some(reason);
        }
    }
    None
}

/// Whether a statement itself (excluding its nested sub-statements) is
/// lowerable.
fn stmt_reason(stmt: &HirStmt, varargout_local: Option<BindingId>) -> Option<String> {
    match &stmt.kind {
        HirStmtKind::ExprStmt(expr, _) => expr_reason(expr),
        HirStmtKind::Assign(place, value, _) => {
            // An anonymous-function definition `f = @(...)` is compile-time only;
            // its validity (single assignment, non-escaping) is checked by
            // [`analyze_handles`]. The `@(...)` literal itself is not a value we
            // lower here.
            if matches!(place, HirPlace::Binding(_))
                && matches!(value.kind, HirExprKind::AnonymousFunction(_))
            {
                return None;
            }
            // A `varargout{k}` target is a cell element write; a plain `Binding`
            // target is the norm; a `Member` target is a struct field write.
            if varargout_index(place, varargout_local).is_some()
                || matches!(place, HirPlace::Binding(_))
            {
                expr_reason(value)
            } else if let HirPlace::Member(base, _) = place {
                expr_reason(base).or_else(|| expr_reason(value))
            } else {
                Some(format!("non-local assignment target {place:?}"))
            }
        }
        HirStmtKind::If { cond, .. } => expr_reason(cond),
        HirStmtKind::While { cond, .. } => expr_reason(cond),
        HirStmtKind::For { range, .. } => expr_reason(range),
        HirStmtKind::Switch { expr, cases, .. } => {
            expr_reason(expr).or_else(|| cases.iter().find_map(|(case, _)| expr_reason(case)))
        }
        HirStmtKind::MultiAssign(..) => {
            Some("multi-assignment (`[a, b] = f()`) is not supported".to_string())
        }
        HirStmtKind::TryCatch { .. } => Some("try/catch is not supported".to_string()),
        HirStmtKind::Global(_) | HirStmtKind::Persistent(_) => None,
        HirStmtKind::Break | HirStmtKind::Continue => None,
        HirStmtKind::Return | HirStmtKind::Import(_) => None,
        other => Some(format!("unsupported statement {other:?}")),
    }
}

/// Map from a handle binding to the anonymous function it holds.
pub type HandleTargets = HashMap<BindingId, FunctionId>;

/// Resolve the non-escaping anonymous-function handles defined in `function`.
///
/// A binding `f` is a supported handle iff it is assigned exactly once, at the
/// top level of the function body, with an `@(...)` literal, and every use of
/// `f` is a direct call `f(args)` (paren indexing). Anything else — the handle
/// is returned, passed on, copied, or otherwise examined as a value — needs the
/// dynamic closure tier (`docs/architecture.md` §11.2) and is deferred.
///
/// The captured bindings (the environment) are *not* resolved here: the caller
/// of this function has the full assembly and resolves them from the target
/// `HirFunction` when it needs the capture order.
pub fn analyze_handles(function: &HirFunction) -> Result<HandleTargets, String> {
    let mut targets: HandleTargets = HashMap::new();
    let mut assignments: HashMap<BindingId, usize> = HashMap::new();

    for stmt in &function.body.statements {
        if let HirStmtKind::Assign(HirPlace::Binding(target), value, _) = &stmt.kind {
            if let HirExprKind::AnonymousFunction(id) = value.kind {
                *assignments.entry(*target).or_insert(0) += 1;
                targets.insert(*target, id);
            }
        }
    }

    if nested_handle_definition(&function.body.statements) {
        return Err(
            "anonymous function handle defined inside control flow is not supported (deferred)"
                .to_string(),
        );
    }

    if targets.is_empty() {
        return Ok(targets);
    }

    for (target, count) in &assignments {
        if *count > 1 {
            return Err(format!(
                "function handle {} is assigned more than once (deferred)",
                target.0
            ));
        }
        if function.abi.fixed_outputs.contains(target) {
            return Err(format!(
                "function handle {} escapes as a return value (needs the dynamic closure tier)",
                target.0
            ));
        }
    }

    check_handle_uses_block(&function.body.statements, &targets)?;
    Ok(targets)
}

/// Whether a handle is defined anywhere in a nested block of `stmts` (that is,
/// anywhere below the top level of the function body).
fn nested_handle_definition(stmts: &[HirStmt]) -> bool {
    stmts.iter().any(|stmt| match &stmt.kind {
        HirStmtKind::If {
            then_body,
            elseif_blocks,
            else_body,
            ..
        } => {
            block_defines_handle(&then_body.statements)
                || elseif_blocks
                    .iter()
                    .any(|(_, block)| block_defines_handle(&block.statements))
                || else_body
                    .as_ref()
                    .is_some_and(|block| block_defines_handle(&block.statements))
        }
        HirStmtKind::While { body, .. } | HirStmtKind::For { body, .. } => {
            block_defines_handle(&body.statements)
        }
        HirStmtKind::Switch {
            cases, otherwise, ..
        } => {
            cases
                .iter()
                .any(|(_, block)| block_defines_handle(&block.statements))
                || otherwise
                    .as_ref()
                    .is_some_and(|block| block_defines_handle(&block.statements))
        }
        _ => false,
    })
}

/// Whether any statement at any depth defines an anonymous-function handle.
fn block_defines_handle(stmts: &[HirStmt]) -> bool {
    stmts.iter().any(|stmt| match &stmt.kind {
        HirStmtKind::Assign(HirPlace::Binding(_), value, _) => {
            matches!(value.kind, HirExprKind::AnonymousFunction(_))
        }
        _ => nested_handle_definition(std::slice::from_ref(stmt)),
    })
}

/// Verify that no handle binding escapes through the statements in `stmts`.
fn check_handle_uses_block(stmts: &[HirStmt], handles: &HandleTargets) -> Result<(), String> {
    for stmt in stmts {
        match &stmt.kind {
            HirStmtKind::Assign(place, value, _) => {
                // The defining assignment `f = @(...)` reads the handle binding
                // as a storage location, not as a value; skip it (the target is
                // checked for reassignment/escape separately).
                if let HirPlace::Binding(target) = place {
                    if handles.contains_key(target)
                        && matches!(value.kind, HirExprKind::AnonymousFunction(_))
                    {
                        continue;
                    }
                }
                check_handle_uses_place(place, handles)?;
                check_handle_uses_expr(value, handles)?;
            }
            HirStmtKind::ExprStmt(expr, _) => check_handle_uses_expr(expr, handles)?,
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                check_handle_uses_expr(cond, handles)?;
                check_handle_uses_block(&then_body.statements, handles)?;
                for (cond, block) in elseif_blocks {
                    check_handle_uses_expr(cond, handles)?;
                    check_handle_uses_block(&block.statements, handles)?;
                }
                if let Some(block) = else_body {
                    check_handle_uses_block(&block.statements, handles)?;
                }
            }
            HirStmtKind::While { cond, body } => {
                check_handle_uses_expr(cond, handles)?;
                check_handle_uses_block(&body.statements, handles)?;
            }
            HirStmtKind::For { range, body, .. } => {
                check_handle_uses_expr(range, handles)?;
                check_handle_uses_block(&body.statements, handles)?;
            }
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
            } => {
                check_handle_uses_expr(expr, handles)?;
                for (case, block) in cases {
                    check_handle_uses_expr(case, handles)?;
                    check_handle_uses_block(&block.statements, handles)?;
                }
                if let Some(block) = otherwise {
                    check_handle_uses_block(&block.statements, handles)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Verify that no handle binding is read through an assignment target.
fn check_handle_uses_place(place: &HirPlace, handles: &HandleTargets) -> Result<(), String> {
    match place {
        HirPlace::Binding(_) => Ok(()),
        HirPlace::Member(base, _) => check_handle_uses_expr(base, handles),
        HirPlace::MemberDynamic(base, key) => {
            check_handle_uses_expr(base, handles)?;
            check_handle_uses_expr(key, handles)
        }
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            check_handle_uses_expr(base, handles)?;
            check_handle_uses_components(indexing, handles)
        }
    }
}

/// Verify that no handle binding escapes through the expression tree `expr`.
fn check_handle_uses_expr(expr: &HirExpr, handles: &HandleTargets) -> Result<(), String> {
    match &expr.kind {
        HirExprKind::Binding(id) => {
            if handles.contains_key(id) {
                return Err(format!(
                    "function handle {} escapes (used as a value, not called) \
                     (needs the dynamic closure tier)",
                    id.0
                ));
            }
            Ok(())
        }
        HirExprKind::Number(_)
        | HirExprKind::IntegerLiteral(_)
        | HirExprKind::String(_)
        | HirExprKind::Constant(_)
        | HirExprKind::Colon
        | HirExprKind::End
        | HirExprKind::FunctionHandle(_)
        | HirExprKind::AnonymousFunction(_)
        | HirExprKind::MetaClass(_)
        | HirExprKind::WorkspaceFirstStaticProperty { .. }
        | HirExprKind::CommandCall(_) => Ok(()),
        HirExprKind::Unary(_, operand)
        | HirExprKind::Member(operand, _)
        | HirExprKind::Await(operand)
        | HirExprKind::Spawn(operand) => check_handle_uses_expr(operand, handles),
        HirExprKind::Binary(lhs, _, rhs) => {
            check_handle_uses_expr(lhs, handles)?;
            check_handle_uses_expr(rhs, handles)
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for element in row {
                    check_handle_uses_expr(element, handles)?;
                }
            }
            Ok(())
        }
        HirExprKind::StructLiteral(fields) | HirExprKind::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                check_handle_uses_expr(value, handles)?;
            }
            Ok(())
        }
        HirExprKind::Range(start, step, end) => {
            check_handle_uses_expr(start, handles)?;
            if let Some(step) = step {
                check_handle_uses_expr(step, handles)?;
            }
            check_handle_uses_expr(end, handles)
        }
        HirExprKind::Index(base, indexing) => {
            // A direct call `f(args)` is the only legal use of a handle value.
            if let HirExprKind::Binding(id) = &base.kind {
                if handles.contains_key(id) {
                    if indexing.kind != IndexKind::Paren {
                        return Err(format!(
                            "function handle {} must be called with `()` (deferred)",
                            id.0
                        ));
                    }
                    for component in &indexing.components {
                        let IndexComponent::Expr(arg) = component else {
                            return Err(format!(
                                "function handle {} call arguments must be values (deferred)",
                                id.0
                            ));
                        };
                        check_handle_uses_expr(arg, handles)?;
                    }
                    return Ok(());
                }
            }
            check_handle_uses_expr(base, handles)?;
            check_handle_uses_components(indexing, handles)
        }
        HirExprKind::MemberDynamic(base, key) => {
            check_handle_uses_expr(base, handles)?;
            check_handle_uses_expr(key, handles)
        }
        HirExprKind::Call(call) => {
            if let HirCallableRef::DynamicExpr(callee) = &call.callee {
                check_handle_uses_expr(callee, handles)?;
            }
            for arg in &call.args {
                check_handle_uses_expr(arg, handles)?;
            }
            Ok(())
        }
    }
}

/// Verify that no handle binding escapes through an index's components.
fn check_handle_uses_components(
    indexing: &IndexingSemantics,
    handles: &HandleTargets,
) -> Result<(), String> {
    for component in &indexing.components {
        if let IndexComponent::Expr(expr) | IndexComponent::Logical(expr) = component {
            check_handle_uses_expr(expr, handles)?;
        }
    }
    Ok(())
}

/// The codegen boundary: decide whether a function is statically lowerable.
///
/// This is the seam where `runmat-static-analysis` (type/shape inference,
/// definite assignment) will plug in to replace the operator whitelist and
/// cover the architecture's type-determined / shape-controlled / closed-world
/// criteria.
pub trait Classifier {
    fn classify(&self, function: &HirFunction) -> Verdict;
}

/// The current classifier: a hand-written whitelist of the scalar-double
/// subset with structured control flow (`if`/`while`/`for`/`switch`), statically
/// shaped array literals, and the pure numeric built-ins in [`builtins`].
pub struct WhitelistClassifier;

impl Classifier for WhitelistClassifier {
    fn classify(&self, function: &HirFunction) -> Verdict {
        let variadics = Variadics::compute(function);

        // Anonymous-function handles must be statically resolvable and must not
        // escape; anything else needs the dynamic closure tier.
        if let Err(reason) = analyze_handles(function) {
            return Verdict::Deferred { reason };
        }

        let mut reason = None;
        for_each_stmt(&function.body.statements, &mut |stmt| {
            if reason.is_none() {
                reason = stmt_reason(stmt, variadics.varargout_local);
            }
        });
        if let Some(reason) = reason {
            return Verdict::Deferred { reason };
        }

        // A dynamic-shape array local is realized by the runtime matrix tier:
        // parameters/outputs use the pointer+length ABI, intermediates get a
        // runtime-length heap buffer. A truly unknown type (`LocalTy::Dynamic`)
        // still crosses the boundary, as does a struct with a dynamic field.
        let locals = infer_locals(function);
        for (local, ty) in &locals {
            let is_dynamic = match ty {
                LocalTy::Dynamic => true,
                LocalTy::Array { .. } => false,
                // Struct fields must be fully static; a dynamic field defers the
                // whole struct.
                LocalTy::Struct { fields } => fields
                    .iter()
                    .any(|(_, field_ty)| matches!(field_ty, LocalTy::Dynamic)),
                LocalTy::Scalar => false,
            };
            if is_dynamic {
                return Verdict::Deferred {
                    reason: format!("unresolved shape for local {}", local.0),
                };
            }
        }

        Verdict::Static
    }
}

/// Classify a HIR function with the default ([`WhitelistClassifier`]) classifier.
pub fn classify(function: &HirFunction) -> Verdict {
    WhitelistClassifier.classify(function)
}

/// Recursively visit every statement in `stmts` (descending into nested
/// control-flow bodies).
fn for_each_stmt<'a>(stmts: &'a [HirStmt], visit: &mut dyn FnMut(&'a HirStmt)) {
    for stmt in stmts {
        visit(stmt);
        match &stmt.kind {
            HirStmtKind::If {
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                for_each_stmt(&then_body.statements, visit);
                for (_, block) in elseif_blocks {
                    for_each_stmt(&block.statements, visit);
                }
                if let Some(block) = else_body {
                    for_each_stmt(&block.statements, visit);
                }
            }
            HirStmtKind::While { body, .. } | HirStmtKind::For { body, .. } => {
                for_each_stmt(&body.statements, visit);
            }
            HirStmtKind::Switch {
                cases, otherwise, ..
            } => {
                for (_, block) in cases {
                    for_each_stmt(&block.statements, visit);
                }
                if let Some(block) = otherwise {
                    for_each_stmt(&block.statements, visit);
                }
            }
            HirStmtKind::TryCatch {
                try_body,
                catch_body,
                ..
            } => {
                for_each_stmt(&try_body.statements, visit);
                for_each_stmt(&catch_body.statements, visit);
            }
            _ => {}
        }
    }
}

// --- Lightweight local type / shape inference ---------------------------------

/// Infer the static type of every local in `function`.
///
/// Parameters default to [`LocalTy::Scalar`], then are promoted by use-site
/// inference: a parameter read/written as a struct becomes
/// [`LocalTy::Struct`]; a parameter used in an array context (`A(i)`, `sum(A)`,
/// `reshape(A, …)`, …) becomes [`LocalTy::Array`] with a [`Shape::Dynamic`]
/// shape (and so is deferred to the runtime tier). Array shapes are otherwise
/// only known for tensor literals; elementwise built-ins preserve their
/// argument's shape; reductions produce a scalar. Anything else resolves to
/// [`LocalTy::Dynamic`].
pub fn infer_locals(function: &HirFunction) -> HashMap<BindingId, LocalTy> {
    let variadics = Variadics::compute(function);
    let mut tys = HashMap::new();

    for binding in &function.abi.fixed_inputs {
        tys.insert(*binding, LocalTy::Scalar);
    }

    // Promote untyped parameters to structs based on their field-access usage,
    // so `function y = f(s); y = s.a + s.b; end` compiles without an explicit
    // type annotation (mirroring MATLAB Coder's use-site struct inference).
    infer_struct_params(function, &mut tys);

    // Promote untyped parameters to dynamic-shape arrays based on their array
    // usage (`A(i)`, `sum(A)`, …). These defer to the runtime tier rather than
    // being silently treated as scalars (see docs/runtime.md §8 step 1).
    infer_array_params(function, &mut tys);

    // Anonymous-function handles are compile-time-only; their bindings are not
    // numeric locals, and their call expressions are typed directly by
    // `expr_ty` against this map.
    let handles = analyze_handles(function).unwrap_or_default();
    infer_block(&function.body.statements, &variadics, &handles, &mut tys);

    tys
}

/// Infer struct types for fixed-input parameters that still carry the default
/// `Scalar` type, by scanning field reads (`s.a`) and writes (`s.a = ...`).
/// Fields are collected in first-seen order; every field is `Scalar` for now
/// (struct fields are scalar-only in the current lowering).
fn infer_struct_params(function: &HirFunction, tys: &mut HashMap<BindingId, LocalTy>) {
    let params: HashSet<BindingId> = function
        .abi
        .fixed_inputs
        .iter()
        .copied()
        .filter(|id| matches!(tys.get(id), Some(LocalTy::Scalar)))
        .collect();
    if params.is_empty() {
        return;
    }

    let mut fields: HashMap<BindingId, Vec<(String, LocalTy)>> = HashMap::new();
    for id in &params {
        fields.insert(*id, Vec::new());
    }
    collect_stmt_fields(&function.body.statements, &params, &mut fields);

    for (id, field_list) in fields {
        if field_list.is_empty() {
            continue;
        }
        tys.insert(id, LocalTy::Struct { fields: field_list });
    }
}

/// Infer array-typed fixed-input parameters from their array usage, so
/// `function y = f(A); y = sum(A); end` (or `A(i)`, `reshape(A, …)`) is
/// classified as a dynamic-shape array and deferred to the runtime tier instead
/// of being silently treated as a scalar.
fn infer_array_params(function: &HirFunction, tys: &mut HashMap<BindingId, LocalTy>) {
    let params: HashSet<BindingId> = function
        .abi
        .fixed_inputs
        .iter()
        .copied()
        .filter(|id| matches!(tys.get(id), Some(LocalTy::Scalar)))
        .collect();
    if params.is_empty() {
        return;
    }

    let mut array_used: HashSet<BindingId> = HashSet::new();
    scan_array_stmt(&function.body.statements, &params, &mut array_used);

    for id in array_used {
        tys.insert(
            id,
            LocalTy::Array {
                shape: Shape::Dynamic,
            },
        );
    }
}

/// Whether `name` called with `nargs` arguments treats its first argument as an
/// array (a shape-unambiguous array usage). `min`/`max` are array reductions
/// only in their single-argument form; the two-argument form is elementwise.
fn array_arg_builtin(name: &str, nargs: usize) -> bool {
    match name {
        "sum" | "prod" | "reshape" | "size" | "numel" | "length" => true,
        "min" | "max" => nargs == 1,
        "mean" | "std" | "median" | "cumsum" | "diff" => true,
        "var" | "isempty" | "logical" | "repmat" | "permute" => true,
        _ => false,
    }
}

/// The dynamic-shape array parameters whose size is queried with `size(...)`,
/// which need the `(rows, cols)` shape-descriptor ABI rather than the lean
/// `(data, n)` one (a bare element count cannot distinguish a row from a column
/// vector). Returns the empty set for the common case.
pub(crate) fn shape_descriptor_params(
    function: &HirFunction,
    tys: &HashMap<BindingId, LocalTy>,
) -> HashSet<BindingId> {
    let params: HashSet<BindingId> = function
        .abi
        .fixed_inputs
        .iter()
        .copied()
        .filter(|id| matches!(tys.get(id), Some(LocalTy::Array { shape }) if shape.is_dynamic()))
        .collect();
    let mut used = HashSet::new();
    if !params.is_empty() {
        scan_shape_stmt(&function.body.statements, &params, &mut used);
    }
    used
}

/// Scan statements for `size(param, ...)` calls on parameters under inference.
fn scan_shape_stmt(stmts: &[HirStmt], params: &HashSet<BindingId>, used: &mut HashSet<BindingId>) {
    for stmt in stmts {
        match &stmt.kind {
            HirStmtKind::ExprStmt(expr, _) => scan_shape_expr(expr, params, used),
            HirStmtKind::Assign(place, value, _) => {
                scan_shape_place(place, params, used);
                scan_shape_expr(value, params, used);
            }
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                scan_shape_expr(cond, params, used);
                scan_shape_stmt(&then_body.statements, params, used);
                for (cond, block) in elseif_blocks {
                    scan_shape_expr(cond, params, used);
                    scan_shape_stmt(&block.statements, params, used);
                }
                if let Some(block) = else_body {
                    scan_shape_stmt(&block.statements, params, used);
                }
            }
            HirStmtKind::While { cond, body } => {
                scan_shape_expr(cond, params, used);
                scan_shape_stmt(&body.statements, params, used);
            }
            HirStmtKind::For { range, body, .. } => {
                scan_shape_expr(range, params, used);
                scan_shape_stmt(&body.statements, params, used);
            }
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => {
                scan_shape_expr(expr, params, used);
                for (case, block) in cases {
                    scan_shape_expr(case, params, used);
                    scan_shape_stmt(&block.statements, params, used);
                }
                if let Some(block) = otherwise {
                    scan_shape_stmt(&block.statements, params, used);
                }
            }
            _ => {}
        }
    }
}

/// Scan an assignment target for `size`-relevant subexpressions.
fn scan_shape_place(place: &HirPlace, params: &HashSet<BindingId>, used: &mut HashSet<BindingId>) {
    match place {
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            scan_shape_expr(base, params, used);
            for component in &indexing.components {
                if let IndexComponent::Expr(e) = component {
                    scan_shape_expr(e, params, used);
                }
            }
        }
        HirPlace::Member(base, _) => scan_shape_expr(base, params, used),
        HirPlace::MemberDynamic(base, expr) => {
            scan_shape_expr(base, params, used);
            scan_shape_expr(expr, params, used);
        }
        HirPlace::Binding(_) => {}
    }
}

/// Scan an expression for `size(param, ...)` calls on parameters under
/// inference.
fn scan_shape_expr(expr: &HirExpr, params: &HashSet<BindingId>, used: &mut HashSet<BindingId>) {
    match &expr.kind {
        HirExprKind::Call(call) => {
            if call_name(&call.callee).as_deref() == Some("size") {
                if let Some(HirExpr {
                    kind: HirExprKind::Binding(id),
                    ..
                }) = call.args.first()
                {
                    if params.contains(id) {
                        used.insert(*id);
                    }
                }
            }
            for arg in &call.args {
                scan_shape_expr(arg, params, used);
            }
        }
        HirExprKind::Index(base, indexing) => {
            scan_shape_expr(base, params, used);
            for component in &indexing.components {
                if let IndexComponent::Expr(e) = component {
                    scan_shape_expr(e, params, used);
                }
            }
        }
        HirExprKind::Unary(_, operand) => scan_shape_expr(operand, params, used),
        HirExprKind::Binary(lhs, _, rhs) => {
            scan_shape_expr(lhs, params, used);
            scan_shape_expr(rhs, params, used);
        }
        HirExprKind::Range(start, step, end) => {
            scan_shape_expr(start, params, used);
            if let Some(step) = step {
                scan_shape_expr(step, params, used);
            }
            scan_shape_expr(end, params, used);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for element in row {
                    scan_shape_expr(element, params, used);
                }
            }
        }
        HirExprKind::Member(base, _) => scan_shape_expr(base, params, used),
        HirExprKind::MemberDynamic(base, expr) => {
            scan_shape_expr(base, params, used);
            scan_shape_expr(expr, params, used);
        }
        HirExprKind::StructLiteral(pairs) => {
            for (_, value) in pairs {
                scan_shape_expr(value, params, used);
            }
        }
        HirExprKind::ObjectLiteral { fields: pairs, .. } => {
            for (_, value) in pairs {
                scan_shape_expr(value, params, used);
            }
        }
        _ => {}
    }
}

/// Record `expr` (a binding under inference) as array-used.
fn mark_array_param(expr: &HirExpr, params: &HashSet<BindingId>, used: &mut HashSet<BindingId>) {
    if let HirExprKind::Binding(id) = expr.kind {
        if params.contains(&id) {
            used.insert(id);
        }
    }
}

/// Scan statements for array usages of parameters under inference.
fn scan_array_stmt(stmts: &[HirStmt], params: &HashSet<BindingId>, used: &mut HashSet<BindingId>) {
    for stmt in stmts {
        match &stmt.kind {
            HirStmtKind::ExprStmt(expr, _) => scan_array_expr(expr, params, used),
            HirStmtKind::Assign(place, value, _) => {
                scan_array_place(place, params, used);
                scan_array_expr(value, params, used);
            }
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                scan_array_expr(cond, params, used);
                scan_array_stmt(&then_body.statements, params, used);
                for (cond, block) in elseif_blocks {
                    scan_array_expr(cond, params, used);
                    scan_array_stmt(&block.statements, params, used);
                }
                if let Some(block) = else_body {
                    scan_array_stmt(&block.statements, params, used);
                }
            }
            HirStmtKind::While { cond, body } => {
                scan_array_expr(cond, params, used);
                scan_array_stmt(&body.statements, params, used);
            }
            HirStmtKind::For { range, body, .. } => {
                scan_array_expr(range, params, used);
                scan_array_stmt(&body.statements, params, used);
            }
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => {
                scan_array_expr(expr, params, used);
                for (case, block) in cases {
                    scan_array_expr(case, params, used);
                    scan_array_stmt(&block.statements, params, used);
                }
                if let Some(block) = otherwise {
                    scan_array_stmt(&block.statements, params, used);
                }
            }
            _ => {}
        }
    }
}

/// Scan an assignment target for array writes (`A(i) = …`) on parameters under
/// inference.
fn scan_array_place(place: &HirPlace, params: &HashSet<BindingId>, used: &mut HashSet<BindingId>) {
    match place {
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            if indexing.kind == IndexKind::Paren {
                mark_array_param(base, params, used);
            }
            scan_array_expr(base, params, used);
            for component in &indexing.components {
                if let IndexComponent::Expr(e) = component {
                    scan_array_expr(e, params, used);
                }
            }
        }
        HirPlace::Member(base, _) => scan_array_expr(base, params, used),
        HirPlace::MemberDynamic(base, expr) => {
            scan_array_expr(base, params, used);
            scan_array_expr(expr, params, used);
        }
        HirPlace::Binding(_) => {}
    }
}

/// Scan an expression for array usages of parameters under inference.
fn scan_array_expr(expr: &HirExpr, params: &HashSet<BindingId>, used: &mut HashSet<BindingId>) {
    match &expr.kind {
        HirExprKind::Index(base, indexing) => {
            if indexing.kind == IndexKind::Paren {
                mark_array_param(base, params, used);
            }
            scan_array_expr(base, params, used);
            for component in &indexing.components {
                if let IndexComponent::Expr(e) = component {
                    scan_array_expr(e, params, used);
                }
            }
        }
        HirExprKind::Unary(_, operand) => scan_array_expr(operand, params, used),
        HirExprKind::Binary(lhs, _, rhs) => {
            scan_array_expr(lhs, params, used);
            scan_array_expr(rhs, params, used);
        }
        HirExprKind::Call(call) => {
            if let Some(name) = call_name(&call.callee) {
                if array_arg_builtin(&name, call.args.len()) {
                    if let Some(first) = call.args.first() {
                        mark_array_param(first, params, used);
                    }
                }
            }
            for arg in &call.args {
                scan_array_expr(arg, params, used);
            }
        }
        HirExprKind::Range(start, step, end) => {
            scan_array_expr(start, params, used);
            if let Some(step) = step {
                scan_array_expr(step, params, used);
            }
            scan_array_expr(end, params, used);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for element in row {
                    scan_array_expr(element, params, used);
                }
            }
        }
        HirExprKind::Member(base, _) => scan_array_expr(base, params, used),
        HirExprKind::MemberDynamic(base, expr) => {
            scan_array_expr(base, params, used);
            scan_array_expr(expr, params, used);
        }
        HirExprKind::StructLiteral(pairs) => {
            for (_, value) in pairs {
                scan_array_expr(value, params, used);
            }
        }
        HirExprKind::ObjectLiteral { fields: pairs, .. } => {
            for (_, value) in pairs {
                scan_array_expr(value, params, used);
            }
        }
        _ => {}
    }
}

/// Record `name` as a scalar field of struct parameter `id`, if `id` is one of
/// the parameters under inference and the field is not already recorded.
fn record_struct_field(
    id: BindingId,
    name: &str,
    params: &HashSet<BindingId>,
    fields: &mut HashMap<BindingId, Vec<(String, LocalTy)>>,
) {
    if !params.contains(&id) {
        return;
    }
    let list = fields.entry(id).or_default();
    if !list.iter().any(|(n, _)| n == name) {
        list.push((name.to_string(), LocalTy::Scalar));
    }
}

/// Scan statements for struct-field access on parameters under inference.
fn collect_stmt_fields(
    stmts: &[HirStmt],
    params: &HashSet<BindingId>,
    fields: &mut HashMap<BindingId, Vec<(String, LocalTy)>>,
) {
    for stmt in stmts {
        match &stmt.kind {
            HirStmtKind::ExprStmt(expr, _) => collect_expr_fields(expr, params, fields),
            HirStmtKind::Assign(place, value, _) => {
                collect_place_fields(place, params, fields);
                collect_expr_fields(value, params, fields);
            }
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                collect_expr_fields(cond, params, fields);
                collect_stmt_fields(&then_body.statements, params, fields);
                for (cond, block) in elseif_blocks {
                    collect_expr_fields(cond, params, fields);
                    collect_stmt_fields(&block.statements, params, fields);
                }
                if let Some(block) = else_body {
                    collect_stmt_fields(&block.statements, params, fields);
                }
            }
            HirStmtKind::While { cond, body } => {
                collect_expr_fields(cond, params, fields);
                collect_stmt_fields(&body.statements, params, fields);
            }
            HirStmtKind::For { range, body, .. } => {
                collect_expr_fields(range, params, fields);
                collect_stmt_fields(&body.statements, params, fields);
            }
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => {
                collect_expr_fields(expr, params, fields);
                for (case, block) in cases {
                    collect_expr_fields(case, params, fields);
                    collect_stmt_fields(&block.statements, params, fields);
                }
                if let Some(block) = otherwise {
                    collect_stmt_fields(&block.statements, params, fields);
                }
            }
            _ => {}
        }
    }
}

/// Scan a `HirPlace` for struct-field writes on parameters under inference.
fn collect_place_fields(
    place: &HirPlace,
    params: &HashSet<BindingId>,
    fields: &mut HashMap<BindingId, Vec<(String, LocalTy)>>,
) {
    match place {
        HirPlace::Member(base, name) => {
            if let HirExprKind::Binding(id) = base.kind {
                record_struct_field(id, &name.0, params, fields);
            }
            collect_expr_fields(base, params, fields);
        }
        HirPlace::MemberDynamic(base, expr) => {
            collect_expr_fields(base, params, fields);
            collect_expr_fields(expr, params, fields);
        }
        HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
            collect_expr_fields(base, params, fields);
            for component in &indexing.components {
                if let IndexComponent::Expr(e) = component {
                    collect_expr_fields(e, params, fields);
                }
            }
        }
        HirPlace::Binding(_) => {}
    }
}

/// Scan an expression for struct-field reads on parameters under inference.
fn collect_expr_fields(
    expr: &HirExpr,
    params: &HashSet<BindingId>,
    fields: &mut HashMap<BindingId, Vec<(String, LocalTy)>>,
) {
    match &expr.kind {
        HirExprKind::Member(base, name) => {
            if let HirExprKind::Binding(id) = base.kind {
                record_struct_field(id, &name.0, params, fields);
            }
            collect_expr_fields(base, params, fields);
        }
        HirExprKind::Unary(_, operand) => collect_expr_fields(operand, params, fields),
        HirExprKind::Binary(lhs, _, rhs) => {
            collect_expr_fields(lhs, params, fields);
            collect_expr_fields(rhs, params, fields);
        }
        HirExprKind::Index(base, indexing) => {
            collect_expr_fields(base, params, fields);
            for component in &indexing.components {
                if let IndexComponent::Expr(e) = component {
                    collect_expr_fields(e, params, fields);
                }
            }
        }
        HirExprKind::Range(start, step, end) => {
            collect_expr_fields(start, params, fields);
            if let Some(step) = step {
                collect_expr_fields(step, params, fields);
            }
            collect_expr_fields(end, params, fields);
        }
        HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
            for row in rows {
                for element in row {
                    collect_expr_fields(element, params, fields);
                }
            }
        }
        HirExprKind::Call(call) => {
            for arg in &call.args {
                collect_expr_fields(arg, params, fields);
            }
        }
        HirExprKind::StructLiteral(pairs) => {
            for (_, value) in pairs {
                collect_expr_fields(value, params, fields);
            }
        }
        HirExprKind::ObjectLiteral { fields: pairs, .. } => {
            for (_, value) in pairs {
                collect_expr_fields(value, params, fields);
            }
        }
        _ => {}
    }
}

fn infer_block(
    stmts: &[HirStmt],
    variadics: &Variadics,
    handles: &HandleTargets,
    tys: &mut HashMap<BindingId, LocalTy>,
) {
    for stmt in stmts {
        match &stmt.kind {
            HirStmtKind::Assign(HirPlace::Binding(target), value, _) => {
                // Handle definitions hold no numeric value; the binding is
                // resolved through `handles` instead.
                if matches!(value.kind, HirExprKind::AnonymousFunction(_)) {
                    continue;
                }
                let ty = expr_ty(value, tys, variadics.varargin_local, handles);
                tys.insert(*target, ty);
            }
            HirStmtKind::If {
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                infer_block(&then_body.statements, variadics, handles, tys);
                for (_, block) in elseif_blocks {
                    infer_block(&block.statements, variadics, handles, tys);
                }
                if let Some(block) = else_body {
                    infer_block(&block.statements, variadics, handles, tys);
                }
            }
            HirStmtKind::While { body, .. } | HirStmtKind::For { body, .. } => {
                infer_block(&body.statements, variadics, handles, tys);
            }
            HirStmtKind::Switch {
                cases, otherwise, ..
            } => {
                for (_, block) in cases {
                    infer_block(&block.statements, variadics, handles, tys);
                }
                if let Some(block) = otherwise {
                    infer_block(&block.statements, variadics, handles, tys);
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn expr_ty(
    expr: &HirExpr,
    tys: &HashMap<BindingId, LocalTy>,
    varargin_local: Option<BindingId>,
    handles: &HandleTargets,
) -> LocalTy {
    match &expr.kind {
        HirExprKind::Binding(id) => tys.get(id).cloned().unwrap_or(LocalTy::Scalar),
        HirExprKind::Number(_) | HirExprKind::IntegerLiteral(_) => LocalTy::Scalar,
        HirExprKind::Constant(symbol) => match symbol.0.as_str() {
            "true" | "false" | "Inf" | "Infinity" | "NaN" => LocalTy::Scalar,
            _ => LocalTy::Dynamic,
        },
        HirExprKind::Unary(op, operand) => {
            unary_ty(*op, expr_ty(operand, tys, varargin_local, handles))
        }
        HirExprKind::Binary(lhs, op, rhs) => binary_ty(
            expr_ty(lhs, tys, varargin_local, handles),
            *op,
            expr_ty(rhs, tys, varargin_local, handles),
        ),
        HirExprKind::Tensor(rows) => {
            let elem_shapes: Option<Vec<Vec<Shape>>> = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| concat_operand_shape(&expr_ty(e, tys, varargin_local, handles)))
                        .collect::<Option<Vec<_>>>()
                })
                .collect();
            match elem_shapes.and_then(|shapes| concat_shape(&shapes)) {
                Some(shape) => LocalTy::Array { shape },
                None => LocalTy::Dynamic,
            }
        }
        HirExprKind::Cell(_) => LocalTy::Dynamic,
        HirExprKind::Call(call) => call_ty(call, tys, handles),
        HirExprKind::Member(base, name) => match expr_ty(base, tys, varargin_local, handles) {
            LocalTy::Struct { fields } => fields
                .iter()
                .find(|(field, _)| field == &name.0)
                .map(|(_, ty)| ty.clone())
                .unwrap_or(LocalTy::Dynamic),
            _ => LocalTy::Dynamic,
        },
        HirExprKind::Index(base, indexing) => {
            // `varargin{k}` (constant `k`) resolves to a scalar extra argument.
            if let HirExprKind::Binding(id) = base.kind {
                if varargin_local == Some(id)
                    && indexing.kind == IndexKind::Brace
                    && brace_index(indexing).is_some()
                {
                    return LocalTy::Scalar;
                }
                // A call through an anonymous-function handle. The MVP only
                // supports scalar-returning handles, so the result is a scalar.
                if handles.contains_key(&id) {
                    return LocalTy::Scalar;
                }
            }
            index_ty(expr_ty(base, tys, varargin_local, handles), indexing)
        }
        _ => LocalTy::Dynamic,
    }
}

/// Elementwise operators that accept array operands (with scalar broadcast).
fn is_elementwise(op: OperatorKind) -> bool {
    matches!(
        op,
        OperatorKind::Add
            | OperatorKind::Subtract
            | OperatorKind::ElementwiseMultiply
            | OperatorKind::ElementwiseDivide
            | OperatorKind::ElementwiseLeftDivide
            | OperatorKind::ElementwisePower
            | OperatorKind::Equal
            | OperatorKind::NotEqual
            | OperatorKind::Less
            | OperatorKind::LessEqual
            | OperatorKind::Greater
            | OperatorKind::GreaterEqual
            | OperatorKind::ElementwiseAnd
            | OperatorKind::ElementwiseOr
    )
}

fn unary_ty(op: OperatorKind, ty: LocalTy) -> LocalTy {
    match (op, ty) {
        // 2-D transpose swaps the first two dimensions.
        (OperatorKind::Transpose | OperatorKind::ConjugateTranspose, LocalTy::Array { shape }) => {
            if !shape.is_dynamic() && shape.rank() == 2 {
                LocalTy::Array {
                    shape: Shape::matrix(shape.dims()[1], shape.dims()[0]),
                }
            } else {
                LocalTy::Dynamic
            }
        }
        // Unary minus/plus/not preserve the shape elementwise.
        (_, LocalTy::Array { shape }) => LocalTy::Array { shape },
        (_, LocalTy::Scalar) => LocalTy::Scalar,
        (_, LocalTy::Struct { .. }) | (_, LocalTy::Dynamic) => LocalTy::Dynamic,
    }
}

fn binary_ty(lhs: LocalTy, op: OperatorKind, rhs: LocalTy) -> LocalTy {
    use LocalTy::*;

    match (lhs, rhs) {
        (Scalar, Scalar) => Scalar,
        (Struct { .. }, _) | (_, Struct { .. }) | (Dynamic, _) | (_, Dynamic) => Dynamic,
        // Matrix power `A ^ scalar` keeps `A`'s shape (a square matrix).
        (Array { shape }, Scalar) if op == OperatorKind::MatrixPower => Array { shape },
        // `scalar ^ matrix` is `expm`; not statically lowerable.
        (Scalar, Array { .. }) if op == OperatorKind::MatrixPower => Dynamic,
        // Scalar broadcast against an array (`*` behaves like `.*` here).
        (Scalar, Array { shape }) | (Array { shape }, Scalar) => {
            if is_elementwise(op) || op == OperatorKind::MatrixMultiply {
                Array { shape }
            } else {
                Dynamic
            }
        }
        // Two arrays: elementwise (with implicit singleton expansion) when the
        // shapes are broadcast-compatible; `*` is a matmul; `^` (matrix ^ matrix)
        // is not defined and is deferred.
        (Array { shape: lhs_shape }, Array { shape: rhs_shape }) => match op {
            OperatorKind::MatrixMultiply => matmul_ty(lhs_shape, rhs_shape),
            OperatorKind::Mldivide => mldivide_ty(lhs_shape, rhs_shape),
            OperatorKind::MatrixPower => Dynamic,
            _ if is_elementwise(op) => match broadcast_shape(lhs_shape, rhs_shape) {
                Some(shape) => Array { shape },
                // Two dynamic-shape (vector) arrays of equal runtime length.
                None if lhs_shape.is_dynamic() && rhs_shape.is_dynamic() => Array {
                    shape: Shape::Dynamic,
                },
                None => Dynamic,
            },
            _ => Dynamic,
        },
    }
}

/// The result shape of implicit singleton expansion (MATLAB broadcasting) for
/// two static 2-D shapes, or `None` when they are not broadcast-compatible.
pub(crate) fn broadcast_shape(lhs: Shape, rhs: Shape) -> Option<Shape> {
    if lhs.is_dynamic() || rhs.is_dynamic() || lhs.rank() != 2 || rhs.rank() != 2 {
        return None;
    }
    let mut dims = Vec::with_capacity(2);
    for (&l, &r) in lhs.dims().iter().zip(rhs.dims()) {
        dims.push(if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return None;
        });
    }
    Shape::from_dims(&dims)
}

/// The shape of `A * B` for two 2-D operands (`m x k` times `k x n`).
fn matmul_ty(lhs: Shape, rhs: Shape) -> LocalTy {
    if lhs.is_dynamic() || rhs.is_dynamic() {
        return LocalTy::Dynamic;
    }
    if lhs.rank() == 2 && rhs.rank() == 2 && lhs.dims()[1] == rhs.dims()[0] {
        LocalTy::Array {
            shape: Shape::matrix(lhs.dims()[0], rhs.dims()[1]),
        }
    } else {
        LocalTy::Dynamic
    }
}

/// The shape of `A \ B` for two static 2-D operands: solving `A X = B` with a
/// square `A` gives an `n x k` result.
fn mldivide_ty(lhs: Shape, rhs: Shape) -> LocalTy {
    if lhs.is_dynamic() || rhs.is_dynamic() {
        return LocalTy::Dynamic;
    }
    if lhs.rank() == 2
        && rhs.rank() == 2
        && lhs.dims()[0] == lhs.dims()[1]
        && lhs.dims()[0] == rhs.dims()[0]
    {
        LocalTy::Array {
            shape: Shape::matrix(lhs.dims()[1], rhs.dims()[1]),
        }
    } else {
        LocalTy::Dynamic
    }
}

fn call_ty(call: &HirCall, tys: &HashMap<BindingId, LocalTy>, handles: &HandleTargets) -> LocalTy {
    // A user-defined function call: the scalar-ABI subset (all scalar args,
    // single scalar result) is lowered directly; arrays defer.
    if is_user_call(call) {
        let args: Vec<LocalTy> = call
            .args
            .iter()
            .map(|arg| expr_ty(arg, tys, None, handles))
            .collect();
        return if args.iter().all(|ty| matches!(ty, LocalTy::Scalar)) {
            LocalTy::Scalar
        } else {
            LocalTy::Dynamic
        };
    }
    let Some(name) = call_name(&call.callee) else {
        return LocalTy::Dynamic;
    };
    // `struct('a', 1, 'b', 2, ...)` constructs a struct with alternating
    // field-name/field-value arguments.
    if name == "struct" {
        return struct_ty(call, tys, handles);
    }
    let Some(builtin) = builtins::lookup(&name) else {
        return LocalTy::Dynamic;
    };

    let args: Vec<LocalTy> = call
        .args
        .iter()
        .map(|arg| expr_ty(arg, tys, None, handles))
        .collect();

    match builtin {
        // Elementwise unary: preserves the argument's shape; `sort` also
        // preserves shape (a sorted copy).
        Builtin::Unary(_) | Builtin::Sort => args.first().cloned().unwrap_or(LocalTy::Dynamic),
        // Reductions: one array argument reduces to a scalar; a second numeric
        // argument selects the dimension to reduce along (array result).
        Builtin::MinMax(_) | Builtin::Reduce(_) => match args.as_slice() {
            [LocalTy::Scalar] | [LocalTy::Array { .. }] | [LocalTy::Scalar, LocalTy::Scalar] => {
                LocalTy::Scalar
            }
            [LocalTy::Array { shape }, _] => match reduction_dim(call) {
                Some(dim) => reduce_axis_ty(*shape, dim),
                None => LocalTy::Dynamic,
            },
            _ => LocalTy::Dynamic,
        },
        // Shape introspection: scalar results, except `size(A)` (a 1x2 vector).
        Builtin::Numel | Builtin::Length => LocalTy::Scalar,
        Builtin::Size => {
            if call.args.len() == 1 {
                LocalTy::Array {
                    shape: Shape::matrix(1, 2),
                }
            } else {
                LocalTy::Scalar
            }
        }
        // Scalar reductions. `mean` accepts a dynamic-shape (vector) parameter;
        // `std`/`median`/`var` are static-only (they need a compile-time temp).
        Builtin::Mean => match args.as_slice() {
            [LocalTy::Scalar] => LocalTy::Scalar,
            [LocalTy::Array { shape }] if is_vector_shape(*shape) => LocalTy::Scalar,
            _ => LocalTy::Dynamic,
        },
        Builtin::Std | Builtin::Median | Builtin::Var => match args.as_slice() {
            [LocalTy::Scalar] => LocalTy::Scalar,
            [LocalTy::Array { shape }] if is_static_vector(*shape) => LocalTy::Scalar,
            _ => LocalTy::Dynamic,
        },
        // `isempty` is a scalar predicate; `logical` is elementwise.
        Builtin::IsEmpty => LocalTy::Scalar,
        Builtin::Logical => args.first().cloned().unwrap_or(LocalTy::Dynamic),
        // Array constructors with constant shape controls.
        Builtin::LinSpace => {
            let n = constant_arg(call, 2).unwrap_or(100);
            LocalTy::Array {
                shape: Shape::matrix(1, n),
            }
        }
        Builtin::Repmat => repmat_ty(call, &args),
        Builtin::Permute => permute_ty(call, &args),
        // Linear algebra on static square matrices / vectors.
        Builtin::Inv => match args.first() {
            Some(LocalTy::Array { shape }) if is_static_square(*shape) => {
                LocalTy::Array { shape: *shape }
            }
            _ => LocalTy::Dynamic,
        },
        Builtin::Det => match args.first() {
            Some(LocalTy::Scalar) => LocalTy::Scalar,
            Some(LocalTy::Array { shape }) if is_static_square(*shape) => LocalTy::Scalar,
            _ => LocalTy::Dynamic,
        },
        Builtin::Norm => match args.first() {
            Some(LocalTy::Scalar) => LocalTy::Scalar,
            Some(LocalTy::Array { shape }) if is_static_vector(*shape) => LocalTy::Scalar,
            _ => LocalTy::Dynamic,
        },
        Builtin::Rand => LocalTy::Scalar,
        // `cumsum` preserves the operand's shape; `diff` shortens a vector by one.
        Builtin::CumSum => match args.as_slice() {
            [LocalTy::Scalar] => LocalTy::Scalar,
            [LocalTy::Array { shape }] if is_static_vector(*shape) => {
                LocalTy::Array { shape: *shape }
            }
            _ => LocalTy::Dynamic,
        },
        Builtin::Diff => match args.as_slice() {
            [LocalTy::Array { shape }] => match diff_result_shape(*shape) {
                Some(shape) => LocalTy::Array { shape },
                None => LocalTy::Dynamic,
            },
            _ => LocalTy::Dynamic,
        },
        // Everything else in the supported set is scalar-valued.
        Builtin::Binary(_) | Builtin::Mod | Builtin::Sign => LocalTy::Scalar,
        // Constructors: array shapes from constant dim arguments (`zeros`/`ones`
        // accept N dimensions; `eye` is 2-D).
        Builtin::Fill(_) => fill_ty(call),
        Builtin::Eye => match (constant_arg(call, 0), constant_arg(call, 1)) {
            (Some(n), None) => LocalTy::Array {
                shape: Shape::matrix(n, n),
            },
            (Some(rows), Some(cols)) => LocalTy::Array {
                shape: Shape::matrix(rows, cols),
            },
            _ => LocalTy::Dynamic,
        },
        Builtin::Reshape => match (constant_arg(call, 1), constant_arg(call, 2)) {
            (Some(rows), Some(cols)) => LocalTy::Array {
                shape: Shape::matrix(rows, cols),
            },
            _ => LocalTy::Dynamic,
        },
    }
}

/// The shape of `zeros`/`ones` from its constant dimension arguments: one
/// argument is a square matrix, more form an N-D array.
fn fill_ty(call: &HirCall) -> LocalTy {
    let mut dims = Vec::with_capacity(call.args.len());
    for i in 0..call.args.len() {
        match constant_arg(call, i) {
            Some(n) => dims.push(n),
            None => return LocalTy::Dynamic,
        }
    }
    if dims.is_empty() {
        return LocalTy::Dynamic;
    }
    let full = if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        dims
    };
    match Shape::from_dims(&full) {
        Some(shape) => LocalTy::Array { shape },
        None => LocalTy::Dynamic,
    }
}

/// The shape of `repmat(A, m, n)` for a static 2-D `A` and constant `m`/`n`.
fn repmat_ty(call: &HirCall, args: &[LocalTy]) -> LocalTy {
    let Some(LocalTy::Array { shape }) = args.first() else {
        return LocalTy::Dynamic;
    };
    if shape.is_dynamic() || shape.rank() != 2 {
        return LocalTy::Dynamic;
    }
    let Some(m) = constant_arg(call, 1) else {
        return LocalTy::Dynamic;
    };
    let n = constant_arg(call, 2).unwrap_or(m);
    LocalTy::Array {
        shape: Shape::matrix(shape.dims()[0] * m, shape.dims()[1] * n),
    }
}

/// The shape of `permute(A, order)` for a static `A` and a constant `order`.
fn permute_ty(call: &HirCall, args: &[LocalTy]) -> LocalTy {
    let Some(LocalTy::Array { shape }) = args.first() else {
        return LocalTy::Dynamic;
    };
    if shape.is_dynamic() {
        return LocalTy::Dynamic;
    }
    let Some(order) = constant_int_list(call, 1) else {
        return LocalTy::Dynamic;
    };
    if order.len() != shape.rank() {
        return LocalTy::Dynamic;
    }
    let mut seen = vec![false; order.len()];
    let mut dims = Vec::with_capacity(order.len());
    for &p in &order {
        if p == 0 || p > order.len() || seen[p - 1] {
            return LocalTy::Dynamic;
        }
        seen[p - 1] = true;
        dims.push(shape.dims()[p - 1]);
    }
    match Shape::from_dims(&dims) {
        Some(shape) => LocalTy::Array { shape },
        None => LocalTy::Dynamic,
    }
}

/// The constant integer values of a row-vector literal argument (e.g. `[2 1]`).
pub(crate) fn constant_int_list(call: &HirCall, index: usize) -> Option<Vec<usize>> {
    let expr = call.args.get(index)?;
    let HirExprKind::Tensor(rows) = &expr.kind else {
        return None;
    };
    if rows.len() != 1 {
        return None;
    }
    rows[0].iter().map(constant_expr_value).collect()
}

/// The type of a `struct('a', 1, 'b', 2, ...)` construction: alternating
/// field-name (`String`) / field-value arguments, in declaration order.
fn struct_ty(
    call: &HirCall,
    tys: &HashMap<BindingId, LocalTy>,
    handles: &HandleTargets,
) -> LocalTy {
    let mut fields = Vec::new();
    let mut args = call.args.iter();
    while let Some(name_arg) = args.next() {
        let Some(value_arg) = args.next() else {
            return LocalTy::Dynamic;
        };
        let HirExprKind::String(field_name) = &name_arg.kind else {
            return LocalTy::Dynamic;
        };
        let field_ty = expr_ty(value_arg, tys, None, handles);
        fields.push((unquote_str(&field_name.0).to_string(), field_ty));
    }
    if fields.is_empty() {
        return LocalTy::Dynamic;
    }
    LocalTy::Struct { fields }
}

/// The constant `usize` value of the `index`-th argument of a call, if present.
fn constant_arg(call: &HirCall, index: usize) -> Option<usize> {
    call.args.get(index).and_then(constant_expr_value)
}

/// The 1-based dimension argument of a reduction call, if present and constant.
fn reduction_dim(call: &HirCall) -> Option<usize> {
    if call.args.len() < 2 {
        return None;
    }
    constant_expr_value(&call.args[1])
}

/// The static 2-D shape of a concatenation operand: a scalar is `1x1`, a static
/// array is its shape; a dynamic/struct operand has no static shape.
pub(crate) fn concat_operand_shape(ty: &LocalTy) -> Option<Shape> {
    match ty {
        LocalTy::Scalar => Some(Shape::matrix(1, 1)),
        LocalTy::Array { shape } if !shape.is_dynamic() => Some(*shape),
        _ => None,
    }
}

/// The shape of a block-concatenation literal `[a b; c d]` from its operand
/// shapes, or `None` when they do not form a consistent block matrix (every
/// block must be 2-D; blocks in one row share a row count; blocks in one column
/// share a column count).
pub(crate) fn concat_shape(rows: &[Vec<Shape>]) -> Option<Shape> {
    if rows.is_empty() || rows[0].is_empty() {
        return None;
    }
    let cols = rows[0].len();
    if rows.iter().any(|row| row.len() != cols) {
        return None;
    }
    if rows
        .iter()
        .flatten()
        .any(|shape| shape.is_dynamic() || shape.rank() != 2)
    {
        return None;
    }
    let mut heights = Vec::with_capacity(rows.len());
    for row in rows {
        let h = row[0].dims()[0];
        if row.iter().any(|shape| shape.dims()[0] != h) {
            return None;
        }
        heights.push(h);
    }
    let mut widths = Vec::with_capacity(cols);
    for c in 0..cols {
        let w = rows[0][c].dims()[1];
        if rows.iter().any(|row| row[c].dims()[1] != w) {
            return None;
        }
        widths.push(w);
    }
    let total_rows: usize = heights.iter().sum();
    let total_cols: usize = widths.iter().sum();
    Some(Shape::matrix(total_rows, total_cols))
}

/// The shape of reducing a 2-D array along dimension `dim` (1 or 2).
fn reduce_axis_ty(shape: Shape, dim: usize) -> LocalTy {
    let Shape::Static { rank, mut dims } = shape else {
        return LocalTy::Dynamic;
    };
    if rank != 2 {
        return LocalTy::Dynamic;
    }
    match dim {
        1 => dims[0] = 1,
        2 => dims[1] = 1,
        _ => return LocalTy::Dynamic,
    }
    LocalTy::Array {
        shape: Shape::Static { rank, dims },
    }
}

/// Whether a shape is a vector (rank-2 with a singleton dimension). A
/// dynamic-shape array parameter is treated as a vector by the runtime ABI.
fn is_vector_shape(shape: Shape) -> bool {
    match shape {
        Shape::Dynamic => true,
        Shape::Static { rank, dims } => rank == 2 && (dims[0] == 1 || dims[1] == 1),
    }
}

/// Whether a shape is a statically-known vector.
fn is_static_vector(shape: Shape) -> bool {
    !shape.is_dynamic() && is_vector_shape(shape)
}

/// Whether a shape is a statically-known square matrix.
fn is_static_square(shape: Shape) -> bool {
    matches!(shape, Shape::Static { rank: 2, dims } if dims[0] == dims[1])
}

/// The shape of `diff` of a static vector (one element shorter), if defined.
fn diff_result_shape(shape: Shape) -> Option<Shape> {
    let Shape::Static { rank, dims } = shape else {
        return None;
    };
    if rank != 2 {
        return None;
    }
    let mut out = dims;
    if out[0] == 1 && out[1] >= 2 {
        out[1] -= 1;
    } else if out[1] == 1 && out[0] >= 2 {
        out[0] -= 1;
    } else {
        return None;
    }
    Some(Shape::Static { rank: 2, dims: out })
}

/// The type of an indexing expression `base(...)`.
fn index_ty(base: LocalTy, indexing: &IndexingSemantics) -> LocalTy {
    let LocalTy::Array { shape } = base else {
        return LocalTy::Dynamic;
    };
    let comps = indexing.components.as_slice();

    if shape.is_dynamic() {
        // `A(:)` on a dynamic vector is the identity; a scalar/tuntime subscript
        // yields a scalar element (the parameter is treated as a vector; see
        // `hir_to_mlir`).
        if comps.len() == 1 && component_is_colon(&comps[0]) {
            return LocalTy::Array {
                shape: Shape::Dynamic,
            };
        }
        return LocalTy::Scalar;
    }

    if shape.rank() != 2 {
        return static_or_dynamic_scalar(comps);
    }

    match comps {
        // Linear indexing: `A(:)` columnizes; a constant range keeps a vector's
        // orientation; a scalar subscript is an element (scalar).
        [c] => {
            if component_is_colon(c) {
                return LocalTy::Array {
                    shape: Shape::matrix(shape.numel(), 1),
                };
            }
            if !is_scalar_selector(c) {
                if let Some(sel) = component_selection(c, shape.numel()) {
                    let k = sel.len();
                    return LocalTy::Array {
                        shape: if shape.dims()[0] == 1 {
                            Shape::matrix(1, k)
                        } else {
                            Shape::matrix(k, 1)
                        },
                    };
                }
            }
            static_or_dynamic_scalar(comps)
        }
        // Two subscripts: a slice `A(i,:)` / `A(:,j)` / `A(i0:i1, j0:j1)`; two
        // scalar selectors are an element (scalar).
        [r, c] => {
            let (Some(rows), Some(cols)) = (
                component_selection(r, shape.dims()[0]),
                component_selection(c, shape.dims()[1]),
            ) else {
                return static_or_dynamic_scalar(comps);
            };
            if is_scalar_selector(r) && is_scalar_selector(c) {
                LocalTy::Scalar
            } else {
                LocalTy::Array {
                    shape: Shape::matrix(rows.len(), cols.len()),
                }
            }
        }
        _ => static_or_dynamic_scalar(comps),
    }
}

/// A scalar subscript (all components constant / `end`) is a scalar element;
/// anything else is deferred.
fn static_or_dynamic_scalar(comps: &[IndexComponent]) -> LocalTy {
    if static_scalar_subscript(comps) {
        LocalTy::Scalar
    } else {
        LocalTy::Dynamic
    }
}

/// Whether every index component is a constant scalar or `end` (a scalar
/// subscript).
fn static_scalar_subscript(comps: &[IndexComponent]) -> bool {
    !comps.is_empty()
        && comps.iter().all(|c| {
            component_end_offset(c).is_some()
                || matches!(c, IndexComponent::Expr(e) if constant_expr_value(e).is_some())
        })
}

/// Whether a component selects one position without a range (`i` or `end`), as
/// opposed to `:` or a range.
fn is_scalar_selector(component: &IndexComponent) -> bool {
    if component_is_colon(component) {
        return false;
    }
    if component_end_offset(component).is_some() {
        return true;
    }
    match component {
        IndexComponent::Expr(e) => {
            !matches!(e.kind, HirExprKind::Range(..)) && constant_expr_value(e).is_some()
        }
        _ => false,
    }
}

/// The 0-based positions selected by an index component along a dimension of
/// length `dim`: `:` selects all, `i`/`end±k` selects one, a constant range
/// selects its values. `None` when not statically resolvable.
fn component_selection(component: &IndexComponent, dim: usize) -> Option<Vec<usize>> {
    if component_is_colon(component) {
        return Some((0..dim).collect());
    }
    if let Some(offset) = component_end_offset(component) {
        let idx = dim as isize + offset;
        if idx < 1 || idx as usize > dim {
            return None;
        }
        return Some(vec![idx as usize - 1]);
    }
    let IndexComponent::Expr(expr) = component else {
        return None;
    };
    if let HirExprKind::Range(start, step, end) = &expr.kind {
        let vals = range_values(start, step.as_deref(), end, dim)?;
        return vals
            .into_iter()
            .map(|v| {
                if v >= 1 && v as usize <= dim {
                    Some(v as usize - 1)
                } else {
                    None
                }
            })
            .collect();
    }
    constant_expr_value(expr).and_then(|v| {
        if v >= 1 && v <= dim {
            Some(vec![v - 1])
        } else {
            None
        }
    })
}

/// The source linear (column-major) index of each element of `base(indexing)`,
/// for a statically-shaped rank-2 base, in destination (column-major) order.
/// `None` when the selection is not statically resolvable.
pub(crate) fn static_index_selection(
    base: Shape,
    indexing: &IndexingSemantics,
) -> Option<Vec<usize>> {
    if base.is_dynamic() || base.rank() != 2 {
        return None;
    }
    let comps = indexing.components.as_slice();
    match comps {
        [c] => {
            if component_is_colon(c) {
                return Some((0..base.numel()).collect());
            }
            component_selection(c, base.numel())
        }
        [r, c] => {
            let rows = component_selection(r, base.dims()[0])?;
            let cols = component_selection(c, base.dims()[1])?;
            let mut out = Vec::with_capacity(rows.len() * cols.len());
            for &col in &cols {
                for &row in &rows {
                    out.push(base.linear(&[row, col]));
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// The 1-based values of a constant range `start:step:end`, with `end` resolved
/// against `dim`.
fn range_values(
    start: &HirExpr,
    step: Option<&HirExpr>,
    end: &HirExpr,
    dim: usize,
) -> Option<Vec<isize>> {
    let s = int_value(start, dim)?;
    let e = int_value(end, dim)?;
    let st = match step {
        Some(expr) => int_value(expr, dim)?,
        None => 1,
    };
    if st == 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut v = s;
    if st > 0 {
        while v <= e {
            out.push(v);
            v += st;
        }
    } else {
        while v >= e {
            out.push(v);
            v += st;
        }
    }
    Some(out)
}

/// The constant integer value of an expression, with `end`/`end±k` resolved
/// against `dim` (1-based).
fn int_value(expr: &HirExpr, dim: usize) -> Option<isize> {
    if let Some(offset) = expr_end_offset(expr) {
        return Some(dim as isize + offset);
    }
    match &expr.kind {
        HirExprKind::Number(text) => text.trim().parse::<isize>().ok(),
        HirExprKind::IntegerLiteral(literal) => Some(literal.bits() as isize),
        HirExprKind::Unary(op, operand) if *op == OperatorKind::UnaryMinus => {
            int_value(operand, dim).and_then(|v| v.checked_neg())
        }
        HirExprKind::Binary(lhs, op, rhs) => {
            let l = int_value(lhs, dim)?;
            let r = int_value(rhs, dim)?;
            match op {
                OperatorKind::Add => l.checked_add(r),
                OperatorKind::Subtract => l.checked_sub(r),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_numel() {
        assert_eq!(Shape::matrix(1, 3).numel(), 3);
        assert_eq!(Shape::matrix(3, 1).numel(), 3);
        assert_eq!(Shape::matrix(2, 3).numel(), 6);
    }

    #[test]
    fn shape_linear_is_column_major() {
        // A 2x3 matrix stores element (row, col) at offset col*rows + row.
        let shape = Shape::matrix(2, 3);
        assert_eq!(shape.linear(&[0, 0]), 0);
        assert_eq!(shape.linear(&[1, 0]), 1);
        assert_eq!(shape.linear(&[0, 1]), 2);
        assert_eq!(shape.linear(&[1, 1]), 3);
        assert_eq!(shape.linear(&[0, 2]), 4);
        assert_eq!(shape.linear(&[1, 2]), 5);
    }

    #[test]
    fn row_and_column_vectors_have_distinct_shapes() {
        assert_eq!(Shape::matrix(1, 3), Shape::matrix(1, 3));
        assert_ne!(Shape::matrix(1, 3), Shape::matrix(3, 1));
    }

    #[test]
    fn shape_static_vs_dynamic() {
        assert!(!Shape::matrix(2, 2).is_dynamic());
        assert!(Shape::Dynamic.is_dynamic());
        assert_ne!(Shape::matrix(2, 2), Shape::Dynamic);
    }

    #[test]
    fn shape_static_accessors() {
        let shape = Shape::matrix(2, 3);
        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.dims(), &[2, 3]);
        assert_eq!(shape.numel(), 6);
        assert_eq!(shape.linear(&[1, 2]), 5);
    }

    #[test]
    #[should_panic]
    fn shape_dynamic_numel_panics() {
        let _ = Shape::Dynamic.numel();
    }
}
