//! Layer 2: the codegen boundary.
//!
//! Classifies each MIR body as statically lowerable (`Verdict::Static`) or
//! deferred to the runtime (`Verdict::Deferred`). This is a function-level
//! decision, so a program can mix compiled and runtime code.
//!
//! The current implementation is a hand-written whitelist of the scalar-double
//! subset (plus statically-shaped array literals and pure numeric built-ins).
//! It is exposed behind [`Classifier`] so a `runmat-static-analysis`-driven
//! classifier (type/shape inference, definite assignment) can replace it
//! without changing callers.

use std::collections::HashMap;

use runmat_hir::{IndexKind, OperatorKind};
use runmat_mir::{
    MirBody, MirCall, MirCallArg, MirCallee, MirConstant, MirIndexComponent, MirIndexing,
    MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminator, MirTerminatorKind,
};

use crate::builtins::{self, Builtin};

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

/// The static type of a MIR local, as inferred by the boundary's lightweight
/// shape analysis (a stand-in for `runmat-static-analysis` in the MVP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalTy {
    /// A scalar `f64` value.
    Scalar,
    /// A `f64` array, with either a static or dynamic shape.
    Array { shape: Shape },
    /// Not a numeric value / not statically resolvable; deferred to the runtime.
    Dynamic,
}

/// The variadic-argument shape of a function body: which MIR locals back
/// `varargin`/`varargout`/`nargin`/`nargout`, and how many extra scalar
/// arguments/outputs the body actually uses.
///
/// In the closed-world, fixed-arity specialization (see `docs/architecture.md`
/// §12), `nargin`/`nargout` fold to `named_* + *_count`, and `varargin{k}` /
/// `varargout{k}` (constant `k`) resolve directly to the `k`-th extra argument
/// or output — no cell is materialized.
#[derive(Debug, Clone, Default)]
pub struct Variadics {
    pub varargin_local: Option<usize>,
    pub varargin_count: usize,
    pub varargout_local: Option<usize>,
    pub varargout_count: usize,
    pub nargin_local: Option<usize>,
    pub nargout_local: Option<usize>,
    /// Number of named (non-variadic) inputs: `fixed_inputs` minus `varargin`.
    pub named_inputs: usize,
    /// Number of named (non-variadic) outputs: `fixed_outputs` minus `varargout`.
    pub named_outputs: usize,
}

impl Variadics {
    /// Compute the variadic shape of `body` by mapping ABI bindings to locals
    /// and scanning for constant `varargin{k}` / `varargout{k}` indices.
    pub fn compute(body: &MirBody) -> Self {
        let mut binding_to_local = HashMap::new();
        for local in &body.locals {
            if let Some(binding) = local.binding {
                binding_to_local.insert(binding, local.id.0);
            }
        }
        let varargin_local = body
            .abi
            .varargin
            .and_then(|b| binding_to_local.get(&b).copied());
        let varargout_local = body
            .abi
            .varargout
            .and_then(|b| binding_to_local.get(&b).copied());
        let nargin_local = body
            .abi
            .implicit_nargin
            .and_then(|b| binding_to_local.get(&b).copied());
        let nargout_local = body
            .abi
            .implicit_nargout
            .and_then(|b| binding_to_local.get(&b).copied());

        let mut varargin_count = 0;
        let mut varargout_count = 0;
        for block in &body.blocks {
            for stmt in &block.statements {
                match &stmt.kind {
                    MirStmtKind::Assign { place, value } => {
                        if let Some(k) = varargout_index(place, varargout_local) {
                            varargout_count = varargout_count.max(k);
                        }
                        scan_varargin(value, varargin_local, &mut varargin_count);
                    }
                    MirStmtKind::Expr(value) => {
                        scan_varargin(value, varargin_local, &mut varargin_count);
                    }
                    MirStmtKind::PlaceMutation(mutation) => {
                        if let Some(k) = varargout_index(&mutation.place, varargout_local) {
                            varargout_count = varargout_count.max(k);
                        }
                    }
                    _ => {}
                }
            }
        }

        Variadics {
            varargin_local,
            varargin_count,
            varargout_local,
            varargout_count,
            nargin_local,
            nargout_local,
            named_inputs: body
                .abi
                .fixed_inputs
                .len()
                .saturating_sub(usize::from(body.abi.varargin.is_some())),
            named_outputs: body
                .abi
                .fixed_outputs
                .len()
                .saturating_sub(usize::from(body.abi.varargout.is_some())),
        }
    }
}

/// The 1-based constant index of a single-component `{}` (cell) index, if known.
pub(crate) fn brace_index(indexing: &MirIndexing) -> Option<usize> {
    match indexing.components.as_slice() {
        [MirIndexComponent::Expr(MirOperand::Constant(MirConstant::Number(text)))] => {
            text.trim().parse::<usize>().ok()
        }
        [MirIndexComponent::Expr(MirOperand::Constant(MirConstant::IntegerLiteral(lit)))] => {
            Some(lit.bits() as usize)
        }
        _ => None,
    }
}

/// Accumulate the largest constant `varargin{k}` index referenced by `rvalue`.
fn scan_varargin(rvalue: &MirRvalue, varargin_local: Option<usize>, count: &mut usize) {
    let Some(varargin_local) = varargin_local else {
        return;
    };
    match rvalue {
        MirRvalue::Index {
            base: MirOperand::Local(id),
            indexing,
        } => {
            if id.0 == varargin_local && indexing.kind == IndexKind::Brace {
                if let Some(k) = brace_index(indexing) {
                    *count = (*count).max(k);
                }
            }
        }
        MirRvalue::ShortCircuit { right_temps, .. } => {
            for stmt in right_temps {
                match &stmt.kind {
                    MirStmtKind::Assign { value, .. } | MirStmtKind::Expr(value) => {
                        scan_varargin(value, Some(varargin_local), count)
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// The 1-based constant index of a `varargout{k}` write target, if `place` is
/// `varargout{k}`.
pub(crate) fn varargout_index(place: &MirPlace, varargout_local: Option<usize>) -> Option<usize> {
    let varargout_local = varargout_local?;
    match place {
        MirPlace::Index(base, indexing) => {
            if let MirPlace::Local(id) = &**base {
                if id.0 == varargout_local && indexing.kind == IndexKind::Brace {
                    return brace_index(indexing);
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
            | OperatorKind::ElementwiseAnd
            | OperatorKind::ElementwiseOr
    )
}

/// Resolve a callable to its source-level name, when one exists.
pub fn call_name(callee: &MirCallee) -> Option<String> {
    match callee {
        MirCallee::Static(identity) => identity.display_name(),
        _ => None,
    }
}

fn operand_reason(operand: &MirOperand) -> Option<String> {
    match operand {
        MirOperand::Local(_) => None,
        MirOperand::Constant(constant) => match constant {
            MirConstant::Number(_) | MirConstant::IntegerLiteral(_) | MirConstant::Bool(_) => None,
            other => Some(format!("unsupported constant {other:?}")),
        },
        other => Some(format!("unsupported operand {other:?}")),
    }
}

fn rvalue_reason(value: &MirRvalue, varargout_local: Option<usize>) -> Option<String> {
    match value {
        MirRvalue::Use(operand) => operand_reason(operand),
        MirRvalue::Unary(op, operand) => {
            if supported_unary(op) {
                operand_reason(operand)
            } else {
                Some(format!("unsupported unary operator {op:?}"))
            }
        }
        MirRvalue::Binary(lhs, op, rhs) => operand_reason(lhs)
            .or_else(|| operand_reason(rhs))
            .or_else(|| {
                if supported_binary(op) {
                    None
                } else {
                    Some(format!("unsupported operator {op:?}"))
                }
            }),
        MirRvalue::ShortCircuit {
            left,
            right,
            right_temps,
            ..
        } => operand_reason(left)
            .or_else(|| operand_reason(right))
            .or_else(|| {
                right_temps
                    .iter()
                    .find_map(|s| stmt_reason(s, varargout_local))
            }),
        MirRvalue::Aggregate { kind, .. } => match kind {
            runmat_mir::MirAggregateKind::Tensor => None,
            runmat_mir::MirAggregateKind::Cell => {
                Some("cell array literals are not supported yet".to_string())
            }
        },
        MirRvalue::Call(call) => call_reason(call),
        MirRvalue::Index { base, indexing } => operand_reason(base).or_else(|| {
            indexing
                .components
                .iter()
                .find_map(|component| match component {
                    MirIndexComponent::Expr(operand) => operand_reason(operand),
                    MirIndexComponent::Colon | MirIndexComponent::End { .. } => None,
                })
        }),
        other => Some(format!("unsupported rvalue {other:?}")),
    }
}

fn call_reason(call: &MirCall) -> Option<String> {
    let Some(name) = call_name(&call.callee) else {
        return Some("dynamic or non-static function call is not supported".to_string());
    };
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
    call.args.iter().find_map(|arg| match arg {
        MirCallArg::Single(operand) => operand_reason(operand),
        MirCallArg::Expansion { .. } => {
            Some("argument expansion (`{:}`/`varargin`) is not supported".to_string())
        }
    })
}

fn stmt_reason(stmt: &MirStmt, varargout_local: Option<usize>) -> Option<String> {
    match &stmt.kind {
        MirStmtKind::Assign { place, value } => {
            // A `varargout{k}` target is a local cell element write; the value
            // still has to be lowerable. A plain `Local` target is the norm.
            if varargout_index(place, varargout_local).is_some()
                || matches!(place, MirPlace::Local(_))
            {
                rvalue_reason(value, varargout_local)
            } else {
                Some(format!("non-local assignment target {place:?}"))
            }
        }
        MirStmtKind::Expr(value) => rvalue_reason(value, varargout_local),
        // `varargout{k}` array-creation place mutations are folded away; any
        // other place mutation is unsupported.
        MirStmtKind::PlaceMutation(mutation) => {
            match varargout_index(&mutation.place, varargout_local) {
                Some(_) => None,
                None => Some(format!("unsupported place mutation {:?}", mutation.kind)),
            }
        }
        other => Some(format!("unsupported statement {other:?}")),
    }
}

fn terminator_reason(terminator: &MirTerminator) -> Option<String> {
    match &terminator.kind {
        MirTerminatorKind::Return(operands) => operands.iter().find_map(operand_reason),
        MirTerminatorKind::Goto(_) | MirTerminatorKind::Unreachable => None,
        MirTerminatorKind::Branch { cond, .. } => operand_reason(cond),
        MirTerminatorKind::Switch { discr, cases, .. } => operand_reason(discr)
            .or_else(|| cases.iter().find_map(|(case, _)| operand_reason(case))),
        MirTerminatorKind::For { iterable, .. } => match iterable {
            MirRvalue::Range { start, step, end } => operand_reason(start)
                .or_else(|| step.as_ref().and_then(operand_reason))
                .or_else(|| operand_reason(end)),
            other => Some(format!("unsupported for-loop iterable {other:?}")),
        },
        other => Some(format!("unsupported terminator {other:?}")),
    }
}

/// The codegen boundary: decide whether a body is statically lowerable.
///
/// This is the seam where `runmat-static-analysis` (type/shape inference,
/// definite assignment) will plug in to replace the operator whitelist and
/// cover the architecture's type-determined / shape-controlled / closed-world
/// criteria.
pub trait Classifier {
    fn classify(&self, body: &MirBody) -> Verdict;
}

/// The current classifier: a hand-written whitelist of the scalar-double
/// subset with structured control flow (`if`/`while`/`for`/`switch`), statically
/// shaped array literals, and the pure numeric built-ins in [`builtins`].
pub struct WhitelistClassifier;

impl Classifier for WhitelistClassifier {
    fn classify(&self, body: &MirBody) -> Verdict {
        let variadics = Variadics::compute(body);
        for block in &body.blocks {
            for stmt in &block.statements {
                if let Some(reason) = stmt_reason(stmt, variadics.varargout_local) {
                    return Verdict::Deferred { reason };
                }
            }
            if let Some(reason) = terminator_reason(&block.terminator) {
                return Verdict::Deferred { reason };
            }
        }

        // Shape analysis must fully resolve every local; anything dynamic
        // (`LocalTy::Dynamic` or a dynamic-shape array) crosses the static
        // boundary and is deferred to the runtime.
        for (local, ty) in infer_locals(body) {
            let is_dynamic = match ty {
                LocalTy::Dynamic => true,
                LocalTy::Array { shape } => shape.is_dynamic(),
                LocalTy::Scalar => false,
            };
            if is_dynamic {
                return Verdict::Deferred {
                    reason: format!("unresolved shape for local {local}"),
                };
            }
        }

        Verdict::Static
    }
}

/// Classify a MIR body with the default ([`WhitelistClassifier`]) classifier.
pub fn classify(body: &MirBody) -> Verdict {
    WhitelistClassifier.classify(body)
}

// --- Lightweight local type / shape inference ---------------------------------

/// Infer the static type of every local in `body`.
///
/// Parameters default to [`LocalTy::Scalar`] (the MVP has no
/// `runmat-static-analysis`, so array-typed parameters are out of scope and are
/// treated as scalars). Array shapes are only known for tensor `Aggregate`
/// literals; elementwise built-ins preserve their argument's shape; reductions
/// produce a scalar. Anything else resolves to [`LocalTy::Dynamic`].
pub fn infer_locals(body: &MirBody) -> HashMap<usize, LocalTy> {
    let variadics = Variadics::compute(body);
    let mut tys = HashMap::new();

    for local in &body.locals {
        if let Some(binding) = local.binding {
            if body.abi.fixed_inputs.contains(&binding) {
                tys.insert(local.id.0, LocalTy::Scalar);
            }
        }
    }

    for block in &body.blocks {
        for stmt in &block.statements {
            if let MirStmtKind::Assign { place, value } = &stmt.kind {
                let MirPlace::Local(target) = place else {
                    continue;
                };
                let ty = rvalue_ty(value, &tys, variadics.varargin_local);
                tys.insert(target.0, ty);
            }
        }
    }

    tys
}

fn operand_ty(operand: &MirOperand, tys: &HashMap<usize, LocalTy>) -> LocalTy {
    match operand {
        MirOperand::Local(id) => tys.get(&id.0).copied().unwrap_or(LocalTy::Scalar),
        MirOperand::Constant(_) => LocalTy::Scalar,
        _ => LocalTy::Dynamic,
    }
}

fn rvalue_ty(
    value: &MirRvalue,
    tys: &HashMap<usize, LocalTy>,
    varargin_local: Option<usize>,
) -> LocalTy {
    match value {
        MirRvalue::Use(operand) => operand_ty(operand, tys),
        MirRvalue::Unary(op, operand) => unary_ty(*op, operand_ty(operand, tys)),
        MirRvalue::Binary(lhs, op, rhs) => {
            binary_ty(operand_ty(lhs, tys), *op, operand_ty(rhs, tys))
        }
        MirRvalue::ShortCircuit { .. } => LocalTy::Scalar,
        MirRvalue::Aggregate {
            kind, rows, cols, ..
        } => match kind {
            runmat_mir::MirAggregateKind::Tensor => LocalTy::Array {
                shape: Shape::matrix(*rows, *cols),
            },
            runmat_mir::MirAggregateKind::Cell => LocalTy::Dynamic,
        },
        MirRvalue::Call(call) => call_ty(call, tys),
        MirRvalue::Index { base, indexing } => {
            // `varargin{k}` (constant `k`) resolves to a scalar extra argument.
            if let MirOperand::Local(id) = base {
                if varargin_local == Some(id.0)
                    && indexing.kind == IndexKind::Brace
                    && brace_index(indexing).is_some()
                {
                    return LocalTy::Scalar;
                }
            }
            index_ty(operand_ty(base, tys), indexing)
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
        (_, LocalTy::Dynamic) => LocalTy::Dynamic,
    }
}

fn binary_ty(lhs: LocalTy, op: OperatorKind, rhs: LocalTy) -> LocalTy {
    use LocalTy::*;

    match (lhs, rhs) {
        (Scalar, Scalar) => Scalar,
        (Dynamic, _) | (_, Dynamic) => Dynamic,
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
        // Two arrays: elementwise when the shapes agree; `*` is a matmul; `^`
        // (matrix ^ matrix) is not defined and is deferred.
        (Array { shape: lhs_shape }, Array { shape: rhs_shape }) => match op {
            OperatorKind::MatrixMultiply => matmul_ty(lhs_shape, rhs_shape),
            OperatorKind::MatrixPower => Dynamic,
            _ if lhs_shape == rhs_shape && is_elementwise(op) => Array { shape: lhs_shape },
            _ => Dynamic,
        },
    }
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

fn call_ty(call: &MirCall, tys: &HashMap<usize, LocalTy>) -> LocalTy {
    let Some(name) = call_name(&call.callee) else {
        return LocalTy::Dynamic;
    };
    let Some(builtin) = builtins::lookup(&name) else {
        return LocalTy::Dynamic;
    };

    let args: Vec<LocalTy> = call
        .args
        .iter()
        .map(|arg| match arg {
            MirCallArg::Single(operand) => operand_ty(operand, tys),
            MirCallArg::Expansion { .. } => LocalTy::Dynamic,
        })
        .collect();

    match builtin {
        // Elementwise unary: preserves the argument's shape.
        Builtin::Unary(_) => args.first().copied().unwrap_or(LocalTy::Dynamic),
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
        // Everything else in the supported set is scalar-valued.
        Builtin::Binary(_) | Builtin::Mod | Builtin::Sign => LocalTy::Scalar,
        // Constructors and reshape: array shapes from constant dim arguments.
        Builtin::Fill(_) | Builtin::Eye => match (constant_arg(call, 0), constant_arg(call, 1)) {
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

/// The constant `usize` value of the `index`-th argument of a call, if present.
fn constant_arg(call: &MirCall, index: usize) -> Option<usize> {
    match call.args.get(index) {
        Some(MirCallArg::Single(MirOperand::Constant(MirConstant::Number(text)))) => {
            text.trim().parse::<usize>().ok()
        }
        _ => None,
    }
}

/// The 1-based dimension argument of a reduction call, if present and constant.
fn reduction_dim(call: &MirCall) -> Option<usize> {
    if call.args.len() < 2 {
        return None;
    }
    match &call.args[1] {
        MirCallArg::Single(MirOperand::Constant(MirConstant::Number(text))) => {
            text.trim().parse::<usize>().ok()
        }
        _ => None,
    }
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

/// The type of an indexing expression `base(...)`.
fn index_ty(base: LocalTy, indexing: &MirIndexing) -> LocalTy {
    let LocalTy::Array { shape } = base else {
        return LocalTy::Dynamic;
    };
    let has_colon = indexing
        .components
        .iter()
        .any(|component| matches!(component, MirIndexComponent::Colon));
    if has_colon {
        // `A(:)` flattens to a column vector; other slices are deferred.
        if indexing.components.len() == 1
            && matches!(indexing.components[0], MirIndexComponent::Colon)
        {
            LocalTy::Array {
                shape: Shape::matrix(shape.numel(), 1),
            }
        } else {
            LocalTy::Dynamic
        }
    } else {
        // Constant or `end` subscript reads produce a scalar.
        LocalTy::Scalar
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
