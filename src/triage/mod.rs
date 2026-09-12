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

use runmat_hir::OperatorKind;
use runmat_mir::{
    MirBody, MirCall, MirCallArg, MirCallee, MirConstant, MirOperand, MirPlace, MirRvalue, MirStmt,
    MirStmtKind, MirTerminator, MirTerminatorKind,
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

/// The static type of a MIR local, as inferred by the boundary's lightweight
/// shape analysis (a stand-in for `runmat-static-analysis` in the MVP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalTy {
    /// A scalar `f64` value.
    Scalar,
    /// A statically-shaped numeric array with `n` elements (flattened).
    Array { n: usize },
    /// Not statically resolvable; the value must be deferred to the runtime.
    Dynamic,
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

fn rvalue_reason(value: &MirRvalue) -> Option<String> {
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
            .or_else(|| right_temps.iter().find_map(stmt_reason)),
        MirRvalue::Aggregate { kind, .. } => match kind {
            runmat_mir::MirAggregateKind::Tensor => None,
            runmat_mir::MirAggregateKind::Cell => {
                Some("cell array literals are not supported yet".to_string())
            }
        },
        MirRvalue::Call(call) => call_reason(call),
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

fn stmt_reason(stmt: &MirStmt) -> Option<String> {
    match &stmt.kind {
        MirStmtKind::Assign { place, value } => {
            if !matches!(place, MirPlace::Local(_)) {
                return Some(format!("non-local assignment target {place:?}"));
            }
            rvalue_reason(value)
        }
        MirStmtKind::Expr(value) => rvalue_reason(value),
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
        for block in &body.blocks {
            for stmt in &block.statements {
                if let Some(reason) = stmt_reason(stmt) {
                    return Verdict::Deferred { reason };
                }
            }
            if let Some(reason) = terminator_reason(&block.terminator) {
                return Verdict::Deferred { reason };
            }
        }

        // Shape analysis must fully resolve every local; anything `Dynamic`
        // crosses the static boundary and is deferred to the runtime.
        for (local, ty) in infer_locals(body) {
            if ty == LocalTy::Dynamic {
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
                let ty = rvalue_ty(value, &tys);
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

fn rvalue_ty(value: &MirRvalue, tys: &HashMap<usize, LocalTy>) -> LocalTy {
    match value {
        MirRvalue::Use(operand) => operand_ty(operand, tys),
        MirRvalue::Unary(_, operand) => match operand_ty(operand, tys) {
            LocalTy::Scalar => LocalTy::Scalar,
            _ => LocalTy::Dynamic,
        },
        MirRvalue::Binary(lhs, _, rhs) => {
            if operand_ty(lhs, tys) == LocalTy::Scalar && operand_ty(rhs, tys) == LocalTy::Scalar {
                LocalTy::Scalar
            } else {
                LocalTy::Dynamic
            }
        }
        MirRvalue::ShortCircuit { .. } => LocalTy::Scalar,
        MirRvalue::Aggregate {
            kind, rows, cols, ..
        } => match kind {
            runmat_mir::MirAggregateKind::Tensor => LocalTy::Array { n: rows * cols },
            runmat_mir::MirAggregateKind::Cell => LocalTy::Dynamic,
        },
        MirRvalue::Call(call) => call_ty(call, tys),
        _ => LocalTy::Dynamic,
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
        // `min`/`max`: reduction over one array, or elementwise over two
        // scalars — both produce a scalar in the supported subset.
        Builtin::MinMax(_) => LocalTy::Scalar,
        // Everything else in the supported set is scalar-valued.
        Builtin::Binary(_) | Builtin::Sign | Builtin::Reduce(_) => LocalTy::Scalar,
    }
}
