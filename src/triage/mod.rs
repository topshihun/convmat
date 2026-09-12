//! Layer 2: the codegen boundary.
//!
//! Classifies each MIR body as statically lowerable (`Verdict::Static`) or
//! deferred to the runtime (`Verdict::Deferred`). This is a function-level
//! decision, so a program can mix compiled and runtime code.
//!
//! The current implementation is a hand-written whitelist of the scalar-double
//! subset. It is exposed behind [`Classifier`] so a
//! `runmat-static-analysis`-driven classifier (type/shape inference, definite
//! assignment) can replace it without changing callers.

use runmat_hir::OperatorKind;
use runmat_mir::{
    MirBody, MirConstant, MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminator,
    MirTerminatorKind,
};

/// A per-function classification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Statically lowerable: type-determined, closed-world, supported subset.
    Static,
    /// Deferred to the runtime with a human-readable reason.
    Deferred { reason: String },
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
        other => Some(format!("unsupported rvalue {other:?}")),
    }
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
/// subset with structured control flow (`if`/`while`/`for`/`switch`).
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

        Verdict::Static
    }
}

/// Classify a MIR body with the default ([`WhitelistClassifier`]) classifier.
pub fn classify(body: &MirBody) -> Verdict {
    WhitelistClassifier.classify(body)
}
