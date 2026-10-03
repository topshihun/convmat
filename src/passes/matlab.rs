//! `matlab`-dialect optimization passes.
//!
//! These operate on the *semantic* IR (dense arrays, scalar cells as one-element
//! arrays, structured control flow). Each pass is independent and idempotent;
//! [`crate::passes::run_matlab_passes`] runs them in sequence and repeats the
//! sequence to a fixpoint, so a rewrite that exposes another (e.g. folding a
//! condition to a constant, then eliminating the branch) is handled by a later
//! round instead of by entangling the passes.
//!
//! Correctness note: [`ConstantPropagationPass`] must not fold a cell value into
//! a loop region — a cell written in the loop body holds a different value on
//! later iterations. Only `if` regions inherit the enclosing cell constants.

use std::collections::{HashMap, HashSet};

use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    pass::{AnalysisManager, Pass, PassResult},
    r#type::Typed,
    region::Region,
    result::Result,
    value::Value,
};

use super::{changed_result, func_entry_block};
use crate::dialects::matlab::{
    self, ArrayType, BinOp, BinOpKind, CmpKind, CmpOp, ConstantOp, IfOp, LoadOp, SelectOp, StoreOp,
    YieldOp,
};

/// Propagate constant scalar-cell values into their loads (`store` → `load`
/// forwarding). A `matlab.load` from a cell whose last store was a constant is
/// replaced by that constant.
#[derive(Default)]
pub struct ConstantPropagationPass;

impl Pass for ConstantPropagationPass {
    fn name(&self) -> &str {
        "matlab-constant-propagation"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        context: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let Some(entry) = func_entry_block(op, context) else {
            return Ok(PassResult::default());
        };
        let mut changed = false;
        let mut cells: HashMap<Value, Option<f64>> = HashMap::new();
        let mut written: HashSet<Value> = HashSet::new();
        propagate_block(context, entry, &mut cells, &mut written, &mut changed);
        Ok(changed_result(changed))
    }
}

/// Fold `matlab.binop`/`matlab.cmp` with constant operands and `matlab.select`
/// with a constant condition (or two equal branches) into constants.
#[derive(Default)]
pub struct ConstantFoldPass;

impl Pass for ConstantFoldPass {
    fn name(&self) -> &str {
        "matlab-constant-fold"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        context: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let Some(entry) = func_entry_block(op, context) else {
            return Ok(PassResult::default());
        };
        let mut changed = false;
        fold_block(context, entry, &mut changed);
        Ok(changed_result(changed))
    }
}

/// Replace a `matlab.if` whose condition is a constant by the taken region's
/// body (dead-branch elimination).
#[derive(Default)]
pub struct DeadBranchPass;

impl Pass for DeadBranchPass {
    fn name(&self) -> &str {
        "matlab-dead-branch"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        context: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let Some(entry) = func_entry_block(op, context) else {
            return Ok(PassResult::default());
        };
        let mut changed = false;
        branch_block(context, entry, &mut changed);
        Ok(changed_result(changed))
    }
}

/// Remove side-effect-free single-value `matlab` ops whose result is unused.
#[derive(Default)]
pub struct DeadValuePass;

impl Pass for DeadValuePass {
    fn name(&self) -> &str {
        "matlab-dead-value"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        context: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let Some(entry) = func_entry_block(op, context) else {
            return Ok(PassResult::default());
        };
        let changed = dead_value_block(context, entry);
        Ok(changed_result(changed))
    }
}

// --- constant propagation -----------------------------------------------------

fn propagate_block(
    context: &mut Context,
    block: Ptr<BasicBlock>,
    cells: &mut HashMap<Value, Option<f64>>,
    written: &mut HashSet<Value>,
    changed: &mut bool,
) {
    let mut consts: HashMap<Value, f64> = HashMap::new();
    let mut erased: HashSet<Ptr<Operation>> = HashSet::new();
    let ops: Vec<Ptr<Operation>> = block.deref(context).iter(context).collect();
    for op in ops {
        if erased.contains(&op) {
            continue;
        }
        if let Some(constant) = Operation::get_op::<ConstantOp>(op, context) {
            let result = op.deref(context).get_result(0);
            consts.insert(result, constant.value(context));
            continue;
        }
        if Operation::is_op::<matlab::AllocaOp>(op, context) {
            let cell = op.deref(context).get_result(0);
            if is_scalar_cell(context, cell) {
                cells.insert(cell, None);
            }
            continue;
        }
        if Operation::is_op::<LoadOp>(op, context) {
            let cell = op.deref(context).get_operand(0);
            let index = op.deref(context).get_operand(1);
            if index_is_zero(&consts, index) {
                if let Some(Some(value)) = cells.get(&cell).copied() {
                    let folded = replace_with_constant(context, op, value);
                    consts.insert(folded, value);
                    *changed = true;
                    erased.insert(op);
                }
            }
            continue;
        }
        if Operation::is_op::<StoreOp>(op, context) {
            let cell = op.deref(context).get_operand(0);
            let index = op.deref(context).get_operand(1);
            let value = op.deref(context).get_operand(2);
            if is_scalar_cell(context, cell) && index_is_zero(&consts, index) {
                written.insert(cell);
                cells.insert(cell, consts.get(&value).copied());
            }
            continue;
        }
        for (region, inherit) in child_regions(context, op) {
            written.extend(propagate_region(context, region, cells, inherit, changed));
        }
    }
}

fn propagate_region(
    context: &mut Context,
    region: Ptr<Region>,
    outer_cells: &mut HashMap<Value, Option<f64>>,
    inherit: bool,
    changed: &mut bool,
) -> HashSet<Value> {
    let Some(block) = region.deref(context).get_entry_block() else {
        return HashSet::new();
    };
    let mut inner_cells = if inherit {
        outer_cells.clone()
    } else {
        HashMap::new()
    };
    let mut written: HashSet<Value> = HashSet::new();
    propagate_block(context, block, &mut inner_cells, &mut written, changed);
    for &cell in &written {
        outer_cells.insert(cell, None);
    }
    written
}

// --- constant folding ---------------------------------------------------------

fn fold_block(context: &mut Context, block: Ptr<BasicBlock>, changed: &mut bool) {
    let mut consts: HashMap<Value, f64> = HashMap::new();
    let mut erased: HashSet<Ptr<Operation>> = HashSet::new();
    let ops: Vec<Ptr<Operation>> = block.deref(context).iter(context).collect();
    for op in ops {
        if erased.contains(&op) {
            continue;
        }
        if let Some(constant) = Operation::get_op::<ConstantOp>(op, context) {
            let result = op.deref(context).get_result(0);
            consts.insert(result, constant.value(context));
            continue;
        }
        if let Some(binop) = Operation::get_op::<BinOp>(op, context) {
            let lhs = binop.lhs(context);
            let rhs = binop.rhs(context);
            if let (Some(&l), Some(&r)) = (consts.get(&lhs), consts.get(&rhs)) {
                if let Some(value) = fold_binop(binop.kind(context), l, r) {
                    let folded = replace_with_constant(context, op, value);
                    consts.insert(folded, value);
                    *changed = true;
                    erased.insert(op);
                }
            }
            continue;
        }
        if let Some(cmp) = Operation::get_op::<CmpOp>(op, context) {
            let lhs = op.deref(context).get_operand(0);
            let rhs = op.deref(context).get_operand(1);
            if let (Some(&l), Some(&r)) = (consts.get(&lhs), consts.get(&rhs)) {
                let value = fold_cmp(cmp.kind(context), l, r);
                let folded = replace_with_constant(context, op, value);
                consts.insert(folded, value);
                *changed = true;
                erased.insert(op);
            }
            continue;
        }
        if Operation::is_op::<SelectOp>(op, context) {
            let condition = op.deref(context).get_operand(0);
            let true_value = op.deref(context).get_operand(1);
            let false_value = op.deref(context).get_operand(2);
            let chosen = match consts.get(&condition).copied() {
                Some(c) => Some(if c != 0.0 { true_value } else { false_value }),
                None => match (consts.get(&true_value), consts.get(&false_value)) {
                    (Some(&t), Some(&f)) if t == f => Some(true_value),
                    _ => None,
                },
            };
            if let Some(chosen) = chosen {
                let result = op.deref(context).get_result(0);
                result.replace_all_uses_with(context, &chosen);
                Operation::erase(op, context);
                *changed = true;
                erased.insert(op);
            }
            continue;
        }
        for (region, _) in child_regions(context, op) {
            let child = region.deref(context).get_entry_block();
            if let Some(child) = child {
                fold_block(context, child, changed);
            }
        }
    }
}

// --- dead-branch elimination --------------------------------------------------

fn branch_block(context: &mut Context, block: Ptr<BasicBlock>, changed: &mut bool) {
    let mut consts: HashMap<Value, f64> = HashMap::new();
    let mut erased: HashSet<Ptr<Operation>> = HashSet::new();
    let ops: Vec<Ptr<Operation>> = block.deref(context).iter(context).collect();
    for op in ops {
        if erased.contains(&op) {
            continue;
        }
        if let Some(constant) = Operation::get_op::<ConstantOp>(op, context) {
            let result = op.deref(context).get_result(0);
            consts.insert(result, constant.value(context));
            continue;
        }
        // Optimize nested regions first.
        for (region, _) in child_regions(context, op) {
            let child = region.deref(context).get_entry_block();
            if let Some(child) = child {
                branch_block(context, child, changed);
            }
        }
        if let Some(if_op) = Operation::get_op::<IfOp>(op, context) {
            let condition = if_op.condition(context);
            if let Some(&c) = consts.get(&condition) {
                let region = if c != 0.0 {
                    if_op.then_region(context)
                } else {
                    if_op.else_region(context)
                };
                splice_region(context, region, op);
                Operation::erase(op, context);
                *changed = true;
                erased.insert(op);
            }
        }
    }
}

// --- dead-value elimination ---------------------------------------------------

fn dead_value_block(context: &mut Context, block: Ptr<BasicBlock>) -> bool {
    let mut changed = false;
    let mut deleted: HashSet<Ptr<Operation>> = HashSet::new();
    let ops: Vec<Ptr<Operation>> = block.deref(context).iter(context).collect();
    for op in ops {
        if deleted.contains(&op) {
            continue;
        }
        if is_pure_value_op(context, op) && !op.deref(context).has_use() {
            Operation::erase(op, context);
            deleted.insert(op);
            changed = true;
        }
    }
    let remaining: Vec<Ptr<Operation>> = block.deref(context).iter(context).collect();
    for op in remaining {
        let regions: Vec<Ptr<Region>> = op.deref(context).regions().collect();
        for region in regions {
            let child = region.deref(context).get_entry_block();
            if let Some(child) = child {
                changed |= dead_value_block(context, child);
            }
        }
    }
    changed
}

fn is_pure_value_op(context: &Context, op: Ptr<Operation>) -> bool {
    Operation::is_op::<ConstantOp>(op, context)
        || Operation::is_op::<BinOp>(op, context)
        || Operation::is_op::<CmpOp>(op, context)
        || Operation::is_op::<SelectOp>(op, context)
        || Operation::is_op::<LoadOp>(op, context)
}

// --- shared helpers -----------------------------------------------------------

/// The child regions of `op`, each paired with whether it may inherit the
/// enclosing block's cell constants (only single-execution `if` regions may).
fn child_regions(context: &Context, op: Ptr<Operation>) -> Vec<(Ptr<Region>, bool)> {
    if let Some(if_op) = Operation::get_op::<IfOp>(op, context) {
        return vec![
            (if_op.then_region(context), true),
            (if_op.else_region(context), true),
        ];
    }
    if let Some(while_op) = Operation::get_op::<matlab::WhileOp>(op, context) {
        return vec![
            (while_op.before_region(context), false),
            (while_op.after_region(context), false),
        ];
    }
    if let Some(for_op) = Operation::get_op::<matlab::ForOp>(op, context) {
        return vec![(for_op.body_region(context), false)];
    }
    if let Some(for_op) = Operation::get_op::<matlab::RangeForOp>(op, context) {
        return vec![(for_op.body_region(context), false)];
    }
    Vec::new()
}

/// Replace a single-result op with a fresh `matlab.constant`, rewiring all uses.
fn replace_with_constant(context: &mut Context, op: Ptr<Operation>, value: f64) -> Value {
    let constant = ConstantOp::new(context, value);
    let constant_op = constant.get_operation();
    constant_op.insert_before(context, op);
    let new_result = constant_op.deref(context).get_result(0);
    let old_result = op.deref(context).get_result(0);
    old_result.replace_all_uses_with(context, &new_result);
    Operation::erase(op, context);
    new_result
}

/// Move every non-terminator op of `region` in front of `mark`, preserving order.
fn splice_region(context: &Context, region: Ptr<Region>, mark: Ptr<Operation>) {
    let Some(block) = region.deref(context).get_entry_block() else {
        return;
    };
    let ops: Vec<Ptr<Operation>> = block.deref(context).iter(context).collect();
    for op in ops {
        if Operation::is_op::<YieldOp>(op, context) {
            continue;
        }
        op.unlink(context);
        op.insert_before(context, mark);
    }
}

fn index_is_zero(consts: &HashMap<Value, f64>, index: Value) -> bool {
    consts.get(&index).copied() == Some(0.0)
}

fn is_scalar_cell(context: &Context, value: Value) -> bool {
    value
        .get_type(context)
        .deref(context)
        .downcast_ref::<ArrayType>()
        .is_some_and(|array| array.numel() == 1)
}

/// Fold a scalar binary arithmetic op, or `None` if it must be left alone
/// (division by zero, or a non-finite result with no clean C literal).
fn fold_binop(kind: BinOpKind, lhs: f64, rhs: f64) -> Option<f64> {
    let value = match kind {
        BinOpKind::Add => lhs + rhs,
        BinOpKind::Sub => lhs - rhs,
        BinOpKind::Mul => lhs * rhs,
        BinOpKind::Div => {
            if rhs == 0.0 {
                return None;
            }
            lhs / rhs
        }
    };
    value.is_finite().then_some(value)
}

/// Fold a scalar comparison to `1.0` (true) or `0.0` (false).
fn fold_cmp(kind: CmpKind, lhs: f64, rhs: f64) -> f64 {
    let result = match kind {
        CmpKind::Eq => lhs == rhs,
        CmpKind::Ne => lhs != rhs,
        CmpKind::Lt => lhs < rhs,
        CmpKind::Le => lhs <= rhs,
        CmpKind::Gt => lhs > rhs,
        CmpKind::Ge => lhs >= rhs,
    };
    if result {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_binop_arithmetic() {
        assert_eq!(fold_binop(BinOpKind::Add, 2.0, 3.0), Some(5.0));
        assert_eq!(fold_binop(BinOpKind::Sub, 2.0, 3.0), Some(-1.0));
        assert_eq!(fold_binop(BinOpKind::Mul, 2.0, 3.0), Some(6.0));
        assert_eq!(fold_binop(BinOpKind::Div, 6.0, 3.0), Some(2.0));
    }

    #[test]
    fn fold_binop_leaves_unsafe_cases() {
        // Division by zero is not folded (IEEE Inf has no clean C literal here).
        assert_eq!(fold_binop(BinOpKind::Div, 1.0, 0.0), None);
        // A non-finite result (overflow) is not folded.
        assert_eq!(fold_binop(BinOpKind::Mul, f64::MAX, f64::MAX), None);
    }

    #[test]
    fn fold_cmp_all_kinds() {
        assert_eq!(fold_cmp(CmpKind::Eq, 2.0, 2.0), 1.0);
        assert_eq!(fold_cmp(CmpKind::Ne, 2.0, 2.0), 0.0);
        assert_eq!(fold_cmp(CmpKind::Lt, 2.0, 3.0), 1.0);
        assert_eq!(fold_cmp(CmpKind::Le, 3.0, 3.0), 1.0);
        assert_eq!(fold_cmp(CmpKind::Gt, 2.0, 3.0), 0.0);
        assert_eq!(fold_cmp(CmpKind::Ge, 3.0, 3.0), 1.0);
    }
}
