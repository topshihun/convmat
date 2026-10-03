//! `emitc`-dialect optimization passes.
//!
//! These operate on the *C-level* IR, after `matlab` → `emitc` lowering. They are
//! the cleanups that only make sense once cells are C declarations/assignments:
//!
//! - [`DeadCellPass`] removes a local cell (`emitc.declare`) that is only ever
//!   written — never loaded and never escaping (not passed to a call, not used
//!   by struct ops) — together with its `emitc.assign`s and `emitc.delete`.
//! - [`DeadValuePass`] removes pure single-value `emitc` ops whose result is
//!   unused (literals/arithmetic/comparisons/ternaries/loads).

use std::collections::{HashMap, HashSet};

use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    pass::{AnalysisManager, Pass, PassResult},
    region::Region,
    result::Result,
    value::Value,
};

use super::{changed_result, func_entry_block};
use crate::dialects::emitc;

/// Remove local cells that are written but never read (nor otherwise used).
#[derive(Default)]
pub struct DeadCellPass;

impl Pass for DeadCellPass {
    fn name(&self) -> &str {
        "emitc-dead-cell"
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
        let mut declares: Vec<Ptr<Operation>> = Vec::new();
        collect_declares(context, entry, &mut declares);
        if declares.is_empty() {
            return Ok(PassResult::default());
        }

        let cells: HashSet<Value> = declares
            .iter()
            .map(|op| op.deref(context).get_result(0))
            .collect();
        let mut usage: HashMap<Value, CellUsage> = cells
            .iter()
            .map(|cell| (*cell, CellUsage::default()))
            .collect();
        classify_uses(context, entry, &cells, &mut usage);

        let mut changed = false;
        for declare in declares {
            let cell = declare.deref(context).get_result(0);
            let Some(usage) = usage.get(&cell) else {
                continue;
            };
            if usage.loaded || usage.escaped {
                continue;
            }
            for &assign in &usage.assigns {
                Operation::erase(assign, context);
            }
            for &delete in &usage.deletes {
                Operation::erase(delete, context);
            }
            Operation::erase(declare, context);
            changed = true;
        }
        Ok(changed_result(changed))
    }
}

/// Remove pure single-value `emitc` ops whose result is unused.
#[derive(Default)]
pub struct DeadValuePass;

impl Pass for DeadValuePass {
    fn name(&self) -> &str {
        "emitc-dead-value"
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

#[derive(Default)]
struct CellUsage {
    /// The cell is read by an `emitc.load`.
    loaded: bool,
    /// The cell is used by something other than assign/delete/load (a call
    /// argument, a struct op, …), so it must be kept.
    escaped: bool,
    assigns: Vec<Ptr<Operation>>,
    deletes: Vec<Ptr<Operation>>,
}

fn collect_declares(context: &Context, block: Ptr<BasicBlock>, out: &mut Vec<Ptr<Operation>>) {
    for op in block.deref(context).iter(context) {
        if Operation::is_op::<emitc::DeclareOp>(op, context) {
            out.push(op);
        }
        let regions: Vec<Ptr<Region>> = op.deref(context).regions().collect();
        for region in regions {
            let child = region.deref(context).get_entry_block();
            if let Some(child) = child {
                collect_declares(context, child, out);
            }
        }
    }
}

fn classify_uses(
    context: &Context,
    block: Ptr<BasicBlock>,
    cells: &HashSet<Value>,
    usage: &mut HashMap<Value, CellUsage>,
) {
    for op in block.deref(context).iter(context) {
        let num_operands = op.deref(context).get_num_operands();
        for i in 0..num_operands {
            let operand = op.deref(context).get_operand(i);
            if !cells.contains(&operand) {
                continue;
            }
            let entry = usage.get_mut(&operand).expect("usage entry for a cell");
            if Operation::is_op::<emitc::AssignOp>(op, context) && i == 0 {
                entry.assigns.push(op);
            } else if Operation::is_op::<emitc::DeleteOp>(op, context) && i == 0 {
                entry.deletes.push(op);
            } else if Operation::is_op::<emitc::LoadOp>(op, context) && i == 0 {
                entry.loaded = true;
            } else {
                entry.escaped = true;
            }
        }
        let regions: Vec<Ptr<Region>> = op.deref(context).regions().collect();
        for region in regions {
            let child = region.deref(context).get_entry_block();
            if let Some(child) = child {
                classify_uses(context, child, cells, usage);
            }
        }
    }
}

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
    Operation::is_op::<emitc::LiteralOp>(op, context)
        || Operation::is_op::<emitc::BinOp>(op, context)
        || Operation::is_op::<emitc::CmpOp>(op, context)
        || Operation::is_op::<emitc::TernaryOp>(op, context)
        || Operation::is_op::<emitc::LoadOp>(op, context)
}
