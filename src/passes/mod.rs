//! Optimization passes, layered by dialect and run at the pipeline stage where
//! that dialect is current.
//!
//! Optimization is deliberately **not** a single monolith. Each dialect gets its
//! own module of independent passes, built on pliron's pass framework
//! ([`Pass`]/[`Passes`]/[`NestedOpsPass`]/[`OpPass`]):
//!
//! - [`matlab`] — *semantic* optimizations on the high-level IR, run right after
//!   `hir_to_mlir`: constant folding, scalar-cell constant propagation,
//!   dead-branch elimination, dead-value elimination.
//! - [`emitc`] — *C-level* cleanups on the low-level IR, run right after
//!   `lowering`: dead write-only local cells (declarations + stores), dead value
//!   elimination.
//!
//! ```text
//! lower(HIR -> matlab) -> run_matlab_passes
//!   -> lowering(matlab -> emitc) -> run_emitc_passes
//!   -> emit_c(emitc -> C)
//! ```
//!
//! Each pipeline runs its passes per `builtin.func` and repeats the whole
//! sequence until a round reports no change (bounded by [`MAX_ROUNDS`]); a
//! rewrite that exposes another uses a later round rather than entangling the
//! passes.

pub mod emitc;
pub mod matlab;

use pliron::{
    basic_block::BasicBlock,
    builtin::ops::{FuncOp, ModuleOp},
    context::{Context, Ptr},
    irbuild::IRStatus,
    op::Op,
    operation::Operation,
    pass::{AnalysisManager, NestedOpsPass, OpPass, Pass, PassResult, Passes},
};

use crate::error::{Error, Result};

/// Safety cap on fixpoint rounds, in case a pass fails to converge.
const MAX_ROUNDS: usize = 16;

/// Run the `matlab`-dialect optimization pipeline on a `matlab` module.
pub fn run_matlab_passes(context: &mut Context, module: &ModuleOp) -> Result<()> {
    let mut passes = Passes::default();
    passes.add_pass(func_pass::<matlab::ConstantPropagationPass>());
    passes.add_pass(func_pass::<matlab::ConstantFoldPass>());
    passes.add_pass(func_pass::<matlab::DeadBranchPass>());
    passes.add_pass(func_pass::<matlab::DeadValuePass>());
    run_to_fixpoint(context, module, passes)
}

/// Run the `emitc`-dialect optimization pipeline on an `emitc` module.
pub fn run_emitc_passes(context: &mut Context, module: &ModuleOp) -> Result<()> {
    let mut passes = Passes::default();
    passes.add_pass(func_pass::<emitc::DeadCellPass>());
    passes.add_pass(func_pass::<emitc::DeadValuePass>());
    run_to_fixpoint(context, module, passes)
}

/// A pass that runs once per nested `builtin.func`.
fn func_pass<P: Pass + Default + 'static>() -> NestedOpsPass {
    NestedOpsPass::new(OpPass::<FuncOp, P>::default())
}

/// Run `passes` on `module`, repeating the sequence until a round is a no-op.
fn run_to_fixpoint(context: &mut Context, module: &ModuleOp, mut passes: Passes) -> Result<()> {
    let root = module.get_operation();
    let mut analyses = AnalysisManager::default();
    for _ in 0..MAX_ROUNDS {
        let result = passes
            .run(root, context, &mut analyses)
            .map_err(|_| Error::Backend("optimization pass failed".to_string()))?;
        if result.ir_changed == IRStatus::Unchanged {
            break;
        }
    }
    Ok(())
}

/// Build a [`PassResult`] reporting whether the IR changed.
pub(crate) fn changed_result(changed: bool) -> PassResult {
    let mut result = PassResult::default();
    result.ir_changed = IRStatus::from(changed);
    result
}

/// The entry block of the `builtin.func` behind `op`, or `None` if `op` is not a
/// function (the [`OpPass`] guard should make that unreachable).
pub(crate) fn func_entry_block(op: Ptr<Operation>, context: &Context) -> Option<Ptr<BasicBlock>> {
    Operation::get_op::<FuncOp>(op, context).map(|func| func.get_entry_block(context))
}
