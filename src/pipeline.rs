//! End-to-end orchestration of the compilation pipeline.

use pliron::context::Context;

use crate::backend::BackendKind;
use crate::error::{Error, Result};
use crate::frontend::SourceFile;
use crate::triage::{dispatch, FunctionPlan, Route};

/// Compile a source file: source -> runmat HIR -> triage -> `matlab` dialect ->
/// `emitc` dialect -> C.
pub fn compile(source: &SourceFile, backend: BackendKind) -> Result<String> {
    let hir = crate::frontend::parse_hir(source)?;

    // Compile-time dispatch: resolve every value's class and each function's
    // route once (see `triage::dispatch`). Statically-routed functions continue
    // to lowering; runtime-routed ones hit the runtime seam (an error until the
    // runtime matrix tier lands).
    let plans: Vec<FunctionPlan> = hir.functions.iter().map(dispatch).collect();
    for plan in &plans {
        if let Route::Runtime(reason) = &plan.route {
            crate::runtime::defer_to_runtime(reason)?;
        }
    }

    let mut context = Context::new();
    let module = crate::hir_to_mlir::lower_to_module_with_plans(&mut context, &hir, &plans)?;
    // Semantic optimizations on the `matlab` dialect (constant folding, cell
    // constant propagation, dead-branch elimination, dead-value elimination).
    crate::passes::run_matlab_passes(&mut context, &module)?;
    let lowered = crate::lowering::lower_module(&mut context, &module)?;
    // C-level cleanups on the `emitc` dialect (dead write-only cells, dead values).
    crate::passes::run_emitc_passes(&mut context, &lowered)?;
    let c = crate::emit_c::emit(&context, &lowered)?;

    match backend {
        BackendKind::C => Ok(c),
        BackendKind::Llvm | BackendKind::Gpu => {
            Err(Error::NotImplemented(format!("{} backend", backend.name())))
        }
    }
}
