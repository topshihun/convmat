//! End-to-end orchestration of the compilation pipeline.

use pliron::context::Context;

use crate::backend::BackendKind;
use crate::error::{Error, Result};
use crate::frontend::SourceFile;
use crate::triage::Verdict;

/// Compile a source file: source -> runmat HIR -> triage -> `matlab` dialect ->
/// `emitc` dialect -> C.
pub fn compile(source: &SourceFile, backend: BackendKind) -> Result<String> {
    let hir = crate::frontend::parse_hir(source)?;

    // The codegen boundary: defer functions that are not statically lowerable.
    // The MVP has no runtime fallback, so a deferred function is an error.
    for function in &hir.functions {
        if let Verdict::Deferred { reason } = crate::triage::classify(function) {
            crate::runtime::defer_to_runtime(&reason)?;
        }
    }

    let mut context = Context::new();
    let module = crate::hir_to_mlir::lower_to_module(&mut context, &hir)?;
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
