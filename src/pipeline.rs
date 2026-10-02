//! End-to-end orchestration of the compilation pipeline.

use pliron::context::Context;

use crate::backend::BackendKind;
use crate::error::{Error, Result};
use crate::frontend::SourceFile;
use crate::triage::Verdict;

/// Compile a source file: source -> runmat MIR -> triage -> `matlab` dialect ->
/// `emitc` dialect -> C.
pub fn compile(source: &SourceFile, backend: BackendKind) -> Result<String> {
    let mir = crate::frontend::parse_mir(source)?;

    // The codegen boundary: defer functions that are not statically lowerable.
    // The MVP has no runtime fallback, so a deferred function is an error.
    for body in mir.bodies.values() {
        if let Verdict::Deferred { reason } = crate::triage::classify(body) {
            crate::runtime::defer_to_runtime(&reason)?;
        }
    }

    let mut context = Context::new();
    let module = crate::mir_to_mlir::lower_to_module(&mut context, &mir)?;
    let functions = crate::lowering::lower_module(&mut context, &module)?;
    let c = crate::emit_c::emit(&context, &functions)?;

    match backend {
        BackendKind::C => Ok(c),
        BackendKind::Llvm | BackendKind::Gpu => {
            Err(Error::NotImplemented(format!("{} backend", backend.name())))
        }
    }
}
