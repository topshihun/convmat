//! End-to-end orchestration of the compilation pipeline.

use crate::backend::BackendKind;
use crate::error::Result;
use crate::frontend::SourceFile;
use crate::triage::Verdict;

/// Compile a source file: source -> runmat MIR -> triage -> MLIR -> emitc -> C.
pub fn compile(source: &SourceFile, backend: BackendKind) -> Result<String> {
    let mir = crate::frontend::parse_mir(source)?;

    // The codegen boundary: defer functions that are not statically lowerable.
    // The MVP has no runtime fallback, so a deferred function is an error.
    for body in mir.bodies.values() {
        if let Verdict::Deferred { reason } = crate::triage::classify(body) {
            // The codegen boundary defers this function to the runtime. The MVP
            // has no runtime shim, so the seam currently reports a hard error.
            crate::runtime::defer_to_runtime(&reason)?;
        }
    }

    // Keep the MLIR context alive across lowering and the pass manager so the
    // in-memory `Module` never needs a text round-trip.
    let context = crate::mir_to_mlir::create_context();
    let mut module = crate::mir_to_mlir::lower_to_module(&context, &mir)?;
    let emitc = crate::passes::convert_to_emitc(&context, &mut module)?;
    backend.emit(&emitc)
}
