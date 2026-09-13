//! Layer 4: run MLIR passes.
//!
//! `convert_to_emitc` lowers the core-dialect MLIR (`func`/`arith`/`scf`/
//! `memref`) to the `emitc` dialect using melior's pass manager. Before the
//! emitc conversion, `canonicalize` and `cse` clean up the lowered IR.

use melior::{
    ir::Module,
    pass::{conversion, transform, PassManager},
};

use crate::error::{Error, Result};

/// Convert core-dialect MLIR to `emitc`-dialect MLIR text via melior.
///
/// The generic `convert-to-emitc` pass lowers every dialect with an emitc
/// conversion interface (`arith`, `scf`, `memref`, `func`) in a single run,
/// which keeps the pipeline free of pass-ordering concerns. The
/// `reconcile-unrealized-casts` pass then folds away the remaining
/// `builtin.unrealized_conversion_cast` operations (e.g. `index` to
/// `!emitc.size_t`) so the result is printable.
pub fn convert_to_emitc(context: &melior::Context, module: &mut Module) -> Result<String> {
    let manager = PassManager::new(context);
    manager.add_pass(transform::create_canonicalizer());
    manager.add_pass(transform::create_cse());
    manager.add_pass(conversion::create_to_emit_c());
    manager.add_pass(conversion::create_reconcile_unrealized_casts());
    manager
        .run(module)
        .map_err(|error| Error::Backend(format!("emitc pass manager failed: {error}")))?;

    Ok(module.as_operation().to_string())
}
