//! Layer 1: read `.m` sources and drive the runmat frontend to produce MIR.

use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

/// A MATLAB/Octave source file.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: String,
    pub source: String,
}

impl SourceFile {
    /// Read a source file from disk.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)?;
        Ok(Self {
            path: path.display().to_string(),
            source,
        })
    }
}

/// Parse a source file through the runmat frontend into MIR.
///
/// Pipeline: `runmat_parser::parse` -> `runmat_hir::lower` ->
/// `runmat_mir::lowering::lower_assembly`.
pub fn parse_mir(source: &SourceFile) -> Result<runmat_mir::MirAssembly> {
    use std::collections::HashMap;

    let program =
        runmat_parser::parse(&source.source).map_err(|e| Error::Frontend(format!("parse: {e}")))?;
    let variables = HashMap::<String, usize>::new();
    let context = runmat_hir::LoweringContext::new(&variables);
    let lowering = runmat_hir::lower(&program, &context)
        .map_err(|e| Error::Frontend(format!("HIR lowering: {e}")))?;
    runmat_mir::lowering::lower_assembly(&lowering.assembly)
        .map_err(|e| Error::Frontend(format!("MIR lowering: {e}")))
}
