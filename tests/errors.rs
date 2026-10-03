//! Negative tests: sources that cross the codegen boundary must be rejected
//! with a clear `NotLowerable` error (the MVP has no runtime fallback). These
//! guard the deferral paths in `triage` and `mir_to_mlir` against silent
//! miscompilation.

use convmat::backend::BackendKind;
use convmat::error::Error;
use convmat::frontend::SourceFile;
use convmat::pipeline;

/// Compile a fixture, returning the error it produced (panicking on success —
/// every fixture here is expected to fail).
fn compile_err(name: &str) -> Error {
    let path = format!("{}/tests/fixtures/{name}.m", env!("CARGO_MANIFEST_DIR"));
    let source = SourceFile::read(path).expect("read fixture");
    match pipeline::compile(&source, BackendKind::C) {
        Ok(_) => panic!("`{name}` unexpectedly compiled to C"),
        Err(err) => err,
    }
}

/// Assert a fixture is rejected as `NotLowerable` with a message containing
/// `needle`.
fn assert_not_lowerable(name: &str, needle: &str) {
    let err = compile_err(name);
    let Error::NotLowerable(reason) = &err else {
        panic!("`{name}` failed with the wrong error kind: {err}");
    };
    assert!(
        reason.contains(needle),
        "`{name}` reason `{reason}` did not contain `{needle}`"
    );
}

#[test]
fn matrix_power_non_integer_exponent_rejected() {
    assert_not_lowerable("mpower_nonint", "non-negative integer");
}

#[test]
fn matrix_power_non_square_rejected() {
    assert_not_lowerable("mpower_nonsquare", "square matrix");
}

#[test]
fn unsupported_builtin_rejected() {
    assert_not_lowerable("unsupported_builtin", "unsupported builtin `sort`");
}

#[test]
fn varargin_variable_index_rejected() {
    // A non-constant `varargin{n}` index cannot be specialized.
    assert_not_lowerable("varargin_var", "unresolved shape");
}

#[test]
fn try_catch_rejected() {
    // C has no native exception handling; `try`/`catch` is deferred.
    assert_not_lowerable("try_catch", "try/catch");
}

#[test]
fn multi_assign_rejected() {
    // `[a, b] = f()` needs multi-value call support (user functions / tuple
    // returns), which is not implemented.
    assert_not_lowerable("multi_assign", "multi-assignment");
}

#[test]
fn arg_expansion_rejected() {
    // `varargin{:}` argument expansion needs a runtime cell ABI.
    assert_not_lowerable("arg_expansion", "expansion");
}

#[test]
fn logical_index_rejected() {
    // Logical indexing produces a runtime-sized result and is deferred.
    assert_not_lowerable("logical_index", "unresolved shape");
}

#[test]
fn cell_literal_rejected() {
    // Cell arrays are deferred: a heterogeneous cell cannot map to a fixed C
    // type without a runtime cell ABI (see docs/architecture.md §10.5).
    assert_not_lowerable("cell_literal", "cell array literals are not supported yet");
}
