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
    assert_not_lowerable("unsupported_builtin", "unsupported builtin `fft`");
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

#[test]
fn anonymous_function_escape_rejected() {
    // Returning an anonymous-function handle needs the dynamic closure tier.
    assert_not_lowerable("anon_escape", "escapes as a return value");
}

#[test]
fn anonymous_function_array_argument_rejected() {
    // A scalar-parameter lambda called with an array would be truncated to the
    // first element, so it is rejected rather than miscompiled.
    assert_not_lowerable("anon_array", "arguments must be scalar");
}

#[test]
fn anonymous_function_array_capture_rejected() {
    // Capturing an array is not supported (scalar captures only).
    assert_not_lowerable("anon_array_capture", "captures must be scalar");
}

#[test]
fn anonymous_function_nested_definition_rejected() {
    // A handle defined inside a loop/conditional is deferred.
    assert_not_lowerable("anon_nested", "defined inside control flow");
}

#[test]
fn anonymous_function_array_result_rejected() {
    // A lambda returning an array needs the array-result ABI, which handle
    // calls do not express yet.
    assert_not_lowerable("anon_array_result", "must return a scalar");
}

#[test]
fn anonymous_function_copy_rejected() {
    // Copying a handle to another binding makes it escape as a value.
    assert_not_lowerable("anon_copy", "escapes");
}

#[test]
fn anonymous_function_arithmetic_rejected() {
    // Using a handle in a non-call expression is an escape.
    assert_not_lowerable("anon_arith", "escapes");
}

#[test]
fn anonymous_function_as_argument_rejected() {
    // Passing a handle as a call argument is an escape.
    assert_not_lowerable("anon_as_arg", "escapes");
}

#[test]
fn anonymous_function_reassign_rejected() {
    // A handle binding that is assigned twice has no single target.
    assert_not_lowerable("anon_reassign", "assigned more than once");
}

#[test]
fn anonymous_function_struct_field_rejected() {
    // Storing a handle in a struct field is out of the scalar-handle subset.
    assert_not_lowerable("anon_struct_field", "AnonymousFunction");
}

#[test]
fn anonymous_function_builtin_handle_rejected() {
    // A handle to a built-in (`@sin`) is not in the subset yet.
    assert_not_lowerable("anon_builtin_handle", "FunctionHandle");
}

#[test]
fn anonymous_function_immediate_call_rejected() {
    // Immediate invocation of an unbound literal is not in the subset.
    assert_not_lowerable("anon_immediate", "AnonymousFunction");
}
