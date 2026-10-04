//! Negative tests: sources that cross the codegen boundary must be rejected
//! with a clear `NotLowerable` error (the MVP has no runtime fallback). These
//! guard the deferral paths in `triage` and `hir_to_mlir` against silent
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
    assert_not_lowerable("unsupported_builtin", "unsupported builtin `svd`");
}

#[test]
fn varargin_variable_index_rejected() {
    // A non-constant `varargin{n}` index cannot be specialized.
    assert_not_lowerable("varargin_var", "unresolved shape");
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
fn graphics_as_value_rejected() {
    // `plot` has no numeric result; using it as a value defers (no shape).
    assert_not_lowerable("plot_value", "unresolved shape");
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
fn anonymous_function_returned_and_called_rejected() {
    // A handle cannot be both returned and called in the closure-value subset.
    assert_not_lowerable("anon_return_call", "returned and also called");
}

#[test]
fn anonymous_function_immediate_call_rejected() {
    // Immediate invocation of an unbound literal is not in the subset.
    assert_not_lowerable("anon_immediate", "AnonymousFunction");
}
