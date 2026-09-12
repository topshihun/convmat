//! End-to-end codegen tests: each fixture is a runmat-supported `.m` snippet
//! that is compiled all the way to C and checked for the expected output.

use convmat::backend::BackendKind;
use convmat::frontend::SourceFile;
use convmat::pipeline;

/// Compile a fixture from `tests/fixtures/<name>.m` to C.
fn compile_fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}.m", env!("CARGO_MANIFEST_DIR"));
    let source = SourceFile::read(path).expect("read fixture");
    pipeline::compile(&source, BackendKind::C).expect("compile to C")
}

/// Lower a fixture to MLIR text (before the emitc conversion).
fn lower_fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}.m", env!("CARGO_MANIFEST_DIR"));
    let source = SourceFile::read(path).expect("read fixture");
    let mir = convmat::frontend::parse_mir(&source).expect("parse to MIR");
    convmat::mir_to_mlir::lower(&mir).expect("lower to MLIR")
}

/// The fixture used to prove the original scalar add path still works.
#[test]
fn fixture_uses_runmat_supported_syntax() {
    let path = format!("{}/tests/fixtures/add.m", env!("CARGO_MANIFEST_DIR"));
    let source = SourceFile::read(path).expect("read fixture");
    assert!(source.source.contains("function y = add(a, b)"));
    assert!(source.source.contains("y = a + b;"));
}

#[test]
fn codegen_add_to_c() {
    let c = compile_fixture("add");
    assert!(c.contains("double add("), "got:\n{c}");
    assert!(c.contains('+'), "got:\n{c}");
}

#[test]
fn codegen_arithmetic_operators() {
    // `*`, `.*`, `/`, `.\` all lower to scalar arithmetic.
    let c = compile_fixture("ops");
    assert!(c.contains("double ops("), "got:\n{c}");
    assert!(c.contains('*'), "got:\n{c}");
    assert!(c.contains('/'), "got:\n{c}");
}

#[test]
fn codegen_comparison_operators() {
    // All six comparisons compile; emitc lowers ordered float comparisons to a
    // NaN-guarded boolean expression.
    let c = compile_fixture("cmp");
    assert!(c.contains("double cmp("), "got:\n{c}");
    assert!(c.contains("bool"), "got:\n{c}");
}

#[test]
fn codegen_logical_and_short_circuit() {
    let c = compile_fixture("logic");
    assert!(c.contains("double logic("), "got:\n{c}");
}

#[test]
fn codegen_unary_operators() {
    // `-a`, `+a`, `~a`, and scalar transpose `a'`.
    let c = compile_fixture("unary");
    assert!(c.contains("double unary("), "got:\n{c}");
    assert!(c.contains('-'), "got:\n{c}");
}

#[test]
fn codegen_if_else() {
    let c = compile_fixture("max");
    assert!(c.contains("double max2("), "got:\n{c}");
    assert!(c.contains("if ("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_elseif_chain() {
    let c = compile_fixture("sign");
    assert!(c.contains("double sign_of("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_while_loop() {
    let c = compile_fixture("countdown");
    assert!(c.contains("double countdown("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
}

#[test]
fn codegen_for_loop() {
    let c = compile_fixture("sumto");
    assert!(c.contains("double sum_to("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
}

#[test]
fn codegen_switch() {
    let c = compile_fixture("grade");
    assert!(c.contains("double grade("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_nested_control_flow() {
    // `for` containing `if` containing `while`.
    let c = compile_fixture("nested");
    assert!(c.contains("double nested("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
}

#[test]
fn codegen_multiple_outputs() {
    let c = compile_fixture("polar");
    assert!(c.contains("std::tuple<double, double> polar("), "got:\n{c}");
}

#[test]
fn codegen_integer_literals() {
    let c = compile_fixture("intlit");
    assert!(c.contains("double intlit("), "got:\n{c}");
}

#[test]
fn codegen_for_loop_with_step() {
    // `for i = 1:2:n` iterates an explicit stride.
    let c = compile_fixture("forstep");
    assert!(c.contains("double odd_sum("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
}

#[test]
fn codegen_one_sided_if() {
    // `if` without an `else` lowers to an `scf.if` with an empty else region.
    let c = compile_fixture("ifonly");
    assert!(c.contains("double clamp_hi("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_void_function() {
    let c = compile_fixture("noop");
    assert!(c.contains("void noop("), "got:\n{c}");
}

#[test]
fn lower_control_flow_to_scf() {
    // The core-dialect MLIR uses `scf` for structured control flow and `memref`
    // for locals, which is what the emitc conversion consumes downstream.
    let mlir = lower_fixture("max");
    assert!(mlir.contains("scf.if"), "got:\n{mlir}");
    assert!(mlir.contains("memref.alloca"), "got:\n{mlir}");
}

// --- More complex codegen fixtures -------------------------------------------

#[test]
fn codegen_factorial_for_loop() {
    let c = compile_fixture("fact");
    assert!(c.contains("double fact("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
    assert!(c.contains('*'), "got:\n{c}");
}

#[test]
fn codegen_sum_of_squares() {
    let c = compile_fixture("sumsq");
    assert!(c.contains("double sumsq("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
    assert!(c.contains('*'), "got:\n{c}");
}

#[test]
fn codegen_parity_while_loop() {
    // Computes parity by repeatedly subtracting two, then comparing to zero.
    let c = compile_fixture("is_even");
    assert!(c.contains("double is_even("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
    assert!(c.contains("bool"), "got:\n{c}");
}

#[test]
fn codegen_absolute_difference() {
    // `abs` is a runtime builtin (not lowerable yet); express it via branches.
    let c = compile_fixture("abs_diff");
    assert!(c.contains("double abs_diff("), "got:\n{c}");
    assert!(c.contains("if ("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_clamp_two_sided() {
    let c = compile_fixture("clamp");
    assert!(c.contains("double clamp("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_polynomial_horner() {
    let c = compile_fixture("horner");
    assert!(c.contains("double horner("), "got:\n{c}");
    assert!(c.contains('*'), "got:\n{c}");
    assert!(c.contains('+'), "got:\n{c}");
}

#[test]
fn codegen_while_compound_condition() {
    let c = compile_fixture("bounded_sum");
    assert!(c.contains("double bounded_sum("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
}

#[test]
fn codegen_for_loop_descending() {
    let c = compile_fixture("count_down");
    assert!(c.contains("double count_down("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
    assert!(c.contains('-'), "got:\n{c}");
}

#[test]
fn codegen_switch_expression_discriminant() {
    // The switch discriminant is a computed expression, not just a variable.
    let c = compile_fixture("dispatch");
    assert!(c.contains("double dispatch("), "got:\n{c}");
    assert!(c.contains("if ("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_three_outputs() {
    let c = compile_fixture("stats3");
    assert!(
        c.contains("std::tuple<double, double, double> stats3("),
        "got:\n{c}"
    );
}

#[test]
fn codegen_literal_formats() {
    // Decimal, scientific notation, and negative literals all fold to doubles.
    let c = compile_fixture("litmix");
    assert!(c.contains("double litmix("), "got:\n{c}");
    assert!(c.contains('-'), "got:\n{c}");
}

#[test]
fn codegen_operator_precedence() {
    let c = compile_fixture("precedence");
    assert!(c.contains("double precedence("), "got:\n{c}");
    assert!(c.contains('*'), "got:\n{c}");
    assert!(c.contains('/'), "got:\n{c}");
}

#[test]
fn codegen_comparison_in_arithmetic() {
    // Comparison results are 0.0/1.0 doubles, usable in arithmetic.
    let c = compile_fixture("cmp_arith");
    assert!(c.contains("double cmp_arith("), "got:\n{c}");
    assert!(c.contains("bool"), "got:\n{c}");
}

#[test]
fn codegen_elseif_band_chain() {
    let c = compile_fixture("band");
    assert!(c.contains("double band("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_switch_in_for_loop() {
    let c = compile_fixture("nested_switch");
    assert!(c.contains("double nested_switch("), "got:\n{c}");
    assert!(c.contains("while ("), "got:\n{c}");
    assert!(c.contains("else"), "got:\n{c}");
}

#[test]
fn codegen_mixed_logical_operators() {
    // Elementwise `&`/`|`, short-circuit `&&`/`||`, and unary `~` together.
    let c = compile_fixture("logic_mix");
    assert!(c.contains("double logic_mix("), "got:\n{c}");
    assert!(c.contains("bool"), "got:\n{c}");
}

#[test]
fn codegen_unary_mix() {
    // Unary minus/plus/not and scalar transpose inside a larger expression.
    let c = compile_fixture("unary_mix");
    assert!(c.contains("double unary_mix("), "got:\n{c}");
    assert!(c.contains('-'), "got:\n{c}");
}
