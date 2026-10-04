//! Compile-and-run tests.
//!
//! `emit_c.rs` only checks that the generated code *contains* the expected
//! fragments; it cannot tell whether the generated code actually compiles or
//! behaves correctly. These tests close that gap: each fixture is compiled all
//! the way to C++, a tiny `main` driver is generated, the whole thing is
//! compiled with a system C++ compiler (`g++`/`clang++`, or `$CXX`), and the
//! program's stdout is compared against the expected value.
//!
//! The C backend emits C++ (it uses `<cmath>`, `<tuple>` and `std::tuple<>`),
//! so we need a C++ compiler. A test fails loudly if none is available so that
//! a missing compiler is never silently skipped.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use convmat::backend::BackendKind;
use convmat::frontend::SourceFile;
use convmat::pipeline;

/// Standard headers included in every generated `main.cpp`, so drivers can use
/// `printf` and forward-declare tuple-returning functions.
const PREAMBLE: &str = "\
#include <cstdio>
#include <cstdint>
#include <tuple>
#include <cmath>
";

/// Tolerance for floating-point runtime comparisons (generous enough to absorb
/// platform libm rounding differences, tight enough to catch real bugs).
const TOLERANCE: f64 = 1e-9;

/// Locate a usable C++ compiler: honour `$CXX`, otherwise probe the usual
/// suspects. The result is cached in a `OnceLock`; the returned `&str` lives
/// for the duration of the process.
fn find_compiler() -> Option<&'static str> {
    static COMPILER: OnceLock<Option<String>> = OnceLock::new();
    COMPILER
        .get_or_init(|| {
            if let Ok(cxx) = std::env::var("CXX") {
                let cxx = cxx.trim().to_string();
                if !cxx.is_empty() {
                    return Some(cxx);
                }
            }
            for name in ["g++", "clang++", "c++"] {
                if Command::new(name).arg("--version").output().is_ok() {
                    return Some(name.to_string());
                }
            }
            None
        })
        .as_deref()
}

/// Create a fresh, unique scratch directory for one test.
fn scratch_dir(fixture: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("convmat-{}-{fixture}-{n}", std::process::id()));
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Compile a fixture from `tests/fixtures/<name>.m` to C++.
fn compile_fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}.m", env!("CARGO_MANIFEST_DIR"));
    let source = SourceFile::read(path).expect("read fixture");
    pipeline::compile(&source, BackendKind::C).expect("compile to C++")
}

/// Compile the generated code together with a `main` driver and run it,
/// returning the trimmed stdout.
fn run_program(fixture: &str, decls: &str, body: &str) -> String {
    let cpp = compile_fixture(fixture);

    let compiler = find_compiler().unwrap_or_else(|| {
        panic!("no C++ compiler found (set `CXX`); compile-and-run tests need one")
    });

    let dir = scratch_dir(fixture);
    let gen_path = dir.join("gen.cpp");
    let main_path = dir.join("main.cpp");
    let bin_path = dir.join("prog");

    fs::write(&gen_path, &cpp).expect("write generated code");
    fs::write(
        &main_path,
        format!("{PREAMBLE}{decls}\nint main() {{\n{body}\n    return 0;\n}}\n"),
    )
    .expect("write main driver");

    let compile = Command::new(compiler)
        .arg("-std=c++17")
        .arg(&gen_path)
        .arg(&main_path)
        .arg("-o")
        .arg(&bin_path)
        .arg("-lm")
        .output()
        .expect("spawn compiler");

    if !compile.status.success() {
        panic!(
            "compiling `{fixture}` with `{compiler}` failed:\n\
             --- generated C++ ---\n{cpp}\n\
             --- compiler stderr ---\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );
    }

    let run = Command::new(&bin_path).output().expect("run program");
    let _ = fs::remove_dir_all(&dir);

    assert!(
        run.status.success(),
        "`{fixture}` program exited with {:?}\nstderr: {}",
        run.status.code(),
        String::from_utf8_lossy(&run.stderr)
    );

    String::from_utf8_lossy(&run.stdout).trim().to_string()
}

/// Assert that a fixture's program prints exactly `expected` (whitespace
/// trimmed). The driver should print results with `%g` (6 significant digits),
/// which is exact for the integer / short-decimal results covered here.
fn run_exact(fixture: &str, decls: &str, body: &str, expected: &str) {
    let got = run_program(fixture, decls, body);
    assert_eq!(got, expected, "`{fixture}` produced wrong output");
}

/// Assert that a fixture's program prints the values in `expected`, one per
/// line, matching within `TOLERANCE`. The driver should print each value with
/// `%.17g` on its own line.
fn run_close(fixture: &str, decls: &str, body: &str, expected: &[f64]) {
    let stdout = run_program(fixture, decls, body);
    let got: Vec<f64> = stdout
        .lines()
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("`{fixture}` printed non-numeric output: {line:?}"))
        })
        .collect();

    assert_eq!(
        got.len(),
        expected.len(),
        "`{fixture}` printed {} values, expected {}",
        got.len(),
        expected.len()
    );
    for (i, (got, want)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= TOLERANCE * (1.0 + want.abs()),
            "`{fixture}` value {i}: got {got}, expected {want}"
        );
    }
}

// --- Scalar arithmetic & literals ---------------------------------------------

#[test]
fn run_add() {
    run_exact(
        "add",
        "double add(double, double);",
        "printf(\"%g\\n\", add(2.0, 3.0));",
        "5",
    );
}

#[test]
fn run_ops() {
    // ops(4, 2): 4*2 + 4.*2 + 4/2 + 2.\4 + (4==2) + (4<=2) + (4 & 2)
    //          =   8  +   8  +  2  + 0.5  +   0    +   0    +   1   = 19.5
    run_exact(
        "ops",
        "double ops(double, double);",
        "printf(\"%g\\n\", ops(4.0, 2.0));",
        "19.5",
    );
}

#[test]
fn run_intlit() {
    run_exact(
        "intlit",
        "double intlit();",
        "printf(\"%g\\n\", intlit());",
        "47",
    );
}

#[test]
fn run_litmix() {
    run_exact(
        "litmix",
        "double litmix();",
        "printf(\"%g\\n\", litmix());",
        "998.025",
    );
}

#[test]
fn run_precedence() {
    run_exact(
        "precedence",
        "double precedence(double, double, double, double);",
        "printf(\"%g\\n\", precedence(1.0, 1.0, 3.0, 1.0));",
        "1",
    );
}

// --- Comparisons & logical operators ------------------------------------------

#[test]
fn run_cmp() {
    // cmp(2, 2): == 1, ~= 0, < 0, <= 1, > 0, >= 1 -> 3
    run_exact(
        "cmp",
        "double cmp(double, double);",
        "printf(\"%g\\n\", cmp(2.0, 2.0));",
        "3",
    );
}

#[test]
fn run_cmp_arith() {
    // Exactly one of (<, >, ==, >=, <=, ~=) is true for any distinct pair,
    // and three are true when a == b; both cases sum to 3.
    run_exact(
        "cmp_arith",
        "double cmp_arith(double, double);",
        "printf(\"%g\\n\", cmp_arith(2.0, 3.0));",
        "3",
    );
}

#[test]
fn run_logic() {
    run_exact(
        "logic",
        "double logic(double, double);",
        "printf(\"%g\\n\", logic(1.0, 0.0));",
        "0",
    );
}

#[test]
fn run_logic_mix() {
    run_exact(
        "logic_mix",
        "double logic_mix(double, double, double);",
        "printf(\"%g\\n\", logic_mix(1.0, 1.0, 1.0));",
        "2",
    );
}

// --- Unary operators ----------------------------------------------------------

#[test]
fn run_unary() {
    // unary(2): -2 + 2 + ~2(0) + 2' (2) = 2
    run_exact(
        "unary",
        "double unary(double);",
        "printf(\"%g\\n\", unary(2.0));",
        "2",
    );
}

#[test]
fn run_unary_mix() {
    // unary_mix(2, 3): -(5) + 2 - ~(0)=1 + 2 = -2
    run_exact(
        "unary_mix",
        "double unary_mix(double, double);",
        "printf(\"%g\\n\", unary_mix(2.0, 3.0));",
        "-2",
    );
}

// --- Control flow -------------------------------------------------------------

#[test]
fn run_max2() {
    run_exact(
        "max",
        "double max2(double, double);",
        "printf(\"%g\\n\", max2(2.0, 5.0));",
        "5",
    );
}

#[test]
fn run_sign_of() {
    run_exact(
        "sign",
        "double sign_of(double);",
        "printf(\"%g\\n\", sign_of(-5.0));",
        "-1",
    );
}

#[test]
fn run_grade() {
    run_exact(
        "grade",
        "double grade(double);",
        "printf(\"%g\\n\", grade(2.0));",
        "20",
    );
}

#[test]
fn run_dispatch() {
    run_exact(
        "dispatch",
        "double dispatch(double);",
        "printf(\"%g\\n\", dispatch(2.0));",
        "20",
    );
}

#[test]
fn run_band() {
    run_exact(
        "band",
        "double band(double);",
        "printf(\"%g\\n\", band(7.0));",
        "50",
    );
}

#[test]
fn run_clamp() {
    run_exact(
        "clamp",
        "double clamp(double, double, double);",
        "printf(\"%g\\n\", clamp(5.0, 0.0, 10.0));",
        "5",
    );
}

#[test]
fn run_clamp_hi() {
    run_exact(
        "ifonly",
        "double clamp_hi(double, double);",
        "printf(\"%g\\n\", clamp_hi(20.0, 10.0));",
        "10",
    );
}

#[test]
fn run_abs_diff() {
    run_exact(
        "abs_diff",
        "double abs_diff(double, double);",
        "printf(\"%g\\n\", abs_diff(3.0, 8.0));",
        "5",
    );
}

// --- Loops --------------------------------------------------------------------

#[test]
fn run_fact() {
    run_exact(
        "fact",
        "double fact(double);",
        "printf(\"%g\\n\", fact(5.0));",
        "120",
    );
}

#[test]
fn run_sum_to() {
    run_exact(
        "sumto",
        "double sum_to(double);",
        "printf(\"%g\\n\", sum_to(10.0));",
        "55",
    );
}

#[test]
fn run_odd_sum() {
    run_exact(
        "forstep",
        "double odd_sum(double);",
        "printf(\"%g\\n\", odd_sum(7.0));",
        "16",
    );
}

#[test]
fn run_countdown() {
    run_exact(
        "countdown",
        "double countdown(double);",
        "printf(\"%g\\n\", countdown(5.0));",
        "5",
    );
}

#[test]
fn run_count_down() {
    run_exact(
        "count_down",
        "double count_down(double);",
        "printf(\"%g\\n\", count_down(4.0));",
        "10",
    );
}

#[test]
fn run_sumsq() {
    run_exact(
        "sumsq",
        "double sumsq(double);",
        "printf(\"%g\\n\", sumsq(4.0));",
        "30",
    );
}

#[test]
fn run_is_even() {
    run_exact(
        "is_even",
        "double is_even(double);",
        "printf(\"%g\\n\", is_even(6.0));",
        "1",
    );
}

#[test]
fn run_bounded_sum() {
    run_exact(
        "bounded_sum",
        "double bounded_sum(double, double);",
        "printf(\"%g\\n\", bounded_sum(5.0, 10.0));",
        "45",
    );
}

// --- Nested control flow ------------------------------------------------------

#[test]
fn run_nested() {
    run_exact(
        "nested",
        "double nested(double);",
        "printf(\"%g\\n\", nested(5.0));",
        "12",
    );
}

#[test]
fn run_nested_switch() {
    run_exact(
        "nested_switch",
        "double nested_switch(double);",
        "printf(\"%g\\n\", nested_switch(4.0));",
        "14",
    );
}

// --- Multiple outputs ---------------------------------------------------------

#[test]
fn run_polar() {
    run_exact(
        "polar",
        "std::tuple<double, double> polar(double, double);",
        "auto [r, t] = polar(3.0, 4.0);\n    printf(\"%g %g\\n\", r, t);",
        "25 0.75",
    );
}

#[test]
fn run_stats3() {
    run_exact(
        "stats3",
        "std::tuple<double, double, double> stats3(double, double, double);",
        "auto [mn, prod, diff] = stats3(1.0, 2.0, 3.0);\n    printf(\"%g %g %g\\n\", mn, prod, diff);",
        "6 6 -4",
    );
}

#[test]
fn run_noop() {
    run_exact(
        "noop",
        "void noop(double);",
        "noop(1.0);\n    printf(\"ok\\n\");",
        "ok",
    );
}

// --- Built-in functions -------------------------------------------------------

#[test]
fn run_trig() {
    let x = 0.5f64;
    run_close(
        "trig",
        "double trig(double);",
        "printf(\"%.17g\\n\", trig(0.5));",
        &[x.sin() + x.cos() + x.tan()],
    );
}

#[test]
fn run_mathfns() {
    let x = 2.0f64;
    run_close(
        "mathfns",
        "double mathfns(double);",
        "printf(\"%.17g\\n\", mathfns(2.0));",
        &[x.sqrt() + x.exp() + x.ln() + x.abs() + x.floor() + x.ceil() + x.round()],
    );
}

#[test]
fn run_sign_builtin() {
    run_exact(
        "sign_builtin",
        "double sign_builtin(double);",
        "printf(\"%g\\n\", sign_builtin(-5.0));",
        "-1",
    );
}

#[test]
fn run_binary_math() {
    // `mod(-5, 3) = 1` (floor semantics, sign of the divisor) and
    // `rem(-5, 3) = -2` (trunc semantics, sign of the dividend): this pair
    // distinguishes `mod`/`rem` from plain `fmod`/`remainder`.
    let a = -5.0f64;
    let b = 3.0f64;
    let matlab_mod = a - b * (a / b).floor();
    let matlab_rem = a - b * (a / b).trunc();
    run_close(
        "binary_math",
        "double binary_math(double, double);",
        "printf(\"%.17g\\n\", binary_math(-5.0, 3.0));",
        &[a.powf(b) + b.atan2(a) + a.hypot(b) + matlab_mod + matlab_rem + a.min(b) + a.max(b)],
    );
}

#[test]
fn run_horner() {
    run_exact(
        "horner",
        "double horner(double, double, double, double);",
        "printf(\"%g\\n\", horner(2.0, 1.0, 2.0, 3.0));",
        "11",
    );
}

// --- Array built-ins & reductions ---------------------------------------------

#[test]
fn run_sin_array() {
    run_close(
        "sin_array",
        "void sin_array(double*);",
        "double o[3]; sin_array(o);\n    printf(\"%.17g\\n%.17g\\n%.17g\\n\", o[0], o[1], o[2]);",
        &[1.0f64.sin(), 3.0f64.sin(), 4.0f64.sin()],
    );
}

#[test]
fn run_abs_array() {
    run_exact(
        "abs_array",
        "void abs_array(double*);",
        "double o[3]; abs_array(o);\n    printf(\"%g %g %g\\n\", o[0], o[1], o[2]);",
        "1 2 3",
    );
}

#[test]
fn run_nested_elementwise() {
    run_close(
        "nested_elementwise",
        "void nested_elementwise(double*);",
        "double o[2]; nested_elementwise(o);\n    printf(\"%.17g\\n%.17g\\n\", o[0], o[1]);",
        &[(0.5f64.cos()).sin(), (1.0f64.cos()).sin()],
    );
}

#[test]
fn run_reduce_sum() {
    run_exact(
        "reduce_sum",
        "double reduce_sum();",
        "printf(\"%g\\n\", reduce_sum());",
        "10",
    );
}

#[test]
fn run_reduce_prod() {
    run_exact(
        "reduce_prod",
        "double reduce_prod();",
        "printf(\"%g\\n\", reduce_prod());",
        "24",
    );
}

#[test]
fn run_reduce_minmax() {
    run_exact(
        "reduce_minmax",
        "double reduce_minmax();",
        "printf(\"%g\\n\", reduce_minmax());",
        "4",
    );
}

// --- Matrix / vector literals -------------------------------------------------

#[test]
fn run_matrix2d() {
    run_exact(
        "matrix2d",
        "void matrix2d(double*);",
        "double o[4]; matrix2d(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "1 3 2 4",
    );
}

#[test]
fn run_colvec() {
    run_exact(
        "colvec",
        "void colvec(double*);",
        "double o[3]; colvec(o);\n    printf(\"%g %g %g\\n\", o[0], o[1], o[2]);",
        "1 2 3",
    );
}

#[test]
fn run_rowvec() {
    run_exact(
        "rowvec",
        "void rowvec(double*);",
        "double o[3]; rowvec(o);\n    printf(\"%g %g %g\\n\", o[0], o[1], o[2]);",
        "1 2 3",
    );
}

#[test]
fn run_sort_vec() {
    // `sort([3, 1, 2])` sorts a vector ascending via the `convmat_sort` helper.
    run_exact(
        "sort_vec",
        "void sort_vec(double*);",
        "double o[3]; sort_vec(o);\n    printf(\"%g %g %g\\n\", o[0], o[1], o[2]);",
        "1 2 3",
    );
}

#[test]
fn run_matrix_sort() {
    // `sort([3 1; 2 4])` sorts each column independently via `convmat_sort_cols`:
    // column 1 [3;2] -> [2;3], column 2 [1;4] -> [1;4] (column-major [2 3 1 4]).
    run_exact(
        "sort_matrix",
        "void sort_matrix(double*);",
        "double o[4]; sort_matrix(o);\n    \
         printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "2 3 1 4",
    );
}

#[test]
fn run_linalg_det3() {
    // det of a 3x3 whose first pivot is zero: the `convmat_det` helper must
    // swap rows. det([0 2 1; 3 1 4; 1 5 2]) == 10.
    run_exact(
        "linalg_det3",
        "double linalg_det3(void);",
        "printf(\"%g\\n\", linalg_det3());",
        "10",
    );
}

#[test]
fn run_linalg_inv3() {
    // inv([4 7 2; 3 6 1; 2 5 3]) == (1/9) * [13 -11 -5; -7 8 2; 3 -6 3]
    // (column-major output).
    run_close(
        "linalg_inv3",
        "void linalg_inv3(double*);",
        "double o[9]; linalg_inv3(o);\n    \
         for (int i = 0; i < 9; i++) printf(\"%.17g\\n\", o[i]);",
        &[
            13.0 / 9.0,
            -7.0 / 9.0,
            1.0 / 3.0,
            -11.0 / 9.0,
            8.0 / 9.0,
            -2.0 / 3.0,
            -5.0 / 9.0,
            2.0 / 9.0,
            1.0 / 3.0,
        ],
    );
}

#[test]
fn run_linalg_solve3() {
    // [2 1 0; 1 3 1; 0 1 2] \ [1; 2; 3] == [0.5; 0; 1.5].
    run_close(
        "linalg_solve3",
        "void linalg_solve3(double*);",
        "double o[3]; linalg_solve3(o);\n    \
         for (int i = 0; i < 3; i++) printf(\"%.17g\\n\", o[i]);",
        &[0.5, 0.0, 1.5],
    );
}

#[test]
fn run_array_param_sum() {
    // A dynamic-shape array parameter lowers to `(double* data, double n)`; the
    // reduction goes through the `convmat_sum` runtime helper.
    run_exact(
        "array_sum",
        "double array_sum(double*, double);",
        "double a[3] = {1.0, 2.0, 3.0};\n    printf(\"%g\\n\", array_sum(a, 3));",
        "6",
    );
}

#[test]
fn run_array_param_numel() {
    // `numel`/`length` of a dynamic-shape array parameter is its runtime length.
    run_exact(
        "array_numel",
        "double array_numel(double*, double);",
        "double a[4] = {0.0, 0.0, 0.0, 0.0};\n    printf(\"%g\\n\", array_numel(a, 4));",
        "4",
    );
}

#[test]
fn run_array_size_descriptor() {
    // `size(A, ...)` forces the `(data, rows, cols)` shape-descriptor ABI.
    run_exact(
        "array_size",
        "double array_size(double*, double, double);",
        "double a[6] = {1, 2, 3, 4, 5, 6};\n    \
         printf(\"%g\\n\", array_size(a, 2.0, 3.0));",
        "2",
    );
}

#[test]
fn run_array_size_col() {
    run_exact(
        "array_size_col",
        "double array_size_col(double*, double, double);",
        "double a[6] = {1, 2, 3, 4, 5, 6};\n    \
         printf(\"%g\\n\", array_size_col(a, 2.0, 3.0));",
        "3",
    );
}

#[test]
fn run_array_size_all() {
    // `size(A)` returns `[rows cols]`; the fixture sums them: 2 + 3 == 5.
    run_exact(
        "array_size_all",
        "double array_size_all(double*, double, double);",
        "double a[6] = {1, 2, 3, 4, 5, 6};\n    \
         printf(\"%g\\n\", array_size_all(a, 2.0, 3.0));",
        "5",
    );
}

#[test]
fn run_array_param_index() {
    // A constant-index read on a dynamic array parameter is `data[i-1]`.
    run_exact(
        "array_param_index",
        "double array_param_index(double*, double);",
        "double a[3] = {7.0, 8.0, 9.0};\n    printf(\"%g\\n\", array_param_index(a, 3));",
        "7",
    );
}

#[test]
fn run_array_get() {
    // A runtime index `i` into a dynamic array parameter is `data[i-1]`.
    run_exact(
        "array_get",
        "double array_get(double*, double, double);",
        "double a[3] = {7.0, 8.0, 9.0};\n    printf(\"%g\\n\", array_get(a, 3, 2));",
        "8",
    );
}

#[test]
fn run_array_last() {
    // `A(end)` on a dynamic array parameter is `data[n-1]`.
    run_exact(
        "array_last",
        "double array_last(double*, double);",
        "double a[3] = {7.0, 8.0, 9.0};\n    printf(\"%g\\n\", array_last(a, 3));",
        "9",
    );
}

#[test]
fn run_array_loop_sum() {
    // A `for i = 1:numel(A)` loop with a runtime bound, indexing `A(i)`.
    run_exact(
        "array_loop_sum",
        "double array_loop_sum(double*, double);",
        "double a[3] = {7.0, 8.0, 9.0};\n    printf(\"%g\\n\", array_loop_sum(a, 3));",
        "24",
    );
}

#[test]
fn run_array_copy() {
    // A dynamic array output: the caller passes an out-buffer and an out-length
    // cell; the callee fills the buffer and reports the actual length.
    run_exact(
        "array_copy",
        "void array_copy(double*, double, double*, double*);",
        "double a[3] = {1.0, 2.0, 3.0};\n    double y[3];\n    double yn;\n    array_copy(a, 3, y, &yn);\n    printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], yn);",
        "1 2 3 3",
    );
}

#[test]
fn run_array_scale() {
    // Scalar broadcast into a dynamic array output (`y = 2 * A`).
    run_exact(
        "array_scale",
        "void array_scale(double*, double, double*, double*);",
        "double a[3] = {1.0, 2.0, 3.0};\n    double y[3];\n    double yn;\n    array_scale(a, 3, y, &yn);\n    printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], yn);",
        "2 4 6 3",
    );
}

#[test]
fn run_array_add() {
    // Two equal-length dynamic arrays, elementwise (`y = A(:) + B(:)`).
    run_exact(
        "array_add",
        "void array_add(double*, double, double*, double, double*, double*);",
        "double a[3] = {1.0, 2.0, 3.0};\n    double b[3] = {10.0, 20.0, 30.0};\n    double y[3];\n    double yn;\n    array_add(a, 3, b, 3, y, &yn);\n    printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], yn);",
        "11 22 33 3",
    );
}

#[test]
fn run_array_mul() {
    // Two equal-length dynamic arrays, elementwise multiply (`y = A(:) .* B(:)`).
    run_exact(
        "array_mul",
        "void array_mul(double*, double, double*, double, double*, double*);",
        "double a[3] = {1.0, 2.0, 3.0};\n    double b[3] = {10.0, 20.0, 30.0};\n    double y[3];\n    double yn;\n    array_mul(a, 3, b, 3, y, &yn);\n    printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], yn);",
        "10 40 90 3",
    );
}

#[test]
fn run_array_neg() {
    // Unary negate into a dynamic array output (`y = -A(:)`).
    run_exact(
        "array_neg",
        "void array_neg(double*, double, double*, double*);",
        "double a[3] = {1.0, -2.0, 3.0};\n    double y[3];\n    double yn;\n    array_neg(a, 3, y, &yn);\n    printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], yn);",
        "-1 2 -3 3",
    );
}

// --- Matrix elementwise operators / broadcast / transpose ---------------------

#[test]
fn run_mat_add() {
    run_exact(
        "mat_add",
        "void mat_add(double*);",
        "double o[4]; mat_add(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "11 33 22 44",
    );
}

#[test]
fn run_mat_broadcast() {
    run_exact(
        "mat_broadcast",
        "void mat_broadcast(double*);",
        "double o[4]; mat_broadcast(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "11 13 12 14",
    );
}

#[test]
fn run_mat_scale() {
    run_exact(
        "mat_scale",
        "void mat_scale(double*);",
        "double o[4]; mat_scale(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "2 6 4 8",
    );
}

#[test]
fn run_mat_transpose() {
    run_exact(
        "mat_transpose",
        "void mat_transpose(double*);",
        "double o[4]; mat_transpose(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "1 2 3 4",
    );
}

#[test]
fn run_mat_cmp() {
    run_exact(
        "mat_cmp",
        "void mat_cmp(double*);",
        "double o[4]; mat_cmp(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "1 1 0 0",
    );
}

// --- Matrix multiplication ----------------------------------------------------

#[test]
fn run_matmul() {
    run_exact(
        "matmul",
        "void matmul(double*);",
        "double o[4]; matmul(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "19 43 22 50",
    );
}

#[test]
fn run_matvec() {
    run_exact(
        "matvec",
        "void matvec(double*);",
        "double o[2]; matvec(o);\n    printf(\"%g %g\\n\", o[0], o[1]);",
        "17 39",
    );
}

// --- Dimension reductions & shape introspection --------------------------------

#[test]
fn run_sum_dim1() {
    run_exact(
        "sum_dim1",
        "void sum_dim1(double*);",
        "double o[2]; sum_dim1(o);\n    printf(\"%g %g\\n\", o[0], o[1]);",
        "4 6",
    );
}

#[test]
fn run_sum_dim2() {
    run_exact(
        "sum_dim2",
        "void sum_dim2(double*);",
        "double o[2]; sum_dim2(o);\n    printf(\"%g %g\\n\", o[0], o[1]);",
        "3 7",
    );
}

#[test]
fn run_shape_intro() {
    // size(A,1)=2 + size(A,2)=2 + numel=4 + length=2 = 10
    run_exact(
        "shape_intro",
        "double shape_intro();",
        "printf(\"%g\\n\", shape_intro());",
        "10",
    );
}

#[test]
fn run_size_vec() {
    run_exact(
        "size_vec",
        "void size_vec(double*);",
        "double o[2]; size_vec(o);\n    printf(\"%g %g\\n\", o[0], o[1]);",
        "2 2",
    );
}

// --- Constructors & reshape ---------------------------------------------------

#[test]
fn run_zeros2x3() {
    run_exact(
        "zeros2x3",
        "void zeros2x3(double*);",
        "double o[6]; zeros2x3(o);\n    printf(\"%g %g %g %g %g %g\\n\", o[0], o[1], o[2], o[3], o[4], o[5]);",
        "0 0 0 0 0 0",
    );
}

#[test]
fn run_ones2x2() {
    run_exact(
        "ones2x2",
        "void ones2x2(double*);",
        "double o[4]; ones2x2(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "1 1 1 1",
    );
}

#[test]
fn run_eye2() {
    run_exact(
        "eye2",
        "void eye2(double*);",
        "double o[4]; eye2(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "1 0 0 1",
    );
}

#[test]
fn run_reshape2x2() {
    run_exact(
        "reshape2x2",
        "void reshape2x2(double*);",
        "double o[4]; reshape2x2(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "1 2 3 4",
    );
}

// --- Indexing -----------------------------------------------------------------

#[test]
fn run_index_scalar() {
    run_exact(
        "index_scalar",
        "double index_scalar();",
        "printf(\"%g\\n\", index_scalar());",
        "3",
    );
}

#[test]
fn run_index_end() {
    run_exact(
        "index_end",
        "double index_end();",
        "printf(\"%g\\n\", index_end());",
        "7",
    );
}

#[test]
fn run_index_colon() {
    run_exact(
        "index_colon",
        "void index_colon(double*);",
        "double o[4]; index_colon(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "1 3 2 4",
    );
}

// --- Power operators (`^` / `.^`) ----------------------------------------------

#[test]
fn run_power_scalar() {
    run_exact(
        "power_scalar",
        "double power_scalar(double, double);",
        "printf(\"%g\\n\", power_scalar(2, 3));",
        "16",
    );
}

#[test]
fn run_power_array() {
    run_exact(
        "power_array",
        "void power_array(double*);",
        "double o[4]; power_array(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "1 9 4 16",
    );
}

#[test]
fn run_mpower() {
    run_exact(
        "mpower",
        "void mpower(double*);",
        "double o[4]; mpower(o);\n    printf(\"%g %g %g %g\\n\", o[0], o[1], o[2], o[3]);",
        "7 15 10 22",
    );
}

// --- varargin / nargin / varargout -------------------------------------------

#[test]
fn run_varargin_sum() {
    run_exact(
        "varargin_sum",
        "double f(double, double);",
        "printf(\"%g\\n\", f(3, 4));",
        "7",
    );
}

#[test]
fn run_varargin_mix() {
    // A named parameter plus a single `varargin{1}` extra argument.
    run_exact(
        "varargin_mix",
        "double mix(double, double);",
        "printf(\"%g\\n\", mix(10, 5));",
        "15",
    );
}

#[test]
fn run_nargin_guard() {
    // `nargin` folds to 1, so the guard `nargin < 1` is false and `y = a`.
    run_exact(
        "nargin_guard",
        "double g(double);",
        "printf(\"%g\\n\", g(5));",
        "5",
    );
}

#[test]
fn run_varargout_two() {
    run_exact(
        "varargout_two",
        "std::tuple<double, double> h(double);",
        "auto [x, y] = h(3.0);\n    printf(\"%g %g\\n\", x, y);",
        "4 6",
    );
}

#[test]
fn run_break_sum() {
    // `break` exits the `for` loop once `i > 3`, so `y = 1 + 2 + 3 = 6`.
    run_exact(
        "break_sum",
        "double break_sum(double);",
        "printf(\"%g\\n\", break_sum(10.0));",
        "6",
    );
}

#[test]
fn run_continue_sum() {
    // `continue` skips `i == 2`, so `y = 1 + 3 + 4 = 8` (for `n = 4`).
    run_exact(
        "continue_sum",
        "double continue_sum(double);",
        "printf(\"%g\\n\", continue_sum(4.0));",
        "8",
    );
}

#[test]
fn run_while_break() {
    // `break` exits the `while` loop once `i > 3`, so `y = 1 + 2 + 3 = 6`.
    run_exact(
        "while_break",
        "double while_break(double);",
        "printf(\"%g\\n\", while_break(10.0));",
        "6",
    );
}

#[test]
fn run_persistent_counter() {
    // `persistent` maps to a C `static`, so `count` retains its value across
    // calls: `counter(1)` -> 1, then `counter(2)` -> 3.
    run_exact(
        "counter",
        "double counter(double);",
        "double a = counter(1.0);\n    double b = counter(2.0);\n    printf(\"%g %g\\n\", a, b);",
        "1 3",
    );
}

#[test]
fn run_global_counter() {
    // `global` (single-function world) also maps to a C `static`.
    run_exact(
        "global_count",
        "double global_count(double);",
        "double a = global_count(1.0);\n    double b = global_count(2.0);\n    printf(\"%g %g\\n\", a, b);",
        "1 3",
    );
}

#[test]
fn run_heap_allocation() {
    // A `100x50` array (5000 elements) is heap-allocated and still reads
    // correctly (`a(1)` == 1).
    run_exact(
        "big_array",
        "double big_array();",
        "printf(\"%g\\n\", big_array());",
        "1",
    );
}

#[test]
fn run_struct_field_access() {
    // `struct('a', 1, 'b', 2)`, then `s.a = 10`, so `y = 10 + 2 = 12`.
    run_exact(
        "struct_test",
        "double struct_test();",
        "printf(\"%g\\n\", struct_test());",
        "12",
    );
}

#[test]
fn run_struct_return() {
    // A function can return a struct by value.
    run_exact(
        "make_struct",
        "struct s0 { double x; double y; };\nstruct s0 make_struct();",
        "auto s = make_struct();\n    printf(\"%g %g\\n\", s.x, s.y);",
        "1 2",
    );
}

#[test]
fn run_struct_arg() {
    // A function can take a struct parameter by value; its fields are inferred
    // from use (`s.a + s.b`), mirroring MATLAB Coder.
    run_exact(
        "struct_arg",
        "struct s0 { double a; double b; };\ndouble struct_arg(struct s0);",
        "struct s0 s{3.0, 4.0};\n    printf(\"%g\\n\", struct_arg(s));",
        "7",
    );
}

#[test]
fn run_struct_mix() {
    // A function can return a struct and a scalar together (tuple ABI).
    run_exact(
        "struct_mix",
        "struct s0 { double x; double y; };\nstd::tuple<struct s0, double> struct_mix();",
        "auto [s, n] = struct_mix();\n    printf(\"%g %g %g\\n\", s.x, s.y, n);",
        "1 2 3",
    );
}

#[test]
fn run_struct_inout() {
    // A struct parameter and struct return: copy in, transform, copy out.
    run_exact(
        "struct_inout",
        "struct s0 { double a; double b; };\nstruct s0 struct_inout(struct s0);",
        "struct s0 s{5.0, 7.0};\n    auto t = struct_inout(s);\n    printf(\"%g %g\\n\", t.a, t.b);",
        "6 7",
    );
}

// --- Anonymous functions -------------------------------------------------------

#[test]
fn run_anon_scale() {
    // `f = @(t) t + 2; y = f(x)` inlines to `x + 2`; no captures.
    run_exact(
        "anon_scale",
        "double anon_scale(double);",
        "printf(\"%g\\n\", anon_scale(5.0));",
        "7",
    );
}

#[test]
fn run_anon_capture() {
    // `f = @(x) x * x + a` captures `a`; `f(3)` with a = 10 is 9 + 10 = 19.
    run_exact(
        "anon_capture",
        "double anon_capture(double);",
        "printf(\"%g\\n\", anon_capture(10.0));",
        "19",
    );
}

#[test]
fn run_anon_two() {
    // Two handles in one function: (3 + a) + (4 * a) with a = 2 is 5 + 8 = 13.
    run_exact(
        "anon_two",
        "double anon_two(double);",
        "printf(\"%g\\n\", anon_two(2.0));",
        "13",
    );
}

#[test]
fn run_anon_snapshot() {
    // MATLAB captures by value at creation: `a` is mutated to 100 *after* the
    // handle is created, so `f(1)` still sees the original a = 5 -> 1 + 5 = 6.
    run_exact(
        "anon_snapshot",
        "double anon_snapshot(double);",
        "printf(\"%g\\n\", anon_snapshot(5.0));",
        "6",
    );
}

#[test]
fn run_anon_lin() {
    // Two lambda parameters plus a capture: `x * t + a` with x = 3, t = 4, a = 1
    // is 12 + 1 = 13.
    run_exact(
        "anon_lin",
        "double anon_lin(double);",
        "printf(\"%g\\n\", anon_lin(1.0));",
        "13",
    );
}

#[test]
fn run_anon_zero() {
    // A zero-argument lambda that only reads a capture: `a * 2` with a = 5 is 10.
    run_exact(
        "anon_zero",
        "double anon_zero(double);",
        "printf(\"%g\\n\", anon_zero(5.0));",
        "10",
    );
}

#[test]
fn run_anon_loop() {
    // The handle is created at the top level but called inside a `for` loop:
    // sum of 2*i for i = 1..3 is 2 + 4 + 6 = 12.
    run_exact(
        "anon_loop",
        "double anon_loop(double);",
        "printf(\"%g\\n\", anon_loop(3.0));",
        "12",
    );
}

#[test]
fn run_anon_while() {
    // The handle is called in a `while` condition/body: `y = y + 1` until y >= 5.
    run_exact(
        "anon_while",
        "double anon_while(double);",
        "printf(\"%g\\n\", anon_while(5.0));",
        "5",
    );
}

#[test]
fn run_anon_expr() {
    // Two handles composed in one expression: f(3)*g(4) + f(1) with a = 2 is
    // 5 * 8 + 3 = 43.
    run_exact(
        "anon_expr",
        "double anon_expr(double);",
        "printf(\"%g\\n\", anon_expr(2.0));",
        "43",
    );
}

#[test]
fn run_anon_cond() {
    // A handle call as an `if` condition: `f(3) > 4` with a = 2 is 5 > 4 -> 1.
    run_exact(
        "anon_cond",
        "double anon_cond(double);",
        "printf(\"%g\\n\", anon_cond(2.0));",
        "1",
    );
}

#[test]
fn run_anon_multi_capture() {
    // Two captures in one lambda: `x + a * b` with x = 2, a = 3, b = 4 is 14.
    run_exact(
        "anon_multi_capture",
        "double anon_multi_capture(double, double);",
        "printf(\"%g\\n\", anon_multi_capture(3.0, 4.0));",
        "14",
    );
}

#[test]
fn run_anon_computed_capture() {
    // A capture of a computed local (not a parameter): k = 5*5, then f(1) = 26.
    run_exact(
        "anon_computed_capture",
        "double anon_computed_capture(double);",
        "printf(\"%g\\n\", anon_computed_capture(5.0));",
        "26",
    );
}

#[test]
fn run_anon_dead() {
    // A handle that is defined but never called still compiles (the helper is
    // emitted but unused; the snapshot is still taken at creation).
    run_exact(
        "anon_dead",
        "double anon_dead(double);",
        "printf(\"%g\\n\", anon_dead(7.0));",
        "7",
    );
}

// --- Optimization pass (src/passes): optimized code still runs correctly ------

#[test]
fn run_const_fold() {
    // `a = 2*3; b = a + 4; y = b*2` is folded at compile time to 20.
    run_exact(
        "const_fold",
        "double const_fold();",
        "printf(\"%g\\n\", const_fold());",
        "20",
    );
}

#[test]
fn run_dead_branch() {
    // The `a > 2` branch is folded and the else-branch removed; result is 1.
    run_exact(
        "dead_branch",
        "double dead_branch();",
        "printf(\"%g\\n\", dead_branch());",
        "1",
    );
}

#[test]
fn run_while_dynamic() {
    // Regression: the loop condition changes each iteration (`a: 2 -> 1`), so
    // the loop must run exactly once, not be folded into an infinite loop.
    run_exact(
        "while_dynamic",
        "double while_dynamic();",
        "printf(\"%g\\n\", while_dynamic());",
        "1",
    );
}

#[test]
fn run_array_intermediate() {
    // `t = A(:); y = sum(t)` allocates a runtime-length heap buffer for `t`.
    run_exact(
        "array_intermediate",
        "double array_intermediate(double*, double);",
        "double a[3] = {1.0, 2.0, 3.0};\n    \
         printf(\"%g\\n\", array_intermediate(a, 3.0));",
        "6",
    );
}

#[test]
fn run_array_intermediate_loop() {
    // A dynamic intermediate assigned inside a loop is block-scoped: its buffer
    // is allocated and freed on each iteration. A=[1 2 3], 2 iterations: 6 + 6.
    run_exact(
        "array_intermediate_loop",
        "double array_intermediate_loop(double*, double, double);",
        "double a[3] = {1.0, 2.0, 3.0};\n    \
         printf(\"%g\\n\", array_intermediate_loop(a, 3.0, 2.0));",
        "12",
    );
}

#[test]
fn run_array_sq() {
    // `n = sum(A); y = A .* A + n` with a dynamic-shape array parameter and a
    // dynamic-shape array output (buffer + out-length). A=[1 2 3] -> [7 10 15].
    run_exact(
        "array_sq",
        "void array_sq(double*, double, double*, double*);",
        "double a[3] = {1.0, 2.0, 3.0};\n    double y[3];\n    double n = 0.0;\n    \
         array_sq(a, 3.0, y, &n);\n    \
         printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], n);",
        "7 10 15 3",
    );
}

#[test]
fn run_array_scalar_ops() {
    // `B = A(:)` is a dynamic intermediate; `(2 - B) ./ k + B .* k` exercises
    // scalar-broadcast subtract/divide/multiply and elementwise add. For
    // A=[1 2 3], k=2 -> [(2-1)/2+2, (2-2)/2+4, (2-3)/2+6] = [2.5 4 5.5].
    run_exact(
        "array_scalar_ops",
        "void array_scalar_ops(double*, double, double, double*, double*);",
        "double a[3] = {1.0, 2.0, 3.0};\n    double y[3];\n    double n = 0.0;\n    \
         array_scalar_ops(a, 3.0, 2.0, y, &n);\n    \
         printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], n);",
        "2.5 4 5.5 3",
    );
}
