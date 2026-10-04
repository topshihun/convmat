//! Survey of the MATLAB Coder test cases in `examples/coder/`.
//!
//! Every `.m` file is compiled end-to-end to C. [`SUPPORTED`] lists the examples
//! convmat currently lowers; every other file is a tracked roadmap item that is
//! expected to fail until its feature lands. The survey fails if an example's
//! outcome disagrees with that classification, so landing a feature is a
//! deliberate "promote it into `SUPPORTED`" step; the failure names the example
//! and shows the compiler error.
//!
//! Examples in `SUPPORTED` are also linked with a tiny `main` driver and run, so
//! "supported" means the generated C both compiles and produces the right
//! answer. Run `cargo test --test coder_examples -- --ignored --nocapture` for a
//! report of every example and why the unsupported ones are rejected.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use convmat::backend::BackendKind;
use convmat::frontend::SourceFile;
use convmat::pipeline;

/// Examples that compile to correct C today. Extend this as features land.
const SUPPORTED: &[&str] = &[
    "addone",
    "array_broadcast",
    "array_col_slice",
    "array_concat",
    "array_linspace",
    "array_logical_index",
    "array_mask_assign",
    "array_nd",
    "array_param_normalize",
    "array_permute",
    "array_repmat",
    "array_row_slice",
    "array_stride",
    "averaging_filter",
    "builtin_binary_math",
    "builtin_clamp",
    "builtin_cumsum",
    "builtin_diff",
    "builtin_mean",
    "builtin_mean_param",
    "builtin_median",
    "builtin_predicates",
    "builtin_std",
    "cell_basic",
    "fib",
    "func_helper",
    "func_recursion",
    "kalmanfilter",
    "linalg_det",
    "linalg_eig",
    "linalg_inv",
    "linalg_norm",
    "linalg_solve",
    "mandelbrot_count",
    "struct_nested",
    "sys_fft",
    "sys_random",
    "sys_trycatch",
    "text_compare",
    "text_switch",
    "type_complex",
    "type_integer",
    "type_logical",
    "value_special",
];

/// Examples convmat currently compiles but miscompiles. They are accepted by the
/// coverage test (they do produce C); any `#[ignore]`d expected-value test
/// documents the intended behavior until the underlying semantics land. Empty
/// today: the last entry (`kalmanfilter`) was fixed by modeling `isempty` of a
/// `persistent` variable.
const KNOWN_BUGS: &[&str] = &[];

/// Tolerance for floating-point runtime comparisons.
const TOLERANCE: f64 = 1e-9;

/// The directory holding the examples.
fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/coder")
}

/// The `.m` examples on disk, sorted, by file stem.
fn example_names() -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(example_dir())
        .expect("read examples/coder")
        .map(|entry| entry.expect("read dir entry").path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("m"))
        .map(|path| path.file_stem().unwrap().to_str().unwrap().to_string())
        .collect();
    names.sort();
    names
}

/// Compile one example to C, returning the compiler result.
fn compile_example(name: &str) -> convmat::error::Result<String> {
    let path = example_dir().join(format!("{name}.m"));
    let source = SourceFile::read(path).expect("read example");
    pipeline::compile(&source, BackendKind::C)
}

/// Locate a usable C++ compiler: honour `$CXX`, otherwise probe the usual
/// suspects (the C backend emits C++).
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

/// Create a fresh, unique scratch directory for one run.
fn scratch_dir(name: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("convmat-coder-{}-{name}-{n}", std::process::id()));
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Compile a supported example with a `main` driver and return trimmed stdout.
fn run_program(name: &str, decls: &str, body: &str) -> String {
    let cpp = compile_example(name).expect("supported example should compile");
    let compiler = find_compiler()
        .unwrap_or_else(|| panic!("no C++ compiler found (set `CXX`); run tests need one"));

    let dir = scratch_dir(name);
    let gen_path = dir.join("gen.cpp");
    let main_path = dir.join("main.cpp");
    let bin_path = dir.join("prog");

    fs::write(&gen_path, &cpp).expect("write generated code");
    fs::write(
        &main_path,
        format!(
            "#include <cstdio>\n#include <cstdint>\n#include <tuple>\n#include <cmath>\n\
             {decls}\nint main() {{\n{body}\n    return 0;\n}}\n"
        ),
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
            "compiling `{name}` with `{compiler}` failed:\n--- generated C++ ---\n{cpp}\n\
             --- compiler stderr ---\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );
    }

    let run = Command::new(&bin_path).output().expect("run program");
    let _ = fs::remove_dir_all(&dir);

    assert!(
        run.status.success(),
        "`{name}` exited with {:?}\nstderr: {}",
        run.status.code(),
        String::from_utf8_lossy(&run.stderr)
    );

    String::from_utf8_lossy(&run.stdout).trim().to_string()
}

/// Assert that a driver prints exactly `expected`.
fn run_exact(name: &str, decls: &str, body: &str, expected: &str) {
    let got = run_program(name, decls, body);
    assert_eq!(got, expected, "`{name}` produced wrong output");
}

/// Assert that a driver prints one value per line, matching within `TOLERANCE`.
fn run_close(name: &str, decls: &str, body: &str, expected: &[f64]) {
    let stdout = run_program(name, decls, body);
    let got: Vec<f64> = stdout
        .lines()
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("`{name}` printed non-numeric output: {line:?}"))
        })
        .collect();
    assert_eq!(
        got.len(),
        expected.len(),
        "`{name}` printed {} values, expected {}",
        got.len(),
        expected.len()
    );
    for (i, (got, want)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= TOLERANCE * (1.0 + want.abs()),
            "`{name}` value {i}: got {got}, expected {want}"
        );
    }
}

/// Each example must compile iff it is listed in [`SUPPORTED`].
#[test]
fn coder_examples_match_coverage() {
    let names = example_names();
    assert!(!names.is_empty(), "no examples found under examples/coder");

    for name in &names {
        let want_ok = SUPPORTED.contains(&name.as_str()) || KNOWN_BUGS.contains(&name.as_str());
        let result = compile_example(name);
        assert_eq!(
            result.is_ok(),
            want_ok,
            "`{name}`: compiled={}, expected compiled={want_ok}; result: {result:?}",
            result.is_ok()
        );
    }

    for name in SUPPORTED.iter().chain(KNOWN_BUGS) {
        assert!(
            names.iter().any(|n| n == name),
            "coverage lists `{name}` but no `examples/coder/{name}.m` exists"
        );
    }
}

/// The supported examples must also behave correctly at run time.
#[test]
fn coder_examples_supported_run() {
    // addone(2) == 3.
    run_exact(
        "addone",
        "double addone(double);",
        "printf(\"%g\\n\", addone(2.0));",
        "3",
    );

    // min(max(x, 0), 1) at the upper, lower and interior bounds.
    run_exact(
        "builtin_clamp",
        "double builtin_clamp(double);",
        "printf(\"%g\\n\", builtin_clamp(2.0));\n    \
         printf(\"%g\\n\", builtin_clamp(-0.5));\n    \
         printf(\"%g\\n\", builtin_clamp(0.5));",
        "1\n0\n0.5",
    );

    // mandelbrot_count stays inside the set for c=0 and breaks at n=3 for c=1
    // with 10 iterations (0 -> 1 -> 2 -> 5, where |z| > 2).
    run_exact(
        "mandelbrot_count",
        "double mandelbrot_count(double, double);",
        "printf(\"%g\\n\", mandelbrot_count(0.0, 10.0));\n    \
         printf(\"%g\\n\", mandelbrot_count(1.0, 10.0));",
        "10\n3",
    );

    // atan2(4, 2) + hypot(4, 2) + mod(4, 2) + rem(4, 2); the last two are 0.
    let expected = 4.0f64.atan2(2.0) + 4.0f64.hypot(2.0);
    run_close(
        "builtin_binary_math",
        "double builtin_binary_math(double, double);",
        "printf(\"%.17g\\n\", builtin_binary_math(4.0, 2.0));",
        &[expected],
    );

    // mean([1 2 3 4]) == 10 / 4.
    run_exact(
        "builtin_mean",
        "double builtin_mean(void);",
        "printf(\"%g\\n\", builtin_mean());",
        "2.5",
    );

    // mean of a dynamic-shape array parameter (pointer + length ABI).
    run_exact(
        "builtin_mean_param",
        "double builtin_mean_param(double*, double);",
        "double a[4] = {1.0, 2.0, 3.0, 4.0};\n    \
         printf(\"%g\\n\", builtin_mean_param(a, 4.0));",
        "2.5",
    );

    // median([3 1 2]) sorts to [1 2 3], middle element 2.
    run_exact(
        "builtin_median",
        "double builtin_median(void);",
        "printf(\"%g\\n\", builtin_median());",
        "2",
    );

    // std([1 2 3 4]) == sqrt(5 / 3) (sample standard deviation, n - 1).
    run_close(
        "builtin_std",
        "double builtin_std(void);",
        "printf(\"%.17g\\n\", builtin_std());",
        &[(5.0f64 / 3.0).sqrt()],
    );

    // cumsum([1 2 3 4]) == [1 3 6 10] (array result via an out-buffer).
    run_exact(
        "builtin_cumsum",
        "void builtin_cumsum(double*);",
        "double y[4];\n    builtin_cumsum(y);\n    \
         printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], y[3]);",
        "1 3 6 10",
    );

    // diff([1 4 9 16]) == [3 5 7].
    run_exact(
        "builtin_diff",
        "void builtin_diff(double*);",
        "double y[3];\n    builtin_diff(y);\n    \
         printf(\"%g %g %g\\n\", y[0], y[1], y[2]);",
        "3 5 7",
    );

    // isnan/isinf: NAN -> 1, INFINITY -> 1, a finite scalar -> 0.
    run_exact(
        "builtin_predicates",
        "double builtin_predicates(double);",
        "printf(\"%g\\n\", builtin_predicates(NAN));\n    \
         printf(\"%g\\n\", builtin_predicates(INFINITY));\n    \
         printf(\"%g\\n\", builtin_predicates(3.0));",
        "1\n1\n0",
    );

    // Mean-centered dynamic-shape vector: A=[1 2 3 4], mean=2.5 -> [-1.5 -0.5 0.5 1.5].
    run_exact(
        "array_param_normalize",
        "void array_param_normalize(double*, double, double*, double*);",
        "double a[4] = {1.0, 2.0, 3.0, 4.0};\n    double y[4];\n    double n = 0.0;\n    \
         array_param_normalize(a, 4.0, y, &n);\n    \
         printf(\"%g %g %g %g %g\\n\", y[0], y[1], y[2], y[3], n);",
        "-1.5 -0.5 0.5 1.5 4",
    );
    // array_concat: [1 2] concatenated with [3 4] is [1 2 3 4]; sum is 10.
    run_exact(
        "array_concat",
        "double array_concat(void);",
        "printf(\"%g\\n\", array_concat());",
        "10",
    );

    // A(1,:) on [1 2 3; 4 5 6] is [1 2 3].
    run_exact(
        "array_row_slice",
        "void array_row_slice(double*);",
        "double y[3];\n    array_row_slice(y);\n    \
         printf(\"%g %g %g\\n\", y[0], y[1], y[2]);",
        "1 2 3",
    );

    // A(:,2) on [1 2 3; 4 5 6] is [2; 5].
    run_exact(
        "array_col_slice",
        "void array_col_slice(double*);",
        "double y[2];\n    array_col_slice(y);\n    \
         printf(\"%g %g\\n\", y[0], y[1]);",
        "2 5",
    );

    // A(1:2:5) on [1 2 3 4 5] is [1 3 5].
    run_exact(
        "array_stride",
        "void array_stride(double*);",
        "double y[3];\n    array_stride(y);\n    \
         printf(\"%g %g %g\\n\", y[0], y[1], y[2]);",
        "1 3 5",
    );

    // Implicit expansion: [1 2; 3 4] + [10 20] == [11 22; 13 24] (column-major
    // [11 13 22 24]).
    run_exact(
        "array_broadcast",
        "void array_broadcast(double*);",
        "double y[4];\n    array_broadcast(y);\n    \
         printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], y[3]);",
        "11 13 22 24",
    );

    // N-D array: zeros(2,2,2) is all zeros, so A(1,2,1) is 0.
    run_exact(
        "array_nd",
        "double array_nd(void);",
        "printf(\"%g\\n\", array_nd());",
        "0",
    );

    // linspace(0, 1, 5) == [0 0.25 0.5 0.75 1].
    run_close(
        "array_linspace",
        "void array_linspace(double*);",
        "double y[5];\n    array_linspace(y);\n    \
         printf(\"%.17g\\n%.17g\\n%.17g\\n%.17g\\n%.17g\\n\", \
             y[0], y[1], y[2], y[3], y[4]);",
        &[0.0, 0.25, 0.5, 0.75, 1.0],
    );

    // repmat([1 2], 2, 1) == [1 2; 1 2] (column-major [1 1 2 2]).
    run_exact(
        "array_repmat",
        "void array_repmat(double*);",
        "double y[4];\n    array_repmat(y);\n    \
         printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], y[3]);",
        "1 1 2 2",
    );

    // permute([1 2; 3 4], [2 1]) transposes to [1 3; 2 4] (column-major [1 2 3 4]).
    run_exact(
        "array_permute",
        "void array_permute(double*);",
        "double y[4];\n    array_permute(y);\n    \
         printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], y[3]);",
        "1 2 3 4",
    );

    // logical(1) + logical(0) == 1.
    run_exact(
        "type_logical",
        "double type_logical(void);",
        "printf(\"%g\\n\", type_logical());",
        "1",
    );

    // det([1 2; 3 4]) == 1*4 - 2*3 == -2.
    run_exact(
        "linalg_det",
        "double linalg_det(void);",
        "printf(\"%g\\n\", linalg_det());",
        "-2",
    );

    // inv([1 2; 3 4]) == [-2 1; 1.5 -0.5] (column-major [-2 1.5 1 -0.5]).
    run_exact(
        "linalg_inv",
        "void linalg_inv(double*);",
        "double y[4];\n    linalg_inv(y);\n    \
         printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], y[3]);",
        "-2 1.5 1 -0.5",
    );

    // norm([3 4]) == 5.
    run_exact(
        "linalg_norm",
        "double linalg_norm(void);",
        "printf(\"%g\\n\", linalg_norm());",
        "5",
    );

    // [2 0; 0 2] \ [2; 4] == [1; 2].
    run_exact(
        "linalg_solve",
        "void linalg_solve(double*);",
        "double y[2];\n    linalg_solve(y);\n    \
         printf(\"%g %g\\n\", y[0], y[1]);",
        "1 2",
    );

    // rand() returns a pseudo-random scalar in [0, 1).
    {
        let out = run_program(
            "sys_random",
            "double sys_random(void);",
            "printf(\"%.17g\\n\", sys_random());",
        );
        let value: f64 = out
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("sys_random printed non-numeric output: {out:?}"));
        assert!(
            (0.0..1.0).contains(&value),
            "rand() must lie in [0, 1), got {value}"
        );
    }

    // isinf(Inf) + isnan(NaN) == 2.
    run_exact(
        "value_special",
        "double value_special(void);",
        "printf(\"%g\\n\", value_special());",
        "2",
    );

    // averaging_filter: a 16-sample moving average over a persistent buffer.
    // Two calls with 16 then 0 both average to 16 / 16 == 1.
    run_exact(
        "averaging_filter",
        "double averaging_filter(double);",
        "printf(\"%g\\n\", averaging_filter(16.0));\n    \
         printf(\"%g\\n\", averaging_filter(0.0));",
        "1\n1",
    );

    // fib(7) == 13 (recursion).
    run_exact(
        "fib",
        "double fib(double);",
        "printf(\"%g\\n\", fib(7.0));",
        "13",
    );

    // func_helper(3) == square(3) + 1 == 10 (call to a later-defined function).
    run_exact(
        "func_helper",
        "double func_helper(double);",
        "printf(\"%g\\n\", func_helper(3.0));",
        "10",
    );

    // func_recursion(5) == 5! == 120 (recursion).
    run_exact(
        "func_recursion",
        "double func_recursion(double);",
        "printf(\"%g\\n\", func_recursion(5.0));",
        "120",
    );

    // kalmanfilter relies on `isempty` of a `persistent` variable being true on
    // the first call, which initializes `P = 1`. First call with z = 1:
    // x = 0, P = 1; Q = 0.01, R = 0.1.
    let (q, r, z) = (0.01f64, 0.1f64, 1.0f64);
    let p = 1.0 + q;
    let k = p / (p + r);
    let x = k * z;
    run_close(
        "kalmanfilter",
        "double kalmanfilter(double);",
        "printf(\"%.17g\\n\", kalmanfilter(1.0));",
        &[x],
    );

    // The nested field `s.a.b` round-trips a scalar (`s.a.b = 1`).
    run_exact(
        "struct_nested",
        "double struct_nested(void);",
        "printf(\"%g\\n\", struct_nested());",
        "1",
    );

    // `strcmp('abc', 'abc')` is true (constant-folded char comparison).
    run_exact(
        "text_compare",
        "double text_compare(void);",
        "printf(\"%g\\n\", text_compare());",
        "1",
    );

    // Switching on a char code point: 'a' (97) -> 1, 'z' (122) -> otherwise 0.
    run_exact(
        "text_switch",
        "double text_switch(double);",
        "printf(\"%g\\n\", text_switch(97.0));\n    \
         printf(\"%g\\n\", text_switch(122.0));",
        "1\n0",
    );

    // `int32(3) + int32(4) == 7` (int32 wraparound arithmetic).
    run_exact(
        "type_integer",
        "double type_integer(void);",
        "printf(\"%g\\n\", type_integer());",
        "7",
    );

    // `A(A > 0)` on `[1 -2 3 -4]` gathers `[1 3]` (dynamic-length output ABI).
    run_exact(
        "array_logical_index",
        "void array_logical_index(double*, double*);",
        "double y[4];\n    double n;\n    \
         array_logical_index(y, &n);\n    \
         printf(\"%g %g %g\\n\", n, y[0], y[1]);",
        "2 1 3",
    );

    // `A(A < 0) = 0` on `[1 -2 3 -4]` gives `[1 0 3 0]`.
    run_exact(
        "array_mask_assign",
        "void array_mask_assign(double v1[4]);",
        "double y[4];\n    array_mask_assign(y);\n    \
         printf(\"%g %g %g %g\\n\", y[0], y[1], y[2], y[3]);",
        "1 0 3 0",
    );

    // `sys_trycatch(2)` = `1 / 2`; the try body runs (no error is raised).
    run_exact(
        "sys_trycatch",
        "double sys_trycatch(double);",
        "printf(\"%g\\n\", sys_trycatch(2.0));",
        "0.5",
    );

    // `c = {1, 2, 3}; y = c{1}` reads the first scalar element of a boxed cell.
    run_exact(
        "cell_basic",
        "double cell_basic(void);",
        "printf(\"%g\\n\", cell_basic());",
        "1",
    );

    // `abs(3 + 4i)` is the magnitude 5.
    run_exact(
        "type_complex",
        "double type_complex(void);",
        "printf(\"%g\\n\", type_complex());",
        "5",
    );

    // `fft([1 2 3 4])` is a boxed complex array with real parts 10, -2, -2, -2.
    run_exact(
        "sys_fft",
        "struct convmat_value;\n\
         double convmat_complex_component(const convmat_value*, double, double);\n\
         void convmat_value_release(convmat_value*);\n\
         convmat_value* sys_fft(void);",
        "convmat_value* z = sys_fft();\n    \
         printf(\"%g %g %g %g\\n\", convmat_complex_component(z, 0.0, 0.0),\n    \
         convmat_complex_component(z, 1.0, 0.0), convmat_complex_component(z, 2.0, 0.0),\n    \
         convmat_complex_component(z, 3.0, 0.0));\n    \
         convmat_value_release(z);",
        "10 -2 -2 -2",
    );

    // `eig([2 0; 0 3])` is a boxed complex vector with real parts 2, 3.
    run_exact(
        "linalg_eig",
        "struct convmat_value;\n\
         double convmat_complex_component(const convmat_value*, double, double);\n\
         void convmat_value_release(convmat_value*);\n\
         convmat_value* linalg_eig(void);",
        "convmat_value* z = linalg_eig();\n    \
         printf(\"%g %g\\n\", convmat_complex_component(z, 0.0, 0.0),\n    \
         convmat_complex_component(z, 1.0, 0.0));\n    \
         convmat_value_release(z);",
        "2 3",
    );
}

/// Manual report: prints the compile outcome (and rejection reason) of every
/// example. Ignored by default; run with `--ignored --nocapture`.
#[test]
#[ignore = "report only"]
fn coder_examples_report() {
    let supported = SUPPORTED
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut passed = 0;
    for name in example_names() {
        match compile_example(&name) {
            Ok(_) => {
                passed += 1;
                let mark = if supported.contains(name.as_str()) {
                    "PASS"
                } else if KNOWN_BUGS.contains(&name.as_str()) {
                    "PASS (known bug)"
                } else {
                    "PASS (not in SUPPORTED)"
                };
                println!("{mark:24} {name}");
            }
            Err(err) => println!("FAIL                     {name}: {err}"),
        }
    }
    println!("\n{passed} / {} examples compile", example_names().len());
}
