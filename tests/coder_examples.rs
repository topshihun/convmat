//! Survey of the MATLAB Coder example programs in `examples/coder/`.
//!
//! Every example is compiled end-to-end to C. The [`EXPECTED`] table records
//! whether convmat's current codegen subset covers each example; the test fails
//! if an example is missing from the table or its outcome changes, so it doubles
//! as a coverage regression guard. Examples marked supported are additionally
//! compiled together with a tiny `main` driver and run, so the survey also
//! checks runtime behavior, not just that C was produced.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use convmat::backend::BackendKind;
use convmat::frontend::SourceFile;
use convmat::pipeline;

/// Whether each example under `examples/coder/` currently compiles. Names are
/// file stems. A missing entry or a changed outcome fails the survey test.
const EXPECTED: &[(&str, bool)] = &[
    ("addone", true),
    ("averaging_filter", false),
    ("dijkstra", false),
    ("fib", false),
    ("kalmanfilter", false),
    ("mandelbrot_count", true),
    ("sierpinski", false),
];

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
fn run_example(name: &str, decls: &str, body: &str) -> String {
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

/// Every example on disk must be classified in [`EXPECTED`], and compiling it
/// must agree with that classification.
#[test]
fn coder_examples_match_coverage() {
    let names = example_names();
    assert!(!names.is_empty(), "no examples found under examples/coder");

    let expected: std::collections::BTreeMap<&str, bool> = EXPECTED
        .iter()
        .copied()
        .collect::<std::collections::BTreeMap<_, _>>(
    );

    for name in &names {
        let Some(&want_ok) = expected.get(name.as_str()) else {
            panic!("example `{name}` is missing from the EXPECTED table");
        };
        let result = compile_example(name);
        assert_eq!(
            result.is_ok(),
            want_ok,
            "`{name}`: compiled={}, expected compiled={want_ok}; result: {result:?}",
            result.is_ok()
        );
    }

    let on_disk: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    for (name, _) in EXPECTED {
        assert!(
            on_disk.contains(name),
            "EXPECTED lists `{name}` but no `examples/coder/{name}.m` exists"
        );
    }
}

/// The examples convmat currently covers must also behave correctly at run time.
#[test]
fn coder_examples_supported_run() {
    // addone(2) == 3.
    assert_eq!(
        run_example(
            "addone",
            "double addone(double);",
            "printf(\"%g\\n\", addone(2.0));"
        ),
        "3"
    );

    // mandelbrot_count stays inside the set for c=0 and breaks at n=3 for c=1
    // with 10 iterations (0 -> 1 -> 2 -> 5, where |z| > 2).
    assert_eq!(
        run_example(
            "mandelbrot_count",
            "double mandelbrot_count(double, double);",
            "printf(\"%g\\n\", mandelbrot_count(0.0, 10.0));\n    \
             printf(\"%g\\n\", mandelbrot_count(1.0, 10.0));",
        ),
        "10\n3"
    );
}
