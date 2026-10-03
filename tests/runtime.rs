//! Compile-and-run tests for the dynamic-tier runtime kernel (`convmat_value`
//! value model: lifecycle, deep copy, and array/cell/struct helpers).
//!
//! The kernel is not wired into `.m` lowering yet (`docs/runtime.md` §8 step 3
//! is the seam), so these tests compile the raw C source directly against a
//! small driver, validating that the value model is self-contained and correct.

use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use convmat::runtime::{DYNAMIC_RUNTIME_C, DYNAMIC_RUNTIME_H};

/// Headers required by the runtime header (`jmp_buf` comes from `<setjmp.h>`).
const PREAMBLE: &str = "\
#include <cstdio>
#include <cstdint>
#include <setjmp.h>
";

/// Compile the runtime kernel plus `body` (a `main` body) and return stdout.
fn compile_and_run(body: &str) -> String {
    let compiler = std::env::var("CXX").unwrap_or_else(|_| {
        for name in ["g++", "clang++", "c++"] {
            if Command::new(name).arg("--version").output().is_ok() {
                return name.to_string();
            }
        }
        panic!("no C++ compiler found (set `CXX`)")
    });

    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("convmat-runtime-{}-{n}", std::process::id()));
    fs::create_dir_all(&dir).expect("create scratch dir");
    let src = dir.join("main.cpp");
    let bin = dir.join("prog");
    let cpp = format!(
        "{PREAMBLE}{DYNAMIC_RUNTIME_H}\n{DYNAMIC_RUNTIME_C}\nint main() {{\n{body}\n    return 0;\n}}\n"
    );
    fs::write(&src, &cpp).expect("write source");

    let compile = Command::new(&compiler)
        .arg("-std=c++17")
        .arg("-Wall")
        .arg("-Wextra")
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("spawn compiler");
    if !compile.status.success() {
        panic!(
            "compiling runtime kernel failed:\n{cpp}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );
    }

    let run = Command::new(&bin).output().expect("run program");
    let _ = fs::remove_dir_all(&dir);
    assert!(
        run.status.success(),
        "runtime test exited {:?}: {}",
        run.status.code(),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).trim().to_string()
}

#[test]
fn scalar_lifecycle_and_refcount() {
    let out = compile_and_run(
        "\
    convmat_value* v = convmat_value_new(CONVMAT_SCALAR, CONVMAT_DOUBLE);
    v->u.scalar.d = 3.5;
    convmat_value_retain(v);
    printf(\"%g %d\\n\", v->u.scalar.d, v->refcount);
    convmat_value_release(v);
    printf(\"%d\\n\", v->refcount);
    convmat_value_release(v);
",
    );
    assert_eq!(out, "3.5 2\n1");
}

#[test]
fn array_create_data_numel_and_linear_index() {
    let out = compile_and_run(
        "\
    int64_t dims[2] = {2, 3};
    convmat_value* a = convmat_array_create(CONVMAT_DOUBLE, 2, dims);
    double* d = static_cast<double*>(convmat_array_data(a));
    for (int64_t i = 0; i < 6; i++) d[i] = static_cast<double>(i + 1);
    printf(\"%g %g\\n\", d[0], d[5]);
    printf(\"%d\\n\", static_cast<int>(convmat_numel(a)));
    int64_t subs[2] = {1, 2};
    printf(\"%d\\n\", static_cast<int>(convmat_linear_index(a, subs)));
    convmat_value_release(a);
",
    );
    // d[0]=1, d[5]=6; numel=6; column-major offset of (row1,col2) in 2x3 = 5.
    assert_eq!(out, "1 6\n6\n5");
}

#[test]
fn cell_set_get_and_deep_copy() {
    let out = compile_and_run(
        "\
    int64_t dims[1] = {3};
    convmat_value* c = convmat_cell_create(1, dims);
    convmat_value* e = convmat_value_new(CONVMAT_SCALAR, CONVMAT_DOUBLE);
    e->u.scalar.d = 42.0;
    convmat_cell_set(c, 0, e);
    convmat_value_release(e);
    convmat_value* got = convmat_cell_get(c, 0);
    printf(\"%g\\n\", got->u.scalar.d);
    convmat_value_release(got);
    convmat_value* cp = convmat_value_copy(c);
    convmat_value* got2 = convmat_cell_get(cp, 0);
    printf(\"%g\\n\", got2->u.scalar.d);
    convmat_value_release(got2);
    convmat_value_release(cp);
    convmat_value_release(c);
",
    );
    assert_eq!(out, "42\n42");
}

#[test]
fn struct_fields_lookup_and_copy() {
    let out = compile_and_run(
        "\
    const char* names[2] = {\"x\", \"y\"};
    int64_t dims[2] = {1, 1};
    convmat_value* s = convmat_struct_create(2, names, 2, dims);
    int64_t fx = convmat_struct_field_index(s, \"x\");
    int64_t fy = convmat_struct_field_index(s, \"y\");
    convmat_value* vx = convmat_value_new(CONVMAT_SCALAR, CONVMAT_DOUBLE);
    vx->u.scalar.d = 1.5;
    convmat_value* vy = convmat_value_new(CONVMAT_SCALAR, CONVMAT_DOUBLE);
    vy->u.scalar.d = 2.5;
    convmat_struct_set(s, fx, 0, vx);
    convmat_struct_set(s, fy, 0, vy);
    convmat_value_release(vx);
    convmat_value_release(vy);
    convmat_value* gx = convmat_struct_get(s, fx, 0);
    convmat_value* gy = convmat_struct_get(s, fy, 0);
    printf(\"%g %g %d\\n\", gx->u.scalar.d, gy->u.scalar.d,
           static_cast<int>(convmat_struct_field_index(s, \"z\")));
    convmat_value_release(gx);
    convmat_value_release(gy);
    convmat_value_release(s);
",
    );
    assert_eq!(out, "1.5 2.5 -1");
}

#[test]
fn deep_copy_is_independent() {
    let out = compile_and_run(
        "\
    int64_t dims[1] = {2};
    convmat_value* a = convmat_array_create(CONVMAT_DOUBLE, 1, dims);
    double* d = static_cast<double*>(convmat_array_data(a));
    d[0] = 1.0; d[1] = 2.0;
    convmat_value* b = convmat_value_copy(a);
    static_cast<double*>(convmat_array_data(b))[0] = 99.0;
    printf(\"%g %g\\n\", static_cast<double*>(convmat_array_data(a))[0],
           static_cast<double*>(convmat_array_data(b))[0]);
    convmat_value_release(a);
    convmat_value_release(b);
",
    );
    assert_eq!(out, "1 99");
}

#[test]
fn array_resize_preserves_and_grows() {
    let out = compile_and_run(
        "\
    int64_t dims2[1] = {2};
    convmat_value* a = convmat_array_create(CONVMAT_DOUBLE, 1, dims2);
    double* d = static_cast<double*>(convmat_array_data(a));
    d[0] = 1.0; d[1] = 2.0;
    int64_t dims5[1] = {5};
    convmat_array_resize(a, 1, dims5);
    double* g = static_cast<double*>(convmat_array_data(a));
    printf(\"%g %g %g\\n\", g[0], g[1], g[4]);
    convmat_value_release(a);
",
    );
    // Newly grown slots are zero-initialized; the first two are preserved.
    assert_eq!(out, "1 2 0");
}
