# MATLAB Coder test cases

A corpus of small MATLAB Coder test cases used to drive convmat's coverage and
to serve as a roadmap: each entry is either lowered today or is a tracked
feature gap. Files in this folder are adapted from the MathWorks MATLAB Coder
documentation/example gallery plus targeted cases for individual language
features.

Compile any one of them with:

```sh
cargo run -- examples/coder/<name>.m
```

The survey lives in `tests/coder_examples.rs`:

- `SUPPORTED` lists the examples that compile **and** run correctly today.
- `KNOWN_BUGS` lists examples that compile but are semantically wrong today.
- Everything else on disk is expected to fail to compile until its feature
  lands. `coder_examples_match_coverage` fails if any example's outcome changes,
  so landing a feature is a deliberate "promote it into `SUPPORTED`" step.
- `cargo test --test coder_examples -- --ignored --nocapture` prints a report of
  every example and the compiler error for the unsupported ones.
- `docs/roadmap.md` groups the remaining failures into workstreams (WS) with the
  root cause, implementation steps and acceptance criteria for each.

Current status: **44 / 50** examples compile (44 correct, 0 miscompiled).

## Supported

| Example              | Checks                            |
|----------------------|-----------------------------------|
| `addone`             | scalar add                        |
| `array_broadcast`    | implicit singleton expansion `A + b` |
| `array_col_slice`    | `A(:, j)` column slice            |
| `array_concat`       | block concatenation `[a, b]`      |
| `array_linspace`      | `linspace(0, 1, 5)`               |
| `array_logical_index` | logical indexing `A(A > 0)`       |
| `array_mask_assign`   | masked write `A(A < 0) = 0`       |
| `array_nd`           | N-D arrays (`zeros(2,2,2)`, `A(i,j,k)`) |
| `array_param_normalize` | dynamic-shape vector mean-centering (runtime matrix) |
| `array_permute`      | `permute(A, [2 1])`               |
| `array_repmat`       | `repmat(A, m, n)`                 |
| `array_row_slice`    | `A(i, :)` row slice               |
| `array_stride`       | `A(1:2:end)` strided range        |
| `averaging_filter`   | 16-sample moving average (persistent + concat) |
| `kalmanfilter`       | scalar Kalman filter (`persistent` + `isempty` initialization) |
| `builtin_clamp`      | nested `min`/`max`                |
| `builtin_binary_math`| `atan2` / `hypot` / `mod` / `rem` |
| `builtin_mean`       | `mean` reduction                  |
| `builtin_mean_param` | `mean` of a dynamic-shape array parameter |
| `builtin_median`     | `median` (sorted via `convmat_sort`) |
| `builtin_std`        | sample standard deviation         |
| `fib`                | recursion (`fib(n-1) + fib(n-2)`) |
| `func_helper`        | call to a later-defined function  |
| `func_recursion`     | recursion (factorial)             |
| `builtin_cumsum`     | cumulative sum (array result)     |
| `builtin_diff`       | adjacent differences (array result) |
| `builtin_predicates` | `isnan` / `isinf`                 |
| `linalg_det`         | `det` of a square matrix          |
| `linalg_inv`         | `inv` of a square matrix          |
| `linalg_norm`        | vector 2-norm                     |
| `linalg_solve`       | `A \ b` linear solve             |
| `sys_random`         | `rand()` in `[0, 1)`              |
| `sys_trycatch`       | `try` / `catch` (`setjmp`)        |
| `mandelbrot_count`   | loops, `abs`, integer power, `break` |
| `type_logical`       | `logical` conversion              |
| `value_special`      | `Inf` / `NaN` literals            |
| `struct_nested`      | nested field `s.a.b` (flattened)  |
| `text_compare`       | `strcmp` of char literals         |
| `text_switch`        | `switch` on a char value          |
| `type_integer`       | `int32` arithmetic (wraparound)   |
| `cell_basic`         | scalar-element cell `{1, 2, 3}`, `c{1}` |
| `type_complex`       | `abs(3 + 4i)` (complex arithmetic) |
| `sys_fft`            | `fft` (complex DFT, boxed result) |
| `linalg_eig`         | `eig` of a 2x2 matrix (complex box) |

## Known bugs (compile, but wrong)

None currently.

## Roadmap (currently rejected)

### Functions & handles

| Example                | Blocker |
|------------------------|---------|
| `func_handle_return`   | escaping anonymous handle |
| `func_handle_builtin`  | handle to a built-in (`@sin`) |

### System / runtime

| Example         | Blocker |
|-----------------|---------|
| `sys_plot`      | `plot` |
| `sys_file_io`   | `fopen` / `fread` |

### MATLAB Coder example gallery

| Example      | Blocker |
|--------------|---------|
| `fib`        | recursive call |
| `dijkstra`   | `Inf` / colon slices / elementwise `min` |
| `sierpinski` | `plot` |

## Provenance

The gallery examples (`addone`, `averaging_filter`, `fib`, `mandelbrot_count`,
`sierpinski`, `dijkstra`, `kalmanfilter`) are adapted, self-contained
reproductions of the corresponding MathWorks MATLAB Coder examples, not verbatim
copies. `mandelbrot_count` keeps `c` real-valued, where the original uses a
complex `c`. The `builtin_*`, `array_*`, `linalg_*`, `type_*`, `func_*`,
`struct_*`, `cell_*`, `text_*`, `value_*` and `sys_*` cases are targeted feature
tests written for this compiler.

MATLAB and MATLAB Coder are trademarks of The MathWorks, Inc. This folder is
only for compiler testing.
