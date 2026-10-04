# MATLAB Coder examples

A small set of example functions adapted from the MathWorks MATLAB Coder
documentation and example gallery, used to survey how much of "real" MATLAB
Coder input the current convmat codegen subset covers.

Each file is compiled to C with:

```sh
cargo run -- examples/coder/<name>.m
```

The survey is automated in `tests/coder_examples.rs`. Every `.m` file here must
be listed in that test's `EXPECTED` table; the test fails if an example is
missing or its compile outcome changes, so it doubles as a coverage regression
guard. Examples that currently compile are also linked with a tiny `main` driver
and run.

## Coverage

| Example              | Compiles | Blocking construct (when not)              |
|----------------------|:--------:|--------------------------------------------|
| `addone`             |   yes    | —                                          |
| `mandelbrot_count`   |   yes    | —                                          |
| `averaging_filter`   |    no    | `isempty` / `mean` / array concatenation   |
| `fib`                |    no    | recursive user-function call               |
| `dijkstra`           |    no    | `Inf` / colon slices / elementwise `min`   |
| `kalmanfilter`       |    no    | `isempty` on a persistent variable         |
| `sierpinski`         |    no    | `plot` (graphics)                          |

## Provenance

These are adapted, self-contained reproductions of the corresponding MathWorks
examples, not verbatim copies:

- `addone` — command-line quick start ("Generate C Code from MATLAB Code").
- `averaging_filter` — the MATLAB Coder tutorial "Generate C Code from MATLAB
  Code".
- `fib` — recursive Fibonacci from the "recursive functions" code-generation
  documentation.
- `mandelbrot_count` — the Mandelbrot Set example (kept real-valued here; the
  original uses a complex `c`).
- `sierpinski` — the Sierpinski Triangle example.
- `dijkstra` — Dijkstra's shortest path example.
- `kalmanfilter` — the scalar Kalman filter example.

MATLAB and MATLAB Coder are trademarks of The MathWorks, Inc. This folder is
only for compiler testing.
