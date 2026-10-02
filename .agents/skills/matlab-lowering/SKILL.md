# MATLAB → IR Lowering

Use this when writing the `convmat` backend that translates MATLAB/Octave
semantics into the `matlab` pliron dialect (and, downstream, into the `emitc`
dialect and C).

## Target dialects

convmat defines two pliron dialects in `src/dialects/`:

- **`matlab`** — MATLAB-level semantics: dense column-major arrays
  (`matlab.array`), `f64`/`bool` scalars, `constant`/`bconst`, `binop`/`cmp`/
  `select`, `alloca`/`load`/`store`, `call` (to `libm`), and structured control
  flow (`if`/`while`/`for`/`condition`/`yield`/`return`).
- **`emitc`** — C-level primitives: `func`/`declare`/`literal`/`binop`/`cmp`/
  `ternary`/`call`/`load`/`assign`, and `if`/`while`/`for`/`condition`/`yield`/
  `break`/`return`.

The pipeline is `MIR -> matlab -> emitc -> C`. Introduce a new op only after the
two existing dialects prove insufficient for MATLAB-specific semantics (e.g.
colon indexing, `end`, dynamic typing).

## Mapping guidelines

1. **Dynamic typing**: MATLAB values are dynamically typed matrices. Carry a
   runtime type descriptor or boxed value until static analysis (via
   `runmat-static-analysis`) proves a concrete element type/shape, then lower
   to `matlab.array`.
2. **Arrays**: static shapes are tracked in `Shape`/`LocalTy` and stored as a
   flattened column-major `matlab.array` with shape metadata; elementwise,
   transpose, matmul, and reductions are expanded to `load`/`store` loops in
   the MIR lowerer (compile-time unrolled for static shapes). Reserve `linalg`
   for later fusion/vectorization, and defer dynamic shapes to the runtime.
3. **Control flow**: `for` loops lower to `while` with a loop-binding cell;
   `while` uses `condition` (before region) + `yield` (after region); `if`/
   `switch` use `if` with then/else regions (a nested if/else chain).
4. **Functions**: `function` files become `builtin.func`; scalar outputs return,
   array outputs become caller-allocated out-pointer parameters (the ABI in
   `docs/architecture.md` §5).
5. **Built-ins**: pure numeric built-ins lower to `matlab.call @libm` via the
   `src/builtins.rs` table; `sign` and `min`/`max` over scalars lower inline.
   Impure or dynamic built-ins defer to the runtime.

## Verification

- Round-trip the generated IR through `mir_to_mlir::lower` and check the dump
  (op names) rather than `mlir-opt` (no external tools are available).
- Add a test that parses a small `.m` snippet, lowers it, and checks the
  resulting IR dump or the emitted C.

Do not fabricate `pliron` API surface. Consult the
[pliron source](https://github.com/pliron-org/pliron) (especially
`examples/kaleidoscope`) and confirm every builder/op call against the crate
before committing.
