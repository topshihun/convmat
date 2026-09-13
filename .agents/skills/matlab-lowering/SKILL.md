---
name: matlab-lowering
description: Guidance for lowering MATLAB/Octave constructs to MLIR dialects using the melior bindings. Use when implementing or reviewing the convmat code generation backend.
---

# MATLAB → MLIR Lowering

Use this when writing the `convmat` backend that translates MATLAB/Octave
semantics into MLIR.

## Target dialects

melior 0.27.8 exposes **typed builders** for these dialects:

- `arith` — scalar arithmetic / comparisons / casts / select.
- `func` — function definitions and calls.
- `scf` / `cf` — structured control flow (`if`/`while`/`for`).
- `memref` — dense arrays (`alloca` / `alloc` / `load` / `store`).
- `index` — loop induction and indexing.

`linalg` / `tensor` / `math` / `emitc` have **no typed builders**, but their
**pass factories are available** (`melior::pass::{linalg,tensor,math,conversion}`).
Emit those ops via the generic `OperationBuilder` (by op name) when needed, and
drive them with the corresponding passes.

Introduce a `convmat` dialect only after the core dialects prove insufficient
for MATLAB-specific semantics (e.g. colon indexing, `end`, dynamic typing).

## Mapping guidelines

1. **Dynamic typing**: MATLAB values are dynamically typed matrices. Carry a
   runtime type descriptor or boxed value until static analysis (via
   `runmat-static-analysis`) proves a concrete element type/shape, then lower
   to `memref`/`tensor`.
2. **Arrays**: static shapes are tracked in `Shape`/`LocalTy` and stored as a
   flattened column-major `memref<nxf64>` with shape metadata; elementwise,
   transpose, matmul, and reductions lower to `scf`/`arith` directly (matmul and
   reductions are compile-time unrolled for static shapes). Reserve `linalg` for
   later fusion/vectorization, and defer dynamic shapes to the runtime.
3. **Control flow**: `for` loops lower to `scf.for`; `while` to `scf.while`;
   `if`/`switch` to `scf.if` (a nested if/else chain).
4. **Functions**: `function` files become `func.func`; scalar outputs return,
   array outputs become caller-allocated out-pointer parameters (the ABI in
   `docs/architecture.md` §5).
5. **Built-ins**: pure numeric built-ins lower to `func.call @libm` via the
   `src/builtins.rs` table (with a private `func.func` declaration emitted once);
   `sign` and `min`/`max` over scalars lower inline. Impure or dynamic built-ins
   defer to the runtime.

## Verification

- Always round-trip the generated MLIR through `mlir-opt` to check validity.
- Add a test that parses a small `.m` snippet, lowers it, and checks the
  resulting MLIR text or a successful execution result.

Do not fabricate `melior` API surface. Consult the
[melior docs](https://mlir-rs.github.io/melior/melior/) and confirm every
builder call against the crate before committing.
