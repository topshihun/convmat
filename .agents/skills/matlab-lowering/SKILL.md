---
name: matlab-lowering
description: Guidance for lowering MATLAB/Octave constructs to MLIR dialects using the melior bindings. Use when implementing or reviewing the convmat code generation backend.
---

# MATLAB → MLIR Lowering

Use this when writing the `convmat` backend that translates MATLAB/Octave
semantics into MLIR.

## Target dialects

Start with the core MLIR dialects available through `melior`:

- `arith` — scalar arithmetic and type casts for elementwise math.
- `linalg` — named structured ops (matmul, reductions, generic) for array code.
- `func` — function definitions and calls.
- `cf` / `scf` — control flow and loops (`for`/`while` lowering).
- `memref` / `tensor` — dense arrays and their lowering path.
- `index` — loop induction and indexing.

Introduce a `convmat` dialect only after the core dialects prove insufficient
for MATLAB-specific semantics (e.g. colon indexing, `end`, dynamic typing).

## Mapping guidelines

1. **Dynamic typing**: MATLAB values are dynamically typed matrices. Carry a
   runtime type descriptor or boxed value until static analysis (via
   `runmat-static-analysis`) proves a concrete element type/shape, then lower
   to `memref`/`tensor`.
2. **Arrays**: map matrix ops to `linalg` on `memref<...x...>` after shape
   inference; fall back to a runtime-library call for shape-dependent code.
3. **Control flow**: `for` loops lower to `scf.for`; `while` to `scf.while`;
   `if`/`switch` to `scf.if`/`cf.cond_br`.
4. **Functions**: `function` files become `func.func`; respect MATLAB
   pass-by-value semantics when lowering arguments.
5. **Built-ins**: call into the runmat builtin/runtime layer first, and only
   inline or pattern-match a builtin once it is on a hot path and its
   semantics are verified.

## Verification

- Always round-trip the generated MLIR through `mlir-opt` to check validity.
- Add a test that parses a small `.m` snippet, lowers it, and checks the
  resulting MLIR text or a successful execution result.

Do not fabricate `melior` API surface. Consult the
[melior docs](https://mlir-rs.github.io/melior/melior/) and confirm every
builder call against the crate before committing.
