# convmat

A MATLAB/Octave-to-MLIR compiler written in Rust. The goal is to become a
better MATLAB Coder: parse MATLAB-style source with the
[RunMat](https://github.com/runmat-org/runmat) frontend, lower it to MLIR with
the [melior](https://github.com/mlir-rs/melior) bindings, and generate native
code.

## Status

End-to-end codegen works for a **scalar-double subset** with structured control
flow (`if` / `while` / `for` / `switch`):

```sh
cargo run -- tests/fixtures/add.m
```

```c
double add(double a, double b) { ... }
```

The codegen boundary (`src/triage`) defers anything it cannot statically lower.
The MVP has no runtime fallback yet, so a deferred function is reported as an
error instead of being compiled.

## Dependencies

The frontend and backend crates are **mandatory** dependencies (code generation
is the whole point of this crate, so they are not feature-gated):

| Crate           | Version | Role                              | Notes                                    |
|-----------------|---------|-----------------------------------|------------------------------------------|
| `runmat-parser` | 0.6.2   | Parse MATLAB/Octave tokens -> HIR | MIT                                      |
| `runmat-hir`    | 0.6.2   | High-level IR                     | MIT                                      |
| `runmat-mir`    | 0.6.2   | Mid-level IR (lowering boundary)  | MIT                                      |
| `melior`        | 0.27.8  | MLIR bindings (safe Rust wrapper) | Apache-2.0; requires a local MLIR build |

`melior` requires a local **MLIR 22** install (`libMLIR` + `libMLIR-C`) and
`mlir-translate` on `PATH` for the C backend. See the
[melior](https://github.com/mlir-rs/melior) build instructions.

> `runmat-static-analysis` is deliberately not used yet: it pulls
> `runmat-vm` -> `runmat-runtime` -> native HDF5/OpenBLAS, which the MVP does
> not need. Type/shape inference is deferred; scalars default to `double`.

## Layout

```
convmat/
├── src/
│   ├── main.rs            # thin CLI wrapper (arg parsing + pipeline)
│   ├── lib.rs             # library crate (pipeline docs + public API)
│   ├── frontend/          # layer 1: read `.m`, drive runmat -> MIR
│   ├── triage/            # layer 2: static vs dynamic classification
│   ├── mir_to_mlir/       # layer 3: MIR -> core-dialect MLIR
│   ├── passes/            # layer 4: MLIR pass pipeline (-> emitc)
│   ├── backend/           # layer 5: emitc -> C (LLVM/GPU reserved)
│   └── runtime/           # layer 6: runtime fallback seam (not implemented)
└── tests/                 # end-to-end codegen tests + fixtures
```

See `AGENTS.md` for the authoritative architecture.

## Building and testing

```sh
cargo check                  # type-check
cargo test --all-targets     # unit + end-to-end codegen tests
cargo clippy --all-targets -- -D warnings
```

## Not yet implemented

- Runtime fallback for deferred functions (`src/runtime` is a seam only).
- `runmat-static-analysis`-driven triage (type/shape inference, definite
  assignment); the current boundary is a hand-written operator whitelist.
- Array / matrix / `linalg` / `tensor` lowering (locals are scalar `memref`).
- LLVM and GPU backends (`src/backend` reserves `Llvm`/`Gpu`).
- Optimization passes (`canonicalize` / `CSE` / `linalg` transforms).
