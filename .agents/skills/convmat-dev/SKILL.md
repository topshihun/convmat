---
name: convmat-dev
description: Build workflow and project conventions for the convmat compiler (MATLAB/Octave to MLIR via runmat + melior). Use when scaffolding modules, wiring dependencies, or running/building/testing convmat.
---

# convmat Development

You are working on `convmat`, a Rust compiler that lowers MATLAB/Octave source
to MLIR and aims to be a better MATLAB Coder.

## Project facts

- Package: `convmat` (a lib + bin crate in a single `Cargo.toml`).
- Edition 2021, but `melior 0.27.8` targets edition 2024, so keep the local
  toolchain recent (Rust 1.85+).
- Frontend dependencies (all 0.6.2, mandatory): `runmat-parser`,
  `runmat-hir`, `runmat-mir`.
- Authoritative architecture and conventions live in `AGENTS.md` (project root);
  follow it and keep this skill in sync.
- Backend dependency: `melior` 0.27.8 (safe MLIR bindings). Requires a local
  MLIR/LLVM 22 install (`libMLIR` + `libMLIR-C`) and `mlir-translate` on
  `PATH` for the C backend.
- `runmat-static-analysis` is intentionally excluded: it pulls
  runmat-vm -> runmat-runtime -> native HDF5/OpenBLAS, which the MVP does not
  need. Type/shape inference is deferred; scalars default to `double`.
- `runmat`/`melior` are mandatory (not feature-gated): code generation is the
  whole point of the crate.

## Commands

```sh
cargo run -- tests/fixtures/add.m   # compile a fixture to C
cargo check                        # type-check without linking
cargo test --all-targets           # unit + end-to-end codegen tests
cargo clippy --all-targets -- -D warnings
```

Do not commit `target/`.

## Conventions

1. Put library code in `src/lib.rs` and re-export it through modules; keep
   `src/main.rs` a thin wrapper.
2. Add new pipeline stages as modules with doc comments explaining their role
   before implementing. Never invent API calls into `runmat` or `melior`
   without verifying them against the crate docs.
3. When you need a dependency, prefer the versions already declared
   (`runmat` 0.6.2, `melior` 0.27.8) unless there is a specific reason to bump.
4. For lowering questions, follow the `matlab-lowering` skill.
