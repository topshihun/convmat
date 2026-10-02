# convmat Development

You are working on `convmat`, a Rust compiler that lowers MATLAB/Octave source
to C and aims to be a better MATLAB Coder.

## Project facts

- Package: `convmat` (a lib + bin crate in a single `Cargo.toml`).
- Edition 2021.
- Frontend dependencies (all 0.6.2, mandatory): `runmat-parser`,
  `runmat-hir`, `runmat-mir`.
- IR dependency: `pliron` 0.18 (a pure-Rust, MLIR-inspired compiler IR
  framework). There is **no** C++ MLIR/LLVM dependency and no `melior`.
- Authoritative architecture lives in `docs/architecture.md`; agent working
  conventions live in `AGENTS.md` (project root). Follow both and keep this
  skill in sync.
- `runmat-static-analysis` is intentionally excluded: it pulls
  runmat-vm -> runmat-runtime -> native HDF5/OpenBLAS, which the MVP does not
  need. Shape inference is lightweight and local (`src/triage::infer_locals`);
  array shapes are static-only (from literals) and array parameters default to
  scalars.
- `runmat`/`pliron` are mandatory (not feature-gated): code generation is the
  whole point of the crate.

## Pipeline

```
.m source
  -> runmat (lexer/parser/HIR/MIR)            [src/frontend]
  -> triage (static vs dynamic)               [src/triage]
  -> MIR -> matlab dialect (pliron)           [src/mir_to_mlir]
  -> matlab -> emitc dialect (pliron)         [src/lowering]
  -> emitc -> C                               [src/emit_c]
```

Two custom pliron dialects are defined in `src/dialects/`:
`matlab` (semantics) and `emitc` (C-level). Every op is declared with the
`#[pliron_op]`/`#[pliron_type]`/`#[pliron_attr]` derive macros.

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
   before implementing. Never invent API calls into `runmat` or `pliron`
   without verifying them against the crate docs.
3. When you need a dependency, prefer the versions already declared
   (`runmat` 0.6.2, `pliron` 0.18) unless there is a specific reason to bump.
4. `pliron` op attribute names are globally unique `dict_key`s; name new
   attributes `<dialect>_<op>_<field>` to avoid collisions.
5. For lowering questions, follow the `matlab-lowering` skill.
