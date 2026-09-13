# convmat

A MATLAB/Octave-to-MLIR compiler written in Rust. The goal is to become a
better MATLAB Coder: parse MATLAB-style source with the
[RunMat](https://github.com/runmat-org/runmat) frontend, lower it to MLIR with
the [melior](https://github.com/mlir-rs/melior) bindings, and generate C.

## Status

End-to-end codegen works for a growing subset of MATLAB:

- **Scalars** (`double`) with arithmetic / comparison / logical operators,
  structured control flow (`if` / `elseif` / `while` / `for` / `switch`), and
  multiple return values.
- **Statically-shaped arrays and matrices** (row / column vectors, 2-D matrices,
  stored column-major): elementwise operators with scalar broadcast, 2-D
  transpose, and matrix multiply.
- **Pure numeric built-ins** (`sin`/`cos`/`sqrt`/`abs`/`floor`/…,
  `sum`/`prod`/`min`/`max` with an optional dimension, `numel`/`length`/`size`,
  `zeros`/`ones`/`eye`, `reshape`) lowered to `libm` `func.call`s.
- **Indexing**: constant subscript `A(i,j)`, linear `A(i)`, `end`, and `A(:)`.

```sh
cargo run -- tests/fixtures/add.m
```

```c
double add(double v1, double v2) { ... }
```

The codegen boundary (`src/triage`) classifies each function as statically
lowerable (`Static`) or deferred to the runtime (`Deferred`). The MVP has no
runtime fallback yet, so a deferred function is reported as an error instead of
being compiled.

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
> not need. Shape inference is lightweight and local (`src/triage`): array
> shapes are static-only (derived from literals), and array parameters default
> to scalars. See `docs/architecture.md` §10.5 for the trade-off.

## Layout

```
convmat/
├── src/
│   ├── main.rs            # thin CLI wrapper (arg parsing + pipeline)
│   ├── lib.rs             # library crate (pipeline + public API)
│   ├── frontend/          # layer 1: read `.m`, drive runmat -> MIR
│   ├── triage/            # layer 2: static vs dynamic classification + shape inference
│   ├── mir_to_mlir/       # layer 3: MIR -> core-dialect MLIR
│   ├── builtins.rs        # builtin name -> lowering recipe table
│   ├── passes/            # layer 4: MLIR pass pipeline (canonicalize/CSE/emitc)
│   ├── backend/           # layer 5: emitc -> C (LLVM/GPU reserved)
│   └── runtime/           # layer 6: runtime fallback seam (not implemented)
├── docs/
│   └── architecture.md    # authoritative architecture
├── tests/                 # end-to-end codegen tests + fixtures
└── AGENTS.md              # agent working conventions
```

See `docs/architecture.md` for the authoritative architecture; `AGENTS.md` holds
agent working conventions.

## Building and testing

```sh
cargo run -- tests/fixtures/add.m   # compile a fixture to C
cargo check                        # type-check without linking
cargo test --all-targets           # unit + end-to-end codegen tests
cargo clippy --all-targets -- -D warnings
```

## Not yet implemented

- Runtime fallback for deferred functions (`src/runtime` is a seam only).
- `runmat-static-analysis`-driven type/shape inference (dynamic shapes); the
  current boundary is a lightweight local inference over the static-from-literal
  subset.
- Full matrix linear algebra: `mrdivide`/`mldivide` (`/` `\`), `^` (mpower),
  array-array broadcasting, N-D transpose.
- Remaining shape transforms: `permute`/`repmat`/`cat`/`horzcat`/`vertcat`.
- Advanced indexing: `A(i,:)` / `A(:,j)` slices, colon ranges, variable indices.
- `linalg` optimization (fusion / tiling / vectorization); the pipeline today
  runs only `canonicalize` + `cse` before the emitc conversion.
- `varargin` / `varargout` (closed-world specialization or runtime cell ABI).
- LLVM and GPU backends (`src/backend` reserves `Llvm`/`Gpu`).
