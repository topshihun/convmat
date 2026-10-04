# convmat

A MATLAB/Octave-to-C compiler written in Rust. The goal is to become a
better MATLAB Coder: parse MATLAB-style source with the
[RunMat](https://github.com/runmat-org/runmat) frontend, lower it to a
[pliron](https://github.com/pliron-org/pliron)-based IR, and generate C.

## Status

End-to-end codegen works for a growing subset of MATLAB:

- **Scalars** (`double`) with arithmetic / comparison / logical operators,
  structured control flow (`if` / `elseif` / `while` / `for` / `switch`), and
  multiple return values.
- **Statically-shaped arrays and matrices** (row / column vectors, 2-D matrices,
  stored column-major): elementwise operators with scalar broadcast (incl. `.^`),
  2-D transpose, matrix multiply, and integer matrix power `A^k`.
- **Full operator coverage**: every `OperatorKind` binary/unary/relational/logical
  operator is lowerable; `^` / `.^` lower to `libm::pow` (scalar) or a `convmat_mpower`
  runtime call (square-matrix integer power).
- **Wrapped runtime helpers**: transpose (`convmat_transpose`), matrix multiply
  (`convmat_matmul`), integer matrix power (`convmat_mpower`), square-matrix
  inverse/determinant/left-division (`convmat_inv`/`convmat_det`/`convmat_solve`),
  vector 2-norm (`convmat_norm`) and `rand` (`convmat_rand`) are emitted as
  calls to a small C runtime library (`src/runtime`) — emitted only when used —
  instead of unrolled loops; see `docs/architecture.md` §10.2.1.
- **Pure numeric built-ins** (`sin`/`cos`/`sqrt`/`abs`/`floor`/…,
  `sum`/`prod`/`min`/`max` with an optional dimension, `numel`/`length`/`size`,
  `zeros`/`ones`/`eye`, `reshape`, `sort`) lowered to `libm` calls.
- **Statistics, differences and predicates**: `mean`/`std`/`median`/`var` (vector
  reductions to a scalar), `cumsum`/`diff` (array results), and
  `isnan`/`isinf`/`isempty`/`logical` and `Inf`/`NaN` literals.
- **Linear algebra & random** (static square matrices / vectors): `inv`, `det`,
  `norm` (vector 2-norm), `A \ b` (square linear solve) and `rand()`.
- **Array constructors & shape ops**: block concatenation `[a b; c d]`, implicit
  singleton expansion (`A + b`), slices `A(i,:)`/`A(:,j)` and ranges `A(a:b:c)`,
  N-D arrays, `linspace`, `repmat`, `permute`.
- **Structs** (scalar fields): `struct('a', 1, …)` construction, `s.a` read/write,
  and struct pass-by-value parameters / returns. Struct parameter layouts are
  inferred from field use (`s.a`), matching MATLAB Coder's use-site inference.
  Nested field paths (`s.a.b`) are flattened to a single C field (`a__b`), and a
  struct local built field-by-field (with no `struct(...)`) is typed from its
  field writes.
- **Char literals & `strcmp`**: a 1-char literal is a scalar code point (so
  `switch c; case 'a'` compares code points) and a longer literal is a `1xN`
  code-unit array; `strcmp` of two char literals folds to a logical scalar.
- **`int32` scalars**: `int32(x)` conversion plus wraparound `+`/`-`/`*` via
  `convmat_int32`/`convmat_iadd`/`convmat_isub`/`convmat_imul`; mixing `int32`
  with `double` is deferred.
- **Dynamic-shape array parameters** as `(double* data, double n)`: reduce
  (`sum`/`prod`/`min`/`max`), `numel`/`length`, runtime indexing `A(i)` / `end`, and
  runtime-bound loops (`for i = 1:numel(A)`) over a parameter whose size is only
  known at run time. Parameters whose size is queried (`size(A)` / `size(A, d)`)
  instead use a `(double* data, double rows, double cols)` **shape-descriptor** ABI
  (usage-driven, so lean parameters keep the shorter signature). **Dynamic-shape
  array outputs** use an out-buffer plus an
  out-length cell the callee fills (`y = A(:)`, `y = -A(:)`, `y = k * A`,
  `y = A(:) + B(:)`, `y = A(:) .* B(:)`). **Dynamic-shape array intermediates**
  are heap-allocated at run time (`matlab.heap_alloc` → `new double[n]`) and freed
  at the end of the block that allocated them (block-scoped, so a `t = A(:)`
  inside a loop allocates and frees each iteration); they support scalar broadcast
  (`+`/`-`/`.*`/`./`) and nested dynamic subexpressions (`y = A .* A + n`); see
  `docs/architecture.md` §5, §10.5 P7 and `docs/runtime.md` §8–§9.
- **Indexing**: constant subscript `A(i,j)`, linear `A(i)`, `end`, `A(:)`, and logical
  indexing `A(A > 0)` / masked assignment `A(A < 0) = 0` (a runtime-sized result via
  `convmat_mask_gather`/`convmat_mask_assign`).
- **`try`/`catch`**: the `matlab.try` op lowers to a `setjmp`-based handler stack
  (`convmat_error_enter`/`check`/`leave`/`throw`); the `try` body is the `if` branch
  and `catch` the `else` branch. See `docs/runtime.md` §7.
- **Cell arrays** (scalar elements): `{1, 2, 3}` builds a boxed `convmat_value`
  cell and `c{i}` reads a scalar element; the boxed value-model kernel
  (`docs/runtime.md`) is emitted on demand, and cells are released block-scoped.
- **Complex numbers**: `i` / `3 + 4i` and complex arithmetic lower to runtime
  helpers (`convmat_complex`/`cadd`/`csub`/`cmul`/`cdiv`); `abs` (magnitude),
  `fft` and `eig` (1x1/2x2) are runtime built-ins, and a complex result is
  returned as a boxed `convmat_value*`. See `docs/runtime.md`.
- **Variadic arguments (closed-world specialization)**: `nargin`/`nargout` fold to
  compile-time constants; `varargin{k}` (constant `k`) resolves to the `k`-th extra
  scalar input; `varargout{k} = scalar` resolves to the `k`-th extra scalar output.
  See `docs/architecture.md` §12.

- **Anonymous functions** (non-escaping, closed-world): `f = @(x) x.^2 + a;`
  assigned in the same function and only called as `f(args)` is specialized at
  compile time to a helper whose captured variables are passed as extra
  arguments, snapshotted at creation (MATLAB capture-by-value semantics); see
  `docs/architecture.md` §13.
- **Closed-world user-function calls** within one file, for the scalar ABI (all
  scalar inputs, a single scalar output), including recursion. The C emitter
  forward-declares every function so calls to later-defined functions resolve.
- **Layered, dialect-based optimizations** (`src/passes`, on pliron's pass framework):
  `matlab`-dialect semantic passes (constant folding, scalar-cell constant
  propagation, dead-branch elimination, dead-value elimination) run after
  `hir_to_mlir`; `emitc`-dialect C-level passes (dead write-only cells, dead values)
  run after `lowering`. Each pass is independent; the pipeline repeats to a fixpoint.
  Loop conditions are never folded against their pre-loop value.

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

The frontend and IR crates are **mandatory** dependencies (code generation is
the whole point of this crate, so they are not feature-gated):

| Crate           | Version | Role                              | Notes                                    |
|-----------------|---------|-----------------------------------|------------------------------------------|
| `runmat-parser` | 0.6.2   | Parse MATLAB/Octave tokens -> HIR | MIT                                      |
| `runmat-hir`    | 0.6.2   | High-level IR                     | MIT                                      |
| `pliron`        | 0.18    | Extensible compiler IR (pure Rust) | Apache-2.0; no C++ MLIR/LLVM needed     |

The build is pure `cargo build` — no local MLIR/LLVM install, no `libMLIR`,
no `mlir-translate`/`mlir-opt` on `PATH`.

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
│   ├── frontend/          # layer 1: read `.m`, drive runmat -> HIR
│   ├── triage/            # layer 2: static vs dynamic classification + shape inference
│   ├── dialects/          # layer 2.5/4.5: `matlab` and `emitc` pliron dialects
│   ├── hir_to_mlir/       # layer 3: HIR -> matlab dialect
│   ├── builtins.rs        # builtin name -> lowering recipe table
│   ├── lowering.rs        # layer 4: matlab -> emitc dialect
│   ├── passes/            # dialect-scoped IR optimizations (matlab / emitc)
│   ├── emit_c.rs          # layer 5: emitc -> C
│   ├── backend/           # backend selection (C today; LLVM/GPU reserved)
│   └── runtime/           # layer 6: runtime fallback seam (not implemented)
├── docs/
│   └── architecture.md    # authoritative architecture
├── examples/
│   └── coder/             # MATLAB Coder example coverage survey
├── tests/                 # end-to-end codegen tests + fixtures
└── AGENTS.md              # agent working conventions
```

See `docs/architecture.md` for the authoritative architecture; `docs/roadmap.md` for
the plan that turns the remaining `examples/coder` failures green; `AGENTS.md` holds
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
- Full matrix linear algebra: `mrdivide` (`/`) at the matrix level and
  `mldivide` (`\`) for non-square/least-squares systems, `scalar ^ matrix`
  (`expm`), `matrix ^ matrix`, non-integer/negative matrix power, array-array
  broadcasting, N-D transpose.
- Remaining shape transforms: `permute`/`repmat`/`cat`/`horzcat`/`vertcat`.
- Advanced indexing: `A(i,:)` / `A(:,j)` slices, colon ranges, variable indices.
- Optimization passes (CSE / canonicalize / linalg fusion / vectorization); the
  pipeline today emits straightforward, unoptimized C.
- `varargin{k}` with a variable index, `varargout{k}` with an array value, and
  `varargin{:}` expansion (runtime cell ABI); the closed-world fixed-arity
  specialization of `nargin`/`nargout`/`varargin`/`varargout` is implemented.
- LLVM and GPU backends (`src/backend` reserves `Llvm`/`Gpu`).
