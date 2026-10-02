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

## pliron 0.18 API quick reference

Verified against docs.rs (crate pages: `pliron` 0.18.0, `pliron-derive` 0.18.0).
Re-verify against the docs before inventing new calls, as the API is pre-1.0.

### Derive macros (in `pliron::derive`)

- `#[pliron_op(...)]`: defines an Op. Options: `name` (required,
  `"dialect.op_name"`), `format` (bare or a format string), `interfaces = [...]`,
  `attributes = (field: Type, ...)`, `operands = (...)`, `results = (...)`, and
  `verifier = "succ"`. Expands to `def_op` + `format_op` +
  `derive_attr_get_set` + `derive_op_interface_impl` etc. The struct is auto-
  derived `Clone, Copy, Hash, PartialEq, Eq`; it gets an `op: Ptr<Operation>`
  field, so declare the user struct as `pub struct FooOp;`.
- `#[pliron_attr(...)]`: defines an Attribute. Options: `name`, `format`,
  `verifier = "succ"`.
- `#[pliron_type(...)]`: defines a Type. Options: `name`, `format`,
  `generate_get = true/false`, `verifier = "succ"`.
- `#[derive_attr_get_set]`-generated getters/setters are named
  `get_attr_<field>` / `set_attr_<field>` and return `Option<Ref<'_, T>>`.
- Registration is automatic via `linkme`/`inventory` (no explicit
  `Context` dialect registration needed).

### Builtin attributes (`pliron::builtin::attributes`)

`StringAttr` (`.new(String)`, `From<&str>`/`From<String>`, `From<StringAttr> for String`),
`BoolAttr` (`.new(bool)`, `From<bool>`, `From<BoolAttr> for bool`),
`FPDoubleAttr` (`From<f64>`), `IntegerAttr`, `VecAttr(Vec<AttrObj>)`,
`DictAttr`, `TypeAttr`, `UnitAttr`, `IdentifierAttr`.

### Builtin op interfaces (`pliron::builtin::op_interfaces`)

`NOpdsInterface<N>` (exactly N operands), `NResultsInterface<N>`,
`NRegionsInterface<N>`, `OneOpdInterface`, `OneResultInterface`,
`OneRegionInterface`, `IsTerminatorInterface`, `SymbolOpInterface`,
`SingleBlockRegionInterface`, `NoTerminatorInterface`. Zero-operand/result,
non-terminator ops use `[NOpdsInterface<0>, NResultsInterface<0>]`.

### Block / region / insertion

- `BasicBlock::new(ctx: &mut Context, label: Option<Identifier>, arg_types:
  Vec<TypeHandle>) -> Ptr<BasicBlock>`.
- `block.insert_at_front(region: Ptr<Region>, ctx: &Context)` inserts the block
  into the region (note: takes `&Context`, not `&mut Context`).
- `Region::get_entry_block(&self) -> Option<Ptr<BasicBlock>>` (inherent, no ctx).
- `block.deref(ctx).iter(ctx)` yields `Ptr<Operation>` (needs
  `pliron::linked_list::ContainsLinkedList` in scope).
- Append an op: `IRInserter::<DummyListener>::new_at_block_end(block)
  .append_op(ctx, op)` (`pliron::irbuild::inserter::{IRInserter, Inserter}`,
  `pliron::irbuild::listener::DummyListener`; `append_op` needs `&Context`).
- Ops are created detached: `Operation::new(ctx, Self::get_concrete_op_info(),
  result_tys, operands, successors, num_regions)`.
- Builtin `FuncOp::new(ctx, name: Identifier, ty: TypedHandle<FunctionType>)`
  creates a single region with an empty entry block (`get_entry_block(ctx)`).
  `ModuleOp::new(ctx, name: Identifier)` creates a single region + block; add a
  func via `module.append_operation(ctx, func.get_operation(), 0)` (needs
  `SingleBlockRegionInterface` in scope).

### The `emitc` top level

`emitc` contributes **only** C-level body ops + top-level directives; the
`module`/`func` containers are pliron builtins.
- Builtin `ModuleOp`/`FuncOp` hold the structure; `lowering::lower_module`
  returns a builtin `ModuleOp` whose `builtin.func` bodies are `emitc.*` ops.
- `emitc::IncludeOp` (`#include`, `header` + `system` flag),
  `emitc::DefineOp` (`#define NAME [VALUE]`, empty value = bare `#define NAME`),
  `emitc::UndefOp` (`#undef NAME`), and `emitc::VerbatimOp` (raw top-level C
  text; the escape hatch for `typedef`/`struct`/`#pragma`/`extern`/...).
- `emit_c::emit` takes `&builtin::ModuleOp`, prints the fixed stdlib preamble
  (`<cmath>`/`<cstdint>`/`<tuple>`), then each top-level op in source order.
