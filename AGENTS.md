# convmat · Agent 工作约定

> convmat 是把 MATLAB/Octave 源码编译成 C（优先）的编译器：前端复用 runmat，
> 中后端复用 melior/MLIR。**权威架构见 `docs/architecture.md`**；本文件只写
> Agent 工作约定与工程规范，做架构/设计类改动前先读 `docs/architecture.md`。

## 模块速查

| 模块 | 职责 |
|------|------|
| `src/frontend/` | 读 `.m` + 驱动 runmat 前端（parse → HIR → MIR） |
| `src/triage/` | 静态 vs 动态边界 + 形状推断（`Shape`/`LocalTy`） |
| `src/mir_to_mlir/` | MIR → 核心方言 MLIR（降级器，含内存分配决策） |
| `src/builtins.rs` | 内建函数名 → 降级配方表 |
| `src/passes/` | MLIR passes（canonicalize/CSE/convert-to-emitc） |
| `src/backend/` | 后端策略（今天 `emitc` → C，`llvm`/`gpu` 预留） |
| `src/runtime/` | 运行时兔底（预留，MVP 报错） |

## 依赖与版本

- `runmat-*` 0.6.2（前端）、`melior` 0.27.8（MLIR 绑定）；升级需说明理由。
- 不重复造轮子：解析/类型/形状推断用 runmat；IR 优化/验证/后端发射用 MLIR + melior；
  矩阵/向量化用 `linalg`（尚未接入）。
- 允许自研的只有：MIR→MLIR 降级器、静态/动态分类（triage）、运行时 shim。

## 工作约定

1. 改前端接入或 MIR 依赖时，集中在 `src/frontend/` 与 `src/mir_to_mlir/`，隔离 runmat
   版本演进的影响。
2. 新增后端实现 `src/backend/` 的策略，不要改动降级器和前端。
3. 想引入自定义方言前，先确认「核心方言 + 运行时调用」确实无法表达；否则一律否。
4. 不要臆造 `runmat`/`melior` 的 API，落地前查对应 crate 文档（runmat 目前 pre-1.0，
   API 可能变化）。
5. 与 runmat/MIR 相关的类型只出现在 `src/frontend/` 与 `src/mir_to_mlir/`，
   其余模块不得直接引用 runmat 类型。

## 代码风格

- edition 2021；公共 API 写 `///` doc comment，模块写 `//!` 头注释。
- 除 `melior`/`runmat` 安全绑定外避免 `unsafe`。
- 错误处理：库代码用 `thiserror`，CLI 入口用 `anyhow` 兜底。
- `src/main.rs` 只做薄封装，逻辑放 `src/lib.rs`。

## 测试

- 单元测试放 `#[cfg(test)] mod tests`；集成测试放 `tests/`。
- 每个新增的 MATLAB→MLIR 降级 pattern 配一个「最小 `.m` → MLIR → 结果」测试。
- 生成的 MLIR 用 `mlir-opt` 跑一遍验证（失败即测试失败）。

## 构建 / 测试 / 提交前检查

```sh
cargo run -- tests/fixtures/add.m   # 编译 fixture 到 C
cargo check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

提交前依次执行并确保全部通过（GitHub CI 同样执行）：

1. `cargo fmt --all`（CI 用 `cargo fmt --all -- --check`）
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo test --all-targets`
4. `cargo check`

> 依赖本机安装 MLIR 22（`melior` 需要 libMLIR + libMLIR-C），且 `mlir-opt`/`mlir-translate`
> 在 `PATH` 上。

## 提交

- 遵循个人 AGENTS.md 提交规范（祈使句、50 字符主题、72 字符正文）。
- 一个提交只做一件事；架构改动同步更新 `docs/architecture.md`，本文件只改 Agent 约定。

> 关于具体 MATLAB→MLIR 映射参考 `matlab-lowering` skill；构建/测试与依赖版本参考
> `convmat-dev` skill。
