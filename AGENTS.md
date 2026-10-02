# convmat · Agent 工作约定

> convmat 是把 MATLAB/Octave 源码编译成 C（优先）的编译器：前端复用 runmat，
> 中后端复用 pliron（纯 Rust 的 MLIR 式 IR 框架）。**权威架构见
> `docs/architecture.md`**；本文件只写 Agent 工作约定与工程规范，做架构/设计类改动前
> 先读 `docs/architecture.md`。

## 模块速查

| 模块 | 职责 |
|------|------|
| `src/frontend/` | 读 `.m` + 驱动 runmat 前端（parse → HIR → MIR） |
| `src/triage/` | 静态 vs 动态边界 + 形状推断（`Shape`/`LocalTy`） |
| `src/dialects/matlab.rs` | `matlab` 方言（语义）：数组类型 + 标量/数组/控制流 op |
| `src/mir_to_mlir/` | MIR → `matlab` 方言（降级器，含内存分配决策） |
| `src/builtins.rs` | 内建函数名 → 降级配方表 |
| `src/lowering.rs` | `matlab` → `emitc` 方言降级（ABI 解析、数组命名、op 重写） |
| `src/dialects/emitc.rs` | `emitc` 方言（C 级）：声明/赋值/三元/调用/控制流 |
| `src/emit_c.rs` | `emitc` → C（pretty-printer；LLVM/GPU 预留） |
| `src/backend/` | 后端策略（今天 C，`llvm`/`gpu` 预留） |
| `src/runtime/` | 运行时兔底（预留，MVP 报错） |

## 依赖与版本

- `runmat-*` 0.6.2（前端）、`pliron` 0.18（IR 骨架，纯 Rust）；升级需说明理由。
- 不重复造轮子：解析/名字解析用 runmat；IR 骨架（SSA/region/block/dialect/op/type/
  verifier/pass 框架）用 pliron；C 发射器与两级降级器自研。
- 不再依赖 melior/libMLIR/`mlir-translate`/`mlir-opt`；构建是纯 `cargo build`。
- 允许自研的只有：MIR→`matlab` 降级器、`matlab`→`emitc` 降级器、C 发射器、
  静态/动态分类（triage）、运行时 shim。**优化 pass（CSE/canonicalize/linalg）本期不做。**

## 工作约定

1. 改前端接入或 MIR 依赖时，集中在 `src/frontend/` 与 `src/mir_to_mlir/`，隔离 runmat
   版本演进的影响。
2. 新增后端实现 `src/emit_c.rs`（或 `src/backend/`）的策略，不要改动降级器和前端。
3. 新增 `matlab`/`emitc` op 前，先确认「现有方言 + 运行时调用」确实无法表达；否则一律否。
   方言 op 只覆盖当前可生成代码子集需要的语义，不追求语法完整。
4. 不要臆造 `runmat`/`pliron` 的 API，落地前查对应 crate 文档（两者目前都是 pre-1.0，
   API 可能变化）。
5. 与 runmat/MIR 相关的类型只出现在 `src/frontend/` 与 `src/mir_to_mlir/`，
   其余模块不得直接引用 runmat 类型。
6. pliron 的 attribute 名（`dict_key`）必须**全局唯一**（跨所有方言）；新属性名用
   `<方言>_<op>_<字段>` 命名，避免与既有 op 冲突。

## 代码风格

- edition 2021；公共 API 写 `///` doc comment，模块写 `//!` 头注释。
- 除 `pliron`/`runmat` 安全绑定外避免 `unsafe`。
- 错误处理：库代码用 `thiserror`，CLI 入口用 `anyhow` 兜底。
- `src/main.rs` 只做薄封装，逻辑放 `src/lib.rs`。

## 测试

- 单元测试放 `#[cfg(test)] mod tests`；集成测试放 `tests/`。
- 每个新增的 MATLAB→IR 降级 pattern 配一个「最小 `.m` → IR → C」测试。
- 生成 IR 用自研 dump（`mir_to_mlir::lower`）做结构断言，不再依赖 `mlir-opt`。

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

> 纯 Rust 构建，无需本机安装 MLIR/LLVM 或任何外部工具。

## 提交

- 遵循个人 AGENTS.md 提交规范（祈使句、50 字符主题、72 字符正文）。
- 一个提交只做一件事；架构改动同步更新 `docs/architecture.md`，本文件只改 Agent 约定。

> 关于具体 MATLAB→IR 映射参考 `matlab-lowering` skill；构建/测试与依赖版本参考
> `convmat-dev` skill。
