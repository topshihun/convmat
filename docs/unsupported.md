# 未支持的 MATLAB 特性清单

> 本文档是「可生成代码子集」的补集：凡是超出 `src/triage` 白名单、`src/builtins`
> 表、`src/mir_to_mlir` 可降级范围、以及 `docs/architecture.md` §10.5 路线表
> 已实现项之外的语言构造，都在此列出。每一项标注**当前行为**（报错信息 / defer）与
> **检测位置**（代码路径）。
>
> 权威边界判定见 `docs/architecture.md` §4；可变参数与动态形状的设计见 §11、§12。

## 1. 函数与调用

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `varargin{k}` 变量下标 | defer（无法特化） | `triage::rvalue_ty` / `mir_to_mlir::lower_index_scalar` |
| `varargout{k}` 赋数组 | defer（仅标量特化） | `triage::stmt_reason` |
| 参数展开 `{:}` / cell 展开 | 报错 `argument expansion ({:}/varargin) is not supported` | `triage::call_reason` |
| 匿名函数句柄 `f = @(x) …`（同作用域、不逃逸、标量参数/捕获） | 支持：编译期特化 + 捕获按创建时快照作为额外形参（见 `docs/architecture.md` §4、§13） | `triage::analyze_handles`、`hir_to_mlir::lower_handle_creation`/`lower_handle_call` |
| 逃逸的函数句柄（作为返回值/实参、存入容器、赋值给其他变量、参与非调用表达式）、在控制流内定义的句柄 | 报错 `function handle N escapes …` / `… defined inside control flow …` | `triage::analyze_handles`、`triage::check_handle_uses_*` |
| 匿名函数的数组实参/捕获/返回值 | 报错 `arguments must be scalar` / `captures must be scalar` / `must return a scalar` | `hir_to_mlir::lower_handle_call`/`lower_handle_creation` |
| 命名函数句柄 `@f`、内建句柄 `@sin`、立即调用 `(@(x) …)(3)` | 报错 `unsupported expression FunctionHandle(...)` / `unsupported expression AnonymousFunction(...)` | `triage::expr_reason`（非赋值位置的 `AnonymousFunction`/`FunctionHandle`） |
| 动态 / 非静态调用（`feval`、多返回值句柄调用、字符串调用） | 报错 `dynamic or non-static function call is not supported` | `triage::call_reason`（`call_name` 返回 `None`） |
| `eval` / `evalin` / `assignin` | 边界外构造（设计上排除，见 §4） | 架构决策，未显式检测 |

## 2. 动态类型与容器

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| cell 数组字面量 `{...}` | 报错 `cell array literals are not supported yet` | `triage::rvalue_reason`（`MirAggregateKind::Cell`） |
| 花括号 `{}` 索引（cell 索引） | 报错 `only paren indexing is supported` | `mir_to_mlir::lower_index_scalar`（`IndexKind != Paren`） |
| 动态字段 / struct 动态访问 | 不可静态解析 | 架构 §4 |
| `classdef` 动态分派 | 不可静态解析 | 架构 §4 |
| `global` / `persistent` 变量 | 不可静态解析（类型未知） | 架构 §4 |

## 3. 字符串与字符

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| 字符 / 字符串字面量 | 报错 `unsupported constant` | `triage::operand_reason`（仅放行 `Number`/`IntegerLiteral`/`Bool`） |
| 字符串拼接 / 操作 | 内建未收录 | `builtins::lookup` |

## 4. 运算符

> 二元/一元/关系/逻辑运算符已**全部覆盖**（`OperatorKind` 全量），含 `^`/`.^`（见
> `docs/architecture.md` §10.2）。下表仅列仍延后的边界情形。

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `scalar ^ matrix`（`expm`）/ `matrix ^ matrix` | 不可静态降级 → defer | `triage::binary_ty` |
| `matrix ^ 非整数/负指数`（需 `expm`/求逆） | defer | `mir_to_mlir::lower_matrix_power` |
| `matrix ^ 运行时指数` | defer | `mir_to_mlir::lower_matrix_power` |
| 矩阵除法 `mrdivide` / `mldivide`（矩阵 `/` `\`） | 标量 `/` `\` 已支持；矩阵级未实现 | `mir_to_mlir::apply_binary`（按标量处理）、架构 §10.2 |
| 数组 × 数组广播（非同形逐元素） | 仅同形数组或标量广播 | `mir_to_mlir::lower_array_binary`、架构 §10.2 |
| N-D 转置（>2 维） | 报错 `only 2-D transpose is supported` | `mir_to_mlir::lower_array_unary` |
| 复数运算 | 元素类型仅 `f64` | 架构 §10.1 |

## 5. 数值类型

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `single` / `int*` / `uint*` | 元素类型仅 `f64`（`LocalTy::Scalar` 即 `f64`） | 架构 §10.1 |
| 逻辑数组（logical array） | 仅标量 `bool`（映射 C++ `bool`） | 架构 §10.1 |

## 6. 数组与形状变换

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `permute` / `repmat` / `cat` / `horzcat` / `vertcat` | 内建未收录 | `builtins::lookup`、架构 §10.5（P4） |
| N-D 数组（rank > 2 的转置/广播等） | 大多未覆盖 | `MAX_RANK`/`lower_array_unary` |
| 动态形状数组 | 数组形参/输出已支持（见下行）；任意动态形状中间值报错 `unresolved shape` / `dynamic array expression` | `triage::classify`、`hir_to_mlir::array_source`、§10.5（P7） |
| 数组形参 | 已支持「指针 + 长度」ABI（`double* v1, double v2`）：归约（`sum/prod/min/max`）、`numel/length`、运行时下标 `A(i)`、`end`、运行时区间 `for` 循环；数组/标量二义用法（`A+B`）仍当标量；无越界检查 | `triage::infer_array_params`、`hir_to_mlir::lower_function`、§10.5 P7 |
| 动态数组输出 | 已支持「缓冲 + 长度回填」ABI：`y = A(:)`、`y = -A(:)`、`y = k * A`（标量广播）、`y = A(:) ± B(:)`、`y = A(:) .* B(:)`（等长双数组）；其他输出表达式未支持 | `hir_to_mlir::lower_array_unary`/`lower_array_binary`、§5、§10.5 P7 |
| 动态数组中间值 | 报错 `unresolved shape`（需体内运行时分配，未实现） | `triage::classify`、§10.5 P7 |
| 数组增长 / 追加（`x(end+1) = ...`） | 需运行时堆分配，未实现 | §11.3 |

## 7. 内建函数（未收录，defer 到运行时）

`builtins::lookup` 只收录纯数值逐元素/归约/形状内省/构造器子集，其余全部
`unsupported builtin`（defer）。代表性未支持项：

- **排序 / 查找**：`find`、`unique`、`ismember`（`sort` 已支持向量升序，见 `docs/architecture.md` §10.3；矩阵列排序未做）
- **统计**：`mean`、`var`、`std`、`median`、`cumsum`、`cumprod`、`diff`、`all`、`any`
- **线性代数**：`dot`、`cross`、`norm`、`det`、`inv`、`eig`、`svd`、`chol`、`lu`、`qr`、`pinv`
- **信号 / 插值**：`fft`、`ifft`、`conv`、`filter`、`polyval`、`polyfit`、`interp1`、`interp2`
- **随机**：`rand`、`randn`、`randi`、`randperm`
- **构造 / 网格**：`linspace`、`logspace`、`diag`、`tril`、`triu`、`fliplr`、`flipud`、`rot90`、`meshgrid`、`ndgrid`、`magic`
- **内省**：`isscalar`、`isvector`、`ismatrix`、`isempty`、`isnan`、`isinf`、`isfinite`、`class`
- **I/O / 副作用**：`disp`、`fprintf`、`sprintf`、`error`、`warning`、`input`、`load`、`save`

## 8. 索引

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| 切片 `A(i,:)` / `A(:,j)` | 报错 `colon slices are not supported yet` | `mir_to_mlir::static_linear_offset` |
| 冒号区间表达式 `v = 1:10` | 报错 `unsupported rvalue`（仅 `for` 循环的 range 可迭代已支持） | `triage::rvalue_reason` / `terminator_reason` |
| 变量下标 | 报错 `variable indices are not supported yet`（仅常量下标） | `mir_to_mlir::constant_index` |
| 逻辑索引 `A(A>0)` | 未支持 | `mir_to_mlir::static_linear_offset` |
| cell 索引 `{}` | 见 §2 | `mir_to_mlir::lower_index_scalar` |

## 9. 控制流

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| 函数中途 `return`（在 `if`/循环内） | 报错 `` `return` inside control flow is not supported yet `` | `mir_to_mlir::lower_region` |
| 非结构化 `goto` | 报错 `unstructured goto to block ...` | `mir_to_mlir::lower_region` |
| `try` / `catch` | 报错 `unsupported statement` | `triage::stmt_reason` |
| `for i = A`（数组迭代） | 报错 `unsupported for-loop iterable`（仅 `start:step:end` range） | `triage::terminator_reason` |
| `parfor` / `spmd` | 并行构造，未支持 | 架构非目标 |

## 10. 内存与运行时兜底

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| 运行时兜底（`defer_to_runtime`） | 仅 seam，被 deferred 的函数直接报错 | `src/runtime/mod.rs` |
| 动态形状描述符 ABI（数据指针 + 形状描述） | 未实现 | §11.2 |
| 运行时堆分配 / 越界 / 重分配 | 未实现 | §11.3 |

## 11. 后端与优化

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| LLVM / GPU 后端 | `NotImplemented`（仅 `C` 后端可用） | `src/pipeline.rs` |
| CSE / canonicalize / linalg 融合 / 向量化 | 整体延后，不做 | §10.4、§10.5（P6） |

---

> **补集说明**：已支持的子集见 `README.md`「Status」与 `docs/architecture.md`
> §10.5 路线状态（P1–P5 已做项）。本文档随白名单 / 内建表 / 降级器扩展持续更新，
> 新增可生成 pattern 时请同步从对应小节移除。
