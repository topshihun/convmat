# 未支持的 MATLAB 特性清单

> 本文档是「可生成代码子集」的补集：凡是超出 `src/triage` 白名单、`src/builtins`
> 表、`src/hir_to_mlir` 可降级范围、以及 `docs/architecture.md` §10.5 路线表
> 已实现项之外的语言构造，都在此列出。每一项标注**当前行为**（报错信息 / defer）与
> **检测位置**（代码路径）。
>
> 权威边界判定见 `docs/architecture.md` §4；可变参数与动态形状的设计见 §11、§12。

## 1. 函数与调用

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `varargin{k}` 变量下标 | defer（无法特化） | `triage::expr_ty` / `hir_to_mlir::lower_index_scalar` |
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
| cell 数组字面量 `{...}` | 标量元素字面量已支持（`convmat_value` box）；非标量元素 defer | `triage::expr_ty`（`HirExprKind::Cell`）、`hir_to_mlir::lower_stmt` |
| 花括号 `{}` 索引（cell 索引） | `c{i}`（常量 `i`）读已支持（标量元素）；变量下标 / `c{i}=` 写 / 非标量元素未支持 | `hir_to_mlir::lower_expr` |
| 动态字段 / struct 动态访问 | 不可静态解析 | 架构 §4 |
| `classdef` 动态分派 | 不可静态解析 | 架构 §4 |
| `global` / `persistent` 变量 | 已支持（编译为 C `static`，仅限可静态定型的封闭世界单函数；`persistent` 变量带 `_not_empty` 标志） | `hir_to_mlir::lower_function`、架构 §10.5 |

## 3. 字符串与字符

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| 字符字面量（简单） | 已支持：单字符=码点标量、多字符=1×N 码点数组；转义序列未解释 | `triage::expr_ty`、`hir_to_mlir::lower_expr` |
| `strcmp`（非字面量操作数） | 仅两个字面量常量折叠；变量/动态 char 数组未支持 | `hir_to_mlir::lower_scalar_call` |
| 字符串拼接 / 操作 | 内建未收录 | `builtins::lookup` |

## 4. 运算符

> 二元/一元/关系/逻辑运算符已**全部覆盖**（`OperatorKind` 全量），含 `^`/`.^`（见
> `docs/architecture.md` §10.2）。下表仅列仍延后的边界情形。

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `scalar ^ matrix`（`expm`）/ `matrix ^ matrix` | 不可静态降级 → defer | `triage::binary_ty` |
| `matrix ^ 非整数/负指数`（需 `expm`/求逆） | defer | `hir_to_mlir::lower_matrix_power` |
| `matrix ^ 运行时指数` | defer | `hir_to_mlir::lower_matrix_power` |
| 矩阵除法 `mrdivide` / `mldivide`（矩阵 `/` `\`） | 标量 `/` `\` 已支持；`\`（方阵 `A \ B`）已支持（高斯消元 helper）；矩阵 `/`（`mrdivide`）及非方阵/最小二乘未实现 | `hir_to_mlir::lower_mldivide`、`triage::mldivide_ty`、架构 §10.2 |
| 数组 × 数组广播（非同形逐元素） | 静态同形 / 单例维隐式扩展已支持；动态数组仅等长逐元素 | `hir_to_mlir::map_binary_broadcast`、`triage::broadcast_shape` |
| N-D 转置（>2 维） | 报错 `only 2-D transpose is supported` | `hir_to_mlir::lower_array_unary` |
| 复数数组运算 | 标量复数 `i`/`3+4i`/算术/`abs`/`fft` 已支持（盒式）；复数数组逐元素/矩阵运算未支持 | 架构 §10.1、§10.5 |

## 5. 数值类型

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `int32` 标量 | 已支持（`LocalTy::Int32`，`f64` 存储 + 32 位回绕运算） | 架构 §10.1、`hir_to_mlir::lower_int_binary` |
| `single` / 其他 `int*` / `uint*` | 元素类型仅 `f64`（标量 `int32` 除外） | 架构 §10.1 |
| 逻辑数组（logical array） | 逻辑值以 0/1 `f64` 表示；`logical`/比较结果已支持 | 架构 §10.1 |

## 6. 数组与形状变换

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `cat` / `horzcat` / `vertcat` | 内建未收录（`permute`/`repmat` 已支持 2-D / 常量） | `builtins::lookup`、架构 §10.5（P4） |
| N-D 数组 | `zeros/ones` 接受 N 维、`A(i,j,k)` 常量下标已支持；N-D 转置/广播未覆盖 | `MAX_RANK`/`lower_array_unary` |
| 动态形状数组 | 数组形参/输出已支持（见下行）；动态数组中间值已在体内（含控制流内）运行时分配（块作用域） | `triage::classify`、`hir_to_mlir::lower_block`、§10.5（P7） |
| 数组形参 | 默认「指针 + 长度」ABI（`double* v1, double v2`）：归约（`sum/prod/min/max`）、`numel/length`、运行时下标 `A(i)`、`end`、运行时区间 `for` 循环；查询 `size` 的形参改为「指针 + 行 + 列」形状描述符 ABI（`double* v1, double v2, double v3`），支持 `size(A)`/`size(A,d)`；无越界检查 | `triage::infer_array_params`/`shape_descriptor_params`、`hir_to_mlir::lower_function`、§10.5 P7 |
| 动态数组输出 | 已支持「缓冲 + 长度回填」ABI：`y = A(:)`、`y = -A(:)`、`y = k * A`（标量广播）、`y = A(:) ± B(:)`、`y = A(:) .* B(:)`（等长双数组）；其他输出表达式未支持 | `hir_to_mlir::lower_array_unary`/`lower_array_binary`、§5、§10.5 P7 |
| 动态数组中间值 | 已支持（含控制流内）：赋值处 `matlab.heap_alloc` 堆缓冲，在所在块末尾 `delete[]`（块作用域）；超出所在块的引用仍报错 | `hir_to_mlir::lower_block`、§10.5 P7 |
| 数组增长 / 追加（`x(end+1) = ...`） | 需运行时堆分配，未实现 | §11.3 |

## 7. 内建函数（未收录，defer 到运行时）

`builtins::lookup` 只收录纯数值逐元素/归约/形状内省/构造器子集，其余全部
`unsupported builtin`（defer）。代表性未支持项：

- **排序 / 查找**：`find`、`unique`、`ismember`（`sort` 已支持向量升序与 2-D 矩阵按列排序，见 `docs/architecture.md` §10.3）
- **统计**：`cumprod`、`all`、`any`（`mean`/`var`/`std`/`median`/`cumsum`/`diff` 已支持）
- **线性代数**：`dot`、`cross`、`norm`（矩阵 2-范数/其他范数）、`det`（非方阵）、`svd`、`chol`、`lu`、`qr`、`pinv`（方阵 `inv`/`det`、向量 `norm`、方阵 `A \ B`、1x1/2x2 `eig` 已支持）
- **信号 / 插值**：`ifft`、`conv`、`filter`、`polyval`、`polyfit`、`interp1`、`interp2`（`fft` 标量复数已支持）
- **随机**：`randn`、`randi`、`randperm`（`rand()` 标量已支持）
- **构造 / 网格**：`linspace`、`logspace`、`diag`、`tril`、`triu`、`fliplr`、`flipud`、`rot90`、`meshgrid`、`ndgrid`、`magic`
- **内省**：`isscalar`、`isvector`、`ismatrix`、`isfinite`、`class`（`isempty`/`isnan`/`isinf` 已支持）
- **I/O / 副作用**：`disp`、`fprintf`、`sprintf`、`error`、`warning`、`input`、`load`、`save`

## 8. 索引

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| 切片 `A(i,:)` / `A(:,j)` | 静态 2-D 已支持（复制选定元素） | `triage::static_index_selection`、`hir_to_mlir` Index 分支 |
| 冒号 / 步长区间 `A(a:b:c)`（含 `end` 边界） | 静态、常量边界已支持 | `triage::component_selection`、`hir_to_mlir` Index 分支 |
| 变量下标（非静态） | 仍不支持（静态数组） | `hir_to_mlir::constant_index` |
| 逻辑索引 `A(mask)` / 掩码写 `A(mask)=v` | 已支持（同形掩码；结果为运行时长度向量） | `triage::mask_index_ty`、`hir_to_mlir::mask_component` |
| 非掩码数组下标列表 `A(idxArray)` | 未支持（同形数组下标一律按掩码处理） | `hir_to_mlir::mask_component` |
| cell 索引 `{}` | 见 §2 | `hir_to_mlir::lower_index_scalar` |

## 9. 控制流

| 特性 | 当前行为 | 检测位置 |
|------|----------|----------|
| `for i = A`（数组迭代） | 报错 `unsupported for-loop iterable`（仅 `start:step:end` range） | `hir_to_mlir::lower_for` |
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
