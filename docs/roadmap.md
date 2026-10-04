# 测试驱动的实现路线（未通过用例 → 问题 → 思路）

> 本文把当前仍**失败**的测试用例（`examples/coder/` 语料库 + `tests/errors.rs` 负向测试）
> 归类为若干工作流，给出根因（所在层次）与实现思路，作为「按测试逐渐实现功能」的路线图。
>
> - 现状：`examples/coder/` **50 个用例，33 个编译**（33 正确，已知误编译已清零）。
> - **第 1 批（A 静态项 + B1）已完成**：块拼接 `[a b; c d]`、切片 `A(i,:)`/`A(:,j)`、
>   步长区间 `A(a:b:c)`、隐式扩展 `A + b`、N-D 数组、`Inf`/`NaN`、`isempty`、`logical`、
>   `var`、`linspace`/`repmat`/`permute`。语料库 13→25。
> - **第 2 批（C 用户函数调用/递归，标量子集）已完成**：同文件封闭世界调用（全标量入参 +
>   单标量输出，含递归）；语料库 25→28。
> - **第 3 批（B2 静态线代/随机运行时 helper）已完成**：`inv`/`det`/`norm`（向量 2-范数）、
>   方阵左除 `A \ B`、`rand()`；语料库 28→33。
> - **第 4 批（F 持久化 `isempty`）已完成**：`persistent` 变量新增 `_not_empty` 静态标志，
>   `isempty(p)` 首次为真，修复 `kalmanfilter`（已知误编译清零；该用例本已能编译，故计数不变）；
>   另完成 `sort(A)` 矩阵按列排序（`convmat_sort_cols`）。
> - **第 5 批（F 形状描述符）已完成**：动态数组形参在查询 `size` 时走「数据 + 行 + 列」
>   形状描述符 ABI（按使用点驱动，既有「数据 + 长度」形参签名不变），`size(A)`/`size(A,d)`、
>   `length=max(rows,cols)` 可用；负向 `array_size` 转正（语料库计数不变，为负向测试）。
> - `tests/errors.rs` **24 个负向测试**：有意拒绝，其中一部分是永久边界，一部分是待实现。
> - 重新生成报告：`cargo test --test coder_examples -- --ignored --nocapture`。
> - 已支持子集见 `README.md`「Status」、`docs/architecture.md` §10.5；边界检测点见 `docs/unsupported.md`。

层次缩写：**T**=`src/triage`（类型/形状/边界）、**L**=`src/hir_to_mlir`（降级）、
**D**=方言（`src/dialects/{matlab,emitc}.rs`）、**E**=`src/emit_c.rs`、**R**=`src/runtime`。

---

## A. 静态数组形状与索引（**不需要运行时 tier**，形状编译期已知）

| 用例 | 现状原因 | 层次 | 思路 | 难度 |
|------|----------|------|------|------|
| `array_row_slice` / `array_col_slice` | `index_ty` 对 `A(i,:)`/`A(:,j)` 返回 `Dynamic` | T,L | `index_ty` 计算切片结果形状；`L` 复制子块到 `dest` | 小 |
| `array_stride` (`A(1:2:5)`) | 非常量区间下标未降级 | T,L | 常量 range → 结果长度；逐元素拷贝 | 小-中 |
| `array_broadcast` (`A + b`) | `binary_ty` 对不同形状返回 `Dynamic` | T,L | 允许 singleton 维隐式扩展；结果形状取各维 max；`L` 用模索引生成 | 中 |
| `array_nd` (`zeros(2,2,2)`) | `Fill` 仅接受 ≤2 维（arity） | T,L,D | 构造器/`Shape`/`linear`/下标推广到 N 维（`MAX_RANK=4` 已在） | 中 |
| `array_concat`（含 `[a,b]`/`[a;b]`） | `Tensor` 把数组元素当标量 → **静默误编译** | T,L | 检测数组元素 → 计算拼接形状并整块拷贝；**先修此 bug（或显式拒绝）** | 中 |
| `array_logical_index` (`A(A>0)`) | 逻辑下标 → `Dynamic` | T,L,R | logical mask + 运行时长度 → 动态输出 ABI + `convmat_mask` | 中-大 |
| `array_mask_assign` (`A(A<0)=0`) | 非局部赋值目标被拒 | T,L,R | 掩码写：`convmat_mask_assign`；需 mask 类型 | 中-大 |

> 这一批多数是「静态」特性缺失，**不依赖动态 tier**，投入产出比最高。

## B. 内建函数（`builtins::lookup` 未收录 → `unsupported builtin`）

**B1. 纯静态，几行即可**
`value_special`（`Inf`/`NaN` 字面量）、`isempty`、`logical`、`var`、`array_linspace`、
`array_repmat`、`array_permute`。
思路：加 `Builtin` 变体；`Inf`/`NaN` 在 `T` 放行、`L` 发 `INFINITY`/`NAN`；`isempty`=`numel==0`
（静态常量 / 动态 `n==0`）；`var` 复用 `std` 机制；构造器在 `lower_array_expr_into` 内展开。
层次 T,L,E；难度 **小**。

**B2. 静态矩阵 + 运行时 helper**（沿用 `helper_source`，如现有 `convmat_matmul`）—— ✅ 已完成
`inv`/`det`/`norm`（向量 2-范数）、方阵 `A \ B`（`convmat_solve`，高斯消元）、`rand`（`convmat_rand`）
均已落地（`tests/coder_examples.rs` 的 `linalg_*`/`sys_random` 用例）。`sys_fft`（DFT）需**复数**，
留待 D（类型系统）。
层次 R,L；难度 **小-中**。

**B3. 需类型系统 / 大件**
`type_integer`（`int32`/`uint*`）、`linalg_eig`（特征值算法）。
层次 T,D,E,R；难度 **大**。

## C. 用户函数调用 / 递归

用例：`fib`、`func_helper`、`func_recursion`；负向 `multi_assign`（`[a,b]=f()`）。
问题：`call_reason` 拒绝任何非内建调用；无跨函数/多返回值 ABI。
思路（封闭世界）：`T` 放行对同 `HirAssembly` 内函数的调用；`L` 按解析后的签名发
`func.call`（标量按值、静态数组/动态数组走输出指针 ABI）；`lower_to_module` 已遍历全部函数，
被调函数已在模块内。多返回值走 tuple / out-ptr。
层次 T,L,D；难度 **中-大**；杠杆高（解锁真实程序）。

## D. 类型系统（元素 dtype）

用例：`type_integer`、`type_logical`、`type_complex`；`docs/unsupported.md` §5。
问题：值模型只有 `f64`（`LocalTy::Scalar` 即 `f64`），无元素类型。
思路：给 `LocalTy`/`ArrayType` 加 `DType { F64, I32, Bool, Complex }`：
整数用 C `int32_t`（含回绕/饱和语义）、logical 用 `bool`、复数用 `{re,im}`（或 `std::complex`）。
层次 T,D,L,E,R；难度 **大**（贯穿各层）。

## E. 动态异构 tier（cell / 字符串 / 嵌套 struct / 动态字段）

用例：`cell_basic`、`text_compare`、`text_switch`、`struct_nested`、`dijkstra`（部分）；
负向 `cell_literal`。
问题：需要 `convmat_value` 盒（`docs/runtime.md` §2 已设计、§9.2 内核已实现，**未接线**）。
思路：落地 `docs/runtime.md` §8 step 3 的 seam——cell/char/struct 用 `convmat_value*`，
`{...}`/`c{i}`/`s.(name)` 降级为运行时调用；字符串用 `dtype=CHAR`；嵌套 struct 用扁平前缀或盒。
层次 R,D,L,T；难度 **大**（这是「运行时矩阵」的下一步主线）。

## F. 系统 / 副作用与动态形状边界

| 用例/负向 | 问题 | 思路 | 难度 |
|-----------|------|------|------|
| `sys_plot` / `sierpinski` / `sys_file_io`（`plot`/`fopen`） | I/O、图形 | 设计上排除，或映射运行时 stub；低优先 | — |
| `sys_fft` | 未收录（需复数类型，见 D） | 见 B2 | 中-大 |
| `sys_trycatch`、`try_catch` | C 无异常 | `setjmp`/`longjmp` + 线程本地错误（`docs/runtime.md` §7） | 中-大 |
| `averaging_filter` / `kalmanfilter` | `isempty` / 持久化 / 拼接 | ✅ 已完成（`persistent` 的 `_not_empty` 标志） | 小-中 |
| `array_size`（负向） | 动态数组「行 vs 列」二义 | ✅ 已完成（查询 `size` 的形参走「数据 + 行 + 列」形状描述符 ABI，按使用点驱动） | 中 |
| `array_intermediate_loop`（负向） | 控制流内的中间值缓冲作用域越界 | ✅ 已完成（块作用域分配/释放，`lower_block`） | 中 |
| `varargin_var` / `arg_expansion` / `multi_assign` | 开放世界可变参数 / `{:}` 展开 / 多值调用 | 运行时 cell + `varargout` ABI（`docs/architecture.md` §12） | 中-大 |
| `mpower_nonint` / `mpower_nonsquare` | 矩阵幂负/非整数、非方阵 | 需求逆/`expm`/`eig`（依赖 B2/B3） | 中 |
| `matrix_sort`（`sort(A)` 矩阵列排序） | 仅支持向量 | ✅ 已完成（`convmat_sort_cols`） | 小 |
| `anon_*`（逃逸/命名/内建句柄、数组参数/捕获/返回、控制流内定义…） | 动态闭包 / 动态分派 | 非逃逸具名/内建句柄可编译期解析；一般情形需动态闭包 tier | 大 |

## 优先级建议

1. **A 的静态项 + B1**：小改动，语料库可一次性多转绿若干（切片/步长/广播/N-D/拼接 + `Inf`/`isempty`/构造器）。
2. **C（用户调用 / 递归）**：杠杆最大，解锁函数式程序与真实代码。
3. ~~**B2（linalg/rand/fft 运行时 helper）**：自包含、无需类型系统改造。~~ ✅ 已完成
   （`inv`/`det`/`norm`/`\`/`rand`；`fft` 因需复数推迟到 D）。
4. ~~**F 的 `array_size`**~~ ✅ 已完成（按使用点驱动的形状描述符 ABI，负向 `array_size` 转正）；
   ~~**ptr 局部**~~ ✅ 已完成（动态数组中间值改为块作用域分配/释放，负向 `array_intermediate_loop` 转正）。
5. **D / E（类型系统与异构盒）**：长期主线，对应「运行时矩阵」的完整形态。

> 每完成一项：从 `docs/unsupported.md` 对应小节移除、更新 `docs/architecture.md` §10.5，
> 并把 `examples/coder/` 的用例提升进 `tests/coder_examples.rs` 的 `SUPPORTED`（含运行驱动）。
