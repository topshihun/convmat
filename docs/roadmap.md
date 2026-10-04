# 测试驱动的实现路线（未通过用例 → 根因 → 方案）

> 本文是「让 `examples/coder/` 语料库**全部通过**」的实现方案与推进顺序。它把当前仍
> 失败的用例归到若干**工作流（WS）**，每个工作流给出根因（所在层次 + 代码路径）、
> 落地步骤、验收标准、难度与依赖，并按批次排出推进顺序。
>
> - 现状：`examples/coder/` **50 个用例，50 个编译**（全部正确，已知误编译 0），**0 个待实现**。
> - 报告：`cargo test --test coder_examples -- --ignored --nocapture`。
> - 负向测试：`tests/errors.rs` 有意拒绝的构造；**落地某项时要同步翻正**（见各 WS 说明）。
> - 每落地一项的**收尾清单**见文末 §4（提升进 `SUPPORTED`、补运行驱动、同步
>   `README.md`/`docs/architecture.md §10.5`/`docs/unsupported.md`）。
>
> 层次缩写：**T**=`src/triage`（类型/形状/边界）、**L**=`src/hir_to_mlir`（降级）、
> **D**=方言（`src/dialects/{matlab,emitc}.rs`）、**E**=`src/emit_c.rs`、**R**=`src/runtime`。

---

## 0. 当前失败的用例

无：`examples/coder/` 50 个用例全部编译且运行正确。

## 1. 工作流总览（按建议顺序）

| # | 工作流 | 解锁用例 | 依赖 | 难度 | 主要层次 |
|---|--------|----------|------|------|----------|
| WS3 | 嵌套 struct（静态展平） | `struct_nested` | ✅ 已完成 | 小-中 | T,L |
| WS2 | char 字面量与文本内建 | `text_compare`、`text_switch` | ✅ 已完成 | 中 | T,D,E,L |
| WS4a | 整数元素类型 | `type_integer` | ✅ 已完成 | 中-大 | T,D,L,E |
| WS1 | 逻辑索引 / 掩码写 | `array_logical_index`、`array_mask_assign` | ✅ 已完成 | 中-大 | T,L,R,E |
| WS6 | try / catch | `sys_trycatch` | ✅ 已完成 | 中-大 | D,L,T,R |
| WS9 | cell 数组（运行时 box 接线） | `cell_basic` | ✅ 已完成 | 大 | R,D,L,T |
| WS4b | 复数 + fft | `type_complex`、`sys_fft` | ✅ 已完成 | 大 | T,D,L,E,R |
| WS5 | 逃逸 / 内建函数句柄 | `func_handle_return`、`func_handle_builtin` | ✅ 已完成 | 大 | D,L,T(,R) |
| WS4c | eig | `linalg_eig` | ✅ 已完成 | 大 | R,L |
| WS7 | 图形 / 文件 I/O | `sys_plot`、`sierpinski`、`sys_file_io` | ✅ 已完成 | 小-中 | R,L |
| WS8 | dijkstra（收口） | `dijkstra` | ✅ 已完成 | 大 | T,L,R |

> 依赖关系决定顺序：WS3/WS2/WS4a 互不依赖、可并行；WS1 是动态数组基建，WS4b 依赖
> WS4a 的类型系统；WS4c 依赖 WS4b；WS8 依赖 WS1；WS9（运行时 box 接线）与 WS5（闭包
> 环境）可共享同一 box 基建。

## 2. 各工作流方案

### WS1 · 逻辑索引与掩码写 ✅ 已完成

> 已落地：`triage::mask_index_ty` 识别同形掩码下标 → `Array { shape: Dynamic }`；
> `A(mask)` 经 `convmat_mask_gather`（写输出缓冲）+ `convmat_mask_count`（长度回填）；
> `A(mask)=v` 经 `convmat_mask_assign`；`lower_array_expr_into` 新增 `Binding` 拷贝分支
> （`y = A`）。验收 `run_array_logical_index`/`run_array_mask_assign`，新增
> `logical_index`/`mask_assign` fixture。

**解锁**：`array_logical_index`（`y = A(A>0)`）、`array_mask_assign`（`A(A<0)=0`）。

**根因**

- 读：`triage::expr_ty` → `index_ty(base, indexing)` 对「下标是布尔掩码」的结果形状无法
  解析，返回 `LocalTy::Dynamic`；`WhitelistClassifier::classify` 见到 `Dynamic` 局部即
  `Deferred { unresolved shape for local N }`。
- 写：`triage::stmt_reason` / `hir_to_mlir::lower_stmt` 对非常量 `Index` 赋值目标返回
  `non-local assignment target`。

**方案（通用，走动态数组基建）**

1. **T**：让同形数组比较（`A > 0`）产出逻辑数组类型
   `Array { shape: 与 A 同 }`；在 `index_ty` 中识别「单个下标且下标类型为布尔数组」，
   结果定为 `Array { shape: Dynamic }`。把 `IndexComponent::Logical` 与「`Expr` 但语义为
   掩码」两条路径归一到同一处识别（`expr_reason` 目前对 `Logical` 直接拒绝，需放宽）。
2. **T**：`stmt_reason` 放行 `A(mask) = v`（`HirPlace::Index` 且下标为掩码；`A` 保持原形状，
   右值为标量）。
3. **L**：
   - `y = A(mask)` → 复用动态输出 ABI（`data` + 长度回填），发
     `convmat_mask_gather(A.data, A.n, mask.data, mask.n, out, out_n)`。
   - `A(mask) = v` → 发 `convmat_mask_assign(A.data, A.n, mask.data, mask.n, v)`（原地写）。
   - `mask` 的产生复用现有逐元素比较降级（生成 0/1 数组）。
4. **R**：在 `src/runtime/mod.rs` 的 `helper_source` 新增 `convmat_mask_gather` /
   `convmat_mask_assign`，并在 `docs/runtime.md §9.1` 表登记；`tests/runtime.rs` 的
   「注册表 ↔ 源码」一致性校验会覆盖。
5. **E**：无需新 op，复用动态输出/堆中间值机制。

**快速路径（小，可选，先行）**：当 base 与 mask 均为编译期常量（静态字面量 + 常量比较）时，
在 `hir_to_mlir` 直接常量折叠出静态结果数组。两个用例都能过，但**不通用**；建议直接做
通用 helper，避免「只为用例特化」。

**验收**：`run_array_logical_index` → `[1 3]`（长度 2）；`run_array_mask_assign` → `[1 0 3 0]`。
**翻正负向**：`tests/errors.rs::logical_index_rejected`（fixture `logical_index.m`）。
**难度** 中-大；层次 T,L,R,E。

### WS2 · char 字面量与文本内建 ✅ 已完成

**解锁**：`text_compare`（`strcmp('abc','abc')`）、`text_switch`（`switch c; case 'a'`）。

**根因**：`triage::expr_reason` 拒绝 `HirExprKind::String`/char 字面量（报
`unsupported expression String(...)`）；`builtins::lookup` 未收录 `strcmp`；值模型无 char
元素类型。`text_switch` 的 `c` 是标量形参，缺的只是 char 字面量/码点。

**方案**

1. **T**：`expr_reason` 放行 `String`；`expr_ty` 把 char 字面量 `'abc'` 定为
   `Array { shape: Static 1×n }`，元素类型为 char（引入 WS4 类型系统里的 `DType`，本 WS 先
   落 char 一种）。单字符 `'a'` 在标量上下文（`switch` case）按 1×1 char → **标量码点**处理。
2. **D/E**：char 元素在 C 里用码点数值（`char`/`uint16_t`）表示；数组字面量发射为码点数组。
3. **L**：新增 `Builtin::Strcmp`（等长逐元素码点相等 → 标量 bool）；`switch` 的 `case 'a'`
   走已有标量比较路径。
4. **R**：可纯内联，无需 helper（或可选 `convmat_strcmp`）。

**最小替代（小，不通用）**：仅对「两个字符串字面量」的 `strcmp` 常量折叠为 1/0，并把 char
字面量当标量码点。可过 `text_compare` 与 `text_switch`，但不支持变量 char 数组。

**验收**：`run_text_compare` → 1；`run_text_switch('a')` → 1、`run_text_switch('z')` → 0。
**难度** 中（通用）/ 小（最小替代）；层次 T,D,E,L。
**边界**：完整字符串语义（拼接、`sprintf`、cellstr、`strncmp` 等）仍延后。

### WS3 · 嵌套 struct（静态展平）✅ 已完成

**解锁**：`struct_nested`（`s.a.b = 1; y = s.a.b`）。

**根因**：字段类型推断 `triage::infer_struct_params` / `record_struct_field` 只认「**参数**
的一级标量字段」（`LocalTy::Struct { fields }` 的元素固定 `Scalar`）；`s.a.b` 的访问链解析为
`LocalTy::Dynamic`。此外 `s` 是**局部变量**，而现有 struct 推断只覆盖参数，需要局部 struct
的 use-site 推断。

**方案（静态展平，推荐）**

1. **T**：把「一级字段」推广为「字段路径」——沿 `Member` 链收集 `s.a.b` 为**点号键**
   `"a.b"`；`record_struct_field` / `collect_*_fields` 不再限定参数，对局部变量同样收集
   字段路径（扩展 `infer_locals`）。
2. **L**：`hir_to_mlir` 的 `Member` 读/写与 struct 布局按展平点号字段处理（`s.a.b` → 一个
   扁平字段，`s_a_b`）。
3. **E**：沿用现有 struct 值语义 ABI（字段仍为标量），无需新 op。

**替代**：真正的嵌套 C struct（ABI/返回布局改动更大），延后；struct 数组、字段为数组仍不在本 WS。

**验收**：`run_struct_nested` → 1。
**难度** 小-中；层次 T,L。

### WS4 · 数值类型系统（整数 / 复数 / eig）

**解锁**：`type_integer`、`type_complex`、`sys_fft`、`linalg_eig`。

**根因**：`LocalTy::Scalar` 即 `f64`，无元素 dtype；`int32`、复数（`i`/`3+4i`）、`abs`
复数重载、`fft`/`eig` 均未收录。

**方案（分三步，可独立落地）**

- **WS4a 整数** ✅ 已完成：新增标量 `LocalTy::Int32`（以 `f64` 存储，运算经 `convmat_int32`/`convmat_iadd`/`convmat_isub`/`convmat_imul`/`convmat_idiv` 回绕/取整到 32 位）；`int32(x)` 转为 `Builtin::Int32`；整数与整数运算保持整数（比较/逻辑得逻辑标量），整数与 double 混用仍 defer。验收 `run_type_integer` → 7，新增 `int32_arith`/`int32_wrap`/`int32_div` fixture。难度 **中-大**；层次 T,D,L,E。
  > 最小替代（不推荐）：把 `int32(x)` 当恒等（按 `f64` 计算）→ 示例可过，但丢失
  > 溢出/取整语义，属误编译，不应采纳。
- **WS4b 复数 + fft** ✅ 已完成：新增 `LocalTy::Complex`（盒式 `convmat_value*`，dtype
  `CONVMAT_COMPLEX`，交错 `[re,im]`）；`i`/`3+4i` 与复数算术全部走运行时 helper
  （`convmat_complex`/`cadd`/`csub`/`cmul`/`cdiv`/`cabs`/`fft`）；`abs` 复数 → `convmat_cabs`；
  `fft` → `convmat_fft`；复数输出以 `convmat_value*` 返回。验收 `run_complex_abs`/`run_complex_fft`。
  难度 **大**；层次 T,D,L,R。
- **WS4c eig** ✅ 已完成：`convmat_eig`（1x1/2x2 解析解，2x2 含复特征值）返回复数盒；
  `eig(A)` 对静态方阵（n≤2）类型为 Complex，n>2 defer。验收 `run_complex_eig` → `[2, 3]`。
  难度 **大**；层次 R,L。

### WS5 · 逃逸 / 内建函数句柄（闭包值）✅ 已完成

> 已落地：函数句柄降级为盒式 `convmat_value`（`CONVMAT_FUNCTION`）：`hir_to_mlir` 为
> 每个逃逸句柄点生成一个 thunk（`convmat_handle_thunk_<id>`）与模块级
> `convmat_handle_dispatch`（select 链，无需新控制流 op），捕获标量快照进 `u.func.env`；
> 运行时 `convmat_function_handle`/`handle_set_env`/`handle_env`/`handle_call`（
> `handle_support_source()`）；`@sin` 直接调用亦支持（`g = @sin; y = g(2)`）。
> 验收 `run_handle_return`/`run_handle_builtin`，coder 例 `func_handle_return`/
> `func_handle_builtin`。边界：一元句柄、标量捕获/返回；多参数、数组捕获、跨函数传递仍 defer。

**解锁**：`func_handle_return`（`f = @(x) x + k` 作为返回值）、
`func_handle_builtin`（`f = @sin`）。

**根因**：`triage::analyze_handles` 拒绝逃逸句柄（`escapes as a return value`）；`@sin` 在
`triage::expr_reason` 报 `unsupported expression FunctionHandle(...)`。两例都是「**返回**句柄」，
无法用「非逃逸静态特化」覆盖。

**方案（动态闭包 tier）**

1. **D/T**：新增**句柄值**类型 = `{ 函数指针, 捕获环境 }`（或 `convmat_value` 的
   `CONVMAT_FUNCTION` kind + 注册表 `handle_id`，见 `docs/runtime.md §2`）。
2. **L**：`@sin` → 注册一个包装 `sin` 的 thunk 并返回其句柄；`@(x) x+k` → 生成特化函数 +
   捕获环境（`k` 存入环境），返回句柄即函数返回值。
3. **调用点**：经函数指针 + 环境分派，而非编译期直接 `convmat_anon_N(...)`。
4. **R**（可选）：若环境用堆盒则接 `convmat_value`；若仅为「固定捕获集」可用 C 结构体，
   避免引入动态 tier。

**验收**：两例可编译；驱动可调用返回句柄（`@sin` → `f(0) == 0`；`@(x) x+k` → `f(1) == 1+k`）。
**翻正负向**：`anonymous_function_escape_rejected` 等（按实现范围）。
**难度** 大；层次 D,L,T（+R）。

### WS6 · try / catch ✅ 已完成

> 已落地：`matlab.try` op（try/catch 两个 region）由 `hir_to_mlir` 生成；`lowering` 把它
> 翻译为 emitc 的**已有 C 控制流**（`convmat_error_enter` → `convmat_error_check`（宏，
> 内联 `setjmp`）→ `emitc.if`（then=try / else=catch）→ `convmat_error_leave`），**不引入
> `emitc.try`**；运行时 `convmat_error_throw` 用 `longjmp` 回到最近的 try。验收
> `run_sys_trycatch`/`run_try_catch`，并在 `tests/runtime.rs` 验证真实抛/捕获路径。

**解锁**：`sys_trycatch`。

**根因**：`triage::stmt_reason` 对 `HirStmtKind::TryCatch` 返回 `try/catch is not supported`。

**方案**（按 `docs/runtime.md §7`）

1. **R**：实现 `convmat_error_enter` / `leave` / `throw` + 线程本地 `convmat_error`
   （`setjmp`/`longjmp`），补 `#include <setjmp.h>`；更新 `docs/runtime.md §9.2`（当前仅声明）。
2. **D/L**：新增 `matlab.try` op（try/catch 两个 region），降级为
   `if (setjmp(...) == 0) { try } else { catch }`。
3. **T**：放行 `TryCatch`。
4. **抛错源**：动态调用/内建出错 → `convmat_error_throw`。注意本例 `y = 1/x` 在 C 中不抛
   （除零得 Inf），`catch` 不可达；该例只验证「可编译 + try 体语义」。

**验收**：`run_sys_trycatch(2)` → 0.5。
**翻正负向**：`try_catch_rejected`（fixture `try_catch.m`）。
**难度** 中-大；层次 D,L,T,R。

### WS9 · cell 数组（运行时 box 接线）✅ 已完成

> 已落地：新增 `matlab.box` 类型（C 层 `convmat_value*`）与 `matlab.cell_new`/
> `cell_set`/`cell_get` op，`lowering` 映射到 `emitc.call_box`/`call_void`/`call`；接线
> `convmat_value` box 内核（`DYNAMIC_RUNTIME_H`+`C`，按需发射）+ 标量包装 `convmat_cell_new`/
> `set_scalar`/`get_scalar`；块作用域释放（`convmat_value_release`）。**当前仅支持标量元素的
> 字面量 cell**；非标量元素/`cell(...)`/cell 形参或返回值仍 defer。

**解锁**：`cell_basic`（`c = {1, 2, 3}; y = c{1}`）。

**根因**：`triage::expr_reason` 对 `HirExprKind::Cell` 直接拒绝（`cell array literals are not
supported yet`）；`hir_to_mlir::lower_index_scalar` 拒绝花括号索引（`{}`）；值模型无 cell。

**方案**（沿用 `docs/runtime.md` 已设计的 `convmat_value` box）

1. **R**：把 `docs/runtime.md §9.2` 已实现但**未接线**的 box 内核通过 seam（§8 step 3）
   接入生成代码：``defer_to_runtime`` 从「报错」改为「桥接 `convmat_value` + 动态 ABI 调用」。
2. **D**：新增 cell 相关 op（构造 / 读 `c{i}` / 写 `c{i}=`），降级为运行时调用
   `convmat_cell_create`/`get`/`set`。
3. **L/T**：`HirExprKind::Cell` 字面量 → cell 构造；`HirPlace`/`HirExpr` 的 `{}` 索引 → cell
   访问；元素的静态类型在 box 内动态承载。
4. **E**：动态值以 `convmat_value*` 传参/返回（需与 §5 ABI 对接）。

**边界**：cell 元素的异构类型、嵌套 cell、cell 作为 struct 字段等随本 WS 一并归到 box 语义。
**验收**：`run_cell_basic` → 1。
**翻正负向**：`cell_literal_rejected`（fixture `cell_literal.m`）。
**难度** 大；层次 R,D,L,T。

### WS7 · 图形 / 文件 I/O ✅ 已完成

**解锁**：`sys_plot`、`sierpinski`（`plot`）、`sys_file_io`（`fopen`/`fread`/`fclose`）。

> 已落地（路线 (a)）：`plot`/`plot3` 降为**文档化的 no-op**（实参先求值，再调用
> `convmat_plot`；作为值使用会 defer）；`fopen(name, mode)`/`fread(fid)`/`fclose(fid)`
> 降为基于 C stdio 的运行时 helper（`FILE*` 表 + 标量 id；`name` 为 char 数组，`mode`
> 为单字符码点；`fread` 读二进制 `double` 返回动态形状数组）。验收 `run_sys_plot`/
> `run_sierpinski`/`run_sys_file_io`（后者在临时目录建 `data.bin` 后读取）。语义边界见
> `docs/unsupported.md` §7.1/§7.2。

**根因**：这些内建未收录；`docs/architecture.md §4` 将它们视为「边界外」。

### WS8 · dijkstra（动态数组收口）✅ 已完成

**解锁**：`dijkstra`。

> 已落地：① `Inf`/`NaN` 作为 `Fill` 构造（`Inf(1,n)`），非恒定维度 → `Array{Dynamic}`
> （`fill_ty`），降为 `convmat_fill`；② 动态数组的**运行时标量下标写** `dist(source)=0`
> （`lower_stmt` 的 Index 分支 → `matlab.store`）；③ **运行时列切片** `W(:, j)`
> （`index_ty` 动态 slice → `Array{Dynamic}`；`convmat_column`，形状来自描述符）；
> ④ 两数组**逐元素** `min`/`max`（`call_ty` + `convmat_ewmin`/`ewmax`）；⑤ `dynamic_array_len`
> 支持 Fill/列切片/元素逐元素 min 与动态输出的 out-length。验收 `run_dijkstra`（3×3 W、
> source=1 → `dist=[0 3 5]`）。语料库 49→50。

**根因**：首报 `unsupported builtin 'Inf'`，但实际是多项动态数组能力的**合成**：

- `Inf(1, n)`：常量填充的运行时形状构造（当前 `Fill` 用 `f64` 常量、dims 来自标量实参）。
- `dist(source) = 0`：动态数组的「**运行时标量下标写**」。
- `W(:, source)`：动态数组的**列切片**。
- `min(dist, ...)`：两参 `min` 对数组**逐元素**（结果为数组，当前两参 `min` 仅标量）。
- `dist(source)` 读、数组 + 标量广播（动态）。

**方案**：在 WS1 的动态输出/中间值基建之上，逐项补齐：运行时运行下标**写**、运行时列切片、
逐元素两参 `min`/`max`、`Inf`/任意常量填充的动态维度。作为动态数组 tier 的收口用例。

**验收**：需为 `dijkstra` 复核语义并设计确定输出的驱动（当前示例较简化）。
**难度** 大（合成）；层次 T,L,R。

## 3. 批次计划

按「小步快赢 → 动态基建 → 大件」推进，每批完成后语料库计数上升且 CI 全绿。

| 批次 | 工作流 | 新增通过 | 说明 |
|------|--------|----------|------|
| 1 ✅ | WS3、WS2、WS4a | +4 | `struct_nested` + 文本 + 整数（已完成） |
| 2 ✅ | WS1 | +2 | 逻辑索引/掩码写（已完成） |
| 3 ✅ | WS6 | +1 | try/catch（已完成） |
| 4 ✅ | WS4b | +2 | 复数主线：`type_complex` + `sys_fft`（已完成） |
| 5 ✅ | WS9 | +1 | 运行时 box 接线：`cell_basic`（已完成） |
| 6 ✅ | WS5 | +2 | 闭包值：逃逸/内建句柄（已完成） |
| 7 ✅ | WS4c | +1 | `eig`（已完成） |
| 8 ✅ | WS7 | +3 | 图形 no-op + stdio 文件 I/O（已完成） |
| 9 ✅ | WS8 | +1 | `dijkstra`（动态数组收口）（已完成） |

> 合计 17。批次内若某 WS 可独立落地，可并行推进（写入范围互不重叠）。

## 4. 每项完成的收尾清单

1. 把用例提升进 `tests/coder_examples.rs` 的 `SUPPORTED`，并补一个 `run_*` 运行驱动
   （`coder_examples_supported_run`）。
2. 翻正/删除对应的 `tests/errors.rs` 负向测试与 `tests/fixtures/` 中的拒绝用 `fixture`。
3. 补「最小 `.m` → IR → C」的单元/集成测试（`tests/emit_c.rs` 或 `#[cfg(test)]`）。
4. 更新 `README.md`「Status」与 `docs/architecture.md §10.5` 路线状态。
5. 从 `docs/unsupported.md` 对应小节移除该项；若新增运行时 helper，登记到
   `docs/runtime.md §9.1/§9.2` 并让 `tests/runtime.rs` 的一致性校验通过。
6. 提交前依次跑通：`cargo fmt --all`、`cargo clippy --all-targets -- -D warnings`、
   `cargo test --all-targets`、`cargo check`。

---

## 5. 已完成批次（历史）

> 保留推进记录，便于回看已落地的能力面。

- **第 1 批（A 静态项 + B1）**：块拼接 `[a b; c d]`、切片 `A(i,:)`/`A(:,j)`、步长区间
  `A(a:b:c)`、隐式扩展 `A + b`、N-D 数组、`Inf`/`NaN`、`isempty`、`logical`、`var`、
  `linspace`/`repmat`/`permute`。语料库 13→25。
- **第 2 批（C 用户函数调用/递归，标量子集）**：同文件封闭世界调用（全标量入参 + 单标量输出，
  含递归）。语料库 25→28。
- **第 3 批（B2 静态线代/随机运行时 helper）**：`inv`/`det`/`norm`（向量 2-范数）、方阵左除
  `A \ B`、`rand()`。语料库 28→33。
- **第 4 批（F 持久化 `isempty`）**：`persistent` 变量新增 `_not_empty` 静态标志，修复
  `kalmanfilter`（该用例本已能编译，计数不变）；另完成 `sort(A)` 矩阵按列排序。
- **第 5 批（F 形状描述符）**：动态数组形参查询 `size` 走「数据 + 行 + 列」形状描述符 ABI，
  `size(A)`/`size(A,d)`、`length=max(rows,cols)` 可用。
- **第 6 批（动态数组中间值 + 输出）**：动态数组输出「缓冲 + 长度回填」ABI
  （`y = A(:)`、`y = -A(:)`、`y = k * A`、`y = A(:) ± B(:)`、`y = A(:) .* B(:)`）；中间值块作用域
  堆分配/释放（含控制流内）。
- **第 7 批（batch 1：WS3 + WS2 + WS4a）**：嵌套 struct 字段路径展平（`s.a.b` → 扁平 C 字段
  `a__b`，局部 struct 按字段写入推断类型）；char 字面量（单字符=码点标量、多字符=1×N 数组）
  与 `strcmp`（字面量常量折叠）；标量 `int32` 类型（`int32(x)` + 回绕运算 helper）。
  语料库 33→37，新增 `run_nested_struct`/`run_int32_arith`/`run_int32_wrap`/`run_char_switch`/
  `run_strcmp_lit`。
- **第 8 批（batch 2：WS1）**：逻辑索引 `A(A>0)`（同形掩码 → 动态输出 +
  `convmat_mask_gather`/`convmat_mask_count`）与掩码写 `A(A<0)=0`（`convmat_mask_assign`）；
  另补齐 `y = A` 静态数组拷贝与 `int32` 除法（`convmat_idiv`）。语料库 37→39。
- **第 9 批（batch 3：WS6）**：`matlab.try` op（try/catch 两 region）→ `lowering` 翻译为
  emitc 的**已有 C 控制流**（`setjmp` 守卫 + `if`/`else`，无新 `emitc.try`）；运行时
  `convmat_error_enter`/`check`（宏，内联 `setjmp`）/`leave`/`throw`（`longjmp`）。语料库 39→40。
- **第 10 批（batch 5：WS9）**：接线 `convmat_value` box 内核。新增 `matlab.box` 类型与
  `cell_new`/`cell_set`/`cell_get` op（`lowering` → `emitc.call_box`/`call_void`/`call`）；
  `triage::LocalTy::Cell`；块作用域释放。语料库 40→41。
- **第 11 批（batch 4：WS4b）**：复数盒式值（`LocalTy::Complex`，`CONVMAT_COMPLEX`）+ 运行时
  复数算术/`cabs`/`fft`（`complex_support_source`）；`matlab.box_call` op；复数输出以
  `convmat_value*` 返回。语料库 41→43。
- **第 12 批（batch 7：WS4c）**：`convmat_eig`（1x1/2x2 解析解）复用复数盒。语料库 43→44。
- **第 13 批（batch 6：WS5）**：函数句柄闭包值（`CONVMAT_FUNCTION` 盒）：逃逸匿名句柄
  （返回捕获快照）、内建句柄（`@sin` 直接调用/返回）；`hir_to_mlir` 生成 thunk +
  `convmat_handle_dispatch`，运行时 `convmat_handle_call`。语料库 44→46。
- **第 14 批（batch 8：WS7）**：图形 `plot`/`plot3` 降为文档化 no-op（`convmat_plot`，
  实参先求值）；`fopen`/`fread`/`fclose` 降为 stdio 运行时 helper（`FILE*` 表 + 标量 id，
  `fread` 读二进制 `double` 返回动态形状数组）。语料库 46→49。
- **第 15 批（batch 9：WS8）**：动态数组收口——`Inf`/`NaN` 动态填充（`convmat_fill`）、
  运行时标量下标写 `dist(source)=0`、运行时列切片 `W(:,j)`（`convmat_column`）、
  两数组逐元素 `min`/`max`（`convmat_ewmin`/`ewmax`）；`dijkstra` 通过。语料库 49→50。
