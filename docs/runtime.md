# 运行时库设计：动态 tier 的统一值模型

> 本文档整合「静态边界之外、必须靠运行时库动态调整类型」的动态特性，并把它们翻译成
> 一套可落地的运行时库接口（C 类型 + 函数签名）。它是 `docs/architecture.md` §4
> （边界）、§5（ABI）、§11（值分级）、§12（varargin/varargout）里「运行时兔底」的
> 具体化。
>
> 现状：`src/runtime/mod.rs` 只有「封装运算符 helper」（`convmat_transpose/matmul/
> mpower`）+ `defer_to_runtime` 报错 seam，尚无动态 tier。本文档是动态 tier 的
> 设计基线，落地依赖 §10.5 的 P7（静态分析）先把「哪些值动态」判出来。

## 1. 哪些动态特性需要运行时库（整合）

按「为什么静态编译解决不了」分成五类。每一类的共同点是：**类型或形状在运行期才确定**，
无法在编译期用 `f64` / `double[N]` / 固定 `struct` 表达。

| # | 类别 | 动态特性 | 为什么必须运行时 |
|---|------|----------|------------------|
| A | 动态形状 | 数组形参（`function y=f(A)`，A 大小随调用变）、逻辑下标 `A(A>0)`、增长数组 `x(end+1)=...`、`repmat`/`cat` 等形状随输入变 | 维度运行期才定，编译期只能给「形状描述符 + 数据指针」 |
| B | 动态类型 | cell 数组（`{...}`、`c{i}`、`c{i}=`、`cell(...)`）、字符串/字符数组、struct 的动态字段 `s.(name)` | 元素个数/类型/字段名运行期才定，需要「带标签的盒子」 |
| C | 动态分派 | `feval`、函数句柄、匿名函数 `@`、字符串调用、`classdef` 方法分派、`eval`/`evalin`/`assignin` | 被调目标运行期才定 |
| D | 错误/控制流语义 | `try`/`catch`、多值调用 `[a,b]=f()`（输出个数运行期定） | C 无原生异常；输出个数是运行期信息 |
| E | 内存/生命周期 | cell/struct 共享、数组增长重分配、逃逸值的所有权 | 需要引用计数 + 堆分配/重分配 |

> **不在运行时库**、而是靠静态分析解决（不引入运行时）：封闭世界的
> `varargin`/`varargout`（§12.2 特化）、常量下标的 `A(i,j)`、静态形状的字面量数组。
> 二者的分界就是 §4「形状可控 + 封闭世界」。

## 2. 统一动态值模型 `convmat_value`

MATLAB 所有动态值用一个带标签、带引用计数的盒子表达（对标 `mxArray`，但只覆盖
动态 tier 需要的那几种）。这是整套运行时库的核心。

```c
// ---- convmat runtime: dynamic tier value model (spec) ----
#include <stdint.h>
#include <stdbool.h>
#include <setjmp.h>

// 元素类型（标量/数组的元素）。
typedef enum convmat_dtype {
    CONVMAT_DOUBLE = 0,   // f64
    CONVMAT_LOGICAL,      // bool（_Bool）
    CONVMAT_INT32,        // int32_t
    CONVMAT_CHAR,         // char（MATLAB 原生 UTF-16，此处存 uint16_t）
} convmat_dtype;

// 值种类。
typedef enum convmat_kind {
    CONVMAT_EMPTY = 0,
    CONVMAT_SCALAR,       // 数值/bool/char 标量
    CONVMAT_ARRAY,        // 稠密数值数组（元素类型 = dtype）
    CONVMAT_CELL,         // cell 数组：元素是 convmat_value*
    CONVMAT_STRUCT,       // struct 数组：字段主序的 convmat_value*
    CONVMAT_FUNCTION,     // 分派令牌（feval/句柄/方法）
} convmat_kind;

// 形状描述符（MATLAB 列主序逻辑维度；dims[0]=行，dims[1]=列，…）。
typedef struct convmat_dims {
    int64_t ndims;      // >= 2（MATLAB 最小二维）
    int64_t *dims;      // 堆分配，长度 ndims，归属所属 value
} convmat_dims;

// 动态值盒子。
typedef struct convmat_value {
    int32_t refcount;       // 引用计数（值语义：copy 深拷贝，跨 cell/struct 共享时 +1）
    convmat_kind kind;
    convmat_dtype dtype;
    union {
        struct { double d; } scalar;
        struct {
            convmat_dims shape;
            void *data;          // 扁平列主序，元素大小按 dtype
            int64_t capacity;    // 已分配 numel（>= shape.numel）
            int owns_data;       // data 是否由本值释放
        } array;
        struct {
            convmat_dims shape;
            convmat_value **elems;  // 扁平列主序，每个元素 refcount 独立
        } cell;
        struct {
            convmat_dims shape;         // struct 数组形状（常见 1x1）
            int64_t nfields;
            const char **field_names;   // UTF-8，nfields 项
            convmat_value **fields;     // 字段主序：fields[f * numel + i]
        } strct;
        struct {
            int64_t handle_id;          // 函数注册表索引
        } func;
    } u;
} convmat_value;
```

设计要点：

- **`kind` 决定 payload**：`scalar` 是标量，`array` 是同类稠密数组，`cell`/`struct`
  递归包含 `convmat_value*`（故能表达异构），`function` 是分派令牌。
- **值语义**：`convmat_value_copy` 深拷贝，与 MATLAB 值类一致；`cell`/`struct` 的
  元素持有各自引用计数，`convmat_value_release` 在计数归零时递归释放。
- **动态形状集中在 `array`/`cell`/`struct` 的 `convmat_dims`**，标量无形状。

## 3. 形状描述符（`convmat_dims`）

静态 tier 用 `Shape::Static(dims)`（编译期已知，入栈/调用方分配）；动态 tier 用
`convmat_dims`（运行期 `int64_t* dims`）。§11.2 说的「数据指针 + 形状描述」就是
`convmat_value.array.data + convmat_value.array.shape`，即一个轻量 emxArray。

- 所有运行期维度/线性化都走列主序，与静态 tier 的 `Shape::linear` 一致。
- 线性索引计算放到运行时 helper（`convmat_linear_index`），避免在静态/动态边界
  各自维护一套下标换算。

## 4. 内存管理与生命周期

```c
convmat_value *convmat_value_new(convmat_kind kind, convmat_dtype dtype);
convmat_value *convmat_value_retain(convmat_value *v);       // refcount++
void           convmat_value_release(convmat_value *v);       // --ref；归零则递归释放
convmat_value *convmat_value_copy(const convmat_value *v);    // 深拷贝（值语义）

// 数组（含增长）
convmat_value *convmat_array_create(convmat_dtype dtype, int64_t ndims, const int64_t *dims);
void          *convmat_array_data(convmat_value *v);                       // 借用
void           convmat_array_resize(convmat_value *v, int64_t ndims, const int64_t *dims);

// cell / struct
convmat_value *convmat_cell_create(int64_t ndims, const int64_t *dims);
convmat_value *convmat_cell_get(const convmat_value *c, int64_t lin);       // retained
void           convmat_cell_set(convmat_value *c, int64_t lin, convmat_value *v); // 取 ref
convmat_value *convmat_struct_create(int64_t nfields, const char *const *names,
                                     int64_t ndims, const int64_t *dims);
int64_t        convmat_struct_field_index(const convmat_value *s, const char *name); // -1 缺失
convmat_value *convmat_struct_get(const convmat_value *s, int64_t field, int64_t lin);
void           convmat_struct_set(convmat_value *s, int64_t field, int64_t lin, convmat_value *v);

// 形状/索引
int64_t        convmat_numel(const convmat_value *v);
int64_t        convmat_linear_index(const convmat_value *v, const int64_t *subs);
```

对应 §11.3 的第 3 条（动态形状/增长走堆 + 运行时 shim 管理）。引用计数是 cell/struct
共享的根基；静态 tier 不共享（栈/输出指针，所有权固定），所以**只有跨入动态 tier 的
值才带 refcount**。

## 5. 运行时 ABI（动态函数调用）

动态 tier 的函数统一走「运行时 ABI」，覆盖开放世界的 `varargin`/`varargout`、多值调用
`[a,b]=f()`、`feval`/函数句柄（§12.3）：

```c
// 入参 (nargs, args)，出参 (nouts, outs)。outs 由调用方预置 nouts 个空槽，被调方填充。
typedef void (*convmat_runtime_fn)(int64_t nargs, convmat_value **args,
                                   int64_t nouts,  convmat_value **outs);
```

- 静态 ABI（§5：标量走返回值、数组走输出指针）与运行时 ABI 在**同一程序共存**，
  按「函数/表达式是否动态」选择（§11.1 函数级 + 表达式级混合）。
- 动态函数的输入/输出都是 `convmat_value*`（带 refcount）；静态侧通过 §6 的桥接
  与它互转。

## 6. 静态 ↔ 动态桥接（seam）

`defer_to_runtime` 是这个 seam 的唯一入口。落地后它不再报错，而是做两件事：

1. **静态 → 动态**（把 `double[N]` / `struct s0` / 标量传入被 defer 的调用）：
   box 成 `convmat_value`（借用的 `array` 用 `owns_data=0`，不拷贝），调用结束 release。
2. **动态 → 静态**（defer 调用返回的 `convmat_value*` 被静态代码消费）：
   若 `kind`/`dtype`/`dims` 匹配某个静态已知布局则 unbox 成 `double[N]`；否则继续
   留在动态 tier（消费方也得是动态函数）。

- 需要把 §4/§11.1「谁分配、谁释放」在 seam 处显式落成 `retain`/`release` 配对，
  避免跨 tier 的所有权歧义。
- seam 也是未来「运行时内建（`sort`/`find`/`mean`/`fft`…，见 `docs/unsupported.md` §7）
  的接入点：这些内建在动态 tier 里实现为操作 `convmat_value` 的 `convmat_*` 函数。

## 7. try/catch 与错误传播

C 无原生异常，运行时库用 `setjmp`/`longjmp` + 线程本地错误记录（MEX 内部同款）：

```c
typedef struct convmat_error {
    jmp_buf jmp;
    char message[256];
    int armed;            // 是否有 catch 在等
} convmat_error;

void convmat_error_throw(const char *msg);   // longjmp 到最近 armed 的 catch
// try 块：convmat_error_enter(&err)（setjmp 返回 0 = 进 try；非 0 = 抛回）；
// catch 块：convmat_error_leave(&err)。
```

`try`/`catch` 降级为：`try_body` 里任何动态调用/内建出错 → `convmat_error_throw` →
跳回 catch 块。静态 tier 的错误（如 `assert`）后续也走同一通道，保证「静态/动态出错
语义一致」。

## 8. 分阶段落地（依赖 P7）

1. **先判值分级**：扩展 `infer_locals`（自写轻量类型/形状推断，见 §10.5 建议），
   标出 `Dynamic` 值 → 只有这些才走运行时库。
2. **运行时库内核**：落地 §2–§4（`convmat_value` + 生命周期 + 数组/cell/struct helper），
   作为内联 C（沿用 `helper_source` 的 `VerbatimOp` 发射机制，见 `src/runtime/mod.rs`）。
3. **seam**：把 `defer_to_runtime` 从报错改成 §6 的桥接 + 动态 ABI 调用。
4. **逐特性接入**：按 §1 优先级（先 A 动态形状，再 B cell/struct/string，再 D
   try/catch，最后 C 动态分派），每接一个补一组「最小 `.m` → IR → C」+ `run_*` 测试。
5. **运行时内建**：`sort`/`find`/`mean`/`fft` 等在动态 tier 实现（§6），逐步从
   `docs/unsupported.md` §7 移除。
