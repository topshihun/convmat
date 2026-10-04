//! The `emitc` dialect: C-level statements, expressions, and top-level
//! directives.
//!
//! The `matlab -> emitc` lowering rewrites MATLAB-level ops into these C-level
//! primitives (declarations, assignments, ternary expressions, `libm` calls,
//! and C-style control flow). The C emitter then pretty-prints this dialect
//! almost 1:1. Value types are shared with the `matlab` dialect (a `f64` is a
//! `f64` everywhere); only the operations differ.
//!
//! Containers (`module`/`func`) are **pliron's builtin**
//! [`ModuleOp`](pliron::builtin::ops::ModuleOp) /
//! [`FuncOp`](pliron::builtin::ops::FuncOp); this dialect only contributes the
//! C-level body ops plus the top-level directives that the builtin dialect can't
//! express: [`IncludeOp`] (`#include`), [`DefineOp`]/[`UndefOp`]
//! (`#define`/`#undef`), and [`VerbatimOp`] as a raw escape hatch for anything
//! else (`typedef`, `struct`, `#pragma`, `extern`, ...).

use pliron::{
    builtin::{
        attributes::{BoolAttr, FPDoubleAttr, StringAttr},
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NRegionsInterface, NResultsInterface,
            OneOpdInterface, OneRegionInterface, OneResultInterface,
        },
        types::FP64Type,
    },
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    region::Region,
    value::Value,
};

use crate::dialects::matlab::{BinOpKind, BoolType, BoxType, CmpKind, PtrType};

/// Declare a local array: `double <name>[<size>]` (stack), optionally `static`,
/// or `double* <name> = new double[<size>]` (heap).
#[pliron_op(
    name = "emitc.declare",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (declare_name: StringAttr, declare_static: BoolAttr, declare_heap: BoolAttr),
    verifier = "succ",
)]
pub struct DeclareOp;

impl DeclareOp {
    pub fn new(ctx: &mut Context, name: &str, array_ty: TypeHandle) -> Self {
        Self::with_flags(ctx, name, array_ty, false, false)
    }

    pub fn new_static(ctx: &mut Context, name: &str, array_ty: TypeHandle) -> Self {
        Self::with_flags(ctx, name, array_ty, true, false)
    }

    pub fn new_heap(ctx: &mut Context, name: &str, array_ty: TypeHandle) -> Self {
        Self::with_flags(ctx, name, array_ty, false, true)
    }

    fn with_flags(
        ctx: &mut Context,
        name: &str,
        array_ty: TypeHandle,
        is_static: bool,
        is_heap: bool,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![array_ty],
            vec![],
            vec![],
            0,
        );
        let op = DeclareOp { op };
        op.set_attr_declare_name(ctx, StringAttr::new(name.to_string()));
        op.set_attr_declare_static(ctx, BoolAttr::new(is_static));
        op.set_attr_declare_heap(ctx, BoolAttr::new(is_heap));
        op
    }

    pub fn name(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_declare_name(ctx)
                .expect("declare name")
                .clone(),
        )
    }

    pub fn is_static(&self, ctx: &Context) -> bool {
        bool::from(
            self.get_attr_declare_static(ctx)
                .expect("declare static")
                .clone(),
        )
    }

    pub fn is_heap(&self, ctx: &Context) -> bool {
        bool::from(
            self.get_attr_declare_heap(ctx)
                .expect("declare heap")
                .clone(),
        )
    }
}

/// Free a heap-allocated array (`delete[] <name>`).
#[pliron_op(
    name = "emitc.delete",
    format,
    interfaces = [OneOpdInterface, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct DeleteOp;

impl DeleteOp {
    pub fn new(ctx: &mut Context, array: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![array],
            vec![],
            0,
        );
        DeleteOp { op }
    }
}

/// Allocate a runtime-length heap buffer: `double* <result> = new
/// double[(int64_t)<len>];`. `len` is an `f64` operand; the buffer is released
/// by [`DeleteOp`].
#[pliron_op(
    name = "emitc.heap_alloc",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    verifier = "succ",
)]
pub struct HeapAllocOp;

impl HeapAllocOp {
    pub fn new(ctx: &mut Context, len: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![PtrType::get(ctx).into()],
            vec![len],
            vec![],
            0,
        );
        HeapAllocOp { op }
    }
}

/// A `f64` literal.
#[pliron_op(
    name = "emitc.literal",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (literal_value: FPDoubleAttr),
    verifier = "succ",
)]
pub struct LiteralOp;

impl LiteralOp {
    pub fn new(ctx: &mut Context, value: f64) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![FP64Type::get(ctx).into()],
            vec![],
            vec![],
            0,
        );
        let op = LiteralOp { op };
        op.set_attr_literal_value(ctx, FPDoubleAttr::from(value));
        op
    }

    pub fn value(&self, ctx: &Context) -> f64 {
        f64::from(
            self.get_attr_literal_value(ctx)
                .expect("literal value")
                .clone(),
        )
    }
}

/// A binary arithmetic expression (`+`, `-`, `*`, `/`) over `f64`.
#[pliron_op(
    name = "emitc.binop",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (ebinop_kind: BinOpKind),
    verifier = "succ",
)]
pub struct BinOp;

impl BinOp {
    pub fn new(ctx: &mut Context, kind: BinOpKind, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![FP64Type::get(ctx).into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let op = BinOp { op };
        op.set_attr_ebinop_kind(ctx, kind);
        op
    }

    pub fn kind(&self, ctx: &Context) -> BinOpKind {
        *self.get_attr_ebinop_kind(ctx).expect("binop kind")
    }
}

/// An ordered comparison over `f64` producing a `bool`.
#[pliron_op(
    name = "emitc.cmp",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (ecmp_kind: CmpKind),
    verifier = "succ",
)]
pub struct CmpOp;

impl CmpOp {
    pub fn new(ctx: &mut Context, kind: CmpKind, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![BoolType::get(ctx).into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let op = CmpOp { op };
        op.set_attr_ecmp_kind(ctx, kind);
        op
    }

    pub fn kind(&self, ctx: &Context) -> CmpKind {
        *self.get_attr_ecmp_kind(ctx).expect("cmp kind")
    }
}

/// A ternary `cond ? true_value : false_value`.
#[pliron_op(
    name = "emitc.ternary",
    format,
    interfaces = [NOpdsInterface<3>, OneResultInterface],
    verifier = "succ",
)]
pub struct TernaryOp;

impl TernaryOp {
    pub fn new(ctx: &mut Context, cond: Value, true_value: Value, false_value: Value) -> Self {
        let result_ty = true_value.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![cond, true_value, false_value],
            vec![],
            0,
        );
        TernaryOp { op }
    }
}

/// A call to an external `libm` function returning `f64`.
#[pliron_op(
    name = "emitc.call",
    format,
    interfaces = [OneResultInterface],
    attributes = (ecall_callee: StringAttr),
    verifier = "succ",
)]
pub struct CallOp;

impl CallOp {
    pub fn new(ctx: &mut Context, callee: &str, args: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![FP64Type::get(ctx).into()],
            args,
            vec![],
            0,
        );
        let op = CallOp { op };
        op.set_attr_ecall_callee(ctx, StringAttr::new(callee.to_string()));
        op
    }

    pub fn callee(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_ecall_callee(ctx)
                .expect("call callee")
                .clone(),
        )
    }
}

/// A call to an external function returning nothing (a runtime helper that
/// writes its result through an out-parameter).
#[pliron_op(
    name = "emitc.call_void",
    format,
    interfaces = [NResultsInterface<0>],
    attributes = (ecall_void_callee: StringAttr),
    verifier = "succ",
)]
pub struct CallVoidOp;

impl CallVoidOp {
    pub fn new(ctx: &mut Context, callee: &str, args: Vec<Value>) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], args, vec![], 0);
        let op = CallVoidOp { op };
        op.set_attr_ecall_void_callee(ctx, StringAttr::new(callee.to_string()));
        op
    }

    pub fn callee(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_ecall_void_callee(ctx)
                .expect("call_void callee")
                .clone(),
        )
    }
}

/// A call to an external function returning a boxed dynamic value
/// (`convmat_value*`), e.g. the cell constructor.
#[pliron_op(
    name = "emitc.call_box",
    format,
    interfaces = [OneResultInterface],
    attributes = (ecall_box_callee: StringAttr),
    verifier = "succ",
)]
pub struct CallBoxOp;

impl CallBoxOp {
    pub fn new(ctx: &mut Context, callee: &str, args: Vec<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![BoxType::get(ctx).into()],
            args,
            vec![],
            0,
        );
        let op = CallBoxOp { op };
        op.set_attr_ecall_box_callee(ctx, StringAttr::new(callee.to_string()));
        op
    }

    pub fn callee(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_ecall_box_callee(ctx)
                .expect("call_box callee")
                .clone(),
        )
    }
}

/// Load `array[index]` (with an `i64` index).
#[pliron_op(
    name = "emitc.load",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    verifier = "succ",
)]
pub struct LoadOp;

impl LoadOp {
    pub fn new(ctx: &mut Context, array: Value, index: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![FP64Type::get(ctx).into()],
            vec![array, index],
            vec![],
            0,
        );
        LoadOp { op }
    }
}

/// Assign `array[index] = value`.
#[pliron_op(
    name = "emitc.assign",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct AssignOp;

impl AssignOp {
    pub fn new(ctx: &mut Context, array: Value, index: Value, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![array, index, value],
            vec![],
            0,
        );
        AssignOp { op }
    }
}

/// Read a named `f64` field from a struct (`cell.field`).
#[pliron_op(
    name = "emitc.struct_get",
    format,
    interfaces = [OneOpdInterface, OneResultInterface],
    attributes = (estruct_get_field: StringAttr),
    verifier = "succ",
)]
pub struct StructGetOp;

impl StructGetOp {
    pub fn new(ctx: &mut Context, cell: Value, field: &str) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![FP64Type::get(ctx).into()],
            vec![cell],
            vec![],
            0,
        );
        let op = StructGetOp { op };
        op.set_attr_estruct_get_field(ctx, StringAttr::new(field.to_string()));
        op
    }

    pub fn field(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_estruct_get_field(ctx)
                .expect("struct_get field")
                .clone(),
        )
    }
}

/// Write a named `f64` field into a struct (`cell.field = value`).
#[pliron_op(
    name = "emitc.struct_set",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (estruct_set_field: StringAttr),
    verifier = "succ",
)]
pub struct StructSetOp;

impl StructSetOp {
    pub fn new(ctx: &mut Context, cell: Value, field: &str, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![cell, value],
            vec![],
            0,
        );
        let op = StructSetOp { op };
        op.set_attr_estruct_set_field(ctx, StringAttr::new(field.to_string()));
        op
    }

    pub fn field(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_estruct_set_field(ctx)
                .expect("struct_set field")
                .clone(),
        )
    }
}

/// Copy a struct value into a struct cell (`dest = src`).
#[pliron_op(
    name = "emitc.struct_copy",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct StructCopyOp;

impl StructCopyOp {
    pub fn new(ctx: &mut Context, dest: Value, src: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![dest, src],
            vec![],
            0,
        );
        StructCopyOp { op }
    }
}

/// A statement-form conditional with `then` and `else` regions.
#[pliron_op(
    name = "emitc.if",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>, NRegionsInterface<2>],
    verifier = "succ",
)]
pub struct IfOp;

impl IfOp {
    pub fn new(ctx: &mut Context, cond: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![cond],
            vec![],
            2,
        );
        IfOp { op }
    }

    pub fn condition(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn then_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn else_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(1)
    }
}

/// A `while` loop (`before` recomputes the condition, `after` is the body).
#[pliron_op(
    name = "emitc.while",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>, NRegionsInterface<2>],
    verifier = "succ",
)]
pub struct WhileOp;

impl WhileOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 2);
        WhileOp { op }
    }

    pub fn before_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn after_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(1)
    }
}

/// A counted `for` loop.
#[pliron_op(
    name = "emitc.for",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>, OneRegionInterface],
    verifier = "succ",
)]
pub struct ForOp;

impl ForOp {
    pub fn new(ctx: &mut Context, start: Value, end: Value, step: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![start, end, step],
            vec![],
            1,
        );
        ForOp { op }
    }

    pub fn body_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}

/// A `for` loop with a runtime direction-aware condition:
/// `for (iv = start; (step >= 0) ? (iv <= end) : (iv >= end); iv += step)`.
/// This is the C form of MATLAB's `for i = start : step : end`, and it gives
/// `break`/`continue` their natural C semantics (unlike a `while(true)` loop).
#[pliron_op(
    name = "emitc.range_for",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>, OneRegionInterface],
    verifier = "succ",
)]
pub struct RangeForOp;

impl RangeForOp {
    pub fn new(ctx: &mut Context, start: Value, end: Value, step: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![start, end, step],
            vec![],
            1,
        );
        RangeForOp { op }
    }

    pub fn body_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}

/// Terminates a `before` region with the loop condition.
#[pliron_op(
    name = "emitc.condition",
    format,
    interfaces = [IsTerminatorInterface, OneOpdInterface, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct ConditionOp;

impl ConditionOp {
    pub fn new(ctx: &mut Context, cond: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![cond],
            vec![],
            0,
        );
        ConditionOp { op }
    }
}

/// Terminates a structured region with no carried values.
#[pliron_op(
    name = "emitc.yield",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NOpdsInterface<0>],
    verifier = "succ",
)]
pub struct YieldOp;

impl YieldOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        YieldOp { op }
    }
}

/// A `break` statement (exits the innermost loop).
#[pliron_op(
    name = "emitc.break",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct BreakOp;

impl BreakOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        BreakOp { op }
    }
}

/// A `continue` statement (jumps to the next loop iteration).
#[pliron_op(
    name = "emitc.continue",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct ContinueOp;

impl ContinueOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        ContinueOp { op }
    }
}

/// Returns the scalar results of a function (a terminator).
#[pliron_op(
    name = "emitc.return",
    format,
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct ReturnOp;

impl ReturnOp {
    pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0);
        ReturnOp { op }
    }
}

/// A `#include <header>` (system) or `#include "header"` (local) directive.
#[pliron_op(
    name = "emitc.include",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (emitc_include_header: StringAttr, emitc_include_system: BoolAttr),
    verifier = "succ",
)]
pub struct IncludeOp;

impl IncludeOp {
    pub fn new(ctx: &mut Context, header: &str, is_system: bool) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let op = IncludeOp { op };
        op.set_attr_emitc_include_header(ctx, StringAttr::new(header.to_string()));
        op.set_attr_emitc_include_system(ctx, BoolAttr::new(is_system));
        op
    }

    /// The header file name (without the angle brackets or quotes).
    pub fn header(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_emitc_include_header(ctx)
                .expect("include header")
                .clone(),
        )
    }

    /// `true` for `#include <...>`, `false` for `#include "..."`.
    pub fn is_system(&self, ctx: &Context) -> bool {
        bool::from(
            self.get_attr_emitc_include_system(ctx)
                .expect("include system")
                .clone(),
        )
    }
}

/// A `#define NAME [VALUE]` macro definition. An empty value emits a bare
/// `#define NAME` (an object-like macro with no replacement text). Function-like
/// macros are represented by including the parameter list in `name`.
#[pliron_op(
    name = "emitc.define",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (emitc_define_name: StringAttr, emitc_define_value: StringAttr),
    verifier = "succ",
)]
pub struct DefineOp;

impl DefineOp {
    pub fn new(ctx: &mut Context, name: &str, value: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let op = DefineOp { op };
        op.set_attr_emitc_define_name(ctx, StringAttr::new(name.to_string()));
        op.set_attr_emitc_define_value(ctx, StringAttr::new(value.to_string()));
        op
    }

    /// The macro name (optionally with a parameter list).
    pub fn name(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_emitc_define_name(ctx)
                .expect("define name")
                .clone(),
        )
    }

    /// The replacement token list (empty for a bare `#define NAME`).
    pub fn value(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_emitc_define_value(ctx)
                .expect("define value")
                .clone(),
        )
    }
}

/// A `#undef NAME` directive.
#[pliron_op(
    name = "emitc.undef",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (emitc_undef_name: StringAttr),
    verifier = "succ",
)]
pub struct UndefOp;

impl UndefOp {
    pub fn new(ctx: &mut Context, name: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let op = UndefOp { op };
        op.set_attr_emitc_undef_name(ctx, StringAttr::new(name.to_string()));
        op
    }

    pub fn name(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_emitc_undef_name(ctx)
                .expect("undef name")
                .clone(),
        )
    }
}

/// A verbatim top-level snippet of C source. This is the escape hatch for any C
/// construct that has no dedicated op yet: `typedef`, `struct`/`enum`, `extern`
/// declarations, `#pragma`, conditional compilation, and so on.
#[pliron_op(
    name = "emitc.verbatim",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (emitc_verbatim_source: StringAttr),
    verifier = "succ",
)]
pub struct VerbatimOp;

impl VerbatimOp {
    pub fn new(ctx: &mut Context, source: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let op = VerbatimOp { op };
        op.set_attr_emitc_verbatim_source(ctx, StringAttr::new(source.to_string()));
        op
    }

    pub fn source(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_emitc_verbatim_source(ctx)
                .expect("verbatim source")
                .clone(),
        )
    }
}
