//! The `emitc` dialect: C-level statements and expressions.
//!
//! The `matlab -> emitc` lowering rewrites MATLAB-level ops into these C-level
//! primitives (declarations, assignments, ternary expressions, `libm` calls,
//! and C-style control flow). The C emitter then pretty-prints this dialect
//! almost 1:1. Value types are shared with the `matlab` dialect (a `f64` is a
//! `f64` everywhere); only the operations differ.

use pliron::{
    builtin::{
        attributes::{FPDoubleAttr, StringAttr},
        op_interfaces::{
            IsTerminatorInterface, NOpdsInterface, NRegionsInterface, NResultsInterface,
            OneOpdInterface, OneRegionInterface, OneResultInterface,
        },
        types::FP64Type,
    },
    context::{Context, Ptr},
    derive::{pliron_attr, pliron_op},
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    region::Region,
    value::Value,
};

use crate::dialects::matlab::{BinOpKind, BoolType, CmpKind};

/// A small `u64` attribute (used for scalar-result counts).
#[pliron_attr(name = "emitc.count", format, verifier = "succ")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CountAttr(pub u64);

/// A function: carries its symbol name and the count of scalar results.
#[pliron_op(
    name = "emitc.func",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>, OneRegionInterface],
    attributes = (func_name: StringAttr, func_nresults: CountAttr),
    verifier = "succ",
)]
pub struct FuncOp;

impl FuncOp {
    pub fn new(ctx: &mut Context, name: &str, n_results: u64) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let op = FuncOp { op };
        op.set_attr_func_name(ctx, StringAttr::new(name.to_string()));
        op.set_attr_func_nresults(ctx, CountAttr(n_results));
        op
    }

    pub fn name(&self, ctx: &Context) -> String {
        String::from(self.get_attr_func_name(ctx).expect("func name").clone())
    }

    pub fn n_results(&self, ctx: &Context) -> u64 {
        self.get_attr_func_nresults(ctx).expect("func n_results").0
    }

    pub fn body_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}

/// Declare a local array `double <name>[<size>]`.
#[pliron_op(
    name = "emitc.declare",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (declare_name: StringAttr),
    verifier = "succ",
)]
pub struct DeclareOp;

impl DeclareOp {
    pub fn new(ctx: &mut Context, name: &str, array_ty: TypeHandle) -> Self {
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
        op
    }

    pub fn name(&self, ctx: &Context) -> String {
        String::from(
            self.get_attr_declare_name(ctx)
                .expect("declare name")
                .clone(),
        )
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
