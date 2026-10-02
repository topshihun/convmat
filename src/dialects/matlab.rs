//! The `matlab` dialect: MATLAB-level semantics produced by `mir_to_mlir`.
//!
//! This is the high-level IR that the MIR lowerer emits. It captures what a
//! MATLAB program *means*: dense column-major arrays, scalar arithmetic,
//! comparisons, `libm` built-in calls, and structured control flow. It is
//! deliberately free of C-specific concerns (pointer ABI, loop syntax), which
//! are introduced later by the `emitc` dialect.

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
    derive::{pliron_attr, pliron_op, pliron_type},
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    region::Region,
    value::Value,
};

/// A MATLAB logical value (maps to C++ `bool`).
#[pliron_type(name = "matlab.bool", format, generate_get = true, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct BoolType;

/// A statically-shaped, column-major dense array of `f64` (the `memref` analog).
#[pliron_type(name = "matlab.array", generate_get = true, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct ArrayType {
    dims: Vec<i64>,
}

impl pliron::printable::Printable for ArrayType {
    fn fmt(
        &self,
        _ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "array<{}>", self.numel())
    }
}

// Array types are never parsed from text (the compiler never round-trips IR
// through a textual form), so the parser is a stub that yields a placeholder.
impl pliron::parsable::Parsable for ArrayType {
    type Arg = ();
    type Parsed = pliron::r#type::TypedHandle<Self>;

    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        _arg: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        use pliron::combine::Parser;
        let ctx = &*state_stream.state.ctx;
        pliron::combine::value(ArrayType::get(ctx, vec![1]))
            .parse_stream(state_stream)
            .into()
    }
}

impl ArrayType {
    /// The static dimensions of the array (column-major).
    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    /// The flattened element count.
    pub fn numel(&self) -> i64 {
        self.dims.iter().product()
    }
}

/// The arithmetic kind of a [`BinOp`].
#[pliron_attr(name = "matlab.binop_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

/// The predicate of a [`CmpOp`].
#[pliron_attr(name = "matlab.cmp_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A `f64` literal.
#[pliron_op(
    name = "matlab.constant",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    attributes = (constant_value: FPDoubleAttr),
    verifier = "succ",
)]
pub struct ConstantOp;

impl ConstantOp {
    pub fn new(ctx: &mut Context, value: f64) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![FP64Type::get(ctx).into()],
            vec![],
            vec![],
            0,
        );
        let op = ConstantOp { op };
        op.set_attr_constant_value(ctx, FPDoubleAttr::from(value));
        op
    }

    pub fn value(&self, ctx: &Context) -> f64 {
        f64::from(
            self.get_attr_constant_value(ctx)
                .expect("constant value")
                .clone(),
        )
    }
}

/// A scalar binary arithmetic operation (`+`, `-`, `*`, `/`) over `f64`.
#[pliron_op(
    name = "matlab.binop",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (binop_kind: BinOpKind),
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
        op.set_attr_binop_kind(ctx, kind);
        op
    }

    pub fn kind(&self, ctx: &Context) -> BinOpKind {
        *self.get_attr_binop_kind(ctx).expect("binop kind")
    }

    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

/// An ordered comparison over `f64` producing a `bool`.
#[pliron_op(
    name = "matlab.cmp",
    format,
    interfaces = [NOpdsInterface<2>, OneResultInterface],
    attributes = (cmp_kind: CmpKind),
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
        op.set_attr_cmp_kind(ctx, kind);
        op
    }

    pub fn kind(&self, ctx: &Context) -> CmpKind {
        *self.get_attr_cmp_kind(ctx).expect("cmp kind")
    }
}

/// A `bool`-conditioned selection: `cond ? true_value : false_value`.
#[pliron_op(
    name = "matlab.select",
    format,
    interfaces = [NOpdsInterface<3>, OneResultInterface],
    verifier = "succ",
)]
pub struct SelectOp;

impl SelectOp {
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
        SelectOp { op }
    }
}

/// A call to an external `libm` function returning `f64`.
#[pliron_op(
    name = "matlab.call",
    format,
    interfaces = [OneResultInterface],
    attributes = (call_callee: StringAttr),
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
        op.set_attr_call_callee(ctx, StringAttr::new(callee.to_string()));
        op
    }

    pub fn callee(&self, ctx: &Context) -> String {
        let attr = self.get_attr_call_callee(ctx).expect("call callee").clone();
        String::from(attr)
    }
}

/// Allocate a statically-shaped array local (stack slot).
#[pliron_op(
    name = "matlab.alloca",
    format,
    interfaces = [NOpdsInterface<0>, OneResultInterface],
    verifier = "succ",
)]
pub struct AllocaOp;

impl AllocaOp {
    pub fn new(ctx: &mut Context, array_ty: TypeHandle) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![array_ty],
            vec![],
            vec![],
            0,
        );
        AllocaOp { op }
    }
}

/// Load a single `f64` element from an array at a linear (column-major) index.
#[pliron_op(
    name = "matlab.load",
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

/// Store a single `f64` element into an array at a linear index.
#[pliron_op(
    name = "matlab.store",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
    verifier = "succ",
)]
pub struct StoreOp;

impl StoreOp {
    pub fn new(ctx: &mut Context, array: Value, index: Value, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![array, index, value],
            vec![],
            0,
        );
        StoreOp { op }
    }
}

/// A statement-form conditional with `then` and `else` regions.
#[pliron_op(
    name = "matlab.if",
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

/// A `while` loop: the `before` region recomputes the condition each iteration
/// (terminated by [`ConditionOp`]); the `after` region is the body.
#[pliron_op(
    name = "matlab.while",
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

/// A counted `for` loop `for i = start : step : end`. The body region's single
/// block has one argument: the induction variable (`f64`).
#[pliron_op(
    name = "matlab.for",
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

/// Terminates the `before` region of a [`WhileOp`], carrying the loop condition.
#[pliron_op(
    name = "matlab.condition",
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

/// Terminates a structured region (if/while/for body) with no carried values.
#[pliron_op(
    name = "matlab.yield",
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

/// Returns the scalar results of a function (a terminator).
#[pliron_op(
    name = "matlab.return",
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
