//! Layer 3: lower MIR to MLIR core dialects.
//!
//! A [`runmat_mir::MirAssembly`] is traversed and lowered to `func`/`arith`/
//! `scf`/`memref` MLIR via `melior`. The result is serialized as MLIR text and
//! handed to [`crate::passes`] for the emitc conversion.
//!
//! The lowering is deliberately straightforward (no optimization). Every MIR
//! local becomes a stack `memref<1xf64>`; assignments are `memref.store`s and
//! reads are `memref.load`s. Structured control flow is recovered from the MIR
//! CFG and emitted as `scf.if`/`scf.while`, which sidesteps explicit SSA phi
//! construction while remaining faithful to the source semantics.

use std::collections::{HashMap, HashSet};

use melior::{
    dialect::{arith, func, memref, scf, DialectRegistry},
    ir::{
        attribute::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        block::BlockLike,
        r#type::{FunctionType, MemRefType},
        Block, Location, Module, Region, RegionLike, Type, Value,
    },
    utility::register_all_dialects,
    Context,
};

use crate::error::{Error, Result};

use runmat_hir::OperatorKind;
use runmat_mir::{
    BasicBlockId, MirAssembly, MirBody, MirConstant, MirLocalId, MirOperand, MirPlace, MirRvalue,
    MirShortCircuitOp, MirStmt, MirStmtKind, MirTerminatorKind,
};

/// Create an MLIR context with all registered dialects loaded.
pub fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}

/// Lower MIR to MLIR text.
pub fn lower(mir: &MirAssembly) -> Result<String> {
    let context = create_context();
    let module = lower_to_module(&context, mir)?;
    Ok(module.as_operation().to_string())
}

/// Lower MIR into a new MLIR module owned by `context`.
pub fn lower_to_module<'c>(context: &'c Context, mir: &MirAssembly) -> Result<Module<'c>> {
    let location = Location::unknown(context);
    let module = Module::new(location);

    let cx = Cx {
        context,
        location,
        float: Type::float64(context),
        index: Type::index(context),
        scalar_memref: MemRefType::contiguous(Type::float64(context), &[1], None),
    };

    for (function_id, body) in &mir.bodies {
        lower_function(&cx, mir, *function_id, body, &module)?;
    }

    Ok(module)
}

/// Immutable lowering context shared across all functions.
struct Cx<'c> {
    context: &'c Context,
    location: Location<'c>,
    float: Type<'c>,
    index: Type<'c>,
    scalar_memref: MemRefType<'c>,
}

/// Lower a single MIR body into a `func.func` appended to `module`.
fn lower_function<'c>(
    cx: &Cx<'c>,
    mir: &MirAssembly,
    function_id: runmat_hir::FunctionId,
    body: &MirBody,
    module: &Module<'c>,
) -> Result<()> {
    let metadata = mir
        .functions
        .get(&function_id)
        .ok_or_else(|| Error::Backend(format!("missing metadata for {function_id:?}")))?;
    let name = metadata.name.0.clone();

    // Map BindingId -> MirLocalId so parameters can be resolved from the ABI.
    let mut binding_to_local = HashMap::new();
    for local in &body.locals {
        if let Some(binding) = local.binding {
            binding_to_local.insert(binding, local.id);
        }
    }

    let params: Vec<MirLocalId> = body
        .abi
        .fixed_inputs
        .iter()
        .map(|binding| {
            binding_to_local
                .get(binding)
                .copied()
                .ok_or_else(|| Error::Backend(format!("no local for input {binding:?}")))
        })
        .collect::<Result<_>>()?;

    // Entry block with one `f64` argument per parameter.
    let entry = Block::new(
        &params
            .iter()
            .map(|_| (cx.float, cx.location))
            .collect::<Vec<_>>(),
    );

    // Allocate one stack cell per local and store incoming parameters into it.
    let mut locals: HashMap<usize, Value> = HashMap::new();
    for local in &body.locals {
        let alloca = entry.append_operation(memref::alloca(
            cx.context,
            cx.scalar_memref,
            &[],
            &[],
            None,
            cx.location,
        ));
        let value: Value = alloca
            .result(0)
            .map_err(|e| Error::Backend(format!("memref.alloca result: {e}")))?
            .into();
        locals.insert(local.id.0, value);
    }

    for (index, param) in params.iter().enumerate() {
        let argument: Value = entry
            .argument(index)
            .map_err(|e| Error::Backend(format!("block argument: {e}")))?
            .into();
        let target = locals
            .get(&param.0)
            .copied()
            .ok_or_else(|| Error::Backend(format!("no cell for parameter {param:?}")))?;
        let zero = index_zero(cx, &entry)?;
        entry.append_operation(memref::store(argument, target, &[zero], cx.location));
    }

    let cfg = compute_cfg(body);
    let lowerer = FuncLowerer {
        cx,
        body,
        locals,
        preds: cfg.preds,
        dominators: cfg.dominators,
    };
    lowerer.lower_region(&entry, 0, None)?;
    drop(lowerer);

    let input_types = vec![cx.float; params.len()];
    let output_types = vec![cx.float; body.abi.fixed_outputs.len()];

    let region = Region::new();
    region.append_block(entry);

    module.body().append_operation(func::func(
        cx.context,
        StringAttribute::new(cx.context, name.as_str()),
        TypeAttribute::new(FunctionType::new(cx.context, &input_types, &output_types).into()),
        region,
        &[],
        cx.location,
    ));

    Ok(())
}

/// Per-function lowering state.
///
/// `'e` is the lifetime of the function entry block; `locals` holds the
/// `memref` values allocated there. Nested control-flow blocks borrow their own
/// values transiently and never outlive this struct.
struct FuncLowerer<'c, 'e, 'x, 'y> {
    cx: &'x Cx<'c>,
    body: &'y MirBody,
    locals: HashMap<usize, Value<'c, 'e>>,
    preds: Vec<Vec<usize>>,
    dominators: Vec<HashSet<usize>>,
}

impl<'c, 'e, 'x, 'y> FuncLowerer<'c, 'e, 'x, 'y> {
    /// Lower a structured region starting at `start` until control flows to
    /// `stop` (exclusive) or the function returns.
    ///
    /// Returns the block reached when the region exits (`Some`), or `None` when
    /// the region ends in a `return`/`unreachable`.
    fn lower_region<'b>(
        &self,
        block: &'b Block<'c>,
        start: usize,
        stop: Option<usize>,
    ) -> Result<Option<usize>> {
        let mut cur = start;

        loop {
            let mir_block = self
                .body
                .blocks
                .get(cur)
                .ok_or_else(|| Error::Backend(format!("block {cur} out of range")))?;

            // Loop headers first: their condition statements must be emitted in
            // the loop's "before" region so they re-run on every iteration.
            match &mir_block.terminator.kind {
                MirTerminatorKind::Branch {
                    cond,
                    then_block,
                    else_block,
                } if self.is_loop_header(cur) => {
                    self.lower_while(
                        block,
                        &mir_block.statements,
                        cond,
                        then_block.0,
                        else_block.0,
                        cur,
                    )?;
                    cur = else_block.0;
                    continue;
                }
                MirTerminatorKind::For {
                    binding,
                    iterable,
                    body_block,
                    exit_block,
                } => {
                    self.lower_for(block, binding, iterable, body_block.0, exit_block.0, cur)?;
                    cur = exit_block.0;
                    continue;
                }
                _ => {}
            }

            for stmt in &mir_block.statements {
                self.lower_stmt(block, stmt)?;
            }

            match &mir_block.terminator.kind {
                MirTerminatorKind::Return(operands) => {
                    if stop.is_some() {
                        return Err(Error::NotLowerable(
                            "`return` inside control flow is not supported yet".to_string(),
                        ));
                    }
                    let mut values = Vec::new();
                    for operand in operands {
                        values.push(self.lower_operand(block, operand)?);
                    }
                    block.append_operation(func::r#return(&values, self.cx.location));
                    return Ok(None);
                }
                MirTerminatorKind::Unreachable => return Ok(None),
                MirTerminatorKind::Goto(target) => {
                    let target = target.0;
                    if stop == Some(target) {
                        return Ok(Some(target));
                    }
                    if self.is_loop_header(target) {
                        cur = target;
                    } else {
                        return Err(Error::NotLowerable(format!(
                            "unstructured goto to block {target} is not supported yet"
                        )));
                    }
                }
                MirTerminatorKind::Branch {
                    cond,
                    then_block,
                    else_block,
                } => {
                    let merge = self.exit_of(then_block.0);
                    self.lower_if(block, cond, then_block.0, else_block.0, merge)?;
                    match merge {
                        Some(merge) => cur = merge,
                        None => return Ok(None),
                    }
                }
                MirTerminatorKind::Switch {
                    discr,
                    cases,
                    otherwise,
                } => {
                    let merge = self.exit_of(otherwise.0);
                    self.lower_switch(block, discr, cases, otherwise.0, merge)?;
                    match merge {
                        Some(merge) => cur = merge,
                        None => return Ok(None),
                    }
                }
                other => {
                    return Err(Error::NotLowerable(format!(
                        "unsupported terminator {other:?}"
                    )));
                }
            }
        }
    }

    /// Lower a statement into the current block.
    fn lower_stmt<'b>(&self, block: &'b Block<'c>, stmt: &MirStmt) -> Result<()> {
        match &stmt.kind {
            MirStmtKind::Assign { place, value } => {
                let MirPlace::Local(target) = place else {
                    return Err(Error::NotLowerable(format!(
                        "non-local assignment target {place:?}"
                    )));
                };
                let value = self.lower_rvalue(block, value)?;
                let cell = self
                    .locals
                    .get(&target.0)
                    .copied()
                    .ok_or_else(|| Error::Backend(format!("no cell for local {:?}", target)))?;
                let zero = index_zero(self.cx, block)?;
                block.append_operation(memref::store(value, cell, &[zero], self.cx.location));
                Ok(())
            }
            MirStmtKind::Expr(value) => {
                // Evaluate for (potential) side effects; the result is dropped.
                self.lower_rvalue(block, value)?;
                Ok(())
            }
            other => Err(Error::NotLowerable(format!(
                "unsupported statement {other:?}"
            ))),
        }
    }

    /// Lower a rvalue to a `f64` value in the current block.
    fn lower_rvalue<'b>(&self, block: &'b Block<'c>, rvalue: &MirRvalue) -> Result<Value<'c, 'b>> {
        match rvalue {
            MirRvalue::Use(operand) => self.lower_operand(block, operand),
            MirRvalue::Unary(op, operand) => {
                let value = self.lower_operand(block, operand)?;
                self.apply_unary(block, op, value)
            }
            MirRvalue::Binary(lhs, op, rhs) => {
                let l = self.lower_operand(block, lhs)?;
                let r = self.lower_operand(block, rhs)?;
                self.apply_binary(block, op, l, r)
            }
            MirRvalue::ShortCircuit {
                left,
                op,
                right_temps,
                right,
            } => {
                let l = self.lower_operand(block, left)?;
                for stmt in right_temps {
                    self.lower_stmt(block, stmt)?;
                }
                let r = self.lower_operand(block, right)?;
                self.apply_short_circuit(block, op, l, r)
            }
            other => Err(Error::NotLowerable(format!("rvalue {other:?}"))),
        }
    }

    /// Lower an operand to a `f64` value in the current block.
    fn lower_operand<'b>(
        &self,
        block: &'b Block<'c>,
        operand: &MirOperand,
    ) -> Result<Value<'c, 'b>> {
        match operand {
            MirOperand::Local(id) => {
                let cell = self
                    .locals
                    .get(&id.0)
                    .copied()
                    .ok_or_else(|| Error::Backend(format!("operand {id:?} has no cell")))?;
                let zero = index_zero(self.cx, block)?;
                let load = block.append_operation(memref::load(cell, &[zero], self.cx.location));
                Ok(load
                    .result(0)
                    .map_err(|e| Error::Backend(format!("memref.load result: {e}")))?
                    .into())
            }
            MirOperand::Constant(constant) => match constant {
                MirConstant::Number(text) => self.number_constant(block, text),
                MirConstant::IntegerLiteral(literal) => {
                    self.float_constant(block, literal.bits() as f64)
                }
                MirConstant::Bool(value) => {
                    self.float_constant(block, if *value { 1.0 } else { 0.0 })
                }
                other => Err(Error::NotLowerable(format!("constant {other:?}"))),
            },
            other => Err(Error::NotLowerable(format!("operand {other:?}"))),
        }
    }

    fn apply_unary<'b>(
        &self,
        block: &'b Block<'c>,
        op: &OperatorKind,
        value: Value<'c, 'b>,
    ) -> Result<Value<'c, 'b>> {
        match op {
            OperatorKind::UnaryPlus
            | OperatorKind::Transpose
            | OperatorKind::ConjugateTranspose => Ok(value),
            OperatorKind::UnaryMinus => {
                let op = block.append_operation(arith::negf(value, self.cx.location));
                Ok(op
                    .result(0)
                    .map_err(|e| Error::Backend(format!("negf result: {e}")))?
                    .into())
            }
            OperatorKind::Not => {
                let truthy = self.truthy(block, value)?;
                let zero = self.float_constant(block, 0.0)?;
                let one = self.float_constant(block, 1.0)?;
                self.select(block, truthy, zero, one)
            }
            other => Err(Error::NotLowerable(format!("unary operator {other:?}"))),
        }
    }

    fn apply_binary<'b>(
        &self,
        block: &'b Block<'c>,
        op: &OperatorKind,
        l: Value<'c, 'b>,
        r: Value<'c, 'b>,
    ) -> Result<Value<'c, 'b>> {
        use melior::dialect::arith::CmpfPredicate;

        match op {
            OperatorKind::Add => self.append_arith(block, arith::addf(l, r, self.cx.location)),
            OperatorKind::Subtract => self.append_arith(block, arith::subf(l, r, self.cx.location)),
            OperatorKind::MatrixMultiply | OperatorKind::ElementwiseMultiply => {
                self.append_arith(block, arith::mulf(l, r, self.cx.location))
            }
            OperatorKind::Mrdivide | OperatorKind::ElementwiseDivide => {
                self.append_arith(block, arith::divf(l, r, self.cx.location))
            }
            // Left division: `a \ b` == `b / a` and `a .\ b` == `b ./ a`.
            OperatorKind::Mldivide | OperatorKind::ElementwiseLeftDivide => {
                self.append_arith(block, arith::divf(r, l, self.cx.location))
            }
            OperatorKind::Equal => self.cmp_select(block, CmpfPredicate::Oeq, l, r),
            OperatorKind::NotEqual => self.cmp_select(block, CmpfPredicate::Une, l, r),
            OperatorKind::Less => self.cmp_select(block, CmpfPredicate::Olt, l, r),
            OperatorKind::LessEqual => self.cmp_select(block, CmpfPredicate::Ole, l, r),
            OperatorKind::Greater => self.cmp_select(block, CmpfPredicate::Ogt, l, r),
            OperatorKind::GreaterEqual => self.cmp_select(block, CmpfPredicate::Oge, l, r),
            OperatorKind::ElementwiseAnd => self.logical_and(block, l, r),
            OperatorKind::ElementwiseOr => self.logical_or(block, l, r),
            other => Err(Error::NotLowerable(format!("binary operator {other:?}"))),
        }
    }

    fn apply_short_circuit<'b>(
        &self,
        block: &'b Block<'c>,
        op: &MirShortCircuitOp,
        l: Value<'c, 'b>,
        r: Value<'c, 'b>,
    ) -> Result<Value<'c, 'b>> {
        match op {
            MirShortCircuitOp::And => self.logical_and(block, l, r),
            MirShortCircuitOp::Or => self.logical_or(block, l, r),
        }
    }

    fn cmp_select<'b>(
        &self,
        block: &'b Block<'c>,
        predicate: melior::dialect::arith::CmpfPredicate,
        l: Value<'c, 'b>,
        r: Value<'c, 'b>,
    ) -> Result<Value<'c, 'b>> {
        let cmp = self.cmpf(block, predicate, l, r)?;
        let one = self.float_constant(block, 1.0)?;
        let zero = self.float_constant(block, 0.0)?;
        self.select(block, cmp, one, zero)
    }

    fn logical_and<'b>(
        &self,
        block: &'b Block<'c>,
        l: Value<'c, 'b>,
        r: Value<'c, 'b>,
    ) -> Result<Value<'c, 'b>> {
        let a = self.truthy(block, l)?;
        let b = self.truthy(block, r)?;
        let and = block.append_operation(arith::andi(a, b, self.cx.location));
        let and: Value = and
            .result(0)
            .map_err(|e| Error::Backend(format!("andi result: {e}")))?
            .into();
        let one = self.float_constant(block, 1.0)?;
        let zero = self.float_constant(block, 0.0)?;
        self.select(block, and, one, zero)
    }

    fn logical_or<'b>(
        &self,
        block: &'b Block<'c>,
        l: Value<'c, 'b>,
        r: Value<'c, 'b>,
    ) -> Result<Value<'c, 'b>> {
        let a = self.truthy(block, l)?;
        let b = self.truthy(block, r)?;
        let or = block.append_operation(arith::ori(a, b, self.cx.location));
        let or: Value = or
            .result(0)
            .map_err(|e| Error::Backend(format!("ori result: {e}")))?
            .into();
        let one = self.float_constant(block, 1.0)?;
        let zero = self.float_constant(block, 0.0)?;
        self.select(block, or, one, zero)
    }

    fn cmpf<'b, 'l, 'r>(
        &self,
        block: &'b Block<'c>,
        predicate: melior::dialect::arith::CmpfPredicate,
        l: Value<'c, 'l>,
        r: Value<'c, 'r>,
    ) -> Result<Value<'c, 'b>> {
        let op = block.append_operation(arith::cmpf(
            self.cx.context,
            predicate,
            l,
            r,
            self.cx.location,
        ));
        Ok(op
            .result(0)
            .map_err(|e| Error::Backend(format!("cmpf result: {e}")))?
            .into())
    }

    fn truthy<'b>(&self, block: &'b Block<'c>, value: Value<'c, '_>) -> Result<Value<'c, 'b>> {
        let zero = self.float_constant(block, 0.0)?;
        self.cmpf(
            block,
            melior::dialect::arith::CmpfPredicate::Une,
            value,
            zero,
        )
    }

    fn select<'b, 'l, 'r>(
        &self,
        block: &'b Block<'c>,
        condition: Value<'c, '_>,
        true_value: Value<'c, 'l>,
        false_value: Value<'c, 'r>,
    ) -> Result<Value<'c, 'b>> {
        let op = block.append_operation(arith::select(
            condition,
            true_value,
            false_value,
            self.cx.location,
        ));
        Ok(op
            .result(0)
            .map_err(|e| Error::Backend(format!("select result: {e}")))?
            .into())
    }

    /// Append an `arith` operation and return its first result.
    fn append_arith<'b>(
        &self,
        block: &'b Block<'c>,
        operation: melior::ir::Operation<'c>,
    ) -> Result<Value<'c, 'b>> {
        block
            .append_operation(operation)
            .result(0)
            .map_err(|e| Error::Backend(format!("arith result: {e}")))
            .map(Into::into)
    }

    fn float_constant<'b>(&self, block: &'b Block<'c>, value: f64) -> Result<Value<'c, 'b>> {
        let attribute = FloatAttribute::new(self.cx.context, self.cx.float, value);
        let op = block.append_operation(arith::constant(
            self.cx.context,
            attribute.into(),
            self.cx.location,
        ));
        Ok(op
            .result(0)
            .map_err(|e| Error::Backend(format!("constant result: {e}")))?
            .into())
    }

    fn number_constant<'b>(&self, block: &'b Block<'c>, text: &str) -> Result<Value<'c, 'b>> {
        let value = parse_number(text)
            .ok_or_else(|| Error::NotLowerable(format!("unsupported number literal {text:?}")))?;
        self.float_constant(block, value)
    }

    /// Emit an `if`/`else` construct.
    fn lower_if<'b>(
        &self,
        block: &'b Block<'c>,
        cond: &MirOperand,
        then: usize,
        els: usize,
        merge: Option<usize>,
    ) -> Result<()> {
        let cond = self.lower_operand(block, cond)?;
        let cond = self.truthy(block, cond)?;

        let then_region = self.body_region(then, merge)?;
        let else_region = self.body_region(els, merge)?;

        block.append_operation(scf::r#if(
            cond,
            &[],
            then_region,
            else_region,
            self.cx.location,
        ));
        Ok(())
    }

    /// Emit a `while` loop. `header` is the loop header block id.
    fn lower_while<'b>(
        &self,
        block: &'b Block<'c>,
        header_statements: &[MirStmt],
        cond: &MirOperand,
        body: usize,
        _exit: usize,
        header: usize,
    ) -> Result<()> {
        let before = Block::new(&[]);
        for stmt in header_statements {
            self.lower_stmt(&before, stmt)?;
        }
        let cond = self.lower_operand(&before, cond)?;
        let cond = self.truthy(&before, cond)?;
        before.append_operation(scf::condition(cond, &[], self.cx.location));
        let before_region = Region::new();
        before_region.append_block(before);

        let after = Block::new(&[]);
        self.lower_region(&after, body, Some(header))?;
        after.append_operation(scf::r#yield(&[], self.cx.location));
        let after_region = Region::new();
        after_region.append_block(after);

        block.append_operation(scf::r#while(
            &[],
            &[],
            before_region,
            after_region,
            self.cx.location,
        ));
        Ok(())
    }

    /// Emit a `for` loop from a colon range. `header` is the loop header id.
    fn lower_for<'b>(
        &self,
        block: &'b Block<'c>,
        binding: &MirLocalId,
        iterable: &MirRvalue,
        body: usize,
        _exit: usize,
        header: usize,
    ) -> Result<()> {
        use melior::dialect::arith::CmpfPredicate;

        let (start, step, end) = match iterable {
            MirRvalue::Range { start, step, end } => (start, step.as_ref(), end),
            other => {
                return Err(Error::NotLowerable(format!(
                    "unsupported for-loop iterable {other:?}"
                )))
            }
        };

        let start = self.lower_operand(block, start)?;
        let step = match step {
            Some(step) => self.lower_operand(block, step)?,
            None => self.float_constant(block, 1.0)?,
        };
        let end = self.lower_operand(block, end)?;

        let cell = self
            .locals
            .get(&binding.0)
            .copied()
            .ok_or_else(|| Error::Backend(format!("no cell for loop binding {binding:?}")))?;
        let zero = index_zero(self.cx, block)?;
        block.append_operation(memref::store(start, cell, &[zero], self.cx.location));

        // Before region: `i <= end` when ascending, `i >= end` when descending.
        let before = Block::new(&[]);
        let i = load_local(self.cx, &before, cell)?;
        let zero = self.float_constant(&before, 0.0)?;
        let ascending = self.cmpf(&before, CmpfPredicate::Oge, step, zero)?;
        let le = self.cmpf(&before, CmpfPredicate::Ole, i, end)?;
        let ge = self.cmpf(&before, CmpfPredicate::Oge, i, end)?;
        let cond = self.select(&before, ascending, le, ge)?;
        before.append_operation(scf::condition(cond, &[], self.cx.location));
        let before_region = Region::new();
        before_region.append_block(before);

        // After region: body, then `i = i + step`.
        let after = Block::new(&[]);
        self.lower_region(&after, body, Some(header))?;
        let i = load_local(self.cx, &after, cell)?;
        let next = {
            let op = after.append_operation(arith::addf(i, step, self.cx.location));
            op.result(0)
                .map_err(|e| Error::Backend(format!("addf result: {e}")))?
                .into()
        };
        let zero = index_zero(self.cx, &after)?;
        after.append_operation(memref::store(next, cell, &[zero], self.cx.location));
        after.append_operation(scf::r#yield(&[], self.cx.location));
        let after_region = Region::new();
        after_region.append_block(after);

        block.append_operation(scf::r#while(
            &[],
            &[],
            before_region,
            after_region,
            self.cx.location,
        ));
        Ok(())
    }

    /// Emit a `switch`/`case`/`otherwise` construct as a nested `if`/`else`
    /// chain (MATLAB switch cases are exclusive and do not fall through).
    fn lower_switch<'b>(
        &self,
        block: &'b Block<'c>,
        discr: &MirOperand,
        cases: &[(MirOperand, BasicBlockId)],
        otherwise: usize,
        merge: Option<usize>,
    ) -> Result<()> {
        let discr = self.lower_operand(block, discr)?;

        let mut case_values = Vec::with_capacity(cases.len());
        for (operand, _) in cases {
            case_values.push(self.lower_operand(block, operand)?);
        }

        self.emit_switch(block, discr, &case_values, cases, otherwise, merge)
    }

    /// Recursively emit the nested `scf.if` chain for a switch into `block`.
    fn emit_switch<'b>(
        &self,
        block: &'b Block<'c>,
        discr: Value<'c, '_>,
        case_values: &[Value<'c, 'b>],
        cases: &[(MirOperand, BasicBlockId)],
        otherwise: usize,
        merge: Option<usize>,
    ) -> Result<()> {
        match case_values.split_first() {
            None => {
                // Innermost `else`: the `otherwise` body, emitted inline into
                // the block that holds this chain.
                self.lower_region(block, otherwise, merge)?;
                Ok(())
            }
            Some((first, rest)) => {
                let first_block = cases[0].1 .0;
                let cmp = self.cmpf(
                    block,
                    melior::dialect::arith::CmpfPredicate::Oeq,
                    discr,
                    *first,
                )?;
                let then_region = self.body_region(first_block, merge)?;
                let else_region =
                    self.switch_else_region(discr, rest, &cases[1..], otherwise, merge)?;
                block.append_operation(scf::r#if(
                    cmp,
                    &[],
                    then_region,
                    else_region,
                    self.cx.location,
                ));
                Ok(())
            }
        }
    }

    fn switch_else_region<'b>(
        &self,
        discr: Value<'c, '_>,
        case_values: &[Value<'c, 'b>],
        cases: &[(MirOperand, BasicBlockId)],
        otherwise: usize,
        merge: Option<usize>,
    ) -> Result<Region<'c>> {
        let block = Block::new(&[]);
        self.emit_switch(&block, discr, case_values, cases, otherwise, merge)?;
        block.append_operation(scf::r#yield(&[], self.cx.location));
        let region = Region::new();
        region.append_block(block);
        Ok(region)
    }

    /// Build a single-block `scf` region for a MIR block range.
    fn body_region(&self, start: usize, stop: Option<usize>) -> Result<Region<'c>> {
        let block = Block::new(&[]);
        self.lower_region(&block, start, stop)?;
        block.append_operation(scf::r#yield(&[], self.cx.location));
        let region = Region::new();
        region.append_block(block);
        Ok(region)
    }

    /// Compute the block a structured region starting at `start` exits to.
    fn exit_of(&self, start: usize) -> Option<usize> {
        let mut cur = start;
        loop {
            let block = self.body.blocks.get(cur)?;
            match &block.terminator.kind {
                MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => return None,
                MirTerminatorKind::Goto(target) => {
                    if self.is_loop_header(target.0) {
                        cur = target.0;
                    } else {
                        return Some(target.0);
                    }
                }
                MirTerminatorKind::Branch {
                    then_block,
                    else_block,
                    ..
                } => {
                    if self.is_loop_header(cur) {
                        cur = else_block.0;
                    } else {
                        return self.exit_of(then_block.0);
                    }
                }
                MirTerminatorKind::For { exit_block, .. } => cur = exit_block.0,
                MirTerminatorKind::Switch { otherwise, .. } => return self.exit_of(otherwise.0),
                _ => return None,
            }
        }
    }

    /// Whether `block` is a loop header, i.e. it has an incoming backedge from
    /// a block it dominates.
    fn is_loop_header(&self, block: usize) -> bool {
        self.preds
            .get(block)
            .map(|preds| {
                preds
                    .iter()
                    .any(|pred| self.dominators[*pred].contains(&block))
            })
            .unwrap_or(false)
    }
}

/// Emit an `arith.constant 0 : index` into `block`.
fn index_zero<'c, 'b>(cx: &Cx<'c>, block: &'b Block<'c>) -> Result<Value<'c, 'b>> {
    let attribute = IntegerAttribute::new(cx.index, 0);
    let op = block.append_operation(arith::constant(cx.context, attribute.into(), cx.location));
    Ok(op
        .result(0)
        .map_err(|e| Error::Backend(format!("index constant result: {e}")))?
        .into())
}

/// Load the scalar stored in `cell` from `block`.
fn load_local<'c, 'b>(
    cx: &Cx<'c>,
    block: &'b Block<'c>,
    cell: Value<'c, '_>,
) -> Result<Value<'c, 'b>> {
    let zero = index_zero(cx, block)?;
    let load = block.append_operation(memref::load(cell, &[zero], cx.location));
    Ok(load
        .result(0)
        .map_err(|e| Error::Backend(format!("memref.load result: {e}")))?
        .into())
}

/// The CFG-derived facts needed to recover structured control flow.
struct Cfg {
    preds: Vec<Vec<usize>>,
    dominators: Vec<HashSet<usize>>,
}

/// Compute the predecessor list and dominator sets for every MIR block
/// (indexed by position, which equals `id.0`).
fn compute_cfg(body: &MirBody) -> Cfg {
    let n = body.blocks.len();
    let mut succs = vec![Vec::new(); n];
    for (index, block) in body.blocks.iter().enumerate() {
        let mut succ = Vec::new();
        match &block.terminator.kind {
            MirTerminatorKind::Goto(target) => succ.push(target.0),
            MirTerminatorKind::Branch {
                then_block,
                else_block,
                ..
            } => {
                succ.push(then_block.0);
                succ.push(else_block.0);
            }
            MirTerminatorKind::Switch {
                cases, otherwise, ..
            } => {
                for (_, target) in cases {
                    succ.push(target.0);
                }
                succ.push(otherwise.0);
            }
            MirTerminatorKind::For {
                body_block,
                exit_block,
                ..
            } => {
                succ.push(body_block.0);
                succ.push(exit_block.0);
            }
            _ => {}
        }
        succs[index] = succ;
    }

    let mut preds = vec![Vec::new(); n];
    for (index, succ) in succs.iter().enumerate() {
        for &target in succ {
            preds[target].push(index);
        }
    }

    let dominators = compute_dominators(&preds);
    Cfg { preds, dominators }
}

/// Compute the dominator set for every block using the iterative dataflow
/// algorithm (entry block is `0`).
fn compute_dominators(preds: &[Vec<usize>]) -> Vec<HashSet<usize>> {
    let n = preds.len();
    let all: HashSet<usize> = (0..n).collect();
    let mut dom = vec![all.clone(); n];
    dom[0] = HashSet::from([0]);

    loop {
        let mut changed = false;
        for block in 0..n {
            if block == 0 {
                continue;
            }
            let mut intersection: Option<HashSet<usize>> = None;
            for &pred in &preds[block] {
                intersection = Some(match intersection {
                    None => dom[pred].clone(),
                    Some(set) => set.intersection(&dom[pred]).copied().collect(),
                });
            }
            let mut new = intersection.unwrap_or_default();
            new.insert(block);
            if new != dom[block] {
                dom[block] = new;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    dom
}

/// Parse a MATLAB numeric literal into an `f64`, including special values.
fn parse_number(text: &str) -> Option<f64> {
    text.trim()
        .parse::<f64>()
        .ok()
        .or_else(|| match text.trim().to_ascii_lowercase().as_str() {
            "inf" | "infinity" => Some(f64::INFINITY),
            "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
            "nan" => Some(f64::NAN),
            _ => None,
        })
}
