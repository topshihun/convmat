//! Layer 3: lower MIR to the `matlab` pliron dialect.
//!
//! A [`runmat_mir::MirAssembly`] is traversed and lowered to the [`matlab`]
//! dialect (dense column-major arrays, scalar arithmetic, comparisons, `libm`
//! calls, and structured control flow) hosted in a builtin `module`/`func`.
//! The result is handed to [`crate::lowering`] for the `emitc` conversion.
//!
//! The lowering is deliberately straightforward (no optimization). Every MIR
//! local becomes a stack array cell (a `1`-element array for scalars, an
//! `N`-element array otherwise); assignments are `store`s and reads are
//! `load`s. Structured control flow is recovered from the MIR CFG and emitted
//! as `if`/`while`/`for`, which sidesteps explicit SSA phi construction while
//! remaining faithful to the source semantics.
//!
//! The recursive control-flow helpers thread the `Context` alongside the MIR
//! block ranges, so several of them naturally exceed Clippy's argument-count
//! heuristic.
#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet};

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::{OneRegionInterface, OneResultInterface, SingleBlockRegionInterface},
        ops::{FuncOp, ModuleOp},
        types::{FP64Type, FunctionType},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        inserter::{IRInserter, Inserter},
        listener::DummyListener,
    },
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    region::Region,
    value::Value,
};

use crate::builtins::{self, Builtin, MinMax, ReduceOp};
use crate::dialects::matlab::{
    AllocaOp, ArrayType, BinOp, BinOpKind, CallOp, CallVoidOp, CmpKind, CmpOp, ConditionOp,
    ConstantOp, ForOp, IfOp, LoadOp, ReturnOp, SelectOp, StoreOp, WhileOp, YieldOp,
};
use crate::error::{Error, Result};
use crate::triage::{
    brace_index, call_name, infer_locals, varargout_index, LocalTy, Shape, Variadics,
};

use runmat_hir::{IndexKind, OperatorKind};
use runmat_mir::{
    BasicBlockId, MirAssembly, MirBody, MirCall, MirCallArg, MirConstant, MirIndexComponent,
    MirIndexing, MirLocalId, MirOperand, MirPlace, MirRvalue, MirShortCircuitOp, MirStmt,
    MirStmtKind, MirTerminatorKind,
};

type OpInserter = IRInserter<DummyListener>;

/// Lower MIR to a readable dump of the generated `matlab`-dialect IR.
pub fn lower(mir: &MirAssembly) -> Result<String> {
    let mut context = Context::new();
    let module = lower_to_module(&mut context, mir)?;
    Ok(dump_module(&context, module))
}

/// Lower MIR into a new `matlab`-dialect module owned by `context`.
pub fn lower_to_module(context: &mut Context, mir: &MirAssembly) -> Result<ModuleOp> {
    let f64_ty: TypeHandle = FP64Type::get(context).into();
    let module = ModuleOp::new(context, Identifier::try_from("convmat").unwrap());

    for (function_id, body) in &mir.bodies {
        lower_function(context, mir, *function_id, body, &module, f64_ty)?;
    }

    Ok(module)
}

/// Lower a single MIR body into a `builtin.func` appended to `module`.
fn lower_function(
    context: &mut Context,
    mir: &MirAssembly,
    function_id: runmat_hir::FunctionId,
    body: &MirBody,
    module: &ModuleOp,
    f64_ty: TypeHandle,
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

    let tys = infer_locals(body);
    let variadics = Variadics::compute(body);

    // Named (non-variadic) scalar parameters: `fixed_inputs` minus `varargin`.
    let varargin_binding = body.abi.varargin;
    let named_params: Vec<MirLocalId> = body
        .abi
        .fixed_inputs
        .iter()
        .filter(|binding| Some(**binding) != varargin_binding)
        .map(|binding| {
            binding_to_local
                .get(binding)
                .copied()
                .ok_or_else(|| Error::Backend(format!("no local for input {binding:?}")))
        })
        .collect::<Result<_>>()?;

    // Named (non-variadic) outputs: `fixed_outputs` minus `varargout`.
    let varargout_binding = body.abi.varargout;
    let named_outputs: Vec<MirLocalId> = body
        .abi
        .fixed_outputs
        .iter()
        .filter(|binding| Some(**binding) != varargout_binding)
        .map(|binding| {
            binding_to_local
                .get(binding)
                .copied()
                .ok_or_else(|| Error::Backend(format!("no local for output {binding:?}")))
        })
        .collect::<Result<_>>()?;

    // Split named outputs by ABI: scalars are returned; arrays become out-pointer
    // parameters supplied (and allocated) by the caller. `varargout` elements are
    // specialized as extra scalar outputs.
    let mut scalar_outputs = Vec::new();
    let mut array_outputs = Vec::new();
    for output in &named_outputs {
        match tys.get(&output.0).copied().unwrap_or(LocalTy::Scalar) {
            LocalTy::Scalar => scalar_outputs.push(*output),
            LocalTy::Array { .. } => array_outputs.push(*output),
            LocalTy::Dynamic => {
                return Err(Error::NotLowerable("dynamic output shape".to_string()))
            }
        }
    }

    let total_inputs = named_params.len() + variadics.varargin_count;
    let total_scalar_outputs = scalar_outputs.len() + variadics.varargout_count;

    // Entry block argument types: scalar params first, then array out-params.
    let scalar_cell_ty: TypeHandle = ArrayType::get(context, vec![1]).into();
    let mut entry_arg_types = vec![f64_ty; total_inputs];
    for output in &array_outputs {
        let LocalTy::Array { shape } = tys[&output.0] else {
            unreachable!("array output must have an array type");
        };
        entry_arg_types.push(ArrayType::get(context, vec![static_numel(shape)? as i64]).into());
    }
    let output_types = vec![f64_ty; total_scalar_outputs];
    let fn_ty = FunctionType::get(context, entry_arg_types.clone(), output_types);
    let name_id = Identifier::try_from(name.as_str())
        .map_err(|e| Error::Backend(format!("bad function name `{name}`: {e}")))?;
    let func = FuncOp::new(context, name_id, fn_ty);
    let entry = func.get_entry_block(context);

    // Allocate a stack cell per local. Array outputs are caller-provided and are
    // wired to their incoming buffers below; the `varargin`/`varargout` cell
    // locals are specialized away (no cell is materialized).
    let mut skip_ids: HashSet<usize> = array_outputs.iter().map(|output| output.0).collect();
    if let Some(local) = variadics.varargin_local {
        skip_ids.insert(local);
    }
    if let Some(local) = variadics.varargout_local {
        skip_ids.insert(local);
    }
    let mut locals: HashMap<usize, Value> = HashMap::new();
    for local in &body.locals {
        if skip_ids.contains(&local.id.0) {
            continue;
        }
        let array_ty = match tys.get(&local.id.0).copied().unwrap_or(LocalTy::Scalar) {
            LocalTy::Scalar => scalar_cell_ty,
            LocalTy::Array { shape } => {
                ArrayType::get(context, vec![static_numel(shape)? as i64]).into()
            }
            LocalTy::Dynamic => return Err(Error::NotLowerable("dynamic local shape".to_string())),
        };
        let alloca = AllocaOp::new(context, array_ty);
        let value = alloca.get_result(context);
        append(context, entry, &alloca);
        locals.insert(local.id.0, value);
    }

    // Store incoming named scalar parameters into their cells.
    for (index, param) in named_params.iter().enumerate() {
        let argument = entry.deref(context).get_argument(index);
        let target = locals
            .get(&param.0)
            .copied()
            .ok_or_else(|| Error::Backend(format!("no cell for parameter {param:?}")))?;
        let zero = emit_constant(context, entry, 0.0)?;
        emit_store(context, entry, target, zero, argument);
    }

    // `varargin{k}` resolves directly to the (k-1)-th extra scalar argument.
    let mut varargin_args = Vec::with_capacity(variadics.varargin_count);
    for k in 0..variadics.varargin_count {
        varargin_args.push(entry.deref(context).get_argument(named_params.len() + k));
    }

    // `nargin`/`nargout` fold to compile-time constants (fixed arity + variadic).
    if let Some(nargin_local) = variadics.nargin_local {
        let value = (variadics.named_inputs + variadics.varargin_count) as f64;
        store_constant(context, entry, locals.get(&nargin_local).copied(), value)?;
    }
    if let Some(nargout_local) = variadics.nargout_local {
        let value = (variadics.named_outputs + variadics.varargout_count) as f64;
        store_constant(context, entry, locals.get(&nargout_local).copied(), value)?;
    }

    // `varargout{k}` writes target a dedicated scalar cell per element.
    let mut varargout_cells = Vec::with_capacity(variadics.varargout_count);
    for _ in 0..variadics.varargout_count {
        let alloca = AllocaOp::new(context, scalar_cell_ty);
        let value = alloca.get_result(context);
        append(context, entry, &alloca);
        varargout_cells.push(value);
    }

    // Wire array outputs to their incoming caller-provided buffers.
    for (offset, output) in array_outputs.iter().enumerate() {
        let argument = entry.deref(context).get_argument(total_inputs + offset);
        locals.insert(output.0, argument);
    }

    let cfg = compute_cfg(body);
    let lowerer = FuncLowerer {
        body,
        f64_ty,
        locals,
        tys,
        preds: cfg.preds,
        dominators: cfg.dominators,
        varargin_local: variadics.varargin_local,
        varargin_args,
        varargout_local: variadics.varargout_local,
        varargout_cells,
    };
    lowerer.lower_region(context, entry, 0, None)?;

    module.append_operation(context, func.get_operation(), 0);
    Ok(())
}

/// Append an op to the end of `block` using a throwaway inserter.
fn append(context: &Context, block: Ptr<BasicBlock>, op: &dyn Op) {
    OpInserter::new_at_block_end(block).append_op(context, op);
}

/// Build and append a `matlab.store` into `block`.
fn emit_store(
    context: &mut Context,
    block: Ptr<BasicBlock>,
    array: Value,
    index: Value,
    value: Value,
) {
    let op = StoreOp::new(context, array, index, value);
    append(context, block, &op);
}

/// Build and append a `matlab.return` into `block`.
fn emit_return(context: &mut Context, block: Ptr<BasicBlock>, values: Vec<Value>) {
    let op = ReturnOp::new(context, values);
    append(context, block, &op);
}

/// Build and append a `matlab.condition` into `block`.
fn emit_condition(context: &mut Context, block: Ptr<BasicBlock>, cond: Value) {
    let op = ConditionOp::new(context, cond);
    append(context, block, &op);
}

/// Build and append a `matlab.yield` into `block`.
fn emit_yield(context: &mut Context, block: Ptr<BasicBlock>) {
    let op = YieldOp::new(context);
    append(context, block, &op);
}

/// Emit a `matlab.constant` `f64` into `block`.
fn emit_constant(context: &mut Context, block: Ptr<BasicBlock>, value: f64) -> Result<Value> {
    let op = ConstantOp::new(context, value);
    let result = op.get_result(context);
    append(context, block, &op);
    Ok(result)
}

/// Store a compile-time constant `f64` into a scalar cell (at index 0).
/// A `None` cell is a no-op (the binding is unused).
fn store_constant(
    context: &mut Context,
    block: Ptr<BasicBlock>,
    cell: Option<Value>,
    value: f64,
) -> Result<()> {
    let Some(cell) = cell else {
        return Ok(());
    };
    let value = emit_constant(context, block, value)?;
    let zero = emit_constant(context, block, 0.0)?;
    emit_store(context, block, cell, zero, value);
    Ok(())
}

/// Per-function lowering state (all read-only during lowering; `context` is
/// threaded separately so the MIR borrow and the IR construction never alias).
struct FuncLowerer<'b> {
    body: &'b MirBody,
    f64_ty: TypeHandle,
    locals: HashMap<usize, Value>,
    tys: HashMap<usize, LocalTy>,
    preds: Vec<Vec<usize>>,
    dominators: Vec<HashSet<usize>>,
    /// The MIR local backing `varargin` (if any), specialized away.
    varargin_local: Option<usize>,
    /// The entry-block argument `Value` for each `varargin{k}` (0-based).
    varargin_args: Vec<Value>,
    /// The MIR local backing `varargout` (if any), specialized away.
    varargout_local: Option<usize>,
    /// The scalar cell for each `varargout{k}` (0-based).
    varargout_cells: Vec<Value>,
}

/// A reduction to apply across an array.
#[derive(Clone, Copy)]
enum Reducer {
    Add,
    Mul,
    Min,
    Max,
}

impl Reducer {
    fn from_reduce(op: ReduceOp) -> Self {
        match op {
            ReduceOp::Sum => Reducer::Add,
            ReduceOp::Prod => Reducer::Mul,
        }
    }

    fn from_minmax(minmax: MinMax) -> Self {
        match minmax {
            MinMax::Min => Reducer::Min,
            MinMax::Max => Reducer::Max,
        }
    }

    /// The accumulator seed for this reduction.
    fn init(self) -> f64 {
        match self {
            Reducer::Add => 0.0,
            Reducer::Mul => 1.0,
            Reducer::Min => f64::INFINITY,
            Reducer::Max => f64::NEG_INFINITY,
        }
    }
}

impl<'b> FuncLowerer<'b> {
    /// Lower a structured region starting at `start` until control flows to
    /// `stop` (exclusive) or the function returns.
    fn lower_region(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        start: usize,
        stop: Option<usize>,
    ) -> Result<Option<usize>> {
        let mut cur = start;

        loop {
            let (stmts, term) = {
                let mir_block = self
                    .body
                    .blocks
                    .get(cur)
                    .ok_or_else(|| Error::Backend(format!("block {cur} out of range")))?;
                (
                    mir_block.statements.clone(),
                    mir_block.terminator.kind.clone(),
                )
            };

            // Loop headers first: their condition statements must be emitted in
            // the loop's "before" region so they re-run on every iteration.
            match &term {
                MirTerminatorKind::Branch {
                    cond,
                    then_block,
                    else_block,
                } if self.is_loop_header(cur) => {
                    self.lower_while(
                        context,
                        block,
                        &stmts,
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
                    self.lower_for(
                        context,
                        block,
                        binding,
                        iterable,
                        body_block.0,
                        exit_block.0,
                        cur,
                    )?;
                    cur = exit_block.0;
                    continue;
                }
                _ => {}
            }

            for stmt in &stmts {
                self.lower_stmt(context, block, stmt)?;
            }

            match &term {
                MirTerminatorKind::Return(operands) => {
                    if stop.is_some() {
                        return Err(Error::NotLowerable(
                            "`return` inside control flow is not supported yet".to_string(),
                        ));
                    }
                    let mut values = Vec::new();
                    for operand in operands {
                        // A `varargout` return expands to its specialized scalar
                        // outputs, in order.
                        if let MirOperand::Local(id) = operand {
                            if self.varargout_local == Some(id.0) {
                                for cell in &self.varargout_cells {
                                    values.push(self.load_local(context, block, *cell)?);
                                }
                                continue;
                            }
                        }
                        match self.operand_ty(operand) {
                            LocalTy::Scalar => {
                                values.push(self.lower_operand(context, block, operand)?);
                            }
                            // Array outputs are already written into their
                            // caller-provided out-parameter buffers.
                            LocalTy::Array { .. } => {}
                            LocalTy::Dynamic => {
                                return Err(Error::NotLowerable(
                                    "dynamic return value".to_string(),
                                ));
                            }
                        }
                    }
                    emit_return(context, block, values);
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
                    self.lower_if(context, block, cond, then_block.0, else_block.0, merge)?;
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
                    self.lower_switch(context, block, discr, cases, otherwise.0, merge)?;
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
    fn lower_stmt(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        stmt: &MirStmt,
    ) -> Result<()> {
        match &stmt.kind {
            MirStmtKind::Assign { place, value } => {
                // `varargout{k} = expr` writes the k-th extra scalar output cell.
                if let Some(k) = varargout_index(place, self.varargout_local) {
                    let value = self.lower_rvalue(context, block, value)?;
                    let cell = self.varargout_cells.get(k - 1).copied().ok_or_else(|| {
                        Error::NotLowerable(format!("varargout{{{k}}} is out of range"))
                    })?;
                    let zero = emit_constant(context, block, 0.0)?;
                    emit_store(context, block, cell, zero, value);
                    return Ok(());
                }
                let MirPlace::Local(target) = place else {
                    return Err(Error::NotLowerable(format!(
                        "non-local assignment target {place:?}"
                    )));
                };
                let cell = self
                    .locals
                    .get(&target.0)
                    .copied()
                    .ok_or_else(|| Error::Backend(format!("no cell for local {:?}", target)))?;
                match self.tys.get(&target.0).copied().unwrap_or(LocalTy::Scalar) {
                    LocalTy::Scalar => {
                        let value = self.lower_rvalue(context, block, value)?;
                        let zero = emit_constant(context, block, 0.0)?;
                        emit_store(context, block, cell, zero, value);
                    }
                    LocalTy::Array { .. } => {
                        self.lower_array_rvalue_into(context, block, value, cell)?;
                    }
                    LocalTy::Dynamic => {
                        return Err(Error::NotLowerable("dynamic assignment target".to_string()));
                    }
                }
                Ok(())
            }
            MirStmtKind::Expr(value) => {
                // Evaluate for (potential) side effects; the result is dropped.
                self.lower_rvalue(context, block, value)?;
                Ok(())
            }
            // `varargout{k}` array-creation place mutations are folded away.
            MirStmtKind::PlaceMutation(_) => Ok(()),
            other => Err(Error::NotLowerable(format!(
                "unsupported statement {other:?}"
            ))),
        }
    }

    /// Lower a rvalue to a `f64` value in the current block.
    fn lower_rvalue(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        rvalue: &MirRvalue,
    ) -> Result<Value> {
        match rvalue {
            MirRvalue::Use(operand) => self.lower_operand(context, block, operand),
            MirRvalue::Unary(op, operand) => {
                let value = self.lower_operand(context, block, operand)?;
                self.apply_unary(context, block, op, value)
            }
            MirRvalue::Binary(lhs, op, rhs) => {
                let l = self.lower_operand(context, block, lhs)?;
                let r = self.lower_operand(context, block, rhs)?;
                self.apply_binary(context, block, op, l, r)
            }
            MirRvalue::ShortCircuit {
                left,
                op,
                right_temps,
                right,
            } => {
                let l = self.lower_operand(context, block, left)?;
                for stmt in right_temps {
                    self.lower_stmt(context, block, stmt)?;
                }
                let r = self.lower_operand(context, block, right)?;
                self.apply_short_circuit(context, block, op, l, r)
            }
            MirRvalue::Call(call) => self.lower_scalar_call(context, block, call),
            MirRvalue::Index { base, indexing } => {
                self.lower_index_scalar(context, block, base, indexing)
            }
            other => Err(Error::NotLowerable(format!("rvalue {other:?}"))),
        }
    }

    /// Lower an operand to a `f64` value in the current block.
    fn lower_operand(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        operand: &MirOperand,
    ) -> Result<Value> {
        match operand {
            MirOperand::Local(id) => {
                let cell = self
                    .locals
                    .get(&id.0)
                    .copied()
                    .ok_or_else(|| Error::Backend(format!("operand {id:?} has no cell")))?;
                self.load_local(context, block, cell)
            }
            MirOperand::Constant(constant) => match constant {
                MirConstant::Number(text) => self.number_constant(context, block, text),
                MirConstant::IntegerLiteral(literal) => {
                    emit_constant(context, block, literal.bits() as f64)
                }
                MirConstant::Bool(value) => {
                    emit_constant(context, block, if *value { 1.0 } else { 0.0 })
                }
                other => Err(Error::NotLowerable(format!("constant {other:?}"))),
            },
            other => Err(Error::NotLowerable(format!("operand {other:?}"))),
        }
    }

    /// The static type of an operand, as determined by triage's shape analysis.
    fn operand_ty(&self, operand: &MirOperand) -> LocalTy {
        match operand {
            MirOperand::Local(id) => self.tys.get(&id.0).copied().unwrap_or(LocalTy::Scalar),
            MirOperand::Constant(_) => LocalTy::Scalar,
            _ => LocalTy::Dynamic,
        }
    }

    /// Lower scalar indexing `A(i, j)` / `A(i)` to a single `f64` load, or
    /// `varargin{k}` to the `k`-th extra scalar argument.
    fn lower_index_scalar(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        base: &MirOperand,
        indexing: &MirIndexing,
    ) -> Result<Value> {
        // `varargin{k}` (constant `k`) is a direct reference to an argument.
        if indexing.kind == IndexKind::Brace {
            if let MirOperand::Local(id) = base {
                if self.varargin_local == Some(id.0) {
                    let Some(k) = brace_index(indexing) else {
                        return Err(Error::NotLowerable(
                            "varargin index must be a constant".to_string(),
                        ));
                    };
                    return self.varargin_args.get(k - 1).copied().ok_or_else(|| {
                        Error::NotLowerable(format!("varargin{{{k}}} is out of range"))
                    });
                }
            }
            return Err(Error::NotLowerable(
                "only paren indexing is supported".to_string(),
            ));
        }
        if indexing.kind != IndexKind::Paren {
            return Err(Error::NotLowerable(
                "only paren indexing is supported".to_string(),
            ));
        }
        let src = self.array_source(base)?;
        let shape = self.array_shape(base)?;
        let offset = self.static_linear_offset(indexing, shape)?;
        let index = emit_constant(context, block, offset as f64)?;
        let load = LoadOp::new(context, src, index);
        let result = load.get_result(context);
        append(context, block, &load);
        Ok(result)
    }

    /// Compile-time column-major linear offset of a scalar index expression.
    fn static_linear_offset(&self, indexing: &MirIndexing, shape: Shape) -> Result<usize> {
        // Linear indexing: a single component indexes the flattened array.
        if indexing.components.len() == 1 {
            return match &indexing.components[0] {
                MirIndexComponent::Expr(operand) => {
                    let one_based = self.constant_index(operand)?;
                    if one_based == 0 {
                        return Err(Error::NotLowerable("indices are 1-based".to_string()));
                    }
                    Ok(one_based - 1)
                }
                MirIndexComponent::End { offset, .. } => {
                    let position = shape.numel() as isize + offset;
                    if position <= 0 {
                        return Err(Error::NotLowerable("end index out of range".to_string()));
                    }
                    Ok((position - 1) as usize)
                }
                MirIndexComponent::Colon => Err(Error::NotLowerable(
                    "colon is only supported in `A(:)`".to_string(),
                )),
            };
        }

        // Subscript indexing: each component addresses one dimension.
        let mut coords = Vec::with_capacity(indexing.components.len());
        for (axis, component) in indexing.components.iter().enumerate() {
            let coord = match component {
                MirIndexComponent::Expr(operand) => {
                    let one_based = self.constant_index(operand)?;
                    if one_based == 0 {
                        return Err(Error::NotLowerable("indices are 1-based".to_string()));
                    }
                    one_based - 1
                }
                MirIndexComponent::End { dim, offset } => {
                    let dim = dim.unwrap_or(axis);
                    let size = *shape
                        .dims()
                        .get(dim)
                        .ok_or_else(|| Error::NotLowerable("end index out of range".to_string()))?;
                    let position = size as isize + offset;
                    if position <= 0 {
                        return Err(Error::NotLowerable("end index out of range".to_string()));
                    }
                    (position - 1) as usize
                }
                MirIndexComponent::Colon => {
                    return Err(Error::NotLowerable(
                        "colon slices are not supported yet".to_string(),
                    ))
                }
            };
            coords.push(coord);
        }
        Ok(shape.linear(&coords))
    }

    /// A constant integer index (1-based) from an operand.
    fn constant_index(&self, operand: &MirOperand) -> Result<usize> {
        match operand {
            MirOperand::Constant(MirConstant::Number(text)) => text
                .trim()
                .parse::<usize>()
                .map_err(|_| Error::NotLowerable(format!("bad index `{text}`"))),
            MirOperand::Constant(MirConstant::IntegerLiteral(literal)) => {
                Ok(literal.bits() as usize)
            }
            _ => Err(Error::NotLowerable(
                "variable indices are not supported yet".to_string(),
            )),
        }
    }

    /// Lower a built-in call that yields a scalar `f64`.
    fn lower_scalar_call(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        call: &MirCall,
    ) -> Result<Value> {
        let name = call_name(&call.callee)
            .ok_or_else(|| Error::NotLowerable("dynamic function call".to_string()))?;
        let builtin = builtins::lookup(&name)
            .ok_or_else(|| Error::NotLowerable(format!("unsupported builtin `{name}`")))?;

        let args: Vec<&MirOperand> = call
            .args
            .iter()
            .map(|arg| match arg {
                MirCallArg::Single(operand) => Ok(operand),
                MirCallArg::Expansion { .. } => {
                    Err(Error::NotLowerable("argument expansion".to_string()))
                }
            })
            .collect::<Result<_>>()?;

        match builtin {
            Builtin::Unary(symbol) => {
                let value = self.lower_operand(context, block, args[0])?;
                self.emit_libm_call(context, block, symbol, &[value])
            }
            Builtin::Binary(symbol) => {
                let l = self.lower_operand(context, block, args[0])?;
                let r = self.lower_operand(context, block, args[1])?;
                self.emit_libm_call(context, block, symbol, &[l, r])
            }
            Builtin::Mod => {
                let l = self.lower_operand(context, block, args[0])?;
                let r = self.lower_operand(context, block, args[1])?;
                self.mod_floor(context, block, l, r)
            }
            Builtin::Sign => {
                let value = self.lower_operand(context, block, args[0])?;
                self.sign(context, block, value)
            }
            Builtin::MinMax(minmax) => match args.len() {
                2 => {
                    let l = self.lower_operand(context, block, args[0])?;
                    let r = self.lower_operand(context, block, args[1])?;
                    let symbol = match minmax {
                        MinMax::Min => "fmin",
                        MinMax::Max => "fmax",
                    };
                    self.emit_libm_call(context, block, symbol, &[l, r])
                }
                1 => self.reduce_arg(context, block, args[0], Reducer::from_minmax(minmax)),
                _ => unreachable!("arity checked by triage"),
            },
            Builtin::Reduce(op) => {
                self.reduce_arg(context, block, args[0], Reducer::from_reduce(op))
            }
            Builtin::Numel => match self.operand_ty(args[0]) {
                LocalTy::Array { shape } => emit_constant(context, block, shape.numel() as f64),
                LocalTy::Scalar => emit_constant(context, block, 1.0),
                LocalTy::Dynamic => Err(Error::NotLowerable("dynamic numel".to_string())),
            },
            Builtin::Length => {
                let max = match self.operand_ty(args[0]) {
                    LocalTy::Array { shape } => shape.dims().iter().copied().max().unwrap_or(1),
                    LocalTy::Scalar => 1,
                    LocalTy::Dynamic => {
                        return Err(Error::NotLowerable("dynamic length".to_string()))
                    }
                };
                emit_constant(context, block, max as f64)
            }
            Builtin::Size => {
                let dim = self.dim_arg(call)?.unwrap_or(1);
                let size = match self.operand_ty(args[0]) {
                    LocalTy::Array { shape } if dim >= 1 && dim <= shape.rank() => {
                        shape.dims()[dim - 1]
                    }
                    _ => 1,
                };
                emit_constant(context, block, size as f64)
            }
            // Constructors and reshape always produce arrays; they are handled
            // by the array path and never reach the scalar path.
            Builtin::Fill(_) | Builtin::Eye | Builtin::Reshape => Err(Error::NotLowerable(
                "constructor reached scalar path".to_string(),
            )),
        }
    }

    /// The constant dimension argument of a built-in call, when present.
    fn dim_arg(&self, call: &MirCall) -> Result<Option<usize>> {
        if call.args.len() < 2 {
            return Ok(None);
        }
        match &call.args[1] {
            MirCallArg::Single(MirOperand::Constant(MirConstant::Number(text))) => text
                .trim()
                .parse::<usize>()
                .map(Some)
                .map_err(|_| Error::NotLowerable(format!("bad dimension `{text}`"))),
            _ => Err(Error::NotLowerable(
                "dimension argument must be a constant".to_string(),
            )),
        }
    }

    /// Lower a reduction over a scalar (identity) or an array (loop).
    fn reduce_arg(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        operand: &MirOperand,
        reducer: Reducer,
    ) -> Result<Value> {
        match self.operand_ty(operand) {
            LocalTy::Scalar => self.lower_operand(context, block, operand),
            LocalTy::Array { shape } => {
                let src = self.array_source(operand)?;
                self.reduce(context, block, reducer, src, shape.numel())
            }
            LocalTy::Dynamic => Err(Error::NotLowerable("dynamic reduction".to_string())),
        }
    }

    /// Lower an array-producing rvalue directly into `dest` (an array cell).
    fn lower_array_rvalue_into(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        rvalue: &MirRvalue,
        dest: Value,
    ) -> Result<()> {
        match rvalue {
            MirRvalue::Aggregate {
                kind,
                rows,
                cols,
                elements,
            } => {
                if !matches!(kind, runmat_mir::MirAggregateKind::Tensor) {
                    return Err(Error::NotLowerable("cell array literal".to_string()));
                }
                // runmat flattens literal elements row-major; our storage is
                // MATLAB column-major, so transpose the index when writing.
                let shape = Shape::matrix(*rows, *cols);
                for row in 0..*rows {
                    for col in 0..*cols {
                        let element = &elements[row * cols + col];
                        let value = self.lower_operand(context, block, element)?;
                        let offset = shape.linear(&[row, col]);
                        let index = emit_constant(context, block, offset as f64)?;
                        emit_store(context, block, dest, index, value);
                    }
                }
                Ok(())
            }
            MirRvalue::Call(call) => {
                let name = call_name(&call.callee)
                    .ok_or_else(|| Error::NotLowerable("dynamic function call".to_string()))?;
                let builtin = builtins::lookup(&name)
                    .ok_or_else(|| Error::NotLowerable(format!("unsupported builtin `{name}`")))?;
                match builtin {
                    Builtin::Unary(symbol) => {
                        let [MirCallArg::Single(arg)] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "expected a single array argument".to_string(),
                            ));
                        };
                        let src = self.array_source(arg)?;
                        let n = self.array_len(arg)?;
                        self.map_unary(context, block, symbol, src, dest, n)
                    }
                    Builtin::Reduce(op) => {
                        let [MirCallArg::Single(arg), _] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "expected array + dimension".to_string(),
                            ));
                        };
                        let dim = self.dim_arg(call)?.unwrap_or(1);
                        let src = self.array_source(arg)?;
                        let shape = self.array_shape(arg)?;
                        self.reduce_axis(
                            context,
                            block,
                            Reducer::from_reduce(op),
                            src,
                            shape,
                            dim,
                            dest,
                        )
                    }
                    Builtin::MinMax(minmax) => {
                        let [MirCallArg::Single(arg), _] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "expected array + dimension".to_string(),
                            ));
                        };
                        let dim = self.dim_arg(call)?.unwrap_or(1);
                        let src = self.array_source(arg)?;
                        let shape = self.array_shape(arg)?;
                        self.reduce_axis(
                            context,
                            block,
                            Reducer::from_minmax(minmax),
                            src,
                            shape,
                            dim,
                            dest,
                        )
                    }
                    Builtin::Size => {
                        let [MirCallArg::Single(arg)] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "expected a single array argument".to_string(),
                            ));
                        };
                        let shape = match self.operand_ty(arg) {
                            LocalTy::Array { shape } => shape,
                            LocalTy::Scalar => Shape::matrix(1, 1),
                            LocalTy::Dynamic => {
                                return Err(Error::NotLowerable("dynamic size".to_string()))
                            }
                        };
                        let rows = emit_constant(context, block, shape.dims()[0] as f64)?;
                        let cols = emit_constant(context, block, shape.dims()[1] as f64)?;
                        let zero = emit_constant(context, block, 0.0)?;
                        let one = emit_constant(context, block, 1.0)?;
                        emit_store(context, block, dest, zero, rows);
                        emit_store(context, block, dest, one, cols);
                        Ok(())
                    }
                    Builtin::Fill(value) => {
                        let (rows, cols) = self.constructor_dims(call)?;
                        let fill = emit_constant(context, block, value)?;
                        for offset in 0..(rows * cols) {
                            let index = emit_constant(context, block, offset as f64)?;
                            emit_store(context, block, dest, index, fill);
                        }
                        Ok(())
                    }
                    Builtin::Eye => {
                        let (rows, cols) = self.constructor_dims(call)?;
                        let shape = Shape::matrix(rows, cols);
                        let zero = emit_constant(context, block, 0.0)?;
                        let one = emit_constant(context, block, 1.0)?;
                        for row in 0..rows {
                            for col in 0..cols {
                                let value = if row == col { one } else { zero };
                                let index = shape.linear(&[row, col]);
                                let idx = emit_constant(context, block, index as f64)?;
                                emit_store(context, block, dest, idx, value);
                            }
                        }
                        Ok(())
                    }
                    Builtin::Reshape => {
                        let [MirCallArg::Single(arg), ..] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "reshape expects an array argument".to_string(),
                            ));
                        };
                        let src = self.array_source(arg)?;
                        let n = self.array_len(arg)?;
                        for offset in 0..n {
                            let idx = emit_constant(context, block, offset as f64)?;
                            let load = LoadOp::new(context, src, idx);
                            let value = load.get_result(context);
                            append(context, block, &load);
                            let index = emit_constant(context, block, offset as f64)?;
                            emit_store(context, block, dest, index, value);
                        }
                        Ok(())
                    }
                    _ => Err(Error::NotLowerable(format!(
                        "array result from `{name}` is not supported"
                    ))),
                }
            }
            MirRvalue::Unary(op, operand) => {
                self.lower_array_unary(context, block, op, operand, dest)
            }
            MirRvalue::Binary(lhs, op, rhs) => {
                self.lower_array_binary(context, block, lhs, op, rhs, dest)
            }
            MirRvalue::Index { base, .. } => {
                // `A(:)` flattens to a column vector (linear order is preserved).
                let src = self.array_source(base)?;
                let n = self.array_len(base)?;
                for offset in 0..n {
                    let idx = emit_constant(context, block, offset as f64)?;
                    let load = LoadOp::new(context, src, idx);
                    let value = load.get_result(context);
                    append(context, block, &load);
                    let index = emit_constant(context, block, offset as f64)?;
                    emit_store(context, block, dest, index, value);
                }
                Ok(())
            }
            other => Err(Error::NotLowerable(format!("array rvalue {other:?}"))),
        }
    }

    /// The constant `(rows, cols)` of a constructor call (`zeros/ones/eye`).
    fn constructor_dims(&self, call: &MirCall) -> Result<(usize, usize)> {
        let rows = self
            .constant_arg(call, 0)?
            .ok_or_else(|| Error::NotLowerable("constructor dims must be constant".to_string()))?;
        let cols = self.constant_arg(call, 1)?.unwrap_or(rows);
        Ok((rows, cols))
    }

    /// The constant `usize` value of the `index`-th argument, if present.
    fn constant_arg(&self, call: &MirCall, index: usize) -> Result<Option<usize>> {
        match call.args.get(index) {
            Some(MirCallArg::Single(MirOperand::Constant(MirConstant::Number(text)))) => text
                .trim()
                .parse::<usize>()
                .map(Some)
                .map_err(|_| Error::NotLowerable(format!("bad constant `{text}`"))),
            None => Ok(None),
            Some(_) => Err(Error::NotLowerable(
                "constructor dims must be constant".to_string(),
            )),
        }
    }

    /// Lower a unary operator over an array (elementwise or 2-D transpose).
    fn lower_array_unary(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        op: &OperatorKind,
        operand: &MirOperand,
        dest: Value,
    ) -> Result<()> {
        match op {
            OperatorKind::Transpose | OperatorKind::ConjugateTranspose => {
                let src = self.array_source(operand)?;
                let shape = self.array_shape(operand)?;
                if shape.rank() != 2 {
                    return Err(Error::NotLowerable(
                        "only 2-D transpose is supported".to_string(),
                    ));
                }
                self.transpose(context, block, src, dest, shape.dims()[0], shape.dims()[1])
            }
            OperatorKind::UnaryMinus | OperatorKind::UnaryPlus | OperatorKind::Not => {
                let src = self.array_source(operand)?;
                let n = self.array_len(operand)?;
                self.map_unary_op(context, block, op, src, dest, n)
            }
            _ => Err(Error::NotLowerable(format!("array unary operator {op:?}"))),
        }
    }

    /// Lower a binary operator over arrays (elementwise or scalar broadcast).
    fn lower_array_binary(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        lhs: &MirOperand,
        op: &OperatorKind,
        rhs: &MirOperand,
        dest: Value,
    ) -> Result<()> {
        // Matrix power is not elementwise: route it to its own helper.
        if *op == OperatorKind::MatrixPower {
            return self.lower_matrix_power(context, block, lhs, rhs, dest);
        }
        match (self.operand_ty(lhs), self.operand_ty(rhs)) {
            (LocalTy::Scalar, LocalTy::Array { shape }) => {
                let src = self.array_source(rhs)?;
                self.map_binary_scalar(context, block, op, lhs, src, dest, shape.numel(), true)
            }
            (LocalTy::Array { shape }, LocalTy::Scalar) => {
                let src = self.array_source(lhs)?;
                self.map_binary_scalar(context, block, op, rhs, src, dest, shape.numel(), false)
            }
            (LocalTy::Array { shape: lhs_shape }, LocalTy::Array { shape: rhs_shape }) => {
                let ls = self.array_source(lhs)?;
                let rs = self.array_source(rhs)?;
                if *op == OperatorKind::MatrixMultiply {
                    self.matmul(context, block, ls, rs, dest, lhs_shape, rhs_shape)
                } else {
                    self.map_binary(context, block, op, ls, rs, dest, lhs_shape.numel())
                }
            }
            _ => Err(Error::NotLowerable(
                "unsupported array binary operands".to_string(),
            )),
        }
    }

    /// Build a single-block `for` loop body region.
    fn for_loop(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        n: usize,
        body: impl FnOnce(&Self, &mut Context, Ptr<BasicBlock>, Value) -> Result<()>,
    ) -> Result<()> {
        let start = emit_constant(context, block, 0.0)?;
        let end = emit_constant(context, block, n as f64)?;
        let step = emit_constant(context, block, 1.0)?;
        let for_op = ForOp::new(context, start, end, step);
        append(context, block, &for_op);
        let body_block = BasicBlock::new(context, None, vec![self.f64_ty]);
        body_block.insert_at_front(for_op.body_region(context), context);
        let i = body_block.deref(context).get_argument(0);
        body(self, context, body_block, i)?;
        emit_yield(context, body_block);
        Ok(())
    }

    /// Apply a unary operator elementwise from `src` into `dest`.
    fn map_unary_op(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        op: &OperatorKind,
        src: Value,
        dest: Value,
        n: usize,
    ) -> Result<()> {
        self.for_loop(context, block, n, |this, ctx, body, i| {
            let load = LoadOp::new(ctx, src, i);
            let x = load.get_result(ctx);
            append(ctx, body, &load);
            let y = this.apply_unary(ctx, body, op, x)?;
            emit_store(ctx, body, dest, i, y);
            Ok(())
        })
    }

    /// Apply a binary operator elementwise from two same-shape arrays into `dest`.
    fn map_binary(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        op: &OperatorKind,
        ls: Value,
        rs: Value,
        dest: Value,
        n: usize,
    ) -> Result<()> {
        self.for_loop(context, block, n, |this, ctx, body, i| {
            let lload = LoadOp::new(ctx, ls, i);
            let l = lload.get_result(ctx);
            append(ctx, body, &lload);
            let rload = LoadOp::new(ctx, rs, i);
            let r = rload.get_result(ctx);
            append(ctx, body, &rload);
            let y = this.apply_binary(ctx, body, op, l, r)?;
            emit_store(ctx, body, dest, i, y);
            Ok(())
        })
    }

    /// Apply a binary operator between a scalar operand and each array element.
    fn map_binary_scalar(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        op: &OperatorKind,
        scalar: &MirOperand,
        src: Value,
        dest: Value,
        n: usize,
        scalar_on_left: bool,
    ) -> Result<()> {
        self.for_loop(context, block, n, |this, ctx, body, i| {
            let scalar = this.lower_operand(ctx, body, scalar)?;
            let load = LoadOp::new(ctx, src, i);
            let x = load.get_result(ctx);
            append(ctx, body, &load);
            let y = if scalar_on_left {
                this.apply_binary(ctx, body, op, scalar, x)?
            } else {
                this.apply_binary(ctx, body, op, x, scalar)?
            };
            emit_store(ctx, body, dest, i, y);
            Ok(())
        })
    }

    /// Apply a unary `libm` function elementwise from `src` into `dest`.
    fn map_unary(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        symbol: &str,
        src: Value,
        dest: Value,
        n: usize,
    ) -> Result<()> {
        self.for_loop(context, block, n, |this, ctx, body, i| {
            let load = LoadOp::new(ctx, src, i);
            let x = load.get_result(ctx);
            append(ctx, body, &load);
            let y = this.emit_libm_call(ctx, body, symbol, &[x])?;
            emit_store(ctx, body, dest, i, y);
            Ok(())
        })
    }

    /// 2-D transpose from `src` (`rows x cols`) into `dest`, wrapped as a
    /// runtime helper call (a pure memory-layout change).
    fn transpose(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        src: Value,
        dest: Value,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let rows = emit_constant(context, block, rows as f64)?;
        let cols = emit_constant(context, block, cols as f64)?;
        self.emit_extern_call(
            context,
            block,
            crate::runtime::TRANSPOSE,
            &[dest, src, rows, cols],
        );
        Ok(())
    }

    /// Matrix multiply `C = A * B` with column-major indexing, wrapped as a
    /// runtime helper call (instead of an unrolled triple loop).
    fn matmul(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        ls: Value,
        rs: Value,
        dest: Value,
        lhs: Shape,
        rhs: Shape,
    ) -> Result<()> {
        let (m, k, n) = (lhs.dims()[0], lhs.dims()[1], rhs.dims()[1]);
        let m = emit_constant(context, block, m as f64)?;
        let k = emit_constant(context, block, k as f64)?;
        let n = emit_constant(context, block, n as f64)?;
        self.emit_extern_call(
            context,
            block,
            crate::runtime::MATMUL,
            &[dest, ls, rs, m, k, n],
        );
        Ok(())
    }

    /// Matrix power `A ^ k` for a square matrix `A` and a compile-time integer
    /// exponent `k >= 0`, wrapped as a runtime helper call. Non-integer or
    /// negative exponents (which need `expm`/inverse) are deferred.
    fn lower_matrix_power(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        lhs: &MirOperand,
        rhs: &MirOperand,
        dest: Value,
    ) -> Result<()> {
        let LocalTy::Array { shape } = self.operand_ty(lhs) else {
            return Err(Error::NotLowerable(
                "matrix power base must be an array".to_string(),
            ));
        };
        if shape.rank() != 2 || shape.dims()[0] != shape.dims()[1] {
            return Err(Error::NotLowerable(
                "matrix power requires a square matrix".to_string(),
            ));
        }
        let Some(exp) = self.const_int_exponent(rhs)? else {
            return Err(Error::NotLowerable(
                "matrix power with a runtime exponent is not supported".to_string(),
            ));
        };
        if exp < 0 {
            return Err(Error::NotLowerable(
                "negative matrix power (inverse) is not supported".to_string(),
            ));
        }
        let src = self.array_source(lhs)?;
        let m = emit_constant(context, block, shape.dims()[0] as f64)?;
        let k = emit_constant(context, block, exp as f64)?;
        self.emit_extern_call(context, block, crate::runtime::MPOWER, &[dest, src, m, k]);
        Ok(())
    }

    /// A compile-time integer exponent from a scalar operand, or `None` when the
    /// operand is not a known integer constant.
    fn const_int_exponent(&self, operand: &MirOperand) -> Result<Option<i64>> {
        match operand {
            MirOperand::Constant(MirConstant::Number(text)) => {
                text.trim().parse::<i64>().map(Some).map_err(|_| {
                    Error::NotLowerable(format!(
                        "matrix power exponent must be a non-negative integer, got `{text}`"
                    ))
                })
            }
            MirOperand::Constant(MirConstant::IntegerLiteral(literal)) => {
                Ok(Some(literal.bits() as i64))
            }
            _ => Ok(None),
        }
    }

    /// Reduce a 2-D array along dimension `dim` (1 or 2).
    fn reduce_axis(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        reducer: Reducer,
        src: Value,
        src_shape: Shape,
        dim: usize,
        dest: Value,
    ) -> Result<()> {
        if src_shape.rank() != 2 {
            return Err(Error::NotLowerable(
                "only 2-D reduction is supported".to_string(),
            ));
        }
        let (rows, cols) = (src_shape.dims()[0], src_shape.dims()[1]);
        match dim {
            1 => {
                let dst_shape = Shape::matrix(1, cols);
                for col in 0..cols {
                    let mut acc = emit_constant(context, block, reducer.init())?;
                    for row in 0..rows {
                        let index = src_shape.linear(&[row, col]);
                        let idx = emit_constant(context, block, index as f64)?;
                        let load = LoadOp::new(context, src, idx);
                        let value = load.get_result(context);
                        append(context, block, &load);
                        acc = self.reduce_step(context, block, reducer, acc, value)?;
                    }
                    let out = dst_shape.linear(&[0, col]);
                    let idx = emit_constant(context, block, out as f64)?;
                    emit_store(context, block, dest, idx, acc);
                }
            }
            2 => {
                let dst_shape = Shape::matrix(rows, 1);
                for row in 0..rows {
                    let mut acc = emit_constant(context, block, reducer.init())?;
                    for col in 0..cols {
                        let index = src_shape.linear(&[row, col]);
                        let idx = emit_constant(context, block, index as f64)?;
                        let load = LoadOp::new(context, src, idx);
                        let value = load.get_result(context);
                        append(context, block, &load);
                        acc = self.reduce_step(context, block, reducer, acc, value)?;
                    }
                    let out = dst_shape.linear(&[row, 0]);
                    let idx = emit_constant(context, block, out as f64)?;
                    emit_store(context, block, dest, idx, acc);
                }
            }
            _ => return Err(Error::NotLowerable(format!("bad reduction dim {dim}"))),
        }
        Ok(())
    }

    /// The cell backing an array-typed operand.
    fn array_source(&self, operand: &MirOperand) -> Result<Value> {
        match operand {
            MirOperand::Local(id) => self
                .locals
                .get(&id.0)
                .copied()
                .ok_or_else(|| Error::Backend(format!("array operand {id:?} has no cell"))),
            _ => Err(Error::NotLowerable(
                "array source must be a local".to_string(),
            )),
        }
    }

    /// The flattened element count of an array-typed operand.
    fn array_len(&self, operand: &MirOperand) -> Result<usize> {
        match self.operand_ty(operand) {
            LocalTy::Array { shape } => Ok(shape.numel()),
            _ => Err(Error::NotLowerable("expected an array operand".to_string())),
        }
    }

    /// The static shape of an array-typed operand.
    fn array_shape(&self, operand: &MirOperand) -> Result<Shape> {
        match self.operand_ty(operand) {
            LocalTy::Array { shape } => Ok(shape),
            _ => Err(Error::NotLowerable("expected an array operand".to_string())),
        }
    }

    /// Reduce an array to a scalar using a scalar accumulator cell.
    fn reduce(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        reducer: Reducer,
        src: Value,
        n: usize,
    ) -> Result<Value> {
        let cell_ty: TypeHandle = ArrayType::get(context, vec![1]).into();
        let alloca = AllocaOp::new(context, cell_ty);
        let acc = alloca.get_result(context);
        append(context, block, &alloca);

        let init = emit_constant(
            context,
            block,
            match reducer {
                Reducer::Add => 0.0,
                Reducer::Mul => 1.0,
                Reducer::Min => f64::INFINITY,
                Reducer::Max => f64::NEG_INFINITY,
            },
        )?;
        let zero = emit_constant(context, block, 0.0)?;
        emit_store(context, block, acc, zero, init);

        self.for_loop(context, block, n, |this, ctx, body, i| {
            let load = LoadOp::new(ctx, src, i);
            let x = load.get_result(ctx);
            append(ctx, body, &load);
            let cur = this.load_local(ctx, body, acc)?;
            let next = this.reduce_step(ctx, body, reducer, cur, x)?;
            let zero = emit_constant(ctx, body, 0.0)?;
            emit_store(ctx, body, acc, zero, next);
            Ok(())
        })?;

        let zero = emit_constant(context, block, 0.0)?;
        let load = LoadOp::new(context, acc, zero);
        let result = load.get_result(context);
        append(context, block, &load);
        Ok(result)
    }

    /// One accumulation step of a reduction.
    fn reduce_step(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        reducer: Reducer,
        acc: Value,
        x: Value,
    ) -> Result<Value> {
        match reducer {
            Reducer::Add => self.append_binop(context, block, BinOpKind::Add, acc, x),
            Reducer::Mul => self.append_binop(context, block, BinOpKind::Mul, acc, x),
            Reducer::Min => {
                let cond = self.cmpf(context, block, CmpKind::Lt, x, acc)?;
                self.select(context, block, cond, x, acc)
            }
            Reducer::Max => {
                let cond = self.cmpf(context, block, CmpKind::Gt, x, acc)?;
                self.select(context, block, cond, x, acc)
            }
        }
    }

    /// Emit a call to an external `libm` function.
    fn emit_libm_call(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        symbol: &str,
        args: &[Value],
    ) -> Result<Value> {
        let op = CallOp::new(context, symbol, args.to_vec());
        let result = op.get_result(context);
        append(context, block, &op);
        Ok(result)
    }

    /// Emit a call to a runtime helper that writes its result into the first
    /// argument (an out-buffer) and returns nothing.
    fn emit_extern_call(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        callee: &str,
        args: &[Value],
    ) {
        let op = CallVoidOp::new(context, callee, args.to_vec());
        append(context, block, &op);
    }

    /// Lower MATLAB `mod(x, y) = x - floor(x / y) * y` inline as
    /// `fmod(fmod(x, y) + y, y)`, which carries the sign of `y` (valid for
    /// `y != 0`).
    fn mod_floor(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        x: Value,
        y: Value,
    ) -> Result<Value> {
        let first = self.emit_libm_call(context, block, "fmod", &[x, y])?;
        let shifted = self.append_binop(context, block, BinOpKind::Add, first, y)?;
        self.emit_libm_call(context, block, "fmod", &[shifted, y])
    }

    /// Lower `sign(x)` inline: `1` for `x > 0`, `-1` for `x < 0`, else `0`.
    fn sign(&self, context: &mut Context, block: Ptr<BasicBlock>, x: Value) -> Result<Value> {
        let zero = emit_constant(context, block, 0.0)?;
        let one = emit_constant(context, block, 1.0)?;
        let neg_one = emit_constant(context, block, -1.0)?;
        let gt = self.cmpf(context, block, CmpKind::Gt, x, zero)?;
        let lt = self.cmpf(context, block, CmpKind::Lt, x, zero)?;
        let inner = self.select(context, block, lt, neg_one, zero)?;
        self.select(context, block, gt, one, inner)
    }

    fn apply_unary(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        op: &OperatorKind,
        value: Value,
    ) -> Result<Value> {
        match op {
            OperatorKind::UnaryPlus
            | OperatorKind::Transpose
            | OperatorKind::ConjugateTranspose => Ok(value),
            OperatorKind::UnaryMinus => {
                let zero = emit_constant(context, block, 0.0)?;
                self.append_binop(context, block, BinOpKind::Sub, zero, value)
            }
            OperatorKind::Not => {
                let truthy = self.truthy(context, block, value)?;
                let zero = emit_constant(context, block, 0.0)?;
                let one = emit_constant(context, block, 1.0)?;
                self.select(context, block, truthy, zero, one)
            }
            other => Err(Error::NotLowerable(format!("unary operator {other:?}"))),
        }
    }

    fn apply_binary(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        op: &OperatorKind,
        l: Value,
        r: Value,
    ) -> Result<Value> {
        match op {
            OperatorKind::Add => self.append_binop(context, block, BinOpKind::Add, l, r),
            OperatorKind::Subtract => self.append_binop(context, block, BinOpKind::Sub, l, r),
            OperatorKind::MatrixMultiply | OperatorKind::ElementwiseMultiply => {
                self.append_binop(context, block, BinOpKind::Mul, l, r)
            }
            OperatorKind::Mrdivide | OperatorKind::ElementwiseDivide => {
                self.append_binop(context, block, BinOpKind::Div, l, r)
            }
            // Left division: `a \ b` == `b / a` and `a .\ b` == `b ./ a`.
            OperatorKind::Mldivide | OperatorKind::ElementwiseLeftDivide => {
                self.append_binop(context, block, BinOpKind::Div, r, l)
            }
            OperatorKind::Equal => self.cmp_select(context, block, CmpKind::Eq, l, r),
            OperatorKind::NotEqual => self.cmp_select(context, block, CmpKind::Ne, l, r),
            OperatorKind::Less => self.cmp_select(context, block, CmpKind::Lt, l, r),
            OperatorKind::LessEqual => self.cmp_select(context, block, CmpKind::Le, l, r),
            OperatorKind::Greater => self.cmp_select(context, block, CmpKind::Gt, l, r),
            OperatorKind::GreaterEqual => self.cmp_select(context, block, CmpKind::Ge, l, r),
            OperatorKind::ElementwiseAnd => self.logical_and(context, block, l, r),
            OperatorKind::ElementwiseOr => self.logical_or(context, block, l, r),
            // Scalar power: `.^` and scalar `^` both lower to `libm::pow`.
            OperatorKind::ElementwisePower | OperatorKind::MatrixPower => {
                self.emit_libm_call(context, block, "pow", &[l, r])
            }
            other => Err(Error::NotLowerable(format!("binary operator {other:?}"))),
        }
    }

    fn apply_short_circuit(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        op: &MirShortCircuitOp,
        l: Value,
        r: Value,
    ) -> Result<Value> {
        match op {
            MirShortCircuitOp::And => self.logical_and(context, block, l, r),
            MirShortCircuitOp::Or => self.logical_or(context, block, l, r),
        }
    }

    fn cmp_select(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        predicate: CmpKind,
        l: Value,
        r: Value,
    ) -> Result<Value> {
        let cmp = self.cmpf(context, block, predicate, l, r)?;
        let one = emit_constant(context, block, 1.0)?;
        let zero = emit_constant(context, block, 0.0)?;
        self.select(context, block, cmp, one, zero)
    }

    fn logical_and(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        l: Value,
        r: Value,
    ) -> Result<Value> {
        let a = self.truthy(context, block, l)?;
        let b = self.truthy(context, block, r)?;
        let one = emit_constant(context, block, 1.0)?;
        let zero = emit_constant(context, block, 0.0)?;
        let inner = self.select(context, block, b, one, zero)?;
        self.select(context, block, a, inner, zero)
    }

    fn logical_or(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        l: Value,
        r: Value,
    ) -> Result<Value> {
        let a = self.truthy(context, block, l)?;
        let b = self.truthy(context, block, r)?;
        let one = emit_constant(context, block, 1.0)?;
        let zero = emit_constant(context, block, 0.0)?;
        let inner = self.select(context, block, b, one, zero)?;
        self.select(context, block, a, one, inner)
    }

    fn cmpf(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        predicate: CmpKind,
        l: Value,
        r: Value,
    ) -> Result<Value> {
        let op = CmpOp::new(context, predicate, l, r);
        let result = op.get_result(context);
        append(context, block, &op);
        Ok(result)
    }

    fn truthy(&self, context: &mut Context, block: Ptr<BasicBlock>, value: Value) -> Result<Value> {
        let zero = emit_constant(context, block, 0.0)?;
        self.cmpf(context, block, CmpKind::Ne, value, zero)
    }

    fn select(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        condition: Value,
        true_value: Value,
        false_value: Value,
    ) -> Result<Value> {
        let op = SelectOp::new(context, condition, true_value, false_value);
        let result = op.get_result(context);
        append(context, block, &op);
        Ok(result)
    }

    /// Append a `matlab.binop` operation and return its result.
    fn append_binop(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        kind: BinOpKind,
        l: Value,
        r: Value,
    ) -> Result<Value> {
        let op = BinOp::new(context, kind, l, r);
        let result = op.get_result(context);
        append(context, block, &op);
        Ok(result)
    }

    fn number_constant(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        text: &str,
    ) -> Result<Value> {
        let value = parse_number(text)
            .ok_or_else(|| Error::NotLowerable(format!("unsupported number literal {text:?}")))?;
        emit_constant(context, block, value)
    }

    /// Load the scalar stored in `cell` from `block`.
    fn load_local(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        cell: Value,
    ) -> Result<Value> {
        let zero = emit_constant(context, block, 0.0)?;
        let load = LoadOp::new(context, cell, zero);
        let result = load.get_result(context);
        append(context, block, &load);
        Ok(result)
    }

    /// Emit an `if`/`else` construct.
    fn lower_if(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        cond: &MirOperand,
        then: usize,
        els: usize,
        merge: Option<usize>,
    ) -> Result<()> {
        let cond = self.lower_operand(context, block, cond)?;
        let cond = self.truthy(context, block, cond)?;

        let if_op = IfOp::new(context, cond);
        append(context, block, &if_op);
        self.fill_region(context, if_op.then_region(context), then, merge)?;
        self.fill_region(context, if_op.else_region(context), els, merge)?;
        Ok(())
    }

    /// Emit a `while` loop. `header` is the loop header block id.
    fn lower_while(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        header_statements: &[MirStmt],
        cond: &MirOperand,
        body: usize,
        _exit: usize,
        header: usize,
    ) -> Result<()> {
        let while_op = WhileOp::new(context);
        append(context, block, &while_op);

        let before = BasicBlock::new(context, None, vec![]);
        before.insert_at_front(while_op.before_region(context), context);
        for stmt in header_statements {
            self.lower_stmt(context, before, stmt)?;
        }
        let cond = self.lower_operand(context, before, cond)?;
        let cond = self.truthy(context, before, cond)?;
        emit_condition(context, before, cond);

        let after = BasicBlock::new(context, None, vec![]);
        after.insert_at_front(while_op.after_region(context), context);
        self.lower_region(context, after, body, Some(header))?;
        emit_yield(context, after);
        Ok(())
    }

    /// Emit a `for` loop from a colon range. `header` is the loop header id.
    fn lower_for(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        binding: &MirLocalId,
        iterable: &MirRvalue,
        body: usize,
        _exit: usize,
        header: usize,
    ) -> Result<()> {
        let (start, step, end) = match iterable {
            MirRvalue::Range { start, step, end } => (start, step.as_ref(), end),
            other => {
                return Err(Error::NotLowerable(format!(
                    "unsupported for-loop iterable {other:?}"
                )))
            }
        };

        let start = self.lower_operand(context, block, start)?;
        let step = match step {
            Some(step) => self.lower_operand(context, block, step)?,
            None => emit_constant(context, block, 1.0)?,
        };
        let end = self.lower_operand(context, block, end)?;

        let cell = self
            .locals
            .get(&binding.0)
            .copied()
            .ok_or_else(|| Error::Backend(format!("no cell for loop binding {binding:?}")))?;
        let zero = emit_constant(context, block, 0.0)?;
        emit_store(context, block, cell, zero, start);

        let while_op = WhileOp::new(context);
        append(context, block, &while_op);

        // Before region: `i <= end` when ascending, `i >= end` when descending.
        let before = BasicBlock::new(context, None, vec![]);
        before.insert_at_front(while_op.before_region(context), context);
        let i = self.load_local(context, before, cell)?;
        let zero = emit_constant(context, before, 0.0)?;
        let ascending = self.cmpf(context, before, CmpKind::Ge, step, zero)?;
        let le = self.cmpf(context, before, CmpKind::Le, i, end)?;
        let ge = self.cmpf(context, before, CmpKind::Ge, i, end)?;
        let cond = self.select(context, before, ascending, le, ge)?;
        emit_condition(context, before, cond);

        // After region: body, then `i = i + step`.
        let after = BasicBlock::new(context, None, vec![]);
        after.insert_at_front(while_op.after_region(context), context);
        self.lower_region(context, after, body, Some(header))?;
        let i = self.load_local(context, after, cell)?;
        let next = self.append_binop(context, after, BinOpKind::Add, i, step)?;
        let zero = emit_constant(context, after, 0.0)?;
        emit_store(context, after, cell, zero, next);
        emit_yield(context, after);
        Ok(())
    }

    /// Emit a `switch`/`case`/`otherwise` construct as a nested `if`/`else`
    /// chain (MATLAB switch cases are exclusive and do not fall through).
    fn lower_switch(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        discr: &MirOperand,
        cases: &[(MirOperand, BasicBlockId)],
        otherwise: usize,
        merge: Option<usize>,
    ) -> Result<()> {
        let discr = self.lower_operand(context, block, discr)?;

        let mut case_values = Vec::with_capacity(cases.len());
        for (operand, _) in cases {
            case_values.push(self.lower_operand(context, block, operand)?);
        }

        self.emit_switch(context, block, discr, &case_values, cases, otherwise, merge)
    }

    /// Recursively emit the nested `if`/`else` chain for a switch into `block`.
    fn emit_switch(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        discr: Value,
        case_values: &[Value],
        cases: &[(MirOperand, BasicBlockId)],
        otherwise: usize,
        merge: Option<usize>,
    ) -> Result<()> {
        match case_values.split_first() {
            None => {
                // Innermost `else`: the `otherwise` body.
                self.lower_region(context, block, otherwise, merge)?;
                Ok(())
            }
            Some((first, rest)) => {
                let first_block = cases[0].1 .0;
                let cmp = self.cmpf(context, block, CmpKind::Eq, discr, *first)?;
                let if_op = IfOp::new(context, cmp);
                append(context, block, &if_op);
                self.fill_region(context, if_op.then_region(context), first_block, merge)?;
                self.fill_switch_else(
                    context,
                    if_op.else_region(context),
                    discr,
                    rest,
                    &cases[1..],
                    otherwise,
                    merge,
                )?;
                Ok(())
            }
        }
    }

    fn fill_switch_else(
        &self,
        context: &mut Context,
        region: Ptr<Region>,
        discr: Value,
        case_values: &[Value],
        cases: &[(MirOperand, BasicBlockId)],
        otherwise: usize,
        merge: Option<usize>,
    ) -> Result<()> {
        let block = BasicBlock::new(context, None, vec![]);
        block.insert_at_front(region, context);
        self.emit_switch(context, block, discr, case_values, cases, otherwise, merge)?;
        emit_yield(context, block);
        Ok(())
    }

    /// Fill a single-block region for a MIR block range, terminating with `yield`.
    fn fill_region(
        &self,
        context: &mut Context,
        region: Ptr<Region>,
        start: usize,
        stop: Option<usize>,
    ) -> Result<()> {
        let block = BasicBlock::new(context, None, vec![]);
        block.insert_at_front(region, context);
        self.lower_region(context, block, start, stop)?;
        emit_yield(context, block);
        Ok(())
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

/// The static element count of a shape, deferring dynamic shapes (they need
/// runtime heap allocation, which is not implemented yet).
fn static_numel(shape: Shape) -> Result<usize> {
    if shape.is_dynamic() {
        Err(Error::NotLowerable(
            "dynamic shape arrays need runtime heap allocation (not implemented)".to_string(),
        ))
    } else {
        Ok(shape.numel())
    }
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

/// Produce a readable, nested dump of the `matlab`-dialect module for tests.
fn dump_module(context: &Context, module: ModuleOp) -> String {
    let mut out = String::new();
    if let Some(block) = module.get_region(context).deref(context).get_entry_block() {
        dump_block(context, &mut out, block, 0);
    }
    out
}

fn dump_block(context: &Context, out: &mut String, block: Ptr<BasicBlock>, indent: usize) {
    for op in block.deref(context).iter(context) {
        let opid = Operation::get_opid(op, context);
        let pad = "  ".repeat(indent);
        if let Some(call) = Operation::get_op::<CallOp>(op, context) {
            out.push_str(&format!("{pad}{opid} @{}\n", call.callee(context)));
        } else if let Some(call) = Operation::get_op::<CallVoidOp>(op, context) {
            out.push_str(&format!("{pad}{opid} @{}\n", call.callee(context)));
        } else {
            out.push_str(&format!("{pad}{opid}\n"));
        }
        for region in op.deref(context).regions() {
            if let Some(child) = region.deref(context).get_entry_block() {
                dump_block(context, out, child, indent + 1);
            }
        }
    }
}
