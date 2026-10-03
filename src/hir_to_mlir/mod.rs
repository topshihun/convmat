//! Layer 3: lower HIR to the `matlab` pliron dialect.
//!
//! A [`runmat_hir::HirAssembly`] is traversed and lowered to the [`matlab`]
//! dialect (dense column-major arrays, scalar arithmetic, comparisons, `libm`
//! calls, and structured control flow) hosted in a builtin `module`/`func`.
//! The result is handed to [`crate::lowering`] for the `emitc` conversion.
//!
//! HIR is structured (statements + expression trees), so there is no CFG to
//! recover: `if`/`while`/`for`/`switch` are emitted directly as the `matlab`
//! dialect's structured control-flow ops. Every binding becomes a stack array
//! cell (a `1`-element array for scalars, an `N`-element array otherwise);
//! assignments are `store`s and reads are `load`s. Inline array literals
//! (`Tensor`) are materialized into a temp cell before use as an array operand.
//!
//! The recursive lowering helpers thread `Context` alongside the HIR nodes, so
//! several of them naturally exceed Clippy's argument-count heuristic.
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
    AllocaOp, ArrayType, BinOp, BinOpKind, BreakOp, CallOp, CallVoidOp, CmpKind, CmpOp,
    ConditionOp, ConstantOp, ContinueOp, DeleteOp, ForOp, IfOp, LoadOp, PtrType, RangeForOp,
    ReturnOp, SelectOp, StoreOp, StructCopyOp, StructGetOp, StructSetOp, StructType, WhileOp,
    YieldOp,
};
use crate::error::{Error, Result};
use crate::triage::{
    analyze_handles, brace_index, call_name, component_end_offset, component_is_colon, expr_ty,
    infer_locals, unquote_str, varargout_index, LocalTy, Shape, Variadics,
};

use runmat_hir::{
    BindingId, BindingStorage, FunctionId, FunctionKind, HirAssembly, HirBlock, HirCall, HirExpr,
    HirExprKind, HirFunction, HirPlace, HirStmt, HirStmtKind, IndexComponent, IndexKind,
    IndexingSemantics, OperatorKind,
};

type OpInserter = IRInserter<DummyListener>;

/// The stack budget for a single statically-shaped array, in elements. Arrays
/// with `numel > STACK_ELEMS_LIMIT` are heap-allocated (and freed on return)
/// instead of taking a stack slot; smaller arrays stay on the stack. At 8 bytes
/// per `double`, this caps a single stack array at ~32 KiB.
const STACK_ELEMS_LIMIT: usize = 4096;

/// Whether a static array of the given shape should live on the heap.
fn alloc_on_heap(shape: Shape) -> bool {
    !shape.is_dynamic() && shape.numel() > STACK_ELEMS_LIMIT
}

/// Lower HIR to a readable dump of the generated `matlab`-dialect IR.
pub fn lower(hir: &HirAssembly) -> Result<String> {
    let mut context = Context::new();
    let module = lower_to_module(&mut context, hir)?;
    Ok(dump_module(&context, module))
}

/// Lower HIR into a new `matlab`-dialect module owned by `context`.
pub fn lower_to_module(context: &mut Context, hir: &HirAssembly) -> Result<ModuleOp> {
    let f64_ty: TypeHandle = FP64Type::get(context).into();
    let module = ModuleOp::new(context, Identifier::try_from("convmat").unwrap());

    // Map each binding to its storage class, so `persistent`/`global` bindings
    // can be allocated as `static` cells.
    let storage: HashMap<BindingId, BindingStorage> = hir
        .bindings
        .iter()
        .map(|binding| (binding.id, binding.storage.clone()))
        .collect();

    for function in &hir.functions {
        lower_function(context, function, hir, &storage, &module, f64_ty)?;
    }

    Ok(module)
}

/// The C-level name of a HIR function. Anonymous functions carry a
/// non-identifier source name (`anonymous#1`), so they are renamed to a stable
/// valid C identifier; every other function keeps its source name.
fn c_function_name(function: &HirFunction) -> String {
    match function.kind {
        FunctionKind::Anonymous => format!("convmat_anon_{}", function.id.0),
        _ => function.name.0.clone(),
    }
}

/// The ordered captured bindings of an anonymous function, matching the extra
/// trailing parameters [`lower_function`] adds for captures.
fn anon_capture_bindings(hir: &HirAssembly, id: FunctionId) -> Result<Vec<BindingId>> {
    let function = hir
        .functions
        .iter()
        .find(|function| function.id == id)
        .ok_or_else(|| Error::Backend(format!("anonymous function #{} not found", id.0)))?;
    Ok(function
        .captures
        .iter()
        .map(|capture| capture.binding)
        .collect())
}

/// Whether an anonymous function returns a single scalar value (the only result
/// ABI a handle call can currently express).
fn anon_returns_scalar(hir: &HirAssembly, id: FunctionId) -> bool {
    let Some(function) = hir.functions.iter().find(|function| function.id == id) else {
        return false;
    };
    if function.outputs.len() != 1 {
        return false;
    }
    matches!(
        infer_locals(function).get(&function.outputs[0]),
        None | Some(LocalTy::Scalar)
    )
}

/// A statically-resolved anonymous-function handle in a caller: the target
/// function, its ordered captured bindings, and one caller-frame snapshot cell
/// per capture (written at handle creation, read at every call).
struct HandleRuntime {
    function: FunctionId,
    captures: Vec<BindingId>,
    snapshot_cells: Vec<Value>,
    /// Whether the target returns a single scalar (the only supported ABI).
    scalar_result: bool,
}

/// Lower a single HIR function into a `builtin.func` appended to `module`.
fn lower_function(
    context: &mut Context,
    function: &HirFunction,
    hir: &HirAssembly,
    storage: &HashMap<BindingId, BindingStorage>,
    module: &ModuleOp,
    f64_ty: TypeHandle,
) -> Result<()> {
    let name = c_function_name(function);

    let mut tys = infer_locals(function);
    let variadics = Variadics::compute(function);

    // Named (non-variadic) scalar parameters: `fixed_inputs` minus `varargin`.
    let varargin_binding = function.abi.varargin;
    let named_params: Vec<BindingId> = function
        .abi
        .fixed_inputs
        .iter()
        .filter(|binding| Some(**binding) != varargin_binding)
        .copied()
        .collect();

    // Named (non-variadic) outputs: `fixed_outputs` minus `varargout`.
    let varargout_binding = function.abi.varargout;
    let named_outputs: Vec<BindingId> = function
        .abi
        .fixed_outputs
        .iter()
        .filter(|binding| Some(**binding) != varargout_binding)
        .copied()
        .collect();

    // Captured bindings of an anonymous function. They are not part of
    // `function.locals` (they belong to the enclosing function), so they become
    // extra trailing parameters of the generated C function.
    let capture_bindings: Vec<BindingId> = function
        .captures
        .iter()
        .map(|capture| capture.binding)
        .collect();

    // Split named outputs by ABI: scalars and structs are returned by value;
    // static-shape arrays become out-pointer parameters; dynamic-shape arrays
    // become an out-pointer buffer *plus* an out-length cell (the callee writes
    // the actual element count there). `varargout` elements are specialized as
    // extra scalar outputs.
    let mut return_outputs: Vec<(BindingId, TypeHandle)> = Vec::new();
    let mut array_outputs: Vec<BindingId> = Vec::new();
    let mut dynamic_outputs: Vec<BindingId> = Vec::new();
    for output in &named_outputs {
        match tys.get(output).cloned().unwrap_or(LocalTy::Scalar) {
            LocalTy::Scalar => return_outputs.push((*output, f64_ty)),
            LocalTy::Struct { fields } => {
                return_outputs.push((*output, struct_type(context, &fields)?));
            }
            LocalTy::Array { shape } if shape.is_dynamic() => dynamic_outputs.push(*output),
            LocalTy::Array { .. } => array_outputs.push(*output),
            LocalTy::Dynamic => {
                return Err(Error::NotLowerable("dynamic output shape".to_string()))
            }
        }
    }

    // Entry block argument layout: named params (scalar: 1 arg; struct: 1 arg;
    // dynamic array: `double* data` + `double n`), then `varargin` scalars, then
    // array out-params. Track each named param's first argument slot so the body
    // can wire its data pointer and length.
    let scalar_cell_ty: TypeHandle = ArrayType::get(context, vec![1]).into();
    let mut entry_arg_types: Vec<TypeHandle> = Vec::new();
    let mut named_param_args: HashMap<BindingId, usize> = HashMap::new();
    let mut array_params: Vec<BindingId> = Vec::new();
    for param in &named_params {
        named_param_args.insert(*param, entry_arg_types.len());
        match tys.get(param).cloned().unwrap_or(LocalTy::Scalar) {
            LocalTy::Scalar => entry_arg_types.push(f64_ty),
            LocalTy::Struct { fields } => entry_arg_types.push(struct_type(context, &fields)?),
            LocalTy::Array { shape } if shape.is_dynamic() => {
                entry_arg_types.push(PtrType::get(context).into());
                entry_arg_types.push(f64_ty);
                array_params.push(*param);
            }
            LocalTy::Array { .. } => {
                return Err(Error::NotLowerable(
                    "static-shape array parameters are not supported".to_string(),
                ))
            }
            LocalTy::Dynamic => return Err(Error::NotLowerable("dynamic parameter".to_string())),
        }
    }
    let named_param_arg_count = entry_arg_types.len();
    entry_arg_types.extend(vec![f64_ty; variadics.varargin_count]);
    for output in &array_outputs {
        let LocalTy::Array { shape } = tys[output] else {
            unreachable!("array output must have an array type");
        };
        entry_arg_types.push(ArrayType::get(context, vec![static_numel(shape)? as i64]).into());
    }
    // Dynamic array outputs: an out-buffer (`double*`) and an out-length
    // (`double*`) the callee writes the actual element count into.
    for _ in &dynamic_outputs {
        entry_arg_types.push(PtrType::get(context).into());
        entry_arg_types.push(PtrType::get(context).into());
    }
    // Anonymous-function captures: one trailing scalar parameter per captured
    // binding, snapshotted by the caller at handle creation.
    let capture_base = entry_arg_types.len();
    entry_arg_types.extend(vec![f64_ty; capture_bindings.len()]);
    // Result types: value outputs (scalar/struct) first, then the extra scalar
    // `varargout` cells.
    let mut output_types: Vec<TypeHandle> = return_outputs.iter().map(|(_, ty)| *ty).collect();
    output_types.extend(vec![f64_ty; variadics.varargout_count]);
    let fn_ty = FunctionType::get(context, entry_arg_types.clone(), output_types);
    let name_id = Identifier::try_from(name.as_str())
        .map_err(|e| Error::Backend(format!("bad function name `{name}`: {e}")))?;
    let func = FuncOp::new(context, name_id, fn_ty);
    let entry = func.get_entry_block(context);

    // Allocate a stack cell per binding. Array outputs and array parameters are
    // caller-provided and are wired to their incoming arguments below; the
    // `varargin`/`varargout` cell bindings are specialized away.
    let mut skip_ids: HashSet<BindingId> = array_outputs.iter().copied().collect();
    skip_ids.extend(dynamic_outputs.iter().copied());
    skip_ids.extend(array_params.iter().copied());
    if let Some(local) = variadics.varargin_local {
        skip_ids.insert(local);
    }
    if let Some(local) = variadics.varargout_local {
        skip_ids.insert(local);
    }
    let mut locals: HashMap<BindingId, Value> = HashMap::new();
    let mut array_lens: HashMap<BindingId, Value> = HashMap::new();
    let mut array_out_lens: HashMap<BindingId, Value> = HashMap::new();
    let mut heap_cells: Vec<Value> = Vec::new();
    for binding in &function.locals {
        if skip_ids.contains(binding) {
            continue;
        }
        let (array_ty, shape) = match tys.get(binding).cloned().unwrap_or(LocalTy::Scalar) {
            LocalTy::Scalar => (scalar_cell_ty, None),
            LocalTy::Array { shape } => (
                ArrayType::get(context, vec![static_numel(shape)? as i64]).into(),
                Some(shape),
            ),
            LocalTy::Struct { fields } => (struct_type(context, &fields)?, None),
            LocalTy::Dynamic => return Err(Error::NotLowerable("dynamic local shape".to_string())),
        };

        // Storage class first (`persistent`/`global` -> `static`), then the heap
        // decision for large, non-static arrays.
        let is_static = matches!(
            storage.get(binding),
            Some(BindingStorage::Persistent) | Some(BindingStorage::Global)
        );
        let on_heap = !is_static && shape.is_some_and(alloc_on_heap);
        let alloca = if is_static {
            AllocaOp::new_static(context, array_ty)
        } else if on_heap {
            AllocaOp::new_heap(context, array_ty)
        } else {
            AllocaOp::new(context, array_ty)
        };
        let value = alloca.get_result(context);
        append(context, entry, &alloca);
        if on_heap {
            heap_cells.push(value);
        }
        locals.insert(*binding, value);
    }

    // Anonymous-function captures: one scalar cell per captured binding, wired
    // from the trailing parameters below. Their types are scalar by definition
    // of the MVP (only scalar captures are supported).
    let mut capture_cells: Vec<Value> = Vec::with_capacity(capture_bindings.len());
    for binding in &capture_bindings {
        let alloca = AllocaOp::new(context, scalar_cell_ty);
        let value = alloca.get_result(context);
        append(context, entry, &alloca);
        capture_cells.push(value);
        locals.insert(*binding, value);
        tys.insert(*binding, LocalTy::Scalar);
    }

    // Wire incoming named parameters into their cells. Scalar params are stored
    // element-wise into a length-1 cell; struct params are copied by value
    // (`StructCopyOp`); dynamic-array params are caller-provided pointers wired
    // directly (data pointer + length).
    for (param, &arg_index) in &named_param_args {
        let argument = entry.deref(context).get_argument(arg_index);
        match tys.get(param).cloned().unwrap_or(LocalTy::Scalar) {
            LocalTy::Array { shape } if shape.is_dynamic() => {
                let len = entry.deref(context).get_argument(arg_index + 1);
                locals.insert(*param, argument);
                array_lens.insert(*param, len);
            }
            _ => {
                let target = locals
                    .get(param)
                    .copied()
                    .ok_or_else(|| Error::Backend(format!("no cell for parameter {param:?}")))?;
                match tys.get(param).cloned().unwrap_or(LocalTy::Scalar) {
                    LocalTy::Struct { .. } => {
                        let copy = StructCopyOp::new(context, target, argument);
                        append(context, entry, &copy);
                    }
                    _ => {
                        let zero = emit_constant(context, entry, 0.0)?;
                        emit_store(context, entry, target, zero, argument);
                    }
                }
            }
        }
    }

    // Anonymous-function captures arrive as trailing scalar arguments; store
    // each into its cell so the body can read it like any other local.
    for (offset, cell) in capture_cells.iter().enumerate() {
        let argument = entry.deref(context).get_argument(capture_base + offset);
        let zero = emit_constant(context, entry, 0.0)?;
        emit_store(context, entry, *cell, zero, argument);
    }

    // `varargin{k}` resolves directly to the (k-1)-th extra scalar argument.
    let mut varargin_args = Vec::with_capacity(variadics.varargin_count);
    for k in 0..variadics.varargin_count {
        varargin_args.push(entry.deref(context).get_argument(named_param_arg_count + k));
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
        let argument = entry
            .deref(context)
            .get_argument(named_param_arg_count + variadics.varargin_count + offset);
        locals.insert(*output, argument);
    }

    // Wire dynamic array outputs to their out-buffer + out-length arguments.
    let dynamic_base = named_param_arg_count + variadics.varargin_count + array_outputs.len();
    for (offset, output) in dynamic_outputs.iter().enumerate() {
        let buffer = entry.deref(context).get_argument(dynamic_base + 2 * offset);
        let len = entry
            .deref(context)
            .get_argument(dynamic_base + 2 * offset + 1);
        locals.insert(*output, buffer);
        array_out_lens.insert(*output, len);
    }

    // Resolve anonymous-function handles defined in this function and allocate
    // one snapshot cell per capture. The cells are written at the handle's
    // definition and read at every call, giving MATLAB's capture-at-creation
    // semantics.
    let mut handles: HashMap<BindingId, HandleRuntime> = HashMap::new();
    let mut handle_targets: HashMap<BindingId, FunctionId> = HashMap::new();
    for (binding, function_id) in analyze_handles(function).map_err(Error::NotLowerable)? {
        let captures = anon_capture_bindings(hir, function_id)?;
        let mut snapshot_cells = Vec::with_capacity(captures.len());
        for _ in &captures {
            let alloca = AllocaOp::new(context, scalar_cell_ty);
            let value = alloca.get_result(context);
            append(context, entry, &alloca);
            snapshot_cells.push(value);
        }
        handles.insert(
            binding,
            HandleRuntime {
                function: function_id,
                captures,
                snapshot_cells,
                scalar_result: anon_returns_scalar(hir, function_id),
            },
        );
        handle_targets.insert(binding, function_id);
    }

    let lowerer = FuncLowerer {
        f64_ty,
        locals,
        tys,
        heap_cells,
        varargin_local: variadics.varargin_local,
        varargin_args,
        varargout_local: variadics.varargout_local,
        varargout_cells,
        array_lens,
        array_out_lens,
        return_outputs,
        handles,
        handle_targets,
    };
    lowerer.lower_block(context, entry, &function.body)?;

    // Implicit return: free heap cells, load the scalar outputs (in order), and
    // emit `matlab.return`.
    lowerer.emit_heap_frees(context, entry);
    lowerer.emit_return_values(context, entry)?;

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
/// threaded separately so the HIR borrow and the IR construction never alias).
struct FuncLowerer {
    f64_ty: TypeHandle,
    locals: HashMap<BindingId, Value>,
    tys: HashMap<BindingId, LocalTy>,
    /// Heap-allocated array cells that must be freed on return.
    heap_cells: Vec<Value>,
    /// The binding backing `varargin` (if any), specialized away.
    varargin_local: Option<BindingId>,
    /// The entry-block argument `Value` for each `varargin{k}` (0-based).
    varargin_args: Vec<Value>,
    /// The binding backing `varargout` (if any), specialized away.
    varargout_local: Option<BindingId>,
    /// The scalar cell for each `varargout{k}` (0-based).
    varargout_cells: Vec<Value>,
    /// The runtime length (`double n`) of each dynamic-shape array parameter.
    array_lens: HashMap<BindingId, Value>,
    /// The out-length cell (`double*`) of each dynamic-shape array output.
    array_out_lens: HashMap<BindingId, Value>,
    /// Value outputs (scalar or struct), in ABI order, with their return type.
    return_outputs: Vec<(BindingId, TypeHandle)>,
    /// Statically-resolved anonymous-function handles: target + capture cells.
    handles: HashMap<BindingId, HandleRuntime>,
    /// Just the handle targets, for `expr_ty`'s call typing.
    handle_targets: HashMap<BindingId, FunctionId>,
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

impl FuncLowerer {
    /// Lower every statement in `hir_block` into `block`.
    fn lower_block(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        hir_block: &HirBlock,
    ) -> Result<()> {
        for stmt in &hir_block.statements {
            self.lower_stmt(context, block, stmt)?;
        }
        Ok(())
    }

    /// Lower a single statement into the current block.
    fn lower_stmt(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        stmt: &HirStmt,
    ) -> Result<()> {
        match &stmt.kind {
            HirStmtKind::ExprStmt(expr, _) => {
                // Evaluate for (potential) side effects; the result is dropped.
                self.lower_expr(context, block, expr)?;
                Ok(())
            }
            HirStmtKind::Assign(place, value, _) => {
                // `varargout{k} = expr` writes the k-th extra scalar output cell.
                if let Some(k) = varargout_index(place, self.varargout_local) {
                    let value = self.lower_expr(context, block, value)?;
                    let cell = self.varargout_cells.get(k - 1).copied().ok_or_else(|| {
                        Error::NotLowerable(format!("varargout{{{k}}} is out of range"))
                    })?;
                    let zero = emit_constant(context, block, 0.0)?;
                    emit_store(context, block, cell, zero, value);
                    return Ok(());
                }
                // `s.field = expr` writes a struct field.
                if let HirPlace::Member(base, name) = place {
                    let cell = self.struct_cell(base)?;
                    let value = self.lower_expr(context, block, value)?;
                    let op = StructSetOp::new(context, cell, &name.0, value);
                    append(context, block, &op);
                    return Ok(());
                }
                let HirPlace::Binding(target) = place else {
                    return Err(Error::NotLowerable(format!(
                        "non-local assignment target {place:?}"
                    )));
                };
                // `f = @(...)`: snapshot the captured bindings at creation time.
                // The handle binding itself has no runtime cell.
                if matches!(value.kind, HirExprKind::AnonymousFunction(_)) {
                    if let Some(handle) = self.handles.get(target) {
                        return self.lower_handle_creation(context, block, handle);
                    }
                }
                let cell = self
                    .locals
                    .get(target)
                    .copied()
                    .ok_or_else(|| Error::Backend(format!("no cell for binding {target:?}")))?;
                match self.tys.get(target).cloned().unwrap_or(LocalTy::Scalar) {
                    LocalTy::Scalar => {
                        let value = self.lower_expr(context, block, value)?;
                        let zero = emit_constant(context, block, 0.0)?;
                        emit_store(context, block, cell, zero, value);
                    }
                    LocalTy::Array { shape } if shape.is_dynamic() => {
                        // Dynamic array output: fill the caller's buffer, then
                        // report the actual element count to the out-length cell
                        // (which may be absent for an intermediate, unused value).
                        self.lower_array_expr_into(context, block, value, cell)?;
                        if let Some(out_len) = self.array_out_lens.get(target).copied() {
                            let len = self.dynamic_array_len(value)?;
                            let zero = emit_constant(context, block, 0.0)?;
                            emit_store(context, block, out_len, zero, len);
                        }
                    }
                    LocalTy::Array { .. } => {
                        self.lower_array_expr_into(context, block, value, cell)?;
                    }
                    LocalTy::Struct { .. } => {
                        self.lower_struct_expr_into(context, block, value, cell)?;
                    }
                    LocalTy::Dynamic => {
                        return Err(Error::NotLowerable("dynamic assignment target".to_string()));
                    }
                }
                Ok(())
            }
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => self.lower_if(context, block, cond, then_body, elseif_blocks, else_body),
            HirStmtKind::While { cond, body } => self.lower_while(context, block, cond, body),
            HirStmtKind::For {
                binding,
                range,
                body,
            } => self.lower_for(context, block, *binding, range, body),
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
            } => self.lower_switch(context, block, expr, cases, otherwise),
            HirStmtKind::Break => {
                let op = BreakOp::new(context);
                append(context, block, &op);
                Ok(())
            }
            HirStmtKind::Continue => {
                let op = ContinueOp::new(context);
                append(context, block, &op);
                Ok(())
            }
            // `global`/`persistent` declarations are already materialized as
            // `static` cells during allocation; the statement itself is a no-op.
            HirStmtKind::Global(_) | HirStmtKind::Persistent(_) => Ok(()),
            HirStmtKind::Return => {
                self.emit_heap_frees(context, block);
                self.emit_return_values(context, block)
            }
            other => Err(Error::NotLowerable(format!(
                "unsupported statement {other:?}"
            ))),
        }
    }

    /// Lower a scalar expression to a `f64` value in the current block.
    fn lower_expr(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        expr: &HirExpr,
    ) -> Result<Value> {
        match &expr.kind {
            HirExprKind::Binding(id) => {
                let cell = self
                    .locals
                    .get(id)
                    .copied()
                    .ok_or_else(|| Error::Backend(format!("binding {id:?} has no cell")))?;
                self.load_local(context, block, cell)
            }
            HirExprKind::Number(text) => self.number_constant(context, block, text),
            HirExprKind::IntegerLiteral(literal) => {
                emit_constant(context, block, literal.bits() as f64)
            }
            HirExprKind::Constant(symbol) => match symbol.0.as_str() {
                "true" => emit_constant(context, block, 1.0),
                "false" => emit_constant(context, block, 0.0),
                other => Err(Error::NotLowerable(format!("constant `{other}`"))),
            },
            HirExprKind::Unary(op, operand) => {
                let value = self.lower_expr(context, block, operand)?;
                self.apply_unary(context, block, op, value)
            }
            HirExprKind::Binary(lhs, op, rhs) => {
                let l = self.lower_expr(context, block, lhs)?;
                let r = self.lower_expr(context, block, rhs)?;
                self.apply_binary(context, block, op, l, r)
            }
            HirExprKind::Call(call) => self.lower_scalar_call(context, block, call),
            HirExprKind::Index(base, indexing) => {
                // A call through an anonymous-function handle. HIR represents
                // `f(args)` on a binding as paren indexing, so this must be
                // resolved before the array-index path.
                if let HirExprKind::Binding(id) = &base.kind {
                    if let Some(handle) = self.handles.get(id) {
                        return self.lower_handle_call(context, block, handle, indexing);
                    }
                }
                self.lower_index_scalar(context, block, base, indexing)
            }
            HirExprKind::Member(base, name) => {
                let cell = self.struct_cell(base)?;
                let op = StructGetOp::new(context, cell, &name.0);
                let result = op.get_result(context);
                append(context, block, &op);
                Ok(result)
            }
            other => Err(Error::NotLowerable(format!("rvalue {other:?}"))),
        }
    }

    /// The struct cell backing an expression that must resolve to a `Binding`.
    fn struct_cell(&self, expr: &HirExpr) -> Result<Value> {
        match &expr.kind {
            HirExprKind::Binding(id) => self
                .locals
                .get(id)
                .copied()
                .ok_or_else(|| Error::Backend(format!("binding {id:?} has no cell"))),
            _ => Err(Error::NotLowerable(
                "struct field base must be a binding".to_string(),
            )),
        }
    }

    /// Lower a handle definition `f = @(...)`: snapshot each captured binding
    /// into its cell. MATLAB captures the current value at handle creation, so
    /// the snapshot is taken here, not at the call site.
    fn lower_handle_creation(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        handle: &HandleRuntime,
    ) -> Result<()> {
        for (capture, cell) in handle.captures.iter().zip(&handle.snapshot_cells) {
            // Only scalar captures are supported; an array/struct capture would
            // be silently truncated to its first element by `load_local`.
            let ty = self.tys.get(capture).cloned().unwrap_or(LocalTy::Scalar);
            if !matches!(ty, LocalTy::Scalar) {
                return Err(Error::NotLowerable(
                    "anonymous function captures must be scalar \
                     (only scalar captures are supported)"
                        .to_string(),
                ));
            }
            let source = self.locals.get(capture).copied().ok_or_else(|| {
                Error::Backend(format!("captured binding {capture:?} has no cell"))
            })?;
            let value = self.load_local(context, block, source)?;
            let zero = emit_constant(context, block, 0.0)?;
            emit_store(context, block, *cell, zero, value);
        }
        Ok(())
    }

    /// Lower a call through an anonymous-function handle to a `matlab.call` on
    /// the specialized helper, passing the call arguments followed by the
    /// captured snapshots (the helper's trailing parameters).
    fn lower_handle_call(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        handle: &HandleRuntime,
        indexing: &IndexingSemantics,
    ) -> Result<Value> {
        if indexing.kind != IndexKind::Paren {
            return Err(Error::NotLowerable(
                "function handle must be called with `()`".to_string(),
            ));
        }
        if !handle.scalar_result {
            return Err(Error::NotLowerable(
                "anonymous function must return a scalar \
                 (array results are not supported)"
                    .to_string(),
            ));
        }
        let mut args: Vec<Value> =
            Vec::with_capacity(indexing.components.len() + handle.snapshot_cells.len());
        for component in &indexing.components {
            let IndexComponent::Expr(expr) = component else {
                return Err(Error::NotLowerable(
                    "function handle call arguments must be values".to_string(),
                ));
            };
            // Only scalar arguments are supported; an array argument would be
            // silently truncated to its first element by `lower_expr`.
            if !matches!(self.operand_type(expr), LocalTy::Scalar) {
                return Err(Error::NotLowerable(
                    "anonymous function arguments must be scalar \
                     (only scalar parameters are supported)"
                        .to_string(),
                ));
            }
            args.push(self.lower_expr(context, block, expr)?);
        }
        for cell in &handle.snapshot_cells {
            args.push(self.load_local(context, block, *cell)?);
        }
        let op = CallOp::new(
            context,
            &format!("convmat_anon_{}", handle.function.0),
            args,
        );
        let result = op.get_result(context);
        append(context, block, &op);
        Ok(result)
    }

    /// Lower a struct-producing expression into `dest` (a struct cell). Only
    /// `struct('a', 1, ...)` construction is supported.
    fn lower_struct_expr_into(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        expr: &HirExpr,
        dest: Value,
    ) -> Result<()> {
        let HirExprKind::Call(call) = &expr.kind else {
            return Err(Error::NotLowerable(
                "struct value must come from `struct(...)`".to_string(),
            ));
        };
        let name = call_name(&call.callee)
            .ok_or_else(|| Error::NotLowerable("dynamic struct call".to_string()))?;
        if name != "struct" {
            return Err(Error::NotLowerable(
                "struct value must come from `struct(...)`".to_string(),
            ));
        }
        let mut args = call.args.iter();
        while let Some(field_name_arg) = args.next() {
            let Some(value_arg) = args.next() else {
                return Err(Error::NotLowerable(
                    "struct requires field values".to_string(),
                ));
            };
            let HirExprKind::String(field_name) = &field_name_arg.kind else {
                return Err(Error::NotLowerable(
                    "struct field names must be strings".to_string(),
                ));
            };
            let value = self.lower_expr(context, block, value_arg)?;
            let op = StructSetOp::new(context, dest, unquote_str(&field_name.0), value);
            append(context, block, &op);
        }
        Ok(())
    }

    /// The static type of an operand expression, as determined by triage's shape
    /// analysis.
    fn operand_type(&self, expr: &HirExpr) -> LocalTy {
        expr_ty(expr, &self.tys, self.varargin_local, &self.handle_targets)
    }

    /// Allocate a temporary array cell of the given static shape in `block`.
    fn alloca_array(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        shape: Shape,
    ) -> Result<Value> {
        let ty: TypeHandle = ArrayType::get(context, vec![static_numel(shape)? as i64]).into();
        let alloca = AllocaOp::new(context, ty);
        let value = alloca.get_result(context);
        append(context, block, &alloca);
        Ok(value)
    }

    /// Lower scalar indexing `A(i, j)` / `A(i)` to a single `f64` load, or
    /// `varargin{k}` to the `k`-th extra scalar argument.
    fn lower_index_scalar(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        base: &HirExpr,
        indexing: &IndexingSemantics,
    ) -> Result<Value> {
        // `varargin{k}` (constant `k`) is a direct reference to an argument.
        if indexing.kind == IndexKind::Brace {
            if let HirExprKind::Binding(id) = base.kind {
                if self.varargin_local == Some(id) {
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
        let src = self.array_source(context, block, base)?;
        let shape = self.array_shape(base)?;
        let index = if shape.is_dynamic() {
            self.dynamic_linear_offset(context, block, indexing, base)?
        } else {
            let offset = self.static_linear_offset(indexing, shape)?;
            emit_constant(context, block, offset as f64)?
        };
        let load = LoadOp::new(context, src, index);
        let result = load.get_result(context);
        append(context, block, &load);
        Ok(result)
    }

    /// Runtime 0-based linear offset for a scalar index into a dynamic-shape
    /// array parameter (treated as a vector). Supports a single component: a
    /// runtime index expression `i` (`i - 1`) or `end`±k (`n + k - 1`).
    fn dynamic_linear_offset(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        indexing: &IndexingSemantics,
        base: &HirExpr,
    ) -> Result<Value> {
        if indexing.components.len() != 1 {
            return Err(Error::NotLowerable(
                "multi-dimensional indexing of a dynamic array is not supported".to_string(),
            ));
        }
        let component = &indexing.components[0];
        if component_is_colon(component) {
            return Err(Error::NotLowerable(
                "colon indexing of a dynamic array is not supported".to_string(),
            ));
        }
        // `end` [+/- k] -> 0-based offset `n + k - 1`.
        if let Some(offset) = component_end_offset(component) {
            let n = self.array_len_value(context, block, base)?;
            let adj = emit_constant(context, block, offset as f64)?;
            let sum = self.append_binop(context, block, BinOpKind::Add, n, adj)?;
            let one = emit_constant(context, block, 1.0)?;
            return self.append_binop(context, block, BinOpKind::Sub, sum, one);
        }
        // A 1-based scalar index `i` -> 0-based offset `i - 1`.
        if let IndexComponent::Expr(expr) = component {
            let i = self.lower_expr(context, block, expr)?;
            let one = emit_constant(context, block, 1.0)?;
            return self.append_binop(context, block, BinOpKind::Sub, i, one);
        }
        Err(Error::NotLowerable(
            "logical indexing of a dynamic array is not supported".to_string(),
        ))
    }

    /// Compile-time column-major linear offset of a scalar index expression.
    fn static_linear_offset(&self, indexing: &IndexingSemantics, shape: Shape) -> Result<usize> {
        // Linear indexing: a single component indexes the flattened array.
        if indexing.components.len() == 1 {
            let component = &indexing.components[0];
            if component_is_colon(component) {
                return Err(Error::NotLowerable(
                    "colon is only supported in `A(:)`".to_string(),
                ));
            }
            if let Some(offset) = component_end_offset(component) {
                let position = shape.numel() as isize + offset;
                if position <= 0 {
                    return Err(Error::NotLowerable("end index out of range".to_string()));
                }
                return Ok((position - 1) as usize);
            }
            if let IndexComponent::Expr(expr) = component {
                let one_based = self.constant_index(expr)?;
                if one_based == 0 {
                    return Err(Error::NotLowerable("indices are 1-based".to_string()));
                }
                return Ok(one_based - 1);
            }
            return Err(Error::NotLowerable(
                "logical indexing is not supported".to_string(),
            ));
        }

        // Subscript indexing: each component addresses one dimension.
        let mut coords = Vec::with_capacity(indexing.components.len());
        for (axis, component) in indexing.components.iter().enumerate() {
            if component_is_colon(component) {
                return Err(Error::NotLowerable(
                    "colon slices are not supported yet".to_string(),
                ));
            }
            let coord = if let Some(offset) = component_end_offset(component) {
                let dim = match component {
                    IndexComponent::End { dim, .. } => *dim,
                    _ => None,
                }
                .unwrap_or(axis);
                let size = *shape
                    .dims()
                    .get(dim)
                    .ok_or_else(|| Error::NotLowerable("end index out of range".to_string()))?;
                let position = size as isize + offset;
                if position <= 0 {
                    return Err(Error::NotLowerable("end index out of range".to_string()));
                }
                (position - 1) as usize
            } else if let IndexComponent::Expr(expr) = component {
                let one_based = self.constant_index(expr)?;
                if one_based == 0 {
                    return Err(Error::NotLowerable("indices are 1-based".to_string()));
                }
                one_based - 1
            } else {
                return Err(Error::NotLowerable(
                    "logical indexing is not supported".to_string(),
                ));
            };
            coords.push(coord);
        }
        Ok(shape.linear(&coords))
    }

    /// A constant integer index (1-based) from an expression.
    fn constant_index(&self, expr: &HirExpr) -> Result<usize> {
        match &expr.kind {
            HirExprKind::Number(text) => text
                .trim()
                .parse::<usize>()
                .map_err(|_| Error::NotLowerable(format!("bad index `{text}`"))),
            HirExprKind::IntegerLiteral(literal) => Ok(literal.bits() as usize),
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
        call: &HirCall,
    ) -> Result<Value> {
        let name = call_name(&call.callee)
            .ok_or_else(|| Error::NotLowerable("dynamic function call".to_string()))?;
        let builtin = builtins::lookup(&name)
            .ok_or_else(|| Error::NotLowerable(format!("unsupported builtin `{name}`")))?;

        let args: Vec<&HirExpr> = call.args.iter().collect();

        match builtin {
            Builtin::Unary(symbol) => {
                let value = self.lower_expr(context, block, args[0])?;
                self.emit_libm_call(context, block, symbol, &[value])
            }
            Builtin::Binary(symbol) => {
                let l = self.lower_expr(context, block, args[0])?;
                let r = self.lower_expr(context, block, args[1])?;
                self.emit_libm_call(context, block, symbol, &[l, r])
            }
            Builtin::Mod => {
                let l = self.lower_expr(context, block, args[0])?;
                let r = self.lower_expr(context, block, args[1])?;
                self.mod_floor(context, block, l, r)
            }
            Builtin::Sign => {
                let value = self.lower_expr(context, block, args[0])?;
                self.sign(context, block, value)
            }
            Builtin::MinMax(minmax) => match args.len() {
                2 => {
                    let l = self.lower_expr(context, block, args[0])?;
                    let r = self.lower_expr(context, block, args[1])?;
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
            Builtin::Numel => match self.operand_type(args[0]) {
                LocalTy::Array { shape } if shape.is_dynamic() => {
                    self.array_len_value(context, block, args[0])
                }
                LocalTy::Array { shape } => emit_constant(context, block, shape.numel() as f64),
                LocalTy::Scalar => emit_constant(context, block, 1.0),
                LocalTy::Struct { .. } | LocalTy::Dynamic => {
                    Err(Error::NotLowerable("dynamic numel".to_string()))
                }
            },
            Builtin::Length => {
                let max = match self.operand_type(args[0]) {
                    // A dynamic array parameter is treated as a vector: its
                    // length is its runtime element count.
                    LocalTy::Array { shape } if shape.is_dynamic() => {
                        return self.array_len_value(context, block, args[0]);
                    }
                    LocalTy::Array { shape } => shape.dims().iter().copied().max().unwrap_or(1),
                    LocalTy::Scalar => 1,
                    LocalTy::Struct { .. } | LocalTy::Dynamic => {
                        return Err(Error::NotLowerable("dynamic length".to_string()))
                    }
                };
                emit_constant(context, block, max as f64)
            }
            Builtin::Size => {
                let dim = self.dim_arg(call)?.unwrap_or(1);
                let size = match self.operand_type(args[0]) {
                    // `size` of a dynamic-shape array parameter is ambiguous
                    // (row vs column orientation is unknown).
                    LocalTy::Array { shape } if shape.is_dynamic() => {
                        return Err(Error::NotLowerable("dynamic size".to_string()))
                    }
                    LocalTy::Array { shape } if dim >= 1 && dim <= shape.rank() => {
                        shape.dims()[dim - 1]
                    }
                    _ => 1,
                };
                emit_constant(context, block, size as f64)
            }
            // Constructors and reshape/sort always produce arrays; they are handled
            // by the array path and never reach the scalar path.
            Builtin::Fill(_) | Builtin::Eye | Builtin::Reshape | Builtin::Sort => Err(
                Error::NotLowerable("constructor reached scalar path".to_string()),
            ),
        }
    }

    /// The constant dimension argument of a built-in call, when present.
    fn dim_arg(&self, call: &HirCall) -> Result<Option<usize>> {
        if call.args.len() < 2 {
            return Ok(None);
        }
        match &call.args[1].kind {
            HirExprKind::Number(text) => text
                .trim()
                .parse::<usize>()
                .map(Some)
                .map_err(|_| Error::NotLowerable(format!("bad dimension `{text}`"))),
            _ => Err(Error::NotLowerable(
                "dimension argument must be a constant".to_string(),
            )),
        }
    }

    /// Lower a reduction over a scalar (identity), a static array (compile-time
    /// loop), or a dynamic-shape array parameter (a `convmat_*` runtime helper).
    fn reduce_arg(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        operand: &HirExpr,
        reducer: Reducer,
    ) -> Result<Value> {
        match self.operand_type(operand) {
            LocalTy::Scalar => self.lower_expr(context, block, operand),
            LocalTy::Array { shape } if shape.is_dynamic() => {
                let src = self.array_source(context, block, operand)?;
                let n = self.array_len_value(context, block, operand)?;
                let callee = match reducer {
                    Reducer::Add => crate::runtime::SUM,
                    Reducer::Mul => crate::runtime::PROD,
                    Reducer::Min => crate::runtime::MIN,
                    Reducer::Max => crate::runtime::MAX,
                };
                // `convmat_*` reductions return the scalar directly.
                self.emit_libm_call(context, block, callee, &[src, n])
            }
            LocalTy::Array { shape } => {
                let src = self.array_source(context, block, operand)?;
                self.reduce(context, block, reducer, src, shape.numel())
            }
            LocalTy::Struct { .. } | LocalTy::Dynamic => {
                Err(Error::NotLowerable("dynamic reduction".to_string()))
            }
        }
    }

    /// Lower an array-producing expression directly into `dest` (an array cell).
    fn lower_array_expr_into(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        expr: &HirExpr,
        dest: Value,
    ) -> Result<()> {
        match &expr.kind {
            HirExprKind::Tensor(rows) => {
                let rows_n = rows.len();
                let cols_n = rows.first().map_or(0, |row| row.len());
                let shape = Shape::matrix(rows_n, cols_n);
                for (row, elements) in rows.iter().enumerate() {
                    for (col, element) in elements.iter().enumerate() {
                        let value = self.lower_expr(context, block, element)?;
                        let offset = shape.linear(&[row, col]);
                        let index = emit_constant(context, block, offset as f64)?;
                        emit_store(context, block, dest, index, value);
                    }
                }
                Ok(())
            }
            HirExprKind::Call(call) => {
                let name = call_name(&call.callee)
                    .ok_or_else(|| Error::NotLowerable("dynamic function call".to_string()))?;
                let builtin = builtins::lookup(&name)
                    .ok_or_else(|| Error::NotLowerable(format!("unsupported builtin `{name}`")))?;
                match builtin {
                    Builtin::Unary(symbol) => {
                        let [arg] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "expected a single array argument".to_string(),
                            ));
                        };
                        let src = self.array_source(context, block, arg)?;
                        let n = self.array_len(arg)?;
                        self.map_unary(context, block, symbol, src, dest, n)
                    }
                    Builtin::Reduce(op) => {
                        let [arg, _] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "expected array + dimension".to_string(),
                            ));
                        };
                        let dim = self.dim_arg(call)?.unwrap_or(1);
                        let src = self.array_source(context, block, arg)?;
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
                        let [arg, _] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "expected array + dimension".to_string(),
                            ));
                        };
                        let dim = self.dim_arg(call)?.unwrap_or(1);
                        let src = self.array_source(context, block, arg)?;
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
                        let [arg] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "expected a single array argument".to_string(),
                            ));
                        };
                        let shape = match self.operand_type(arg) {
                            LocalTy::Array { shape } if shape.is_dynamic() => {
                                return Err(Error::NotLowerable("dynamic size".to_string()))
                            }
                            LocalTy::Array { shape } => shape,
                            LocalTy::Scalar => Shape::matrix(1, 1),
                            LocalTy::Struct { .. } | LocalTy::Dynamic => {
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
                        let [arg, ..] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "reshape expects an array argument".to_string(),
                            ));
                        };
                        let src = self.array_source(context, block, arg)?;
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
                    Builtin::Sort => {
                        // Sort a vector ascending into `dest` (a wrapped runtime
                        // helper). Matrix (rank-2 with both dims > 1) column sort
                        // is not supported yet.
                        let [arg] = call.args.as_slice() else {
                            return Err(Error::NotLowerable(
                                "sort expects a single array argument".to_string(),
                            ));
                        };
                        let src = self.array_source(context, block, arg)?;
                        let shape = self.array_shape(arg)?;
                        if shape.is_dynamic() {
                            return Err(Error::NotLowerable(
                                "sort of a dynamic array is not supported".to_string(),
                            ));
                        }
                        if shape.rank() != 2 || (shape.dims()[0] != 1 && shape.dims()[1] != 1) {
                            return Err(Error::NotLowerable(
                                "sort is only supported for vectors".to_string(),
                            ));
                        }
                        let n = emit_constant(context, block, shape.numel() as f64)?;
                        self.emit_extern_call(
                            context,
                            block,
                            crate::runtime::SORT,
                            &[dest, src, n],
                        );
                        Ok(())
                    }
                    _ => Err(Error::NotLowerable(format!(
                        "array result from `{name}` is not supported"
                    ))),
                }
            }
            HirExprKind::Unary(op, operand) => {
                self.lower_array_unary(context, block, op, operand, dest)
            }
            HirExprKind::Binary(lhs, op, rhs) => {
                self.lower_array_binary(context, block, lhs, op, rhs, dest)
            }
            HirExprKind::Index { .. } => {
                // `A(:)` flattens to a column vector (linear order is preserved).
                let HirExprKind::Index(base, indexing) = &expr.kind else {
                    unreachable!();
                };
                let _ = indexing;
                let src = self.array_source(context, block, base)?;
                if self.array_shape(base)?.is_dynamic() {
                    // Dynamic source: copy elementwise via a runtime helper.
                    let n = self.array_len_value(context, block, base)?;
                    self.emit_extern_call(context, block, crate::runtime::COPY, &[dest, src, n]);
                    return Ok(());
                }
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

    /// The cell backing an array-typed expression (materializing inline array
    /// literals into a temporary cell when necessary).
    fn array_source(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        expr: &HirExpr,
    ) -> Result<Value> {
        match &expr.kind {
            HirExprKind::Binding(id) => self
                .locals
                .get(id)
                .copied()
                .ok_or_else(|| Error::Backend(format!("binding {id:?} has no cell"))),
            // `A(:)` on a dynamic (vector) array is the identity: the same data
            // pointer (linear order is unchanged), so it needs no temporary.
            HirExprKind::Index(base, _) if matches!(self.operand_type(base), LocalTy::Array { shape } if shape.is_dynamic()) => {
                self.array_source(context, block, base)
            }
            _ => {
                let LocalTy::Array { shape } = self.operand_type(expr) else {
                    return Err(Error::NotLowerable(
                        "array source must be an array".to_string(),
                    ));
                };
                if shape.is_dynamic() {
                    return Err(Error::NotLowerable(
                        "dynamic array expression (not a parameter) is not supported".to_string(),
                    ));
                }
                let temp = self.alloca_array(context, block, shape)?;
                self.lower_array_expr_into(context, block, expr, temp)?;
                Ok(temp)
            }
        }
    }

    /// The flattened element count of an array-typed expression. Dynamic-shape
    /// arrays have no compile-time count and are rejected here (use
    /// [`Self::array_len_value`] for a runtime length).
    fn array_len(&self, expr: &HirExpr) -> Result<usize> {
        match self.operand_type(expr) {
            LocalTy::Array { shape } if shape.is_dynamic() => Err(Error::NotLowerable(
                "compile-time array length required here".to_string(),
            )),
            LocalTy::Array { shape } => Ok(shape.numel()),
            _ => Err(Error::NotLowerable("expected an array operand".to_string())),
        }
    }

    /// The element count of an array-typed expression as an IR value: a
    /// compile-time constant for static shapes, or a runtime length for a
    /// dynamic-shape expression (see [`Self::dynamic_array_len`]).
    fn array_len_value(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        expr: &HirExpr,
    ) -> Result<Value> {
        match self.operand_type(expr) {
            LocalTy::Array { shape } if shape.is_dynamic() => self.dynamic_array_len(expr),
            LocalTy::Array { shape } => emit_constant(context, block, shape.numel() as f64),
            _ => Err(Error::NotLowerable("expected an array operand".to_string())),
        }
    }

    /// The runtime element count of a dynamic-shape array expression, by
    /// walking to the array operand that determines its length. Elementwise
    /// operators (`scalar op A`, `-A`) and `A(:)` preserve the operand's length.
    fn dynamic_array_len(&self, expr: &HirExpr) -> Result<Value> {
        match &expr.kind {
            HirExprKind::Binding(id) => self
                .array_lens
                .get(id)
                .copied()
                .ok_or_else(|| Error::Backend(format!("no length for array {id:?}"))),
            HirExprKind::Index(base, _) | HirExprKind::Unary(_, base) => {
                self.dynamic_array_len(base)
            }
            HirExprKind::Binary(lhs, _, rhs) => {
                if matches!(self.operand_type(lhs), LocalTy::Array { .. }) {
                    self.dynamic_array_len(lhs)
                } else {
                    self.dynamic_array_len(rhs)
                }
            }
            _ => Err(Error::NotLowerable(
                "dynamic array length of this expression is not supported".to_string(),
            )),
        }
    }

    /// The static shape of an array-typed expression.
    fn array_shape(&self, expr: &HirExpr) -> Result<Shape> {
        match self.operand_type(expr) {
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

        let init = emit_constant(context, block, reducer.init())?;
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
            OperatorKind::ShortCircuitAnd | OperatorKind::ElementwiseAnd => {
                self.logical_and(context, block, l, r)
            }
            OperatorKind::ShortCircuitOr | OperatorKind::ElementwiseOr => {
                self.logical_or(context, block, l, r)
            }
            // Scalar power: `.^` and scalar `^` both lower to `libm::pow`.
            OperatorKind::ElementwisePower | OperatorKind::MatrixPower => {
                self.emit_libm_call(context, block, "pow", &[l, r])
            }
            other => Err(Error::NotLowerable(format!("binary operator {other:?}"))),
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

    /// Emit an `if`/`elseif`/`else` chain.
    fn lower_if(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        cond: &HirExpr,
        then_body: &HirBlock,
        elseif_blocks: &[(HirExpr, HirBlock)],
        else_body: &Option<HirBlock>,
    ) -> Result<()> {
        let cond = self.lower_expr(context, block, cond)?;
        let cond = self.truthy(context, block, cond)?;

        let if_op = IfOp::new(context, cond);
        append(context, block, &if_op);
        self.fill_block(context, if_op.then_region(context), then_body)?;
        self.lower_if_else(
            context,
            if_op.else_region(context),
            elseif_blocks,
            else_body,
        )
    }

    fn lower_if_else(
        &self,
        context: &mut Context,
        region: Ptr<Region>,
        elseif_blocks: &[(HirExpr, HirBlock)],
        else_body: &Option<HirBlock>,
    ) -> Result<()> {
        let block = BasicBlock::new(context, None, vec![]);
        block.insert_at_front(region, context);
        match elseif_blocks.split_first() {
            None => {
                if let Some(else_body) = else_body {
                    self.lower_block(context, block, else_body)?;
                }
                emit_yield(context, block);
            }
            Some(((cond, body), rest)) => {
                let cond = self.lower_expr(context, block, cond)?;
                let cond = self.truthy(context, block, cond)?;
                let if_op = IfOp::new(context, cond);
                append(context, block, &if_op);
                self.fill_block(context, if_op.then_region(context), body)?;
                self.lower_if_else(context, if_op.else_region(context), rest, else_body)?;
                emit_yield(context, block);
            }
        }
        Ok(())
    }

    /// Emit a `while` loop.
    fn lower_while(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        cond: &HirExpr,
        body: &HirBlock,
    ) -> Result<()> {
        let while_op = WhileOp::new(context);
        append(context, block, &while_op);

        let before = BasicBlock::new(context, None, vec![]);
        before.insert_at_front(while_op.before_region(context), context);
        let cond = self.lower_expr(context, before, cond)?;
        let cond = self.truthy(context, before, cond)?;
        emit_condition(context, before, cond);

        let after = BasicBlock::new(context, None, vec![]);
        after.insert_at_front(while_op.after_region(context), context);
        self.lower_block(context, after, body)?;
        emit_yield(context, after);
        Ok(())
    }

    /// Emit a `for` loop from a colon range.
    fn lower_for(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        binding: BindingId,
        range: &HirExpr,
        body: &HirBlock,
    ) -> Result<()> {
        let (start, step, end) = match &range.kind {
            HirExprKind::Range(start, step, end) => (start, step.as_deref(), end),
            _ => {
                return Err(Error::NotLowerable(
                    "unsupported for-loop iterable".to_string(),
                ))
            }
        };

        let start = self.lower_expr(context, block, start)?;
        let step = match step {
            Some(step) => self.lower_expr(context, block, step)?,
            None => emit_constant(context, block, 1.0)?,
        };
        let end = self.lower_expr(context, block, end)?;

        let cell = self
            .locals
            .get(&binding)
            .copied()
            .ok_or_else(|| Error::Backend(format!("no cell for loop binding {binding:?}")))?;

        // Seed the cell with `start` so the first read of the loop variable sees
        // it (the `range_for` body then refreshes it every iteration).
        let zero = emit_constant(context, block, 0.0)?;
        emit_store(context, block, cell, zero, start);

        let for_op = RangeForOp::new(context, start, end, step);
        append(context, block, &for_op);

        // Body region: the block argument is the induction variable `iv`; store
        // it into the loop cell at the top of every iteration, then lower the
        // body statements. `break`/`continue` keep their natural C semantics
        // because the whole loop is emitted as a C `for`.
        let body_block = BasicBlock::new(context, None, vec![self.f64_ty]);
        body_block.insert_at_front(for_op.body_region(context), context);
        let iv = body_block.deref(context).get_argument(0);
        let zero = emit_constant(context, body_block, 0.0)?;
        emit_store(context, body_block, cell, zero, iv);
        self.lower_block(context, body_block, body)?;
        emit_yield(context, body_block);
        Ok(())
    }

    /// Emit a `switch`/`case`/`otherwise` construct as a nested `if`/`else`
    /// chain (MATLAB switch cases are exclusive and do not fall through).
    fn lower_switch(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        discr: &HirExpr,
        cases: &[(HirExpr, HirBlock)],
        otherwise: &Option<HirBlock>,
    ) -> Result<()> {
        let discr = self.lower_expr(context, block, discr)?;

        let mut case_values = Vec::with_capacity(cases.len());
        for (case, _) in cases {
            case_values.push(self.lower_expr(context, block, case)?);
        }

        self.emit_switch(context, block, discr, &case_values, cases, otherwise)
    }

    /// Recursively emit the nested `if`/`else` chain for a switch into `block`.
    fn emit_switch(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        discr: Value,
        case_values: &[Value],
        cases: &[(HirExpr, HirBlock)],
        otherwise: &Option<HirBlock>,
    ) -> Result<()> {
        match case_values.split_first() {
            None => {
                // Innermost `else`: the `otherwise` body.
                if let Some(otherwise) = otherwise {
                    self.lower_block(context, block, otherwise)?;
                }
                Ok(())
            }
            Some((first, rest)) => {
                let first_block = &cases[0].1;
                let cmp = self.cmpf(context, block, CmpKind::Eq, discr, *first)?;
                let if_op = IfOp::new(context, cmp);
                append(context, block, &if_op);
                self.fill_block(context, if_op.then_region(context), first_block)?;
                self.fill_switch_else(
                    context,
                    if_op.else_region(context),
                    discr,
                    rest,
                    &cases[1..],
                    otherwise,
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
        cases: &[(HirExpr, HirBlock)],
        otherwise: &Option<HirBlock>,
    ) -> Result<()> {
        let block = BasicBlock::new(context, None, vec![]);
        block.insert_at_front(region, context);
        self.emit_switch(context, block, discr, case_values, cases, otherwise)?;
        emit_yield(context, block);
        Ok(())
    }

    /// Fill a single-block region with the statements of `hir_block`, ending in
    /// a `yield`.
    fn fill_block(
        &self,
        context: &mut Context,
        region: Ptr<Region>,
        hir_block: &HirBlock,
    ) -> Result<()> {
        let block = BasicBlock::new(context, None, vec![]);
        block.insert_at_front(region, context);
        self.lower_block(context, block, hir_block)?;
        emit_yield(context, block);
        Ok(())
    }

    /// Lower a unary operator over an array (elementwise or 2-D transpose).
    fn lower_array_unary(
        &self,
        context: &mut Context,
        block: Ptr<BasicBlock>,
        op: &OperatorKind,
        operand: &HirExpr,
        dest: Value,
    ) -> Result<()> {
        match op {
            OperatorKind::Transpose | OperatorKind::ConjugateTranspose => {
                let src = self.array_source(context, block, operand)?;
                let shape = self.array_shape(operand)?;
                if shape.is_dynamic() {
                    return Err(Error::NotLowerable(
                        "transpose of a dynamic array is not supported".to_string(),
                    ));
                }
                if shape.rank() != 2 {
                    return Err(Error::NotLowerable(
                        "only 2-D transpose is supported".to_string(),
                    ));
                }
                self.transpose(context, block, src, dest, shape.dims()[0], shape.dims()[1])
            }
            OperatorKind::UnaryMinus | OperatorKind::UnaryPlus | OperatorKind::Not => {
                let src = self.array_source(context, block, operand)?;
                if matches!(self.operand_type(operand), LocalTy::Array { shape } if shape.is_dynamic())
                {
                    let n = self.array_len_value(context, block, operand)?;
                    let callee = match op {
                        OperatorKind::UnaryMinus => crate::runtime::NEG,
                        // `+A` is the identity; a copy into the out-buffer suffices.
                        _ => crate::runtime::COPY,
                    };
                    self.emit_extern_call(context, block, callee, &[dest, src, n]);
                    return Ok(());
                }
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
        lhs: &HirExpr,
        op: &OperatorKind,
        rhs: &HirExpr,
        dest: Value,
    ) -> Result<()> {
        // Matrix power is not elementwise: route it to its own helper.
        if *op == OperatorKind::MatrixPower {
            return self.lower_matrix_power(context, block, lhs, rhs, dest);
        }
        // Dynamic-shape operands: scalar broadcast via `convmat_scale`, or two
        // equal-length dynamic arrays via `convmat_add`/`sub`/`ewmul`.
        let dyn_lhs =
            matches!(self.operand_type(lhs), LocalTy::Array { shape } if shape.is_dynamic());
        let dyn_rhs =
            matches!(self.operand_type(rhs), LocalTy::Array { shape } if shape.is_dynamic());
        if dyn_lhs || dyn_rhs {
            if dyn_lhs && dyn_rhs {
                // Two dynamic arrays, elementwise, same length.
                let callee = match op {
                    OperatorKind::Add => crate::runtime::ADD,
                    OperatorKind::Subtract => crate::runtime::SUB,
                    OperatorKind::ElementwiseMultiply => crate::runtime::EWMUL,
                    _ => {
                        return Err(Error::NotLowerable(
                            "unsupported elementwise operator for dynamic arrays".to_string(),
                        ))
                    }
                };
                let a = self.array_source(context, block, lhs)?;
                let b = self.array_source(context, block, rhs)?;
                let n = self.array_len_value(context, block, lhs)?;
                self.emit_extern_call(context, block, callee, &[dest, a, b, n]);
                return Ok(());
            }
            // One dynamic array + one scalar: broadcast via `convmat_scale` (`*`).
            if *op != OperatorKind::ElementwiseMultiply && *op != OperatorKind::MatrixMultiply {
                return Err(Error::NotLowerable(
                    "only `.*`/`*` scalar broadcast is supported for dynamic arrays".to_string(),
                ));
            }
            let (array, scalar) = if dyn_rhs { (rhs, lhs) } else { (lhs, rhs) };
            if !matches!(self.operand_type(scalar), LocalTy::Scalar) {
                return Err(Error::NotLowerable(
                    "only scalar broadcast is supported for dynamic arrays".to_string(),
                ));
            }
            let src = self.array_source(context, block, array)?;
            let n = self.array_len_value(context, block, array)?;
            let k = self.lower_expr(context, block, scalar)?;
            self.emit_extern_call(context, block, crate::runtime::SCALE, &[dest, src, n, k]);
            return Ok(());
        }
        match (self.operand_type(lhs), self.operand_type(rhs)) {
            (LocalTy::Scalar, LocalTy::Array { shape }) => {
                let src = self.array_source(context, block, rhs)?;
                self.map_binary_scalar(context, block, op, lhs, src, dest, shape.numel(), true)
            }
            (LocalTy::Array { shape }, LocalTy::Scalar) => {
                let src = self.array_source(context, block, lhs)?;
                self.map_binary_scalar(context, block, op, rhs, src, dest, shape.numel(), false)
            }
            (LocalTy::Array { shape: lhs_shape }, LocalTy::Array { shape: rhs_shape }) => {
                let ls = self.array_source(context, block, lhs)?;
                let rs = self.array_source(context, block, rhs)?;
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
        scalar: &HirExpr,
        src: Value,
        dest: Value,
        n: usize,
        scalar_on_left: bool,
    ) -> Result<()> {
        self.for_loop(context, block, n, |this, ctx, body, i| {
            let scalar = this.lower_expr(ctx, body, scalar)?;
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
        lhs: &HirExpr,
        rhs: &HirExpr,
        dest: Value,
    ) -> Result<()> {
        let LocalTy::Array { shape } = self.operand_type(lhs) else {
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
        let src = self.array_source(context, block, lhs)?;
        let m = emit_constant(context, block, shape.dims()[0] as f64)?;
        let k = emit_constant(context, block, exp as f64)?;
        self.emit_extern_call(context, block, crate::runtime::MPOWER, &[dest, src, m, k]);
        Ok(())
    }

    /// A compile-time integer exponent from a scalar expression, or `None` when
    /// the operand is not a known integer constant.
    fn const_int_exponent(&self, expr: &HirExpr) -> Result<Option<i64>> {
        match &expr.kind {
            HirExprKind::Number(text) => text.trim().parse::<i64>().map(Some).map_err(|_| {
                Error::NotLowerable(format!(
                    "matrix power exponent must be a non-negative integer, got `{text}`"
                ))
            }),
            HirExprKind::IntegerLiteral(literal) => Ok(Some(literal.bits() as i64)),
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

    /// The constant `(rows, cols)` of a constructor call (`zeros/ones/eye`).
    fn constructor_dims(&self, call: &HirCall) -> Result<(usize, usize)> {
        let rows = self
            .constant_arg(call, 0)?
            .ok_or_else(|| Error::NotLowerable("constructor dims must be constant".to_string()))?;
        let cols = self.constant_arg(call, 1)?.unwrap_or(rows);
        Ok((rows, cols))
    }

    /// The constant `usize` value of the `index`-th argument, if present.
    fn constant_arg(&self, call: &HirCall, index: usize) -> Result<Option<usize>> {
        match call.args.get(index) {
            Some(expr) => match &expr.kind {
                HirExprKind::Number(text) => text
                    .trim()
                    .parse::<usize>()
                    .map(Some)
                    .map_err(|_| Error::NotLowerable(format!("bad constant `{text}`"))),
                HirExprKind::IntegerLiteral(literal) => Ok(Some(literal.bits() as usize)),
                _ => Err(Error::NotLowerable(
                    "constructor dims must be constant".to_string(),
                )),
            },
            None => Ok(None),
        }
    }

    /// Free every heap-allocated cell (in reverse allocation order).
    fn emit_heap_frees(&self, context: &mut Context, block: Ptr<BasicBlock>) {
        for cell in self.heap_cells.iter().rev() {
            let op = DeleteOp::new(context, *cell);
            append(context, block, &op);
        }
    }

    /// Load the value outputs (scalar or struct, in ABI order) and emit
    /// `matlab.return`.
    fn emit_return_values(&self, context: &mut Context, block: Ptr<BasicBlock>) -> Result<()> {
        let mut values = Vec::new();
        for (output, ty) in &self.return_outputs {
            let cell = self
                .locals
                .get(output)
                .copied()
                .ok_or_else(|| Error::Backend(format!("no cell for output {output:?}")))?;
            if ty.deref(context).is::<StructType>() {
                // A struct cell is returned by value directly (C struct copy).
                values.push(cell);
            } else {
                values.push(self.load_local(context, block, cell)?);
            }
        }
        for cell in &self.varargout_cells {
            values.push(self.load_local(context, block, *cell)?);
        }
        emit_return(context, block, values);
        Ok(())
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

/// Build a `matlab.struct` type from its field name/type list. Only scalar
/// (`f64`) fields are supported for now; array and nested-struct fields are
/// deferred.
fn struct_type(context: &mut Context, fields: &[(String, LocalTy)]) -> Result<TypeHandle> {
    let field_tys = fields
        .iter()
        .map(|(name, ty)| {
            let f_ty: TypeHandle = match ty {
                LocalTy::Scalar => FP64Type::get(context).into(),
                LocalTy::Array { .. } | LocalTy::Struct { .. } | LocalTy::Dynamic => {
                    return Err(Error::NotLowerable(
                        "only scalar struct fields are supported yet".to_string(),
                    ))
                }
            };
            Ok((name.clone(), f_ty))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StructType::get(context, field_tys).into())
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
