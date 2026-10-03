//! The `matlab` -> `emitc` lowering pass.
//!
//! Walks the `matlab`-dialect module and produces `emitc`-dialect functions.
//! This is where MATLAB-level semantics become C-level ones: the function ABI
//! (scalar results vs. caller-allocated array out-params) is resolved, every
//! array local gets a concrete C name, and each op is rewritten into its C
//! primitive. The result is a near-1:1 mirror of the `matlab` IR that the C
//! emitter can print directly.

use std::collections::{BTreeSet, HashMap};

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::{OneRegionInterface, SingleBlockRegionInterface, SymbolOpInterface},
        ops::{FuncOp, ModuleOp},
        type_interfaces::FunctionTypeInterface,
        types::FunctionType,
    },
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::inserter::Inserter,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    region::Region,
    value::Value,
};

use crate::{
    dialects::{emitc, matlab},
    error::{Error, Result},
};

/// Translate a `matlab`-dialect module into a builtin `ModuleOp` whose bodies
/// are C-level `emitc` ops. Containers (`module`/`func`) are pliron builtins;
/// only the C-level body ops and directives come from the `emitc` dialect.
pub fn lower_module(context: &mut Context, module: &ModuleOp) -> Result<ModuleOp> {
    let emitc_module = ModuleOp::new(context, Identifier::try_from("convmat").unwrap());

    // Emit only the runtime helpers this module actually references, and emit
    // them first so the wrapped-operator calls in the functions resolve. An
    // unknown helper name is a bug (typo) and fails loudly rather than emitting
    // an undefined symbol.
    for name in collect_used_helpers(context, module) {
        let source = crate::runtime::helper_source(&name)
            .ok_or_else(|| Error::Backend(format!("unknown runtime helper `{name}`")))?;
        let verbatim = emitc::VerbatimOp::new(context, source);
        emitc_module.append_operation(context, verbatim.get_operation(), 0);
    }

    // Collect the `matlab` functions first (immutable pass), then lower each
    // (mutable pass) so the module borrow and the context borrow never alias.
    let mut matlab_funcs = Vec::new();
    if let Some(block) = module.get_region(context).deref(context).get_entry_block() {
        for op in block.deref(context).iter(context) {
            if let Some(func) = Operation::get_op::<FuncOp>(op, context) {
                matlab_funcs.push(func);
            }
        }
    }
    for func in &matlab_funcs {
        let efunc = lower_func(context, func)?;
        emitc_module.append_operation(context, efunc.get_operation(), 0);
    }
    Ok(emitc_module)
}

/// Translate a single `matlab` function into a builtin `FuncOp` whose body is
/// C-level `emitc` ops.
fn lower_func(context: &mut Context, func: &FuncOp) -> Result<FuncOp> {
    let name = func.get_symbol_name(context).as_ref().to_string();
    let (arg_types, res_types) = {
        let fn_ty = func.get_type(context).deref(context);
        let ft = fn_ty
            .downcast_ref::<FunctionType>()
            .expect("func carries a function type");
        (ft.arg_types(), ft.res_types())
    };

    let name_id = Identifier::try_from(name.as_str())
        .map_err(|e| Error::Backend(format!("bad function name `{name}`: {e}")))?;
    let fn_ty = FunctionType::get(context, arg_types.clone(), res_types);
    let efunc = FuncOp::new(context, name_id, fn_ty);
    let entry = efunc.get_entry_block(context);

    let mut lowerer = Lowerer {
        values: HashMap::new(),
        decl_counter: 0,
    };
    let m_entry = func.get_entry_block(context);
    for i in 0..arg_types.len() {
        lowerer.values.insert(
            m_entry.deref(context).get_argument(i),
            entry.deref(context).get_argument(i),
        );
    }
    lowerer.lower_block(context, m_entry, entry)?;
    Ok(efunc)
}

struct Lowerer {
    values: HashMap<Value, Value>,
    decl_counter: usize,
}

impl Lowerer {
    fn lower_block(
        &mut self,
        context: &mut Context,
        src: Ptr<BasicBlock>,
        dst: Ptr<BasicBlock>,
    ) -> Result<()> {
        let ops: Vec<Ptr<Operation>> = src.deref(context).iter(context).collect();
        for op in ops {
            self.lower_op(context, op, dst)?;
        }
        Ok(())
    }

    fn lower_op(
        &mut self,
        context: &mut Context,
        op: Ptr<Operation>,
        dst: Ptr<BasicBlock>,
    ) -> Result<()> {
        let m = op.deref(context).get_num_operands();

        if let Some(c) = Operation::get_op::<matlab::ConstantOp>(op, context) {
            let e = emitc::LiteralOp::new(context, c.value(context));
            self.map_result(context, op, &e, 0);
            append(context, dst, &e);
        } else if let Some(c) = Operation::get_op::<matlab::BinOp>(op, context) {
            let e = emitc::BinOp::new(
                context,
                c.kind(context),
                self.opd(context, op, 0),
                self.opd(context, op, 1),
            );
            self.map_result(context, op, &e, 0);
            append(context, dst, &e);
        } else if let Some(c) = Operation::get_op::<matlab::CmpOp>(op, context) {
            let e = emitc::CmpOp::new(
                context,
                c.kind(context),
                self.opd(context, op, 0),
                self.opd(context, op, 1),
            );
            self.map_result(context, op, &e, 0);
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::SelectOp>(op, context) {
            let e = emitc::TernaryOp::new(
                context,
                self.opd(context, op, 0),
                self.opd(context, op, 1),
                self.opd(context, op, 2),
            );
            self.map_result(context, op, &e, 0);
            append(context, dst, &e);
        } else if let Some(c) = Operation::get_op::<matlab::CallOp>(op, context) {
            let args: Vec<Value> = (0..m).map(|i| self.opd(context, op, i)).collect();
            let e = emitc::CallOp::new(context, &c.callee(context), args);
            self.map_result(context, op, &e, 0);
            append(context, dst, &e);
        } else if let Some(c) = Operation::get_op::<matlab::CallVoidOp>(op, context) {
            let args: Vec<Value> = (0..m).map(|i| self.opd(context, op, i)).collect();
            let e = emitc::CallVoidOp::new(context, &c.callee(context), args);
            append(context, dst, &e);
        } else if let Some(c) = Operation::get_op::<matlab::AllocaOp>(op, context) {
            let array_ty = op.deref(context).get_result(0).get_type(context);
            let name = format!("a{}", self.decl_counter);
            self.decl_counter += 1;
            let e = if c.is_heap(context) {
                emitc::DeclareOp::new_heap(context, &name, array_ty)
            } else if c.is_static(context) {
                emitc::DeclareOp::new_static(context, &name, array_ty)
            } else {
                emitc::DeclareOp::new(context, &name, array_ty)
            };
            self.map_result(context, op, &e, 0);
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::LoadOp>(op, context) {
            let e = emitc::LoadOp::new(context, self.opd(context, op, 0), self.opd(context, op, 1));
            self.map_result(context, op, &e, 0);
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::StoreOp>(op, context) {
            let e = emitc::AssignOp::new(
                context,
                self.opd(context, op, 0),
                self.opd(context, op, 1),
                self.opd(context, op, 2),
            );
            append(context, dst, &e);
        } else if let Some(g) = Operation::get_op::<matlab::StructGetOp>(op, context) {
            let e = emitc::StructGetOp::new(context, self.opd(context, op, 0), &g.field(context));
            self.map_result(context, op, &e, 0);
            append(context, dst, &e);
        } else if let Some(s) = Operation::get_op::<matlab::StructSetOp>(op, context) {
            let e = emitc::StructSetOp::new(
                context,
                self.opd(context, op, 0),
                &s.field(context),
                self.opd(context, op, 1),
            );
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::StructCopyOp>(op, context) {
            let e = emitc::StructCopyOp::new(
                context,
                self.opd(context, op, 0),
                self.opd(context, op, 1),
            );
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::DeleteOp>(op, context) {
            let e = emitc::DeleteOp::new(context, self.opd(context, op, 0));
            append(context, dst, &e);
        } else if let Some(c) = Operation::get_op::<matlab::IfOp>(op, context) {
            let e = emitc::IfOp::new(context, self.opd(context, op, 0));
            append(context, dst, &e);
            self.lower_region(context, c.then_region(context), e.then_region(context))?;
            self.lower_region(context, c.else_region(context), e.else_region(context))?;
        } else if let Some(c) = Operation::get_op::<matlab::WhileOp>(op, context) {
            let e = emitc::WhileOp::new(context);
            append(context, dst, &e);
            self.lower_region(context, c.before_region(context), e.before_region(context))?;
            self.lower_region(context, c.after_region(context), e.after_region(context))?;
        } else if let Some(c) = Operation::get_op::<matlab::ForOp>(op, context) {
            let e = emitc::ForOp::new(
                context,
                self.opd(context, op, 0),
                self.opd(context, op, 1),
                self.opd(context, op, 2),
            );
            append(context, dst, &e);
            self.lower_region(context, c.body_region(context), e.body_region(context))?;
        } else if let Some(c) = Operation::get_op::<matlab::RangeForOp>(op, context) {
            let e = emitc::RangeForOp::new(
                context,
                self.opd(context, op, 0),
                self.opd(context, op, 1),
                self.opd(context, op, 2),
            );
            append(context, dst, &e);
            self.lower_region(context, c.body_region(context), e.body_region(context))?;
        } else if let Some(_c) = Operation::get_op::<matlab::ConditionOp>(op, context) {
            let e = emitc::ConditionOp::new(context, self.opd(context, op, 0));
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::YieldOp>(op, context) {
            let e = emitc::YieldOp::new(context);
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::BreakOp>(op, context) {
            let e = emitc::BreakOp::new(context);
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::ContinueOp>(op, context) {
            let e = emitc::ContinueOp::new(context);
            append(context, dst, &e);
        } else if let Some(_c) = Operation::get_op::<matlab::ReturnOp>(op, context) {
            let values: Vec<Value> = (0..m).map(|i| self.opd(context, op, i)).collect();
            let e = emitc::ReturnOp::new(context, values);
            append(context, dst, &e);
        } else {
            return Err(Error::Backend(format!(
                "unexpected op in lowering: {}",
                Operation::get_opid(op, context)
            )));
        }

        Ok(())
    }

    /// The emitc value corresponding to the `i`-th operand of `op`.
    fn opd(&self, context: &Context, op: Ptr<Operation>, i: usize) -> Value {
        let v = op.deref(context).get_operand(i);
        *self.values.get(&v).unwrap_or(&v)
    }

    fn lower_region(
        &mut self,
        context: &mut Context,
        src: Ptr<Region>,
        dst: Ptr<Region>,
    ) -> Result<()> {
        // Copy the source block's argument types, then map args 1:1.
        let src_block = src
            .deref(context)
            .get_entry_block()
            .expect("single-block region");
        let arg_types: Vec<TypeHandle> = src_block
            .deref(context)
            .arguments()
            .map(|v| v.get_type(context))
            .collect();
        let dst_block = BasicBlock::new(context, None, arg_types);
        dst_block.insert_at_front(dst, context);

        for (i, arg) in src_block.deref(context).arguments().enumerate() {
            self.values
                .insert(arg, dst_block.deref(context).get_argument(i));
        }
        self.lower_block(context, src_block, dst_block)?;
        Ok(())
    }

    fn map_result(&mut self, context: &Context, src: Ptr<Operation>, dst: &dyn Op, idx: usize) {
        let e = dst.get_operation();
        self.values.insert(
            src.deref(context).get_result(idx),
            e.deref(context).get_result(idx),
        );
    }
}

/// Append an op to the end of `block`.
fn append(context: &Context, block: Ptr<BasicBlock>, op: &dyn Op) {
    pliron::irbuild::inserter::IRInserter::<pliron::irbuild::listener::DummyListener>::new_at_block_end(block)
        .append_op(context, op);
}

/// Collect the runtime helper names referenced by `matlab.call_void` ops
/// anywhere in the module (including nested `if`/`while`/`for` regions).
fn collect_used_helpers(context: &Context, module: &ModuleOp) -> BTreeSet<String> {
    let mut used = BTreeSet::new();
    if let Some(block) = module.get_region(context).deref(context).get_entry_block() {
        for op in block.deref(context).iter(context) {
            collect_helpers_from_op(context, op, &mut used);
        }
    }
    used
}

fn collect_helpers_from_op(context: &Context, op: Ptr<Operation>, used: &mut BTreeSet<String>) {
    if let Some(call) = Operation::get_op::<matlab::CallVoidOp>(op, context) {
        used.insert(call.callee(context));
    }
    for region in op.deref(context).regions() {
        if let Some(block) = region.deref(context).get_entry_block() {
            for nested in block.deref(context).iter(context) {
                collect_helpers_from_op(context, nested, used);
            }
        }
    }
}
