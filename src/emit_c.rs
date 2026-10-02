//! The C emitter: pretty-print the `emitc` dialect as C++ source.
//!
//! The output is C++-style source (`double`/`void`/`std::tuple` return types,
//! array out-params as `double name[N]`, `bool` for comparisons, and C-style
//! `if`/`while`/`for`). No optimization is applied; every SSA value is assigned
//! a fresh name.

use std::collections::HashMap;

use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    region::Region,
    value::Value,
};

use crate::{
    dialects::{
        emitc,
        matlab::{ArrayType, BinOpKind, CmpKind},
    },
    error::Result,
};

/// Emit C++ source for a list of `emitc` functions.
pub fn emit(context: &Context, functions: &[emitc::FuncOp]) -> Result<String> {
    let mut emitter = Emitter {
        context,
        names: HashMap::new(),
        counter: 0,
        out: String::new(),
    };
    emitter
        .out
        .push_str("#include <cmath>\n#include <cstdint>\n#include <tuple>\n\n");
    for func in functions {
        emitter.emit_func(func);
    }
    Ok(emitter.out)
}

struct Emitter<'a> {
    context: &'a Context,
    names: HashMap<Value, String>,
    counter: usize,
    out: String,
}

impl<'a> Emitter<'a> {
    fn fresh(&mut self) -> String {
        self.counter += 1;
        format!("v{}", self.counter)
    }

    fn expr(&self, value: Value) -> String {
        self.names
            .get(&value)
            .cloned()
            .unwrap_or_else(|| "<unresolved>".to_string())
    }

    fn emit_func(&mut self, func: &emitc::FuncOp) {
        let name = func.name(self.context);
        let n_results = func.n_results(self.context);
        let entry = func
            .body_region(self.context)
            .deref(self.context)
            .get_entry_block()
            .expect("func has an entry block");

        let args: Vec<Value> = entry.deref(self.context).arguments().collect();
        let mut params = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let pname = format!("v{}", i + 1);
            self.names.insert(*arg, pname.clone());
            params.push(self.param_decl(&pname, arg.get_type(self.context)));
        }

        let ret = match n_results {
            0 => "void".to_string(),
            1 => "double".to_string(),
            n => format!("std::tuple<{}>", vec!["double"; n as usize].join(", ")),
        };

        self.out
            .push_str(&format!("{ret} {name}({}", params.join(", ")));
        self.out.push_str(") {\n");
        self.emit_block(entry, 1);
        self.out.push_str("}\n\n");
    }

    fn param_decl(&self, name: &str, ty: TypeHandle) -> String {
        if is_array(self.context, ty) {
            let numel = array_numel(self.context, ty);
            format!("double {name}[{numel}]")
        } else {
            format!("double {name}")
        }
    }

    fn emit_block(&mut self, block: Ptr<BasicBlock>, indent: usize) {
        // Map block arguments (function params or loop induction vars) to names.
        let args: Vec<Value> = block.deref(self.context).arguments().collect();
        for arg in args {
            let name = self.fresh();
            self.names.insert(arg, name);
        }
        for op in block.deref(self.context).iter(self.context) {
            self.emit_op(op, indent);
        }
    }

    fn emit_op(&mut self, op: Ptr<Operation>, indent: usize) {
        let pad = "  ".repeat(indent);

        if let Some(d) = Operation::get_op::<emitc::DeclareOp>(op, self.context) {
            let result = op.deref(self.context).get_result(0);
            let name = d.name(self.context);
            self.names.insert(result, name.clone());
            let ty = result.get_type(self.context);
            if is_array(self.context, ty) && array_numel(self.context, ty) == 1 {
                self.out.push_str(&format!("{pad}double {name};\n"));
            } else {
                let numel = array_numel(self.context, ty);
                self.out
                    .push_str(&format!("{pad}double {name}[{numel}];\n"));
            }
        } else if let Some(l) = Operation::get_op::<emitc::LiteralOp>(op, self.context) {
            let name = self.assign_name(op);
            self.out.push_str(&format!(
                "{pad}double {name} = {};\n",
                fmt_f64(l.value(self.context))
            ));
        } else if let Some(b) = Operation::get_op::<emitc::BinOp>(op, self.context) {
            let name = self.assign_name(op);
            let lhs = self.expr(op.deref(self.context).get_operand(0));
            let rhs = self.expr(op.deref(self.context).get_operand(1));
            let op_str = binop_str(b.kind(self.context));
            self.out
                .push_str(&format!("{pad}double {name} = ({lhs} {op_str} {rhs});\n"));
        } else if let Some(c) = Operation::get_op::<emitc::CmpOp>(op, self.context) {
            let name = self.assign_name(op);
            let lhs = self.expr(op.deref(self.context).get_operand(0));
            let rhs = self.expr(op.deref(self.context).get_operand(1));
            let op_str = cmp_str(c.kind(self.context));
            self.out
                .push_str(&format!("{pad}bool {name} = ({lhs} {op_str} {rhs});\n"));
        } else if let Some(_t) = Operation::get_op::<emitc::TernaryOp>(op, self.context) {
            let name = self.assign_name(op);
            let cond = self.expr(op.deref(self.context).get_operand(0));
            let tv = self.expr(op.deref(self.context).get_operand(1));
            let fv = self.expr(op.deref(self.context).get_operand(2));
            self.out
                .push_str(&format!("{pad}double {name} = ({cond} ? {tv} : {fv});\n"));
        } else if let Some(c) = Operation::get_op::<emitc::CallOp>(op, self.context) {
            let name = self.assign_name(op);
            let callee = c.callee(self.context);
            let args: Vec<String> = op
                .deref(self.context)
                .operands()
                .map(|v| self.expr(v))
                .collect();
            self.out.push_str(&format!(
                "{pad}double {name} = {callee}({});\n",
                args.join(", ")
            ));
        } else if let Some(_l) = Operation::get_op::<emitc::LoadOp>(op, self.context) {
            let name = self.assign_name(op);
            let array = op.deref(self.context).get_operand(0);
            let index = op.deref(self.context).get_operand(1);
            let access = self.access(array, index);
            self.out
                .push_str(&format!("{pad}double {name} = {access};\n"));
        } else if let Some(_a) = Operation::get_op::<emitc::AssignOp>(op, self.context) {
            let array = op.deref(self.context).get_operand(0);
            let index = op.deref(self.context).get_operand(1);
            let value = self.expr(op.deref(self.context).get_operand(2));
            let access = self.access(array, index);
            self.out.push_str(&format!("{pad}{access} = {value};\n"));
        } else if let Some(i) = Operation::get_op::<emitc::IfOp>(op, self.context) {
            let cond = self.expr(i.condition(self.context));
            self.out.push_str(&format!("{pad}if ({cond}) {{\n"));
            self.emit_region(i.then_region(self.context), indent + 1);
            self.out.push_str(&format!("{pad}}} else {{\n"));
            self.emit_region(i.else_region(self.context), indent + 1);
            self.out.push_str(&format!("{pad}}}\n"));
        } else if let Some(w) = Operation::get_op::<emitc::WhileOp>(op, self.context) {
            self.out.push_str(&format!("{pad}while (true) {{\n"));
            self.emit_region(w.before_region(self.context), indent + 1);
            self.emit_region(w.after_region(self.context), indent + 1);
            self.out.push_str(&format!("{pad}}}\n"));
        } else if let Some(f) = Operation::get_op::<emitc::ForOp>(op, self.context) {
            let start = self.expr(op.deref(self.context).get_operand(0));
            let end = self.expr(op.deref(self.context).get_operand(1));
            let step = self.expr(op.deref(self.context).get_operand(2));
            let body = f
                .body_region(self.context)
                .deref(self.context)
                .get_entry_block()
                .expect("for body block");
            let iv = body.deref(self.context).get_argument(0);
            let iv_name = self.fresh();
            self.names.insert(iv, iv_name.clone());
            self.out.push_str(&format!(
                "{pad}for (double {iv_name} = {start}; {iv_name} < {end}; {iv_name} += {step}) {{\n"
            ));
            self.emit_region(f.body_region(self.context), indent + 1);
            self.out.push_str(&format!("{pad}}}\n"));
        } else if let Some(_c) = Operation::get_op::<emitc::ConditionOp>(op, self.context) {
            let cond = self.expr(op.deref(self.context).get_operand(0));
            self.out.push_str(&format!("{pad}if (!{cond}) break;\n"));
        } else if let Some(_y) = Operation::get_op::<emitc::YieldOp>(op, self.context) {
            // Yield carries no values; nothing to emit.
        } else if let Some(_b) = Operation::get_op::<emitc::BreakOp>(op, self.context) {
            self.out.push_str(&format!("{pad}break;\n"));
        } else if let Some(_r) = Operation::get_op::<emitc::ReturnOp>(op, self.context) {
            let values: Vec<String> = op
                .deref(self.context)
                .operands()
                .map(|v| self.expr(v))
                .collect();
            match values.len() {
                0 => self.out.push_str(&format!("{pad}return;\n")),
                1 => self.out.push_str(&format!("{pad}return {};\n", values[0])),
                _ => self
                    .out
                    .push_str(&format!("{pad}return {{ {} }};\n", values.join(", "))),
            }
        } else {
            self.out.push_str(&format!(
                "{pad}// unknown op: {}\n",
                Operation::get_opid(op, self.context)
            ));
        }
    }

    fn emit_region(&mut self, region: Ptr<Region>, indent: usize) {
        if let Some(block) = region.deref(self.context).get_entry_block() {
            self.emit_block(block, indent);
        }
    }

    /// Assign a fresh name to an op's result, returning the name.
    fn assign_name(&mut self, op: Ptr<Operation>) -> String {
        let name = self.fresh();
        let result = op.deref(self.context).get_result(0);
        self.names.insert(result, name.clone());
        name
    }

    /// The C access expression for `array[index]` (or the bare name for a
    /// 1-element scalar cell).
    fn access(&self, array: Value, index: Value) -> String {
        let ty = array.get_type(self.context);
        if is_array(self.context, ty) && array_numel(self.context, ty) == 1 {
            self.expr(array)
        } else {
            let idx = self.expr(index);
            format!("{}[(int64_t){idx}]", self.expr(array))
        }
    }
}

fn is_array(context: &Context, ty: TypeHandle) -> bool {
    ty.deref(context).is::<ArrayType>()
}

fn array_numel(context: &Context, ty: TypeHandle) -> i64 {
    ty.deref(context)
        .downcast_ref::<ArrayType>()
        .map(|a| a.numel())
        .unwrap_or(1)
}

fn binop_str(kind: BinOpKind) -> &'static str {
    match kind {
        BinOpKind::Add => "+",
        BinOpKind::Sub => "-",
        BinOpKind::Mul => "*",
        BinOpKind::Div => "/",
    }
}

fn cmp_str(kind: CmpKind) -> &'static str {
    match kind {
        CmpKind::Eq => "==",
        CmpKind::Ne => "!=",
        CmpKind::Lt => "<",
        CmpKind::Le => "<=",
        CmpKind::Gt => ">",
        CmpKind::Ge => ">=",
    }
}

fn fmt_f64(value: f64) -> String {
    if value.is_nan() {
        "NAN".to_string()
    } else if value == f64::INFINITY {
        "INFINITY".to_string()
    } else if value == f64::NEG_INFINITY {
        "(-INFINITY)".to_string()
    } else {
        let s = format!("{value}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}
