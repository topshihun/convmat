//! The C emitter: pretty-print the `emitc` dialect as C++ source.
//!
//! The output is C++-style source (`double`/`void`/`std::tuple` return types,
//! array out-params as `double name[N]`, `bool` for comparisons, and C-style
//! `if`/`while`/`for`). No optimization is applied; every SSA value is assigned
//! a fresh name.

use std::collections::HashMap;

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::{OneRegionInterface, SymbolOpInterface},
        ops::{FuncOp, ModuleOp},
        type_interfaces::FunctionTypeInterface,
        types::FunctionType,
    },
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
        matlab::{ArrayType, BinOpKind, CmpKind, PtrType, StructType},
    },
    error::Result,
};

/// Emit C++ source for a builtin `module` whose functions carry `emitc`-level
/// bodies: its top-level directives (`#include`, `#define`, ...) followed by its
/// functions.
pub fn emit(context: &Context, module: &ModuleOp) -> Result<String> {
    let mut emitter = Emitter {
        context,
        names: HashMap::new(),
        struct_types: HashMap::new(),
        counter: 0,
        out: String::new(),
    };
    // These standard-library headers are always required by the generated code.
    emitter
        .out
        .push_str("#include <cmath>\n#include <cstdint>\n#include <tuple>\n\n");
    // First pass: emit every `struct` definition at file scope (before the
    // functions that reference them).
    emitter.emit_struct_definitions(module);
    if let Some(block) = module.get_region(context).deref(context).get_entry_block() {
        let ops: Vec<Ptr<Operation>> = block.deref(context).iter(context).collect();
        let mut prototypes_emitted = false;
        for op in ops {
            // Emit every forward declaration right before the first function
            // body, so top-level directives keep their source order and calls to
            // later-defined functions still resolve.
            if !prototypes_emitted && Operation::get_op::<FuncOp>(op, context).is_some() {
                emitter.emit_function_prototypes(module);
                prototypes_emitted = true;
            }
            emitter.emit_top_level(op);
        }
    }
    Ok(emitter.out)
}

struct Emitter<'a> {
    context: &'a Context,
    names: HashMap<Value, String>,
    /// Map a `matlab.struct` type to its emitted C `struct` tag name.
    struct_types: HashMap<TypeHandle, String>,
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

    fn emit_func(&mut self, func: &FuncOp) {
        let name = func.get_symbol_name(self.context).as_ref().to_string();
        let res_types = {
            let fn_ty = func.get_type(self.context).deref(self.context);
            let ft = fn_ty
                .downcast_ref::<FunctionType>()
                .expect("func carries a function type");
            ft.res_types()
        };
        let entry = func.get_entry_block(self.context);

        let args: Vec<Value> = entry.deref(self.context).arguments().collect();
        let mut params = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let pname = format!("v{}", i + 1);
            self.names.insert(*arg, pname.clone());
            params.push(self.param_decl(&pname, arg.get_type(self.context)));
        }
        // Reserve the parameter names so later `fresh()` calls start after them.
        self.counter = args.len();

        let ret = match res_types.as_slice() {
            [] => "void".to_string(),
            [ty] => self.c_type(*ty),
            tys => format!(
                "std::tuple<{}>",
                tys.iter()
                    .map(|ty| self.c_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };

        self.out
            .push_str(&format!("{ret} {name}({}", params.join(", ")));
        self.out.push_str(") {\n");
        self.emit_block(entry, 1);
        self.out.push_str("}\n\n");
    }

    /// Emit every `struct` definition referenced by the module at file scope.
    fn emit_struct_definitions(&mut self, module: &ModuleOp) {
        if let Some(block) = module
            .get_region(self.context)
            .deref(self.context)
            .get_entry_block()
        {
            for op in block.deref(self.context).iter(self.context) {
                // Function signatures may reference `struct` types as parameters
                // or results.
                if let Some(func) = Operation::get_op::<FuncOp>(op, self.context) {
                    let fn_ty = func.get_type(self.context).deref(self.context);
                    if let Some(ft) = fn_ty.downcast_ref::<FunctionType>() {
                        for ty in ft.arg_types().into_iter().chain(ft.res_types()) {
                            if is_struct(self.context, ty) {
                                self.struct_type_name(ty);
                            }
                        }
                    }
                }
                self.collect_structs_in_op(op);
            }
        }
    }

    /// Recurse through `op` and emit `struct` definitions for any `matlab.struct`
    /// result type (a `DeclareOp` result, or a function argument/result).
    fn collect_structs_in_op(&mut self, op: Ptr<Operation>) {
        for i in 0..op.deref(self.context).get_num_results() {
            let ty = op.deref(self.context).get_result(i).get_type(self.context);
            if is_struct(self.context, ty) {
                self.struct_type_name(ty);
            }
        }
        for region in op.deref(self.context).regions() {
            if let Some(block) = region.deref(self.context).get_entry_block() {
                for nested in block.deref(self.context).iter(self.context) {
                    self.collect_structs_in_op(nested);
                }
            }
        }
    }

    /// Dispatch a top-level op (a directive or a function).
    fn emit_top_level(&mut self, op: Ptr<Operation>) {
        if let Some(func) = Operation::get_op::<FuncOp>(op, self.context) {
            self.emit_func(&func);
        } else if let Some(inc) = Operation::get_op::<emitc::IncludeOp>(op, self.context) {
            self.emit_include(&inc);
        } else if let Some(d) = Operation::get_op::<emitc::DefineOp>(op, self.context) {
            self.emit_define(&d);
        } else if let Some(u) = Operation::get_op::<emitc::UndefOp>(op, self.context) {
            self.emit_undef(&u);
        } else if let Some(v) = Operation::get_op::<emitc::VerbatimOp>(op, self.context) {
            self.emit_verbatim(&v);
        } else {
            self.out.push_str(&format!(
                "// unknown top-level op: {}\n",
                Operation::get_opid(op, self.context)
            ));
        }
    }

    fn emit_include(&mut self, inc: &emitc::IncludeOp) {
        let header = inc.header(self.context);
        if inc.is_system(self.context) {
            self.out.push_str(&format!("#include <{header}>\n"));
        } else {
            self.out.push_str(&format!("#include \"{header}\"\n"));
        }
    }

    fn emit_define(&mut self, d: &emitc::DefineOp) {
        let name = d.name(self.context);
        let value = d.value(self.context);
        if value.is_empty() {
            self.out.push_str(&format!("#define {name}\n"));
        } else {
            self.out.push_str(&format!("#define {name} {value}\n"));
        }
    }

    fn emit_undef(&mut self, u: &emitc::UndefOp) {
        self.out
            .push_str(&format!("#undef {}\n", u.name(self.context)));
    }

    fn emit_verbatim(&mut self, v: &emitc::VerbatimOp) {
        let source = v.source(self.context);
        self.out.push_str(&source);
        if !source.ends_with('\n') {
            self.out.push('\n');
        }
    }

    fn param_decl(&mut self, name: &str, ty: TypeHandle) -> String {
        if is_array(self.context, ty) {
            let numel = array_numel(self.context, ty);
            format!("double {name}[{numel}]")
        } else if is_ptr(self.context, ty) {
            format!("double* {name}")
        } else if is_struct(self.context, ty) {
            format!("struct {} {name}", self.struct_type_name(ty))
        } else {
            format!("double {name}")
        }
    }

    /// Emit a C prototype for every `builtin.func` in the module, so calls to
    /// functions defined later (e.g. a helper called before its definition) still
    /// resolve.
    fn emit_function_prototypes(&mut self, module: &ModuleOp) {
        let Some(block) = module
            .get_region(self.context)
            .deref(self.context)
            .get_entry_block()
        else {
            return;
        };
        let ops: Vec<Ptr<Operation>> = block.deref(self.context).iter(self.context).collect();
        let mut protos = Vec::new();
        for op in ops {
            if let Some(func) = Operation::get_op::<FuncOp>(op, self.context) {
                if let Some(proto) = self.prototype(&func) {
                    protos.push(proto);
                }
            }
        }
        if protos.is_empty() {
            return;
        }
        for proto in protos {
            self.out.push_str(&proto);
            self.out.push('\n');
        }
        self.out.push('\n');
    }

    /// The C prototype of a function (`ret name(params);`).
    fn prototype(&mut self, func: &FuncOp) -> Option<String> {
        let name = func.get_symbol_name(self.context).as_ref().to_string();
        let (arg_types, res_types) = {
            let fn_ty = func.get_type(self.context).deref(self.context);
            let ft = fn_ty.downcast_ref::<FunctionType>()?;
            (ft.arg_types(), ft.res_types())
        };
        let params: Vec<String> = arg_types
            .iter()
            .enumerate()
            .map(|(i, ty)| self.param_decl(&format!("v{}", i + 1), *ty))
            .collect();
        let ret = match res_types.as_slice() {
            [] => "void".to_string(),
            [ty] => self.c_type(*ty),
            tys => format!(
                "std::tuple<{}>",
                tys.iter()
                    .map(|ty| self.c_type(*ty))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        Some(format!("{ret} {name}({});", params.join(", ")))
    }

    /// The emitted C `struct` tag name for a `matlab.struct` type, defining the
    /// struct (once) on first use.
    fn struct_type_name(&mut self, ty: TypeHandle) -> String {
        if let Some(name) = self.struct_types.get(&ty).cloned() {
            return name;
        }
        let name = format!("s{}", self.struct_types.len());
        self.struct_types.insert(ty, name.clone());
        let struct_ty = ty.deref(self.context);
        let fields = struct_ty
            .downcast_ref::<StructType>()
            .expect("struct type")
            .fields();
        let mut body = String::new();
        for (field_name, _field_ty) in fields {
            body.push_str(&format!("  double {field_name};\n"));
        }
        self.out.push_str(&format!("struct {name} {{\n{body}}};\n"));
        name
    }

    /// The C type spelling of a `f64`, array, or `matlab.struct` type.
    fn c_type(&mut self, ty: TypeHandle) -> String {
        if is_array(self.context, ty) {
            format!("double[{}]", array_numel(self.context, ty))
        } else if is_struct(self.context, ty) {
            format!("struct {}", self.struct_type_name(ty))
        } else {
            "double".to_string()
        }
    }

    fn emit_block(&mut self, block: Ptr<BasicBlock>, indent: usize) {
        // Map block arguments (function params or loop induction vars) to
        // names. Skip arguments that already have one: function params are
        // named in `emit_func`, and `for`-loop induction variables are named in
        // `emit_op`, so re-assigning here would desynchronise the loop header
        // from its body.
        let args: Vec<Value> = block.deref(self.context).arguments().collect();
        for arg in args {
            if !self.names.contains_key(&arg) {
                let name = self.fresh();
                self.names.insert(arg, name);
            }
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
            if d.is_heap(self.context) {
                let numel = array_numel(self.context, ty);
                self.out
                    .push_str(&format!("{pad}double* {name} = new double[{numel}];\n"));
            } else if is_struct(self.context, ty) {
                let tag = self.struct_type_name(ty);
                self.out.push_str(&format!("{pad}struct {tag} {name};\n"));
            } else {
                let static_kw = if d.is_static(self.context) {
                    "static "
                } else {
                    ""
                };
                if is_array(self.context, ty) && array_numel(self.context, ty) == 1 {
                    self.out
                        .push_str(&format!("{pad}{static_kw}double {name};\n"));
                } else {
                    let numel = array_numel(self.context, ty);
                    self.out
                        .push_str(&format!("{pad}{static_kw}double {name}[{numel}];\n"));
                }
            }
        } else if let Some(_a) = Operation::get_op::<emitc::HeapAllocOp>(op, self.context) {
            let name = self.assign_name(op);
            let len = self.expr(op.deref(self.context).get_operand(0));
            self.out.push_str(&format!(
                "{pad}double* {name} = new double[(int64_t)({len})];\n"
            ));
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
        } else if let Some(c) = Operation::get_op::<emitc::CallVoidOp>(op, self.context) {
            let callee = c.callee(self.context);
            let args: Vec<String> = op
                .deref(self.context)
                .operands()
                .map(|v| self.expr(v))
                .collect();
            self.out
                .push_str(&format!("{pad}{callee}({});\n", args.join(", ")));
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
        } else if let Some(g) = Operation::get_op::<emitc::StructGetOp>(op, self.context) {
            let name = self.assign_name(op);
            let cell = self.expr(op.deref(self.context).get_operand(0));
            let field = g.field(self.context);
            self.out
                .push_str(&format!("{pad}double {name} = {cell}.{field};\n"));
        } else if let Some(s) = Operation::get_op::<emitc::StructSetOp>(op, self.context) {
            let cell = self.expr(op.deref(self.context).get_operand(0));
            let value = self.expr(op.deref(self.context).get_operand(1));
            let field = s.field(self.context);
            self.out
                .push_str(&format!("{pad}{cell}.{field} = {value};\n"));
        } else if let Some(_c) = Operation::get_op::<emitc::StructCopyOp>(op, self.context) {
            let dest = self.expr(op.deref(self.context).get_operand(0));
            let src = self.expr(op.deref(self.context).get_operand(1));
            self.out.push_str(&format!("{pad}{dest} = {src};\n"));
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
        } else if let Some(f) = Operation::get_op::<emitc::RangeForOp>(op, self.context) {
            let start = self.expr(op.deref(self.context).get_operand(0));
            let end = self.expr(op.deref(self.context).get_operand(1));
            let step = self.expr(op.deref(self.context).get_operand(2));
            let body = f
                .body_region(self.context)
                .deref(self.context)
                .get_entry_block()
                .expect("range for body");
            let iv = body.deref(self.context).get_argument(0);
            let iv_name = self.fresh();
            self.names.insert(iv, iv_name.clone());
            self.out.push_str(&format!(
                "{pad}for (double {iv_name} = {start}; ({step} >= 0) ? ({iv_name} <= {end}) : ({iv_name} >= {end}); {iv_name} += {step}) {{\n"
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
        } else if let Some(_c) = Operation::get_op::<emitc::ContinueOp>(op, self.context) {
            self.out.push_str(&format!("{pad}continue;\n"));
        } else if let Some(_d) = Operation::get_op::<emitc::DeleteOp>(op, self.context) {
            let array = op.deref(self.context).get_operand(0);
            let name = self.expr(array);
            self.out.push_str(&format!("{pad}delete[] {name};\n"));
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

fn is_ptr(context: &Context, ty: TypeHandle) -> bool {
    ty.deref(context).is::<PtrType>()
}

fn is_struct(context: &Context, ty: TypeHandle) -> bool {
    ty.deref(context).is::<StructType>()
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

#[cfg(test)]
mod tests {
    use super::emit;
    use crate::dialects::emitc;

    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OneRegionInterface;
    use pliron::builtin::ops::{FuncOp, ModuleOp};
    use pliron::builtin::types::FunctionType;
    use pliron::context::{Context, Ptr};
    use pliron::identifier::Identifier;
    use pliron::irbuild::inserter::{IRInserter, Inserter};
    use pliron::irbuild::listener::DummyListener;
    use pliron::op::Op;

    /// Append an op to the end of a block (mirrors `lowering::append`).
    fn append(ctx: &Context, block: Ptr<BasicBlock>, op: &dyn Op) {
        IRInserter::<DummyListener>::new_at_block_end(block).append_op(ctx, op);
    }

    #[test]
    fn emits_top_level_directives() {
        let mut ctx = Context::new();

        let module = ModuleOp::new(&mut ctx, Identifier::try_from("convmat").unwrap());
        let body = module
            .get_region(&ctx)
            .deref(&ctx)
            .get_entry_block()
            .expect("module block");

        let inc_math = emitc::IncludeOp::new(&mut ctx, "math.h", true);
        let inc_local = emitc::IncludeOp::new(&mut ctx, "myutil.h", false);
        let def_pi = emitc::DefineOp::new(&mut ctx, "PI", "3.14159265358979");
        let def_nodebug = emitc::DefineOp::new(&mut ctx, "NODEBUG", "");
        let undef_pi = emitc::UndefOp::new(&mut ctx, "PI");
        let verbatim = emitc::VerbatimOp::new(&mut ctx, "typedef double scalar_t;");

        // A bare function after the directives, to check source ordering.
        let fn_ty = FunctionType::get(&ctx, vec![], vec![]);
        let func = FuncOp::new(&mut ctx, Identifier::try_from("helper").unwrap(), fn_ty);

        append(&ctx, body, &inc_math);
        append(&ctx, body, &inc_local);
        append(&ctx, body, &def_pi);
        append(&ctx, body, &def_nodebug);
        append(&ctx, body, &undef_pi);
        append(&ctx, body, &verbatim);
        append(&ctx, body, &func);

        let c = emit(&ctx, &module).unwrap();

        assert!(c.contains("#include <cmath>"), "got:\n{c}");
        assert!(c.contains("#include <math.h>"), "got:\n{c}");
        assert!(c.contains("#include \"myutil.h\""), "got:\n{c}");
        assert!(c.contains("#define PI 3.14159265358979"), "got:\n{c}");
        assert!(c.contains("#define NODEBUG\n"), "got:\n{c}");
        assert!(c.contains("#undef PI"), "got:\n{c}");
        assert!(c.contains("typedef double scalar_t;"), "got:\n{c}");
        assert!(c.contains("void helper()"), "got:\n{c}");

        // Directives precede the function in the emitted source.
        let helper_pos = c.find("void helper()").unwrap();
        let include_pos = c.find("#include <math.h>").unwrap();
        assert!(include_pos < helper_pos, "got:\n{c}");
    }
}
