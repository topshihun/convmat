//! Compile-time dispatch: decide each value's class and each function's route.
//!
//! MATLAB is dynamically typed, so before lowering we must resolve, **at compile
//! time**, how every value is realized: as an `f64` scalar, a statically-shaped
//! matrix, a runtime matrix (pointer + length ABI), a scalar-field struct, or not
//! at all. This module is that phase: [`dispatch`] computes, once per function, a
//! [`FunctionPlan`] holding each binding's resolved type and the function's
//! [`Route`]. The pipeline routes on the plan and the HIR lowerer reads the
//! plan's resolved types, so the "scalar vs runtime matrix" choice is made in one
//! place instead of being re-derived across stages.

use std::collections::HashMap;

use runmat_hir::{BindingId, HirFunction};

use super::{classify, infer_locals, LocalTy, Verdict};

/// How a value is realized at run time, after compile-time dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueClass {
    /// A single `f64`/`bool` scalar.
    Scalar,
    /// A matrix whose shape is known at compile time (`matlab.array`).
    StaticMatrix(super::Shape),
    /// A matrix whose shape is only known at run time (pointer + length ABI).
    RuntimeMatrix,
    /// A scalar-field struct with a compile-time field layout.
    Struct,
    /// A boxed cell array (`convmat_value*`).
    Cell,
    /// A boxed complex value (`convmat_value*`).
    Complex,
    /// A value the static subset cannot realize (deferred to the runtime tier).
    Unsupported,
}

impl ValueClass {
    /// Classify a resolved local type.
    pub fn from_local_ty(ty: &LocalTy) -> Self {
        match ty {
            LocalTy::Scalar | LocalTy::Int32 => ValueClass::Scalar,
            LocalTy::Array { shape } if shape.is_dynamic() => ValueClass::RuntimeMatrix,
            LocalTy::Array { shape } => ValueClass::StaticMatrix(*shape),
            LocalTy::Struct { .. } => ValueClass::Struct,
            LocalTy::Cell => ValueClass::Cell,
            LocalTy::Complex => ValueClass::Complex,
            LocalTy::Dynamic => ValueClass::Unsupported,
        }
    }

    /// Whether the value lives in the runtime-matrix tier.
    pub fn is_runtime_matrix(self) -> bool {
        matches!(self, ValueClass::RuntimeMatrix)
    }
}

/// Whether a function is lowered statically or handed to the runtime tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// Every value is scalar / static-matrix / struct: lower to `matlab`.
    Static,
    /// A value needs the runtime matrix tier, or the function uses an
    /// unsupported construct; the string is the deferral reason.
    Runtime(String),
}

/// The compile-time dispatch plan for one function.
#[derive(Debug, Clone)]
pub struct FunctionPlan {
    /// The function's route.
    pub route: Route,
    /// The resolved type of every binding. The rich [`LocalTy`] is retained (not
    /// just [`ValueClass`]) so the lowerer keeps struct fields and static shapes.
    pub values: HashMap<BindingId, LocalTy>,
}

impl FunctionPlan {
    /// The dispatch class of one binding.
    pub fn class_of(&self, id: BindingId) -> ValueClass {
        self.values
            .get(&id)
            .map(ValueClass::from_local_ty)
            .unwrap_or(ValueClass::Scalar)
    }

    /// Whether any binding is realized by the runtime matrix tier.
    pub fn uses_runtime_matrix(&self) -> bool {
        self.values
            .values()
            .any(|ty| ValueClass::from_local_ty(ty).is_runtime_matrix())
    }
}

/// Dispatch one function: resolve every value's class and the function route.
pub fn dispatch(function: &HirFunction) -> FunctionPlan {
    let values = infer_locals(function);
    let route = match classify(function) {
        Verdict::Static => Route::Static,
        Verdict::Deferred { reason } => Route::Runtime(reason),
    };
    FunctionPlan { route, values }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{parse_hir, SourceFile};

    fn plan_of(source: &str) -> FunctionPlan {
        let file = SourceFile {
            path: "test.m".to_string(),
            source: source.to_string(),
        };
        let hir = parse_hir(&file).expect("parse to HIR");
        dispatch(&hir.functions[0])
    }

    #[test]
    fn scalar_function_is_static_scalar() {
        let plan = plan_of("function y = f(x)\ny = x + 1;\nend\n");
        assert_eq!(plan.route, Route::Static);
        assert!(!plan.uses_runtime_matrix());
    }

    #[test]
    fn array_parameter_is_runtime_matrix() {
        let plan = plan_of("function y = f(A)\ny = sum(A);\nend\n");
        assert_eq!(plan.route, Route::Static);
        assert!(plan.uses_runtime_matrix());
    }

    #[test]
    fn unsupported_builtin_routes_to_runtime() {
        let plan = plan_of("function y = f()\ny = svd([1, 2, 3, 4]);\nend\n");
        assert!(matches!(plan.route, Route::Runtime(_)));
    }
}
