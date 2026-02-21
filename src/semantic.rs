//! Semantic analyzer invariants:
//! - No variable shadowing across nested scopes.
//! - Every expression identifier resolves to a declared symbol.
//! - Assignment preserves declared type, except explicit Int -> Float widening.
//! - Model/dataset identifiers are globally unique.
//! - Property contracts are validated via shared `rules.rs` constants.

use std::collections::{HashMap, HashSet};

use crate::ast::{BinaryOp, Expr, Program, Property, Span, Stmt};
use crate::diagnostics::best_suggestion;
use crate::rules::{
    ACTIVATIONS, DATASET_KNOWN_PROPERTIES, DEVICES, MODEL_KNOWN_PROPERTIES,
    MODEL_REQUIRED_PROPERTIES, OPTIMIZERS, TRAIN_KNOWN_PROPERTIES, TRAIN_REQUIRED_PROPERTIES,
};

#[derive(Debug, Clone)]
pub struct SemanticError {
    pub message: String,
    pub span: Span,
    pub hint: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SemanticWarning {
    pub message: String,
    pub span: Span,
}

pub struct SemanticAnalyzer {
    scopes: Vec<HashMap<String, ValueType>>,
    models: HashSet<String>,
    datasets: HashSet<String>,
    functions: HashMap<String, FunctionSig>,
    in_function_depth: usize,
    declared_variables: HashMap<String, Span>,
    used_variables: HashSet<String>,
    warnings: Vec<SemanticWarning>,
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
    Int,
    Float,
    Bool,
    Str,
    Unknown,
}

#[derive(Debug, Clone, Copy)]
struct FunctionSig {
    arity: usize,
    return_type: ValueType,
    span: Span,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            scopes: Vec::new(),
            models: HashSet::new(),
            datasets: HashSet::new(),
            functions: HashMap::new(),
            in_function_depth: 0,
            declared_variables: HashMap::new(),
            used_variables: HashSet::new(),
            warnings: Vec::new(),
        }
    }

    pub fn warnings(&self) -> &[SemanticWarning] {
        &self.warnings
    }

    pub fn analyze(&mut self, program: &Program) -> Result<(), SemanticError> {
        self.scopes.clear();
        self.models.clear();
        self.datasets.clear();
        self.functions.clear();
        self.in_function_depth = 0;
        self.declared_variables.clear();
        self.used_variables.clear();
        self.warnings.clear();

        self.push_scope();
        if let Err(err) = self.collect_named_entities(&program.statements) {
            self.pop_scope();
            return Err(err);
        }

        let result = self.analyze_statements(&program.statements);
        self.emit_unused_variable_warnings();
        self.pop_scope();
        result
    }

    fn collect_named_entities(&mut self, statements: &[Stmt]) -> Result<(), SemanticError> {
        for stmt in statements {
            match stmt {
                Stmt::Model { name, span, .. } => {
                    if self.models.contains(name) {
                        return Err(Self::error(
                            format!("Duplicate model declaration: '{name}'"),
                            *span,
                        ));
                    }
                    self.models.insert(name.clone());
                }
                Stmt::Dataset { name, span, .. } => {
                    if self.datasets.contains(name) {
                        return Err(Self::error(
                            format!("Duplicate dataset declaration: '{name}'"),
                            *span,
                        ));
                    }
                    self.datasets.insert(name.clone());
                }
                Stmt::Function {
                    name,
                    params,
                    body,
                    span,
                } => {
                    if self.functions.contains_key(name) {
                        return Err(Self::error(
                            format!("Duplicate function declaration: '{name}'"),
                            *span,
                        ));
                    }
                    self.functions.insert(
                        name.clone(),
                        FunctionSig {
                            arity: params.len(),
                            return_type: ValueType::Unknown,
                            span: *span,
                        },
                    );
                    self.collect_named_entities(body)?;
                }
                Stmt::Loop { body, .. } => self.collect_named_entities(body)?,
                Stmt::If {
                    then_branch,
                    elif_branches,
                    else_branch,
                    ..
                } => {
                    self.collect_named_entities(then_branch)?;
                    for (_, branch) in elif_branches {
                        self.collect_named_entities(branch)?;
                    }
                    if let Some(branch) = else_branch {
                        self.collect_named_entities(branch)?;
                    }
                }
                Stmt::VarDecl { .. }
                | Stmt::Assign { .. }
                | Stmt::Train { .. }
                | Stmt::Save { .. }
                | Stmt::Load { .. }
                | Stmt::Return { .. }
                | Stmt::Print { .. } => {}
            }
        }

        Ok(())
    }

    fn analyze_statements(&mut self, statements: &[Stmt]) -> Result<(), SemanticError> {
        for stmt in statements {
            let _ = stmt.span();
            self.analyze_stmt(stmt)?;
        }
        Ok(())
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) -> Result<(), SemanticError> {
        match stmt {
            Stmt::VarDecl { name, value, span } => {
                if self.any_scope_contains(name) {
                    return Err(Self::error(
                        format!("Variable shadowing is not allowed: '{name}'"),
                        *span,
                    ));
                }

                let value_type = self.infer_expr_type(value)?;
                self.declare_variable(name.clone(), value_type, *span)?;
                self.declared_variables.insert(name.clone(), *span);
            }
            Stmt::Assign { name, value, span } => {
                let target_type = self.lookup_variable(name).ok_or_else(|| {
                    Self::error_with_hint(
                        format!("Assignment to undefined variable: '{name}'"),
                        *span,
                        format!(
                            "Declare '{name}' before assigning to it, e.g. `{name} 0` then `{name} = ...`."
                        ),
                    )
                })?;

                let value_type = self.infer_expr_type(value)?;
                if !is_assignable(target_type, value_type) {
                    return Err(Self::error(
                        format!(
                            "Type mismatch in assignment to '{}': expected {:?}, found {:?}",
                            name, target_type, value_type
                        ),
                        *span,
                    ));
                }
            }
            Stmt::Function {
                name,
                params,
                body,
                span,
            } => {
                self.push_scope();
                self.in_function_depth += 1;

                for param in params {
                    if self.any_scope_contains(param) {
                        self.pop_scope();
                        self.in_function_depth = self.in_function_depth.saturating_sub(1);
                        return Err(Self::error(
                            format!("Function parameter shadowing is not allowed: '{param}'"),
                            *span,
                        ));
                    }
                    self.declare_variable(param.clone(), ValueType::Unknown, *span)?;
                }

                let body_result = self.analyze_statements(body);
                self.pop_scope();
                self.in_function_depth = self.in_function_depth.saturating_sub(1);
                body_result?;

                if let Some(sig) = self.functions.get_mut(name) {
                    let _ = sig.span;
                    sig.return_type = ValueType::Unknown;
                }
            }
            Stmt::Return { value, span } => {
                if self.in_function_depth == 0 {
                    return Err(Self::error(
                        "Return statement is only allowed inside function".to_string(),
                        *span,
                    ));
                }
                if let Some(expr) = value {
                    let _ = self.infer_expr_type(expr)?;
                }
            }
            Stmt::Model { props, .. } => {
                self.validate_properties("model", props, MODEL_REQUIRED_PROPERTIES)?;
            }
            Stmt::Dataset { props, .. } => {
                self.validate_properties("dataset", props, &[])?;
            }
            Stmt::Train {
                model,
                data,
                props,
                span,
            } => {
                if !self.models.contains(model) {
                    return Err(Self::error_with_hint(
                        format!("Undefined model in train: '{model}'"),
                        *span,
                        format!(
                            "Declare model '{model}' before training, for example:\nmodel {model}\n    layers 2 1"
                        ),
                    ));
                }
                if !self.datasets.contains(data) {
                    return Err(Self::error_with_hint(
                        format!("Undefined dataset in train: '{data}'"),
                        *span,
                        format!(
                            "Declare dataset '{data}' before training, for example:\ndataset {data}\n    batch 32"
                        ),
                    ));
                }

                self.validate_properties("train", props, TRAIN_REQUIRED_PROPERTIES)?;
            }
            Stmt::Save { model, span, .. } => {
                if !self.models.contains(model) {
                    return Err(Self::error_with_hint(
                        format!("Undefined model in save: '{model}'"),
                        *span,
                        "Define the model before saving it in this script run.".to_string(),
                    ));
                }
            }
            Stmt::Load { model, span, .. } => {
                if !self.models.contains(model) {
                    return Err(Self::error_with_hint(
                        format!("Undefined model in load: '{model}'"),
                        *span,
                        "Declare the model name first, then load weights into it.".to_string(),
                    ));
                }
            }
            Stmt::Print { expr, span } => {
                let _ = span;
                self.infer_expr_type(expr)?;
            }
            Stmt::Loop { count, body, span } => {
                let count_type = self.infer_expr_type(count)?;
                if count_type != ValueType::Int {
                    return Err(Self::error("Loop count must be integer".to_string(), *span));
                }

                if let Expr::Int { value, .. } = count {
                    if *value < 0 {
                        return Err(Self::error(
                            "Loop count must be non-negative".to_string(),
                            count.span(),
                        ));
                    }
                    if *value == 0 {
                        self.warnings.push(SemanticWarning {
                            message: "Loop has zero iterations; body is unreachable".to_string(),
                            span: *span,
                        });
                    }
                }

                self.push_scope();
                let result = self.analyze_statements(body);
                self.pop_scope();
                result?;
            }
            Stmt::If {
                condition,
                then_branch,
                elif_branches,
                else_branch,
                span,
            } => {
                let cond_type = self.infer_expr_type(condition)?;
                if cond_type != ValueType::Bool {
                    return Err(Self::error(
                        "If condition must be boolean".to_string(),
                        *span,
                    ));
                }

                if let Expr::Bool { value, .. } = condition {
                    self.warnings.push(SemanticWarning {
                        message: if *value {
                            "else/elif branches may be unreachable due to constant true condition"
                                .to_string()
                        } else {
                            "then branch may be unreachable due to constant false condition"
                                .to_string()
                        },
                        span: *span,
                    });
                }

                self.push_scope();
                let then_result = self.analyze_statements(then_branch);
                self.pop_scope();
                then_result?;

                for (elif_condition, branch) in elif_branches {
                    let elif_type = self.infer_expr_type(elif_condition)?;
                    if elif_type != ValueType::Bool {
                        return Err(Self::error(
                            "Elif condition must be boolean".to_string(),
                            elif_condition.span(),
                        ));
                    }

                    self.push_scope();
                    let branch_result = self.analyze_statements(branch);
                    self.pop_scope();
                    branch_result?;
                }

                if let Some(branch) = else_branch {
                    self.push_scope();
                    let else_result = self.analyze_statements(branch);
                    self.pop_scope();
                    else_result?;
                }
            }
        }

        Ok(())
    }

    fn validate_properties(
        &mut self,
        context: &str,
        props: &[Property],
        required: &[&str],
    ) -> Result<(), SemanticError> {
        self.ensure_unique_properties(context, props)?;
        self.ensure_required_properties(context, props, required)?;

        let known = known_property_names(context);
        for prop in props {
            if !known.contains(&prop.key.as_str()) {
                let suggestion = best_suggestion(&prop.key, known);
                self.warnings.push(SemanticWarning {
                    message: match suggestion {
                        Some(candidate) => format!(
                            "Unknown property '{}' in {context} block. Did you mean '{}'?",
                            prop.key, candidate
                        ),
                        None => format!("Unknown property '{}' in {context} block", prop.key),
                    },
                    span: prop.span,
                });
            }
        }

        for prop in props {
            self.validate_property_value_types(context, prop)?;
        }

        Ok(())
    }

    fn ensure_unique_properties(
        &self,
        context: &str,
        props: &[Property],
    ) -> Result<(), SemanticError> {
        let mut seen = HashSet::new();
        for prop in props {
            if !seen.insert(prop.key.as_str()) {
                return Err(Self::error(
                    format!("Duplicate property '{}' in {context} block", prop.key),
                    prop.span,
                ));
            }
        }
        Ok(())
    }

    fn ensure_required_properties(
        &self,
        context: &str,
        props: &[Property],
        required: &[&str],
    ) -> Result<(), SemanticError> {
        let present: HashSet<&str> = props.iter().map(|p| p.key.as_str()).collect();
        for key in required {
            if !present.contains(key) {
                return Err(Self::error_with_hint(
                    format!("Missing required property '{key}' in {context} block"),
                    props.first().map(|p| p.span).unwrap_or(Span::unknown()),
                    format!(
                        "Add `{key} ...` to the {context} block. Required set: {}",
                        required.join(", ")
                    ),
                ));
            }
        }
        Ok(())
    }

    fn validate_property_value_types(
        &mut self,
        context: &str,
        prop: &Property,
    ) -> Result<(), SemanticError> {
        match (context, prop.key.as_str()) {
            ("model", "layers") => {
                self.expect_min_count(context, prop, 2)?;
                for value in &prop.values {
                    self.expect_value_type(context, &prop.key, value, ValueType::Int)?;
                }
            }
            ("model", "activation") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_allowed_symbol(context, &prop.key, &prop.values[0], ACTIVATIONS)?;
            }
            ("model", "optimizer") => {
                if prop.values.is_empty() || prop.values.len() > 2 {
                    return Err(Self::error(
                        "Property 'optimizer' in model block requires 1 or 2 values".to_string(),
                        prop.span,
                    ));
                }
                self.expect_allowed_symbol(context, &prop.key, &prop.values[0], OPTIMIZERS)?;
                if prop.values.len() == 2 {
                    let rate_ty = self.infer_expr_type(&prop.values[1])?;
                    if !matches!(rate_ty, ValueType::Int | ValueType::Float) {
                        return Err(Self::error(
                            "Second optimizer value must be numeric (learning rate)".to_string(),
                            prop.values[1].span(),
                        ));
                    }
                    match &prop.values[1] {
                        Expr::Int { value, .. } if *value <= 0 => {
                            return Err(Self::error(
                                "Second optimizer value (learning rate) must be > 0".to_string(),
                                prop.values[1].span(),
                            ));
                        }
                        Expr::Float { value, .. } if *value <= 0.0 => {
                            return Err(Self::error(
                                "Second optimizer value (learning rate) must be > 0".to_string(),
                                prop.values[1].span(),
                            ));
                        }
                        _ => {}
                    }
                }
            }
            ("model", "precision") | ("model", "memory") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_symbol(context, &prop.key, &prop.values[0])?;
            }
            ("dataset", "batch") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_value_type(context, &prop.key, &prop.values[0], ValueType::Int)?;
            }
            ("dataset", "shuffle") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_value_type(context, &prop.key, &prop.values[0], ValueType::Bool)?;
            }
            ("train", "epochs") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_value_type(context, &prop.key, &prop.values[0], ValueType::Int)?;
                if let Expr::Int { value, .. } = &prop.values[0]
                    && *value < 0
                {
                    return Err(Self::error(
                        "Property 'epochs' in train block must be non-negative".to_string(),
                        prop.values[0].span(),
                    ));
                }
            }
            ("train", "device") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_allowed_symbol(context, &prop.key, &prop.values[0], DEVICES)?;
            }
            ("train", "optimizer") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_allowed_symbol(context, &prop.key, &prop.values[0], OPTIMIZERS)?;
            }
            ("train", "lr") => {
                self.expect_exact_count(context, prop, 1)?;
                let lr_ty = self.infer_expr_type(&prop.values[0])?;
                if !matches!(lr_ty, ValueType::Int | ValueType::Float) {
                    return Err(Self::error(
                        "Property 'lr' in train block must be numeric".to_string(),
                        prop.values[0].span(),
                    ));
                }

                match &prop.values[0] {
                    Expr::Int { value, .. } if *value <= 0 => {
                        return Err(Self::error(
                            "Property 'lr' in train block must be > 0".to_string(),
                            prop.values[0].span(),
                        ));
                    }
                    Expr::Float { value, .. } if *value <= 0.0 => {
                        return Err(Self::error(
                            "Property 'lr' in train block must be > 0".to_string(),
                            prop.values[0].span(),
                        ));
                    }
                    _ => {}
                }
            }
            ("train", "batch") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_value_type(context, &prop.key, &prop.values[0], ValueType::Int)?;
                if let Expr::Int { value, .. } = &prop.values[0]
                    && *value <= 0
                {
                    return Err(Self::error(
                        "Property 'batch' in train block must be > 0".to_string(),
                        prop.values[0].span(),
                    ));
                }
            }
            ("train", "precision") => {
                self.expect_exact_count(context, prop, 1)?;
                self.expect_symbol(context, &prop.key, &prop.values[0])?;
            }
            _ => {}
        }

        Ok(())
    }

    fn expect_exact_count(
        &self,
        context: &str,
        prop: &Property,
        count: usize,
    ) -> Result<(), SemanticError> {
        if prop.values.len() != count {
            return Err(Self::error(
                format!(
                    "Property '{}' in {context} block requires exactly {count} value(s)",
                    prop.key
                ),
                prop.span,
            ));
        }
        Ok(())
    }

    fn expect_min_count(
        &self,
        context: &str,
        prop: &Property,
        min: usize,
    ) -> Result<(), SemanticError> {
        if prop.values.len() < min {
            return Err(Self::error(
                format!(
                    "Property '{}' in {context} block requires at least {min} value(s)",
                    prop.key
                ),
                prop.span,
            ));
        }
        Ok(())
    }

    fn expect_symbol(&self, context: &str, key: &str, value: &Expr) -> Result<(), SemanticError> {
        match value {
            Expr::Ident { .. } | Expr::Str { .. } => Ok(()),
            _ => Err(Self::error(
                format!("Property '{key}' in {context} block expects identifier or string value"),
                value.span(),
            )),
        }
    }

    fn expect_allowed_symbol(
        &mut self,
        context: &str,
        key: &str,
        value: &Expr,
        allowed: &[&str],
    ) -> Result<(), SemanticError> {
        if let Expr::Ident { name, .. } = value
            && let Some(var_type) = self.lookup_variable(name)
        {
            if var_type == ValueType::Str {
                self.used_variables.insert(name.clone());
                return Ok(());
            }

            return Err(Self::error(
                format!(
                    "Property '{key}' in {context} block expects symbol/string; variable '{name}' is not string"
                ),
                value.span(),
            ));
        }

        let symbol = self.extract_symbol(value).ok_or_else(|| {
            Self::error(
                format!("Property '{key}' in {context} block expects identifier or string value"),
                value.span(),
            )
        })?;

        if allowed.contains(&symbol.as_str()) {
            Ok(())
        } else {
            let suggestion = best_suggestion(&symbol, allowed);
            let hint = match suggestion {
                Some(candidate) => format!(
                    "Did you mean '{candidate}'? Allowed values: {}",
                    allowed.join(", ")
                ),
                None => format!("Allowed values: {}", allowed.join(", ")),
            };
            Err(Self::error_with_hint(
                format!(
                    "Invalid value '{}' for property '{}' in {context} block",
                    symbol, key
                ),
                value.span(),
                hint,
            ))
        }
    }

    fn expect_value_type(
        &mut self,
        context: &str,
        key: &str,
        value: &Expr,
        expected: ValueType,
    ) -> Result<(), SemanticError> {
        let value_type = self.infer_expr_type(value)?;
        if value_type != expected {
            return Err(Self::error(
                format!(
                    "Property '{key}' in {context} block expects {:?}, found {:?}",
                    expected, value_type
                ),
                value.span(),
            ));
        }
        Ok(())
    }

    fn infer_expr_type(&mut self, expr: &Expr) -> Result<ValueType, SemanticError> {
        match expr {
            Expr::Int { .. } => Ok(ValueType::Int),
            Expr::Float { .. } => Ok(ValueType::Float),
            Expr::Bool { .. } => Ok(ValueType::Bool),
            Expr::Str { .. } => Ok(ValueType::Str),
            Expr::Call { callee, args, span } => {
                let sig =
                    self.functions.get(callee).copied().ok_or_else(|| {
                        Self::error(format!("Undefined function: '{callee}'"), *span)
                    })?;

                if args.len() != sig.arity {
                    return Err(Self::error(
                        format!(
                            "Function '{}' expects {} argument(s), found {}",
                            callee,
                            sig.arity,
                            args.len()
                        ),
                        *span,
                    ));
                }

                for arg in args {
                    let _ = self.infer_expr_type(arg)?;
                }

                Ok(sig.return_type)
            }
            Expr::Ident { name, span } => {
                let ty = self.lookup_variable(name).ok_or_else(|| {
                    Self::error(
                        format!("Variable used before declaration or undefined: '{name}'"),
                        *span,
                    )
                })?;
                self.used_variables.insert(name.clone());
                Ok(ty)
            }
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => {
                let left_ty = self.infer_expr_type(left)?;
                let right_ty = self.infer_expr_type(right)?;

                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                        if left_ty == ValueType::Unknown || right_ty == ValueType::Unknown {
                            return Ok(ValueType::Unknown);
                        }
                        if !is_numeric(left_ty) || !is_numeric(right_ty) {
                            return Err(Self::error(
                                "Arithmetic operators require numeric operands".to_string(),
                                *span,
                            ));
                        }
                        if left_ty == ValueType::Float || right_ty == ValueType::Float {
                            Ok(ValueType::Float)
                        } else {
                            Ok(ValueType::Int)
                        }
                    }
                    BinaryOp::Greater | BinaryOp::Less | BinaryOp::GreaterEq | BinaryOp::LessEq => {
                        if left_ty == ValueType::Unknown || right_ty == ValueType::Unknown {
                            return Ok(ValueType::Bool);
                        }
                        if !is_numeric(left_ty) || !is_numeric(right_ty) {
                            return Err(Self::error(
                                "Comparison operators require numeric operands".to_string(),
                                *span,
                            ));
                        }
                        Ok(ValueType::Bool)
                    }
                    BinaryOp::Equal | BinaryOp::NotEqual => {
                        if left_ty == ValueType::Unknown || right_ty == ValueType::Unknown {
                            return Ok(ValueType::Bool);
                        }
                        if left_ty == right_ty || (is_numeric(left_ty) && is_numeric(right_ty)) {
                            Ok(ValueType::Bool)
                        } else {
                            Err(Self::error(
                                "Equality operators require matching types (numeric mixing allowed)"
                                    .to_string(),
                                *span,
                            ))
                        }
                    }
                }
            }
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        debug_assert!(!self.scopes.is_empty());
    }

    fn pop_scope(&mut self) {
        if !self.scopes.is_empty() {
            self.scopes.pop();
        }
    }

    fn declare_variable(
        &mut self,
        name: String,
        value_type: ValueType,
        span: Span,
    ) -> Result<(), SemanticError> {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, value_type);
            Ok(())
        } else {
            Err(Self::error(
                "Internal semantic error: missing scope".to_string(),
                span,
            ))
        }
    }

    fn any_scope_contains(&self, name: &str) -> bool {
        for scope in self.scopes.iter().rev() {
            if scope.contains_key(name) {
                return true;
            }
        }
        false
    }

    fn lookup_variable(&self, name: &str) -> Option<ValueType> {
        for scope in self.scopes.iter().rev() {
            if let Some(value_type) = scope.get(name) {
                return Some(*value_type);
            }
        }
        None
    }

    fn extract_symbol(&self, value: &Expr) -> Option<String> {
        match value {
            Expr::Str { value, .. } => Some(value.clone()),
            Expr::Ident { name, .. } => Some(name.clone()),
            _ => None,
        }
    }

    fn emit_unused_variable_warnings(&mut self) {
        let mut names: Vec<String> = self.declared_variables.keys().cloned().collect();
        names.sort();

        for name in names {
            if !self.used_variables.contains(&name) {
                let span = self
                    .declared_variables
                    .get(&name)
                    .copied()
                    .unwrap_or(Span::unknown());
                self.warnings.push(SemanticWarning {
                    message: format!("Variable '{}' is declared but never used", name),
                    span,
                });
            }
        }
    }

    fn error(message: String, span: Span) -> SemanticError {
        SemanticError {
            message,
            span,
            hint: None,
        }
    }

    fn error_with_hint(message: String, span: Span, hint: String) -> SemanticError {
        SemanticError {
            message,
            span,
            hint: Some(hint),
        }
    }
}

fn is_numeric(value_type: ValueType) -> bool {
    matches!(value_type, ValueType::Int | ValueType::Float)
}

fn is_assignable(target: ValueType, source: ValueType) -> bool {
    target == ValueType::Unknown
        || source == ValueType::Unknown
        || target == source
        || (target == ValueType::Float && source == ValueType::Int)
}

fn known_property_names(context: &str) -> &'static [&'static str] {
    match context {
        "model" => MODEL_KNOWN_PROPERTIES,
        "dataset" => DATASET_KNOWN_PROPERTIES,
        "train" => TRAIN_KNOWN_PROPERTIES,
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse(source: &str) -> Program {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        parser.parse_program().expect("parse failed")
    }

    #[test]
    fn reports_undefined_variable_with_span() {
        let program = parse("print x\n");
        let mut analyzer = SemanticAnalyzer::new();
        let err = analyzer.analyze(&program).expect_err("must fail");
        assert!(err.message.contains("undefined"));
        assert_eq!(err.span.line, 1);
    }

    #[test]
    fn rejects_shadowing() {
        let src = "x 1\nif true\n    x 2\n";
        let program = parse(src);
        let mut analyzer = SemanticAnalyzer::new();
        let err = analyzer.analyze(&program).expect_err("must fail");
        assert!(err.message.contains("shadowing"));
    }

    #[test]
    fn rejects_assignment_type_mismatch() {
        let src = "x 1\nx = \"s\"\n";
        let program = parse(src);
        let mut analyzer = SemanticAnalyzer::new();
        let err = analyzer.analyze(&program).expect_err("must fail");
        assert!(err.message.contains("Type mismatch"));
    }

    #[test]
    fn warns_on_unused_variable() {
        let program = parse("x 1\n");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program).expect("analysis should pass");
        assert!(
            analyzer
                .warnings()
                .iter()
                .any(|w| w.message.contains("never used"))
        );
    }

    #[test]
    fn warns_on_zero_loop() {
        let program = parse("loop 0\n    print \"x\"\n");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program).expect("analysis should pass");
        assert!(
            analyzer
                .warnings()
                .iter()
                .any(|w| w.message.contains("zero iterations"))
        );
    }

    #[test]
    fn rejects_negative_epochs() {
        let src = "model m\n    layers 2 1\n    activation relu\n    optimizer adam\ndataset d\n    batch 1\ntrain m on d\n    epochs -1\n    device cpu\n";
        let program = parse(src);
        let mut analyzer = SemanticAnalyzer::new();
        let err = analyzer.analyze(&program).expect_err("must fail");
        assert!(err.message.contains("non-negative"));
    }

    #[test]
    fn warns_on_unknown_property() {
        let src = "model m\n    layers 2 1\n    activation relu\n    optimizer adam\n    foo 1\n";
        let program = parse(src);
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program).expect("analysis should pass");
        assert!(
            analyzer
                .warnings()
                .iter()
                .any(|w| w.message.contains("Unknown property"))
        );
    }

    #[test]
    fn rejects_return_outside_function() {
        let program = parse("return 1\n");
        let mut analyzer = SemanticAnalyzer::new();
        let err = analyzer.analyze(&program).expect_err("must fail");
        assert!(err.message.contains("only allowed inside function"));
    }

    #[test]
    fn rejects_function_call_arity_mismatch() {
        let src = "fn add(a, b)\n    return a + b\nx add(1)\n";
        let program = parse(src);
        let mut analyzer = SemanticAnalyzer::new();
        let err = analyzer.analyze(&program).expect_err("must fail");
        assert!(err.message.contains("expects 2 argument"));
    }

    #[test]
    fn allows_train_without_properties_for_autopilot() {
        let src = "model m\n    layers 2 1\ndataset d\ntrain m on d\n";
        let program = parse(src);
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program).expect("analysis should pass");
    }

    #[test]
    fn rejects_negative_train_batch() {
        let src = "model m\n    layers 2 1\ndataset d\ntrain m on d\n    batch -1\n";
        let program = parse(src);
        let mut analyzer = SemanticAnalyzer::new();
        let err = analyzer.analyze(&program).expect_err("must fail");
        assert!(err.message.contains("must be > 0"));
    }
}
