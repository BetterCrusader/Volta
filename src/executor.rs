//! Runtime invariants:
//! - Scope stack is never popped below zero.
//! - Variable declarations never shadow outer scopes.
//! - Assignment updates nearest visible binding.
//! - Model/dataset maps contain unique names.
//! - Numeric operations are checked for overflow, infinity, and division-by-zero.

use std::collections::{HashMap, HashSet};

use crate::ast::{BinaryOp, Expr, Program, Property, Span, Stmt};
use crate::autopilot::{
    AutopilotContext, AutopilotEngine, DatasetAutopilotInput, ModelAutopilotInput,
    TrainAutopilotInput,
};
use crate::diagnostics::best_suggestion;
use crate::rules::{
    ACTIVATIONS, AUTOPILOT_DEFAULT_ACTIVATION, DEVICES, FLOAT_EPSILON, MAX_SAFE_INT_F64, OPTIMIZERS,
};

/// A runtime value produced by evaluating a Volta expression.
///
/// All arithmetic is performed on `Int` and `Float` variants.
/// `Str` holds UTF-8 text; `Bool` stores boolean conditions;
/// `Unit` represents the absence of a meaningful return value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Signed 64-bit integer.
    Int(i64),
    /// 64-bit IEEE-754 floating-point number.
    Float(f64),
    /// Boolean true/false.
    Bool(bool),
    /// UTF-8 string.
    Str(String),
    /// The unit value, analogous to `()` in Rust.
    Unit,
}

/// Persistent state of a declared `model` block after successful parsing.
#[derive(Debug, Clone)]
pub struct ModelState {
    /// Number of neurons in each layer (at least 2 values required).
    pub layers: Vec<i64>,
    /// Activation function name, e.g. `"relu"` or `"sigmoid"`.
    pub activation: String,
    /// Optional optimizer name, e.g. `"adam"`.
    pub optimizer: Option<String>,
    /// Optional learning rate for the model-level optimizer.
    pub optimizer_lr: Option<f64>,
    /// Optional numerical precision tag, e.g. `"fp32"`.
    pub precision: Option<String>,
    /// Optional memory layout tag, e.g. `"pinned"`.
    pub memory: Option<String>,
    /// Total number of epochs this model has been trained for across all `train` calls.
    pub trained_epochs: i64,
}

/// Persistent state of a declared `dataset` block after successful parsing.
#[derive(Debug, Clone)]
pub struct DatasetState {
    /// Number of samples per training batch, if specified.
    pub batch: Option<i64>,
    /// Whether to shuffle the dataset before each epoch, if specified.
    pub shuffle: Option<bool>,
}

/// A runtime error produced during execution of a Volta program.
///
/// Always carries a human-readable `message`, the source `span` for
/// line/column reporting, and an optional `hint` with a suggested fix.
#[derive(Debug, Clone)]
pub struct RuntimeError {
    /// Human-readable description of the error.
    pub message: String,
    /// Source location at which the error occurred.
    pub span: Span,
    /// Optional suggestion to help the user fix the problem.
    pub hint: Option<String>,
}

struct Runtime {
    variables: Vec<HashMap<String, Value>>,
    models: HashMap<String, ModelState>,
    datasets: HashMap<String, DatasetState>,
    functions: HashMap<String, FunctionDef>,
}

#[derive(Debug, Clone)]
struct FunctionDef {
    params: Vec<String>,
    body: Vec<Stmt>,
    span: Span,
}

#[derive(Debug, Clone, Default)]
struct TrainPropsExplicit {
    epochs: Option<i64>,
    device: Option<String>,
    optimizer: Option<String>,
    lr: Option<f64>,
    batch: Option<i64>,
    precision: Option<String>,
}

/// The main Volta interpreter.
///
/// Call [`Executor::new`] to create an instance, then [`Executor::execute`]
/// to run a parsed [`Program`]. The executor is stateful — models and
/// datasets accumulate across multiple `execute` calls on the same instance.
pub struct Executor {
    runtime: Runtime,
    in_function_depth: usize,
    return_slot: Option<Value>,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    /// Creates a new `Executor` with empty runtime state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            runtime: Runtime {
                variables: Vec::new(),
                models: HashMap::new(),
                datasets: HashMap::new(),
                functions: HashMap::new(),
            },
            in_function_depth: 0,
            return_slot: None,
        }
    }

    /// Executes a Volta [`Program`], resetting all runtime state first.
    ///
    /// Returns `Ok(())` on success, or a [`RuntimeError`] describing the
    /// first failure along with its source location and optional fix hint.
    ///
    /// # Errors
    ///
    /// Returns `Err` when:
    /// - a variable or function is used before declaration
    /// - arithmetic overflow, division by zero, or non-finite result
    /// - a `model` or `dataset` block contains invalid property values
    /// - a `train` statement references an undefined model or dataset
    pub fn execute(&mut self, program: &Program) -> Result<(), RuntimeError> {
        self.runtime.variables.clear();
        self.runtime.models.clear();
        self.runtime.datasets.clear();
        self.runtime.functions.clear();
        self.in_function_depth = 0;
        self.return_slot = None;

        self.push_scope();
        let result = self.exec_block_no_scope(&program.statements);
        let pop_result = self.pop_scope();

        match (result, pop_result) {
            (Err(err), _) | (Ok(()), Err(err)) => Err(err),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    fn exec_block_no_scope(&mut self, statements: &[Stmt]) -> Result<(), RuntimeError> {
        for stmt in statements {
            self.exec_stmt(stmt)?;
            if self.return_slot.is_some() {
                break;
            }
        }
        Ok(())
    }

    fn exec_block_with_scope(&mut self, statements: &[Stmt]) -> Result<(), RuntimeError> {
        self.push_scope();
        let result = self.exec_block_no_scope(statements);
        self.pop_scope()?;
        result
    }

    #[allow(clippy::too_many_lines)]
    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<(), RuntimeError> {
        match stmt {
            Stmt::VarDecl { name, value, span } => {
                let evaluated = self.eval_expr(value)?;
                self.set_var(name.clone(), evaluated, *span)?;
            }
            Stmt::Assign { name, value, span } => {
                let evaluated = self.eval_expr(value)?;
                self.assign_var(name, evaluated, *span)?;
            }
            Stmt::Print { expr, span } => {
                let _ = span;
                let value = self.eval_expr(expr)?;
                println!("{}", value.to_display());
            }
            Stmt::Function {
                name,
                params,
                body,
                span,
            } => {
                if self.runtime.functions.contains_key(name) {
                    return Err(Self::error(format!("Duplicate function: '{name}'"), *span));
                }
                self.runtime.functions.insert(
                    name.clone(),
                    FunctionDef {
                        params: params.clone(),
                        body: body.clone(),
                        span: *span,
                    },
                );
            }
            Stmt::Return { value, span } => {
                if self.in_function_depth == 0 {
                    return Err(Self::error(
                        "Return statement is only allowed inside function".to_string(),
                        *span,
                    ));
                }

                let result = if let Some(expr) = value {
                    self.eval_expr(expr)?
                } else {
                    Value::Unit
                };
                self.return_slot = Some(result);
            }
            Stmt::If {
                condition,
                then_branch,
                elif_branches,
                else_branch,
                ..
            } => self.exec_if_stmt(condition, then_branch, elif_branches, else_branch)?,
            Stmt::Loop { count, body, span } => {
                let count_value = self.eval_expr(count)?;
                let iterations = Self::as_int(&count_value, *span)?;
                if iterations < 0 {
                    return Err(Self::error(
                        format!("Invalid loop count: {iterations} (must be non-negative)"),
                        *span,
                    ));
                }

                let loop_count = usize::try_from(iterations)
                    .map_err(|_| Self::error("Loop count out of usize range".to_string(), *span))?;

                for _ in 0..loop_count {
                    self.exec_block_with_scope(body)?;
                    if self.return_slot.is_some() {
                        break;
                    }
                }
            }
            Stmt::Model { name, props, span } => {
                let model = self.build_model_state(name, props, *span)?;
                if self.runtime.models.insert(name.clone(), model).is_some() {
                    return Err(Self::error(format!("Duplicate model: '{name}'"), *span));
                }
            }
            Stmt::Dataset { name, props, span } => {
                let dataset = self.build_dataset_state(name, props, *span)?;
                if self
                    .runtime
                    .datasets
                    .insert(name.clone(), dataset)
                    .is_some()
                {
                    return Err(Self::error(format!("Duplicate dataset: '{name}'"), *span));
                }
            }
            Stmt::Train {
                model,
                data,
                props,
                span,
            } => self.exec_train_stmt(model, data, props, *span)?,
            Stmt::Save { model, path, span } => {
                if !self.runtime.models.contains_key(model) {
                    return Err(Self::error_with_hint(
                        format!("Undefined model: '{model}'"),
                        *span,
                        "Define the model before saving it in this run.".to_string(),
                    ));
                }
                println!("Saving model '{model}' to {path}");
            }
            Stmt::Load { model, path, span } => {
                if !self.runtime.models.contains_key(model) {
                    return Err(Self::error_with_hint(
                        format!("Undefined model: '{model}'"),
                        *span,
                        "Declare the model name first, then load weights into it.".to_string(),
                    ));
                }
                println!("Loading model '{model}' from {path}");
            }
        }

        Ok(())
    }

    fn exec_if_stmt(
        &mut self,
        condition: &Expr,
        then_branch: &[Stmt],
        elif_branches: &[(Expr, Vec<Stmt>)],
        else_branch: &Option<Vec<Stmt>>,
    ) -> Result<(), RuntimeError> {
        let cond_value = self.eval_expr(condition)?;
        if Self::as_bool(&cond_value, condition.span())? {
            self.exec_block_with_scope(then_branch)?;
            return Ok(());
        }

        for (elif_condition, branch) in elif_branches {
            let elif_value = self.eval_expr(elif_condition)?;
            if Self::as_bool(&elif_value, elif_condition.span())? {
                self.exec_block_with_scope(branch)?;
                return Ok(());
            }
        }

        if let Some(branch) = else_branch {
            self.exec_block_with_scope(branch)?;
        }

        Ok(())
    }

    fn exec_train_stmt(
        &mut self,
        model: &str,
        data: &str,
        props: &[Property],
        span: Span,
    ) -> Result<(), RuntimeError> {
        if !self.runtime.datasets.contains_key(data) {
            return Err(Self::error_with_hint(
                format!("Undefined dataset: '{data}'"),
                span,
                format!(
                    "Declare dataset '{data}' before training, for example:\ndataset {data}\n    batch 32"
                ),
            ));
        }

        let explicit_train = self.parse_train_props(props, span)?;

        let dataset_state = self
            .runtime
            .datasets
            .get(data)
            .ok_or_else(|| Self::error(format!("Undefined dataset: '{data}'"), span))?
            .clone();

        let model_state = self.runtime.models.get_mut(model).ok_or_else(|| {
            Self::error_with_hint(
                format!("Undefined model: '{model}'"),
                span,
                format!(
                    "Declare model '{model}' before training, for example:\nmodel {model}\n    layers 4 3 2"
                ),
            )
        })?;

        let autopilot = AutopilotEngine::new();
        let resolved = autopilot.resolve(&AutopilotContext {
            model: ModelAutopilotInput {
                layer_count: model_state.layers.len(),
                optimizer: model_state.optimizer.clone(),
                lr: model_state.optimizer_lr,
                precision: model_state.precision.clone(),
            },
            dataset: DatasetAutopilotInput {
                batch: dataset_state.batch,
            },
            train: TrainAutopilotInput {
                epochs: explicit_train.epochs,
                optimizer: explicit_train.optimizer.clone(),
                lr: explicit_train.lr,
                batch: explicit_train.batch,
                precision: explicit_train.precision.clone(),
                device: explicit_train.device.clone(),
            },
            gpu_available: detect_gpu_available(),
        });

        model_state.trained_epochs = model_state
            .trained_epochs
            .checked_add(resolved.epochs)
            .ok_or_else(|| {
                Self::error("Math overflow: trained_epochs overflow".to_string(), span)
            })?;

        println!("Auto configuration:");
        for decision in &resolved.decisions {
            println!(
                "    {} = {} ({:?}: {})",
                decision.key, decision.value, decision.source, decision.reason
            );
        }

        println!(
            "Training model '{}' on dataset '{}' for {} epoch(s) on {} with {} lr={} batch={} precision={}",
            model,
            data,
            resolved.epochs,
            resolved.device,
            resolved.optimizer,
            resolved.lr,
            resolved.batch,
            resolved.precision
        );

        Ok(())
    }

    fn build_model_state(
        &mut self,
        model_name: &str,
        props: &[Property],
        span: Span,
    ) -> Result<ModelState, RuntimeError> {
        Self::ensure_unique_properties(props, "model", span)?;

        let layers_expr = Self::get_required_prop(props, "layers", "model", span)?;
        if layers_expr.len() < 2 {
            return Err(Self::error(
                "model.layers must have at least 2 values".to_string(),
                span,
            ));
        }

        let mut layers = Vec::with_capacity(layers_expr.len());
        for expr in layers_expr {
            let v = self.expect_int_value(expr, "model.layers")?;
            // BUG FIX: layer sizes of 0 or negative are meaningless and would
            // silently produce degenerate models downstream.
            if v <= 0 {
                return Err(Self::error(
                    format!("model.layers values must be > 0, found {v}"),
                    expr.span(),
                ));
            }
            layers.push(v);
        }

        let activation = self
            .expect_single_symbol_optional(props, "activation", span)?
            .unwrap_or_else(|| AUTOPILOT_DEFAULT_ACTIVATION.to_string());
        Self::ensure_allowed_symbol("model.activation", &activation, ACTIVATIONS, span)?;

        let (optimizer, optimizer_lr) = self.parse_optimizer_prop(props, span)?;
        let precision = self.expect_single_symbol_optional(props, "precision", span)?;
        let memory = self.expect_single_symbol_optional(props, "memory", span)?;

        let _ = model_name;

        Ok(ModelState {
            layers,
            activation,
            optimizer,
            optimizer_lr,
            precision,
            memory,
            trained_epochs: 0,
        })
    }

    fn build_dataset_state(
        &mut self,
        _dataset_name: &str,
        props: &[Property],
        span: Span,
    ) -> Result<DatasetState, RuntimeError> {
        Self::ensure_unique_properties(props, "dataset", span)?;

        let batch = match Self::find_prop(props, "batch") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "dataset.batch expects exactly one value".to_string(),
                        span,
                    ));
                }
                Some(self.expect_int_value(&values[0], "dataset.batch")?)
            }
            None => None,
        };

        let shuffle = match Self::find_prop(props, "shuffle") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "dataset.shuffle expects exactly one value".to_string(),
                        span,
                    ));
                }
                Some(self.expect_bool_value(&values[0], "dataset.shuffle")?)
            }
            None => None,
        };

        Ok(DatasetState { batch, shuffle })
    }

    #[allow(clippy::too_many_lines)]
    fn parse_train_props(
        &mut self,
        props: &[Property],
        span: Span,
    ) -> Result<TrainPropsExplicit, RuntimeError> {
        Self::ensure_unique_properties(props, "train", span)?;

        let epochs = match Self::find_prop(props, "epochs") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "train.epochs expects exactly one value".to_string(),
                        span,
                    ));
                }
                let parsed = self.expect_int_value(&values[0], "train.epochs")?;
                if parsed < 0 {
                    return Err(Self::error(
                        "train.epochs must be non-negative".to_string(),
                        values[0].span(),
                    ));
                }
                Some(parsed)
            }
            None => None,
        };

        let device = match Self::find_prop(props, "device") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "train.device expects exactly one value".to_string(),
                        span,
                    ));
                }
                let parsed = self.expect_symbol_value(&values[0], "train.device")?;
                Self::ensure_allowed_symbol("train.device", &parsed, DEVICES, values[0].span())?;
                Some(parsed)
            }
            None => None,
        };

        let optimizer = match Self::find_prop(props, "optimizer") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "train.optimizer expects exactly one value".to_string(),
                        span,
                    ));
                }
                let parsed = self.expect_symbol_value(&values[0], "train.optimizer")?;
                Self::ensure_allowed_symbol(
                    "train.optimizer",
                    &parsed,
                    OPTIMIZERS,
                    values[0].span(),
                )?;
                Some(parsed)
            }
            None => None,
        };

        let lr = match Self::find_prop(props, "lr") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "train.lr expects exactly one value".to_string(),
                        span,
                    ));
                }
                let raw = self.eval_expr(&values[0])?;
                let parsed = Self::to_f64(&raw, values[0].span())?;
                if parsed <= 0.0 {
                    return Err(Self::error(
                        "train.lr must be > 0".to_string(),
                        values[0].span(),
                    ));
                }
                Some(parsed)
            }
            None => None,
        };

        let batch = match Self::find_prop(props, "batch") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "train.batch expects exactly one value".to_string(),
                        span,
                    ));
                }
                let parsed = self.expect_int_value(&values[0], "train.batch")?;
                if parsed <= 0 {
                    return Err(Self::error(
                        "train.batch must be > 0".to_string(),
                        values[0].span(),
                    ));
                }
                Some(parsed)
            }
            None => None,
        };

        let precision = match Self::find_prop(props, "precision") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "train.precision expects exactly one value".to_string(),
                        span,
                    ));
                }
                Some(self.expect_symbol_value(&values[0], "train.precision")?)
            }
            None => None,
        };

        Ok(TrainPropsExplicit {
            epochs,
            device,
            optimizer,
            lr,
            batch,
            precision,
        })
    }

    fn parse_optimizer_prop(
        &mut self,
        props: &[Property],
        span: Span,
    ) -> Result<(Option<String>, Option<f64>), RuntimeError> {
        if let Some(values) = Self::find_prop(props, "optimizer") {
            if values.is_empty() || values.len() > 2 {
                return Err(Self::error(
                    "model.optimizer expects 1 or 2 values".to_string(),
                    span,
                ));
            }

            let optimizer = self.expect_symbol_value(&values[0], "model.optimizer")?;
            Self::ensure_allowed_symbol(
                "model.optimizer",
                &optimizer,
                OPTIMIZERS,
                values[0].span(),
            )?;

            let optimizer_lr = if values.len() == 2 {
                let lr_value = self.eval_expr(&values[1])?;
                let lr_float = Self::to_f64(&lr_value, values[1].span())?;
                if lr_float <= 0.0 {
                    return Err(Self::error(
                        "model.optimizer learning rate must be > 0".to_string(),
                        values[1].span(),
                    ));
                }
                Some(lr_float)
            } else {
                None
            };

            Ok((Some(optimizer), optimizer_lr))
        } else {
            Ok((None, None))
        }
    }

    fn ensure_unique_properties(
        props: &[Property],
        context: &str,
        _span: Span,
    ) -> Result<(), RuntimeError> {
        let mut seen = HashSet::new();
        for prop in props {
            if !seen.insert(prop.key.as_str()) {
                return Err(Self::error(
                    format!("duplicate '{}' in {} block", prop.key, context),
                    prop.span,
                ));
            }
        }
        Ok(())
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, RuntimeError> {
        match expr {
            Expr::Int { value, .. } => Ok(Value::Int(*value)),
            Expr::Float { value, .. } => Ok(Value::Float(*value)),
            Expr::Bool { value, .. } => Ok(Value::Bool(*value)),
            Expr::Str { value, .. } => Ok(Value::Str(value.clone())),
            Expr::Call { callee, args, span } => self.invoke_function(callee, args, *span),
            Expr::Ident { name, span } => self.get_var(name).ok_or_else(|| {
                Self::error_with_hint(
                    format!("Undefined variable: '{name}'"),
                    *span,
                    format!("Declare '{name}' before using it in expressions."),
                )
            }),
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => {
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                self.eval_binary(*op, &l, &r, *span)
            }
        }
    }

    fn eval_binary(
        &self,
        op: BinaryOp,
        lhs: &Value,
        rhs: &Value,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        match op {
            BinaryOp::Add => Self::eval_add(lhs, rhs, span),
            BinaryOp::Sub => Self::eval_sub(lhs, rhs, span),
            BinaryOp::Mul => Self::eval_mul(lhs, rhs, span),
            BinaryOp::Div => Self::eval_div(lhs, rhs, span),
            BinaryOp::Greater => Self::eval_cmp(lhs, rhs, CmpOp::Greater, span),
            BinaryOp::Less => Self::eval_cmp(lhs, rhs, CmpOp::Less, span),
            BinaryOp::GreaterEq => Self::eval_cmp(lhs, rhs, CmpOp::GreaterEq, span),
            BinaryOp::LessEq => Self::eval_cmp(lhs, rhs, CmpOp::LessEq, span),
            BinaryOp::Equal => Self::eval_eq(lhs, rhs, true, span),
            BinaryOp::NotEqual => Self::eval_eq(lhs, rhs, false, span),
        }
    }

    fn eval_add(left: &Value, right: &Value, span: Span) -> Result<Value, RuntimeError> {
        if let (Value::Int(a), Value::Int(b)) = (left, right) {
            return a.checked_add(*b).map(Value::Int).ok_or_else(|| {
                Self::error("Math overflow: int addition overflow".to_string(), span)
            });
        }
        let result = Self::to_f64(left, span)? + Self::to_f64(right, span)?;
        Self::finite_float(result, "Math overflow: float addition overflow", span).map(Value::Float)
    }

    fn eval_sub(left: &Value, right: &Value, span: Span) -> Result<Value, RuntimeError> {
        if let (Value::Int(a), Value::Int(b)) = (left, right) {
            return a.checked_sub(*b).map(Value::Int).ok_or_else(|| {
                Self::error("Math overflow: int subtraction overflow".to_string(), span)
            });
        }
        let result = Self::to_f64(left, span)? - Self::to_f64(right, span)?;
        Self::finite_float(result, "Math overflow: float subtraction overflow", span)
            .map(Value::Float)
    }

    fn eval_mul(left: &Value, right: &Value, span: Span) -> Result<Value, RuntimeError> {
        if let (Value::Int(a), Value::Int(b)) = (left, right) {
            return a.checked_mul(*b).map(Value::Int).ok_or_else(|| {
                Self::error(
                    "Math overflow: int multiplication overflow".to_string(),
                    span,
                )
            });
        }
        let result = Self::to_f64(left, span)? * Self::to_f64(right, span)?;
        Self::finite_float(result, "Math overflow: float multiplication overflow", span)
            .map(Value::Float)
    }

    fn eval_div(left: &Value, right: &Value, span: Span) -> Result<Value, RuntimeError> {
        let numerator = Self::to_f64(left, span)?;
        let denominator = Self::to_f64(right, span)?;
        if denominator.abs() < f64::EPSILON {
            return Err(Self::error_with_hint(
                "Division by zero".to_string(),
                span,
                "Guard the divisor with a non-zero check before division.".to_string(),
            ));
        }
        let result = numerator / denominator;
        Self::finite_float(result, "Math overflow: float division overflow", span).map(Value::Float)
    }

    fn eval_cmp(left: &Value, right: &Value, op: CmpOp, span: Span) -> Result<Value, RuntimeError> {
        let l = Self::to_f64(left, span)?;
        let r = Self::to_f64(right, span)?;

        let result = match op {
            CmpOp::Greater => l > r,
            CmpOp::Less => l < r,
            CmpOp::GreaterEq => l >= r,
            CmpOp::LessEq => l <= r,
        };

        Ok(Value::Bool(result))
    }

    fn eval_eq(
        left: &Value,
        right: &Value,
        equality: bool,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let result = match (left, right) {
            (Value::Int(_), Value::Float(_)) | (Value::Float(_), Value::Int(_)) => {
                (Self::to_f64(left, span)? - Self::to_f64(right, span)?).abs() <= FLOAT_EPSILON
            }
            (Value::Float(a), Value::Float(b)) => (a - b).abs() <= FLOAT_EPSILON,
            _ => left == right,
        };

        Ok(Value::Bool(if equality { result } else { !result }))
    }

    fn to_f64(value: &Value, span: Span) -> Result<f64, RuntimeError> {
        match value {
            Value::Int(v) => {
                let magnitude = v.checked_abs().map_or(i64::MAX, |abs| abs);
                if magnitude > MAX_SAFE_INT_F64 {
                    return Err(Self::error(
                        "Invalid operation: lossy int-to-float coercion for value outside f64 safe integer range"
                            .to_string(),
                        span,
                    ));
                }
                #[allow(clippy::cast_precision_loss)]
                let f = *v as f64;
                Ok(f)
            }
            Value::Float(v) => Ok(*v),
            _ => Err(Self::error(
                "Invalid operation: numeric operator requires numeric operands".to_string(),
                span,
            )),
        }
    }

    fn as_bool(value: &Value, span: Span) -> Result<bool, RuntimeError> {
        match value {
            Value::Bool(v) => Ok(*v),
            _ => Err(Self::error_with_hint(
                "Invalid operation: condition must evaluate to boolean".to_string(),
                span,
                "Use a comparison such as `x > 0` or `flag == true`.".to_string(),
            )),
        }
    }

    fn as_int(value: &Value, span: Span) -> Result<i64, RuntimeError> {
        match value {
            Value::Int(v) => Ok(*v),
            _ => Err(Self::error_with_hint(
                "Invalid operation: loop count must evaluate to integer".to_string(),
                span,
                "Loop counts must be integer literals or integer variables.".to_string(),
            )),
        }
    }

    fn expect_int_value(&mut self, expr: &Expr, label: &str) -> Result<i64, RuntimeError> {
        match self.eval_expr(expr)? {
            Value::Int(v) => Ok(v),
            other => Err(Self::error(
                format!(
                    "Invalid property: {label} expects int, found {}",
                    other.type_name()
                ),
                expr.span(),
            )),
        }
    }

    fn expect_bool_value(&mut self, expr: &Expr, label: &str) -> Result<bool, RuntimeError> {
        match self.eval_expr(expr)? {
            Value::Bool(v) => Ok(v),
            other => Err(Self::error(
                format!(
                    "Invalid property: {label} expects bool, found {}",
                    other.type_name()
                ),
                expr.span(),
            )),
        }
    }

    fn expect_symbol_value(&self, expr: &Expr, label: &str) -> Result<String, RuntimeError> {
        match expr {
            Expr::Ident { name, span } => {
                if let Some(value) = self.get_var(name) {
                    match value {
                        Value::Str(text) => Ok(text),
                        other => Err(Self::error(
                            format!(
                                "Invalid property: {label} variable '{name}' must be string, found {}",
                                other.type_name()
                            ),
                            *span,
                        )),
                    }
                } else {
                    Ok(name.clone())
                }
            }
            Expr::Str { value, .. } => Ok(value.clone()),
            _ => Err(Self::error(
                format!("Invalid property: {label} expects identifier or string"),
                expr.span(),
            )),
        }
    }

    fn expect_single_symbol_optional(
        &self,
        props: &[Property],
        key: &str,
        span: Span,
    ) -> Result<Option<String>, RuntimeError> {
        if let Some(values) = Self::find_prop(props, key) {
            if values.len() != 1 {
                return Err(Self::error(
                    format!("model.{key} expects exactly one value"),
                    span,
                ));
            }
            let parsed = self.expect_symbol_value(&values[0], &format!("model.{key}"))?;
            Ok(Some(parsed))
        } else {
            Ok(None)
        }
    }

    fn find_prop<'a>(props: &'a [Property], key: &str) -> Option<&'a [Expr]> {
        for prop in props {
            if prop.key == key {
                return Some(prop.values.as_slice());
            }
        }
        None
    }

    fn get_required_prop<'a>(
        props: &'a [Property],
        key: &str,
        context: &str,
        span: Span,
    ) -> Result<&'a [Expr], RuntimeError> {
        Self::find_prop(props, key).ok_or_else(|| {
            Self::error(
                format!("Invalid property: missing required property '{key}' in {context}"),
                span,
            )
        })
    }

    fn ensure_allowed_symbol(
        label: &str,
        value: &str,
        allowed: &[&str],
        span: Span,
    ) -> Result<(), RuntimeError> {
        if allowed.contains(&value) {
            Ok(())
        } else {
            let suggestion = best_suggestion(value, allowed);
            let joined = allowed.join(", ");
            let hint = match suggestion {
                Some(candidate) => format!("Did you mean '{candidate}'? Allowed values: {joined}"),
                None => format!("Allowed values: {joined}"),
            };
            Err(Self::error_with_hint(
                format!("Invalid property: {label} has invalid value '{value}'; allowed: {joined}"),
                span,
                hint,
            ))
        }
    }

    fn finite_float(value: f64, context: &str, span: Span) -> Result<f64, RuntimeError> {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Self::error(context.to_string(), span))
        }
    }

    fn push_scope(&mut self) {
        self.runtime.variables.push(HashMap::new());
        debug_assert!(!self.runtime.variables.is_empty());
    }

    fn invoke_function(
        &mut self,
        callee: &str,
        args: &[Expr],
        span: Span,
    ) -> Result<Value, RuntimeError> {
        // BUG FIX: unbounded recursion would cause a stack overflow. Cap at 128.
        const MAX_CALL_DEPTH: usize = 128;
        if self.in_function_depth >= MAX_CALL_DEPTH {
            return Err(Self::error(
                format!(
                    "Call stack overflow: function '{callee}' exceeded maximum call depth of {MAX_CALL_DEPTH}"
                ),
                span,
            ));
        }

        let function = self.runtime.functions.get(callee).cloned().ok_or_else(|| {
            Self::error_with_hint(
                format!("Undefined function: '{callee}'"),
                span,
                format!("Declare `fn {callee}(...)` before calling it."),
            )
        })?;

        if args.len() != function.params.len() {
            return Err(Self::error(
                format!(
                    "Function '{}' expects {} argument(s), found {}",
                    callee,
                    function.params.len(),
                    args.len()
                ),
                span,
            ));
        }

        let _ = function.span;
        let previous_return = self.return_slot.take();
        self.push_scope();
        // BUG FIX: use a scopeguard-style pattern so in_function_depth is
        // decremented and scope is popped even when exec_block_no_scope errors.
        self.in_function_depth += 1;

        let exec_result = (|| {
            for (param, arg_expr) in function.params.iter().zip(args.iter()) {
                let arg_value = self.eval_expr(arg_expr)?;
                if let Some(scope) = self.runtime.variables.last_mut() {
                    scope.insert(param.clone(), arg_value);
                }
            }
            self.exec_block_no_scope(&function.body)
        })();

        // Always restore depth and pop scope regardless of success/failure.
        self.in_function_depth = self.in_function_depth.saturating_sub(1);
        self.pop_scope()?;

        exec_result?;

        let ret = self.return_slot.take().unwrap_or(Value::Unit);
        self.return_slot = previous_return;
        Ok(ret)
    }

    fn pop_scope(&mut self) -> Result<(), RuntimeError> {
        if self.runtime.variables.pop().is_some() {
            Ok(())
        } else {
            Err(Self::error(
                "Internal runtime error: scope underflow".to_string(),
                Span::unknown(),
            ))
        }
    }

    fn set_var(&mut self, name: String, value: Value, span: Span) -> Result<(), RuntimeError> {
        if self.runtime.variables.is_empty() {
            return Err(Self::error(
                "Internal runtime error: scope underflow".to_string(),
                span,
            ));
        }

        for existing_scope in self.runtime.variables.iter().rev() {
            if existing_scope.contains_key(&name) {
                return Err(Self::error(
                    format!("Variable already declared (shadowing forbidden): '{name}'"),
                    span,
                ));
            }
        }

        if let Some(scope) = self.runtime.variables.last_mut() {
            scope.insert(name, value);
            Ok(())
        } else {
            Err(Self::error(
                "Internal runtime error: scope underflow".to_string(),
                span,
            ))
        }
    }

    fn assign_var(&mut self, name: &str, value: Value, span: Span) -> Result<(), RuntimeError> {
        for scope in self.runtime.variables.iter_mut().rev() {
            if let Some(current) = scope.get(name) {
                if !is_runtime_assignable(current, &value) {
                    return Err(Self::error(
                        format!(
                            "Invalid operation: assignment type mismatch for '{}': expected {}, found {}",
                            name,
                            current.type_name(),
                            value.type_name()
                        ),
                        span,
                    ));
                }

                let stored = if matches!(current, Value::Float(_)) {
                    #[allow(clippy::cast_precision_loss)]
                    if let Value::Int(v) = value {
                        Value::Float(v as f64)
                    } else {
                        value
                    }
                } else {
                    value
                };

                scope.insert(name.to_string(), stored);
                return Ok(());
            }
        }

        Err(Self::error_with_hint(
            format!("Undefined variable: '{name}'"),
            span,
            format!("Declare '{name}' before assigning to it."),
        ))
    }

    fn get_var(&self, name: &str) -> Option<Value> {
        for scope in self.runtime.variables.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value.clone());
            }
        }
        None
    }

    #[must_use]
    fn error(message: String, span: Span) -> RuntimeError {
        RuntimeError {
            message,
            span,
            hint: None,
        }
    }

    #[must_use]
    fn error_with_hint(message: String, span: Span, hint: String) -> RuntimeError {
        RuntimeError {
            message,
            span,
            hint: Some(hint),
        }
    }
}

impl Value {
    #[must_use]
    fn type_name(&self) -> &'static str {
        match self {
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Bool(_) => "bool",
            Self::Str(_) => "str",
            Self::Unit => "unit",
        }
    }

    #[must_use]
    fn to_display(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Float(v) => v.to_string(),
            Self::Bool(v) => v.to_string(),
            Self::Str(v) => v.clone(),
            Self::Unit => String::new(),
        }
    }
}

fn is_runtime_assignable(target: &Value, source: &Value) -> bool {
    target.type_name() == source.type_name()
        || (matches!(target, Value::Float(_)) && matches!(source, Value::Int(_)))
}

fn detect_gpu_available() -> bool {
    std::env::var("VOLTA_GPU_AVAILABLE")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

#[derive(Debug, Clone, Copy)]
enum CmpOp {
    Greater,
    Less,
    GreaterEq,
    LessEq,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::semantic::SemanticAnalyzer;

    fn parse(source: &str) -> Program {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        parser.parse_program().expect("parse failed")
    }

    fn analyze(program: &Program) {
        let mut sem = SemanticAnalyzer::new();
        sem.analyze(program).expect("semantic failed")
    }

    #[test]
    fn executes_assignment_chain() {
        let program = parse("x 1\nx = x + 2\n");
        analyze(&program);
        let mut ex = Executor::new();
        ex.execute(&program).expect("runtime failed");
    }

    #[test]
    fn fails_on_int_overflow() {
        let program = parse("x 9223372036854775807\ny x + 1\n");
        analyze(&program);
        let mut ex = Executor::new();
        let err = ex.execute(&program).expect_err("must overflow");
        assert!(err.message.contains("overflow"));
        assert_eq!(err.span.line, 2);
    }

    #[test]
    fn fails_on_division_by_zero() {
        let program = parse("x 1 / 0\n");
        analyze(&program);
        let mut ex = Executor::new();
        let err = ex.execute(&program).expect_err("must fail");
        assert!(err.message.contains("Division by zero"));
    }

    #[test]
    fn fails_on_runtime_invalid_device() {
        let src = "model m\n    layers 2 1\n    activation relu\n    optimizer adam\ndataset d\n    batch 1\ntrain m on d\n    epochs 1\n    device quantum\n";
        let program = parse(src);
        let mut ex = Executor::new();
        let err = ex.execute(&program).expect_err("must fail");
        assert!(err.message.contains("invalid value"));
    }

    #[test]
    fn fails_on_lossy_int_to_float_coercion() {
        let huge = "9007199254740993";
        let src = format!("x {}\ny x / 2\n", huge);
        let program = parse(&src);
        analyze(&program);
        let mut ex = Executor::new();
        let err = ex.execute(&program).expect_err("must fail");
        assert!(err.message.contains("lossy int-to-float"));
    }

    #[test]
    fn nearest_scope_assignment_updates_visible_binding() {
        let src = "x 1\nloop 1\n    x = 2\n";
        let program = parse(src);
        analyze(&program);
        let mut ex = Executor::new();
        ex.execute(&program).expect("runtime failed");
    }

    #[test]
    fn runtime_state_and_value_variants_are_exercised() {
        let src = "model m\n    layers 3 2\n    activation relu\n    optimizer adam\n    precision auto\n    memory balanced\ndataset d\n    batch 16\n    shuffle true\ntrain m on d\n    epochs 2\n    device cpu\n";
        let program = parse(src);
        analyze(&program);

        let mut ex = Executor::new();
        ex.execute(&program).expect("runtime failed");

        let model = ex.runtime.models.get("m").expect("model exists");
        assert_eq!(model.layers, vec![3, 2]);
        assert_eq!(model.activation, "relu");
        assert_eq!(model.optimizer.as_deref(), Some("adam"));
        assert_eq!(model.optimizer_lr, None);
        assert_eq!(model.precision.as_deref(), Some("auto"));
        assert_eq!(model.memory.as_deref(), Some("balanced"));
        assert_eq!(model.trained_epochs, 2);

        let dataset = ex.runtime.datasets.get("d").expect("dataset exists");
        assert_eq!(dataset.batch, Some(16));
        assert_eq!(dataset.shuffle, Some(true));

        let intv = Value::Int(1);
        let boolv = Value::Bool(true);
        let strv = Value::Str(String::from("x"));
        assert_eq!(intv.type_name(), "int");
        assert_eq!(boolv.type_name(), "bool");
        assert_eq!(strv.type_name(), "str");
    }

    #[test]
    fn executes_function_call_and_return() {
        let src = "fn double(x)\n    return x * 2\ny double(4)\n";
        let program = parse(src);
        analyze(&program);
        let mut ex = Executor::new();
        ex.execute(&program).expect("runtime failed");
    }

    #[test]
    fn executes_recursive_factorial() {
        let src = "fn fact(n)\n    if n <= 1\n        return 1\n    return n * fact(n - 1)\nresult fact(6)\n";
        let program = parse(src);
        analyze(&program);

        let mut ex = Executor::new();
        ex.push_scope();
        ex.exec_block_no_scope(&program.statements)
            .expect("runtime failed");

        assert_eq!(ex.get_var("result"), Some(Value::Int(720)));
    }

    #[test]
    fn executes_recursive_fibonacci() {
        let src = "fn fib(n)\n    if n <= 1\n        return n\n    return fib(n - 1) + fib(n - 2)\nvalue fib(8)\n";
        let program = parse(src);
        analyze(&program);

        let mut ex = Executor::new();
        ex.push_scope();
        ex.exec_block_no_scope(&program.statements)
            .expect("runtime failed");

        assert_eq!(ex.get_var("value"), Some(Value::Int(21)));
    }

    #[test]
    fn supports_deep_recursive_calls() {
        let src =
            "fn deep(n)\n    if n <= 0\n        return 0\n    return deep(n - 1)\nout deep(100)\n";
        let program = parse(src);
        analyze(&program);

        let mut ex = Executor::new();
        ex.push_scope();
        ex.exec_block_no_scope(&program.statements)
            .expect("runtime failed");

        assert_eq!(ex.get_var("out"), Some(Value::Int(0)));
    }

    #[test]
    fn allows_call_inside_expression_and_return_in_if() {
        let src = "fn id(x)\n    if x == 0\n        return 0\n    return x\nz id(3) + id(4)\n";
        let program = parse(src);
        analyze(&program);

        let mut ex = Executor::new();
        ex.push_scope();
        ex.exec_block_no_scope(&program.statements)
            .expect("runtime failed");

        assert_eq!(ex.get_var("z"), Some(Value::Int(7)));
    }

    #[test]
    fn train_uses_autopilot_defaults_when_properties_omitted() {
        let src = "model m\n    layers 2 4 1\ndataset d\ntrain m on d\n";
        let program = parse(src);
        analyze(&program);

        let mut ex = Executor::new();
        ex.execute(&program).expect("runtime failed");

        let model = ex.runtime.models.get("m").expect("model exists");
        assert_eq!(model.trained_epochs, 10);
    }

    #[test]
    fn train_prefers_explicit_overrides_over_autopilot() {
        let src = "model m\n    layers 2 3 4 5 6 7 8\ndataset d\n    batch 64\ntrain m on d\n    epochs 2\n    optimizer sgd\n    lr 0.1\n    batch 16\n    device cpu\n    precision fp32\n";
        let program = parse(src);
        analyze(&program);

        let mut ex = Executor::new();
        ex.execute(&program).expect("runtime failed");

        let model = ex.runtime.models.get("m").expect("model exists");
        assert_eq!(model.trained_epochs, 2);
    }

    // ── Regression tests for bugs found in the production code audit ──────────

    #[test]
    fn model_layer_of_zero_is_rejected() {
        // BUG FIX: model.layers values must be > 0.
        // Before the fix, a layer size of 0 was silently accepted and propagated
        // as a degenerate model to the autopilot and executor, causing silent
        // incorrect behaviour downstream.
        let src = "model m\n    layers 128 0 10\n";
        let program = parse(src);
        let mut ex = Executor::new();
        let err = ex
            .execute(&program)
            .expect_err("zero layer must be rejected");
        assert!(
            err.message.contains("must be > 0"),
            "expected rejection of zero layer, got: {}",
            err.message
        );
    }

    #[test]
    fn model_layer_of_negative_is_rejected() {
        let src = "model m\n    layers 128 -4 10\n";
        let program = parse(src);
        let mut ex = Executor::new();
        let err = ex
            .execute(&program)
            .expect_err("negative layer must be rejected");
        assert!(
            err.message.contains("must be > 0"),
            "expected rejection of negative layer, got: {}",
            err.message
        );
    }

    #[test]
    fn call_stack_overflow_is_caught_at_depth_128() {
        // BUG FIX: before the fix, infinite self-recursion would cause an
        // OS-level stack overflow. Now the runtime catches it at depth 128.
        // We spawn a thread with a larger stack (8 MB) to give the test safe
        // headroom while still validating the guard.
        let src = "fn inf(n)\n    return inf(n + 1)\nresult inf(0)\n".to_string();
        let handle = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let program = parse(&src);
                let mut ex = Executor::new();
                ex.execute(&program).expect_err("must hit call-depth limit")
            })
            .expect("thread spawn failed");
        let err = handle.join().expect("thread panicked");
        assert!(
            err.message.contains("Call stack overflow"),
            "expected call stack error, got: {}",
            err.message
        );
    }

    #[test]
    fn execution_continues_correctly_after_error_inside_function() {
        // BUG FIX: before the fix, a runtime error inside a function body left
        // `in_function_depth` incremented. This caused subsequent `return`
        // statements outside any function to be accepted silently.
        //
        // Strategy: register boom() and ok_fn() at top level, call boom() via a
        // VarDecl (the only call-statement form the parser supports), confirm it
        // errors, then call ok_fn() and verify it returns 42.
        let src = concat!(
            "fn boom()\n",
            "    _x 1 / 0\n", // division by zero inside boom
            "fn ok_fn()\n",
            "    return 42\n",
        );
        let program_setup = parse(src);
        let mut ex = Executor::new();
        ex.push_scope();
        ex.exec_block_no_scope(&program_setup.statements)
            .expect("function registration must not fail");

        // Call boom() — it errors inside; depth must be restored afterwards.
        // Wrap in VarDecl because the language has no bare call statement.
        let boom_prog = parse("_e boom()\n");
        let _ = ex.exec_block_no_scope(&boom_prog.statements); // ignore error

        // ok_fn() must still work correctly after the error.
        let ok_prog = parse("result ok_fn()\n");
        ex.exec_block_no_scope(&ok_prog.statements)
            .expect("ok_fn must succeed after boom() error recovery");

        assert_eq!(ex.get_var("result"), Some(Value::Int(42)));
    }
}
