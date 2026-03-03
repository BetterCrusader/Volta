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
use crate::ir::tensor::Tensor;
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
    /// Optional random seed for deterministic initialization and shuffling.
    pub seed: Option<i64>,
    /// Trained weights, populated after a successful `train` call.
    pub weights: Option<std::collections::HashMap<String, crate::ir::tensor::Tensor>>,
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
    /// The source CSV file path for this dataset.
    pub source: Option<String>,
    /// The fraction of data to reserve for validation (e.g. 0.2).
    pub val_split: Option<f64>,
    /// 0-based column index of the class label (enables one-hot auto-encoding).
    pub label_col: Option<usize>,
    /// Number of output classes; required when `label_col` is set.
    pub num_classes: Option<usize>,
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
            Stmt::Infer {
                model,
                input_csv,
                out_csv,
                span,
            } => {
                self.exec_infer_stmt(model, input_csv, out_csv, *span)?;
            }
            Stmt::Save { model, path, span } => {
                let model_state = self.runtime.models.get(model).ok_or_else(|| {
                    Self::error_with_hint(
                        format!("Undefined model: '{model}'"),
                        *span,
                        "Define the model before saving it in this run.".to_string(),
                    )
                })?;
                let weights = model_state.weights.clone().ok_or_else(|| {
                    Self::error(
                        format!("Model '{model}' has no trained weights to save. Train it first."),
                        *span,
                    )
                })?;

                // Versioned binary format: magic + version + model name + weights list
                let mut buf: Vec<u8> = Vec::new();
                buf.extend_from_slice(b"VOLT"); // magic
                buf.extend_from_slice(&1u32.to_le_bytes()); // format version
                let name_bytes = model.as_bytes();
                buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(name_bytes);

                let entries: Vec<(&String, &Vec<f32>, &Vec<usize>)> = weights
                    .iter()
                    .map(|(k, t)| (k, &t.data, &t.shape))
                    .collect();
                let n = entries.len() as u32;
                buf.extend_from_slice(&n.to_le_bytes());
                for (param_name, data, shape) in &entries {
                    let kb = param_name.as_bytes();
                    buf.extend_from_slice(&(kb.len() as u32).to_le_bytes());
                    buf.extend_from_slice(kb);
                    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
                    for &d in shape.iter() {
                        buf.extend_from_slice(&(d as u32).to_le_bytes());
                    }
                    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
                    for &f in data.iter() {
                        buf.extend_from_slice(&f.to_le_bytes());
                    }
                }

                std::fs::write(path, &buf).map_err(|e| {
                    Self::error(format!("Failed to save model to '{path}': {e}"), *span)
                })?;
                println!(
                    "Saved model '{model}' → '{path}' ({} parameters)",
                    entries.len()
                );
            }
            Stmt::Load { model, path, span } => {
                if !self.runtime.models.contains_key(model) {
                    return Err(Self::error_with_hint(
                        format!("Undefined model: '{model}'"),
                        *span,
                        "Declare the model name first, then load weights into it.".to_string(),
                    ));
                }
                let buf = std::fs::read(path).map_err(|e| {
                    Self::error(format!("Failed to read checkpoint '{path}': {e}"), *span)
                })?;
                if buf.len() < 8 || &buf[0..4] != b"VOLT" {
                    return Err(Self::error(
                        format!("File '{path}' is not a valid Volta checkpoint (bad magic)."),
                        *span,
                    ));
                }
                let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
                if version != 1 {
                    return Err(Self::error(
                        format!("Checkpoint version {version} is not supported (expected 1)."),
                        *span,
                    ));
                }
                let mut pos = 8usize;
                let read_u32 = |p: &mut usize, b: &[u8]| -> u32 {
                    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
                    *p += 4;
                    v
                };
                let read_str = |p: &mut usize, b: &[u8]| -> String {
                    let len = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap()) as usize;
                    *p += 4;
                    let s = String::from_utf8_lossy(&b[*p..*p + len]).to_string();
                    *p += len;
                    s
                };
                let _saved_model_name = read_str(&mut pos, &buf);
                let n_params = read_u32(&mut pos, &buf) as usize;
                let mut loaded: std::collections::HashMap<String, crate::ir::tensor::Tensor> =
                    std::collections::HashMap::with_capacity(n_params);
                for _ in 0..n_params {
                    let param_name = read_str(&mut pos, &buf);
                    let ndim = read_u32(&mut pos, &buf) as usize;
                    let mut shape = Vec::with_capacity(ndim);
                    for _ in 0..ndim {
                        shape.push(read_u32(&mut pos, &buf) as usize);
                    }
                    let nvals = read_u32(&mut pos, &buf) as usize;
                    let mut data = Vec::with_capacity(nvals);
                    for _ in 0..nvals {
                        data.push(f32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()));
                        pos += 4;
                    }
                    let tensor = crate::ir::tensor::Tensor::new(shape, data).map_err(|e| {
                        Self::error(format!("Bad tensor in checkpoint: {}", e.message), *span)
                    })?;
                    loaded.insert(param_name, tensor);
                }

                let ms = self.runtime.models.get_mut(model).unwrap();
                ms.weights = Some(loaded.clone());
                println!(
                    "Loaded model '{model}' from '{path}' ({} parameters)",
                    loaded.len()
                );
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

    fn exec_infer_stmt(
        &mut self,
        model_name: &str,
        input_csv: &str,
        out_csv: &str,
        span: Span,
    ) -> Result<(), RuntimeError> {
        let model_state = self.runtime.models.get(model_name).ok_or_else(|| {
            Self::error_with_hint(
                format!("Undefined model: '{model_name}'"),
                span,
                format!("Declare model '{model_name}' before passing it to infer."),
            )
        })?;

        if model_state.weights.is_none() {
            return Err(Self::error(
                format!("Model '{model_name}' has no weights. Train or load it before inference."),
                span,
            ));
        }

        let in_size = *model_state.layers.first().unwrap() as usize;

        // 1. Load CSV data (inputs only, no targets, batch_size=32 as default inference chunk)
        let batch_size = 32;
        let iter_batches = self.load_csv_inputs_only(input_csv, batch_size, in_size, span)?;

        // 2. Build inference graph
        let mut lower_ctx = crate::ir::lowering::LoweringContext::new();
        let input_node = lower_ctx
            .push_op(crate::ir::Op::Input("input".to_string()))
            .unwrap();

        // Lower the forward pass (without loss, but with activation)
        let _logits_node = lower_ctx.lower_model_to_graph(
            &model_state.layers,
            &model_state.activation,
            input_node,
            true,
        );

        let mut graph = lower_ctx.into_graph();

        // We bind dynamic shape for inference to handle the last partial batch
        graph.bind_input_shape("input", vec![0, in_size]);

        for w in model_state.layers.windows(2) {
            let in_f = w[0] as usize;
            let out_f = w[1] as usize;
            graph.bind_parameter_shape(&format!("weight_{in_f}_{out_f}"), vec![in_f, out_f]);
            graph.bind_parameter_shape(&format!("bias_{out_f}"), vec![1, out_f]);
        }

        let execution_plan = crate::ir::execution_plan::build_execution_plan(
            &graph,
            &std::collections::HashSet::new(),
        )
        .map_err(|e| Self::error(format!("Execution plan error: {}", e.message), span))?;

        let loaded_weights = model_state.weights.clone().unwrap();

        let mut all_predictions: Vec<Vec<f32>> = Vec::new();

        println!("Running inference on {} batches...", iter_batches.len());
        let backend = crate::ir::backend::CpuBackend;

        for batch_tensor in iter_batches {
            let actual_batch_size = batch_tensor.shape[0];

            let mut inputs = std::collections::HashMap::new();
            inputs.insert("input".to_string(), batch_tensor.clone());

            let mut context = crate::ir::interpreter::ExecutionContext {
                inputs: std::collections::HashMap::new(),
                parameters: std::collections::HashMap::new(),
            };

            // Populate parameters
            for (k, tensor) in &loaded_weights {
                context.parameters.insert(
                    k.clone(),
                    crate::ir::RuntimeValue::Tensor(std::sync::Arc::new(tensor.clone())),
                );
            }

            // Populate inputs
            for (k, tensor) in inputs {
                context.inputs.insert(
                    k,
                    crate::ir::RuntimeValue::Tensor(std::sync::Arc::new(tensor)),
                );
            }

            // Rebind the actual shape for this specific execution
            let dynamic_plan = execution_plan.clone();
            // dynamic_plan does not have a graph, the graph is what maps strings to values.
            // We'll rely on interpreter ignoring dynamic shape changes mid-flight and trusting the buffer size.

            // Execute terminal backend node
            let result_tensors = crate::ir::execute_terminal_with_backend(
                &graph,
                &dynamic_plan,
                &dynamic_plan.schedule.ordered_nodes,
                &backend,
                &context,
            )
            .map_err(|e| Self::error(format!("Inference execution failed: {}", e.message), span))?;

            let Some(out_runtime) = result_tensors else {
                return Err(Self::error(
                    "Inference did not return a tensor".to_string(),
                    span,
                ));
            };

            let crate::ir::RuntimeValue::Tensor(out_tensor) = out_runtime else {
                return Err(Self::error(
                    "Inference did not return a tensor".to_string(),
                    span,
                ));
            };

            for i in 0..actual_batch_size {
                let start = i * out_tensor.shape[1];
                let end = start + out_tensor.shape[1];
                let row = out_tensor.data[start..end].to_vec();
                all_predictions.push(row);
            }
        }

        // 4. Save to CSV
        let mut wtr = csv::Writer::from_path(out_csv).map_err(|e| {
            Self::error(format!("Failed to open output CSV '{out_csv}': {e}"), span)
        })?;

        // Write header (pred_0, pred_1, ...)
        if !all_predictions.is_empty() {
            let num_classes = all_predictions[0].len();
            let headers: Vec<String> = (0..num_classes).map(|i| format!("pred_{i}")).collect();
            wtr.write_record(&headers)
                .map_err(|e| Self::error(format!("Failed to write CSV header: {e}"), span))?;
        }

        for row in all_predictions {
            let string_row: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            wtr.write_record(&string_row)
                .map_err(|e| Self::error(format!("Failed to write CSV row: {e}"), span))?;
        }

        wtr.flush()
            .map_err(|e| Self::error(format!("Failed to flush CSV: {e}"), span))?;

        println!("Inference complete. Results saved to '{}'", out_csv);
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

        // --- AST to IR Lowering & Training ---

        use crate::ir::lowering::LoweringContext;
        use crate::ir::optimizer::*;
        use crate::ir::tensor::Tensor;
        use crate::ir::train::{TrainConfig, TrainSample, train_graph_with_backend};
        use crate::ir::{Backend, CpuBackend, CudaBackend, Op};

        let mut lower_ctx = LoweringContext::new();

        // 1. Inputs and Labels for Training Graph
        let input_node = lower_ctx.push_op(Op::Input("input".to_string())).unwrap();
        let target_node = lower_ctx.push_op(Op::Input("target".to_string())).unwrap();

        let logits = lower_ctx.lower_model_to_graph(
            &model_state.layers,
            &model_state.activation,
            input_node,
            false,
        );

        // 3. Loss node
        let is_classification = model_state.activation == "softmax";
        let loss_value = if is_classification {
            lower_ctx.build_softmax_cross_entropy_loss(logits, target_node)
        } else {
            lower_ctx.build_mse_loss(logits, target_node)
        };

        let mut graph = lower_ctx.into_graph();

        let in_size = *model_state.layers.first().unwrap() as usize;
        let out_size = *model_state.layers.last().unwrap() as usize;
        let batch_size = resolved.batch as usize;

        // 3.5 Bind shapes for Autograd & Shape Inference
        graph.bind_input_shape("input", vec![batch_size, in_size]);
        graph.bind_input_shape("target", vec![batch_size, out_size]);

        for w in model_state.layers.windows(2) {
            let in_features = w[0] as usize;
            let out_features = w[1] as usize;
            graph.bind_parameter_shape(
                &format!("weight_{in_features}_{out_features}"),
                vec![in_features, out_features],
            );
            graph.bind_parameter_shape(&format!("bias_{out_features}"), vec![1, out_features]);
        }

        // 4. Initialize random weights using Xavier/Glorot uniform
        let mut initial_parameters = std::collections::HashMap::new();

        use rand::SeedableRng;
        use rand::distr::{Distribution, Uniform};
        let global_seed = model_state.seed.unwrap_or(42) as u64;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(global_seed);
        let unit_uniform = Uniform::new(-1.0f32, 1.0f32).unwrap();

        for w in model_state.layers.windows(2) {
            let in_features = w[0] as usize;
            let out_features = w[1] as usize;

            // Weights – Xavier/Glorot uniform
            let weight_name = format!("weight_{in_features}_{out_features}");
            let weight_size = in_features * out_features;
            let weight_limit = (6.0_f32 / (in_features as f32 + out_features as f32)).sqrt();
            let w_dist = Uniform::new(-weight_limit, weight_limit).unwrap();
            let weight_data: Vec<f32> = (0..weight_size).map(|_| w_dist.sample(&mut rng)).collect();
            initial_parameters.insert(
                weight_name,
                Tensor::new(vec![in_features, out_features], weight_data).unwrap(),
            );

            // Bias (zeros)
            let bias_name = format!("bias_{out_features}");
            let bias_data = vec![0.0_f32; out_features];
            initial_parameters.insert(
                bias_name,
                Tensor::new(vec![1, out_features], bias_data).unwrap(),
            );
        }

        let (dataset, val_dataset) = if let Some(source_path) = &dataset_state.source {
            // Resolve effective output size: num_classes overrides out_size in label-col mode.
            let effective_out = if let Some(nc) = dataset_state.num_classes {
                if nc != out_size {
                    return Err(Self::error(
                        format!(
                            "dataset.num_classes ({nc}) must match model output layer ({out_size}). \
                             Fix 'layers' last value or 'num_classes'."
                        ),
                        span,
                    ));
                }
                nc
            } else {
                out_size
            };
            self.load_csv_dataset(
                source_path,
                batch_size,
                in_size,
                effective_out,
                dataset_state.val_split,
                dataset_state.shuffle.unwrap_or(false),
                global_seed,
                dataset_state.label_col,
                span,
            )?
        } else {
            // 5. Generate a synthetic dataset block
            let input_data: Vec<f32> = (0..batch_size * in_size)
                .map(|_| unit_uniform.sample(&mut rng))
                .collect();
            let _ = input_data.len(); // suppress unused warning
            let target_data: Vec<f32> = (0..batch_size * out_size)
                .map(|_| unit_uniform.sample(&mut rng))
                .collect();
            let _ = target_data.len();

            let sample = TrainSample {
                inputs: [
                    (
                        "input".to_string(),
                        Tensor::new(vec![batch_size, in_size], input_data).unwrap(),
                    ),
                    (
                        "target".to_string(),
                        Tensor::new(vec![batch_size, out_size], target_data).unwrap(),
                    ),
                ]
                .into_iter()
                .collect(),
            };

            // For simulation, we'll train over exactly 1 sample over N epochs
            (vec![sample], vec![])
        };

        // 6. Optimizer configuration
        let mut opt_config = OptimizerConfig::Sgd {
            lr: resolved.lr as f32,
        };
        if resolved.optimizer == "adam" {
            opt_config = OptimizerConfig::Adam {
                lr: resolved.lr as f32,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            };
        }

        let train_config = TrainConfig {
            epochs: resolved.epochs as usize,
            optimizer: opt_config,
        };

        // 7. Select appropriate runtime execution backend
        let (cpu, cuda);
        let dyn_backend: &dyn Backend = if resolved.device == "cuda" {
            cuda = CudaBackend;
            &cuda
        } else {
            cpu = CpuBackend;
            &cpu
        };

        // 8. Execute Backend Training Loop!
        let logits_opt = if is_classification {
            Some(logits)
        } else {
            None
        };
        match train_graph_with_backend(
            &graph,
            loss_value,
            logits_opt,
            initial_parameters,
            &dataset,
            &val_dataset,
            &train_config,
            dyn_backend,
        ) {
            Ok(result) => {
                if let Some(val_loss) = result.final_val_loss {
                    println!(
                        "Training completed. Final loss: {:.4}, Val loss: {:.4}",
                        result.final_loss, val_loss
                    );
                } else {
                    println!("Training completed. Final loss: {:.4}", result.final_loss);
                }
                // Store the trained weights in the model state for later Save/Load
                if let Some(ms) = self.runtime.models.get_mut(model) {
                    ms.weights = Some(result.final_parameters);
                    ms.trained_epochs += resolved.epochs;
                }
            }
            Err(e) => {
                return Err(Self::error(
                    format!("Training failed in graph execution: {}", e.message),
                    span,
                ));
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn load_csv_dataset(
        &self,
        source_path: &str,
        batch_size: usize,
        in_size: usize,
        out_size: usize,
        val_split: Option<f64>,
        shuffle: bool,
        seed: u64,
        label_col: Option<usize>,
        span: Span,
    ) -> Result<
        (
            Vec<crate::ir::train::TrainSample>,
            Vec<crate::ir::train::TrainSample>,
        ),
        RuntimeError,
    > {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(source_path)
            .map_err(|e| Self::error(format!("Failed to open CSV dataset: {e}"), span))?;

        let mut samples = Vec::new();
        for (i, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| Self::error(format!("CSV parsing error: {e}"), span))?;

            let mut inputs = Vec::with_capacity(in_size);
            let mut targets = Vec::with_capacity(out_size);

            if let Some(lc) = label_col {
                // ── label_col mode: one integer class index → one-hot ──────────────
                if record.len() <= lc {
                    return Err(Self::error(
                        format!(
                            "CSV row {} has {} columns but label_col={lc} is out of bounds.",
                            i + 2,
                            record.len()
                        ),
                        span,
                    ));
                }
                // Features = all columns except label_col, left-to-right
                let feature_cols = (0..record.len()).filter(|&c| c != lc);
                let mut taken = 0usize;
                for col in feature_cols {
                    if taken == in_size {
                        break;
                    }
                    let cell = record[col].trim();
                    let val: f32 = cell.parse().map_err(|_| {
                        Self::error(format!("Не вдалося перетворити значення '{cell}' на число у рядку {} (колонка {}).", i + 2, col + 1), span)
                    })?;
                    inputs.push(val);
                    taken += 1;
                }
                if taken < in_size {
                    return Err(Self::error(
                        format!(
                            "CSV row {} has only {taken} feature columns (excluding label), but model expects {in_size}.",
                            i + 2
                        ),
                        span,
                    ));
                }
                // Parse label
                let cell = record[lc].trim();
                if cell.is_empty() {
                    return Err(Self::error(
                        format!("Empty label at row {} (column {}).", i + 2, lc + 1),
                        span,
                    ));
                }
                let label_f: f32 = cell.parse().map_err(|_| {
                    Self::error(format!("Invalid label '{cell}' at row {} (column {}). Expected an integer class index.", i + 2, lc + 1), span)
                })?;
                let label_i = label_f.round();
                if label_i < 0.0 || label_i >= out_size as f32 {
                    return Err(Self::error(
                        format!(
                            "Label {label_i} at row {} is out of range [0, {out_size}). Set 'num_classes' correctly.",
                            i + 2
                        ),
                        span,
                    ));
                }
                let mut one_hot = vec![0.0f32; out_size];
                one_hot[label_i as usize] = 1.0;
                targets.extend_from_slice(&one_hot);
            } else {
                // ── legacy mode: last out_size columns are float targets ────────────
                if record.len() != in_size + out_size {
                    return Err(Self::error(
                        format!(
                            "CSV data shape mismatch at row {}. Model requires {} columns ({} inputs + {} targets), but row has {}.",
                            i + 2,
                            in_size + out_size,
                            in_size,
                            out_size,
                            record.len()
                        ),
                        span,
                    ));
                }
                for j in 0..in_size {
                    let val: f32 = record[j].trim().parse().map_err(|_| {
                        Self::error(format!("Не вдалося перетворити значення '{}' на число у рядку {} (колонка {}).", &record[j], i + 2, j + 1), span)
                    })?;
                    inputs.push(val);
                }
                for j in 0..out_size {
                    let val: f32 = record[in_size + j].trim().parse().map_err(|_| {
                        Self::error(format!("Не вдалося перетворити значення '{}' на число у рядку {} (колонка {}).", &record[in_size + j], i + 2, in_size + j + 1), span)
                    })?;
                    targets.push(val);
                }
            }
            samples.push((inputs, targets));
        }

        let total_samples = samples.len();
        if total_samples == 0 {
            return Err(Self::error("CSV dataset is empty".to_string(), span));
        }

        if shuffle {
            use rand::SeedableRng;
            use rand::seq::SliceRandom;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            samples.shuffle(&mut rng);
        }

        let train_samples_count = if let Some(ratio) = val_split {
            let val_count = (total_samples as f64 * ratio).floor() as usize;
            total_samples.saturating_sub(val_count)
        } else {
            total_samples
        };

        let train_samples = &samples[0..train_samples_count];
        let val_samples = &samples[train_samples_count..];

        // Train: DropLast — only full batches for stable gradients
        #[allow(clippy::needless_range_loop)]
        let make_train_batches =
            |data: &[(Vec<f32>, Vec<f32>)]| -> Vec<crate::ir::train::TrainSample> {
                let num_batches = data.len() / batch_size;
                let mut batches = Vec::with_capacity(num_batches);
                for b in 0..num_batches {
                    let start = b * batch_size;
                    let end = start + batch_size;
                    let mut batch_inputs = Vec::with_capacity(batch_size * in_size);
                    let mut batch_targets = Vec::with_capacity(batch_size * out_size);
                    for k in start..end {
                        batch_inputs.extend_from_slice(&data[k].0);
                        batch_targets.extend_from_slice(&data[k].1);
                    }
                    let in_tensor =
                        crate::ir::tensor::Tensor::new(vec![batch_size, in_size], batch_inputs)
                            .unwrap();
                    let out_tensor =
                        crate::ir::tensor::Tensor::new(vec![batch_size, out_size], batch_targets)
                            .unwrap();
                    batches.push(crate::ir::train::TrainSample {
                        inputs: [
                            ("input".to_string(), in_tensor),
                            ("target".to_string(), out_tensor),
                        ]
                        .into_iter()
                        .collect(),
                    });
                }
                batches
            };

        // Val: keep remainder — every sample counts, even partial last batch
        #[allow(clippy::needless_range_loop)]
        let make_val_batches =
            |data: &[(Vec<f32>, Vec<f32>)]| -> Vec<crate::ir::train::TrainSample> {
                if data.is_empty() {
                    return Vec::new();
                }
                let mut batches = Vec::new();
                let mut offset = 0;
                while offset < data.len() {
                    let actual = (data.len() - offset).min(batch_size);
                    let end = offset + actual;
                    let mut batch_inputs = Vec::with_capacity(actual * in_size);
                    let mut batch_targets = Vec::with_capacity(actual * out_size);
                    for k in offset..end {
                        batch_inputs.extend_from_slice(&data[k].0);
                        batch_targets.extend_from_slice(&data[k].1);
                    }
                    let in_tensor =
                        crate::ir::tensor::Tensor::new(vec![actual, in_size], batch_inputs)
                            .unwrap();
                    let out_tensor =
                        crate::ir::tensor::Tensor::new(vec![actual, out_size], batch_targets)
                            .unwrap();
                    batches.push(crate::ir::train::TrainSample {
                        inputs: [
                            ("input".to_string(), in_tensor),
                            ("target".to_string(), out_tensor),
                        ]
                        .into_iter()
                        .collect(),
                    });
                    offset = end;
                }
                batches
            };

        let train_batches = make_train_batches(train_samples);
        let val_batches = make_val_batches(val_samples);

        if train_batches.is_empty() {
            return Err(Self::error(
                format!(
                    "Train set has {} samples, which is fewer than batch size {}",
                    train_samples.len(),
                    batch_size
                ),
                span,
            ));
        }

        if val_samples.is_empty() {
            println!(
                "Loaded {:<4} total samples from '{}'. Train: {} batches. Validation skipped: no samples after split.",
                total_samples,
                source_path,
                train_batches.len()
            );
        } else {
            let partial = if val_samples.len() % batch_size != 0 {
                " (incl. partial)"
            } else {
                ""
            };
            println!(
                "Loaded {:<4} total samples from '{}'. Train: {} batches, Val: {} batches{}.",
                total_samples,
                source_path,
                train_batches.len(),
                val_batches.len(),
                partial
            );
        }

        Ok((train_batches, val_batches))
    }

    fn load_csv_inputs_only(
        &self,
        path: &str,
        batch_size: usize,
        in_features: usize,
        span: Span,
    ) -> Result<Vec<Tensor>, RuntimeError> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .map_err(|e| {
                Self::error(
                    format!("Failed to open inference CSV dataset '{path}': {e}"),
                    span,
                )
            })?;

        let mut all_inputs = Vec::new();

        for (row_idx, result) in rdr.records().enumerate() {
            let record = match result {
                Ok(r) => r,
                Err(e) => {
                    return Err(Self::error(
                        format!("Failed to read CSV row {row_idx}: {e}"),
                        span,
                    ));
                }
            };

            // Skip empty rows
            if record.is_empty() {
                continue;
            }

            if record.len() < in_features {
                return Err(Self::error(
                    format!(
                        "Inference CSV row {} has {} columns, expected at least {} input features",
                        row_idx + 1,
                        record.len(),
                        in_features
                    ),
                    span,
                ));
            }

            for col_idx in 0..in_features {
                let cell = &record[col_idx];
                let val: f32 = cell.trim().parse().map_err(|_| {
                    Self::error(
                        format!(
                            "Invalid float in inference CSV at row {}, column {}: '{}'",
                            row_idx + 1,
                            col_idx + 1,
                            cell
                        ),
                        span,
                    )
                })?;
                all_inputs.push(val);
            }
        }

        let total_samples = all_inputs.len() / in_features;
        if total_samples == 0 {
            return Err(Self::error(
                "Inference dataset is empty or invalid".to_string(),
                span,
            ));
        }

        let mut batches = Vec::new();
        let mut offset = 0;

        while offset < all_inputs.len() {
            let remaining_samples = (all_inputs.len() - offset) / in_features;
            let current_batch_size = remaining_samples.min(batch_size);

            let chunk_size = current_batch_size * in_features;
            let chunk_data = all_inputs[offset..offset + chunk_size].to_vec();

            let batch_tensor = Tensor::new(vec![current_batch_size, in_features], chunk_data)
                .map_err(|e| {
                    Self::error(
                        format!("Failed to create inference batch tensor: {:?}", e),
                        span,
                    )
                })?;

            batches.push(batch_tensor);
            offset += chunk_size;
        }

        Ok(batches)
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

        let seed = match Self::find_prop(props, "seed") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "model.seed expects exactly one value".to_string(),
                        span,
                    ));
                }
                Some(self.expect_int_value(&values[0], "model.seed")?)
            }
            None => None,
        };

        let _ = model_name;

        Ok(ModelState {
            layers,
            activation,
            optimizer,
            optimizer_lr,
            precision,
            memory,
            seed,
            weights: None,
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

        let source = match Self::find_prop(props, "source") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "dataset.source expects exactly one string value".to_string(),
                        span,
                    ));
                }
                Some(self.expect_symbol_value(&values[0], "dataset.source")?)
            }
            None => None,
        };

        let val_split = match Self::find_prop(props, "val_split") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "dataset.val_split expects exactly one number value".to_string(),
                        span,
                    ));
                }
                let ratio = self.expect_float_value(&values[0], "dataset.val_split")?;
                if !(0.0..=1.0).contains(&ratio) {
                    return Err(Self::error(
                        "dataset.val_split must be between 0.0 and 1.0".to_string(),
                        span,
                    ));
                }
                Some(ratio)
            }
            None => None,
        };

        let label_col = match Self::find_prop(props, "label_col") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "dataset.label_col expects one integer.".into(),
                        span,
                    ));
                }
                let v = self.expect_int_value(&values[0], "dataset.label_col")?;
                if v < 0 {
                    return Err(Self::error("dataset.label_col must be >= 0.".into(), span));
                }
                Some(v as usize)
            }
            None => None,
        };

        let num_classes = match Self::find_prop(props, "num_classes") {
            Some(values) => {
                if values.len() != 1 {
                    return Err(Self::error(
                        "dataset.num_classes expects one integer.".into(),
                        span,
                    ));
                }
                let v = self.expect_int_value(&values[0], "dataset.num_classes")?;
                if v < 2 {
                    return Err(Self::error(
                        "dataset.num_classes must be >= 2.".into(),
                        span,
                    ));
                }
                Some(v as usize)
            }
            None => None,
        };

        if label_col.is_some() && num_classes.is_none() {
            return Err(Self::error(
                "'num_classes' is required when 'label_col' is set.".into(),
                span,
            ));
        }

        Ok(DatasetState {
            batch,
            shuffle,
            source,
            val_split,
            label_col,
            num_classes,
        })
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

    fn expect_float_value(&mut self, expr: &Expr, label: &str) -> Result<f64, RuntimeError> {
        match self.eval_expr(expr)? {
            Value::Float(v) => Ok(v),
            Value::Int(v) => Ok(v as f64),
            other => Err(Self::error(
                format!(
                    "Invalid property: {label} expects float, found {}",
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
