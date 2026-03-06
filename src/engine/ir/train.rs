use std::collections::HashMap;
use std::sync::Arc;

use crate::ir::autograd::build_reverse_graph;
use crate::ir::interpreter::{ExecutionContext, RuntimeValue};
use crate::ir::optimizer::{OptimizerConfig, OptimizerState};
use crate::ir::tensor::Tensor;
use crate::ir::{
    Backend, CompilerFlags, CpuBackend, Graph, Op, ValueId, build_execution_plan,
    execute_terminal_with_backend_buffered, execute_value_with_backend,
    execute_forward_and_save, execute_backward_with_saved_activations,
    verify_graph,
};

#[derive(Debug, Clone)]
pub struct TrainSample {
    pub inputs: HashMap<String, Tensor>,
}

#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Number of epochs to wait for improvement before stopping.
    pub patience: usize,
    /// Minimum change in monitored loss to qualify as improvement.
    pub min_delta: f32,
    /// Whether to restore best weights on early stop (true) or keep final (false).
    pub restore_best_weights: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self { patience: 10, min_delta: 1e-4, restore_best_weights: true }
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub optimizer: OptimizerConfig,
    pub lr_schedule: Option<crate::ir::optimizer::LrSchedule>,
    pub clip_grad: Option<f32>,
    /// Optional early stopping configuration.
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Number of micro-batches to accumulate gradients over before applying update.
    /// 1 = standard SGD (no accumulation). N > 1 = effective batch size N×batch.
    pub gradient_accumulation_steps: usize,
}

impl TrainConfig {
    pub fn new(epochs: usize, optimizer: OptimizerConfig) -> Self {
        Self { epochs, optimizer, lr_schedule: None, clip_grad: None, early_stopping: None, gradient_accumulation_steps: 1 }
    }
}

#[derive(Debug, Clone)]
pub struct TrainResult {
    pub final_parameters: HashMap<String, Tensor>,
    pub final_loss: f32,
    pub final_val_loss: Option<f32>,
    pub optimizer_state: OptimizerState,
}

#[derive(Debug, Clone)]
pub struct TrainError {
    pub message: String,
}

impl From<crate::ir::tensor::TensorError> for TrainError {
    fn from(err: crate::ir::tensor::TensorError) -> Self {
        TrainError {
            message: err.message,
        }
    }
}

pub fn train_graph(
    forward_graph: &Graph,
    loss_value: ValueId,
    initial_parameters: HashMap<String, Tensor>,
    dataset: &[TrainSample],
    val_dataset: &[TrainSample],
    config: &TrainConfig,
) -> Result<TrainResult, TrainError> {
    let backend = CpuBackend;
    train_graph_with_backend(
        forward_graph,
        loss_value,
        None,
        initial_parameters,
        dataset,
        val_dataset,
        config,
        &backend,
    )
}

/// Trains a graph using a caller-provided backend implementation.
///
/// The function executes forward and backward graphs, computes gradients for
/// parameter nodes, and applies optimizer updates for each sample/epoch pair.
///
/// # Errors
/// Returns [`TrainError`] when graph verification fails, backward graph
/// construction fails, execution fails on the selected backend, or optimizer
/// updates cannot be applied.
#[allow(clippy::too_many_arguments)]
pub fn train_graph_with_backend(
    forward_graph: &Graph,
    loss_value: ValueId,
    logits_value: Option<ValueId>,
    initial_parameters: HashMap<String, Tensor>,
    dataset: &[TrainSample],
    val_dataset: &[TrainSample],
    config: &TrainConfig,
    backend: &dyn Backend,
) -> Result<TrainResult, TrainError> {
    verify_graph(forward_graph).map_err(|err| TrainError {
        message: format!("Forward graph failed verification: {}", err.message),
    })?;
    let forward_plan = build_execution_plan(forward_graph, &std::collections::HashSet::new())
        .map_err(|err| TrainError {
            message: format!("Failed to build forward execution plan: {}", err.message),
        })?;
    let determinism = CompilerFlags::from_env().determinism;

    let parameter_values = collect_parameter_values(forward_graph);

    // raw_params: owned tensors for optimizer updates
    // arc_params: Arc-wrapped for zero-copy context building
    let mut raw_params: HashMap<ValueId, Tensor> = HashMap::new();
    let mut arc_params: HashMap<ValueId, Arc<Tensor>> = HashMap::new();
    let mut tracked_params = Vec::new();

    for (name, initial_tensor) in initial_parameters {
        if let Some(value_id) = parameter_values.get(&name).copied() {
            let arc = Arc::new(initial_tensor.clone());
            arc_params.insert(value_id, arc);
            raw_params.insert(value_id, initial_tensor);
            tracked_params.push((name, value_id));
        }
    }

    let mut optimizer_state = OptimizerState::default();
    let param_value_ids = tracked_params.iter().map(|(_, v)| *v).collect::<Vec<_>>();

    let gradient_graph =
        build_reverse_graph(forward_graph, loss_value, &param_value_ids).map_err(|e| {
            TrainError {
                message: format!("Failed to build reverse graph: {}", e.message),
            }
        })?;
    verify_graph(&gradient_graph.backward).map_err(|err| TrainError {
        message: format!("Backward graph failed verification: {}", err.message),
    })?;
    let gradient_values = gradient_graph
        .gradients
        .values()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let backward_plan =
        build_execution_plan(&gradient_graph.backward, &gradient_values).map_err(|err| {
            TrainError {
                message: format!("Failed to build backward execution plan: {}", err.message),
            }
        })?;

    // Pre-compute gradient targets once — stable across all epochs/batches.
    let grad_targets: Vec<(String, ValueId, ValueId)> = tracked_params
        .iter()
        .filter_map(|(name, value_id)| {
            gradient_graph.gradients.get(value_id).copied()
                .map(|grad_value_id| (name.clone(), *value_id, grad_value_id))
        })
        .collect();
    let grad_value_ids: Vec<ValueId> = grad_targets.iter().map(|(_, _, gv)| *gv).collect();

    let mut final_val_loss_out = None;
    // Pre-allocated execution buffers — reused across all epochs/samples.
    // fwd_buf: holds ALL forward activations so the backward pass can skip
    //          recomputing the forward sub-graph that was cloned into the
    //          backward graph (the biggest single source of overhead).
    // bwd_buf: backward-pass intermediate values buffer.
    let mut fwd_buf: Vec<Option<crate::ir::RuntimeValue>> = Vec::new();
    let mut bwd_buf: Vec<Option<crate::ir::RuntimeValue>> = Vec::new();

    // Early stopping state
    let mut es_best_loss = f32::INFINITY;
    let mut es_patience_counter = 0usize;
    let mut es_best_params: Option<HashMap<ValueId, Tensor>> = None;
    let mut es_stopped_epoch = None::<usize>;

    for epoch in 0..config.epochs {
        let mut epoch_loss_sum = 0.0;
        let mut train_batches = 0;

        // Apply LR schedule if configured
        let epoch_optimizer = if let Some(schedule) = &config.lr_schedule {
            let base_lr = get_optimizer_lr(&config.optimizer);
            let scaled_lr = schedule.compute_lr(base_lr, epoch);
            set_optimizer_lr(config.optimizer.clone(), scaled_lr)
        } else {
            config.optimizer.clone()
        };

        let accum_steps = config.gradient_accumulation_steps.max(1);
        let mut accum_grads: HashMap<ValueId, Tensor> = HashMap::new();
        let mut accum_count = 0usize;

        for sample in dataset {
            let mut context = build_context_with_ids(sample, &arc_params, &tracked_params);
            // Forward pass: execute AND save all intermediate activations.
            // fwd_buf will contain every computed value — the backward pass
            // pre-fills its buffer with these to skip re-running forward ops.
            let loss_value_runtime = execute_forward_and_save(
                forward_graph,
                &forward_plan,
                &forward_plan.schedule.ordered_nodes,
                backend,
                &context,
                &mut fwd_buf,
            )
            .map_err(|e| TrainError {
                message: format!("Forward execute failed: {}", e.message),
            })?;
            let Some(loss_runtime) = loss_value_runtime else {
                return Err(TrainError {
                    message: "Forward graph produced no loss output".to_string(),
                });
            };

            epoch_loss_sum +=
                scalar_from_runtime(loss_runtime.clone()).map_err(|e| TrainError { message: e })?;
            train_batches += 1;

            let seed = ones_like(&loss_runtime).map_err(|e| TrainError { message: e })?;
            context.inputs.insert("__loss_grad".to_string(), seed);
            // Backward pass: pre-fill with saved forward activations so the
            // backward graph skips re-running the cloned forward sub-graph.
            let all_grads = execute_backward_with_saved_activations(
                &gradient_graph.backward,
                &backward_plan,
                &grad_value_ids,
                &backward_plan.schedule.ordered_nodes,
                backend,
                &context,
                &fwd_buf,
                &mut bwd_buf,
            )
            .map_err(|e| TrainError {
                message: format!("Backward execute failed: {}", e.message),
            })?;

            // Accumulate gradients
            for (name, value_id, grad_value_id) in &grad_targets {
                let grad_runtime = all_grads.get(grad_value_id).ok_or_else(|| TrainError {
                    message: format!("Missing gradient for '{name}'"),
                })?.clone();
                let grad_tensor = runtime_to_tensor(grad_runtime).map_err(|e| TrainError {
                    message: format!("Gradient for '{name}' is invalid: {e}"),
                })?;
                match accum_grads.get_mut(value_id) {
                    Some(existing) => {
                        let data = Arc::make_mut(&mut existing.data);
                        for (a, &b) in data.iter_mut().zip(grad_tensor.data.iter()) {
                            *a += b;
                        }
                    }
                    None => {
                        accum_grads.insert(*value_id, grad_tensor);
                    }
                }
            }
            accum_count += 1;

            // Only apply update after accumulating accum_steps gradients
            if accum_count < accum_steps {
                continue;
            }

            // Average accumulated gradients
            if accum_steps > 1 {
                let scale = 1.0 / accum_steps as f32;
                for grad in accum_grads.values_mut() {
                    let data = Arc::make_mut(&mut grad.data);
                    for v in data.iter_mut() { *v *= scale; }
                }
            }

            let gradients_by_id = std::mem::take(&mut accum_grads);
            accum_count = 0;

            // Gradient clipping by global norm
            let mut gradients_by_id = gradients_by_id;
            if let Some(max_norm) = config.clip_grad {
                if max_norm > 0.0 {
                    let global_sq_norm: f32 = gradients_by_id.values()
                        .flat_map(|t| t.data.iter())
                        .map(|v| v * v)
                        .sum();
                    let global_norm = global_sq_norm.sqrt();
                    if global_norm > max_norm {
                        let scale = max_norm / global_norm;
                        for grad in gradients_by_id.values_mut() {
                            let data = Arc::make_mut(&mut grad.data);
                            for v in data.iter_mut() { *v *= scale; }
                        }
                    }
                }
            }

            backend
                .apply_gradients(
                    &mut raw_params,
                    &gradients_by_id,
                    &epoch_optimizer,
                    &mut optimizer_state,
                    determinism,
                )
                .map_err(|e| TrainError {
                    message: format!("Optimizer step failed: {}", e.message),
                })?;

            // Sync arc_params with updated raw_params (one Arc::new per updated param)
            for (value_id, tensor) in &raw_params {
                if gradients_by_id.contains_key(value_id) {
                    arc_params.insert(*value_id, Arc::new(tensor.clone()));
                }
            }
        }

        // Flush remaining accumulated gradients (when dataset size < accum_steps,
        // or dataset size is not a multiple of accum_steps)
        if accum_count > 0 && !accum_grads.is_empty() {
            // Average by actual count (not accum_steps)
            if accum_count > 1 {
                let scale = 1.0 / accum_count as f32;
                for grad in accum_grads.values_mut() {
                    let data = Arc::make_mut(&mut grad.data);
                    for v in data.iter_mut() { *v *= scale; }
                }
            }

            let mut gradients_by_id = accum_grads;

            // Gradient clipping
            if let Some(max_norm) = config.clip_grad {
                if max_norm > 0.0 {
                    let global_sq_norm: f32 = gradients_by_id.values()
                        .flat_map(|t| t.data.iter())
                        .map(|v| v * v)
                        .sum();
                    let global_norm = global_sq_norm.sqrt();
                    if global_norm > max_norm {
                        let scale = max_norm / global_norm;
                        for grad in gradients_by_id.values_mut() {
                            let data = Arc::make_mut(&mut grad.data);
                            for v in data.iter_mut() { *v *= scale; }
                        }
                    }
                }
            }

            backend
                .apply_gradients(
                    &mut raw_params,
                    &gradients_by_id,
                    &epoch_optimizer,
                    &mut optimizer_state,
                    determinism,
                )
                .map_err(|e| TrainError {
                    message: format!("Optimizer step (flush) failed: {}", e.message),
                })?;

            for (value_id, tensor) in &raw_params {
                if gradients_by_id.contains_key(value_id) {
                    arc_params.insert(*value_id, Arc::new(tensor.clone()));
                }
            }
        }

        let epoch_train_loss = if train_batches > 0 {
            epoch_loss_sum / train_batches as f32
        } else {
            0.0
        };

        if !val_dataset.is_empty() {
            let mut val_loss_sum = 0.0;
            let mut val_correct = 0;
            let mut val_total = 0;

            for sample in val_dataset {
                let context = build_context_with_ids(sample, &arc_params, &tracked_params);
                let loss_value_runtime = execute_terminal_with_backend_buffered(
                    forward_graph,
                    &forward_plan,
                    &forward_plan.schedule.ordered_nodes,
                    backend,
                    &context,
                    &mut fwd_buf,
                )
                .map_err(|e| TrainError {
                    message: format!("Validation forward execute failed: {}", e.message),
                })?;
                let Some(loss_runtime) = loss_value_runtime else {
                    return Err(TrainError {
                        message: "Forward graph produced no loss output during validation"
                            .to_string(),
                    });
                };
                val_loss_sum +=
                    scalar_from_runtime(loss_runtime).map_err(|e| TrainError { message: e })?;

                // Compute accuracy if logits are available
                #[allow(clippy::collapsible_if)]
                if let Some(logits_id) = logits_value {
                    if let Ok(logits_rt) = execute_value_with_backend(
                        forward_graph,
                        &forward_plan,
                        logits_id,
                        &forward_plan.schedule.ordered_nodes,
                        backend,
                        &context,
                    ) {
                        if let Ok(logits_t) = runtime_to_tensor(logits_rt) {
                            if let Ok(predictions) = logits_t.argmax_axis_1() {
                                // Extract targets from context (assuming it's keyed as "target")
                                if let Some(target_rt) = context.inputs.get("target") {
                                    let target_val: RuntimeValue = target_rt.clone();
                                    if let Ok(target_t) = runtime_to_tensor(target_val) {
                                        if let Ok(targets_argmax) = target_t.argmax_axis_1() {
                                            for (p, t) in
                                                predictions.iter().zip(targets_argmax.iter())
                                            {
                                                if p == t {
                                                    val_correct += 1;
                                                }
                                                val_total += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let avg_val_loss = val_loss_sum / val_dataset.len() as f32;
            final_val_loss_out = Some(avg_val_loss);

            if (epoch + 1) % 10 == 0 || epoch == config.epochs - 1 {
                if val_total > 0 {
                    let acc = (val_correct as f32 / val_total as f32) * 100.0;
                    println!(
                        "Epoch {:>3}/{} - loss: {:.4} - val_loss: {:.4} - val_acc: {:.1}%",
                        epoch + 1,
                        config.epochs,
                        epoch_train_loss,
                        avg_val_loss,
                        acc
                    );
                } else {
                    println!(
                        "Epoch {:>3}/{} - loss: {:.4} - val_loss: {:.4}",
                        epoch + 1,
                        config.epochs,
                        epoch_train_loss,
                        avg_val_loss
                    );
                }
            }
        } else if (epoch + 1) % 10 == 0 || epoch == config.epochs - 1 {
            println!(
                "Epoch {:>3}/{} - loss: {:.4}",
                epoch + 1,
                config.epochs,
                epoch_train_loss
            );
        }

        // Early stopping check
        if let Some(es) = &config.early_stopping {
            let monitored = final_val_loss_out.unwrap_or(epoch_train_loss);
            if monitored < es_best_loss - es.min_delta {
                es_best_loss = monitored;
                es_patience_counter = 0;
                if es.restore_best_weights {
                    es_best_params = Some(raw_params.clone());
                }
            } else {
                es_patience_counter += 1;
                if es_patience_counter >= es.patience {
                    println!(
                        "Early stopping at epoch {} (best loss: {:.4})",
                        epoch + 1, es_best_loss
                    );
                    es_stopped_epoch = Some(epoch + 1);
                    break;
                }
            }
        }
    }

    // Restore best weights if early stopping was triggered and restore_best_weights=true
    if let (Some(es), Some(_)) = (&config.early_stopping, &es_stopped_epoch) {
        if es.restore_best_weights {
            if let Some(best) = es_best_params {
                raw_params = best;
                // Sync arc_params
                for (value_id, tensor) in &raw_params {
                    arc_params.insert(*value_id, Arc::new(tensor.clone()));
                }
            }
        }
    }

    let final_loss = if let Some(sample) = dataset.last() {
        let context = build_context_with_ids(sample, &arc_params, &tracked_params);
        let loss_runtime = execute_terminal_with_backend_buffered(
            forward_graph,
            &forward_plan,
            &forward_plan.schedule.ordered_nodes,
            backend,
            &context,
            &mut fwd_buf,
        )
        .map_err(|e| TrainError {
            message: format!("Final loss execute failed: {}", e.message),
        })?;
        let Some(loss_runtime) = loss_runtime else {
            return Err(TrainError {
                message: "Forward graph produced no final loss output".to_string(),
            });
        };
        scalar_from_runtime(loss_runtime).map_err(|e| TrainError { message: e })?
    } else {
        0.0
    };

    let mut final_parameters = HashMap::new();
    for (name, value_id) in tracked_params {
        if let Some(tensor) = raw_params.remove(&value_id) {
            final_parameters.insert(name, tensor);
        }
    }

    Ok(TrainResult {
        final_parameters,
        final_loss,
        final_val_loss: final_val_loss_out,
        optimizer_state,
    })
}

fn collect_parameter_values(graph: &Graph) -> HashMap<String, ValueId> {
    let mut values = HashMap::new();
    for node in &graph.nodes {
        if let Op::Parameter(name) = &node.op {
            values.insert(name.clone(), node.output);
        }
    }
    values
}

fn build_context_with_ids(
    sample: &TrainSample,
    parameter_tensors: &HashMap<ValueId, Arc<Tensor>>,
    tracked_params: &[(String, ValueId)],
) -> ExecutionContext {
    let mut context = ExecutionContext::default();
    for (name, tensor) in &sample.inputs {
        context.inputs.insert(
            name.clone(),
            RuntimeValue::Tensor(Arc::new(tensor.clone())),
        );
    }
    for (name, value_id) in tracked_params {
        if let Some(arc_tensor) = parameter_tensors.get(value_id) {
            context.parameters.insert(
                name.clone(),
                RuntimeValue::Tensor(Arc::clone(arc_tensor)),
            );
        }
    }
    context
}

fn runtime_to_tensor(value: RuntimeValue) -> Result<Tensor, String> {
    match value {
        RuntimeValue::Tensor(tensor) => Ok((*tensor).clone()),
        _ => Err("Expected tensor runtime value".to_string()),
    }
}

fn scalar_from_runtime(value: RuntimeValue) -> Result<f32, String> {
    match value {
        RuntimeValue::Tensor(tensor) => {
            if tensor.logical_len() != 1 {
                return Err(format!(
                    "Expected scalar tensor for loss, got shape {:?}",
                    tensor.shape
                ));
            }
            Ok(tensor.data[0])
        }
        RuntimeValue::Float(v) => Ok(v as f32),
        RuntimeValue::Int(v) => Ok(v as f32),
    }
}

fn ones_like(value: &RuntimeValue) -> Result<RuntimeValue, String> {
    match value {
        RuntimeValue::Tensor(tensor) => Ok(RuntimeValue::Tensor(std::sync::Arc::new(
            Tensor::new(tensor.shape.clone(), vec![1.0; tensor.logical_len()]).unwrap(),
        ))),
        RuntimeValue::Float(_) => Ok(RuntimeValue::Float(1.0)),
        RuntimeValue::Int(_) => Ok(RuntimeValue::Int(1)),
    }
}

fn get_optimizer_lr(config: &OptimizerConfig) -> f32 {
    match config {
        OptimizerConfig::Sgd { lr } => *lr,
        OptimizerConfig::Adam { lr, .. } => *lr,
        OptimizerConfig::AdamW { lr, .. } => *lr,
        OptimizerConfig::RmsProp { lr, .. } => *lr,
        OptimizerConfig::Adagrad { lr, .. } => *lr,
        OptimizerConfig::Lars { lr, .. } => *lr,
    }
}

fn set_optimizer_lr(config: OptimizerConfig, new_lr: f32) -> OptimizerConfig {
    match config {
        OptimizerConfig::Sgd { .. } => OptimizerConfig::Sgd { lr: new_lr },
        OptimizerConfig::Adam { beta1, beta2, epsilon, .. } =>
            OptimizerConfig::Adam { lr: new_lr, beta1, beta2, epsilon },
        OptimizerConfig::AdamW { beta1, beta2, epsilon, weight_decay, .. } =>
            OptimizerConfig::AdamW { lr: new_lr, beta1, beta2, epsilon, weight_decay },
        OptimizerConfig::RmsProp { alpha, epsilon, weight_decay, momentum, .. } =>
            OptimizerConfig::RmsProp { lr: new_lr, alpha, epsilon, weight_decay, momentum },
        OptimizerConfig::Adagrad { epsilon, weight_decay, .. } =>
            OptimizerConfig::Adagrad { lr: new_lr, epsilon, weight_decay },
        OptimizerConfig::Lars { momentum, weight_decay, trust_coeff, epsilon, .. } =>
            OptimizerConfig::Lars { lr: new_lr, momentum, weight_decay, trust_coeff, epsilon },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::{Graph, Op, OptimizerConfig, Tensor, TrainConfig, TrainSample, train_graph};

    #[test]
    fn end_to_end_train_with_sgd_reduces_loss() {
        let (graph, loss) = build_linear_mse_graph();
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![0.0]).expect("valid tensor"),
        );

        let dataset = vec![
            sample(1.0, 2.0),
            sample(2.0, 4.0),
            sample(3.0, 6.0),
            sample(4.0, 8.0),
        ];

        let result = train_graph(
            &graph,
            loss,
            params,
            &dataset,
            &[],
            &TrainConfig::new(200, OptimizerConfig::Sgd { lr: 0.01 }),
        )
        .expect("training should succeed");

        assert!(result.final_loss < 0.05);
    }

    #[test]
    fn end_to_end_train_with_adam_reduces_loss() {
        let (graph, loss) = build_linear_mse_graph();
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![0.0]).expect("valid tensor"),
        );

        let dataset = vec![
            sample(1.0, 2.0),
            sample(2.0, 4.0),
            sample(3.0, 6.0),
            sample(4.0, 8.0),
        ];

        let result = train_graph(
            &graph,
            loss,
            params,
            &dataset,
            &[],
            &TrainConfig::new(100, OptimizerConfig::Adam {
                lr: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            }),
        )
        .expect("training should succeed");

        assert!(result.final_loss < 0.05);
    }

    fn build_linear_mse_graph() -> (Graph, crate::ir::ValueId) {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = graph
            .add_op(block, Op::Parameter("w".to_string()))
            .expect("add op should succeed");
        let (_, y) = graph
            .add_op(block, Op::Input("y".to_string()))
            .expect("add op should succeed");
        let (_, pred) = graph
            .add_op(block, Op::MatMul(x, w))
            .expect("add op should succeed");
        let (_, diff) = graph
            .add_op(block, Op::Sub(pred, y))
            .expect("add op should succeed");
        let (_, sq) = graph
            .add_op(block, Op::Mul(diff, diff))
            .expect("add op should succeed");
        let (_, loss) = graph
            .add_op(block, Op::Output(sq))
            .expect("add op should succeed");
        (graph, loss)
    }

    fn sample(x: f32, y: f32) -> TrainSample {
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            Tensor::new(vec![1, 1], vec![x]).expect("valid tensor"),
        );
        inputs.insert(
            "y".to_string(),
            Tensor::new(vec![1, 1], vec![y]).expect("valid tensor"),
        );
        TrainSample { inputs }
    }
}
