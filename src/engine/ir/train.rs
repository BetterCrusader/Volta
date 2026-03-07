use std::collections::HashMap;
use std::sync::Arc;

use crate::ir::autograd::build_reverse_graph;
use crate::ir::interpreter::{ExecutionContext, RuntimeValue};
use crate::ir::optimizer::{OptimizerConfig, OptimizerState};
use crate::ir::tensor::Tensor;
use crate::ir::{
    Backend, CompilerFlags, CpuBackend, ExecutionPhase, Graph, Op, ValueId, build_execution_plan,
    execute_backward_with_saved_activations, execute_forward_and_save,
    execute_terminal_with_backend_buffered, execute_value_with_backend, verify_graph,
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
        Self {
            patience: 10,
            min_delta: 1e-4,
            restore_best_weights: true,
        }
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
        Self {
            epochs,
            optimizer,
            lr_schedule: None,
            clip_grad: None,
            early_stopping: None,
            gradient_accumulation_steps: 1,
        }
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
    let determinism = CompilerFlags::from_env().determinism;
    backend
        .capabilities()
        .validate(ExecutionPhase::Training, determinism)
        .map_err(|err| TrainError {
            message: format!("Backend capability check failed: {}", err.message),
        })?;
    let optimizer_name = match &config.optimizer {
        crate::ir::optimizer::OptimizerConfig::Adam { .. } => "adam",
        crate::ir::optimizer::OptimizerConfig::AdamW { .. } => "adamw",
        _ => "sgd",
    };
    backend
        .capabilities()
        .validate_optimizer(optimizer_name)
        .map_err(|err| TrainError {
            message: format!("Backend optimizer check failed: {}", err.message),
        })?;

    let forward_plan = build_execution_plan(forward_graph, &std::collections::HashSet::new())
        .map_err(|err| TrainError {
            message: format!("Failed to build forward execution plan: {}", err.message),
        })?;

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
    let tracked_param_names: HashMap<ValueId, String> = tracked_params
        .iter()
        .map(|(name, value_id)| (*value_id, name.clone()))
        .collect();

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
            gradient_graph
                .gradients
                .get(value_id)
                .copied()
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

            let batch_loss =
                scalar_from_runtime(loss_runtime.clone()).map_err(|e| TrainError { message: e })?;
            ensure_finite_scalar("forward", "loss", batch_loss)?;
            epoch_loss_sum += batch_loss;
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
                let grad_runtime = all_grads
                    .get(grad_value_id)
                    .ok_or_else(|| TrainError {
                        message: format!("Missing gradient for '{name}'"),
                    })?
                    .clone();
                let grad_tensor = runtime_to_tensor(grad_runtime).map_err(|e| TrainError {
                    message: format!("Gradient for '{name}' is invalid: {e}"),
                })?;
                ensure_finite_tensor("backward", &format!("gradient '{name}'"), &grad_tensor)?;
                match accum_grads.get_mut(value_id) {
                    Some(existing) => {
                        let data = Arc::make_mut(&mut existing.data);
                        for (a, &b) in data.iter_mut().zip(grad_tensor.data.iter()) {
                            *a += b;
                        }
                        ensure_finite_tensor(
                            "backward",
                            &format!("accumulated gradient '{name}'"),
                            existing,
                        )?;
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
                    for v in data.iter_mut() {
                        *v *= scale;
                    }
                }
            }

            let gradients_by_id = std::mem::take(&mut accum_grads);
            accum_count = 0;

            // Gradient clipping by global norm
            let mut gradients_by_id = gradients_by_id;
            ensure_finite_named_tensors(
                "backward",
                "gradient",
                &gradients_by_id,
                &tracked_param_names,
            )?;
            if let Some(max_norm) = config.clip_grad {
                if max_norm > 0.0 {
                    let global_sq_norm: f32 = gradients_by_id
                        .values()
                        .flat_map(|t| t.data.iter())
                        .map(|v| v * v)
                        .sum();
                    let global_norm = global_sq_norm.sqrt();
                    if global_norm > max_norm {
                        let scale = max_norm / global_norm;
                        for grad in gradients_by_id.values_mut() {
                            let data = Arc::make_mut(&mut grad.data);
                            for v in data.iter_mut() {
                                *v *= scale;
                            }
                        }
                    }
                }
            }
            ensure_finite_named_tensors(
                "backward",
                "gradient",
                &gradients_by_id,
                &tracked_param_names,
            )?;

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
            ensure_finite_named_tensors(
                "optimizer",
                "parameter",
                &raw_params,
                &tracked_param_names,
            )?;
            ensure_finite_optimizer_state("optimizer", &optimizer_state, &tracked_param_names)?;

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
                    for v in data.iter_mut() {
                        *v *= scale;
                    }
                }
            }

            let mut gradients_by_id = accum_grads;
            ensure_finite_named_tensors(
                "backward",
                "gradient",
                &gradients_by_id,
                &tracked_param_names,
            )?;

            // Gradient clipping
            if let Some(max_norm) = config.clip_grad {
                if max_norm > 0.0 {
                    let global_sq_norm: f32 = gradients_by_id
                        .values()
                        .flat_map(|t| t.data.iter())
                        .map(|v| v * v)
                        .sum();
                    let global_norm = global_sq_norm.sqrt();
                    if global_norm > max_norm {
                        let scale = max_norm / global_norm;
                        for grad in gradients_by_id.values_mut() {
                            let data = Arc::make_mut(&mut grad.data);
                            for v in data.iter_mut() {
                                *v *= scale;
                            }
                        }
                    }
                }
            }
            ensure_finite_named_tensors(
                "backward",
                "gradient",
                &gradients_by_id,
                &tracked_param_names,
            )?;

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
            ensure_finite_named_tensors(
                "optimizer",
                "parameter",
                &raw_params,
                &tracked_param_names,
            )?;
            ensure_finite_optimizer_state("optimizer", &optimizer_state, &tracked_param_names)?;

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
        ensure_finite_scalar("forward", "epoch_train_loss", epoch_train_loss)?;

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
                let val_loss =
                    scalar_from_runtime(loss_runtime).map_err(|e| TrainError { message: e })?;
                ensure_finite_scalar("forward", "validation_loss", val_loss)?;
                val_loss_sum += val_loss;

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
            ensure_finite_scalar("forward", "epoch_validation_loss", avg_val_loss)?;
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
                        epoch + 1,
                        es_best_loss
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
                ensure_finite_named_tensors(
                    "early_stopping_restore",
                    "parameter",
                    &raw_params,
                    &tracked_param_names,
                )?;
                ensure_finite_optimizer_state(
                    "early_stopping_restore",
                    &optimizer_state,
                    &tracked_param_names,
                )?;
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
        let loss = scalar_from_runtime(loss_runtime).map_err(|e| TrainError { message: e })?;
        ensure_finite_scalar("forward", "final_loss", loss)?;
        loss
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

fn ensure_finite_scalar(stage: &str, label: &str, value: f32) -> Result<(), TrainError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TrainError {
            message: format!("Non-finite {stage} {label}: {value}"),
        })
    }
}

fn ensure_finite_tensor(stage: &str, label: &str, tensor: &Tensor) -> Result<(), TrainError> {
    let contiguous = tensor.make_contiguous().map_err(|err| TrainError {
        message: format!("Invalid {stage} {label}: {}", err.message),
    })?;
    for (index, value) in contiguous
        .data
        .iter()
        .take(contiguous.logical_len())
        .enumerate()
    {
        if !value.is_finite() {
            return Err(TrainError {
                message: format!("Non-finite {stage} {label} at element {index}: {value}"),
            });
        }
    }
    Ok(())
}

fn ensure_finite_named_tensors(
    stage: &str,
    tensor_kind: &str,
    tensors: &HashMap<ValueId, Tensor>,
    tracked_param_names: &HashMap<ValueId, String>,
) -> Result<(), TrainError> {
    for (value_id, tensor) in tensors {
        let label = tracked_param_names
            .get(value_id)
            .map(|name| format!("{tensor_kind} '{name}' ({value_id})"))
            .unwrap_or_else(|| format!("{tensor_kind} {value_id}"));
        ensure_finite_tensor(stage, &label, tensor)?;
    }
    Ok(())
}

fn ensure_finite_optimizer_state(
    stage: &str,
    state: &OptimizerState,
    tracked_param_names: &HashMap<ValueId, String>,
) -> Result<(), TrainError> {
    ensure_finite_named_tensors(
        stage,
        "optimizer_state.adam_m",
        &state.adam_m,
        tracked_param_names,
    )?;
    ensure_finite_named_tensors(
        stage,
        "optimizer_state.adam_v",
        &state.adam_v,
        tracked_param_names,
    )?;
    ensure_finite_named_tensors(
        stage,
        "optimizer_state.rms_v",
        &state.rms_v,
        tracked_param_names,
    )?;
    ensure_finite_named_tensors(
        stage,
        "optimizer_state.rms_buf",
        &state.rms_buf,
        tracked_param_names,
    )?;
    ensure_finite_named_tensors(
        stage,
        "optimizer_state.adagrad_acc",
        &state.adagrad_acc,
        tracked_param_names,
    )?;
    ensure_finite_named_tensors(
        stage,
        "optimizer_state.lars_vel",
        &state.lars_vel,
        tracked_param_names,
    )?;
    Ok(())
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
        context
            .inputs
            .insert(name.clone(), RuntimeValue::Tensor(Arc::new(tensor.clone())));
    }
    for (name, value_id) in tracked_params {
        if let Some(arc_tensor) = parameter_tensors.get(value_id) {
            context
                .parameters
                .insert(name.clone(), RuntimeValue::Tensor(Arc::clone(arc_tensor)));
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
        OptimizerConfig::Adam {
            beta1,
            beta2,
            epsilon,
            ..
        } => OptimizerConfig::Adam {
            lr: new_lr,
            beta1,
            beta2,
            epsilon,
        },
        OptimizerConfig::AdamW {
            beta1,
            beta2,
            epsilon,
            weight_decay,
            ..
        } => OptimizerConfig::AdamW {
            lr: new_lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        },
        OptimizerConfig::RmsProp {
            alpha,
            epsilon,
            weight_decay,
            momentum,
            ..
        } => OptimizerConfig::RmsProp {
            lr: new_lr,
            alpha,
            epsilon,
            weight_decay,
            momentum,
        },
        OptimizerConfig::Adagrad {
            epsilon,
            weight_decay,
            ..
        } => OptimizerConfig::Adagrad {
            lr: new_lr,
            epsilon,
            weight_decay,
        },
        OptimizerConfig::Lars {
            momentum,
            weight_decay,
            trust_coeff,
            epsilon,
            ..
        } => OptimizerConfig::Lars {
            lr: new_lr,
            momentum,
            weight_decay,
            trust_coeff,
            epsilon,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::optimizer::LrSchedule;
    use crate::ir::{
        Backend, BackendCapabilities, BackendError, BackendKind, BackendMaturity, BackendVendor,
        CompiledProgram, DeterminismLevel, DeviceClass, EarlyStoppingConfig, ExecutionContext,
        ExecutionPlan, Graph, NodeId, Op, OptimizerConfig, Tensor, TrainConfig, TrainResult,
        TrainSample, ValueId, train_graph, train_graph_with_backend,
    };

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
            &TrainConfig::new(
                100,
                OptimizerConfig::Adam {
                    lr: 0.05,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                },
            ),
        )
        .expect("training should succeed");

        assert!(result.final_loss < 0.05);
    }

    #[derive(Debug, Clone, Copy)]
    struct InferenceOnlyBackend;

    impl Backend for InferenceOnlyBackend {
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                backend: BackendKind::Cpu,
                device_class: DeviceClass::Cpu,
                vendor: BackendVendor::GenericCpu,
                maturity: BackendMaturity::Experimental,
                supports_inference: true,
                supports_training: false,
                supports_runtime_execution: true,
                supports_gradient_updates: false,
                supports_adam: false,
                supports_strict_determinism: true,
                supports_balanced_determinism: true,
                supports_fast_determinism: true,
                default_determinism: DeterminismLevel::Strict,
            }
        }

        fn compile(&self, plan: &ExecutionPlan) -> Result<CompiledProgram, BackendError> {
            Ok(CompiledProgram {
                schedule_len: plan.schedule.ordered_nodes.len(),
                peak_bytes: plan.allocation.peak_bytes,
                fingerprint: 1,
            })
        }

        fn execute_terminal(
            &self,
            _graph: &Graph,
            _plan: &ExecutionPlan,
            _ordered_nodes: &[NodeId],
            _context: &ExecutionContext,
            _determinism: DeterminismLevel,
        ) -> Result<Option<crate::ir::RuntimeValue>, BackendError> {
            Err(BackendError {
                message: "should not execute".to_string(),
            })
        }

        fn execute_value(
            &self,
            _graph: &Graph,
            _plan: &ExecutionPlan,
            _target: ValueId,
            _ordered_nodes: &[NodeId],
            _context: &ExecutionContext,
            _determinism: DeterminismLevel,
        ) -> Result<crate::ir::RuntimeValue, BackendError> {
            Err(BackendError {
                message: "should not execute".to_string(),
            })
        }
    }

    #[test]
    fn training_rejects_backend_without_training_support() {
        let (graph, loss) = build_linear_mse_graph();
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![0.0]).expect("valid tensor"),
        );
        let dataset = vec![sample(1.0, 2.0)];
        let backend = InferenceOnlyBackend;

        let err = train_graph_with_backend(
            &graph,
            loss,
            None,
            params,
            &dataset,
            &[],
            &TrainConfig::new(1, OptimizerConfig::Sgd { lr: 0.01 }),
            &backend,
        )
        .expect_err("backend without training support must fail");

        assert!(err.message.contains("capability check failed"));
        assert!(err.message.contains("Training"));
    }

    #[test]
    fn lr_schedule_step_decay_changes_training_updates() {
        let (graph, loss) = build_linear_mse_graph();
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![1.0]).expect("valid tensor"),
        );
        let dataset = vec![sample(1.0, 0.0)];

        let mut config = TrainConfig::new(2, OptimizerConfig::Sgd { lr: 0.1 });
        config.lr_schedule = Some(LrSchedule::Step {
            step_size: 1,
            gamma: 0.5,
        });

        let result = train_graph(&graph, loss, params, &dataset, &[], &config)
            .expect("training with step lr schedule should succeed");

        assert!((weight_scalar(&result, "w") - 0.72).abs() < 1e-6);
        assert!((result.final_loss - 0.5184).abs() < 1e-6);
    }

    #[test]
    fn early_stopping_restores_best_weights() {
        let (graph, loss) = build_linear_mse_graph();
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![0.0]).expect("valid tensor"),
        );
        let dataset = vec![sample(1.0, 1.0)];
        let val_dataset = vec![sample(1.0, 1.0)];

        let mut config = TrainConfig::new(5, OptimizerConfig::Sgd { lr: 1.0 });
        config.early_stopping = Some(EarlyStoppingConfig {
            patience: 1,
            min_delta: 1e-6,
            restore_best_weights: true,
        });

        let result = train_graph(&graph, loss, params, &dataset, &val_dataset, &config)
            .expect("training with early stopping should succeed");

        assert_eq!(result.final_val_loss, Some(1.0));
        assert!((weight_scalar(&result, "w") - 2.0).abs() < 1e-6);
        assert!((result.final_loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn train_graph_rejects_non_finite_forward_loss() {
        let (graph, loss) = build_linear_mse_graph();
        let dataset = vec![sample(f32::NAN, 1.0)];

        let err = train_graph(
            &graph,
            loss,
            single_weight_params(0.0),
            &dataset,
            &[],
            &TrainConfig::new(1, OptimizerConfig::Sgd { lr: 0.01 }),
        )
        .expect_err("non-finite forward loss must fail");

        assert!(err.message.contains("Non-finite forward loss"));
    }

    #[test]
    fn train_graph_rejects_non_finite_optimizer_parameters() {
        let (graph, loss) = build_linear_mse_graph();
        let dataset = vec![sample(1.0e10, 0.0)];

        let err = train_graph(
            &graph,
            loss,
            single_weight_params(1.0),
            &dataset,
            &[],
            &TrainConfig::new(1, OptimizerConfig::Sgd { lr: 1.0e20 }),
        )
        .expect_err("non-finite optimizer output must fail");

        assert!(err.message.contains("Non-finite optimizer parameter"));
        assert!(err.message.contains("w"));
    }

    #[test]
    fn train_graph_long_loop_sgd_is_deterministic() {
        assert_long_loop_is_deterministic(OptimizerConfig::Sgd { lr: 0.01 });
    }

    #[test]
    fn train_graph_long_loop_adam_is_deterministic() {
        assert_long_loop_is_deterministic(OptimizerConfig::Adam {
            lr: 0.03,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });
    }

    #[test]
    fn train_graph_long_loop_adamw_is_deterministic() {
        assert_long_loop_is_deterministic(OptimizerConfig::AdamW {
            lr: 0.02,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        });
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

    fn single_weight_params(weight: f32) -> HashMap<String, Tensor> {
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![weight]).expect("valid tensor"),
        );
        params
    }

    fn assert_long_loop_is_deterministic(optimizer: OptimizerConfig) {
        let first = run_seeded_long_loop(0x5EED, optimizer.clone());
        let second = run_seeded_long_loop(0x5EED, optimizer);

        assert_eq!(first.final_loss.to_bits(), second.final_loss.to_bits());
        assert_eq!(first.final_parameters, second.final_parameters);
    }

    fn run_seeded_long_loop(seed: u64, optimizer: OptimizerConfig) -> TrainResult {
        let (graph, loss) = build_linear_mse_graph();
        let config = TrainConfig::new(24, optimizer);
        let params = single_weight_params(seeded_range(seed, -0.25, 0.25));
        let dataset = seeded_linear_dataset(seed, 6);

        let result = train_graph(&graph, loss, params, &dataset, &[], &config)
            .expect("long-loop training should succeed");

        assert!(result.final_loss.is_finite());
        result
    }

    fn seeded_linear_dataset(seed: u64, len: usize) -> Vec<TrainSample> {
        let mut state = seed;
        let mut dataset = Vec::with_capacity(len);
        for _ in 0..len {
            let x = seeded_next(&mut state) * 2.0 - 1.0;
            dataset.push(sample(x, x * 2.0));
        }
        dataset
    }

    fn seeded_range(seed: u64, min: f32, max: f32) -> f32 {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        min + (max - min) * seeded_next(&mut state)
    }

    fn seeded_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (*state >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }

    fn weight_scalar(result: &TrainResult, name: &str) -> f32 {
        result
            .final_parameters
            .get(name)
            .expect("parameter must exist")
            .data[0]
    }
}
