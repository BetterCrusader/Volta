use std::collections::HashMap;

use crate::ir::autograd::build_reverse_graph;
use crate::ir::interpreter::{ExecutionContext, RuntimeValue};
use crate::ir::optimizer::{OptimizerConfig, OptimizerState};
use crate::ir::tensor::Tensor;
use crate::ir::{
    Backend, CompilerFlags, CpuBackend, Graph, Op, ValueId, build_execution_plan,
    execute_terminal_with_backend, execute_value_with_backend, verify_graph,
};

#[derive(Debug, Clone)]
pub struct TrainSample {
    pub inputs: HashMap<String, Tensor>,
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub optimizer: OptimizerConfig,
}

#[derive(Debug, Clone)]
pub struct TrainResult {
    pub final_parameters: HashMap<String, Tensor>,
    pub final_loss: f32,
    pub optimizer_state: OptimizerState,
}

#[derive(Debug, Clone)]
pub struct TrainError {
    pub message: String,
}

pub fn train_graph(
    forward_graph: &Graph,
    loss_value: ValueId,
    initial_parameters: HashMap<String, Tensor>,
    dataset: &[TrainSample],
    config: &TrainConfig,
) -> Result<TrainResult, TrainError> {
    let backend = CpuBackend;
    train_graph_with_backend(
        forward_graph,
        loss_value,
        initial_parameters,
        dataset,
        config,
        &backend,
    )
}

pub fn train_graph_with_backend(
    forward_graph: &Graph,
    loss_value: ValueId,
    initial_parameters: HashMap<String, Tensor>,
    dataset: &[TrainSample],
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
    let parameter_names = collect_parameter_names(forward_graph);

    let mut parameter_tensors = initial_parameters;
    let mut optimizer_state = OptimizerState::default();

    let tracked_params = parameter_names
        .iter()
        .filter_map(|name| {
            parameter_values
                .get(name)
                .copied()
                .map(|value| (name.clone(), value))
        })
        .collect::<Vec<_>>();
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

    for _ in 0..config.epochs {
        for sample in dataset {
            let mut context = build_context(sample, &parameter_tensors);
            let loss_value_runtime = execute_terminal_with_backend(
                forward_graph,
                &forward_plan,
                &forward_plan.schedule.ordered_nodes,
                backend,
                &context,
            )
            .map_err(|e| TrainError {
                message: format!("Forward execute failed: {}", e.message),
            })?;
            let Some(loss_runtime) = loss_value_runtime else {
                return Err(TrainError {
                    message: "Forward graph produced no loss output".to_string(),
                });
            };
            let seed = ones_like(&loss_runtime).map_err(|e| TrainError { message: e })?;
            context.inputs.insert("__loss_grad".to_string(), seed);

            let mut gradients_by_name = HashMap::new();
            for (name, value_id) in &tracked_params {
                let Some(grad_value_id) = gradient_graph.gradients.get(value_id).copied() else {
                    continue;
                };
                let grad_runtime = execute_value_with_backend(
                    &gradient_graph.backward,
                    &backward_plan,
                    grad_value_id,
                    &backward_plan.schedule.ordered_nodes,
                    backend,
                    &context,
                )
                .map_err(|e| TrainError {
                    message: format!("Backward execute failed for '{name}': {}", e.message),
                })?;
                let grad_tensor = runtime_to_tensor(grad_runtime).map_err(|e| TrainError {
                    message: format!("Gradient for '{name}' is invalid: {e}"),
                })?;
                gradients_by_name.insert(name.clone(), grad_tensor);
            }

            backend
                .apply_gradients(
                    &mut parameter_tensors,
                    &gradients_by_name,
                    &config.optimizer,
                    &mut optimizer_state,
                    determinism,
                )
                .map_err(|e| TrainError {
                    message: format!("Optimizer step failed: {}", e.message),
                })?;
        }
    }

    let final_loss = if let Some(sample) = dataset.last() {
        let context = build_context(sample, &parameter_tensors);
        let loss_runtime = execute_terminal_with_backend(
            forward_graph,
            &forward_plan,
            &forward_plan.schedule.ordered_nodes,
            backend,
            &context,
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

    Ok(TrainResult {
        final_parameters: parameter_tensors,
        final_loss,
        optimizer_state,
    })
}

fn collect_parameter_names(graph: &Graph) -> Vec<String> {
    let mut names = Vec::new();
    for node in &graph.nodes {
        if let Op::Parameter(name) = &node.op {
            names.push(name.clone());
        }
    }
    names
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

fn build_context(sample: &TrainSample, parameters: &HashMap<String, Tensor>) -> ExecutionContext {
    let mut context = ExecutionContext::default();
    for (name, tensor) in &sample.inputs {
        context.inputs.insert(
            name.clone(),
            RuntimeValue::Tensor {
                shape: tensor.shape.clone(),
                data: tensor.data.clone(),
            },
        );
    }
    for (name, tensor) in parameters {
        context.parameters.insert(
            name.clone(),
            RuntimeValue::Tensor {
                shape: tensor.shape.clone(),
                data: tensor.data.clone(),
            },
        );
    }
    context
}

fn runtime_to_tensor(value: RuntimeValue) -> Result<Tensor, String> {
    match value {
        RuntimeValue::Tensor { shape, data } => Tensor::new(shape, data).map_err(|e| e.message),
        _ => Err("Expected tensor runtime value".to_string()),
    }
}

fn scalar_from_runtime(value: RuntimeValue) -> Result<f32, String> {
    match value {
        RuntimeValue::Tensor { shape, data } => {
            if data.len() != 1 {
                return Err(format!(
                    "Expected scalar tensor for loss, got shape {:?}",
                    shape
                ));
            }
            Ok(data[0])
        }
        RuntimeValue::Float(v) => Ok(v as f32),
        RuntimeValue::Int(v) => Ok(v as f32),
    }
}

fn ones_like(value: &RuntimeValue) -> Result<RuntimeValue, String> {
    match value {
        RuntimeValue::Tensor { shape, data } => Ok(RuntimeValue::Tensor {
            shape: shape.clone(),
            data: vec![1.0; data.len()],
        }),
        RuntimeValue::Float(_) => Ok(RuntimeValue::Float(1.0)),
        RuntimeValue::Int(_) => Ok(RuntimeValue::Int(1)),
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
            &TrainConfig {
                epochs: 200,
                optimizer: OptimizerConfig::Sgd { lr: 0.01 },
            },
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
            &TrainConfig {
                epochs: 100,
                optimizer: OptimizerConfig::Adam {
                    lr: 0.05,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                },
            },
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
