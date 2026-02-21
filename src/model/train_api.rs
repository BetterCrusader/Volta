use std::collections::HashMap;

use crate::ir::{
    Backend, CpuBackend, ExecutionContext, OptimizerConfig, OptimizerState, RuntimeValue, Tensor,
    TrainConfig, TrainSample, execute_value_with_backend, train_graph_with_backend,
};

use crate::model::{
    BatchIterator, CompiledModel, Dataset, Example, GradientCheckpointingConfig, load_checkpoint,
    plan_gradient_checkpointing, save_checkpoint,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReproducibilityMode {
    Deterministic,
    Fast,
}

#[derive(Debug, Clone)]
pub struct TrainApiConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub shuffle: bool,
    pub shuffle_seed: u64,
    pub optimizer: OptimizerConfig,
    pub gradient_checkpointing: Option<GradientCheckpointingConfig>,
    pub reproducibility: ReproducibilityMode,
    pub checkpoint_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TrainApiResult {
    pub final_parameters: HashMap<String, Tensor>,
    pub final_loss: f32,
    pub optimizer_state: OptimizerState,
}

#[derive(Debug, Clone)]
pub struct TrainApiError {
    pub message: String,
}

pub fn infer(
    model: &CompiledModel,
    parameters: &HashMap<String, Tensor>,
    inputs: &HashMap<String, Tensor>,
) -> Result<Tensor, TrainApiError> {
    let backend = CpuBackend;
    infer_with_backend(model, parameters, inputs, &backend)
}

pub fn infer_with_backend(
    model: &CompiledModel,
    parameters: &HashMap<String, Tensor>,
    inputs: &HashMap<String, Tensor>,
    backend: &dyn Backend,
) -> Result<Tensor, TrainApiError> {
    let mut context = ExecutionContext::default();
    for (name, tensor) in inputs {
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

    let runtime = execute_value_with_backend(
        &model.graph,
        &model.inference_plan,
        model.output,
        &model.inference_ordered_nodes,
        backend,
        &context,
    )
    .map_err(|err| TrainApiError {
        message: format!("Infer execution failed: {}", err.message),
    })?;

    match runtime {
        RuntimeValue::Tensor { shape, data } => {
            Tensor::new(shape, data).map_err(|err| TrainApiError {
                message: format!("Infer output is invalid tensor: {}", err.message),
            })
        }
        RuntimeValue::Float(_) | RuntimeValue::Int(_) => Err(TrainApiError {
            message: "Infer output must be a tensor".to_string(),
        }),
    }
}

pub fn train<D: Dataset>(
    model: &CompiledModel,
    dataset: &D,
    config: &TrainApiConfig,
) -> Result<TrainApiResult, TrainApiError> {
    let backend = CpuBackend;
    train_with_backend(model, dataset, config, &backend)
}

pub fn train_with_backend<D: Dataset>(
    model: &CompiledModel,
    dataset: &D,
    config: &TrainApiConfig,
    backend: &dyn Backend,
) -> Result<TrainApiResult, TrainApiError> {
    let mut parameters = model.parameters.clone();
    if let Some(path) = &config.checkpoint_path {
        let loaded = load_checkpoint(path)?;
        for (name, tensor) in loaded {
            parameters.insert(name, tensor);
        }
    }

    let shuffle = match config.reproducibility {
        ReproducibilityMode::Deterministic => config.shuffle,
        ReproducibilityMode::Fast => config.shuffle,
    };

    let mut samples = Vec::<TrainSample>::new();
    for batch in BatchIterator::new(
        dataset.len(),
        config.batch_size,
        shuffle,
        config.shuffle_seed,
    ) {
        for index in batch {
            let Example { inputs } = dataset.example(index)?;
            samples.push(TrainSample { inputs });
        }
    }

    let Some(loss) = model.loss else {
        return Err(TrainApiError {
            message: "Compiled model has no loss node".to_string(),
        });
    };

    if let Some(checkpointing) = &config.gradient_checkpointing {
        let _ = plan_gradient_checkpointing(model, checkpointing).map_err(|err| TrainApiError {
            message: format!("Gradient checkpointing planning failed: {}", err.message),
        })?;
    }

    let result = train_graph_with_backend(
        &model.graph,
        loss,
        parameters,
        &samples,
        &TrainConfig {
            epochs: config.epochs,
            optimizer: config.optimizer.clone(),
        },
        backend,
    )
    .map_err(|err| TrainApiError {
        message: format!("Train pipeline failed: {}", err.message),
    })?;

    if let Some(path) = &config.checkpoint_path {
        save_checkpoint(path, &result.final_parameters)?;
    }

    Ok(TrainApiResult {
        final_parameters: result.final_parameters,
        final_loss: result.final_loss,
        optimizer_state: result.optimizer_state,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::{Graph, NodeId, OptimizerConfig, Tensor, ValueId, build_execution_plan};
    use crate::model::{
        CompiledModel, Dataset, Example, ReproducibilityMode, TensorShape, TrainApiConfig,
        build_tiny_transformer_fixture_for_tests, infer, train,
    };

    struct TinyDataset {
        rows: Vec<(f32, f32)>,
    }

    impl Dataset for TinyDataset {
        fn len(&self) -> usize {
            self.rows.len()
        }

        fn example(&self, index: usize) -> Result<Example, crate::model::TrainApiError> {
            let (x, y) = self.rows[index];
            let mut inputs = HashMap::new();
            inputs.insert(
                "x".to_string(),
                Tensor::new(vec![1, 1], vec![x]).expect("valid tensor"),
            );
            inputs.insert(
                "y".to_string(),
                Tensor::new(vec![1, 1], vec![y]).expect("valid tensor"),
            );
            Ok(Example { inputs })
        }
    }

    #[test]
    fn deterministic_mode_produces_stable_result() {
        let (model, dataset) = fixture();
        let config = TrainApiConfig {
            epochs: 20,
            batch_size: 2,
            shuffle: true,
            shuffle_seed: 7,
            optimizer: OptimizerConfig::Sgd { lr: 0.01 },
            gradient_checkpointing: None,
            reproducibility: ReproducibilityMode::Deterministic,
            checkpoint_path: None,
        };

        let a = train(&model, &dataset, &config).expect("train should pass");
        let b = train(&model, &dataset, &config).expect("train should pass");
        assert!((a.final_loss - b.final_loss).abs() < 1e-9);
    }

    #[test]
    fn infer_returns_output_tensor_with_expected_shape() {
        let (model, _dataset, _cfg, infer_input) = build_tiny_transformer_fixture_for_tests();
        let out = infer(&model, &model.parameters, &infer_input).expect("infer should pass");
        assert_eq!(out.shape, model.output_shape.0);
    }

    #[test]
    fn infer_is_repeatable_for_same_inputs_and_parameters() {
        let (model, _dataset, _cfg, infer_input) = build_tiny_transformer_fixture_for_tests();

        let first =
            infer(&model, &model.parameters, &infer_input).expect("first infer should pass");
        let second =
            infer(&model, &model.parameters, &infer_input).expect("second infer should pass");

        assert_eq!(
            first, second,
            "infer must be deterministic for fixed inputs"
        );
    }

    #[test]
    fn infer_does_not_mutate_trained_parameters_or_loss_result() {
        let (model, dataset, cfg, infer_input) = build_tiny_transformer_fixture_for_tests();
        let trained = train(&model, &dataset, &cfg).expect("train should pass");

        let params_before = trained.final_parameters.clone();
        let loss_before = trained.final_loss;

        let _ = infer(&model, &trained.final_parameters, &infer_input).expect("infer should pass");

        assert_eq!(trained.final_parameters, params_before);
        assert_eq!(trained.final_loss, loss_before);
    }

    fn fixture() -> (CompiledModel, TinyDataset) {
        let mut graph = crate::ir::Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, crate::ir::Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = graph
            .add_op(block, crate::ir::Op::Parameter("w".to_string()))
            .expect("add op should succeed");
        let (_, y) = graph
            .add_op(block, crate::ir::Op::Input("y".to_string()))
            .expect("add op should succeed");
        let (_, pred) = graph
            .add_op(block, crate::ir::Op::MatMul(x, w))
            .expect("add op should succeed");
        let (_, diff) = graph
            .add_op(block, crate::ir::Op::Sub(pred, y))
            .expect("add op should succeed");
        let (_, sq) = graph
            .add_op(block, crate::ir::Op::Mul(diff, diff))
            .expect("add op should succeed");
        let (_, loss) = graph
            .add_op(block, crate::ir::Op::Output(sq))
            .expect("add op should succeed");

        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![0.0]).expect("valid tensor"),
        );
        let mut param_values = HashMap::new();
        param_values.insert("w".to_string(), w);

        let inference_plan = build_execution_plan(&graph, &std::collections::HashSet::new())
            .expect("infer plan should build");
        let inference_ordered_nodes =
            infer_nodes_for_target(&graph, pred, &inference_plan.schedule.ordered_nodes)
                .expect("infer dependency nodes should resolve");

        let model = CompiledModel {
            graph,
            output: pred,
            output_shape: TensorShape(vec![1, 1]),
            loss: Some(loss),
            parameters: params,
            parameter_values: param_values,
            inference_plan,
            inference_ordered_nodes,
        };

        let dataset = TinyDataset {
            rows: vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)],
        };

        (model, dataset)
    }

    fn infer_nodes_for_target(
        graph: &Graph,
        target: ValueId,
        ordered_nodes: &[NodeId],
    ) -> Result<Vec<NodeId>, String> {
        if target.0 >= graph.nodes.len() {
            return Err(format!("target out of range: {}", target.0));
        }

        let mut required_values = std::collections::HashSet::<ValueId>::new();
        let mut stack = vec![target];
        while let Some(value) = stack.pop() {
            if !required_values.insert(value) {
                continue;
            }
            let node = graph
                .nodes
                .get(value.0)
                .ok_or_else(|| format!("dependency out of range: {}", value.0))?;
            for input in node.op.input_values() {
                stack.push(input);
            }
        }

        let mut filtered = Vec::new();
        for node_id in ordered_nodes {
            let node = graph
                .nodes
                .get(node_id.0)
                .ok_or_else(|| format!("schedule node out of range: {}", node_id.0))?;
            if required_values.contains(&node.output) {
                filtered.push(*node_id);
            }
        }

        Ok(filtered)
    }
}
