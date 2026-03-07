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
            RuntimeValue::Tensor(std::sync::Arc::new(tensor.clone())),
        );
    }
    for (name, tensor) in parameters {
        context.parameters.insert(
            name.clone(),
            RuntimeValue::Tensor(std::sync::Arc::new(tensor.clone())),
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
        RuntimeValue::Tensor(tensor) => Tensor::new(tensor.shape.clone(), tensor.data.to_vec())
            .map_err(|err| TrainApiError {
                message: format!("Infer output is invalid tensor: {}", err.message),
            }),
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

/// Trains a compiled model with an explicit backend.
///
/// This API handles checkpoint preloading, dataset batching/shuffling policy,
/// delegates training to IR-level training, and optionally writes checkpoints.
///
/// # Errors
/// Returns [`TrainApiError`] when dataset access fails, checkpoint IO/format
/// fails, graph training fails, or result conversion cannot be completed.
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
        None,
        parameters,
        &samples,
        &[],
        &TrainConfig::new(config.epochs, config.optimizer.clone()),
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

    use crate::ir::{Graph, NodeId, Op, OptimizerConfig, Tensor, ValueId, build_execution_plan};
    use crate::model::{
        CompiledModel, Conv2DLayer, Dataset, Example, LinearLayer, ModelBuilder, Parameter,
        ReLULayer, ReproducibilityMode, Sequential, TensorShape, TrainApiConfig,
        build_tiny_transformer_fixture_for_tests, infer, train,
    };

    #[derive(Debug, Clone)]
    struct TensorDataset {
        examples: Vec<Example>,
    }

    impl Dataset for TensorDataset {
        fn len(&self) -> usize {
            self.examples.len()
        }

        fn example(&self, index: usize) -> Result<Example, crate::model::TrainApiError> {
            Ok(self.examples[index].clone())
        }
    }

    #[test]
    fn mlp_long_loop_sgd_is_deterministic() {
        assert_mlp_long_loop_is_deterministic(OptimizerConfig::Sgd { lr: 0.05 });
    }

    #[test]
    fn mlp_long_loop_adam_is_deterministic() {
        assert_mlp_long_loop_is_deterministic(OptimizerConfig::Adam {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });
    }

    #[test]
    fn mlp_long_loop_adamw_is_deterministic() {
        assert_mlp_long_loop_is_deterministic(OptimizerConfig::AdamW {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        });
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

    fn build_mlp_long_loop_fixture_for_tests() -> (CompiledModel, TensorDataset) {
        let mut graph = crate::ir::Graph::new();
        let block = graph.create_block();

        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, target) = graph
            .add_op(block, Op::Input("target".to_string()))
            .expect("add op should succeed");
        let (_, w1) = graph
            .add_op(block, Op::Parameter("w1".to_string()))
            .expect("add op should succeed");
        let (_, b1) = graph
            .add_op(block, Op::Parameter("b1".to_string()))
            .expect("add op should succeed");
        let (_, w2) = graph
            .add_op(block, Op::Parameter("w2".to_string()))
            .expect("add op should succeed");
        let (_, b2) = graph
            .add_op(block, Op::Parameter("b2".to_string()))
            .expect("add op should succeed");

        let (_, hidden) = graph
            .add_op(
                block,
                Op::Gemm {
                    lhs: x,
                    rhs: w1,
                    bias: Some(b1),
                    alpha: 1.0,
                    beta: 1.0,
                },
            )
            .expect("add op should succeed");
        let (_, hidden_relu) = graph
            .add_op(block, Op::Relu(hidden))
            .expect("add op should succeed");
        let (_, pred) = graph
            .add_op(
                block,
                Op::Gemm {
                    lhs: hidden_relu,
                    rhs: w2,
                    bias: Some(b2),
                    alpha: 1.0,
                    beta: 1.0,
                },
            )
            .expect("add op should succeed");
        let (_, diff) = graph
            .add_op(block, Op::Sub(pred, target))
            .expect("add op should succeed");
        let (_, sq) = graph
            .add_op(block, Op::Mul(diff, diff))
            .expect("add op should succeed");
        let (_, loss) = graph
            .add_op(
                block,
                Op::ReduceMean {
                    input: sq,
                    axis: None,
                    keepdims: false,
                },
            )
            .expect("add op should succeed");

        graph.bind_input_shape("x", vec![2, 3]);
        graph.bind_input_shape("target", vec![2, 2]);
        graph.bind_parameter_shape("w1", vec![3, 4]);
        graph.bind_parameter_shape("b1", vec![4]);
        graph.bind_parameter_shape("w2", vec![4, 2]);
        graph.bind_parameter_shape("b2", vec![2]);

        let parameters = HashMap::from([
            (
                "w1".to_string(),
                Tensor::new(
                    vec![3, 4],
                    vec![
                        0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.2, -0.3,
                    ],
                )
                .expect("valid tensor"),
            ),
            (
                "b1".to_string(),
                Tensor::new(vec![4], vec![0.05, -0.1, 0.15, 0.2]).expect("valid tensor"),
            ),
            (
                "w2".to_string(),
                Tensor::new(vec![4, 2], vec![0.2, -0.4, 0.1, 0.3, -0.5, 0.7, 0.6, -0.2])
                    .expect("valid tensor"),
            ),
            (
                "b2".to_string(),
                Tensor::new(vec![2], vec![0.25, -0.35]).expect("valid tensor"),
            ),
        ]);
        let parameter_values = HashMap::from([
            ("w1".to_string(), w1),
            ("b1".to_string(), b1),
            ("w2".to_string(), w2),
            ("b2".to_string(), b2),
        ]);

        let inference_plan = build_execution_plan(&graph, &std::collections::HashSet::new())
            .expect("infer plan should build");
        let inference_ordered_nodes =
            infer_nodes_for_target(&graph, pred, &inference_plan.schedule.ordered_nodes)
                .expect("infer dependency nodes should resolve");

        let model = CompiledModel {
            graph,
            output: pred,
            output_shape: TensorShape(vec![2, 2]),
            loss: Some(loss),
            parameters,
            parameter_values,
            inference_plan,
            inference_ordered_nodes,
        };

        let dataset = TensorDataset {
            examples: vec![
                Example {
                    inputs: HashMap::from([
                        (
                            "x".to_string(),
                            Tensor::new(vec![2, 3], vec![0.2, -0.1, 0.3, 0.7, 0.5, -0.4])
                                .expect("valid tensor"),
                        ),
                        (
                            "target".to_string(),
                            Tensor::new(vec![2, 2], vec![0.1, -0.2, 0.3, 0.4])
                                .expect("valid tensor"),
                        ),
                    ]),
                },
                Example {
                    inputs: HashMap::from([
                        (
                            "x".to_string(),
                            Tensor::new(vec![2, 3], vec![0.4, 0.2, -0.5, -0.3, 0.6, 0.8])
                                .expect("valid tensor"),
                        ),
                        (
                            "target".to_string(),
                            Tensor::new(vec![2, 2], vec![0.2, 0.05, -0.1, 0.6])
                                .expect("valid tensor"),
                        ),
                    ]),
                },
            ],
        };

        (model, dataset)
    }

    fn build_mlp_long_loop_config(optimizer: OptimizerConfig) -> TrainApiConfig {
        TrainApiConfig {
            epochs: 24,
            batch_size: 1,
            shuffle: true,
            shuffle_seed: 0x5EED,
            optimizer,
            gradient_checkpointing: None,
            reproducibility: ReproducibilityMode::Deterministic,
            checkpoint_path: None,
        }
    }

    fn assert_mlp_long_loop_is_deterministic(optimizer: OptimizerConfig) {
        let (model, dataset) = build_mlp_long_loop_fixture_for_tests();
        let config = build_mlp_long_loop_config(optimizer);
        let initial_loss = average_mlp_dataset_loss(&model, &model.parameters, &dataset);

        let first = train(&model, &dataset, &config).expect("first long-loop train should pass");
        let second = train(&model, &dataset, &config).expect("second long-loop train should pass");

        assert!(first.final_loss.is_finite());
        assert!(second.final_loss.is_finite());
        assert_eq!(first.final_loss.to_bits(), second.final_loss.to_bits());
        assert_eq!(first.final_parameters, second.final_parameters);
        assert_all_tensors_are_finite(&first.final_parameters);

        let final_loss = average_mlp_dataset_loss(&model, &first.final_parameters, &dataset);
        assert!(
            final_loss < initial_loss,
            "long-loop fixture should reduce average MLP loss: before={initial_loss}, after={final_loss}"
        );
    }

    #[test]
    fn convnet_train_api_is_deterministic_and_reduces_loss() {
        let (model, dataset, config) = build_convnet_fixture_for_tests();
        assert_model_contains_conv2d(&model);

        let initial_loss = average_convnet_dataset_loss(&model, &model.parameters, &dataset);
        let first = train(&model, &dataset, &config).expect("first ConvNet train should pass");
        let second = train(&model, &dataset, &config).expect("second ConvNet train should pass");

        assert!(first.final_loss.is_finite());
        assert!(second.final_loss.is_finite());
        assert_eq!(first.final_loss.to_bits(), second.final_loss.to_bits());
        assert_eq!(first.final_parameters, second.final_parameters);
        assert_all_tensors_are_finite(&first.final_parameters);

        let final_loss = average_convnet_dataset_loss(&model, &first.final_parameters, &dataset);
        assert!(
            final_loss < initial_loss,
            "ConvNet fixture should reduce average dataset loss: before={initial_loss}, after={final_loss}"
        );
    }

    fn average_mlp_dataset_loss(
        model: &CompiledModel,
        parameters: &HashMap<String, Tensor>,
        dataset: &TensorDataset,
    ) -> f32 {
        let mut total = 0.0;
        for example in &dataset.examples {
            let predicted = infer(
                model,
                parameters,
                &HashMap::from([(
                    "x".to_string(),
                    example
                        .inputs
                        .get("x")
                        .expect("MLP example should contain x")
                        .clone(),
                )]),
            )
            .expect("infer should pass on MLP long-loop fixture");
            let target = example
                .inputs
                .get("target")
                .expect("MLP example should contain target");
            total += mean_squared_error(&predicted, target);
        }
        total / dataset.len() as f32
    }

    fn mean_squared_error(lhs: &Tensor, rhs: &Tensor) -> f32 {
        assert_eq!(
            lhs.shape, rhs.shape,
            "MSE tensors must share the same shape"
        );
        let element_count = lhs.logical_len();
        let squared_error_sum: f32 = lhs
            .data
            .iter()
            .zip(rhs.data.iter())
            .take(element_count)
            .map(|(actual, expected)| {
                let delta = actual - expected;
                delta * delta
            })
            .sum();
        squared_error_sum / element_count as f32
    }

    fn build_convnet_fixture_for_tests() -> (CompiledModel, TensorDataset, TrainApiConfig) {
        let mut builder = ModelBuilder::new();
        let x = builder
            .input_with_shape("x", vec![3, 3])
            .expect("ConvNet input should build");
        let target = builder
            .input_with_shape("target", vec![2, 1])
            .expect("ConvNet target should build");

        let mut seq = Sequential::new();
        seq.push(Conv2DLayer {
            name: "conv".to_string(),
            kernel: Tensor::new(vec![2, 2], vec![0.35, -0.05, 0.1, 0.2]).expect("valid kernel"),
        });
        seq.push(ReLULayer);
        seq.push(LinearLayer {
            name: "proj".to_string(),
            weight: Tensor::new(vec![2, 1], vec![0.15, -0.05]).expect("valid projection"),
        });

        let (pred, pred_shape) = seq
            .build(&mut builder, x, TensorShape(vec![3, 3]))
            .expect("ConvNet body should build");
        let proj_bias = builder
            .add_parameter(Parameter::new(
                "proj.bias",
                Tensor::new(vec![1], vec![0.02]).expect("valid bias"),
                true,
            ))
            .expect("ConvNet bias should build");
        let pred = builder
            .add_op(Op::Add(pred, proj_bias))
            .expect("ConvNet bias add should build");

        let diff = builder
            .add_op(Op::Sub(pred, target))
            .expect("ConvNet diff should build");
        let sq = builder
            .add_op(Op::Mul(diff, diff))
            .expect("ConvNet square should build");
        let loss = builder
            .add_op(Op::ReduceMean {
                input: sq,
                axis: None,
                keepdims: false,
            })
            .expect("ConvNet loss should build");

        let model = builder
            .finalize(pred, pred_shape, Some(loss))
            .expect("ConvNet model should finalize");

        let dataset = TensorDataset {
            examples: vec![
                Example {
                    inputs: HashMap::from([
                        (
                            "x".to_string(),
                            Tensor::new(
                                vec![3, 3],
                                vec![1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0],
                            )
                            .expect("valid tensor"),
                        ),
                        (
                            "target".to_string(),
                            Tensor::new(vec![2, 1], vec![0.87, 0.14]).expect("valid tensor"),
                        ),
                    ]),
                },
                Example {
                    inputs: HashMap::from([
                        (
                            "x".to_string(),
                            Tensor::new(
                                vec![3, 3],
                                vec![0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 2.0],
                            )
                            .expect("valid tensor"),
                        ),
                        (
                            "target".to_string(),
                            Tensor::new(vec![2, 1], vec![-0.18, 0.45]).expect("valid tensor"),
                        ),
                    ]),
                },
                Example {
                    inputs: HashMap::from([
                        (
                            "x".to_string(),
                            Tensor::new(
                                vec![3, 3],
                                vec![1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                            )
                            .expect("valid tensor"),
                        ),
                        (
                            "target".to_string(),
                            Tensor::new(vec![2, 1], vec![-0.11, -0.52]).expect("valid tensor"),
                        ),
                    ]),
                },
            ],
        };

        let config = TrainApiConfig {
            epochs: 24,
            batch_size: 1,
            shuffle: true,
            shuffle_seed: 0xC0DE_0017,
            optimizer: OptimizerConfig::Sgd { lr: 0.02 },
            gradient_checkpointing: None,
            reproducibility: ReproducibilityMode::Deterministic,
            checkpoint_path: None,
        };

        (model, dataset, config)
    }

    fn assert_model_contains_conv2d(model: &CompiledModel) {
        let conv_count = model
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(node.op, Op::Conv2D(_, _)))
            .count();
        assert_eq!(conv_count, 1, "ConvNet fixture must include one Conv2D op");
    }

    fn average_convnet_dataset_loss(
        model: &CompiledModel,
        parameters: &HashMap<String, Tensor>,
        dataset: &TensorDataset,
    ) -> f32 {
        let mut total = 0.0;
        for example in &dataset.examples {
            let predicted = infer(
                model,
                parameters,
                &HashMap::from([(
                    "x".to_string(),
                    example
                        .inputs
                        .get("x")
                        .expect("ConvNet example should contain x")
                        .clone(),
                )]),
            )
            .expect("infer should pass on ConvNet fixture");
            let target = example
                .inputs
                .get("target")
                .expect("ConvNet example should contain target");
            total += mean_squared_error(&predicted, target);
        }
        total / dataset.len() as f32
    }

    fn assert_all_tensors_are_finite(parameters: &HashMap<String, Tensor>) {
        for (name, tensor) in parameters {
            for (index, value) in tensor.data.iter().take(tensor.logical_len()).enumerate() {
                assert!(
                    value.is_finite(),
                    "parameter '{name}' contains non-finite value at element {index}: {value}"
                );
            }
        }
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
