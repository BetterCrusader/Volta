use std::collections::{HashMap, HashSet};

use crate::ir::{
    Graph, NodeId, Op, OptimizerConfig, Tensor, TransformerConfig, ValueId,
    add_transformer_encoder_block, build_execution_plan,
};

use super::train_api::{ReproducibilityMode, TrainApiConfig, TrainApiError};
use super::{CompiledModel, Dataset, Example, TensorShape};

#[derive(Debug, Clone)]
struct FixtureSample {
    x: [f32; 8],
    target: [f32; 8],
}

#[derive(Debug, Clone)]
pub struct TinyTransformerFixtureDataset {
    rows: Vec<FixtureSample>,
}

impl TinyTransformerFixtureDataset {
    fn new() -> Self {
        Self {
            rows: vec![
                FixtureSample {
                    x: [0.2, -0.1, 0.3, 0.4, 0.7, 0.5, -0.4, 0.1],
                    target: [0.05, -0.1, 0.2, 0.3, 0.4, -0.2, 0.1, 0.5],
                },
                FixtureSample {
                    x: [0.4, 0.2, -0.5, 0.6, -0.3, 0.8, 0.1, -0.7],
                    target: [0.15, 0.05, -0.2, 0.25, -0.1, 0.3, 0.6, -0.4],
                },
            ],
        }
    }
}

impl Dataset for TinyTransformerFixtureDataset {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn example(&self, index: usize) -> Result<Example, TrainApiError> {
        let Some(row) = self.rows.get(index) else {
            return Err(TrainApiError {
                message: format!("tiny-transformer fixture index out of bounds: {index}"),
            });
        };

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            Tensor::new(vec![1, 2, 4], row.x.to_vec()).map_err(|err| TrainApiError {
                message: format!("fixture tensor x is invalid: {}", err.message),
            })?,
        );
        inputs.insert(
            "target".to_string(),
            Tensor::new(vec![2, 4], row.target.to_vec()).map_err(|err| TrainApiError {
                message: format!("fixture tensor target is invalid: {}", err.message),
            })?,
        );
        Ok(Example { inputs })
    }
}

#[must_use]
pub fn build_tiny_transformer_fixture_for_tests() -> (
    CompiledModel,
    TinyTransformerFixtureDataset,
    TrainApiConfig,
    HashMap<String, Tensor>,
) {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph
        .add_op(block, Op::Input("x".to_string()))
        .expect("fixture input x must build");
    let (_, target) = graph
        .add_op(block, Op::Input("target".to_string()))
        .expect("fixture input target must build");

    let mut parameter_values = HashMap::new();
    for name in [
        "w_q", "w_k", "w_v", "w_o", "b_q", "b_k", "b_v", "b_o", "ln1_w", "ln1_b", "ffn_w1",
        "ffn_b1", "ffn_w2", "ffn_b2", "ln2_w", "ln2_b",
    ] {
        let (_, value) = graph
            .add_op(block, Op::Parameter(name.to_string()))
            .expect("fixture parameter must build");
        parameter_values.insert(name.to_string(), value);
    }

    let config = TransformerConfig {
        d_model: 4,
        num_heads: 2,
        ffn_dim: 6,
        dropout: 0.0,
        causal: false,
        epsilon: 1e-5,
    };

    let out = add_transformer_encoder_block(
        &mut graph,
        block,
        x,
        parameter_values["w_q"],
        parameter_values["w_k"],
        parameter_values["w_v"],
        parameter_values["w_o"],
        parameter_values["b_q"],
        parameter_values["b_k"],
        parameter_values["b_v"],
        parameter_values["b_o"],
        parameter_values["ln1_w"],
        parameter_values["ln1_b"],
        parameter_values["ffn_w1"],
        parameter_values["ffn_b1"],
        parameter_values["ffn_w2"],
        parameter_values["ffn_b2"],
        parameter_values["ln2_w"],
        parameter_values["ln2_b"],
        &config,
    )
    .expect("fixture transformer block must build");

    let (_, diff) = graph
        .add_op(block, Op::Sub(out, target))
        .expect("fixture diff must build");
    let (_, sq) = graph
        .add_op(block, Op::Mul(diff, diff))
        .expect("fixture square must build");
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceMean {
                input: sq,
                axis: None,
                keepdims: false,
            },
        )
        .expect("fixture loss must build");

    graph.bind_input_shape("x", vec![1, 2, 4]);
    graph.bind_input_shape("target", vec![2, 4]);
    graph.bind_parameter_shape("w_q", vec![4, 4]);
    graph.bind_parameter_shape("w_k", vec![4, 4]);
    graph.bind_parameter_shape("w_v", vec![4, 4]);
    graph.bind_parameter_shape("w_o", vec![4, 4]);
    graph.bind_parameter_shape("b_q", vec![4]);
    graph.bind_parameter_shape("b_k", vec![4]);
    graph.bind_parameter_shape("b_v", vec![4]);
    graph.bind_parameter_shape("b_o", vec![4]);
    graph.bind_parameter_shape("ln1_w", vec![4]);
    graph.bind_parameter_shape("ln1_b", vec![4]);
    graph.bind_parameter_shape("ffn_w1", vec![4, 6]);
    graph.bind_parameter_shape("ffn_b1", vec![6]);
    graph.bind_parameter_shape("ffn_w2", vec![6, 4]);
    graph.bind_parameter_shape("ffn_b2", vec![4]);
    graph.bind_parameter_shape("ln2_w", vec![4]);
    graph.bind_parameter_shape("ln2_b", vec![4]);

    crate::ir::verify_graph(&graph).expect("tiny-transformer fixture graph must verify");
    let inference_plan =
        build_execution_plan(&graph, &HashSet::new()).expect("fixture infer plan must build");
    let inference_ordered_nodes =
        dependency_ordered_nodes(&graph, out, &inference_plan.schedule.ordered_nodes)
            .expect("fixture infer dependency order must resolve");

    let parameters = tiny_transformer_initial_parameters();
    let model = CompiledModel {
        graph,
        output: out,
        output_shape: TensorShape(vec![2, 4]),
        loss: Some(loss),
        parameters,
        parameter_values,
        inference_plan,
        inference_ordered_nodes,
    };

    let dataset = TinyTransformerFixtureDataset::new();
    let train_config = TrainApiConfig {
        epochs: 2,
        batch_size: 1,
        shuffle: false,
        shuffle_seed: 11,
        optimizer: OptimizerConfig::Adam {
            lr: 0.005,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        gradient_checkpointing: None,
        reproducibility: ReproducibilityMode::Deterministic,
        checkpoint_path: None,
    };

    let mut infer_input = HashMap::new();
    infer_input.insert(
        "x".to_string(),
        Tensor::new(
            vec![1, 2, 4],
            vec![0.2, -0.1, 0.3, 0.4, 0.7, 0.5, -0.4, 0.1],
        )
        .expect("fixture infer input must be valid"),
    );

    (model, dataset, train_config, infer_input)
}

fn tiny_transformer_initial_parameters() -> HashMap<String, Tensor> {
    HashMap::from([
        (
            "w_q".to_string(),
            Tensor::new(
                vec![4, 4],
                vec![
                    0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 0.1, -0.2, 0.7, 0.2, -0.3, 0.5, 0.4, -0.6, 0.8,
                    0.1,
                ],
            )
            .expect("valid tensor"),
        ),
        (
            "w_k".to_string(),
            Tensor::new(
                vec![4, 4],
                vec![
                    0.1, 0.2, -0.4, 0.3, 0.5, -0.7, 0.6, 0.2, -0.3, 0.8, 0.4, -0.1, 0.2, 0.1, 0.5,
                    -0.6,
                ],
            )
            .expect("valid tensor"),
        ),
        (
            "w_v".to_string(),
            Tensor::new(
                vec![4, 4],
                vec![
                    0.3, -0.2, 0.1, 0.7, 0.6, 0.4, -0.5, 0.2, 0.2, -0.8, 0.9, 0.1, -0.4, 0.3, 0.2,
                    0.5,
                ],
            )
            .expect("valid tensor"),
        ),
        (
            "w_o".to_string(),
            Tensor::new(
                vec![4, 4],
                vec![
                    0.4, -0.3, 0.2, 0.1, 0.5, 0.6, -0.7, 0.2, -0.1, 0.8, 0.3, -0.4, 0.7, -0.2, 0.5,
                    0.6,
                ],
            )
            .expect("valid tensor"),
        ),
        (
            "b_q".to_string(),
            Tensor::new(vec![4], vec![0.05, -0.1, 0.15, -0.2]).expect("valid tensor"),
        ),
        (
            "b_k".to_string(),
            Tensor::new(vec![4], vec![-0.05, 0.2, -0.15, 0.1]).expect("valid tensor"),
        ),
        (
            "b_v".to_string(),
            Tensor::new(vec![4], vec![0.1, 0.05, -0.2, 0.25]).expect("valid tensor"),
        ),
        (
            "b_o".to_string(),
            Tensor::new(vec![4], vec![-0.1, 0.15, 0.05, -0.05]).expect("valid tensor"),
        ),
        (
            "ln1_w".to_string(),
            Tensor::new(vec![4], vec![1.0, 0.9, 1.1, -0.8]).expect("valid tensor"),
        ),
        (
            "ln1_b".to_string(),
            Tensor::new(vec![4], vec![0.05, -0.1, 0.15, 0.2]).expect("valid tensor"),
        ),
        (
            "ffn_w1".to_string(),
            Tensor::new(
                vec![4, 6],
                vec![
                    0.2, -0.3, 0.1, 0.5, 0.4, -0.2, 0.6, 0.7, -0.5, 0.2, -0.1, 0.3, -0.4, 0.8, 0.9,
                    -0.6, 0.2, 0.1, 0.3, -0.7, 0.4, 0.5, -0.8, 0.6,
                ],
            )
            .expect("valid tensor"),
        ),
        (
            "ffn_b1".to_string(),
            Tensor::new(vec![6], vec![0.1, -0.2, 0.05, 0.15, -0.1, 0.2]).expect("valid tensor"),
        ),
        (
            "ffn_w2".to_string(),
            Tensor::new(
                vec![6, 4],
                vec![
                    0.3, -0.4, 0.2, 0.1, 0.5, 0.6, -0.7, 0.2, -0.1, 0.8, 0.3, -0.4, 0.7, -0.2, 0.5,
                    0.6, 0.4, 0.1, -0.3, 0.2, -0.5, 0.9, 0.6, -0.7,
                ],
            )
            .expect("valid tensor"),
        ),
        (
            "ffn_b2".to_string(),
            Tensor::new(vec![4], vec![0.2, -0.15, 0.05, 0.1]).expect("valid tensor"),
        ),
        (
            "ln2_w".to_string(),
            Tensor::new(vec![4], vec![0.95, -1.05, 0.85, 1.1]).expect("valid tensor"),
        ),
        (
            "ln2_b".to_string(),
            Tensor::new(vec![4], vec![-0.05, 0.1, -0.15, 0.2]).expect("valid tensor"),
        ),
    ])
}

fn dependency_ordered_nodes(
    graph: &Graph,
    target: ValueId,
    ordered_nodes: &[NodeId],
) -> Result<Vec<NodeId>, String> {
    if target.0 >= graph.nodes.len() {
        return Err(format!("target out of range: {}", target.0));
    }

    let mut required_values = HashSet::<ValueId>::new();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_transformer_fixture_builds_verified_graph() {
        let (model, dataset, _cfg, infer_input) = build_tiny_transformer_fixture_for_tests();

        assert_eq!(
            dataset.len(),
            2,
            "fixture dataset size changed unexpectedly"
        );
        assert_eq!(model.output_shape.0, vec![2, 4]);
        assert!(
            model.loss.is_some(),
            "fixture model must include loss value"
        );

        let mha_count = model
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(node.op, Op::MultiHeadAttention { .. }))
            .count();
        let ln_count = model
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(node.op, Op::LayerNorm { .. }))
            .count();
        let gemm_count = model
            .graph
            .nodes
            .iter()
            .filter(|node| matches!(node.op, Op::Gemm { .. }))
            .count();

        assert_eq!(mha_count, 6, "fixture must emit full MHA output family");
        assert_eq!(ln_count, 2, "fixture must include two LayerNorm ops");
        assert_eq!(gemm_count, 2, "fixture must include FFN Gemm ops");
        assert_eq!(
            infer_input
                .get("x")
                .expect("fixture infer input x exists")
                .shape,
            vec![1, 2, 4]
        );
    }
}
