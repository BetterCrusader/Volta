use std::collections::HashMap;

use crate::ir::{Op, OptimizerConfig, Tensor};

use super::train_api::{ReproducibilityMode, TrainApiConfig, TrainApiError};
use super::{CompiledModel, Dataset, Example, MSELoss, ModelBuilder, Parameter, TensorShape};

#[derive(Debug, Clone)]
struct FixtureSample {
    x: [f32; 4],
    target: [f32; 1],
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
                    x: [1.0, 0.0, 0.0, 0.0],
                    target: [1.0],
                },
                FixtureSample {
                    x: [0.0, 1.0, 0.0, 0.0],
                    target: [0.0],
                },
                FixtureSample {
                    x: [1.0, 1.0, 0.0, 0.0],
                    target: [1.0],
                },
                FixtureSample {
                    x: [0.5, 0.5, 0.0, 0.0],
                    target: [0.5],
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
            Tensor::new(vec![1, 4], row.x.to_vec()).map_err(|err| TrainApiError {
                message: format!("fixture tensor x is invalid: {}", err.message),
            })?,
        );
        inputs.insert(
            "target".to_string(),
            Tensor::new(vec![1, 1], row.target.to_vec()).map_err(|err| TrainApiError {
                message: format!("fixture tensor target is invalid: {}", err.message),
            })?,
        );
        Ok(Example { inputs })
    }
}

pub fn build_tiny_transformer_fixture_for_tests() -> (
    CompiledModel,
    TinyTransformerFixtureDataset,
    TrainApiConfig,
    HashMap<String, Tensor>,
) {
    let mut builder = ModelBuilder::new();
    let x = builder
        .input_with_shape("x", vec![1, 4])
        .expect("fixture input x must build");
    let target = builder
        .input_with_shape("target", vec![1, 1])
        .expect("fixture input target must build");

    let w_q = builder
        .add_parameter(Parameter::new(
            "tiny.w_q",
            Tensor::new(
                vec![4, 4],
                vec![
                    1.0, 0.0, 0.0, 0.0, // row 0
                    0.0, 1.0, 0.0, 0.0, // row 1
                    0.0, 0.0, 0.5, 0.0, // row 2
                    0.0, 0.0, 0.0, 0.5, // row 3
                ],
            )
            .expect("fixture w_q tensor must be valid"),
            true,
        ))
        .expect("fixture parameter tiny.w_q must build");

    let b_q = builder
        .add_parameter(Parameter::new(
            "tiny.b_q",
            Tensor::new(vec![1, 4], vec![0.0, 0.0, 0.0, 0.0])
                .expect("fixture b_q tensor must be valid"),
            true,
        ))
        .expect("fixture parameter tiny.b_q must build");

    let w_o = builder
        .add_parameter(Parameter::new(
            "tiny.w_o",
            Tensor::new(
                vec![4, 1],
                vec![
                    0.1, // row 0
                    0.0, // row 1
                    0.0, // row 2
                    0.0, // row 3
                ],
            )
            .expect("fixture w_o tensor must be valid"),
            true,
        ))
        .expect("fixture parameter tiny.w_o must build");

    let b_o = builder
        .add_parameter(Parameter::new(
            "tiny.b_o",
            Tensor::new(vec![1, 1], vec![0.0]).expect("fixture b_o tensor must be valid"),
            true,
        ))
        .expect("fixture parameter tiny.b_o must build");

    let q_linear = builder
        .add_op(Op::MatMul(x, w_q))
        .expect("fixture matmul q must build");
    let q_biased = builder
        .add_op(Op::Add(q_linear, b_q))
        .expect("fixture add q+b must build");
    let hidden = builder
        .add_op(Op::Relu(q_biased))
        .expect("fixture relu must build");
    let out_linear = builder
        .add_op(Op::MatMul(hidden, w_o))
        .expect("fixture matmul out must build");
    let logits = builder
        .add_op(Op::Add(out_linear, b_o))
        .expect("fixture add out+b must build");

    let output_shape = TensorShape(vec![1, 1]);
    let loss = MSELoss
        .build(&mut builder, logits, &output_shape, target, &output_shape)
        .expect("fixture loss must build");

    let model = builder
        .finalize(logits, output_shape, Some(loss))
        .expect("fixture model must finalize");

    let dataset = TinyTransformerFixtureDataset::new();
    let train_config = TrainApiConfig {
        epochs: 30,
        batch_size: 2,
        shuffle: true,
        shuffle_seed: 11,
        optimizer: OptimizerConfig::Sgd { lr: 0.05 },
        reproducibility: ReproducibilityMode::Deterministic,
        checkpoint_path: None,
    };

    let mut infer_input = HashMap::new();
    infer_input.insert(
        "x".to_string(),
        Tensor::new(vec![1, 4], vec![1.0, 0.5, 0.0, 0.0]).expect("fixture infer input is valid"),
    );

    (model, dataset, train_config, infer_input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_transformer_fixture_builds_verified_graph() {
        let (model, dataset, _cfg, infer_input) = build_tiny_transformer_fixture_for_tests();

        assert_eq!(
            dataset.len(),
            4,
            "fixture dataset size changed unexpectedly"
        );
        assert_eq!(model.output_shape.0, vec![1, 1]);
        assert!(
            model.loss.is_some(),
            "fixture model must include loss value"
        );

        let mut matmul_count = 0usize;
        let mut add_count = 0usize;
        let mut relu_count = 0usize;
        for node in &model.graph.nodes {
            match node.op {
                Op::MatMul(_, _) => matmul_count += 1,
                Op::Add(_, _) => add_count += 1,
                Op::Relu(_) => relu_count += 1,
                _ => {}
            }
        }

        assert!(matmul_count >= 2, "fixture must include matmul path");
        assert!(add_count >= 2, "fixture must include add path");
        assert!(relu_count >= 1, "fixture must include relu path");
        assert_eq!(
            infer_input
                .get("x")
                .expect("fixture infer input x exists")
                .shape,
            vec![1, 4]
        );

        crate::ir::verify_graph(&model.graph).expect("graph must verify");
    }
}
