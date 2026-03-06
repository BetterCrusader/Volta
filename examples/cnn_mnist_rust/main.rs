// CNN-style MNIST classifier using the Volta Rust API.
//
// Architecture (dense classification head on MNIST-sized inputs):
//   Linear(784->256) -> ReLU -> Linear(256->64) -> ReLU -> Linear(64->10)
//
// Demonstrates low-level ModelBuilder API for multi-layer classifiers.

use std::collections::HashMap;
use volta::ir::{Op, Tensor};
use volta::model::{CompiledModel, ModelBuilder, Parameter, TensorShape, infer};

fn build_cnn() -> Result<CompiledModel, String> {
    let mut builder = ModelBuilder::new();

    let input = builder
        .input_with_shape("image", vec![1, 784])
        .map_err(|e| e.message)?;

    // FC1: 784 -> 256 + ReLU
    let w1 = Tensor::zeros(vec![784, 256]).map_err(|e| e.message)?;
    let b1 = Tensor::zeros(vec![1, 256]).map_err(|e| e.message)?;
    let w1_id = builder
        .add_parameter(Parameter::new("fc1.weight", w1, true))
        .map_err(|e| e.message)?;
    let b1_id = builder
        .add_parameter(Parameter::new("fc1.bias", b1, true))
        .map_err(|e| e.message)?;
    let h1 = builder
        .add_op(Op::MatMul(input, w1_id))
        .map_err(|e| e.message)?;
    let h1b = builder.add_op(Op::Add(h1, b1_id)).map_err(|e| e.message)?;
    let h1r = builder.add_op(Op::Relu(h1b)).map_err(|e| e.message)?;

    // FC2: 256 -> 64 + ReLU
    let w2 = Tensor::zeros(vec![256, 64]).map_err(|e| e.message)?;
    let b2 = Tensor::zeros(vec![1, 64]).map_err(|e| e.message)?;
    let w2_id = builder
        .add_parameter(Parameter::new("fc2.weight", w2, true))
        .map_err(|e| e.message)?;
    let b2_id = builder
        .add_parameter(Parameter::new("fc2.bias", b2, true))
        .map_err(|e| e.message)?;
    let h2 = builder
        .add_op(Op::MatMul(h1r, w2_id))
        .map_err(|e| e.message)?;
    let h2b = builder.add_op(Op::Add(h2, b2_id)).map_err(|e| e.message)?;
    let h2r = builder.add_op(Op::Relu(h2b)).map_err(|e| e.message)?;

    // FC3: 64 -> 10 (logits)
    let w3 = Tensor::zeros(vec![64, 10]).map_err(|e| e.message)?;
    let b3 = Tensor::zeros(vec![1, 10]).map_err(|e| e.message)?;
    let w3_id = builder
        .add_parameter(Parameter::new("fc3.weight", w3, true))
        .map_err(|e| e.message)?;
    let b3_id = builder
        .add_parameter(Parameter::new("fc3.bias", b3, true))
        .map_err(|e| e.message)?;
    let h3 = builder
        .add_op(Op::MatMul(h2r, w3_id))
        .map_err(|e| e.message)?;
    let logits = builder.add_op(Op::Add(h3, b3_id)).map_err(|e| e.message)?;

    let target = builder
        .input_with_shape("target", vec![1, 10])
        .map_err(|e| e.message)?;
    let loss = builder
        .add_op(Op::Sub(logits, target))
        .map_err(|e| e.message)?;

    builder
        .finalize(logits, TensorShape(vec![1, 10]), Some(loss))
        .map_err(|e| e.message)
}

fn main() {
    println!("Building CNN-style MNIST classifier (784->256->64->10)...");

    let model = match build_cnn() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Build error: {e}");
            std::process::exit(1);
        }
    };

    let total: usize = model
        .parameters
        .values()
        .map(|t| t.shape.iter().product::<usize>())
        .sum();
    println!("Total parameters: {total}");

    let mut inputs = HashMap::new();
    inputs.insert(
        "image".to_string(),
        Tensor::zeros(vec![1, 784]).expect("input"),
    );

    match infer(&model, &model.parameters.clone(), &inputs) {
        Ok(out) => println!("Forward pass OK. Output shape: {:?}", out.shape),
        Err(e) => eprintln!("Inference error: {}", e.message),
    }
}
