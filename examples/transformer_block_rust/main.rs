// Transformer FFN block using the Volta Rust API.
//
// Architecture (standard GPT-2 FFN sublayer):
//   Linear(model_dim -> 4*model_dim) -> GELU -> Linear(4*model_dim -> model_dim)
//
// model_dim=256, hidden_dim=1024

use std::collections::HashMap;
use volta::ir::{Op, Tensor};
use volta::model::{CompiledModel, ModelBuilder, Parameter, TensorShape, infer};

const MODEL_DIM: usize = 256;
const HIDDEN_DIM: usize = 1024;

fn build_ffn() -> Result<CompiledModel, String> {
    let mut builder = ModelBuilder::new();

    let input = builder
        .input_with_shape("x", vec![1, MODEL_DIM])
        .map_err(|e| e.message)?;

    // FC1: model_dim -> hidden_dim + GELU
    let w1 = Tensor::zeros(vec![MODEL_DIM, HIDDEN_DIM]).map_err(|e| e.message)?;
    let b1 = Tensor::zeros(vec![1, HIDDEN_DIM]).map_err(|e| e.message)?;
    let w1_id = builder
        .add_parameter(Parameter::new("ffn.w1", w1, true))
        .map_err(|e| e.message)?;
    let b1_id = builder
        .add_parameter(Parameter::new("ffn.b1", b1, true))
        .map_err(|e| e.message)?;
    let h1 = builder
        .add_op(Op::MatMul(input, w1_id))
        .map_err(|e| e.message)?;
    let h1b = builder.add_op(Op::Add(h1, b1_id)).map_err(|e| e.message)?;
    let h1_gelu = builder.add_op(Op::Gelu(h1b)).map_err(|e| e.message)?;

    // FC2: hidden_dim -> model_dim
    let w2 = Tensor::zeros(vec![HIDDEN_DIM, MODEL_DIM]).map_err(|e| e.message)?;
    let b2 = Tensor::zeros(vec![1, MODEL_DIM]).map_err(|e| e.message)?;
    let w2_id = builder
        .add_parameter(Parameter::new("ffn.w2", w2, true))
        .map_err(|e| e.message)?;
    let b2_id = builder
        .add_parameter(Parameter::new("ffn.b2", b2, true))
        .map_err(|e| e.message)?;
    let h2 = builder
        .add_op(Op::MatMul(h1_gelu, w2_id))
        .map_err(|e| e.message)?;
    let output = builder.add_op(Op::Add(h2, b2_id)).map_err(|e| e.message)?;

    let target = builder
        .input_with_shape("target", vec![1, MODEL_DIM])
        .map_err(|e| e.message)?;
    let loss = builder
        .add_op(Op::Sub(output, target))
        .map_err(|e| e.message)?;

    builder
        .finalize(output, TensorShape(vec![1, MODEL_DIM]), Some(loss))
        .map_err(|e| e.message)
}

fn main() {
    println!("Building Transformer FFN block (model_dim={MODEL_DIM}, hidden_dim={HIDDEN_DIM})...");

    let model = match build_ffn() {
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
        "x".to_string(),
        Tensor::zeros(vec![1, MODEL_DIM]).expect("input"),
    );

    match infer(&model, &model.parameters.clone(), &inputs) {
        Ok(out) => println!("Forward pass OK. Output shape: {:?}", out.shape),
        Err(e) => eprintln!("Inference error: {}", e.message),
    }
}
