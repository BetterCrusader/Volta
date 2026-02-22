#[path = "common/cuda.rs"]
mod cuda_helpers;

use std::collections::HashMap;

use volta::ir::{CpuBackend, CudaBackend, Tensor};
use volta::model::{CompiledModel, ModelBuilder, Parameter, TensorShape, infer_with_backend};

#[test]
fn cuda_infer_matches_cpu_for_seeded_fuzz_cases() {
    cuda_helpers::with_determinism("strict", || {
        if !cuda_helpers::cuda_runtime_available() {
            eprintln!("[SKIP] cuda_infer_matches_cpu_for_seeded_fuzz_cases — no CUDA device available");
            return;
        }

        let (model, mut infer_input) = build_fixture_model();
        let cpu = CpuBackend;
        let cuda = CudaBackend;

        let mut seed = 0xC0DEC0DE_u64;
        for case_idx in 0..32 {
            let x = vec![
                next_in_range(&mut seed, -3.0, 3.0),
                next_in_range(&mut seed, -3.0, 3.0),
            ];
            let logits = vec![
                next_in_range(&mut seed, -4.0, 4.0),
                next_in_range(&mut seed, -4.0, 4.0),
            ];

            infer_input.insert(
                "x".to_string(),
                Tensor::new(vec![1, 2], x).expect("seeded x tensor should be valid"),
            );
            infer_input.insert(
                "logits".to_string(),
                Tensor::new(vec![2], logits).expect("seeded logits tensor should be valid"),
            );

            let cpu_out = infer_with_backend(&model, &model.parameters, &infer_input, &cpu)
                .expect("cpu infer should succeed");
            let cuda_out = match infer_with_backend(&model, &model.parameters, &infer_input, &cuda)
            {
                Ok(out) => out,
                Err(err) if cuda_helpers::is_cuda_unavailable(&err.message) => return,
                Err(err) => panic!("cuda infer failed for case {case_idx}: {}", err.message),
            };

            assert_eq!(
                cpu_out.shape, cuda_out.shape,
                "shape mismatch for case {case_idx}"
            );
            assert_eq!(
                cpu_out.data.len(),
                cuda_out.data.len(),
                "tensor len mismatch for case {case_idx}"
            );

            let abs = max_abs_diff(&cpu_out.data, &cuda_out.data);
            let rel = max_rel_diff(&cpu_out.data, &cuda_out.data);
            assert!(
                abs <= 1.0e-5,
                "abs diff too large for case {case_idx}: abs={abs}"
            );
            assert!(
                rel <= 1.0e-5,
                "rel diff too large for case {case_idx}: rel={rel}"
            );
        }
    });
}

fn build_fixture_model() -> (CompiledModel, HashMap<String, Tensor>) {
    let mut builder = ModelBuilder::new();
    let x = builder
        .input_with_shape("x", vec![1, 2])
        .expect("x input should be added");
    let logits = builder
        .input_with_shape("logits", vec![2])
        .expect("logits input should be added");

    let w = builder
        .add_parameter(Parameter::new(
            "fuzz.w",
            Tensor::new(vec![2, 2], vec![0.5, -1.0, 1.5, 2.0]).expect("valid weight tensor"),
            true,
        ))
        .expect("w parameter should be added");
    let b = builder
        .add_parameter(Parameter::new(
            "fuzz.b",
            Tensor::new(vec![1, 2], vec![0.25, -0.75]).expect("valid bias tensor"),
            true,
        ))
        .expect("b parameter should be added");

    let mm = builder
        .add_op(volta::ir::Op::MatMul(x, w))
        .expect("matmul op");
    let sum = builder.add_op(volta::ir::Op::Add(mm, b)).expect("add op");
    let out = builder.add_op(volta::ir::Op::Relu(sum)).expect("relu op");
    builder
        .add_op(volta::ir::Op::Softmax(logits))
        .expect("softmax op");

    let model = builder
        .finalize(out, TensorShape(vec![1, 2]), None)
        .expect("fixture model should finalize");

    let mut infer_input = HashMap::new();
    infer_input.insert(
        "x".to_string(),
        Tensor::new(vec![1, 2], vec![1.0, -2.0]).expect("valid x seed"),
    );
    infer_input.insert(
        "logits".to_string(),
        Tensor::new(vec![2], vec![0.1, 0.9]).expect("valid logits seed"),
    );
    (model, infer_input)
}

fn next_in_range(seed: &mut u64, min: f32, max: f32) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let unit = ((*seed >> 32) as u32) as f32 / (u32::MAX as f32);
    min + (max - min) * unit
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn max_rel_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let denom = a.abs().max(b.abs()).max(1.0e-12);
            (a - b).abs() / denom
        })
        .fold(0.0_f32, f32::max)
}

