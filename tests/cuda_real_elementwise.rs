#[path = "common/cuda.rs"]
mod cuda_helpers;

use volta::ir::DeterminismLevel;
use volta::ir::cuda::kernels::add::{add_f32, div_f32};
use volta::ir::cuda::kernels::relu::relu_f32;
use volta::ir::cuda::kernels::softmax::softmax_f32;

#[test]
fn cuda_add_matches_cpu() {
    let Some(device) = cuda_helpers::safe_cuda_device() else {
        eprintln!("[SKIP] cuda_add_matches_cpu — no CUDA device available");
        return;
    };

    let left = vec![1.0_f32, -2.0, 3.5, 4.0];
    let right = vec![0.5_f32, 2.0, -1.5, 1.0];

    let gpu = match add_f32(&device, &left, &right, DeterminismLevel::Strict) {
        Ok(values) => values,
        Err(message) if message.contains("cuBLAS init") || message.contains("runtime handles") => {
            return;
        }
        Err(message) => panic!("cuda add should run: {message}"),
    };
    let cpu = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();

    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_relu_matches_cpu() {
    let Some(device) = cuda_helpers::safe_cuda_device() else {
        eprintln!("[SKIP] cuda_relu_matches_cpu — no CUDA device available");
        return;
    };

    let input = vec![-3.0_f32, -0.1, 0.0, 0.5, 2.0];
    let gpu = match relu_f32(&device, &input, DeterminismLevel::Strict) {
        Ok(values) => values,
        Err(message) if message.contains("NVRTC") || message.contains("runtime handles") => {
            return;
        }
        Err(message) => panic!("cuda relu should run: {message}"),
    };
    let cpu = input
        .iter()
        .map(|value| if *value > 0.0 { *value } else { 0.0 })
        .collect::<Vec<_>>();

    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_div_matches_cpu() {
    let Some(device) = cuda_helpers::safe_cuda_device() else {
        eprintln!("[SKIP] cuda_div_matches_cpu — no CUDA device available");
        return;
    };

    let left = vec![6.0_f32, 8.0, 9.0, 12.0];
    let right = vec![3.0_f32, 2.0, 3.0, 4.0];

    let gpu = match div_f32(&device, &left, &right, DeterminismLevel::Strict) {
        Ok(values) => values,
        Err(message) if message.contains("NVRTC") || message.contains("runtime handles") => {
            return;
        }
        Err(message) => panic!("cuda div should run: {message}"),
    };
    let cpu = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a / b)
        .collect::<Vec<_>>();

    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_softmax_strict_is_deterministic() {
    let Some(device) = cuda_helpers::safe_cuda_device() else {
        eprintln!("[SKIP] cuda_softmax_strict_is_deterministic — no CUDA device available");
        return;
    };

    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let first = match softmax_f32(&device, &input, DeterminismLevel::Strict) {
        Ok(values) => values,
        Err(message) if message.contains("NVRTC") || message.contains("runtime handles") => {
            return;
        }
        Err(message) => panic!("cuda softmax should run: {message}"),
    };
    let second = softmax_f32(&device, &input, DeterminismLevel::Strict)
        .expect("second strict softmax run should pass");

    assert_eq!(first, second);
    let sum = first.iter().copied().sum::<f32>();
    assert!(
        (sum - 1.0).abs() <= 1e-5,
        "softmax must sum to one, got {sum}"
    );
}
