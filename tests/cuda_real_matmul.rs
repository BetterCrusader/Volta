use volta::ir::DeterminismLevel;
use volta::ir::cuda::device::CudaDevice;
use volta::ir::cuda::kernels::matmul::matmul_f32;

#[test]
fn cuda_device_reports_runtime_properties_when_available() {
    let Some(device) = safe_cuda_device() else {
        return;
    };

    assert!(device.total_memory_bytes > 0);
    assert!(
        device.compute_capability_major > 0 || device.compute_capability_minor > 0,
        "real CUDA device must expose compute capability"
    );
    assert!(device.has_runtime_handles());
}

#[test]
fn cuda_matmul_matches_cpu() {
    let Some(device) = safe_cuda_device() else {
        return;
    };

    let m = 2usize;
    let k = 3usize;
    let n = 2usize;
    let lhs = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let gpu = match matmul_f32(&device, &lhs, &rhs, m, n, k, DeterminismLevel::Strict) {
        Ok(out) => out,
        Err(message) if message.contains("cuBLAS init") || message.contains("runtime handles") => {
            return;
        }
        Err(message) => panic!("cuda matmul should run: {message}"),
    };
    let cpu = cpu_matmul(&lhs, &rhs, m, n, k);

    assert_eq!(gpu.len(), cpu.len());
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        assert!((*g - *c).abs() <= 1e-5, "gpu={} cpu={}", g, c);
    }
}

fn cpu_matmul(lhs: &[f32], rhs: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for inner in 0..k {
                acc += lhs[row * k + inner] * rhs[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn safe_cuda_device() -> Option<CudaDevice> {
    let result = std::panic::catch_unwind(|| CudaDevice::new(0));
    match result {
        Ok(Ok(device)) => Some(device),
        Ok(Err(_)) => None,
        Err(_) => None,
    }
}
