use volta::ir::{Backend, BackendKind, CpuBackend, CudaBackend, DeterminismLevel};

#[test]
fn cpu_backend_reports_strict_deterministic_capabilities() {
    let backend = CpuBackend;
    let caps = backend.capabilities();

    assert_eq!(caps.backend, BackendKind::Cpu);
    assert!(caps.supports_inference);
    assert!(caps.supports_training);
    assert!(caps.supports_strict_determinism);
    assert_eq!(caps.default_determinism, DeterminismLevel::Strict);
}

#[test]
fn cuda_backend_reports_inference_only_placeholder_capabilities() {
    let backend = CudaBackend;
    let caps = backend.capabilities();

    assert_eq!(caps.backend, BackendKind::Cuda);
    assert!(caps.supports_inference);
    assert!(!caps.supports_training);
    assert!(caps.supports_strict_determinism);
    assert_eq!(caps.default_determinism, DeterminismLevel::Balanced);
}
