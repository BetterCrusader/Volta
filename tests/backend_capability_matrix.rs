use volta::ir::{Backend, BackendKind, CpuBackend, CudaBackend, LlvmBackend};

fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"))
}

#[test]
fn governance_docs_define_backend_matrix_and_perf_slos() {
    let matrix = read_text("docs/governance/backend-capability-matrix.md");
    assert!(matrix.contains("## Backend Capability Matrix"));
    assert!(matrix.contains("CPU"));
    assert!(matrix.contains("CUDA"));
    assert!(matrix.contains("LLVM"));
    assert!(matrix.contains("strict determinism"));

    let perf_slo = read_text("docs/governance/perf-slo.md");
    assert!(perf_slo.contains("## Runtime SLOs"));
    assert!(perf_slo.contains("P95"));
    assert!(perf_slo.contains("plan cache"));
    assert!(perf_slo.contains("memory budget"));
}

#[test]
fn backend_capabilities_match_matrix_contract() {
    let cpu = CpuBackend.capabilities();
    let cuda = CudaBackend.capabilities();
    let llvm = LlvmBackend.capabilities();

    assert_eq!(cpu.backend, BackendKind::Cpu);
    assert!(cpu.supports_inference);
    assert!(cpu.supports_training);
    assert!(cpu.supports_strict_determinism);

    assert_eq!(cuda.backend, BackendKind::Cuda);
    assert!(cuda.supports_inference);
    assert!(cuda.supports_training);
    assert!(cuda.supports_strict_determinism);

    assert_eq!(llvm.backend, BackendKind::Llvm);
    assert!(!llvm.supports_inference);
    assert!(!llvm.supports_training);
    assert!(!llvm.supports_strict_determinism);
}
