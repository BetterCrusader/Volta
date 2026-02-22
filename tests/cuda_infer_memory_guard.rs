#[path = "common/cuda.rs"]
mod cuda_helpers;

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use volta::ir::cuda::{CudaMemoryClass, lower_plan, profile_memory};
use volta::ir::{Backend, CpuBackend, CudaBackend, Op, Tensor, build_execution_plan};
use volta::model::{CompiledModel, ModelBuilder, Parameter, TensorShape};

const MEMORY_BASELINE_PATH: &str = "benchmarks/baselines/cuda-infer-memory-peak-bytes.txt";

#[derive(Debug, Clone)]
struct MemoryBaseline {
    backend_signature: String,
    determinism_level: String,
    device_capability_fingerprint: String,
    placement_fingerprint: u64,
    peak_device_bytes: usize,
    peak_host_transfer_bytes: usize,
    peak_temp_bytes: usize,
    alignment_consistent: bool,
    tolerance_ratio: f32,
}

#[test]
fn cuda_memory_guard_enforces_baseline_and_contracts() {
    if !cuda_helpers::cuda_runtime_available() {
        eprintln!("[SKIP] cuda_memory_guard_enforces_baseline_and_contracts — no CUDA device available");
        return;
    }
    let strict_runs = (0..5)
        .map(|_| profile_for_level("strict"))
        .collect::<Vec<_>>();
    for run in &strict_runs[1..] {
        assert_eq!(
            *run, strict_runs[0],
            "strict memory profile must be stable across runs"
        );
    }

    let strict_profile = strict_runs[0];
    let balanced_profile = profile_for_level("balanced");
    assert_eq!(
        strict_profile.peak_device_bytes, balanced_profile.peak_device_bytes,
        "strict mode must not change peak_device_bytes"
    );
    assert_eq!(
        strict_profile.peak_host_transfer_bytes, balanced_profile.peak_host_transfer_bytes,
        "strict mode must not change host transfer bytes"
    );
    assert_eq!(
        strict_profile.peak_temp_bytes, balanced_profile.peak_temp_bytes,
        "strict mode must not change temp bytes"
    );

    let baseline = load_memory_baseline(MEMORY_BASELINE_PATH).unwrap_or_else(|message| {
        panic!(
            "{message}; suggested baseline: backend_signature={}, determinism_level=strict, device_capability_fingerprint={}, placement_fingerprint={}, peak_device_bytes={}, peak_host_transfer_bytes={}, peak_temp_bytes={}, alignment_consistent={}, tolerance_ratio=0.10",
            runtime_backend_signature(),
            device_capability_fingerprint(),
            strict_profile.placement_fingerprint,
            strict_profile.peak_device_bytes,
            strict_profile.peak_host_transfer_bytes,
            strict_profile.peak_temp_bytes,
            strict_profile.alignment_consistent,
        );
    });

    assert_eq!(baseline.backend_signature, runtime_backend_signature());
    assert_eq!(baseline.determinism_level, "strict");
    let runtime_fingerprint = device_capability_fingerprint();
    assert!(
        device_fingerprint_matches(
            &baseline.device_capability_fingerprint,
            &runtime_fingerprint
        ),
        "device_capability_fingerprint mismatch: baseline='{}' runtime='{}'",
        baseline.device_capability_fingerprint,
        runtime_fingerprint,
    );
    assert_eq!(
        strict_profile.placement_fingerprint, baseline.placement_fingerprint,
        "placement hints changed; update baseline only after review"
    );
    assert_eq!(
        strict_profile.alignment_consistent, baseline.alignment_consistent,
        "alignment consistency mismatch"
    );

    assert_with_tolerance(
        "peak_device_bytes",
        strict_profile.peak_device_bytes,
        baseline.peak_device_bytes,
        baseline.tolerance_ratio,
    );
    assert_with_tolerance(
        "peak_host_transfer_bytes",
        strict_profile.peak_host_transfer_bytes,
        baseline.peak_host_transfer_bytes,
        baseline.tolerance_ratio,
    );
    assert_with_tolerance(
        "peak_temp_bytes",
        strict_profile.peak_temp_bytes,
        baseline.peak_temp_bytes,
        baseline.tolerance_ratio,
    );
}

#[test]
fn cuda_memory_guard_rejects_placement_mapping_drift() {
    if !cuda_helpers::cuda_runtime_available() {
        eprintln!("[SKIP] cuda_memory_guard_rejects_placement_mapping_drift — no CUDA device available");
        return;
    }
    let model = build_memory_fixture_model();
    let plan = build_execution_plan(&model.graph, &std::collections::HashSet::new())
        .expect("plan should build");
    let mut lowered = lower_plan(&plan).expect("cuda lowering should pass");
    lowered.memory_bindings[0].class = CudaMemoryClass::Output;

    let err = profile_memory(&model.graph, &plan, &lowered)
        .expect_err("memory profiler must reject placement mapping drift");
    assert!(
        err.message.contains("placement mapping mismatch"),
        "unexpected error: {}",
        err.message
    );
}

fn profile_for_level(level: &str) -> volta::ir::cuda::CudaMemoryProfile {
    with_determinism(level, || {
        let model = build_memory_fixture_model();
        let plan = build_execution_plan(&model.graph, &std::collections::HashSet::new())
            .expect("plan should build");
        let lowered = lower_plan(&plan).expect("cuda lowering should pass");
        profile_memory(&model.graph, &plan, &lowered).expect("cuda memory profile should build")
    })
}

fn build_memory_fixture_model() -> CompiledModel {
    let mut builder = ModelBuilder::new();
    let x = builder
        .input_with_shape("x", vec![1, 2])
        .expect("add input should succeed");
    let logits = builder
        .input_with_shape("logits", vec![2])
        .expect("add logits should succeed");

    let w = builder
        .add_parameter(Parameter::new(
            "guard.w",
            Tensor::new(vec![2, 2], vec![0.5, -1.0, 1.5, 2.0]).expect("valid tensor"),
            true,
        ))
        .expect("add parameter should succeed");
    let b = builder
        .add_parameter(Parameter::new(
            "guard.b",
            Tensor::new(vec![1, 2], vec![0.25, -0.75]).expect("valid tensor"),
            true,
        ))
        .expect("add parameter should succeed");

    let mm = builder
        .add_op(Op::MatMul(x, w))
        .expect("add matmul should succeed");
    let sum = builder
        .add_op(Op::Add(mm, b))
        .expect("add add should succeed");
    let out = builder
        .add_op(Op::Relu(sum))
        .expect("add relu should succeed");
    builder
        .add_op(Op::Softmax(logits))
        .expect("add softmax should succeed");

    builder
        .finalize(out, TensorShape(vec![1, 2]), None)
        .expect("fixture model should finalize")
}

fn assert_with_tolerance(metric: &str, current: usize, baseline: usize, tolerance_ratio: f32) {
    let tolerance = ((baseline as f64) * (tolerance_ratio as f64)).ceil() as usize;
    let allowed = baseline.saturating_add(tolerance);
    assert!(
        current <= allowed,
        "{} regression: current={} baseline={} allowed={} (tolerance_ratio={})",
        metric,
        current,
        baseline,
        allowed,
        tolerance_ratio
    );
}

fn runtime_backend_signature() -> String {
    let cpu = CpuBackend;
    let cuda = CudaBackend;
    let cpu_caps = cpu.capabilities();
    let cuda_caps = cuda.capabilities();
    format!(
        "{:?}-{:?}-runtime-gateway-v1",
        cpu_caps.backend, cuda_caps.backend
    )
}

fn device_capability_fingerprint() -> String {
    let device = volta::ir::cuda::device::CudaDevice::default();
    format!(
        "{}-sm{}{}",
        device.name, device.compute_capability_major, device.compute_capability_minor
    )
}

fn device_fingerprint_matches(expected: &str, actual: &str) -> bool {
    if expected.eq_ignore_ascii_case("any") {
        return true;
    }

    let expected_lower = expected.to_ascii_lowercase();
    if let Some(sm) = expected_lower.strip_prefix("any-sm") {
        if sm.is_empty() {
            return false;
        }
        return actual.to_ascii_lowercase().ends_with(&format!("-sm{sm}"));
    }

    expected.eq_ignore_ascii_case(actual)
}

fn load_memory_baseline(path: &str) -> Result<MemoryBaseline, String> {
    let path_ref = Path::new(path);
    if !path_ref.exists() {
        return Err(format!(
            "missing CUDA memory baseline at '{}'",
            path_ref.display()
        ));
    }

    let raw = fs::read_to_string(path_ref)
        .map_err(|err| format!("failed to read baseline '{}': {}", path_ref.display(), err))?;
    let map = parse_key_value_file(&raw)?;

    Ok(MemoryBaseline {
        backend_signature: get_string(&map, "backend_signature")?,
        determinism_level: get_string(&map, "determinism_level")?,
        device_capability_fingerprint: get_string(&map, "device_capability_fingerprint")?,
        placement_fingerprint: get_u64(&map, "placement_fingerprint")?,
        peak_device_bytes: get_usize(&map, "peak_device_bytes")?,
        peak_host_transfer_bytes: get_usize(&map, "peak_host_transfer_bytes")?,
        peak_temp_bytes: get_usize(&map, "peak_temp_bytes")?,
        alignment_consistent: get_bool(&map, "alignment_consistent")?,
        tolerance_ratio: get_f32(&map, "tolerance_ratio")?,
    })
}

fn parse_key_value_file(raw: &str) -> Result<HashMap<String, String>, String> {
    let mut map = HashMap::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            return Err(format!("invalid baseline line: '{}'", trimmed));
        };
        map.insert(key.trim().to_string(), value.trim().to_string());
    }
    Ok(map)
}

fn get_string(map: &HashMap<String, String>, key: &str) -> Result<String, String> {
    map.get(key)
        .cloned()
        .ok_or_else(|| format!("missing '{}' in memory baseline", key))
}

fn get_usize(map: &HashMap<String, String>, key: &str) -> Result<usize, String> {
    let value = get_string(map, key)?;
    value
        .parse::<usize>()
        .map_err(|err| format!("invalid usize for '{}': {} ({})", key, value, err))
}

fn get_u64(map: &HashMap<String, String>, key: &str) -> Result<u64, String> {
    let value = get_string(map, key)?;
    value
        .parse::<u64>()
        .map_err(|err| format!("invalid u64 for '{}': {} ({})", key, value, err))
}

fn get_f32(map: &HashMap<String, String>, key: &str) -> Result<f32, String> {
    let value = get_string(map, key)?;
    value
        .parse::<f32>()
        .map_err(|err| format!("invalid f32 for '{}': {} ({})", key, value, err))
}

fn get_bool(map: &HashMap<String, String>, key: &str) -> Result<bool, String> {
    let value = get_string(map, key)?;
    if value.eq_ignore_ascii_case("true") {
        Ok(true)
    } else if value.eq_ignore_ascii_case("false") {
        Ok(false)
    } else {
        Err(format!("invalid bool for '{}': {}", key, value))
    }
}

fn with_determinism<T>(level: &str, run: impl FnOnce() -> T) -> T {
    cuda_helpers::with_determinism_ret(level, run)
}
