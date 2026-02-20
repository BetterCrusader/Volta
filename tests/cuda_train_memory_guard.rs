use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use volta::ir::cuda::{lower_plan, profile_memory};
use volta::ir::{Backend, CpuBackend, CudaBackend, build_execution_plan, build_reverse_graph};
use volta::model::build_tiny_transformer_fixture_for_tests;

const MEMORY_BASELINE_PATH: &str = "benchmarks/baselines/cuda-train-memory-peak-bytes.txt";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrainMemoryProfile {
    forward_placement_fingerprint: u64,
    backward_placement_fingerprint: u64,
    peak_device_bytes: usize,
    peak_host_transfer_bytes: usize,
    peak_temp_bytes: usize,
    workspace_total_bytes: usize,
}

#[derive(Debug, Clone)]
struct MemoryBaseline {
    backend_signature: String,
    determinism_level: String,
    device_capability_fingerprint: String,
    forward_placement_fingerprint: u64,
    backward_placement_fingerprint: u64,
    peak_device_bytes: usize,
    peak_host_transfer_bytes: usize,
    peak_temp_bytes: usize,
    workspace_total_bytes: usize,
    tolerance_ratio: f32,
}

#[test]
fn cuda_train_memory_guard_enforces_baseline_and_contracts() {
    let strict_runs = (0..5)
        .map(|_| profile_train_for_level("strict"))
        .collect::<Vec<_>>();
    for run in &strict_runs[1..] {
        assert_eq!(
            *run, strict_runs[0],
            "strict training memory profile must be stable across runs"
        );
    }

    let strict_profile = strict_runs[0];
    assert!(
        strict_profile.workspace_total_bytes > 0,
        "training lowering must allocate deterministic workspace buffers"
    );

    let baseline = load_memory_baseline(MEMORY_BASELINE_PATH).unwrap_or_else(|message| {
        panic!(
            "{message}; suggested baseline: backend_signature={}, determinism_level=strict, device_capability_fingerprint={}, forward_placement_fingerprint={}, backward_placement_fingerprint={}, peak_device_bytes={}, peak_host_transfer_bytes={}, peak_temp_bytes={}, workspace_total_bytes={}, tolerance_ratio=0.10",
            runtime_backend_signature(),
            device_capability_fingerprint(),
            strict_profile.forward_placement_fingerprint,
            strict_profile.backward_placement_fingerprint,
            strict_profile.peak_device_bytes,
            strict_profile.peak_host_transfer_bytes,
            strict_profile.peak_temp_bytes,
            strict_profile.workspace_total_bytes,
        );
    });

    assert_eq!(baseline.backend_signature, runtime_backend_signature());
    assert_eq!(baseline.determinism_level, "strict");
    assert_eq!(
        baseline.device_capability_fingerprint,
        device_capability_fingerprint()
    );
    assert_eq!(
        strict_profile.forward_placement_fingerprint, baseline.forward_placement_fingerprint,
        "forward placement hints changed; update baseline only after review"
    );
    assert_eq!(
        strict_profile.backward_placement_fingerprint, baseline.backward_placement_fingerprint,
        "backward placement hints changed; update baseline only after review"
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
    assert_with_tolerance(
        "workspace_total_bytes",
        strict_profile.workspace_total_bytes,
        baseline.workspace_total_bytes,
        baseline.tolerance_ratio,
    );
}

fn profile_train_for_level(level: &str) -> TrainMemoryProfile {
    with_determinism(level, || {
        let (model, _dataset, _train_cfg, _infer_input) =
            build_tiny_transformer_fixture_for_tests();

        let forward_plan =
            build_execution_plan(&model.graph, &HashSet::new()).expect("forward plan should build");
        let forward_lowered = lower_plan(&forward_plan).expect("forward cuda lowering should pass");
        let forward_profile = profile_memory(&model.graph, &forward_plan, &forward_lowered)
            .expect("forward memory profile should pass");

        let loss = model.loss.expect("fixture must include loss");
        let parameter_values = model.parameter_values.values().copied().collect::<Vec<_>>();
        let reverse = build_reverse_graph(&model.graph, loss, &parameter_values)
            .expect("reverse graph should build");

        let gradient_values = reverse.gradients.values().copied().collect::<HashSet<_>>();
        let backward_plan = build_execution_plan(&reverse.backward, &gradient_values)
            .expect("backward plan should build");
        let backward_lowered =
            lower_plan(&backward_plan).expect("backward cuda lowering should pass");
        let backward_profile = profile_memory(&reverse.backward, &backward_plan, &backward_lowered)
            .expect("backward memory profile should pass");

        TrainMemoryProfile {
            forward_placement_fingerprint: forward_profile.placement_fingerprint,
            backward_placement_fingerprint: backward_profile.placement_fingerprint,
            peak_device_bytes: forward_profile
                .peak_device_bytes
                .max(backward_profile.peak_device_bytes),
            peak_host_transfer_bytes: forward_profile
                .peak_host_transfer_bytes
                .saturating_add(backward_profile.peak_host_transfer_bytes),
            peak_temp_bytes: forward_profile
                .peak_temp_bytes
                .max(backward_profile.peak_temp_bytes),
            workspace_total_bytes: backward_lowered
                .workspace_buffers
                .iter()
                .map(|buffer| buffer.bytes)
                .sum::<usize>(),
        }
    })
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

fn load_memory_baseline(path: &str) -> Result<MemoryBaseline, String> {
    let path_ref = Path::new(path);
    if !path_ref.exists() {
        return Err(format!(
            "missing CUDA training memory baseline at '{}'",
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
        forward_placement_fingerprint: get_u64(&map, "forward_placement_fingerprint")?,
        backward_placement_fingerprint: get_u64(&map, "backward_placement_fingerprint")?,
        peak_device_bytes: get_usize(&map, "peak_device_bytes")?,
        peak_host_transfer_bytes: get_usize(&map, "peak_host_transfer_bytes")?,
        peak_temp_bytes: get_usize(&map, "peak_temp_bytes")?,
        workspace_total_bytes: get_usize(&map, "workspace_total_bytes")?,
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

fn with_determinism<T>(level: &str, run: impl FnOnce() -> T) -> T {
    let _guard = match env_lock().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let _restore = EnvVarRestore::set("VOLTA_DETERMINISM", level);
    run()
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvVarRestore {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe {
                std::env::set_var(self.key, value);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}
