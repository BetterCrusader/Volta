use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use volta::ir::{
    Backend, CpuBackend, CudaBackend, Graph, Op, Tensor, build_execution_plan,
    execute_terminal_with_backend,
};
use volta::model::{CompiledModel, ModelBuilder, Parameter, TensorShape, infer_with_backend};

const PARITY_BASELINE_PATH: &str = "benchmarks/baselines/cuda-infer-parity.json";

#[derive(Debug, Clone)]
struct ParityBaseline {
    backend_signature: String,
    determinism_level: String,
    device_capability_fingerprint: String,
    max_abs_diff_budget: f32,
    max_rel_diff_budget: f32,
}

#[test]
fn cuda_infer_parity_stays_within_strict_budget() {
    with_determinism("strict", || {
        let baseline = load_parity_baseline(PARITY_BASELINE_PATH).unwrap_or_else(|message| {
            panic!("{message}");
        });
        assert_eq!(baseline.backend_signature, runtime_backend_signature());
        assert_eq!(baseline.determinism_level, "strict");
        assert_eq!(
            baseline.device_capability_fingerprint,
            device_capability_fingerprint()
        );

        let (model, infer_input) = build_parity_fixture_model();
        let cpu = CpuBackend;
        let cuda = CudaBackend;

        let mut max_abs_seen = 0.0_f32;
        let mut max_rel_seen = 0.0_f32;
        let mut first_cuda = None;

        for _ in 0..5 {
            let cpu_out = infer_with_backend(&model, &model.parameters, &infer_input, &cpu)
                .expect("cpu infer should pass");
            let cuda_out = infer_with_backend(&model, &model.parameters, &infer_input, &cuda)
                .expect("cuda infer should pass for supported kernel subset");

            assert_eq!(cpu_out.shape, cuda_out.shape, "shape mismatch");
            assert_eq!(
                cpu_out.data.len(),
                cuda_out.data.len(),
                "tensor length mismatch"
            );

            if let Some(ref first) = first_cuda {
                assert_eq!(
                    *first, cuda_out,
                    "strict mode must be exactly repeatable across runs"
                );
            } else {
                first_cuda = Some(cuda_out.clone());
            }

            let abs = max_abs_diff(&cpu_out.data, &cuda_out.data);
            let rel = max_rel_diff(&cpu_out.data, &cuda_out.data);
            max_abs_seen = max_abs_seen.max(abs);
            max_rel_seen = max_rel_seen.max(rel);
        }

        assert!(
            max_abs_seen <= baseline.max_abs_diff_budget,
            "max_abs_diff regression: observed={} budget={}",
            max_abs_seen,
            baseline.max_abs_diff_budget
        );
        assert!(
            max_abs_seen <= 1e-6,
            "strict parity must stay <= 1e-6, observed={}",
            max_abs_seen
        );
        assert!(
            max_rel_seen <= baseline.max_rel_diff_budget,
            "max_rel_diff regression: observed={} budget={}",
            max_rel_seen,
            baseline.max_rel_diff_budget
        );
    });
}

#[test]
fn cuda_infer_path_has_no_silent_cpu_fallback() {
    with_determinism("strict", || {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add const should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add const should succeed");
        let (_, product) = graph
            .add_op(block, Op::Mul(a, b))
            .expect("add mul should succeed");
        graph
            .add_op(block, Op::Output(product))
            .expect("add output should succeed");

        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");

        let cpu = CpuBackend;
        execute_terminal_with_backend(
            &graph,
            &plan,
            &plan.schedule.ordered_nodes,
            &cpu,
            &Default::default(),
        )
        .expect("cpu path should execute unsupported-cuda graph");

        let cuda = CudaBackend;
        let err = execute_terminal_with_backend(
            &graph,
            &plan,
            &plan.schedule.ordered_nodes,
            &cuda,
            &Default::default(),
        )
        .expect_err("cuda path must fail instead of silently falling back to cpu");
        assert!(
            err.message
                .contains("unsupported CUDA kernel class: Elementwise"),
            "unexpected error: {}",
            err.message
        );
    });
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

fn build_parity_fixture_model() -> (CompiledModel, std::collections::HashMap<String, Tensor>) {
    let mut builder = ModelBuilder::new();
    let x = builder
        .input_with_shape("x", vec![1, 2])
        .expect("add input should succeed");
    let logits = builder
        .input_with_shape("logits", vec![2])
        .expect("add logits input should succeed");

    let w = builder
        .add_parameter(Parameter::new(
            "parity.w",
            Tensor::new(vec![2, 2], vec![0.5, -1.0, 1.5, 2.0]).expect("valid parameter tensor"),
            true,
        ))
        .expect("add parameter should succeed");
    let b = builder
        .add_parameter(Parameter::new(
            "parity.b",
            Tensor::new(vec![1, 2], vec![0.25, -0.75]).expect("valid parameter tensor"),
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

    let model = builder
        .finalize(out, TensorShape(vec![1, 2]), None)
        .expect("model should finalize");

    let mut infer_input = std::collections::HashMap::new();
    infer_input.insert(
        "x".to_string(),
        Tensor::new(vec![1, 2], vec![1.0, -2.0]).expect("valid infer input"),
    );
    infer_input.insert(
        "logits".to_string(),
        Tensor::new(vec![2], vec![0.1, 0.9]).expect("valid logits input"),
    );

    (model, infer_input)
}

fn device_capability_fingerprint() -> String {
    let device = volta::ir::cuda::device::CudaDevice::default();
    format!(
        "{}-sm{}{}",
        device.name, device.compute_capability_major, device.compute_capability_minor
    )
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

fn load_parity_baseline(path: &str) -> Result<ParityBaseline, String> {
    let path_ref = Path::new(path);
    if !path_ref.exists() {
        return Err(format!(
            "missing parity baseline at '{}'; create it and commit backend signature, determinism level, device fingerprint, and diff budgets",
            path_ref.display()
        ));
    }

    let raw = fs::read_to_string(path_ref)
        .map_err(|err| format!("failed to read baseline '{}': {}", path_ref.display(), err))?;

    Ok(ParityBaseline {
        backend_signature: parse_json_string(&raw, "backend_signature")?,
        determinism_level: parse_json_string(&raw, "determinism_level")?,
        device_capability_fingerprint: parse_json_string(&raw, "device_capability_fingerprint")?,
        max_abs_diff_budget: parse_json_f32(&raw, "max_abs_diff_budget")?,
        max_rel_diff_budget: parse_json_f32(&raw, "max_rel_diff_budget")?,
    })
}

fn parse_json_string(raw: &str, key: &str) -> Result<String, String> {
    let marker = format!("\"{}\"", key);
    let start = raw
        .find(&marker)
        .ok_or_else(|| format!("missing '{}' in parity baseline", key))?;
    let after_key = &raw[start + marker.len()..];
    let colon = after_key
        .find(':')
        .ok_or_else(|| format!("missing ':' for '{}' in parity baseline", key))?;
    let after_colon = after_key[colon + 1..].trim_start();
    let Some(after_quote) = after_colon.strip_prefix('"') else {
        return Err(format!("'{}' must be a JSON string", key));
    };
    let end = after_quote
        .find('"')
        .ok_or_else(|| format!("unterminated string for '{}'", key))?;
    Ok(after_quote[..end].to_string())
}

fn parse_json_f32(raw: &str, key: &str) -> Result<f32, String> {
    let marker = format!("\"{}\"", key);
    let start = raw
        .find(&marker)
        .ok_or_else(|| format!("missing '{}' in parity baseline", key))?;
    let after_key = &raw[start + marker.len()..];
    let colon = after_key
        .find(':')
        .ok_or_else(|| format!("missing ':' for '{}' in parity baseline", key))?;
    let after_colon = after_key[colon + 1..].trim_start();
    let end = after_colon
        .find([',', '}'])
        .ok_or_else(|| format!("missing numeric terminator for '{}'", key))?;
    let token = after_colon[..end].trim();
    token
        .parse::<f32>()
        .map_err(|err| format!("invalid float for '{}': {} ({})", key, token, err))
}

fn with_determinism(level: &str, run: impl FnOnce()) {
    let _guard = match env_lock().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let _restore = EnvVarRestore::set("VOLTA_DETERMINISM", level);
    run();
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
