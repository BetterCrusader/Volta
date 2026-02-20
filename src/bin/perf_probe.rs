use std::time::Instant;

use volta::ir::{Tensor, TensorError};

#[derive(Debug, Clone, Copy)]
struct ProbeConfig {
    dim: usize,
    samples: usize,
    matmul_iters: usize,
    relu_iters: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            dim: 96,
            samples: 9,
            matmul_iters: 3,
            relu_iters: 16,
        }
    }
}

#[derive(Debug, Clone)]
struct MetricSummary {
    median: f64,
    stdev: f64,
    lower_is_better: bool,
}

#[derive(Debug, Clone)]
struct ProbeReport {
    dim: usize,
    samples: usize,
    matmul_ms: MetricSummary,
    relu_ms: MetricSummary,
    checksum: f64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let report = run_probe(config).map_err(|err| err.message)?;
    println!("{}", render_json(&report));
    Ok(())
}

fn parse_args(args: &[String]) -> Result<ProbeConfig, String> {
    let mut config = ProbeConfig::default();
    let mut index = 1usize;

    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--dim" => {
                let value = parse_usize_value(args, &mut index, flag)?;
                if value == 0 {
                    return Err("--dim must be greater than zero".to_string());
                }
                config.dim = value;
            }
            "--samples" => {
                let value = parse_usize_value(args, &mut index, flag)?;
                if value == 0 {
                    return Err("--samples must be greater than zero".to_string());
                }
                config.samples = value;
            }
            "--matmul-iters" => {
                let value = parse_usize_value(args, &mut index, flag)?;
                if value == 0 {
                    return Err("--matmul-iters must be greater than zero".to_string());
                }
                config.matmul_iters = value;
            }
            "--relu-iters" => {
                let value = parse_usize_value(args, &mut index, flag)?;
                if value == 0 {
                    return Err("--relu-iters must be greater than zero".to_string());
                }
                config.relu_iters = value;
            }
            "-h" | "--help" => {
                println!(
                    "Usage: cargo run --release --bin perf_probe -- [--dim N] [--samples N] [--matmul-iters N] [--relu-iters N]"
                );
                std::process::exit(0);
            }
            _ => {
                return Err(format!("Unknown flag: {flag}"));
            }
        }
    }

    Ok(config)
}

fn parse_usize_value(args: &[String], index: &mut usize, flag: &str) -> Result<usize, String> {
    if *index + 1 >= args.len() {
        return Err(format!("Missing value for {flag}"));
    }
    let raw = &args[*index + 1];
    *index += 2;
    raw.parse::<usize>()
        .map_err(|_| format!("Invalid value for {flag}: '{raw}'"))
}

fn run_probe(config: ProbeConfig) -> Result<ProbeReport, TensorError> {
    let lhs = make_tensor(config.dim, config.dim, 17)?;
    let rhs = make_tensor(config.dim, config.dim, 31)?;
    let relu_input = make_tensor(config.dim, config.dim, 47)?;

    let mut matmul_samples = Vec::with_capacity(config.samples);
    let mut relu_samples = Vec::with_capacity(config.samples);
    let mut checksum = 0.0_f64;

    for sample_index in 0..config.samples {
        let matmul_start = Instant::now();
        for _ in 0..config.matmul_iters {
            let out = lhs.matmul(&rhs)?;
            checksum += out.data[sample_index % out.data.len()] as f64;
        }
        let matmul_ms = matmul_start.elapsed().as_secs_f64() * 1000.0 / config.matmul_iters as f64;
        matmul_samples.push(matmul_ms);

        let relu_start = Instant::now();
        for _ in 0..config.relu_iters {
            let out = relu_input.relu()?;
            checksum += out.data[sample_index % out.data.len()] as f64;
        }
        let relu_ms = relu_start.elapsed().as_secs_f64() * 1000.0 / config.relu_iters as f64;
        relu_samples.push(relu_ms);
    }

    Ok(ProbeReport {
        dim: config.dim,
        samples: config.samples,
        matmul_ms: MetricSummary {
            median: median(&matmul_samples),
            stdev: stdev(&matmul_samples),
            lower_is_better: true,
        },
        relu_ms: MetricSummary {
            median: median(&relu_samples),
            stdev: stdev(&relu_samples),
            lower_is_better: true,
        },
        checksum,
    })
}

fn make_tensor(rows: usize, cols: usize, seed: u64) -> Result<Tensor, TensorError> {
    let mut data = Vec::with_capacity(rows * cols);
    let mut state = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);

    for _ in 0..rows * cols {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bucket = (state % 10_007) as f32;
        data.push((bucket / 10_007.0) - 0.5);
    }

    Tensor::new(vec![rows, cols], data)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn stdev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn render_json(report: &ProbeReport) -> String {
    format!(
        "{{\"version\":1,\"dim\":{},\"samples\":{},\"metrics\":{{\"matmul_ms\":{{\"median\":{:.6},\"stdev\":{:.6},\"lower_is_better\":{}}},\"relu_ms\":{{\"median\":{:.6},\"stdev\":{:.6},\"lower_is_better\":{}}}}},\"checksum\":{:.6}}}",
        report.dim,
        report.samples,
        report.matmul_ms.median,
        report.matmul_ms.stdev,
        report.matmul_ms.lower_is_better,
        report.relu_ms.median,
        report.relu_ms.stdev,
        report.relu_ms.lower_is_better,
        report.checksum,
    )
}
