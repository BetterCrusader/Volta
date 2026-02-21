use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_file(name: &str, content: &str) -> PathBuf {
    let mut path = env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    path.push(format!("volta_cli_{name}_{nanos}.vt"));
    fs::write(&path, content).expect("temp script should be writable");
    path
}

fn run_volta(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_volta"))
        .args(args)
        .output()
        .expect("volta binary should execute")
}

fn run_volta_with_env(args: &[&str], key: &str, value: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_volta"))
        .args(args)
        .env(key, value)
        .output()
        .expect("volta binary should execute")
}

#[test]
fn check_command_passes_for_valid_script() {
    let path = unique_temp_file("check_ok", "x 1\nprint x\n");

    let output = run_volta(&["check", path.to_str().expect("utf8 path")]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "check should pass: {stdout}");
    assert!(stdout.contains("Check passed"));
}

#[test]
fn run_command_executes_program() {
    let path = unique_temp_file("run_ok", "x 1\nprint x\n");

    let output = run_volta(&["run", path.to_str().expect("utf8 path")]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "run should pass: {stdout}");
    assert!(stdout.contains("1"));
    assert!(stdout.contains("Run completed"));
}

#[test]
fn info_command_prints_summary() {
    let path = unique_temp_file("info_ok", "x 1\nloop 2\n    print x\n");

    let output = run_volta(&["info", path.to_str().expect("utf8 path")]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "info should pass: {stdout}");
    assert!(stdout.contains("Top-level statements: 2"));
    assert!(stdout.contains("loop: 1"));
    assert!(stdout.contains("print: 1"));
}

#[test]
fn invalid_syntax_returns_nonzero() {
    let path = unique_temp_file("parse_fail", "x =\n");

    let output = run_volta(&["check", path.to_str().expect("utf8 path")]);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(), "expected failure for parse error");
    assert!(stderr.contains("Parse error"));
}

#[test]
fn file_commands_reject_empty_path() {
    let output = run_volta(&["run", ""]);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "empty path should return non-zero exit code"
    );
    assert!(stderr.contains("non-empty file path"));
}

#[test]
fn bare_file_argument_runs_for_backward_compatibility() {
    let path = unique_temp_file("run_compat", "x 1\nprint x\n");
    let output = run_volta(&[path.to_str().expect("utf8 path")]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "run should pass: {stdout}");
    assert!(stdout.contains("Run completed"));
}

#[test]
fn legacy_bench_and_tune_flags_exit_successfully() {
    let bench = run_volta(&[
        "--bench-infer",
        "--runs",
        "1",
        "--warmup",
        "0",
        "--tokens",
        "4",
    ]);
    assert!(bench.status.success(), "legacy bench-infer should succeed");

    let tune = run_volta(&["--tune-matmul", "--dim", "64", "--runs", "1"]);
    assert!(tune.status.success(), "legacy tune-matmul should succeed");
}

#[test]
fn doctor_command_reports_environment() {
    let output = run_volta(&["doctor"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "doctor should pass: {stdout}");
    assert!(stdout.contains("Volta doctor"));
    assert!(stdout.contains("cpu_threads:"));
    assert!(stdout.contains("gpu_available:"));
    assert!(stdout.contains("feature_onnx_import:"));
}

#[test]
fn doctor_command_supports_json_mode() {
    let output = run_volta(&["doctor", "--json"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "doctor --json should pass: {stdout}"
    );
    assert!(stdout.trim_start().starts_with('{'));
    assert!(stdout.contains("\"tool\":\"volta-doctor\""));
    assert!(stdout.contains("\"cpu_threads\":"));
    assert!(stdout.contains("\"gpu_available\":"));
}

#[test]
fn doctor_command_rejects_unknown_flag() {
    let output = run_volta(&["doctor", "--yaml"]);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "doctor with unknown flag must fail"
    );
    assert!(stderr.contains("accepts only optional '--json'"));
}

#[test]
fn doctor_reports_invalid_gpu_env_value() {
    let output = run_volta_with_env(&["doctor"], "VOLTA_GPU_AVAILABLE", "maybe");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "doctor should still succeed");
    assert!(stdout.contains("gpu_env_raw: maybe"));
    assert!(stdout.contains("warning: VOLTA_GPU_AVAILABLE has invalid value"));
}

#[test]
fn doctor_json_reports_invalid_gpu_env_value() {
    let output = run_volta_with_env(&["doctor", "--json"], "VOLTA_GPU_AVAILABLE", "maybe");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "doctor --json should still succeed");
    assert!(stdout.contains("\"gpu_env_raw\":\"maybe\""));
    assert!(stdout.contains("\"gpu_env_valid\":false"));
}
