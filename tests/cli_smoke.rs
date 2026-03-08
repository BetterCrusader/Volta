// tests/cli_smoke.rs
// CLI integration smoke tests — Wave 0 prerequisite for Phase 6.
//
// Run with: cargo test --quiet --test cli_smoke
//
// Each test is independent (no shared state). All tests use CARGO_BIN_EXE_volta
// to find the binary built by cargo — never `cargo run`.
//
// Relative example paths are resolved via CARGO_MANIFEST_DIR as current_dir.

use std::fs;
use std::io::Write;
use std::process::Command;

use serde_json::Value;

fn volta_bin() -> &'static str {
    env!("CARGO_BIN_EXE_volta")
}

fn manifest_dir() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

/// `volta version` exits 0 and stdout contains a non-empty version string.
#[test]
fn smoke_version() {
    let output = Command::new(volta_bin())
        .arg("version")
        .current_dir(manifest_dir())
        .output()
        .expect("failed to spawn volta");

    assert!(
        output.status.success(),
        "volta version exited non-zero: {:?}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "volta version produced empty stdout"
    );
    // Should contain "volta" followed by a version
    assert!(
        stdout.contains("volta"),
        "expected 'volta' in version output, got: {stdout}"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "unexpected panic in volta version: {stderr}"
    );
}

/// `volta doctor` exits 0 and stdout contains the structured sections.
#[test]
fn smoke_doctor() {
    let output = Command::new(volta_bin())
        .arg("doctor")
        .current_dir(manifest_dir())
        .output()
        .expect("failed to spawn volta");

    assert!(
        output.status.success(),
        "volta doctor exited non-zero: {:?}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    for section in &[
        "Volta Doctor",
        "Capability Matrix",
        "AOT Codegen",
        "Environment Variables",
        "Next Steps",
    ] {
        assert!(
            stdout.contains(section),
            "expected '{section}' in volta doctor output, got: {stdout}"
        );
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "unexpected panic in volta doctor: {stderr}"
    );
}

/// `volta doctor --json` exits 0 and stdout is parseable JSON containing
/// the keys "tool", "healthy", and "backends".
#[test]
fn doctor_json_fields() {
    let output = Command::new(volta_bin())
        .args(["doctor", "--json"])
        .current_dir(manifest_dir())
        .output()
        .expect("failed to spawn volta");

    assert!(
        output.status.success(),
        "volta doctor --json exited non-zero: {:?}",
        output.status
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();

    let payload: Value =
        serde_json::from_str(trimmed).expect("volta doctor --json must produce valid JSON");
    assert_eq!(payload["tool"], "volta-doctor");
    assert!(
        payload.get("healthy").is_some(),
        "missing healthy field: {trimmed}"
    );
    assert!(
        payload["backends"].is_array(),
        "backends must be an array: {trimmed}"
    );
    assert_eq!(payload["cpu_target_mode"], "portable");
    assert!(
        payload["cpu_support_tier"].is_string(),
        "missing cpu_support_tier: {trimmed}"
    );
    assert!(
        payload["cpu_isa"].is_array(),
        "missing cpu_isa array: {trimmed}"
    );
    assert!(
        payload.get("mkl_available").is_some(),
        "missing mkl_available: {trimmed}"
    );
    assert!(
        payload.get("llvm_available").is_some(),
        "missing llvm_available: {trimmed}"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "unexpected panic in volta doctor --json: {stderr}"
    );
}

/// `volta run examples/xor.vt --quiet` exits 0 with no "error" in stderr.
#[test]
fn smoke_run_xor() {
    let output = Command::new(volta_bin())
        .args(["run", "examples/xor.vt", "--quiet"])
        .current_dir(manifest_dir())
        .output()
        .expect("failed to spawn volta");

    assert!(
        output.status.success(),
        "volta run examples/xor.vt --quiet exited non-zero: {:?}",
        output.status
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.to_ascii_lowercase().contains("error"),
        "unexpected 'error' in stderr for volta run xor: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "unexpected panic in volta run xor: {stderr}"
    );
}

/// `volta check examples/bench_real.vt --quiet` exits 0 with no "error" in stderr.
#[test]
fn smoke_check_bench_real() {
    let output = Command::new(volta_bin())
        .args(["check", "examples/bench_real.vt", "--quiet"])
        .current_dir(manifest_dir())
        .output()
        .expect("failed to spawn volta");

    assert!(
        output.status.success(),
        "volta check examples/bench_real.vt --quiet exited non-zero: {:?}",
        output.status
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.to_ascii_lowercase().contains("error"),
        "unexpected 'error' in stderr for volta check bench_real: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "unexpected panic in volta check bench_real: {stderr}"
    );
}

/// `volta compile-train` on a function-backed (non-MLP) model exits non-zero
/// or outputs "unsupported"/"MLP-only" to confirm the rejection path.
///
/// compile-train is MLP-only. When a model uses a function template (via `use`
/// keyword), it is rejected with "MLP-only today". This test creates a minimal
/// temp .vt file that triggers that path and asserts the expected failure.
#[test]
fn smoke_compile_train_rejects_non_mlp() {
    // Minimal .vt: define a function, then a model that uses it.
    // The executor runs the script (trains via the MLP path with layers 2 3 1),
    // then compile_first_model_to_train_dll sees use_fn is set and rejects with
    // "MLP-only today".
    let vt_source = r#"fn encoder_block(x)
    return x

model tiny_enc
    layers 2 3 1
    activation relu
    use encoder_block

dataset tiny_data
    type synthetic
    batch 2

train tiny_enc on tiny_data
    epochs 1
    optimizer sgd
    lr 0.01
    device cpu
"#;

    let mut tmp_path = std::env::temp_dir();
    tmp_path.push("volta_smoke_nonmlp_test.vt");

    {
        let mut f = fs::File::create(&tmp_path).expect("failed to create temp .vt file");
        f.write_all(vt_source.as_bytes())
            .expect("failed to write temp .vt file");
    }

    let output = Command::new(volta_bin())
        .arg("compile-train")
        .arg(tmp_path.to_str().expect("temp path is valid UTF-8"))
        .current_dir(manifest_dir())
        .output()
        .expect("failed to spawn volta");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

    let rejected = !output.status.success()
        || combined.to_ascii_lowercase().contains("unsupported")
        || combined.contains("MLP-only");

    assert!(
        rejected,
        "compile-train on a function-backed model should exit non-zero or emit 'MLP-only'/'unsupported', \
         but got exit={:?}, stdout={stdout:?}, stderr={stderr:?}",
        output.status
    );

    assert!(
        !combined.contains("panicked"),
        "unexpected panic in compile-train: {combined}"
    );

    // Clean up temp file (best-effort)
    let _ = fs::remove_file(&tmp_path);
}

#[test]
fn smoke_compile_train_c_path_rejects_adam_optimizer() {
    let vt_source = r#"model demo
    layers 2 4 1
    activation relu
    optimizer adam

dataset tiny_data
    type synthetic
    batch 2

train demo on tiny_data
    epochs 1
    optimizer adam
    lr 0.001
    device cpu
"#;

    let mut tmp_path = std::env::temp_dir();
    tmp_path.push("volta_smoke_adam_c_path_test.vt");
    fs::write(&tmp_path, vt_source).expect("failed to write temp .vt file");

    let output = Command::new(volta_bin())
        .args([
            "compile-train",
            tmp_path.to_str().expect("temp path is valid UTF-8"),
            "--cpu-target",
            "portable",
        ])
        .current_dir(manifest_dir())
        .output()
        .expect("failed to spawn volta");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "C path should reject Adam, got: {combined}"
    );
    assert!(
        combined.contains("supports only SGD today"),
        "unexpected output: {combined}"
    );
    assert!(
        !combined.contains("panicked"),
        "unexpected panic: {combined}"
    );

    let _ = fs::remove_file(&tmp_path);
}

#[test]
fn smoke_compile_train_rust_path_rejects_rmsprop_optimizer() {
    let vt_source = r#"model demo
    layers 2 4 1
    activation relu
    optimizer rmsprop

dataset tiny_data
    type synthetic
    batch 2

train demo on tiny_data
    epochs 1
    optimizer rmsprop
    lr 0.001
    device cpu
"#;

    let mut tmp_path = std::env::temp_dir();
    tmp_path.push("volta_smoke_rust_optimizer_test.vt");
    fs::write(&tmp_path, vt_source).expect("failed to write temp .vt file");

    let output = Command::new(volta_bin())
        .args([
            "compile-train",
            tmp_path.to_str().expect("temp path is valid UTF-8"),
            "--rust",
            "--cpu-target",
            "portable",
        ])
        .current_dir(manifest_dir())
        .output()
        .expect("failed to spawn volta");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "Rust path should reject unsupported RMSProp, got: {combined}"
    );
    assert!(
        combined.contains("supports only SGD, Adam, AdamW, and Adagrad"),
        "unexpected output: {combined}"
    );
    assert!(
        !combined.contains("panicked"),
        "unexpected panic: {combined}"
    );

    let _ = fs::remove_file(&tmp_path);
}
