/// Integration tests for flexible label_col + num_classes dataset feature.
use std::io::Write;
use std::process::Command;

fn volta_bin() -> String {
    std::env::current_dir()
        .unwrap()
        .join("target/debug/volta.exe")
        .to_string_lossy()
        .to_string()
}

fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(volta_bin()).args(args).output().expect("volta binary not found");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().unwrap_or(-1), combined)
}

struct TempDir(std::path::PathBuf);
impl TempDir {
    fn path(&self) -> &std::path::Path { &self.0 }
}
impl Drop for TempDir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}
fn tempdir() -> TempDir {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    let path = std::env::temp_dir().join(format!("volta_label_{ts}"));
    std::fs::create_dir_all(&path).unwrap();
    TempDir(path)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_label_col_trains_successfully() {
    let (code, out) = run(&["tests/integration/iris_indexed.vt"]);
    assert_eq!(code, 0, "iris_indexed.vt failed:\n{out}");
    assert!(out.contains("Training completed"), "Missing 'Training completed':\n{out}");
    assert!(!out.contains("NaN"), "Output contains NaN:\n{out}");
}

#[test]
fn test_backward_compat_one_hot_csv_unchanged() {
    // Existing one-hot pipeline must still work — no label_col, no num_classes
    let (code, out) = run(&["tests/integration/iris.vt"]);
    assert_eq!(code, 0, "iris.vt (legacy one-hot) failed:\n{out}");
    assert!(out.contains("Training completed"), "Missing 'Training completed':\n{out}");
}

#[test]
fn test_label_col_without_num_classes_is_error() {
    let tmp = tempdir();
    let script = tmp.path().join("bad.vt");
    let mut f = std::fs::File::create(&script).unwrap();
    writeln!(f, "model m\n    layers 4 8 3\n    activation \"softmax\"\n\ndataset d\n    source \"tests/data/iris_indexed.csv\"\n    batch 4\n    label_col 4\n\ntrain m on d\n    epochs 1\n    lr 0.01").unwrap();

    let (code, out) = run(&[script.to_str().unwrap()]);
    assert_ne!(code, 0, "Expected error for missing num_classes, got exit 0:\n{out}");
    assert!(
        out.contains("num_classes"),
        "Error message should mention 'num_classes':\n{out}"
    );
}

#[test]
fn test_label_out_of_range_is_error() {
    let tmp = tempdir();
    // CSV with label=9 but num_classes=3
    let csv = tmp.path().join("bad_label.csv");
    std::fs::write(&csv, "f0,f1,f2,f3,label\n1.0,2.0,3.0,4.0,9\n").unwrap();
    let script = tmp.path().join("bad_label.vt");
    let mut f = std::fs::File::create(&script).unwrap();
    writeln!(f,
        "model m\n    layers 4 8 3\n    activation \"softmax\"\n\ndataset d\n    source \"{}\"\n    batch 1\n    label_col 4\n    num_classes 3\n\ntrain m on d\n    epochs 1\n    lr 0.01",
        csv.to_string_lossy().replace('\\', "/")
    ).unwrap();

    let (code, out) = run(&[script.to_str().unwrap()]);
    assert_ne!(code, 0, "Expected error for out-of-range label:\n{out}");
    assert!(
        out.contains("out of range"),
        "Error should mention 'out of range':\n{out}"
    );
}
