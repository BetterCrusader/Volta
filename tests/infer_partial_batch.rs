/// Integration test: inference does not drop the partial last batch
///
/// Creates a CSV with N rows where N is NOT divisible by the default
/// inference batch_size (32). Verifies that all N rows appear in the output.
use std::io::Write;
use std::process::Command;

fn volta_bin() -> String {
    std::env::current_dir()
        .unwrap()
        .join("target")
        .join("debug")
        .join("volta.exe")
        .to_string_lossy()
        .to_string()
}

/// Minimal tempdir helper (no extra crates needed)
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
    let path = std::env::temp_dir().join(format!("volta_infer_partial_{ts}"));
    std::fs::create_dir_all(&path).unwrap();
    TempDir(path)
}

fn count_csv_data_rows(path: &str) -> usize {
    let content = std::fs::read_to_string(path).expect("Failed to read output CSV");
    // Skip header line
    content.lines().filter(|l| !l.trim().is_empty()).count().saturating_sub(1)
}

#[test]
fn test_infer_all_samples_including_partial_batch() {
    // We need a pre-trained checkpoint. First, ensure the standard checkpoint exists.
    let setup_status = Command::new(volta_bin())
        .arg("tests/integration/iris_train_save_load_infer.vt")
        .status()
        .expect("Failed to run setup script");
    assert!(setup_status.success(), "Setup script failed");

    let tmp = tempdir();

    // Build a CSV with 13 rows (NOT divisible by 32, the default inference batch_size)
    // Using iris-like feature structure: 4 features per row
    let input_csv = tmp.path().join("partial_input.csv");
    {
        let mut f = std::fs::File::create(&input_csv).unwrap();
        // Header (features only, no label column for inference)
        writeln!(f, "f0,f1,f2,f3").unwrap();
        for i in 0..13usize {
            writeln!(f, "{:.2},{:.2},{:.2},{:.2}",
                5.0 + i as f64 * 0.1,
                3.0 + i as f64 * 0.05,
                1.5 + i as f64 * 0.2,
                0.3 + i as f64 * 0.03
            ).unwrap();
        }
    }

    let output_csv = tmp.path().join("partial_output.csv");

    // Write a .vt script that loads the pre-trained checkpoint and infers on our 13-row CSV
    let script_path = tmp.path().join("partial_infer.vt");
    let script = format!(
        "model iris_net\n    layers 4 8 3\n    activation \"softmax\"\n\nload iris_net as \"tests/data/iris_model.vt\"\n\ninfer iris_net on \"{}\" as \"{}\"\n",
        input_csv.to_string_lossy().replace('\\', "/"),
        output_csv.to_string_lossy().replace('\\', "/"),
    );
    std::fs::write(&script_path, script).unwrap();

    // Run inference
    let out = Command::new(volta_bin())
        .arg(script_path.to_str().unwrap())
        .output()
        .expect("Failed to run infer script");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "Inference script failed:\n{combined}"
    );

    let out_csv_str = output_csv.to_string_lossy().to_string();
    let row_count = count_csv_data_rows(&out_csv_str);
    assert_eq!(
        row_count, 13,
        "Expected 13 prediction rows (no partial batch dropped), got {row_count}"
    );
}
