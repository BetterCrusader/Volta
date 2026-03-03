/// Integration test: save → load → infer parity
///
/// Verifies that running inference before and after a save/load round-trip
/// produces bit-identical output (same floats within 1e-6 tolerance).
use std::process::Command;

fn volta_bin() -> String {
    // Use the debug build produced by cargo test
    let bin = std::env::current_dir()
        .unwrap()
        .join("target")
        .join("debug")
        .join("volta.exe");
    bin.to_string_lossy().to_string()
}

fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(volta_bin())
        .args(args)
        .output()
        .expect("Failed to run volta binary");
    let code = out.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    (code, format!("{stdout}{stderr}"))
}

fn parse_csv_floats(path: &str) -> Vec<Vec<f32>> {
    let mut rdr = csv::Reader::from_path(path).expect("Failed to open CSV");
    rdr.records()
        .map(|rec| {
            rec.unwrap()
                .iter()
                .map(|cell| cell.trim().parse::<f32>().expect("Not a float"))
                .collect()
        })
        .collect()
}

#[test]
fn test_save_load_infer_parity() {
    // Step 1: Full train → save → infer cycle
    let (code1, out1) = run(&["tests/integration/iris_train_save_load_infer.vt"]);
    assert_eq!(code1, 0, "First run failed:\n{out1}");

    // Capture predictions from first run
    let predictions_first = parse_csv_floats("tests/data/out.csv");
    assert!(!predictions_first.is_empty(), "First inference produced no rows");

    // Step 2: Load saved weights → infer again (no retrain)
    let (code2, out2) = run(&["tests/integration/iris_parity_load_infer.vt"]);
    assert_eq!(code2, 0, "Second run (load+infer) failed:\n{out2}");

    let predictions_second = parse_csv_floats("tests/data/out.csv");
    assert_eq!(
        predictions_first.len(),
        predictions_second.len(),
        "Row count mismatch between runs"
    );

    // Step 3: Compare float-by-float within tolerance
    const TOLERANCE: f32 = 1e-5;
    for (row_idx, (row1, row2)) in predictions_first.iter().zip(predictions_second.iter()).enumerate() {
        assert_eq!(row1.len(), row2.len(), "Column count mismatch at row {row_idx}");
        for (col_idx, (a, b)) in row1.iter().zip(row2.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < TOLERANCE,
                "Parity mismatch at row {row_idx}, col {col_idx}: {a} vs {b} (diff={diff:.2e})"
            );
        }
    }
}
