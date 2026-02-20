use std::fs;
use std::path::Path;

const BYTES_BASELINE_PATH: &str = "benchmarks/baselines/tiny-transformer-memory-peak-bytes.txt";
const VALUES_BASELINE_PATH: &str = "benchmarks/baselines/tiny-transformer-memory-peak-values.txt";

#[test]
fn tiny_transformer_peak_memory_stays_within_budget() {
    let (model, _dataset, _cfg, _infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();
    let plan = volta::ir::plan_memory(&model.graph).expect("memory plan should build");
    let current_bytes = plan.peak_live_bytes;
    let current_values = plan.peak_live_values;

    println!(
        "tiny-transformer peak_live_values={} peak_live_bytes={}",
        current_values, current_bytes
    );

    let baseline_bytes = load_usize_baseline(BYTES_BASELINE_PATH, current_bytes, "peak_live_bytes")
        .unwrap_or_else(|message| {
            panic!("{message}");
        });
    let baseline_values =
        load_usize_baseline(VALUES_BASELINE_PATH, current_values, "peak_live_values")
            .unwrap_or_else(|message| {
                panic!("{message}");
            });

    let allowed_bytes = baseline_bytes.saturating_add(baseline_bytes.div_ceil(10));
    let allowed_values = baseline_values.saturating_add(baseline_values.div_ceil(10));

    assert!(
        current_values <= allowed_values,
        "memory liveness regression: current_values={} baseline_values={} allowed_values={} (+10%)",
        current_values,
        baseline_values,
        allowed_values
    );
    assert!(
        current_bytes <= allowed_bytes,
        "memory bytes regression: current_bytes={} baseline_bytes={} allowed_bytes={} (+10%)",
        current_bytes,
        baseline_bytes,
        allowed_bytes
    );
}

fn load_usize_baseline(path: &str, current: usize, metric_name: &str) -> Result<usize, String> {
    let path_ref = Path::new(path);
    if !path_ref.exists() {
        return Err(format!(
            "missing {} baseline at '{}'; current {}={}. Create the file with that integer value and commit it.",
            metric_name,
            path_ref.display(),
            metric_name,
            current
        ));
    }

    let raw = fs::read_to_string(path_ref)
        .map_err(|err| format!("failed to read baseline '{}': {}", path_ref.display(), err))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("baseline '{}' is empty", path_ref.display()));
    }

    trimmed.parse::<usize>().map_err(|err| {
        format!(
            "baseline '{}' must be a plain integer, got '{}': {}",
            path_ref.display(),
            trimmed,
            err
        )
    })
}
