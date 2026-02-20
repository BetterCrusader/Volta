use std::fs;
use std::path::Path;

const BASELINE_PATH: &str = "benchmarks/baselines/tiny-transformer-memory-peak-bytes.txt";

#[test]
fn tiny_transformer_peak_memory_stays_within_budget() {
    let (model, _dataset, _cfg, _infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();
    let plan = volta::ir::plan_memory(&model.graph).expect("memory plan should build");
    let current = plan.peak_live_bytes;

    println!("tiny-transformer peak_live_bytes={current}");

    let baseline = load_peak_live_bytes_baseline(current).unwrap_or_else(|message| {
        panic!("{message}");
    });
    let allowed = baseline.saturating_add((baseline + 9) / 10);

    assert!(
        current <= allowed,
        "memory regression: current={} baseline={} allowed={} (+10%)",
        current,
        baseline,
        allowed
    );
}

fn load_peak_live_bytes_baseline(current: usize) -> Result<usize, String> {
    let path = Path::new(BASELINE_PATH);
    if !path.exists() {
        return Err(format!(
            "missing memory baseline at '{}'; current peak_live_bytes={}. Create the file with that integer value and commit it.",
            BASELINE_PATH, current
        ));
    }

    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read baseline '{}': {}", BASELINE_PATH, err))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("baseline '{}' is empty", BASELINE_PATH));
    }

    trimmed.parse::<usize>().map_err(|err| {
        format!(
            "baseline '{}' must be a plain integer, got '{}': {}",
            BASELINE_PATH, trimmed, err
        )
    })
}
