use volta::ir::{StaticMemoryBudget, evaluate_static_memory_budget};
use volta::model::{ScalingProfile, build_tiny_transformer_fixture_for_tests, enforce_xl_budget};

#[test]
fn xl_profile_budget_reports_non_zero_usage_for_tiny_transformer_fixture() {
    let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();
    let report = enforce_xl_budget(&model, &ScalingProfile::xl_default())
        .expect("xl default profile should accept fixture model");

    assert!(report.peak_live_bytes > 0);
    assert!(report.peak_live_values > 0);
    assert!(report.within_budget);
}

#[test]
fn static_budget_rejects_model_when_peak_bytes_exceed_limit() {
    let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();

    let err = evaluate_static_memory_budget(
        &model.graph,
        &StaticMemoryBudget {
            max_peak_live_bytes: 1,
            max_peak_live_values: 1,
        },
    )
    .expect_err("budget must fail when limits are unrealistically low");

    assert!(
        err.message.contains("peak_live_bytes") || err.message.contains("peak_live_values"),
        "unexpected error: {}",
        err.message
    );
}
