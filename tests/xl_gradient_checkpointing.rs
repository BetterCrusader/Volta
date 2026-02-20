use volta::model::{
    GradientCheckpointingConfig, build_tiny_transformer_fixture_for_tests,
    plan_gradient_checkpointing,
};

#[test]
fn checkpoint_plan_is_deterministic_for_same_graph_and_config() {
    let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();
    let config = GradientCheckpointingConfig {
        interval_nodes: 2,
        min_tensor_bytes: 16,
    };

    let first = plan_gradient_checkpointing(&model, &config)
        .expect("checkpoint plan should build on first run");
    let second = plan_gradient_checkpointing(&model, &config)
        .expect("checkpoint plan should build on second run");

    assert_eq!(first, second);
    assert!(!first.checkpoint_values.is_empty());
    assert!(first.estimated_saved_bytes > 0);
}

#[test]
fn checkpoint_plan_rejects_zero_interval() {
    let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();

    let err = plan_gradient_checkpointing(
        &model,
        &GradientCheckpointingConfig {
            interval_nodes: 0,
            min_tensor_bytes: 0,
        },
    )
    .expect_err("zero interval must be rejected");

    assert!(err.message.contains("interval_nodes"));
}
