#[path = "common/cuda.rs"]
mod cuda_helpers;

use volta::ir::CudaBackend;
use volta::model::{
    GradientCheckpointingConfig, build_tiny_transformer_fixture_for_tests,
    plan_gradient_checkpointing, train_with_backend,
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

#[test]
fn strict_training_with_and_without_checkpointing_produces_identical_result() {
    if !cuda_helpers::cuda_runtime_available() {
        eprintln!("[SKIP] strict_training_with_and_without_checkpointing_produces_identical_result — no CUDA device available");
        return;
    }
    cuda_helpers::with_determinism("strict", || {
        let (model, dataset, mut config, _infer_input) = build_tiny_transformer_fixture_for_tests();
        let cuda = CudaBackend;

        config.epochs = 10;
        config.gradient_checkpointing = None;
        let without_checkpointing = train_with_backend(&model, &dataset, &config, &cuda)
            .expect("strict train without checkpointing should pass");

        let mut with_checkpointing_config = config.clone();
        with_checkpointing_config.gradient_checkpointing = Some(GradientCheckpointingConfig {
            interval_nodes: 2,
            min_tensor_bytes: 16,
        });
        let with_checkpointing =
            train_with_backend(&model, &dataset, &with_checkpointing_config, &cuda)
                .expect("strict train with checkpointing should pass");

        assert_eq!(
            without_checkpointing.final_parameters,
            with_checkpointing.final_parameters
        );
        assert_eq!(
            without_checkpointing.optimizer_state,
            with_checkpointing.optimizer_state
        );
        assert_eq!(
            without_checkpointing.final_loss.to_bits(),
            with_checkpointing.final_loss.to_bits()
        );
    });
}
