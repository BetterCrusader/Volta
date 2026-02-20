use std::time::{SystemTime, UNIX_EPOCH};

fn temp_checkpoint_path(label: &str) -> String {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock must be after unix epoch")
        .as_nanos();
    std::env::temp_dir()
        .join(format!("volta-task6-{label}-{nonce}.txt"))
        .to_string_lossy()
        .into_owned()
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "diff inputs must have equal length"
    );
    let mut max_diff = 0.0_f32;
    for (a, b) in left.iter().zip(right.iter()) {
        let diff = (*a - *b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

#[test]
fn tiny_transformer_cpu_train_infer_save_load_roundtrip() {
    let (model, dataset, train_config, infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();

    let baseline_config = volta::model::TrainApiConfig {
        epochs: 0,
        ..train_config.clone()
    };
    let baseline = volta::model::train(&model, &dataset, &baseline_config).expect("baseline train");
    assert!(
        baseline.final_loss.is_finite(),
        "baseline loss must be finite"
    );

    let before =
        volta::model::infer(&model, &model.parameters, &infer_input).expect("infer before train");

    let trained = volta::model::train(&model, &dataset, &train_config).expect("train");
    assert!(
        trained.final_loss.is_finite(),
        "trained loss must be finite"
    );

    let after = volta::model::infer(&model, &trained.final_parameters, &infer_input)
        .expect("infer after train");

    let checkpoint_path = temp_checkpoint_path("parity");
    volta::model::save_checkpoint(&checkpoint_path, &trained.final_parameters)
        .expect("save checkpoint");
    let reloaded = volta::model::load_checkpoint(&checkpoint_path).expect("load checkpoint");
    let after_reload =
        volta::model::infer(&model, &reloaded, &infer_input).expect("infer after reload");

    let epsilon = 1e-6_f32;
    let parity_diff = max_abs_diff(&after_reload.data, &after.data);

    println!(
        "task4-metrics lr={} epochs={} initial_loss={} final_loss={} epsilon={} max_abs_diff={}",
        match train_config.optimizer {
            volta::ir::OptimizerConfig::Sgd { lr } => lr,
            volta::ir::OptimizerConfig::Adam { lr, .. } => lr,
        },
        train_config.epochs,
        baseline.final_loss,
        trained.final_loss,
        epsilon,
        parity_diff
    );

    assert_eq!(before.shape, model.output_shape.0);
    assert_eq!(after.shape, model.output_shape.0);
    assert_eq!(after_reload.shape, model.output_shape.0);
    assert!(
        parity_diff <= epsilon,
        "expected parity diff <= epsilon, got diff={} epsilon={}",
        parity_diff,
        epsilon
    );
    assert!(
        trained.final_loss < baseline.final_loss,
        "expected train loss to decrease: initial={} final={}",
        baseline.final_loss,
        trained.final_loss
    );
}
