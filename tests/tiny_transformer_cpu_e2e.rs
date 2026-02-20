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

    println!(
        "task4-metrics lr={} epochs={} initial_loss={} final_loss={}",
        match train_config.optimizer {
            volta::ir::OptimizerConfig::Sgd { lr } => lr,
            volta::ir::OptimizerConfig::Adam { lr, .. } => lr,
        },
        train_config.epochs,
        baseline.final_loss,
        trained.final_loss
    );

    assert_eq!(before.shape, model.output_shape.0);
    assert_eq!(after.shape, model.output_shape.0);
    assert!(
        trained.final_loss < baseline.final_loss,
        "expected train loss to decrease: initial={} final={}",
        baseline.final_loss,
        trained.final_loss
    );
}
