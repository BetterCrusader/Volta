#[test]
fn tiny_transformer_cpu_train_infer_save_load_roundtrip() {
    let (model, dataset, train_config, infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();
    let trained = volta::model::train(&model, &dataset, &train_config).expect("train");
    let before =
        volta::model::infer(&model, &trained.final_parameters, &infer_input).expect("infer");
    assert_eq!(before.shape, model.output_shape.0);
}
