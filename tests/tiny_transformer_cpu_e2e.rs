#[test]
#[should_panic(expected = "inference API is introduced in Task 3")]
fn tiny_transformer_cpu_train_infer_save_load_roundtrip() {
    let (model, dataset, train_config, infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();
    let trained = volta::model::train(&model, &dataset, &train_config).expect("train");
    let _before =
        volta::model::infer(&model, &trained.final_parameters, &infer_input).expect("infer");
}
