use volta::model::{build_tiny_transformer_fixture_for_tests, export_compiled_model_manifest};

#[test]
fn model_export_manifest_contains_graph_and_parameter_metadata() {
    let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();
    let manifest = export_compiled_model_manifest(&model).expect("manifest export should pass");

    assert!(manifest.contains("\"output_shape\""));
    assert!(manifest.contains("\"parameters\""));
    assert!(manifest.contains("\"graph_fingerprint\""));
    assert!(manifest.contains("tiny.w_q"));
}
