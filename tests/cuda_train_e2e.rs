use volta::ir::CudaBackend;
use volta::model::{
    build_tiny_transformer_fixture_for_tests, infer_with_backend, train_with_backend,
};

#[test]
fn tiny_transformer_training_runs_with_cuda_backend() {
    let (model, dataset, train_config, infer_input) = build_tiny_transformer_fixture_for_tests();
    let cuda = CudaBackend;

    let trained = train_with_backend(&model, &dataset, &train_config, &cuda)
        .expect("tiny transformer training should run with cuda backend");
    assert!(trained.final_loss.is_finite());

    let out = infer_with_backend(&model, &trained.final_parameters, &infer_input, &cuda)
        .expect("tiny transformer infer should run with cuda backend");
    assert_eq!(out.shape, model.output_shape.0);
}
