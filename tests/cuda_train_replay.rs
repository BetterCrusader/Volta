#[path = "common/cuda.rs"]
mod cuda_helpers;

use volta::ir::CudaBackend;
use volta::model::{build_tiny_transformer_fixture_for_tests, train_with_backend};

#[test]
fn strict_cuda_training_replay_is_bitwise_stable() {
    if !cuda_helpers::cuda_runtime_available() {
        eprintln!("[SKIP] strict_cuda_training_replay_is_bitwise_stable — no CUDA device available");
        return;
    }
    cuda_helpers::with_determinism("strict", || {
        let (model, dataset, train_config, _infer_input) =
            build_tiny_transformer_fixture_for_tests();
        let cuda = CudaBackend;

        let first = train_with_backend(&model, &dataset, &train_config, &cuda)
            .expect("first strict cuda train run should pass");
        let second = train_with_backend(&model, &dataset, &train_config, &cuda)
            .expect("second strict cuda train run should pass");

        assert_eq!(
            first.final_parameters, second.final_parameters,
            "strict cuda replay must be bitwise stable"
        );
        assert_eq!(first.final_loss.to_bits(), second.final_loss.to_bits());
    });
}
