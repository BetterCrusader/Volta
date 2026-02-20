use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;
use volta::ir::Tensor;

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    fn relu_never_outputs_negative(values in proptest::collection::vec(-1_000.0f32..1_000.0f32, 1..128)) {
        let input = Tensor::new(vec![values.len()], values)
            .expect("vector shape should match element count");
        let output = input.relu().expect("relu should be valid for rank-1 tensors");
        prop_assert!(output.data.iter().all(|value| *value >= 0.0));
    }
}
