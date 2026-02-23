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

    #[test]
    fn sigmoid_outputs_are_bounded_and_finite(values in proptest::collection::vec(-80.0f32..80.0f32, 1..128)) {
        let input = Tensor::new(vec![values.len()], values)
            .expect("vector shape should match element count");
        let output = input.sigmoid().expect("sigmoid should succeed on finite inputs");
        prop_assert!(output.data.iter().all(|value| value.is_finite()));
        prop_assert!(output.data.iter().all(|value| *value >= 0.0 && *value <= 1.0));
    }

    #[test]
    fn softmax_outputs_sum_to_one(values in proptest::collection::vec(-30.0f32..30.0f32, 1..64)) {
        let input = Tensor::new(vec![values.len()], values)
            .expect("vector shape should match element count");
        let output = input.softmax().expect("softmax should succeed for rank-1 input");
        let sum: f32 = output.data.iter().copied().sum();
        prop_assert!((sum - 1.0).abs() <= 1e-4, "sum was {sum}");
        prop_assert!(output.data.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn broadcast_add_is_commutative_for_rank_extended_shapes(
        m in 1usize..8,
        n in 1usize..8,
        a in proptest::collection::vec(-100.0f32..100.0f32, 1..8),
        b in proptest::collection::vec(-100.0f32..100.0f32, 1..8)
    ) {
        let lhs_data: Vec<f32> = (0..m).map(|i| a[i % a.len()]).collect();
        let rhs_data: Vec<f32> = (0..n).map(|j| b[j % b.len()]).collect();

        let lhs = Tensor::new(vec![m, 1], lhs_data).expect("lhs tensor must be valid");
        let rhs = Tensor::new(vec![1, n], rhs_data).expect("rhs tensor must be valid");

        let out_lr = lhs
            .add_broadcast(&rhs)
            .expect("lhs + rhs broadcast should succeed");
        let out_rl = rhs
            .add_broadcast(&lhs)
            .expect("rhs + lhs broadcast should succeed");

        prop_assert_eq!(out_lr.shape, out_rl.shape);
        for (x, y) in out_lr.data.iter().zip(out_rl.data.iter()) {
            prop_assert!((x - y).abs() <= 1e-6);
        }
    }
}
