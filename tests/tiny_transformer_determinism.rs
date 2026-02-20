use std::collections::{BTreeSet, HashMap};

fn max_parameter_abs_diff(
    left: &HashMap<String, volta::ir::Tensor>,
    right: &HashMap<String, volta::ir::Tensor>,
) -> (f32, usize) {
    let left_keys = left.keys().cloned().collect::<BTreeSet<_>>();
    let right_keys = right.keys().cloned().collect::<BTreeSet<_>>();
    assert_eq!(left_keys, right_keys, "parameter key sets must match");

    let mut max_diff = 0.0_f32;
    let mut compared_values = 0usize;

    for key in &left_keys {
        let left_tensor = left.get(key).expect("left tensor exists for key");
        let right_tensor = right.get(key).expect("right tensor exists for key");

        assert_eq!(
            left_tensor.shape, right_tensor.shape,
            "shape mismatch for parameter {}",
            key
        );

        for (a, b) in left_tensor.data.iter().zip(right_tensor.data.iter()) {
            let diff = (*a - *b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            compared_values += 1;
        }
    }

    (max_diff, compared_values)
}

#[test]
fn tiny_transformer_training_is_repeatable_for_fixed_seed() {
    let (model, dataset, mut cfg, _infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();
    cfg.reproducibility = volta::model::ReproducibilityMode::Deterministic;
    cfg.shuffle = true;
    cfg.shuffle_seed = 19;
    cfg.checkpoint_path = None;

    let first = volta::model::train(&model, &dataset, &cfg).expect("first train should pass");
    let second = volta::model::train(&model, &dataset, &cfg).expect("second train should pass");

    let loss_diff = (first.final_loss - second.final_loss).abs();
    let (param_diff, compared_values) =
        max_parameter_abs_diff(&first.final_parameters, &second.final_parameters);

    assert!(loss_diff <= 1e-9, "loss diff too high: {}", loss_diff);
    assert!(compared_values > 0, "no parameter values compared");
    assert!(
        param_diff <= 1e-9,
        "parameter diff too high: {}",
        param_diff
    );
}
