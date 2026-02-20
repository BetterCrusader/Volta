fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"))
}

#[test]
fn cuda_training_determinism_policy_doc_exists_and_has_required_sections() {
    let text = read_text("docs/governance/cuda-training-determinism.md");
    assert!(text.contains("## Strict Mode Guarantees"));
    assert!(text.contains("reduction topology"));
    assert!(text.contains("optimizer state"));
    assert!(text.contains("no silent CPU fallback"));
}

#[test]
fn cuda_train_replay_baseline_exists_and_contains_expected_keys() {
    let text = read_text("benchmarks/baselines/cuda-train-replay.json");
    assert!(text.contains("\"determinism_level\""));
    assert!(text.contains("\"optimizer\""));
    assert!(text.contains("\"final_loss_bits\""));
    assert!(text.contains("\"final_parameters\""));
}
