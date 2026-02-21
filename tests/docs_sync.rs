fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"))
}

#[test]
fn readme_mentions_quality_fortress_wave1() {
    let text = read_text("README.md");
    assert!(
        text.contains("Quality Fortress"),
        "README must mention Quality Fortress governance"
    );
    assert!(
        text.contains("Quality Fortress Wave 2/3 checks"),
        "README must include Wave 2/3 verification section"
    );
    assert!(
        text.contains("tiny-transformer CPU milestone"),
        "README must mention tiny-transformer CPU milestone"
    );
    assert!(
        text.contains("deterministic replay gate"),
        "README must describe deterministic replay gate"
    );
    assert!(
        text.contains("CUDA inference MVP"),
        "README must mention CUDA inference MVP"
    );
    assert!(
        text.contains("scripts/ci/cuda_infer_verify.sh"),
        "README must include CUDA inference verify script"
    );
    assert!(
        text.contains("scripts/ci/cuda_train_verify.sh"),
        "README must include CUDA training verify script"
    );
    assert!(
        text.contains("scripts/ci/xl_verify.sh"),
        "README must include XL verify script"
    );
}

#[test]
fn design_doc_mentions_wave1_foundation() {
    let text = read_text("docs/plans/2026-02-21-volta-roadmap-v3.md");
    assert!(
        text.contains("Volta is not a blank project"),
        "roadmap must document the current baseline explicitly"
    );
}

#[test]
fn wave23_plan_mentions_release_and_nightly() {
    let text = read_text("docs/plans/2026-02-21-volta-roadmap-v3.md");
    assert!(
        text.contains("ONNX Interop Wave 2"),
        "roadmap must include ONNX interop wave 2 scope"
    );
    assert!(
        text.contains("70% Core Fortress"),
        "roadmap must include 70/20/10 execution split"
    );
    assert!(
        text.contains("release-gates.yml"),
        "roadmap must reference release gate workflow"
    );
}

#[test]
fn cuda_determinism_policy_is_documented() {
    let text = read_text("docs/governance/cuda-determinism-policy.md");
    assert!(
        text.contains("strict"),
        "cuda determinism policy must document strict mode"
    );
    assert!(
        text.contains("no atomics"),
        "cuda determinism policy must ban atomics in strict mode"
    );
    assert!(
        text.contains("TF32"),
        "cuda determinism policy must describe TF32 policy"
    );
    assert!(
        text.contains("fast-math"),
        "cuda determinism policy must describe fast-math policy"
    );
}

#[test]
fn workflows_include_xl_verify_lanes() {
    let pr = read_text(".github/workflows/pr-gates.yml");
    let release = read_text(".github/workflows/release-gates.yml");
    let nightly = read_text(".github/workflows/nightly-quality.yml");

    assert!(
        pr.contains("scripts/ci/xl_verify.sh"),
        "pr-gates must include xl verify lane"
    );
    assert!(
        release.contains("scripts/ci/xl_verify.sh"),
        "release-gates must include xl verify lane"
    );
    assert!(
        nightly.contains("scripts/ci/xl_verify.sh"),
        "nightly-quality must include xl verify lane"
    );
}
