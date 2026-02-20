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
}

#[test]
fn design_doc_mentions_wave1_foundation() {
    let text = read_text("docs/plans/2026-02-20-volta-quality-fortress-design.md");
    assert!(
        text.contains("Wave 1"),
        "design doc must mention Wave 1 governance foundation"
    );
}

#[test]
fn wave23_plan_mentions_release_and_nightly() {
    let text = read_text("docs/plans/2026-02-20-volta-quality-fortress-wave23-implementation.md");
    assert!(text.contains("Wave 2"), "wave23 plan must mention Wave 2");
    assert!(text.contains("Wave 3"), "wave23 plan must mention Wave 3");
}
