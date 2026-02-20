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
}

#[test]
fn prd_mentions_wave1_foundation() {
    let text = read_text("PRD.md");
    assert!(
        text.contains("Wave 1"),
        "PRD must mention Wave 1 governance foundation"
    );
}
