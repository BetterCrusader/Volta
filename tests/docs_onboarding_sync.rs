use std::fs;
use std::path::Path;

#[test]
fn onboarding_docs_exist_and_are_linked_from_readme() {
    assert!(Path::new("CONTRIBUTING.md").exists());
    assert!(Path::new("scripts/release/cut_v1.ps1").exists());

    let readme = fs::read_to_string("README.md").expect("readme");
    assert!(readme.contains("CONTRIBUTING.md"));
    assert!(readme.contains("scripts/release/cut_v1.ps1"));
}
