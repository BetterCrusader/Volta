#[test]
fn repo_policy_files_exist() {
    assert!(
        std::path::Path::new(".github/CODEOWNERS").exists(),
        "missing .github/CODEOWNERS"
    );
    assert!(
        std::path::Path::new(".github/pull_request_template.md").exists(),
        "missing .github/pull_request_template.md"
    );
}
