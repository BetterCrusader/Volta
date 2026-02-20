#[test]
fn governance_docs_exist() {
    let required = [
        "docs/governance/contracts-tier-a.md",
        "docs/governance/determinism-scope.md",
        "docs/governance/operational-policy.md",
        "docs/governance/owner-model.md",
        "docs/governance/rfc-template.md",
        "docs/governance/incident-playbook.md",
        "docs/governance/perf-governance.md",
        "docs/governance/ci-topology.md",
    ];

    for path in required {
        assert!(std::path::Path::new(path).exists(), "missing {path}");
    }
}
