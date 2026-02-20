fn read_doc(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"))
}

#[test]
fn contracts_doc_has_required_sections() {
    let text = read_doc("docs/governance/contracts-tier-a.md");
    for required in [
        "## Tier A Invariants",
        "## Numeric Tolerance Table",
        "## Device Fallback Semantics",
    ] {
        assert!(text.contains(required), "missing section: {required}");
    }
}

#[test]
fn determinism_doc_has_scope_sections() {
    let text = read_doc("docs/governance/determinism-scope.md");
    for required in [
        "## Guaranteed Determinism",
        "## Non-Guaranteed Areas",
        "## Seed and Replay Policy",
    ] {
        assert!(text.contains(required), "missing section: {required}");
    }
}

#[test]
fn policy_docs_include_required_rules() {
    let op = read_doc("docs/governance/operational-policy.md");
    let owner = read_doc("docs/governance/owner-model.md");
    assert!(
        op.contains("rollback"),
        "operational policy must mention rollback"
    );
    assert!(
        op.contains("exp/*"),
        "operational policy must mention exp/* branch policy"
    );
    assert!(
        owner.contains("Tier A PRs require 2 independent approvals"),
        "owner model must define Tier A approval rule"
    );
}

#[test]
fn rfc_and_incident_docs_cover_failure_analysis() {
    let rfc = read_doc("docs/governance/rfc-template.md");
    let incident = read_doc("docs/governance/incident-playbook.md");
    assert!(
        rfc.contains("Failure Analysis"),
        "rfc template must include failure analysis"
    );
    assert!(
        incident.contains("Rollback"),
        "incident playbook must include rollback"
    );
    assert!(
        incident.contains("Severity"),
        "incident playbook must include severity"
    );
}
