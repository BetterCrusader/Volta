# Volta Quality Fortress Wave 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver Wave 1 governance foundation with test-enforced docs, tier-aware PR gates, policy checks, and local verification scripts.

**Architecture:** Add governance artifacts first, enforce them with tests, then wire CI automation (`detect_tiers`, `policy_check`, PR gates, CLI smoke, property-fast).

**Tech Stack:** Rust tests, Python scripts, GitHub Actions, shell/PowerShell CI helpers.

---

### Task 1: Governance document presence guard
- Files: `docs/governance/*`, `tests/governance_docs_presence.rs`
- Red: add test expecting required governance docs
- Green: create required governance docs
- Verify: `cargo test governance_docs_exist -- --nocapture`

### Task 2: Tier A contract content guard
- Files: `docs/governance/contracts-tier-a.md`, `tests/governance_docs_content.rs`
- Red: assert required sections exist
- Green: define invariants/tolerances/fallback sections
- Verify: `cargo test contracts_doc_has_required_sections -- --nocapture`

### Task 3: Determinism scope content guard
- Files: `docs/governance/determinism-scope.md`, `tests/governance_docs_content.rs`
- Red: require deterministic/non-deterministic/scope sections
- Green: define guaranteed vs non-guaranteed domains
- Verify: `cargo test determinism_doc_has_scope_sections -- --nocapture`

### Task 4: Operational and owner policy guard
- Files: `docs/governance/operational-policy.md`, `docs/governance/owner-model.md`, `tests/governance_docs_content.rs`
- Red: require rollback, `exp/*`, and Tier A approval rules
- Green: fill branch, rollback, accountability policy
- Verify: `cargo test policy_docs_include_required_rules -- --nocapture`

### Task 5: RFC and incident governance guard
- Files: `docs/governance/rfc-template.md`, `docs/governance/incident-playbook.md`, `tests/governance_docs_content.rs`
- Red: require Failure Analysis / Severity / Rollback content
- Green: finalize RFC and incident playbook templates
- Verify: `cargo test rfc_and_incident_docs_cover_failure_analysis -- --nocapture`

### Task 6: Ownership and PR metadata files
- Files: `.github/CODEOWNERS`, `.github/pull_request_template.md`, `tests/repo_policy_presence.rs`
- Red: assert files exist
- Green: add Tier A/B/C mapping and PR quality checklist
- Verify: `cargo test repo_policy_files_exist -- --nocapture`

### Task 7: Tier detection script + tests
- Files: `scripts/ci/detect_tiers.py`, `scripts/ci/tests/test_detect_tiers.py`
- Red: unit tests for A/B/C and precedence
- Green: implement mapping and CLI output (`tier=`, `changed=`)
- Verify: `python -m unittest scripts.ci.tests.test_detect_tiers -v`

### Task 8: Cross-platform CLI smoke scripts
- Files: `scripts/ci/cli_smoke.sh`, `scripts/ci/cli_smoke.ps1`, `examples/mnist.vt`
- Red: run script before creation (fails)
- Green: implement smoke commands and failure surfacing
- Verify: `bash scripts/ci/cli_smoke.sh`

### Task 9: Property-fast skeleton
- Files: `Cargo.toml`, `Cargo.lock`, `tests/property_fast.rs`
- Red: property-fast test fails without dev dependency
- Green: add `proptest` and Tier A invariant property
- Verify: `cargo test --test property_fast -- --nocapture`

### Task 10: PR gate workflow
- Files: `.github/workflows/pr-gates.yml`, `scripts/ci/wave1_local_verify.sh`
- Red: no dedicated PR gate workflow
- Green: add fmt/clippy/tests/release/cli-smoke/property-fast jobs + tier detection
- Verify: `bash scripts/ci/wave1_local_verify.sh`

### Task 11: Policy checker and workflow
- Files: `scripts/ci/policy_check.py`, `scripts/ci/tests/test_policy_check.py`, `.github/workflows/policy.yml`
- Red: missing policy automation
- Green: enforce RFC marker for governance files and `exp/*` hardening label
- Verify:
  - `python -m unittest scripts.ci.tests.test_policy_check -v`
  - `python scripts/ci/policy_check.py --paths docs/governance/contracts-tier-a.md --pr-body ""` (expect fail)

### Task 12: Docs synchronization and final verification
- Files: `README.md`, `PRD.md`, `tests/docs_sync.rs`
- Red: docs sync tests fail
- Green: document Wave 1 and Quality Fortress policy entry points
- Verify:
  - `cargo test --test docs_sync -- --nocapture`
  - `bash scripts/ci/wave1_local_verify.sh`
  - `python -m unittest scripts.ci.tests.test_detect_tiers -v`
  - `python -m unittest scripts.ci.tests.test_policy_check -v`

## Exit Criteria

- Wave 1 artifacts are in-repo and test-enforced
- PR gates are defined with tier-aware detection
- Policy automation is test-covered
- Local verification script reproduces CI-grade checks
