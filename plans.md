# Volta 1.2.0 Hyper-Detailed Execution Plan

Status: Active Draft
Owner: Volta maintainers
Release target: `release-v1.2.0`
Branch policy: `main` remains releasable
Planning horizon: 8 weeks
Last updated: 2026-02-23

## Execution Log (2026-02-23)

Completed this cycle:

- Milestone `1.2.0` created in GitHub.
- Initial issue pack created: `#23`, `#24`, `#27`, `#32`, `#33`, `#34`, `#35`, `#36`, `#37`, `#38`.
- `#24` completed: PR gates now include blocking `test-onnx-import` lane (`cargo test --features onnx-import`).
- CI reliability governance docs landed:
  - `docs/governance/ci-flaky-registry.md`
  - `docs/governance/ci-weekly-health-template.md`
  - `docs/governance/ci-stale-reference-sweep-2026-02-23.md`
  - `docs/governance/ci-topology.md` updates
- Determinism evidence tracking landed:
  - `docs/governance/determinism-evidence-1.2.0.md`
  - `docs/governance/evidence-2026-02-23-local-verification.md`
- Docs-contract audit ledger started:
  - `docs/governance/docs-contract-audit-1.2.0.md`
- ONNX operator intake contract started:
  - `docs/governance/onnx-wave2-5-intake.md`

In progress:

- `#33` determinism evidence completion (needs additional week-level stability artifacts).
- `#38` CI reliability completion (needs one-week stable blocker-lane signal).
- `#37` docs-contract final pass (pending ONNX/CUDA final evidence rows).
- `#27` ONNX implementation batch work (intake contract done, operator implementation pending).

Instruction mode:

- Work is being accumulated without final merge in this phase.

## Table of Contents

1. Program Charter
2. Reality Check and Baseline
3. Operating Principles and Guardrails
4. Scope and Non-Scope
5. Success Metrics and Release KPIs
6. Workstream Architecture
7. Detailed Workstream Plans (A-H)
8. Weekly Milestone Plan (Week 1 to Week 8)
9. Detailed Next Steps (24h to 60d)
10. Verification and Quality Matrix
11. Risk Register and Mitigations
12. Governance, RFC, and Policy Process
13. Issue Taxonomy and Backlog Model
14. RACI Ownership Model
15. Reporting Cadence and Templates
16. Release Readiness Gates
17. RC, Tagging, and Post-Release Runbook
18. Appendices (command sets, checklists, templates)

## 1) Program Charter

### 1.1 Core mission

Deliver Volta 1.2.0 as a reliability-forward capability expansion release.

The release must improve practical model support and runtime robustness while preserving deterministic-first behavior in all strict execution paths.

### 1.2 Hard outcomes

- Expand supported ONNX operator surface in a controlled, test-backed way.
- Increase confidence in CUDA strict-lane behavior for currently supported operation families.
- Improve deterministic reproducibility confidence through stronger regression evidence.
- Keep policy and docs claims fully synchronized with executable behavior.

### 1.3 Why this release matters

Volta already has strong deterministic infrastructure.
1.2.0 converts this from a hardened base into a repeatable delivery model where capability can grow without quality collapse.

### 1.4 Release constraints

- No silent fallback in strict mode.
- No scope creep that weakens CI reliability.
- No docs claims without executable proof.
- No last-minute architecture rewrites in freeze window.

## 2) Reality Check and Baseline

### 2.1 What is already true

- Deterministic schedule and fingerprint checks exist.
- Governance docs and policy checks are active.
- PR, release, and nightly workflows are in place.
- ONNX support is partial but explicit.
- CUDA support is partial but fail-fast controlled.

### 2.2 What is currently risky

- Coverage expansion can outrun quality evidence.
- CI flakiness can hide true regressions.
- Documentation can drift when many parallel changes land.
- Performance baselines can drift and mask regressions.

### 2.3 Baseline assumptions

Assumption A1: deterministic-first policy remains non-negotiable.
Assumption A2: release quality gates remain blocking.
Assumption A3: unsupported features stay explicit.
Assumption A4: there is no tolerance for hidden fallback behavior.

### 2.4 Baseline references

- `README.md`
- `plan.md`
- `docs/ONNX_COVERAGE.md`
- `docs/governance/QUALITY_POLICY.md`
- `docs/governance/determinism-scope.md`
- `docs/governance/cuda-training-determinism.md`
- `.github/workflows/pr-gates.yml`
- `.github/workflows/release-gates.yml`
- `.github/workflows/nightly-quality.yml`

## 3) Operating Principles and Guardrails

### 3.1 Principles

1. Determinism is a contract.
2. Correctness beats speed when they conflict.
3. Explicit unsupported is better than implicit wrong behavior.
4. Governance must be executable, not aspirational.
5. CI is part of product quality, not separate from it.
6. Documentation is contract surface.
7. Every release must be reproducible.

### 3.2 Guardrails

G1: no strict-mode silent fallback.
G2: no new support claims without tests.
G3: no policy-sensitive merge without traceability.
G4: no RC cut with open P0 blockers.
G5: no release tag before rollback verify signal.

### 3.3 Enforcement rules

- Block merge on failing mandatory gates.
- Block release on failing release gates.
- Escalate nightly regressions to tracked issues.
- Require ownership for every red lane.

## 4) Scope and Non-Scope

### 4.1 In scope

- ONNX Wave 2.5 static-safe operator growth.
- CUDA strict-lane hardening for supported operation families.
- Autograd correctness expansion for high-impact patterns.
- Numerical stability policy enforcement depth.
- CI reliability and stale-reference cleanup.
- Docs and policy synchronization hardening.

### 4.2 Out of scope

- Full ONNX breadth.
- Dynamic-shape control-flow import.
- Hidden fallback behavior.
- Breaking syntax redesign.
- Throughput-only hacks that reduce determinism guarantees.

### 4.3 Scope freeze policy

- New in-scope work allowed until end of Week 4.
- After Week 4, only P0/P1 bugfix and stabilization.
- After Week 6, only release blockers and documentation correctness.

## 5) Success Metrics and Release KPIs

### 5.1 KPI set (must pass)

KPI-01: all mandatory gates green on release candidate commit.
KPI-02: `cargo test --features onnx-import` blocking and green.
KPI-03: minimum 4 new ONNX operator paths landed end-to-end.
KPI-04: zero known strict silent-fallback behavior.
KPI-05: determinism regressions stable under repeated and threaded runs.
KPI-06: perf SLO checks pass or approved baseline updates exist.
KPI-07: docs-contract audit finds zero mismatch.
KPI-08: nightly has 7 clean cycles or blockers are triaged with owner + ETA.
KPI-09: release rollback verify is green.
KPI-10: changelog claims match shipped behavior exactly.

### 5.2 KPI evidence format

- command output snapshot
- workflow URL
- artifact reference path
- owner sign-off
- timestamp

## 6) Workstream Architecture

### 6.1 Workstream map

- WS-A: Deterministic Core
- WS-B: ONNX Expansion
- WS-C: CUDA Strict Hardening
- WS-D: Autograd Correctness
- WS-E: Numerical Stability
- WS-F: Performance and Cache Governance
- WS-G: CI and Pipeline Reliability
- WS-H: Docs and Governance Sync

### 6.2 Workstream sequencing

Phase 1 (Weeks 1-2): WS-G + WS-A foundations.
Phase 2 (Weeks 3-4): WS-B expansion batches.
Phase 3 (Weeks 5-6): WS-C + WS-D + WS-E correctness hardening.
Phase 4 (Weeks 7-8): WS-F + WS-H freeze and release closure.

### 6.3 Cross-stream dependencies

Dependency D1: WS-B depends on WS-A deterministic guarantees.
Dependency D2: WS-C depends on WS-E stability policy tests.
Dependency D3: WS-H depends on WS-B/WS-C implementation status.
Dependency D4: WS-F depends on WS-G artifact quality.

## 7) Detailed Workstream Plans

## WS-A: Deterministic Core

### Mission

Ensure graph transformation and schedule generation remain stable and reproducible under stress.

### Deliverables

- A-DEL-01 deterministic iteration audit report
- A-DEL-02 first-divergence diagnostics helpers
- A-DEL-03 expanded deterministic stress tests
- A-DEL-04 schedule/fingerprint stability evidence set

### Epic backlog

A-EPIC-01: pass iteration audit.
A-EPIC-02: deterministic container normalization.
A-EPIC-03: divergence snapshots and triage support.
A-EPIC-04: repeated-run and threaded-run matrix expansion.

### Tasks

- A-T01 audit pass internals for map/set iteration order sensitivity.
- A-T02 catalog order-sensitive code paths and owners.
- A-T03 replace unstable iteration with explicit ordering where needed.
- A-T04 add pass-level snapshot capture utility.
- A-T05 add first divergent pass comparator helper.
- A-T06 add 100-run deterministic schedule test for representative graphs.
- A-T07 add threaded schedule determinism test expansion.
- A-T08 expand fingerprint stability tests for feature combinations.
- A-T09 document deterministic assumptions per pass.
- A-T10 link evidence in weekly report.

### Acceptance criteria

- A-AC-01 all determinism tests are green in debug and release.
- A-AC-02 no flaky failures in 3 consecutive reruns.
- A-AC-03 first-divergence tooling can isolate first mismatch.

### Risks

- hidden iteration in newly added code paths
- unstable fused ordering behavior

### Mitigations

- pre-merge deterministic audit checklist
- mandatory regression run before merge for WS-A tagged PRs

## WS-B: ONNX Expansion

### Mission

Increase practical ONNX import capability without breaking explicit contract boundaries.

### Target operators (candidate)

- `LeakyRelu`
- `BatchNorm` inference profile
- `MaxPool` static subset
- `AveragePool` static subset
- `LayerNorm` static-safe profile
- additional low-risk op by Week 4 if budget allows

### Deliverables

- B-DEL-01 operator contract definitions
- B-DEL-02 parser + importer + lowering implementation
- B-DEL-03 runtime support or explicit fail-fast wiring
- B-DEL-04 fixtures + edge tests + coverage doc updates

### Epic backlog

B-EPIC-01 operator intake and contract lock.
B-EPIC-02 batch-1 implementation and validation.
B-EPIC-03 batch-2 implementation and validation.
B-EPIC-04 docs coverage synchronization.

### Tasks

- B-T01 define static constraints for each candidate operator.
- B-T02 classify unsupported attributes and error text requirements.
- B-T03 implement parser path for operator #1.
- B-T04 implement contract checks for operator #1.
- B-T05 implement lowering path for operator #1.
- B-T06 implement runtime path or explicit fail-fast for operator #1.
- B-T07 add importer tests for operator #1 valid and invalid forms.
- B-T08 repeat B-T03..B-T07 for operator #2.
- B-T09 repeat B-T03..B-T07 for operator #3.
- B-T10 repeat B-T03..B-T07 for operator #4.
- B-T11 add real fixture combination tests.
- B-T12 update `docs/ONNX_COVERAGE.md` exactly as shipped.

### Acceptance criteria

- B-AC-01 minimum 4 operators fully landed with tests and docs.
- B-AC-02 unsupported attrs fail explicitly with useful diagnostics.
- B-AC-03 no aspirational support claims in docs.

### Risks

- operator complexity expands beyond static-safe boundary
- importer behavior diverges from runtime assumptions

### Mitigations

- strict intake rubric
- operator-by-operator go/no-go gates

## WS-C: CUDA Strict Hardening

### Mission

Raise strict-lane confidence for supported CUDA paths.

### Deliverables

- C-DEL-01 strict replay regression expansion
- C-DEL-02 no-fallback proof set expansion
- C-DEL-03 unsupported-kernel diagnostics improvement
- C-DEL-04 deterministic allocation stress evidence

### Epic backlog

C-EPIC-01 strict replay matrix.
C-EPIC-02 no-fallback regression suite.
C-EPIC-03 error quality and diagnostics.
C-EPIC-04 allocation determinism checks.

### Tasks

- C-T01 add strict replay test for representative inference graph A.
- C-T02 add strict replay test for representative training graph B.
- C-T03 add strict replay test for mixed reduction graph C.
- C-T04 add no-fallback assertion tests for unsupported graph classes.
- C-T05 verify explicit failure messages include op/class context.
- C-T06 add parity tests for supported op families under strict mode.
- C-T07 add memory guard regression case for placement drift.
- C-T08 add allocation ordering stability checks.
- C-T09 ensure failures remain deterministic and reproducible.
- C-T10 document strict vs balanced expectations in governance docs.

### Acceptance criteria

- C-AC-01 strict lane tests are consistently green.
- C-AC-02 no silent fallback observed in strict mode.
- C-AC-03 unsupported behavior emits explicit actionable errors.

## WS-D: Autograd Correctness

### Mission

Strengthen backward correctness and deterministic optimizer behavior.

### Deliverables

- D-DEL-01 expanded gradcheck matrix
- D-DEL-02 forward->backward->step integration suite
- D-DEL-03 unsupported backward path policy alignment

### Epic backlog

D-EPIC-01 gradcheck expansion.
D-EPIC-02 integration training path reliability.
D-EPIC-03 unsupported backward policy clarity.

### Tasks

- D-T01 add gradcheck for broadcast-heavy combinations.
- D-T02 add gradcheck for reduction-heavy combinations.
- D-T03 add gradcheck for new ONNX operator interactions.
- D-T04 add single-step deterministic optimizer tests.
- D-T05 add multi-step deterministic optimizer tests.
- D-T06 add finite-gradient and shape-correctness invariants.
- D-T07 enforce explicit non-support tests for unsupported backward ops.
- D-T08 align docs with backward support matrix.

### Acceptance criteria

- D-AC-01 gradcheck suite passes and is stable.
- D-AC-02 integration training loop tests remain deterministic.
- D-AC-03 unsupported backward behavior is explicit and tested.

## WS-E: Numerical Stability

### Mission

Convert numerical policy into executable evidence.

### Deliverables

- E-DEL-01 NaN/Inf/extreme-value matrix
- E-DEL-02 CPU/CUDA parity behavior checks
- E-DEL-03 tolerance policy alignment by op class

### Epic backlog

E-EPIC-01 policy-to-test mapping.
E-EPIC-02 edge-case expansion.
E-EPIC-03 tolerance and exception governance.

### Tasks

- E-T01 map each policy clause to one or more tests.
- E-T02 add exp/log extreme range tests.
- E-T03 add softmax overflow/underflow tests.
- E-T04 add reduction NaN/Inf propagation tests.
- E-T05 add matmul/gemm NaN/Inf behavior tests.
- E-T06 add tolerance checks for CUDA parity where relevant.
- E-T07 codify policy exceptions with explicit rationale.
- E-T08 enforce exception tests for any deviation.

### Acceptance criteria

- E-AC-01 policy-linked tests are green.
- E-AC-02 NaN/Inf behavior matches docs.
- E-AC-03 exceptions are explicit and validated.

## WS-F: Performance and Cache Governance

### Mission

Preserve performance predictability while keeping correctness first.

### Deliverables

- F-DEL-01 improved perf-gate artifact clarity
- F-DEL-02 cache behavior diagnostics
- F-DEL-03 stable release double-pass perf report

### Epic backlog

F-EPIC-01 baseline governance.
F-EPIC-02 cache observability.
F-EPIC-03 release perf sign-off quality.

### Tasks

- F-T01 audit baseline files and update policy references.
- F-T02 improve perf artifact naming and retention consistency.
- F-T03 add cache hit/miss reporting for repeated runs.
- F-T04 add fallback baseline resolution diagnostics.
- F-T05 validate double-pass perf report interpretability.
- F-T06 document baseline update approval process.

### Acceptance criteria

- F-AC-01 perf regressions are visible and attributable.
- F-AC-02 baseline changes are explicit and reviewed.
- F-AC-03 release perf evidence is easy to audit.

## WS-G: CI and Pipeline Reliability

### Mission

Improve CI signal quality and reduce false blocker cost.

### Deliverables

- G-DEL-01 onnx-import lane blocking in PR gates
- G-DEL-02 stale-reference cleanup
- G-DEL-03 flake ownership model
- G-DEL-04 weekly CI health summary process

### Epic backlog

G-EPIC-01 gate completeness.
G-EPIC-02 stale command eradication.
G-EPIC-03 flake tracking and ownership.
G-EPIC-04 CI health governance.

### Tasks

- G-T01 add/verify onnx-import lane in PR workflow.
- G-T02 grep and remove stale command references.
- G-T03 create flaky test registry with owners.
- G-T04 define escalation path for recurring flakes.
- G-T05 add weekly CI health report checklist.
- G-T06 verify release and nightly parity with local scripts.
- G-T07 ensure wave1 scripts use available python launcher robustly.
- G-T08 validate docs command examples are executable.

### Acceptance criteria

- G-AC-01 no stale CI command references remain.
- G-AC-02 blocking lanes have clear owners.
- G-AC-03 one-week stable signal window achieved.

## WS-H: Docs and Governance Sync

### Mission

Keep all public and governance claims truthful and synchronized.

### Deliverables

- H-DEL-01 docs-contract audit report
- H-DEL-02 coverage matrix synchronization
- H-DEL-03 release checklist and contributor guidance updates

### Epic backlog

H-EPIC-01 docs claim audit.
H-EPIC-02 governance consistency updates.
H-EPIC-03 release communication correctness.

### Tasks

- H-T01 run claim-by-claim audit before RC.
- H-T02 align README commands with actual scripts.
- H-T03 align ONNX coverage with exact shipped behavior.
- H-T04 update governance docs for strict mode boundaries.
- H-T05 update changelog entries with verifiable claims.
- H-T06 ensure release notes include known limits section.

### Acceptance criteria

- H-AC-01 docs-contract audit has zero mismatches.
- H-AC-02 release notes are factual and bounded.
- H-AC-03 all command examples execute as documented.

## 8) Weekly Milestone Plan

## Week 1 - Scope Lock and Planning Setup

Objectives:

- lock 1.2.0 scope boundaries
- assign owners for WS-A..WS-H
- create P0/P1/P2 issue board

Planned outputs:

- M1-O1 milestone and issue map
- M1-O2 acceptance criteria per P0 issue
- M1-O3 release checklist draft

Exit criteria:

- all P0 issues created with owners and ETAs
- no unclear ownership for blocker streams

## Week 2 - CI and Determinism Foundations

Objectives:

- complete CI lane upgrades
- finish stale command cleanup
- begin deterministic core audit

Planned outputs:

- M2-O1 onnx-import lane blocking
- M2-O2 stale reference cleanup report
- M2-O3 first deterministic audit findings

Exit criteria:

- CI remains green after upgrades

## Week 3 - ONNX Expansion Batch 1

Objectives:

- implement first operator batch end-to-end
- add tests and docs for shipped behavior

Planned outputs:

- M3-O1 operator #1 and #2 merged
- M3-O2 fixture tests for batch 1
- M3-O3 docs coverage updates

Exit criteria:

- batch-1 interop tests stable

## Week 4 - ONNX Expansion Batch 2

Objectives:

- implement second operator batch
- complete minimum operator count target

Planned outputs:

- M4-O1 operator #3 and #4 merged
- M4-O2 expanded edge-case tests
- M4-O3 mid-cycle readiness review

Exit criteria:

- minimum 4 operators landed end-to-end

## Week 5 - CUDA Strict Hardening Sprint

Objectives:

- strengthen replay/no-fallback evidence
- improve diagnostics for unsupported behavior

Planned outputs:

- M5-O1 strict replay test expansion
- M5-O2 no-fallback regression suite expansion
- M5-O3 diagnostics quality improvements

Exit criteria:

- strict lanes green without unresolved blockers

## Week 6 - Autograd and Stability Sprint

Objectives:

- complete gradcheck and integration expansion
- map numerical policy clauses to tests

Planned outputs:

- M6-O1 autograd suite enhancements
- M6-O2 numerical stability matrix initial completion
- M6-O3 policy exception catalog

Exit criteria:

- no unknown high-risk autograd/stability gaps

## Week 7 - Freeze and Hardening

Objectives:

- freeze feature scope
- close remaining blocker defects
- finalize docs/changelog truthfulness

Planned outputs:

- M7-O1 blocker bugfix merges
- M7-O2 docs-contract audit pass
- M7-O3 release note draft

Exit criteria:

- zero open P0 blockers

## Week 8 - RC and Release

Objectives:

- cut RC
- run full release verification
- publish release

Planned outputs:

- M8-O1 RC verification report
- M8-O2 release tag and release note
- M8-O3 post-release follow-up list

Exit criteria:

- all release gates pass
- rollback verify pass

## 9) Detailed Next Steps

## 9.1 Next 24 hours

1. Create/confirm `1.2.0` milestone.
2. Ensure P0 issue list exists and owners are assigned.
3. Verify `onnx-import` lane is blocking in PR gates.
4. Run stale reference grep in docs/scripts/workflows.
5. Publish a one-page status note for kickoff.

## 9.2 Next 72 hours

1. Merge CI stale-reference fixes.
2. Finalize operator intake list for batch 1.
3. Start WS-A deterministic audit and open findings issues.
4. Start WS-B operator #1 implementation.
5. Publish first risk register snapshot.

## 9.3 Next 7 days

1. Ship first ONNX operator end-to-end.
2. Expand deterministic stress tests.
3. Add first numerical stability matrix tests.
4. Build flake ownership table and triage SLA.
5. Confirm docs command examples run.

## 9.4 Next 14 days

1. Ship second ONNX operator batch.
2. Ship WS-G CI reliability baseline improvements.
3. Add strict CUDA replay/no-fallback test expansion.
4. Publish milestone health report with KPIs.

## 9.5 Next 30 days

1. Reach minimum 4 ONNX operators landed.
2. Complete deterministic audit fixes from WS-A findings.
3. Complete core numerical stability matrix.
4. Validate perf reporting and baseline governance quality.
5. Run midpoint release readiness review.

## 9.6 Next 60 days

1. Complete freeze checklist.
2. Close all P0 blockers.
3. Produce final RC evidence package.
4. Tag and publish 1.2.0.

## 10) Verification and Quality Matrix

### 10.1 Mandatory command set

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo test --features onnx-import
cargo check --manifest-path fuzz/Cargo.toml
```

### 10.2 Local lane replication

```bash
bash scripts/ci/wave1_local_verify.sh
bash scripts/ci/wave23_local_verify.sh
bash scripts/ci/cuda_infer_verify.sh
bash scripts/ci/cuda_train_verify.sh
bash scripts/ci/xl_verify.sh
bash scripts/ci/interop_onnx_verify.sh
bash scripts/ci/release_perf_double_pass.sh
bash scripts/ci/nightly_perf_matrix.sh
```

### 10.3 Required weekly evidence

- EV-01 latest PR gate status snapshot
- EV-02 latest release lane status snapshot
- EV-03 deterministic stress outputs
- EV-04 perf artifact links
- EV-05 docs-contract audit delta

### 10.4 Evidence retention

- keep all artifacts for at least the full 1.2.0 cycle
- keep final RC artifacts until next minor release

## 11) Risk Register

R-01 Scope creep in ONNX expansion.
R-02 Hidden nondeterministic iteration in new pass code.
R-03 CI flakiness causing false blockers.
R-04 Baseline drift masking perf regressions.
R-05 Docs drift from shipped behavior.
R-06 Late-cycle high-risk changes destabilizing RC.
R-07 Incomplete ownership for critical lanes.
R-08 Unclear unsupported behavior diagnostics.
R-09 Inadequate replay metadata for incident triage.
R-10 Feature velocity outrunning test depth.

### Mitigation matrix

M-01 strict operator intake and Week-4 scope freeze.
M-02 deterministic audit checklist and stress suites.
M-03 flaky registry with owner and SLA.
M-04 explicit baseline approval process.
M-05 docs-contract audit before RC.
M-06 freeze policy after Week 6.
M-07 RACI ownership review weekly.
M-08 diagnostics quality tests and review.
M-09 replay metadata checklist in failures.
M-10 test plan required in every feature PR.

## 12) Governance, RFC, and Policy Process

### 12.1 When RFC is required

- pass-order contract changes
- deterministic mode semantics changes
- governance-sensitive policy behavior changes
- release gate definition changes

### 12.2 Policy-sensitive PR checklist

- includes RFC reference or linked RFC document change
- includes deterministic impact note
- includes docs impact note
- includes rollback note

### 12.3 Incident handling

- every blocker regression gets a tracked issue
- issue includes reproduction metadata
- issue has owner and ETA

## 13) Issue Taxonomy and Backlog Model

### 13.1 Labels

- `release-1.2.0`
- `priority-p0`
- `priority-p1`
- `priority-p2`
- `ws-determinism`
- `ws-onnx`
- `ws-cuda`
- `ws-autograd`
- `ws-stability`
- `ws-perf`
- `ws-ci`
- `ws-docs`

### 13.2 Issue types

- Feature
- Bug
- Governance
- Performance
- Docs-contract
- CI reliability

### 13.3 Required issue fields

- scope
- acceptance criteria
- required tests
- governance impact
- rollback note
- owner
- ETA

### 13.4 Backlog policy

- P0 cannot be deferred without release owner decision.
- P1 can be deferred only after explicit risk assessment.
- P2 is auto-deferred when P0 risk appears.

## 14) RACI Ownership Model

### 14.1 Roles

- Release Owner
- Workstream Lead
- Reviewer
- CI Reliability Owner
- Docs Owner

### 14.2 RACI per stream

WS-A: R=Determinism lead, A=Release Owner, C=CI owner, I=Docs owner.
WS-B: R=Interop lead, A=Release Owner, C=Runtime lead, I=Docs owner.
WS-C: R=CUDA lead, A=Release Owner, C=Determinism lead, I=CI owner.
WS-D: R=Autograd lead, A=Release Owner, C=Runtime lead, I=Docs owner.
WS-E: R=Stability lead, A=Release Owner, C=CUDA lead, I=CI owner.
WS-F: R=Perf lead, A=Release Owner, C=CI owner, I=Docs owner.
WS-G: R=CI owner, A=Release Owner, C=All leads, I=All.
WS-H: R=Docs owner, A=Release Owner, C=All leads, I=All.

## 15) Reporting Cadence and Templates

### 15.1 Daily status template

- Yesterday:
- Today:
- Blockers:
- Needed help:
- Risk delta:

### 15.2 Weekly report template

- KPI status (green/yellow/red)
- P0 progress summary
- New risks and mitigations
- CI health summary
- docs-contract delta
- next week priorities

### 15.3 Mid-cycle review template

- shipped vs planned
- scope adjustments
- release risk outlook
- decision log

## 16) Release Readiness Gates

### 16.1 Gate A - Infrastructure readiness (end Week 2)

- CI lane completeness verified
- stale references removed
- deterministic audit started

### 16.2 Gate B - Capability midpoint (end Week 4)

- at least 4 operator paths in progress with evidence
- interop matrix updated for shipped changes

### 16.3 Gate C - Correctness and stability (end Week 6)

- strict replay/no-fallback evidence acceptable
- autograd/stability expansions landed

### 16.4 Gate D - Final release readiness (Week 8)

- all P0 closed
- all mandatory and release gates green
- docs-contract audit pass
- rollback verify pass

## 17) RC, Tagging, and Post-Release Runbook

### 17.1 Pre-RC checklist

- feature freeze enabled
- blocker bugfix only policy active
- full verification matrix run
- docs-contract audit run

### 17.2 RC cut steps

1. mark RC candidate commit
2. run full release gates
3. collect and archive artifacts
4. publish RC verification summary

### 17.3 Final release steps

1. create `release-v1.2.0` tag
2. run release workflows
3. publish release notes
4. verify rollback path

### 17.4 Post-release steps

1. capture lessons learned
2. triage carryover items to 1.2.1
3. update docs if post-release findings appear

## 18) Appendix A - P0 Checklist (Detailed)

- [ ] P0-01 onnx-import lane blocking and green
- [ ] P0-02 deterministic audit complete
- [ ] P0-03 100-run determinism stress green
- [ ] P0-04 threaded determinism stress green
- [ ] P0-05 operator #1 landed end-to-end
- [ ] P0-06 operator #2 landed end-to-end
- [ ] P0-07 operator #3 landed end-to-end
- [ ] P0-08 operator #4 landed end-to-end
- [ ] P0-09 strict no-fallback regression expansion landed
- [ ] P0-10 strict replay suite expansion landed
- [ ] P0-11 autograd expansion landed
- [ ] P0-12 numerical stability matrix v1 landed
- [ ] P0-13 perf double-pass gate green
- [ ] P0-14 docs-contract audit pass
- [ ] P0-15 rollback verify pass

## 19) Appendix B - Detailed 56-Day Task Track

Use this as granular execution guidance. Shift tasks if blockers appear, but keep dependencies intact.

### Day 01

- confirm milestone and owners
- confirm P0 list
- publish kickoff note

### Day 02

- create WS issue tree
- assign acceptance criteria
- define ETA per P0 issue

### Day 03

- verify onnx-import lane config
- run stale command grep
- open stale cleanup tasks

### Day 04

- merge stale command fixes
- verify wave scripts locally
- refresh docs command examples

### Day 05

- start deterministic pass audit
- log order-sensitive hotspots
- assign hotspot owners

### Day 06

- implement first deterministic fix set
- run determinism regressions
- log evidence artifacts

### Day 07

- weekly review #1
- update risks and ETAs
- confirm Week 2 plan

### Day 08

- operator intake review batch 1
- lock operator #1 and #2 contracts
- open implementation tasks

### Day 09

- implement parser/contract for operator #1
- add invalid-attr rejection tests
- run interop checks

### Day 10

- implement lowering/runtime for operator #1
- add fixture test for operator #1
- update coverage docs draft

### Day 11

- implement parser/contract for operator #2
- add edge tests for operator #2
- run integration checks

### Day 12

- implement lowering/runtime for operator #2
- complete docs updates for batch 1
- run full mandatory gates

### Day 13

- deterministic stress expansion pass
- compare baseline hashes
- investigate drift if any

### Day 14

- weekly review #2
- confirm batch 1 completion
- lock Week 3 objectives

### Day 15

- operator intake review batch 2
- lock operator #3 and #4 contracts
- schedule implementation order

### Day 16

- implement parser/contract for operator #3
- add tests for supported attrs
- add tests for unsupported attrs

### Day 17

- implement lowering/runtime for operator #3
- add fixtures and parity checks
- update docs draft

### Day 18

- implement parser/contract for operator #4
- add tests for invalid shape constraints
- run interop suites

### Day 19

- implement lowering/runtime for operator #4
- finalize batch 2 docs updates
- run mandatory command gates

### Day 20

- run midpoint interop review
- compare KPI progress vs plan
- adjust scope if needed

### Day 21

- weekly review #3
- freeze new operator intake
- move to strict hardening phase

### Day 22

- expand strict replay suite scenario set A
- add evidence artifacts
- assign failures immediately

### Day 23

- expand strict replay suite scenario set B
- add no-fallback assertions
- run CUDA verify lanes

### Day 24

- improve unsupported-kernel diagnostics texts
- add diagnostic quality tests
- run strict-lane checks

### Day 25

- add allocation determinism stress checks
- run repeated checks
- archive result snapshots

### Day 26

- expand no-fallback regressions for unsupported graphs
- validate explicit error classes
- update docs if behavior clarified

### Day 27

- run combined CUDA strict hardening regression pass
- triage any blockers
- patch blockers

### Day 28

- weekly review #4
- sign off strict-lane progress
- start autograd/stability sprint

### Day 29

- expand gradcheck set A
- run and triage failures
- patch low-risk fixes

### Day 30

- expand gradcheck set B
- add integration step tests
- validate deterministic optimizer ordering

### Day 31

- add numerical stability tests exp/log extremes
- add NaN/Inf propagation tests
- run policy-linked suites

### Day 32

- add softmax/reduction edge matrices
- run CPU/CUDA parity checks
- record tolerance results

### Day 33

- add gemm/matmul edge behavior tests
- verify policy exception handling
- update stability docs draft

### Day 34

- consolidate autograd + stability findings
- close or escalate blockers
- prepare freeze candidate list

### Day 35

- weekly review #5
- confirm no new high-risk scope
- prepare freeze plan

### Day 36

- start freeze week
- bug triage and prioritization
- blocker-only merge policy active

### Day 37

- resolve highest priority blocker
- re-run affected lanes
- attach evidence

### Day 38

- resolve second priority blocker
- validate no regression spillover
- update changelog draft

### Day 39

- run docs-contract audit pass 1
- file and fix mismatches
- verify command examples

### Day 40

- run perf report quality pass
- verify baseline references
- update approval logs

### Day 41

- run nightly readiness rehearsal
- inspect issue auto-update behavior
- tune flaky quarantine if needed

### Day 42

- weekly review #6
- confirm RC entry criteria trajectory
- lock final blocker list

### Day 43

- run full mandatory command matrix
- patch immediate blockers
- rerun until green

### Day 44

- run wave and cuda verify lanes
- patch lane-specific regressions
- rerun with artifacts

### Day 45

- run release perf double-pass rehearsal
- inspect regression deltas
- approve/update baseline if justified

### Day 46

- run docs-contract audit pass 2
- finalize README and coverage alignment
- finalize governance deltas

### Day 47

- prepare RC candidate commit
- verify freeze policy compliance
- gather release notes inputs

### Day 48

- weekly review #7
- decide RC go/no-go
- if go, proceed to RC cut

### Day 49

- cut RC candidate
- run full release gates
- archive all artifacts

### Day 50

- triage RC regressions
- patch blockers only
- rerun impacted suites

### Day 51

- finalize release notes draft
- include known limits section
- verify factual claim mapping

### Day 52

- run rollback verify rehearsal
- confirm scripts and docs alignment
- sign off rollback readiness

### Day 53

- final full verification run
- lock release candidate
- confirm all KPI evidence complete

### Day 54

- tag release candidate as final if green
- trigger release workflows
- monitor workflow completion

### Day 55

- publish release notes
- publish post-release watch checklist
- monitor first post-release signals

### Day 56

- post-release retrospective
- capture carryover items to 1.2.1
- close milestone and archive evidence

## 20) Appendix C - PR Template for 1.2.0 Work

Use this structure in every 1.2.0 PR:

### Summary

- what changed
- why now
- scope boundary

### Determinism impact

- none / low / medium / high
- details

### Governance impact

- docs updated yes/no
- RFC reference yes/no

### Test plan

- exact commands
- expected result

### Risks and rollback

- primary risk
- rollback path

## 21) Appendix D - Definition of Done

Release 1.2.0 is done only when all are true:

1. all P0 tasks complete with evidence
2. mandatory and release gates green
3. docs-contract audit has zero mismatch
4. strict no-fallback behavior validated
5. rollback verify passes
6. release notes are factual and bounded

## 22) Appendix E - Immediate Commands for Kickoff

```bash
git status
git pull --ff-only
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo test --features onnx-import
bash scripts/ci/wave1_local_verify.sh
```

## 23) Appendix F - Decision Log (fill as you go)

DL-01:
- date:
- decision:
- rationale:
- alternatives considered:
- owner:

DL-02:
- date:
- decision:
- rationale:
- alternatives considered:
- owner:

DL-03:
- date:
- decision:
- rationale:
- alternatives considered:
- owner:

DL-04:
- date:
- decision:
- rationale:
- alternatives considered:
- owner:

DL-05:
- date:
- decision:
- rationale:
- alternatives considered:
- owner:

This document is intentionally detailed to serve as both execution guide and release audit trail.
