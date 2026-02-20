import subprocess
import sys
import unittest
from pathlib import Path

from scripts.ci.policy_check import HARDENING_LABEL, validate


class PolicyCheckTests(unittest.TestCase):
    def test_requires_rfc_reference_for_tier_a_policy_changes(self):
        result = validate(
            changed_paths=["docs/governance/contracts-tier-a.md"],
            pr_body="",
            branch_name="task/quality-wave1",
            labels=[],
        )
        self.assertFalse(result.ok)
        self.assertTrue(
            any("RFC reference required" in issue for issue in result.errors)
        )

    def test_accepts_rfc_marker_for_policy_change(self):
        result = validate(
            changed_paths=["docs/governance/operational-policy.md"],
            pr_body="Closes RFC: RFC-2026-001",
            branch_name="task/quality-wave1",
            labels=[],
        )
        self.assertTrue(result.ok)

    def test_accepts_changed_rfc_document_for_policy_change(self):
        result = validate(
            changed_paths=[
                "docs/governance/contracts-tier-a.md",
                "docs/governance/rfcs/RFC-2026-001-wave1-foundation.md",
            ],
            pr_body="",
            branch_name="task/quality-wave1",
            labels=[],
        )
        self.assertTrue(result.ok)

    def test_requires_rfc_reference_for_tier_a_source_changes(self):
        result = validate(
            changed_paths=["./src/ir/verifier.rs"],
            pr_body="",
            branch_name="task/quality-wave1",
            labels=[],
        )
        self.assertFalse(result.ok)
        self.assertTrue(
            any("RFC reference required" in issue for issue in result.errors)
        )

    def test_blocks_exp_branch_without_hardening_label(self):
        result = validate(
            changed_paths=["README.md"],
            pr_body="",
            branch_name="exp/new-experiment",
            labels=[],
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("hardening-approved" in issue for issue in result.errors))

    def test_allows_exp_branch_with_hardening_label(self):
        result = validate(
            changed_paths=["README.md"],
            pr_body="",
            branch_name="exp/new-experiment",
            labels=[HARDENING_LABEL],
        )
        self.assertTrue(result.ok)

    def test_cli_accepts_null_labels_json(self):
        script = Path("scripts/ci/policy_check.py")
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--labels-json",
                "null",
                "--paths",
                "README.md",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
