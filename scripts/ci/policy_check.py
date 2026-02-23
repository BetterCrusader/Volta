#!/usr/bin/env python3
"""Policy checks for governance-sensitive pull requests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
from typing import Iterable, List, Sequence


RFC_REQUIRED_PATH_PREFIXES = (
    "src/ir/",
    "src/device/",
    "src/ir/tensor.rs",
    "docs/governance/contracts-tier-a.md",
    "docs/governance/determinism-scope.md",
    "docs/governance/operational-policy.md",
)

RFC_DOCUMENT_PREFIX = "docs/governance/rfcs/rfc-"

HARDENING_LABEL = "hardening-approved"

RFC_TOKEN_PATTERN = re.compile(r"\brfc-[a-z0-9][a-z0-9\-]*\b", re.IGNORECASE)


@dataclass(frozen=True)
class PolicyResult:
    ok: bool
    errors: List[str]


def _normalize(paths: Iterable[str]) -> List[str]:
    normalized = []
    for path in paths:
        candidate = path.replace("\\", "/").strip()
        while candidate.startswith("./"):
            candidate = candidate[2:]
        if candidate:
            normalized.append(candidate)
    return normalized


def _has_rfc_reference(pr_title: str, pr_body: str) -> bool:
    combined = "\n".join(value for value in [pr_title or "", pr_body or ""] if value)
    return bool(RFC_TOKEN_PATTERN.search(combined))


def _has_rfc_document(changed_paths: Sequence[str]) -> bool:
    return any(path.lower().startswith(RFC_DOCUMENT_PREFIX) for path in changed_paths)


def _requires_rfc_for_paths(paths: Sequence[str]) -> bool:
    return any(
        any(path.startswith(prefix) for prefix in RFC_REQUIRED_PATH_PREFIXES)
        for path in paths
    )


def validate(
    changed_paths: Sequence[str],
    pr_title: str,
    pr_body: str,
    branch_name: str = "",
    labels: Sequence[str] | None = None,
) -> PolicyResult:
    normalized_paths = _normalize(changed_paths)
    issues: List[str] = []

    if _requires_rfc_for_paths(normalized_paths) and not (
        _has_rfc_reference(pr_title, pr_body) or _has_rfc_document(normalized_paths)
    ):
        issues.append("RFC reference required for governance/Tier A policy changes")

    normalized_labels = {label.strip() for label in (labels or [])}
    if branch_name.startswith("exp/") and HARDENING_LABEL not in normalized_labels:
        issues.append("exp/* branch merges require hardening-approved label")

    return PolicyResult(ok=len(issues) == 0, errors=issues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository policy checks")
    parser.add_argument(
        "--paths", nargs="*", default=[], help="Changed repository paths"
    )
    parser.add_argument("--pr-title", default="", help="Pull request title")
    parser.add_argument("--pr-body", default="", help="Pull request body")
    parser.add_argument("--branch", default="", help="Source branch name")
    parser.add_argument("--labels", nargs="*", default=[], help="PR labels")
    parser.add_argument(
        "--labels-json",
        default="",
        help="JSON array of labels (CI friendly)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    labels = list(args.labels)
    if args.labels_json:
        try:
            parsed = json.loads(args.labels_json)
        except json.JSONDecodeError as err:
            print("policy=fail")
            print(f"error=invalid labels json: {err}")
            return 1

        if parsed is None:
            labels = []
        elif isinstance(parsed, list):
            labels = [str(value) for value in parsed]
        else:
            print("policy=fail")
            print("error=--labels-json must be a JSON list or null")
            return 1

    result = validate(
        changed_paths=args.paths,
        pr_title=args.pr_title,
        pr_body=args.pr_body,
        branch_name=args.branch,
        labels=labels,
    )

    if result.ok:
        print("policy=pass")
        return 0

    print("policy=fail")
    for issue in result.errors:
        print(f"error={issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
