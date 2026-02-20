#!/usr/bin/env python3
"""Detects highest impacted quality tier from changed paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class TierMatchRule:
    tier: str
    prefix: str


TIER_RULES: List[TierMatchRule] = [
    TierMatchRule("A", "src/ir/"),
    TierMatchRule("A", "src/device/"),
    TierMatchRule("A", "src/ir/tensor.rs"),
    TierMatchRule("B", "src/model/"),
    TierMatchRule("C", "src/lexer.rs"),
    TierMatchRule("C", "src/parser.rs"),
    TierMatchRule("C", "src/semantic.rs"),
    TierMatchRule("C", "src/executor.rs"),
    TierMatchRule("C", "src/autopilot.rs"),
]

TIER_PRIORITY = {"A": 3, "B": 2, "C": 1, "NONE": 0}


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def detect_tier(changed_paths: Iterable[str]) -> str:
    best = "NONE"
    for path in changed_paths:
        normalized = _normalize_path(path)
        if not normalized:
            continue
        for rule in TIER_RULES:
            if (
                normalized.startswith(rule.prefix)
                and TIER_PRIORITY[rule.tier] > TIER_PRIORITY[best]
            ):
                best = rule.tier
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect impacted quality tier")
    parser.add_argument(
        "--paths", nargs="*", default=[], help="Changed repository paths"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed_paths = args.paths
    tier = detect_tier(changed_paths)
    print(f"tier={tier}")
    print(f"changed={'true' if len(changed_paths) > 0 else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
