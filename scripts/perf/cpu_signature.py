#!/usr/bin/env python3
"""Builds a stable CPU signature string for perf baselines."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CpuIdentity:
    system: str
    machine: str
    cpu_name: str
    signature: str


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_component(value: str) -> str:
    lowered = value.strip().lower()
    if not lowered:
        return "unknown"
    normalized = _NON_ALNUM.sub("-", lowered).strip("-")
    return normalized or "unknown"


def detect_cpu_name() -> str:
    system = platform.system().lower()
    if system == "linux":
        path = "/proc/cpuinfo"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        _, _, value = line.partition(":")
                        candidate = value.strip()
                        if candidate:
                            return candidate

    if system == "darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        candidate = completed.stdout.strip()
        if completed.returncode == 0 and candidate:
            return candidate

    if system == "windows":
        candidate = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if candidate:
            return candidate

    candidate = platform.processor().strip()
    return candidate or "unknown-cpu"


def detect_identity() -> CpuIdentity:
    system = platform.system() or "unknown-system"
    machine = platform.machine() or "unknown-machine"
    cpu_name = detect_cpu_name()

    signature = "-".join(
        [
            normalize_component(system),
            normalize_component(machine),
            normalize_component(cpu_name),
        ]
    )

    return CpuIdentity(
        system=system,
        machine=machine,
        cpu_name=cpu_name,
        signature=signature,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect CPU signature for perf baselines"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of signature only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity = detect_identity()

    if args.json:
        print(
            json.dumps(
                {
                    "signature": identity.signature,
                    "system": identity.system,
                    "machine": identity.machine,
                    "cpu_name": identity.cpu_name,
                },
                sort_keys=True,
            )
        )
        return 0

    print(identity.signature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
