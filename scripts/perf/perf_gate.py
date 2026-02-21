#!/usr/bin/env python3
"""Compares current perf probe output against CPU-keyed baseline."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_PROBE_CMD = [
    "cargo",
    "run",
    "--release",
    "--bin",
    "perf_probe",
    "--",
    "--samples",
    "9",
    "--dim",
    "96",
    "--matmul-iters",
    "3",
    "--relu-iters",
    "16",
]


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline_median: float
    current_median: float
    delta_percent: float
    threshold_percent: float
    regressed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate performance regressions")
    parser.add_argument(
        "--baseline-dir",
        default="benchmarks/baselines",
        help="Directory that stores <signature>.json baseline files",
    )
    parser.add_argument(
        "--signature",
        default="",
        help="Override CPU signature for baseline lookup",
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=5.0,
        help="Allowed regression threshold in percent",
    )
    parser.add_argument(
        "--probe-file",
        default="",
        help="Read probe JSON from file instead of running the probe command",
    )
    parser.add_argument(
        "--probe-cmd",
        nargs="*",
        default=None,
        help="Custom probe command tokens",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="Do not fail if baseline file is missing; create one from current metrics",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Write machine-readable comparison result to this file",
    )
    return parser.parse_args()


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parse_probe_stdout(output: str) -> Dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("probe output did not contain a JSON object line")


def read_probe_result(args: argparse.Namespace) -> Dict[str, Any]:
    if args.probe_file:
        return _load_json(pathlib.Path(args.probe_file))

    cmd = args.probe_cmd if args.probe_cmd else DEFAULT_PROBE_CMD
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "perf probe command failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return _parse_probe_stdout(completed.stdout)


def _metric_items(
    probe_payload: Dict[str, Any],
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    metrics = probe_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("probe payload missing 'metrics' object")
    for name, metric in metrics.items():
        if not isinstance(metric, dict):
            raise ValueError(f"metric '{name}' is not an object")
        if "median" not in metric:
            raise ValueError(f"metric '{name}' missing 'median' value")
        yield name, metric


def compare_metrics(
    current_payload: Dict[str, Any],
    baseline_payload: Dict[str, Any],
    threshold_percent: float,
) -> List[MetricDelta]:
    deltas: List[MetricDelta] = []

    baseline_metrics = baseline_payload.get("metrics")
    if not isinstance(baseline_metrics, dict):
        raise ValueError("baseline payload missing 'metrics' object")

    for name, current_metric in _metric_items(current_payload):
        baseline_metric = baseline_metrics.get(name)
        if not isinstance(baseline_metric, dict):
            continue

        baseline_median = float(baseline_metric.get("median", 0.0))
        current_median = float(current_metric.get("median", 0.0))
        lower_is_better = bool(current_metric.get("lower_is_better", True))

        if baseline_median == 0.0:
            delta_percent = 0.0
            regressed = False
        elif lower_is_better:
            delta_percent = (
                (current_median - baseline_median) / baseline_median
            ) * 100.0
            regressed = delta_percent > threshold_percent
        else:
            delta_percent = (
                (baseline_median - current_median) / baseline_median
            ) * 100.0
            regressed = delta_percent > threshold_percent

        deltas.append(
            MetricDelta(
                name=name,
                baseline_median=baseline_median,
                current_median=current_median,
                delta_percent=delta_percent,
                threshold_percent=threshold_percent,
                regressed=regressed,
            )
        )

    return deltas


def baseline_path(baseline_dir: pathlib.Path, signature: str) -> pathlib.Path:
    return baseline_dir / f"{signature}.json"


def baseline_candidates(signature: str) -> List[str]:
    candidates = [signature]

    parts = signature.split("-")
    if len(parts) >= 2:
        generic_signature = f"{parts[0]}-{parts[1]}-generic"
        if generic_signature not in candidates:
            candidates.append(generic_signature)
        example_signature = f"example-{parts[0]}-{parts[1]}-generic"
        if example_signature not in candidates:
            candidates.append(example_signature)

    return candidates


def resolve_baseline_file(
    baseline_dir: pathlib.Path, signature: str
) -> Tuple[str, pathlib.Path]:
    for candidate in baseline_candidates(signature):
        candidate_path = baseline_path(baseline_dir, candidate)
        if candidate_path.exists():
            return candidate, candidate_path
    return signature, baseline_path(baseline_dir, signature)


def detect_signature() -> str:
    signature_script = pathlib.Path(__file__).with_name("cpu_signature.py")
    completed = subprocess.run(
        [sys.executable, str(signature_script)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        signature = completed.stdout.strip()
        if signature:
            return signature
    return "unknown-system-unknown-machine-unknown-cpu"


def main() -> int:
    args = parse_args()
    signature = args.signature.strip() or detect_signature()
    baseline_dir = pathlib.Path(args.baseline_dir)
    current = read_probe_result(args)

    baseline_signature, baseline_file = resolve_baseline_file(baseline_dir, signature)
    if not baseline_file.exists():
        if args.allow_missing_baseline:
            payload = {
                "signature": signature,
                "created_by": "perf_gate",
                "probe": current,
            }
            _dump_json(baseline_file, payload)

            result = {
                "status": "baseline_created",
                "signature": signature,
                "resolved_signature": signature,
                "baseline_file": str(baseline_file),
            }
            if args.output_json:
                _dump_json(pathlib.Path(args.output_json), result)
            print(json.dumps(result, sort_keys=True))
            return 0

        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "missing_baseline",
                    "signature": signature,
                    "resolved_signature": signature,
                    "tried_signatures": baseline_candidates(signature),
                    "baseline_file": str(baseline_file),
                },
                sort_keys=True,
            )
        )
        return 1

    baseline = _load_json(baseline_file).get("probe", {})
    deltas = compare_metrics(current, baseline, args.threshold_percent)
    failures = [delta for delta in deltas if delta.regressed]

    payload = {
        "status": "failed" if failures else "passed",
        "signature": signature,
        "resolved_signature": baseline_signature,
        "baseline_file": str(baseline_file),
        "threshold_percent": args.threshold_percent,
        "metrics": [
            {
                "name": delta.name,
                "baseline_median": delta.baseline_median,
                "current_median": delta.current_median,
                "delta_percent": delta.delta_percent,
                "regressed": delta.regressed,
            }
            for delta in deltas
        ],
    }

    if args.output_json:
        _dump_json(pathlib.Path(args.output_json), payload)
    print(json.dumps(payload, sort_keys=True))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
