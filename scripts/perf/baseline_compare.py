#!/usr/bin/env python3
"""Renders a markdown comparison between baseline and candidate perf results."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render perf baseline comparison")
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--output-markdown", default="")
    return parser.parse_args()


def _load_probe(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "probe" in payload:
        return payload["probe"]
    return payload


def _build_markdown(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})

    lines: List[str] = []
    lines.append("## Perf Comparison")
    lines.append("")
    lines.append("| Metric | Baseline median | Candidate median | Delta % |")
    lines.append("| --- | ---: | ---: | ---: |")

    for name in sorted(candidate_metrics.keys()):
        candidate_median = float(candidate_metrics[name].get("median", 0.0))
        baseline_median = float(baseline_metrics.get(name, {}).get("median", 0.0))

        if baseline_median == 0.0:
            delta = 0.0
        else:
            delta = ((candidate_median - baseline_median) / baseline_median) * 100.0

        lines.append(
            f"| `{name}` | {baseline_median:.6f} | {candidate_median:.6f} | {delta:+.2f}% |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    baseline_probe = _load_probe(pathlib.Path(args.baseline_file))
    candidate_probe = _load_probe(pathlib.Path(args.candidate_file))
    markdown = _build_markdown(baseline_probe, candidate_probe)

    if args.output_markdown:
        output_path = pathlib.Path(args.output_markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
