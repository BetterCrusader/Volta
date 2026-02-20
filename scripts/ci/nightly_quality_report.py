#!/usr/bin/env python3
"""Creates a lightweight nightly quality report markdown artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render nightly quality report")
    parser.add_argument(
        "--report-dir",
        default="benchmarks/reports/nightly",
        help="Directory containing nightly perf report files",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/reports/nightly/quality-report.md",
        help="Path to markdown output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = pathlib.Path(args.report_dir)
    output = pathlib.Path(args.output)

    report_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in report_dir.glob("**/*") if path.is_file())

    lines = [
        "# Nightly Quality Report",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Artifacts",
        "",
    ]

    if not files:
        lines.append("- No artifacts found.")
    else:
        for path in files:
            rel = path.relative_to(report_dir)
            lines.append(f"- `{rel.as_posix()}`")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
