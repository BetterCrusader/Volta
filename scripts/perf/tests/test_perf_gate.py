import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.perf.perf_gate import (
    baseline_candidates,
    compare_metrics,
    resolve_baseline_file,
)


def make_probe_payload(matmul_median: float, relu_median: float) -> dict:
    return {
        "version": 1,
        "metrics": {
            "matmul_ms": {
                "median": matmul_median,
                "stdev": 0.1,
                "lower_is_better": True,
            },
            "relu_ms": {
                "median": relu_median,
                "stdev": 0.01,
                "lower_is_better": True,
            },
        },
    }


class PerfGateCoreTests(unittest.TestCase):
    def test_compare_metrics_accepts_small_delta(self):
        current = make_probe_payload(10.2, 2.0)
        baseline = make_probe_payload(10.0, 2.0)
        deltas = compare_metrics(current, baseline, threshold_percent=5.0)
        self.assertTrue(all(not item.regressed for item in deltas))

    def test_compare_metrics_flags_large_regression(self):
        current = make_probe_payload(12.0, 2.0)
        baseline = make_probe_payload(10.0, 2.0)
        deltas = compare_metrics(current, baseline, threshold_percent=5.0)
        self.assertTrue(
            any(item.regressed and item.name == "matmul_ms" for item in deltas)
        )


class PerfGateBaselineLookupTests(unittest.TestCase):
    def test_baseline_candidates_include_generic_and_example_fallbacks(self):
        candidates = baseline_candidates("linux-x86_64-ryzen-7")
        self.assertEqual(
            candidates,
            [
                "linux-x86_64-ryzen-7",
                "linux-x86_64-generic",
                "example-linux-x86_64-generic",
            ],
        )

    def test_resolve_baseline_file_prefers_generic_when_exact_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir) / "baselines"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            generic_file = baseline_dir / "linux-x86_64-generic.json"
            generic_file.write_text("{}", encoding="utf-8")

            resolved_signature, resolved_file = resolve_baseline_file(
                baseline_dir, "linux-x86_64-ryzen-7"
            )
            self.assertEqual(resolved_signature, "linux-x86_64-generic")
            self.assertEqual(resolved_file, generic_file)


class PerfGateCliTests(unittest.TestCase):
    def test_cli_creates_baseline_when_missing_and_allowed(self):
        script = Path("scripts/perf/perf_gate.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir) / "baselines"
            probe_path = Path(temp_dir) / "probe.json"
            probe_path.write_text(
                json.dumps(make_probe_payload(10.0, 2.0)), encoding="utf-8"
            )

            cmd = [
                sys.executable,
                str(script),
                "--signature",
                "test-cpu",
                "--baseline-dir",
                str(baseline_dir),
                "--probe-file",
                str(probe_path),
                "--allow-missing-baseline",
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            self.assertTrue((baseline_dir / "test-cpu.json").exists())

    def test_cli_fails_on_regression(self):
        script = Path("scripts/perf/perf_gate.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir) / "baselines"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            baseline_payload = {
                "signature": "test-cpu",
                "probe": make_probe_payload(10.0, 2.0),
            }
            (baseline_dir / "test-cpu.json").write_text(
                json.dumps(baseline_payload), encoding="utf-8"
            )

            current_path = Path(temp_dir) / "current.json"
            current_path.write_text(
                json.dumps(make_probe_payload(13.0, 2.0)), encoding="utf-8"
            )

            cmd = [
                sys.executable,
                str(script),
                "--signature",
                "test-cpu",
                "--baseline-dir",
                str(baseline_dir),
                "--probe-file",
                str(current_path),
                "--threshold-percent",
                "5",
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(
                completed.returncode, 1, completed.stdout + completed.stderr
            )

    def test_cli_uses_generic_fallback_baseline(self):
        script = Path("scripts/perf/perf_gate.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir) / "baselines"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            baseline_payload = {
                "signature": "linux-x86_64-generic",
                "probe": make_probe_payload(10.0, 2.0),
            }
            generic_file = baseline_dir / "linux-x86_64-generic.json"
            generic_file.write_text(json.dumps(baseline_payload), encoding="utf-8")

            current_path = Path(temp_dir) / "current.json"
            current_path.write_text(
                json.dumps(make_probe_payload(10.1, 2.0)), encoding="utf-8"
            )

            cmd = [
                sys.executable,
                str(script),
                "--signature",
                "linux-x86_64-intel-core-i7",
                "--baseline-dir",
                str(baseline_dir),
                "--probe-file",
                str(current_path),
                "--threshold-percent",
                "5",
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )

            payload = json.loads(completed.stdout.strip())
            self.assertEqual(payload["resolved_signature"], "linux-x86_64-generic")
            self.assertEqual(payload["baseline_file"], str(generic_file))

    def test_cli_reports_tried_signatures_on_missing_baseline(self):
        script = Path("scripts/perf/perf_gate.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir) / "baselines"
            probe_path = Path(temp_dir) / "probe.json"
            probe_path.write_text(
                json.dumps(make_probe_payload(10.0, 2.0)), encoding="utf-8"
            )

            cmd = [
                sys.executable,
                str(script),
                "--signature",
                "linux-x86_64-intel-core-i7",
                "--baseline-dir",
                str(baseline_dir),
                "--probe-file",
                str(probe_path),
            ]
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(
                completed.returncode, 1, completed.stdout + completed.stderr
            )

            payload = json.loads(completed.stdout.strip())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["reason"], "missing_baseline")
            self.assertEqual(
                payload["tried_signatures"],
                [
                    "linux-x86_64-intel-core-i7",
                    "linux-x86_64-generic",
                    "example-linux-x86_64-generic",
                ],
            )


if __name__ == "__main__":
    unittest.main()
