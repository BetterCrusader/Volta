import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.perf.perf_gate import compare_metrics


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


if __name__ == "__main__":
    unittest.main()
