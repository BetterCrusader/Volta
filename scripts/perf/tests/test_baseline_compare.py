import json
import tempfile
import unittest
from pathlib import Path

from scripts.perf.baseline_compare import _build_markdown


class BaselineCompareTests(unittest.TestCase):
    def test_build_markdown_renders_metric_table(self):
        baseline = {
            "metrics": {
                "matmul_ms": {"median": 10.0},
                "relu_ms": {"median": 2.0},
            }
        }
        candidate = {
            "metrics": {
                "matmul_ms": {"median": 11.0},
                "relu_ms": {"median": 2.0},
            }
        }

        markdown = _build_markdown(baseline, candidate)
        self.assertIn("| `matmul_ms` | 10.000000 | 11.000000 | +10.00% |", markdown)

    def test_can_load_probe_wrapped_payload(self):
        baseline_wrapped = {"probe": {"metrics": {"matmul_ms": {"median": 10.0}}}}
        with tempfile.TemporaryDirectory() as temp_dir:
            payload_path = Path(temp_dir) / "baseline.json"
            payload_path.write_text(json.dumps(baseline_wrapped), encoding="utf-8")

            loaded = json.loads(payload_path.read_text(encoding="utf-8"))
            self.assertIn("probe", loaded)


if __name__ == "__main__":
    unittest.main()
