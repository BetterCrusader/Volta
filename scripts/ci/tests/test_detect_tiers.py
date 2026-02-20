import unittest

from scripts.ci.detect_tiers import detect_tier


class DetectTierTests(unittest.TestCase):
    def test_detects_tier_a_when_ir_changed(self):
        self.assertEqual(detect_tier(["src/ir/tensor.rs"]), "A")

    def test_detects_tier_b_for_model_only_changes(self):
        self.assertEqual(detect_tier(["src/model/layers.rs"]), "B")

    def test_detects_tier_c_for_language_paths(self):
        self.assertEqual(detect_tier(["src/parser.rs"]), "C")

    def test_returns_none_when_no_mapped_paths(self):
        self.assertEqual(detect_tier(["README.md"]), "NONE")

    def test_uses_highest_priority_tier(self):
        self.assertEqual(
            detect_tier(
                ["src/model/train_api.rs", "src/ir/verifier.rs", "src/parser.rs"]
            ),
            "A",
        )

    def test_normalizes_windows_style_paths(self):
        self.assertEqual(detect_tier([r"src\ir\graph.rs"]), "A")

    def test_normalizes_dot_prefixed_paths(self):
        self.assertEqual(detect_tier(["./src/model/layers.rs"]), "B")

    def test_ignores_whitespace_only_paths(self):
        self.assertEqual(detect_tier(["   ", "\t", "README.md"]), "NONE")


if __name__ == "__main__":
    unittest.main()
