import unittest

from scripts.perf.cpu_signature import detect_identity, normalize_component


class CpuSignatureTests(unittest.TestCase):
    def test_normalize_component_replaces_symbols(self):
        self.assertEqual(
            normalize_component("Intel(R) Core i7-12700K"), "intel-r-core-i7-12700k"
        )

    def test_normalize_component_falls_back_to_unknown(self):
        self.assertEqual(normalize_component("   "), "unknown")

    def test_detect_identity_returns_non_empty_signature(self):
        identity = detect_identity()
        self.assertTrue(identity.signature)
        self.assertNotIn(" ", identity.signature)


if __name__ == "__main__":
    unittest.main()
