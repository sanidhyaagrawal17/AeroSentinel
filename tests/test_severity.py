import unittest
from src.inference.severity import SeverityClassifier
from pathlib import Path

class TestSeverityClassifier(unittest.TestCase):
    def setUp(self):
        # We can pass None to use the default config, assuming we run from root
        config_path = Path(__file__).parent.parent / 'configs' / 'inference_config.yaml'
        self.classifier = SeverityClassifier(config_path=config_path if config_path.exists() else None)

    def test_critical_severity(self):
        # Area = 100 * 100 = 10000 (>5000), conf = 0.9 (>0.80)
        bbox = [0, 0, 100, 100]
        conf = 0.90
        self.assertEqual(self.classifier.classify(bbox, conf), "Critical")

    def test_medium_severity_due_to_area(self):
        # Area = 50 * 50 = 2500 (>2000, <5000), conf = 0.9 (>0.80 but area is medium)
        bbox = [0, 0, 50, 50]
        conf = 0.90
        self.assertEqual(self.classifier.classify(bbox, conf), "Medium")

    def test_medium_severity_due_to_conf(self):
        # Area = 100 * 100 = 10000 (>5000), conf = 0.70 (>0.60, <0.80)
        bbox = [0, 0, 100, 100]
        conf = 0.70
        self.assertEqual(self.classifier.classify(bbox, conf), "Medium")

    def test_low_severity(self):
        # Area = 10 * 10 = 100 (<2000), conf = 0.30 (>0.25, <0.60)
        bbox = [0, 0, 10, 10]
        conf = 0.30
        self.assertEqual(self.classifier.classify(bbox, conf), "Low")

if __name__ == '__main__':
    unittest.main()
