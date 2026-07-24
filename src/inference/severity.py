import yaml
from pathlib import Path

class SeverityClassifier:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'inference_config.yaml'
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.rules = self.config.get('severity_rules', {
            'critical': {'min_confidence': 0.80, 'min_area': 5000},
            'medium': {'min_confidence': 0.60, 'min_area': 2000},
            'low': {'min_confidence': 0.25, 'min_area': 0}
        })

    def classify(self, bbox, confidence):
        """
        Heuristic rule-based triage on detection area + confidence.
        bbox: [x1, y1, x2, y2]
        confidence: float
        """
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Check critical first
        if confidence >= self.rules['critical']['min_confidence'] and area >= self.rules['critical']['min_area']:
            return "Critical"
            
        # Override for tiny but extremely high confidence detections (e.g. hairline cracks)
        if confidence >= 0.90:
            return "Medium"
            
        # Check medium
        if confidence >= self.rules['medium']['min_confidence'] and area >= self.rules['medium']['min_area']:
            return "Medium"
            
        return "Low"
