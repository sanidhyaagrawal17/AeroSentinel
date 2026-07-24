import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import Preprocessor
from src.inference.engine import InferenceEngine
from src.inference.severity import SeverityClassifier

# Constants for testing
DEMO_DIR = Path("demo/sample_images")
PATCHES_IMG = DEMO_DIR / "test_patches.jpg"
SCRATCHES_IMG = DEMO_DIR / "test_scratches.jpg"

@pytest.fixture
def preprocessor():
    return Preprocessor()

@pytest.fixture
def severity_classifier():
    return SeverityClassifier()

@pytest.fixture
def inference_engine():
    return InferenceEngine()

def test_preprocessor_shape(preprocessor):
    """Test if preprocessing outputs the correct (1, 3, 640, 640) shape and normalizes correctly."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tensor, orig_img, ratio, pad = preprocessor.process(img)
    
    assert tensor.shape == (1, 3, 640, 640), "Tensor shape should be exactly [1, 3, 640, 640]"
    assert tensor.dtype == np.float32, "Tensor must be float32 for ONNX"
    assert tensor.max() <= 1.0, "Tensor should be normalized between 0 and 1"
    assert tensor.min() >= 0.0, "Tensor should be normalized between 0 and 1"

def test_severity_classifier_critical(severity_classifier):
    """Test severity logic: high confidence + large area = Critical"""
    # area = 100 * 100 = 10000 (Critical is usually > 5000 and conf > 0.8)
    bbox = [0, 0, 100, 100]
    conf = 0.95
    severity = severity_classifier.classify(bbox, conf)
    assert severity == "Critical", f"Expected Critical, got {severity}"

def test_severity_classifier_low(severity_classifier):
    """Test severity logic: low confidence = Low"""
    bbox = [0, 0, 100, 100]
    conf = 0.30
    severity = severity_classifier.classify(bbox, conf)
    assert severity == "Low", f"Expected Low, got {severity}"

def test_inference_empty_image(inference_engine):
    """Test inference on a completely blank image (should return no defects gracefully)."""
    blank_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results, annotated_img, latency = inference_engine.predict(blank_img)
    
    assert len(results) == 0, "Expected 0 defects on a blank image"
    assert latency > 0, "Latency should be calculated"

def test_inference_real_image(inference_engine):
    """Test inference on real test_patches image."""
    img = cv2.imread(str(PATCHES_IMG))
    assert img is not None, "Test image could not be loaded"
    
    results, annotated_img, latency = inference_engine.predict(img)
    
    assert len(results) > 0, "Should detect defects in test_patches.jpg"
    assert all('severity' in r for r in results), "Results should have severity grading"
    assert all(r['confidence'] > 0.25 for r in results), "Results should pass default threshold"

def test_inference_low_confidence_filtering(inference_engine):
    """Test if threshold filtering works properly by setting a very high threshold."""
    img = cv2.imread(str(PATCHES_IMG))
    
    # Temporarily force threshold to 0.99
    inference_engine.conf_threshold = 0.99
    results, annotated_img, latency = inference_engine.predict(img)
    
    # There shouldn't be anything above 0.99 in the patches image
    assert len(results) == 0, "Threshold filtering failed, detected something above 0.99 conf"
