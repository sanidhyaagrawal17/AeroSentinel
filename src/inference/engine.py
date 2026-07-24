import cv2
import onnxruntime as ort
import numpy as np
import time
import yaml
from pathlib import Path
from src.preprocessing.pipeline import Preprocessor
from src.inference.severity import SeverityClassifier

class InferenceEngine:
    def __init__(self, model_path=None, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'inference_config.yaml'
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / self.config.get('model_path', 'models/aerosentinel_int8.onnx')
            
        if not Path(model_path).exists():
            print(f"Warning: Model not found at {model_path}")
            self.session = None
        else:
            # CPU execution provider for edge devices
            self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            
        self.preprocessor = Preprocessor(config_path=Path(__file__).parent.parent.parent / 'configs' / 'clahe_config.yaml')
        self.severity_classifier = SeverityClassifier(config_path=config_path)
        
        self.conf_threshold = self.config.get('conf_threshold', 0.25)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        
        # Mapped from data.yaml
        self.classes = {
            0: 'Crazing (Cracks)',
            1: 'Inclusion (Fastener Anomaly)',
            2: 'Patches (Delamination)',
            3: 'Pitted Surface (Corrosion)',
            4: 'Rolled-in Scale (Fastener Anomaly)',
            5: 'Scratches (Cracks)'
        }

    def predict(self, image_source):
        """
        Runs full pipeline: preprocessing -> inference -> NMS -> severity classification.
        Returns:
            results: List of dicts with bbox, conf, class_id, class_name, severity
            annotated_img: Image with drawn boxes
            latency_ms: Inference time in milliseconds
        """
        if self.session is None:
            raise RuntimeError("Model not loaded.")
            
        # 1. Preprocess
        tensor, orig_img, ratio, pad = self.preprocessor.process(image_source)
        
        # 2. Inference
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: tensor})
        latency_ms = (time.time() - start_time) * 1000
        
        # YOLOv8 ONNX output shape is (1, num_classes + 4, num_anchors)
        # We need to transpose to (1, num_anchors, num_classes + 4)
        preds = np.transpose(outputs[0], (0, 2, 1))[0]
        
        # 3. Post-process (NMS)
        results = self._postprocess(preds, orig_img.shape, ratio, pad)
        
        # 4. Severity classification & formatting
        final_results = []
        annotated_img = orig_img.copy()
        
        # Colors based on severity
        colors = {
            "Low": (0, 255, 0),      # Green
            "Medium": (0, 165, 255), # Orange
            "Critical": (0, 0, 255)  # Red
        }
        
        for r in results:
            bbox = r['bbox']
            conf = r['conf']
            cls_id = r['class_id']
            cls_name = self.classes.get(cls_id, f"Unknown_{cls_id}")
            
            severity = self.severity_classifier.classify(bbox, conf)
            
            final_results.append({
                'bbox': bbox,
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': cls_name,
                'severity': severity
            })
            
            # Draw on image
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = colors.get(severity, (255, 255, 255))
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} ({conf:.2f}) - {severity}"
            cv2.putText(annotated_img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return final_results, annotated_img, latency_ms

    def _postprocess(self, preds, orig_shape, ratio, pad):
        """Perform Non-Maximum Suppression and scale boxes to original image."""
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        
        # Get max score and corresponding class for each anchor
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        # Convert cx, cy, w, h to x1, y1, x2, y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), 
            confidences.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes_xyxy[i]
                
                # Scale box to original image
                box[0] = (box[0] - pad[0]) / ratio[0]
                box[1] = (box[1] - pad[1]) / ratio[1]
                box[2] = (box[2] - pad[0]) / ratio[0]
                box[3] = (box[3] - pad[1]) / ratio[1]
                
                # Clip to image boundaries
                box[0] = np.clip(box[0], 0, orig_shape[1])
                box[1] = np.clip(box[1], 0, orig_shape[0])
                box[2] = np.clip(box[2], 0, orig_shape[1])
                box[3] = np.clip(box[3], 0, orig_shape[0])
                
                results.append({
                    'bbox': box.tolist(),
                    'conf': float(confidences[i]),
                    'class_id': int(class_ids[i])
                })
                
        return results
