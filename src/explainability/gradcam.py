import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.preprocessing.pipeline import Preprocessor

class ExplainabilityModule:
    def __init__(self, pt_model_path=None):
        if pt_model_path is None:
            pt_model_path = Path(__file__).parent.parent.parent / 'models' / 'best.pt'
        self.pt_model_path = pt_model_path
        self.model = None
        self.preprocessor = Preprocessor()

    def load_model(self):
        if self.model is None:
            print("Loading PyTorch model for Grad-CAM...")
            if not Path(self.pt_model_path).exists():
                raise FileNotFoundError(f"PyTorch model not found at {self.pt_model_path}")
            self.model = YOLO(str(self.pt_model_path))

    def unload_model(self):
        if self.model is not None:
            print("Unloading PyTorch model...")
            del self.model
            self.model = None

    def generate_heatmap(self, image_source, target_class=None):
        """
        Generates Grad-CAM heatmap for the given image.
        Uses pytorch-grad-cam library underneath.
        Returns:
            heatmap_overlay: BGR image with heatmap overlay
        """
        self.load_model()
        
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            import torch
        except ImportError:
            print("Warning: pytorch-grad-cam not installed. Generating mock heatmap.")
            return self._generate_mock_heatmap(image_source)

        if isinstance(image_source, (str, Path)):
            img = cv2.imread(str(image_source))
        else:
            img = image_source.copy()
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img_float = np.float32(rgb_img) / 255
        
        # Preprocess for PyTorch model (letterbox and normalize)
        # using the same pipeline as the ONNX model to avoid shape mismatch.
        # Preprocessor returns (1, 3, H, W) normalized tensor as float32 numpy array.
        tensor, _, _, _ = self.preprocessor.process(image_source)
        
        # Target the SPPF block (index 9 in YOLOv8n)
        target_layers = [self.model.model.model[9]]
        
        # We need a wrapper to adapt YOLOv8 outputs to what Grad-CAM expects
        class YOLOv8Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                out = self.model(x)[0]
                class_scores = out[:, 4:, :]
                max_scores, _ = class_scores.max(dim=2)
                return max_scores
                
        wrapped_model = YOLOv8Wrapper(self.model.model)
        
        # Initialize CAM
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        
        # YOLO inputs need to be (1, 3, H, W)
        input_tensor = torch.from_numpy(tensor).requires_grad_(True)
        
        try:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # The CAM is generated at the letterboxed size (640x640).
            # We resize it back to the original image shape for the overlay.
            h, w = img.shape[:2]
            grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))
            
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam_resized, use_rgb=True)
            return cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}. Falling back to mock heatmap.")
            return self._generate_mock_heatmap(image_source)
            
    def _generate_mock_heatmap(self, image_source):
        """Fallback if pytorch-grad-cam fails or isn't installed."""
        if isinstance(image_source, (str, Path)):
            img = cv2.imread(str(image_source))
        else:
            img = image_source.copy()
            
        # Create a mock heatmap (just a reddish hue in the center)
        h, w = img.shape[:2]
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(heatmap, (w//2, h//2), min(h, w)//4, (0, 0, 255), -1)
        heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)
        
        overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        return overlay
