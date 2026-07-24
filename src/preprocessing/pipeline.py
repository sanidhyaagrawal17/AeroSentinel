import cv2
import numpy as np
import yaml
from pathlib import Path

class Preprocessor:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'clahe_config.yaml'
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.get('clip_limit', 2.0),
            tileGridSize=tuple(self.config.get('tile_grid_size', [8, 8]))
        )
        # YOLO standard input size
        self.img_size = 640

    def load_image(self, image_source):
        """Loads an image from a path or returns the array if already loaded."""
        if isinstance(image_source, (str, Path)):
            img = cv2.imread(str(image_source))
            if img is None:
                raise ValueError(f"Could not load image at {image_source}")
            return img
        elif isinstance(image_source, np.ndarray):
            return image_source
        else:
            raise TypeError("image_source must be a path or a numpy array")

    def process(self, image_source):
        """
        Preprocesses an image for YOLOv8 ONNX inference.
        Returns:
            preprocessed_tensor: (1, 3, 640, 640) float32 numpy array
            original_image: For display purposes
            ratio: (rx, ry) scaling ratios
            pad: (dw, dh) paddings
        """
        img0 = self.load_image(image_source)
        
        # Apply CLAHE on the L channel of LAB color space
        lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Letterbox resizing (YOLO standard)
        img, ratio, pad = self._letterbox(img_clahe, new_shape=(self.img_size, self.img_size))
        
        # Convert BGR to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        
        # Normalize 0-255 to 0.0-1.0
        tensor = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, img0, ratio, pad

    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
