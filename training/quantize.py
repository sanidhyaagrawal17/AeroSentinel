import yaml
from pathlib import Path
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType, CalibrationDataReader
import os
import sys

# Add the project root to sys.path so we can import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocessing.pipeline import Preprocessor

class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_folder, image_count, input_name):
        super().__init__()
        self.image_folder = Path(image_folder)
        self.input_name = input_name
        self.image_files = list(self.image_folder.glob("*.jpg")) + list(self.image_folder.glob("*.png"))
        self.image_files = self.image_files[:image_count]
        
        # Initialize the EXACT same preprocessor used in inference
        self.preprocessor = Preprocessor()
        self.enum_data = iter(self.image_files)
        
    def get_next(self):
        try:
            img_path = next(self.enum_data)
            # CRITICAL: We use the EXACT same preprocessor as inference to guarantee 
            # the calibration data matches the input distribution during runtime.
            # This prevents silent accuracy drops caused by preprocessing mismatch.
            tensor, _, _, _ = self.preprocessor.process(str(img_path))
            return {self.input_name: tensor}
        except StopIteration:
            return None

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def quantize_onnx():
    config_path = Path(__file__).parent.parent / 'configs' / 'quantize_config.yaml'
    config = load_config(config_path)
    
    model_path = Path(__file__).parent.parent / 'models' / 'aerosentinel.onnx'
    output_path = Path(__file__).parent.parent / 'models' / 'aerosentinel_int8.onnx'
    
    if not model_path.exists():
        print(f"Error: ONNX model not found at {model_path}. Run export_onnx.py first.")
        return
        
    weight_type_str = config.get('weight_type', 'QInt8')
    weight_type = QuantType.QInt8 if weight_type_str == 'QInt8' else QuantType.QUInt8
    
    mode = config.get('quantization_mode', 'dynamic')
    optimize = config.get('optimize_model', True)
    per_channel = config.get('per_channel', True)
    
    if mode == 'static':
        calib_dir = Path(__file__).parent.parent / config.get('calibration_image_dir', 'data/processed/images/train')
        calib_count = config.get('calibration_image_count', 100)
        
        # Pre-flight check
        if not calib_dir.exists() or len(list(calib_dir.glob("*.jpg"))) == 0:
            if config.get('fallback_to_dynamic', False):
                print(f"Warning: Calibration directory {calib_dir} is empty or missing. Falling back to dynamic quantization.")
                mode = 'dynamic'
            else:
                print(f"Error: Calibration directory {calib_dir} is missing or empty.")
                print("Please run augment_offline() to generate the dataset, or set fallback_to_dynamic: true in config.")
                return
    
    if mode == 'static':
        print(f"Quantizing ONNX model statically: {model_path}")
        print(f"Using {calib_count} images from {calib_dir} for calibration.")
        
        # Get the input name from the ONNX model
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        dr = YoloCalibrationDataReader(calib_dir, calib_count, input_name)
        
        quantize_static(
            model_input=str(model_path),
            model_output=str(output_path),
            calibration_data_reader=dr,
            quant_format=config.get('quant_format', 'QOperator'),
            per_channel=per_channel,
            weight_type=weight_type,
            activation_type=QuantType.QUInt8
        )
    else:
        print(f"Quantizing ONNX model dynamically: {model_path}")
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=weight_type,
            optimize_model=optimize,
            per_channel=per_channel
        )
    
    print(f"Quantization complete. INT8 model saved to {output_path}")
    
    # Print file sizes
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Quantized size: {quant_size:.2f} MB")
    print(f"Reduction: {(1 - quant_size/orig_size)*100:.2f}%")
    print("\nNext Step: Run evaluate.py on the INT8 model to verify mAP drop.")

if __name__ == '__main__':
    quantize_onnx()
