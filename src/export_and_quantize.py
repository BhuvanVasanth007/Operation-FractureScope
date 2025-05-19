from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

# Export
model = YOLO("weights/final_synthetic_defect.pt")
model.export(format="onnx", imgsz=512, dynamic=True, simplify=True, output="weights/final_synthetic_defect.onnx")

# Quantize
quantize_dynamic(
    "weights/final_synthetic_defect.onnx",
    "weights/final_synthetic_defect_q.onnx",
    weight_type=QuantType.QInt8
)
