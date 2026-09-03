from ultralytics import YOLO
import os

BEST_MODEL_PATH = "fire_smoke_training/best.pt"

print("=" * 70)
print("CONVERTING YOLO MODEL TO ONNX")
print("=" * 70)

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")

# Load PyTorch model
model = YOLO(BEST_MODEL_PATH)

# Export to ONNX
onnx_path = model.export(
    format="onnx",
    imgsz=768,
    opset=12,
    simplify=True,
    dynamic=False,
    half=False
)

print("=" * 70)
print("ONNX EXPORT COMPLETED")
print("=" * 70)
print(f"ONNX model saved at: {onnx_path}")