import os
import argparse
import cv2
import torch
import re
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from ultralytics import __file__ as ultralytics_file
import onnxruntime as ort

# ============================================================
# CONFIGURATION
# ============================================================
ONNX_MODEL_PATH = "fire_smoke_training/best.onnx"
parser = argparse.ArgumentParser()
parser.add_argument("input_video", nargs="?", default="input_video.mp4")
parser.add_argument("output_video", nargs="?", default="nput_videooutput_tracked_onnx.mp4")
args = parser.parse_args()
INPUT_VIDEO = args.input_video
OUTPUT_VIDEO = args.output_video
CONF_THRESHOLD = 0.35
IMGSZ = 768
IOU_THRESHOLD = 0.45
MAX_DET = 300
CLASS_NAMES = {0: "Smoke", 1: "Fire"}
TARGET_CLASSES = [0, 1]
USE_REID = False
FRAME_THRESHOLD = 5
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.4
LABEL_FONT_THICKNESS = 1
ALERT_FONT_SCALE = 0.5
ALERT_FONT_THICKNESS = 1
STATUS_FONT_SCALE = 0.6
STATUS_FONT_THICKNESS = 1
LABEL_PADDING_X = 4
LABEL_PADDING_Y = 3
ALERT_PADDING_X = 6
ALERT_PADDING_Y = 4

# ============================================================
# GPU ENVIRONMENT INFORMATION
# ============================================================
print("\n" + "=" * 70)
print("GPU ENVIRONMENT")
print("=" * 70)

print(f"PyTorch version       : {torch.__version__}")
print(f"CUDA available        : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU                   : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version          : {torch.version.cuda}")
    print(f"CUDA device count    : {torch.cuda.device_count()}")
    print(f"Current CUDA device  : {torch.cuda.current_device()}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"CUDA memory cached   : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("⚠️  CUDA is not available. Inference will run on CPU.")

print(f"ONNX Runtime version  : {ort.__version__}")
print(f"ONNX Runtime providers: {ort.get_available_providers()}")

if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("✅ ONNX Runtime CUDA provider is available")
else:
    print("⚠️  ONNX Runtime CUDA provider is NOT available")

print("=" * 70)

# ============================================================
# PREPARE BoT-SORT CONFIGURATION
# ============================================================
ULTRALYTICS_DIR = os.path.dirname(ultralytics_file)
DEFAULT_TRACKER_PATH = os.path.join(ULTRALYTICS_DIR, "cfg", "trackers", "botsort.yaml")
PROJECT_DIR = os.getcwd()
CUSTOM_TRACKER_PATH = os.path.join(PROJECT_DIR, "custom_botsort.yaml")

print("\n" + "=" * 70)
print("PREPARING BoT-SORT CONFIGURATION")
print("=" * 70)

if not os.path.exists(DEFAULT_TRACKER_PATH):
    raise FileNotFoundError(f"BoT-SORT configuration not found:\n{DEFAULT_TRACKER_PATH}")

with open(DEFAULT_TRACKER_PATH, "r", encoding="utf-8") as f:
    tracker_config = f.read()

if re.search(r"^\s*with_reid\s*:.*$", tracker_config, re.MULTILINE):
    tracker_config = re.sub(r"^\s*with_reid\s*:.*$", f"with_reid: {str(USE_REID)}", tracker_config, flags=re.MULTILINE)
else:
    tracker_config += f"\nwith_reid: {str(USE_REID)}\n"

if USE_REID:
    if re.search(r"^\s*model\s*:.*$", tracker_config, re.MULTILINE):
        tracker_config = re.sub(r"^\s*model\s*:.*$", "model: auto", tracker_config, flags=re.MULTILINE)
    else:
        tracker_config += "\nmodel: auto\n"

with open(CUSTOM_TRACKER_PATH, "w", encoding="utf-8") as f:
    f.write(tracker_config)

print("Custom BoT-SORT configuration created successfully.")

# ============================================================
# LOAD ONNX MODEL
# ============================================================
print("\n" + "=" * 70)
print("LOADING FIRE & SMOKE ONNX MODEL")
print("=" * 70)

if not os.path.exists(ONNX_MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}\nPlease run the ONNX export code first.")

model = YOLO(ONNX_MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Model loaded     : {ONNX_MODEL_PATH}")
print("Model format     : ONNX")
print(f"Device requested : {device}")
print(f"Image size       : {IMGSZ}")
print(f"Confidence       : {CONF_THRESHOLD}")
print("Tracker          : BoT-SORT")
print(f"ReID             : {'ENABLED' if USE_REID else 'DISABLED'}")
print(f"Alert threshold  : {FRAME_THRESHOLD} consecutive frames")
print("\nCLASS MAPPING:")
for class_id, class_name in CLASS_NAMES.items():
    print(f"  Class {class_id} -> {class_name}")

# ============================================================
# CHECK VIDEO FILE
# ============================================================
if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(f"Video not found: {INPUT_VIDEO}")

# ============================================================
# OPEN VIDEO
# ============================================================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 30))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================
output_dir = os.path.dirname(OUTPUT_VIDEO)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# ============================================================
# VIDEO WRITER
# ============================================================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

# ============================================================
# VIDEO INFORMATION
# ============================================================
print("\n" + "=" * 70)
print("STARTING ONNX VIDEO INFERENCE")
print("=" * 70)
print(f"Input video      : {INPUT_VIDEO}")
print(f"Output video     : {OUTPUT_VIDEO}")
print(f"Total frames     : {total_frames}")
print(f"FPS              : {fps}")
print(f"Resolution       : {width}x{height}")
print(f"Inference size   : {IMGSZ}x{IMGSZ}")
print(f"Confidence       : {CONF_THRESHOLD}")
print(f"BoT-SORT ReID    : {'ON' if USE_REID else 'OFF'}")

# ============================================================
# TRACKING VARIABLES
# ============================================================
frame_count = 0
total_detections = 0
alert_count = 0
class_detection_count = {0: 0, 1: 0}
class_alert_count = {0: 0, 1: 0}
consecutive_frames = defaultdict(int)
last_seen_frame = {}
track_classes = {}
alerted_tracks = set()

# ============================================================
# INFERENCE LOOP
# ============================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMGSZ,
        tracker=CUSTOM_TRACKER_PATH,
        classes=TARGET_CLASSES,
        max_det=MAX_DET,
        verbose=False,
        device=device
    )

    result = results[0]
    current_track_ids = set()

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
            current_track_ids.add(track_id)
            x1, y1, x2, y2 = box
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            class_name = CLASS_NAMES.get(class_id, "Unknown")
            class_detection_count[class_id] = class_detection_count.get(class_id, 0) + 1
            total_detections += 1
            track_classes[track_id] = class_name

            if track_id in last_seen_frame and last_seen_frame[track_id] == frame_count - 1:
                consecutive_frames[track_id] += 1
            else:
                consecutive_frames[track_id] = 1
            last_seen_frame[track_id] = frame_count

            color = (0, 0, 255) if class_name == "Fire" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            label = f"{class_name} #{track_id}"
            (text_width, text_height), baseline = cv2.getTextSize(label, FONT, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)
            label_width = text_width + LABEL_PADDING_X * 2
            label_height = text_height + baseline + LABEL_PADDING_Y * 2
            label_x = x1
            label_top = y2 + 2
            label_bottom = label_top + label_height

            if label_bottom > height:
                label_bottom = y1 - 2
                label_top = label_bottom - label_height
            if label_x + label_width > width:
                label_x = width - label_width - 2
            if label_x < 2:
                label_x = 2

            cv2.rectangle(frame, (label_x, label_top), (label_x + label_width, label_bottom), color, -1)
            text_x = label_x + LABEL_PADDING_X
            text_y = label_bottom - LABEL_PADDING_Y - baseline
            cv2.putText(frame, label, (text_x, text_y), FONT, LABEL_FONT_SCALE, (255, 255, 255), LABEL_FONT_THICKNESS, cv2.LINE_AA)

            if consecutive_frames[track_id] >= FRAME_THRESHOLD and track_id not in alerted_tracks:
                alerted_tracks.add(track_id)
                alert_count += 1
                class_alert_count[class_id] = class_alert_count.get(class_id, 0) + 1
                print(f"[ALERT] Frame {frame_count}: {class_name} detected for {FRAME_THRESHOLD} consecutive frames (Track ID: {track_id}, Conf: {conf:.2f})")

            if track_id in alerted_tracks:
                alert_text = f"ALERT: {class_name}"
                (alert_width, alert_height), alert_baseline = cv2.getTextSize(alert_text, FONT, ALERT_FONT_SCALE, ALERT_FONT_THICKNESS)
                alert_box_width = alert_width + ALERT_PADDING_X * 2
                alert_box_height = alert_height + alert_baseline + ALERT_PADDING_Y * 2
                alert_x = x1
                alert_top = label_bottom + 2
                alert_bottom = alert_top + alert_box_height

                if alert_bottom <= height:
                    if alert_x + alert_box_width > width:
                        alert_x = width - alert_box_width - 2
                    if alert_x < 2:
                        alert_x = 2
                    cv2.rectangle(frame, (alert_x, alert_top), (alert_x + alert_box_width, alert_bottom), (0, 0, 255), -1)
                    alert_text_x = alert_x + ALERT_PADDING_X
                    alert_text_y = alert_bottom - ALERT_PADDING_Y - alert_baseline
                    cv2.putText(frame, alert_text, (alert_text_x, alert_text_y), FONT, ALERT_FONT_SCALE, (255, 255, 255), ALERT_FONT_THICKNESS, cv2.LINE_AA)

    disappeared_tracks = []
    for track_id in list(last_seen_frame.keys()):
        if track_id not in current_track_ids:
            if last_seen_frame[track_id] == frame_count - 1:
                consecutive_frames[track_id] = 0
            if frame_count - last_seen_frame[track_id] > 30:
                disappeared_tracks.append(track_id)

    for track_id in disappeared_tracks:
        consecutive_frames.pop(track_id, None)
        last_seen_frame.pop(track_id, None)
        track_classes.pop(track_id, None)

    status_text = f"BoT-SORT | ReID: {'ON' if USE_REID else 'OFF'} | Alerts: {alert_count}"
    cv2.putText(frame, status_text, (10, 25), FONT, STATUS_FONT_SCALE, (255, 255, 255), STATUS_FONT_THICKNESS, cv2.LINE_AA)

    out.write(frame)

    if frame_count % 100 == 0:
        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
        print(f"Processing: {frame_count}/{total_frames} ({progress:.1f}%)")

# ============================================================
# RELEASE RESOURCES
# ============================================================
cap.release()
out.release()
cv2.destroyAllWindows()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ONNX INFERENCE COMPLETED")
print("=" * 70)
print(f"Output video       : {OUTPUT_VIDEO}")
print(f"Frames processed   : {frame_count}")
print(f"Total detections   : {total_detections}")
print(f"Total alerts       : {alert_count}")
print(f"BoT-SORT ReID      : {'ENABLED' if USE_REID else 'DISABLED'}")
print("\nDETECTION STATISTICS BY CLASS:")
for class_id, class_name in CLASS_NAMES.items():
    detections_count = class_detection_count.get(class_id, 0)
    alerts_count = class_alert_count.get(class_id, 0)
    print(f"  {class_name} (Class {class_id}): {detections_count} detections, {alerts_count} alerts")

if alerted_tracks:
    print("\nALERT TRIGGERED FOR TRACKS:")
    for track_id in sorted(alerted_tracks):
        class_name = track_classes.get(track_id, "Unknown")
        print(f"  - Track ID {track_id}: {class_name}")
else:
    print("\nNo alerts triggered. No fire or smoke satisfied the temporal alert condition.")

print("=" * 70)
print("COMPLETE")
print("=" * 70)

# ============================================================
# DISPLAY OUTPUT VIDEO IN JUPYTER
# ============================================================
try:
    from IPython.display import Video, display
    print("\nDisplaying output video...")
    display(Video(OUTPUT_VIDEO, width=900, embed=True))
except Exception as e:
    print("\nVideo was saved successfully, but could not be displayed inside Jupyter.")
    print(f"Reason: {e}")
print(f"\nVideo saved successfully at:\n{os.path.abspath(OUTPUT_VIDEO)}")