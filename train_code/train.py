# ============================================================
# FIRE & SMOKE DETECTION - YOLOv8m
# D-Fire Dataset
# ============================================================



import os
import random
import shutil
import numpy as np
import torch

# ============================================================
# DISABLE WANDB
# ============================================================

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# Disable Ultralytics W&B integration
os.environ["ULTRALYTICS_WANDB"] = "false"

from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# Dataset
DATA_YAML = "/content/Smoke-Fire-Detection-YOLO/data.yaml"

# Google Drive
DRIVE_ROOT = "/content/drive/MyDrive"

DRIVE_RUNS = os.path.join(
    DRIVE_ROOT,
    "fire_smoke_training"
)

DRIVE_BEST = os.path.join(
    DRIVE_RUNS,
    "best.pt"
)

DRIVE_LAST = os.path.join(
    DRIVE_RUNS,
    "last.pt"
)

# Local YOLO output
LOCAL_PROJECT = "/content/runs"
LOCAL_RUN = "/content/runs/fire_smoke"

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# ENVIRONMENT
# ============================================================

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print(f"PyTorch: {torch.__version__}")
print(f"NumPy:   {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

# ============================================================
# CHECK DATASET
# ============================================================

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

assert os.path.exists(DATA_YAML), (
    f"Dataset YAML not found: {DATA_YAML}"
)

print(f"Dataset YAML: {DATA_YAML}")

# ============================================================
# GOOGLE DRIVE
# ============================================================

print("\n" + "=" * 70)
print("GOOGLE DRIVE")
print("=" * 70)

assert os.path.exists(DRIVE_ROOT), (
    "Google Drive is not mounted."
)

os.makedirs(
    DRIVE_RUNS,
    exist_ok=True
)

print(f"Checkpoint directory: {DRIVE_RUNS}")

# ============================================================
# CALLBACK
# ============================================================

def save_checkpoints_to_drive(trainer):

    epoch = trainer.epoch + 1

    local_best = os.path.join(
        trainer.save_dir,
        "weights",
        "best.pt"
    )

    local_last = os.path.join(
        trainer.save_dir,
        "weights",
        "last.pt"
    )

    # --------------------------------------------------------
    # Save LAST checkpoint
    # --------------------------------------------------------

    if os.path.exists(local_last):

        shutil.copy2(
            local_last,
            DRIVE_LAST
        )

    # --------------------------------------------------------
    # Save BEST checkpoint
    # --------------------------------------------------------

    if os.path.exists(local_best):

        shutil.copy2(
            local_best,
            DRIVE_BEST
        )

    print(
        f"\n[Google Drive] "
        f"Epoch {epoch}: checkpoints saved"
    )

# ============================================================
# LOAD YOLOv8m
# ============================================================

print("\n" + "=" * 70)
print("LOADING YOLOv8m")
print("=" * 70)

model = YOLO("yolov8m.pt")

# Register checkpoint callback
model.add_callback(
    "on_model_save",
    save_checkpoints_to_drive
)

print("YOLOv8m loaded successfully.")

# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

results = model.train(

    data=DATA_YAML,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    imgsz=768,
    epochs=120,
    batch=16,
    device=0,
    workers=8,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.001,

    # --------------------------------------------------------
    # Learning Rate
    # --------------------------------------------------------

    cos_lr=True,
    warmup_epochs=3.0,

    # --------------------------------------------------------
    # Data Augmentation
    # --------------------------------------------------------

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=5.0,
    translate=0.10,
    scale=0.50,
    shear=2.0,

    fliplr=0.5,

    # --------------------------------------------------------
    # Disabled Augmentations
    # --------------------------------------------------------

    flipud=0.0,
    perspective=0.0,
    mixup=0.0,
    copy_paste=0.0,

    # --------------------------------------------------------
    # Mosaic
    # --------------------------------------------------------

    mosaic=1.0,
    close_mosaic=10,

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    dropout=0.10,

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val=True,
    plots=True,

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    amp=True,
    cache=False,

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    patience=20,

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    save=True,
    save_period=1,

    # IMPORTANT:
    # Do NOT use /content/runs/fire_smoke here.
    # W&B interprets project as its project name.
    project="runs",
    name="fire_smoke",
    exist_ok=True,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    verbose=True
)

# ============================================================
# FINAL COPY TO GOOGLE DRIVE
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECKPOINT COPY")
print("=" * 70)

LOCAL_BEST = os.path.join(
    LOCAL_RUN,
    "weights",
    "best.pt"
)

LOCAL_LAST = os.path.join(
    LOCAL_RUN,
    "weights",
    "last.pt"
)

# Save final BEST
if os.path.exists(LOCAL_BEST):

    shutil.copy2(
        LOCAL_BEST,
        DRIVE_BEST
    )

    print(f"Best model copied to:")
    print(DRIVE_BEST)

else:

    print("WARNING: best.pt not found.")

# Save final LAST
if os.path.exists(LOCAL_LAST):

    shutil.copy2(
        LOCAL_LAST,
        DRIVE_LAST
    )

    print(f"Last model copied to:")
    print(DRIVE_LAST)

else:

    print("WARNING: last.pt not found.")

# ============================================================
# TRAINING COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"Local training directory:")
print(LOCAL_RUN)

print()

print(f"Best model:")
print(DRIVE_BEST)

print()

print(f"Last model:")
print(DRIVE_LAST)

print("=" * 70)
print("ALL DONE")
print("=" * 70)