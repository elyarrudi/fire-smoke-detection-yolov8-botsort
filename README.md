# Fire and Smoke Detection

Video detection of fire and smoke with YOLOv8m, BoT-SORT tracking, and an alert when the same track stays visible for 5 consecutive frames. Inference uses the exported ONNX model. CUDA is used when it is available; otherwise the script falls back to CPU.

## Result

![Fire and smoke sample 1](sample1.gif)

![Fire and smoke sample 2](sample2.gif)

![Fire and smoke sample 3](sample3.gif)

| | Sample |
| --- | --- |
| ![Sample 1](1.png) | ![Sample 2](2.png) |
| ![Sample 3](3.png) | ![Sample 4](4.png) |

Test metrics from `train_result_with_pt/metrics_summary.json` and `train_result_with_onnx/metrics_summary.json`:

| Metric | PyTorch test | ONNX test |
| --- | ---: | ---: |
| Precision | 0.787 | 0.789 |
| Recall | 0.736 | 0.734 |
| mAP50 | 0.789 | 0.781 |
| mAP50-95 | 0.454 | 0.445 |

ONNX test mAP50 is 0.008 below the PyTorch test number. ONNX test mAP50-95 is 0.009 below it.

Validation, from the same files:

| Metric | PyTorch val | ONNX val |
| --- | ---: | ---: |
| Precision | 0.798 | 0.793 |
| Recall | 0.728 | 0.724 |
| mAP50 | 0.787 | 0.783 |
| mAP50-95 | 0.465 | 0.457 |

Per-class test mAP50-95, class 0 then class 1, in the order stored in those JSON files (`smoke`, `fire`):

| Class | PyTorch | ONNX |
| --- | ---: | ---: |
| Smoke | 0.524 | 0.513 |
| Fire | 0.385 | 0.378 |

`train_result/train_result_with_pt/threshold_comparison.csv` compares confidence thresholds. The video script uses confidence 0.35.

## What the video script does

`main.py` reads a video, runs `fire_smoke_training/best.onnx` through Ultralytics, tracks with BoT-SORT, and writes an annotated MP4.

| Setting | Value in `main.py` |
| --- | --- |
| Image size | 768 |
| Confidence | 0.35 |
| IoU | 0.45 |
| Max detections | 300 |
| Classes | 0 smoke, 1 fire |
| ReID | off (`USE_REID = False`) |
| Alert | 5 consecutive frames for one track id |
| Duplicate alerts | a track id is alerted once |

Smoke is drawn in orange and fire in red. After the alert condition, the frame shows `ALERT: Fire` or `ALERT: Smoke`. The script prints a line such as `[ALERT] Frame 245: Fire detected for 5 consecutive frames`.

On startup it rewrites `custom_botsort.yaml` from the BoT-SORT file shipped with the installed Ultralytics package. That yaml is gitignored.

## Train and export

Training is `train_code/train.py`. It fine-tunes `yolov8m.pt` on the [Smoke-Fire-Detection-YOLO](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) dataset, a YOLO-format version of D-Fire. The dataset is not in this repository. `train_result_with_pt/data.yaml` records 14,122 training images, 3,099 validation images, and 4,306 test images.

Training settings in `train.py`, and a 120-epoch curve in `train_result/train_result_with_pt/results/results.csv`:

| Parameter | Value |
| --- | --- |
| Image size | 768 |
| Epochs | 120 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial LR / final LR factor | 0.001 / 0.01 |
| Weight decay | 0.001 |
| Schedule | cosine, 3 warmup epochs |
| Early-stopping patience | 20 |
| AMP | on |
| Dropout | 0.10 |
| Mosaic / close mosaic | 1.0 / 10 |
| Seed | 42 |

Augmentation in that script: HSV jitter, rotation 5°, translation 0.10, scale 0.50, shear 2°, horizontal flip 0.5. Vertical flip, perspective, mixup, and copy-paste are off.

Export with the checkpoint present:

```powershell
python train_code/export_onnx.py
```

`train_code/export_onnx.py` writes `fire_smoke_training/best.onnx` at opset 12, image size 768, static shape, FP32, with `simplify=True`.

`evaluate_pt.py` and `evaluate_onnx.py` produced the JSON metrics above. Both scripts are gitignored.

## Run

Pass the input video and the output video. Sample outputs already in the folder are `sample1.mp4`, `sample2.mp4`, and `sample3.mp4`.

```powershell
python main.py 1.mp4 output_video1.mp4
```

Docker, from this folder. The image is CUDA 12.8. GPU inference needs an NVIDIA driver and the NVIDIA Container Toolkit. Without `--gpus all`, the script uses CPU.

Windows PowerShell:

```powershell
docker build -t fire-smoke .
docker run --gpus all --rm -v "${PWD}:/app" fire-smoke python3 main.py 1.mp4 output_video_docker.mp4
```

Linux:

```bash
docker build -t fire-smoke .
docker run --gpus all --rm -v "$(pwd):/app" fire-smoke python3 main.py 1.mp4 output_video_docker.mp4
```

CPU-only containers use the same command without `--gpus all`.

## Project files

```text
fire-smoke/
├── main.py
├── Dockerfile
├── requirements.txt
├── train_code/
│   ├── train.py
│   └── export_onnx.py
├── fire_smoke_training/
│   ├── best.pt
│   └── best.onnx
├── train_result_with_pt/
├── train_result_with_onnx/
├── train_result/train_result_with_pt/
├── sample1.gif, sample2.gif, sample3.gif
└── sample1.mp4, sample2.mp4, sample3.mp4
```

`dataset/`, input videos `1.mp4` `2.mp4` `3.mp4`, and `yolo26n-cls.pt` are gitignored. The last file is not used by this pipeline.

## Limits

- Fire test mAP50-95 is lower than smoke (0.385 vs 0.524 on the PyTorch test numbers).
- The alert rule is temporal persistence of a track, not a separate fire classifier.
- ReID is off unless `USE_REID` is edited in `main.py`.
- Real-time speed depends on the video, the image size of 768, and the GPU.
