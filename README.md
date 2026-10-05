# AI Parking Space Detector

Real-time parking occupancy detection using **Python, YOLOv8 and OpenCV**. The project detects vehicles in a video stream and determines whether configured parking polygons are occupied using vehicle/slot overlap.

## Features

- YOLOv8 vehicle detection (car, motorcycle, bus, truck)
- OpenCV webcam/video/RTSP processing
- Polygon-based parking-slot configuration
- Configurable confidence and occupancy-overlap thresholds
- Live free/occupied count overlay
- Optional annotated MP4 output
- Unit test for occupancy logic

## Architecture

`Video source -> YOLOv8 -> vehicle bounding boxes -> slot overlap -> FREE/OCCUPIED -> OpenCV overlay`

Parking polygons are intentionally stored separately in `data/parking_slots.json`, so the same detector can be used with different parking lots.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the selected Ultralytics model (for example `yolov8n.pt`).

## Run

```bash
python src/parking_detector.py --source 0 --slots data/parking_slots.json
```

Video file:

```bash
python src/parking_detector.py --source parking.mp4 --slots data/parking_slots.json --output result.mp4
```

Press `q` to stop.

## Slot configuration

Replace the example polygons with coordinates from your camera view. Each polygon is a parking-space boundary. A slot is marked occupied when the intersection between the vehicle bounding box and the slot polygon reaches the configured overlap threshold.

## Evaluation

Do not claim a fixed accuracy unless you have measured it on a labeled dataset. For a portfolio benchmark, create frame-level ground truth and report accuracy, precision, recall and F1. The repository deliberately keeps the detector's metric configurable rather than embedding an unsupported 98% claim.

## GitHub

```bash
git init
git add .
git commit -m "Initial parking detector"
git branch -M main
git remote add origin https://github.com/<username>/ai-parking-space-detector.git
git push -u origin main
```

## License

MIT
