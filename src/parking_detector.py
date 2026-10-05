from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck in COCO


@dataclass
class ParkingSlot:
    slot_id: int
    polygon: np.ndarray

    @property
    def area(self) -> float:
        return float(cv2.contourArea(self.polygon.astype(np.float32)))


def load_slots(path: str | Path) -> list[ParkingSlot]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ParkingSlot(int(item["id"]), np.asarray(item["polygon"], dtype=np.int32)) for item in data["slots"]]


def intersection_over_slot(slot: ParkingSlot, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    mask = np.zeros((max(y2, 1), max(x2, 1)), dtype=np.uint8)
    # Use a local canvas to avoid assumptions about frame dimensions.
    w = max(x2 + 1, int(slot.polygon[:, 0].max()) + 1)
    h = max(y2 + 1, int(slot.polygon[:, 1].max()) + 1)
    mask_slot = np.zeros((h, w), dtype=np.uint8)
    mask_box = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_slot, [slot.polygon], 1)
    cv2.rectangle(mask_box, (x1, y1), (x2, y2), 1, -1)
    return float(np.logical_and(mask_slot, mask_box).sum()) / max(slot.area, 1.0)


def slot_occupancy(slots: Iterable[ParkingSlot], boxes: list[tuple[int, int, int, int]], threshold: float = 0.20) -> dict[int, bool]:
    result = {}
    for slot in slots:
        result[slot.slot_id] = any(intersection_over_slot(slot, box) >= threshold for box in boxes)
    return result


def draw_status(frame: np.ndarray, slots: list[ParkingSlot], status: dict[int, bool]) -> np.ndarray:
    output = frame.copy()
    for slot in slots:
        occupied = status[slot.slot_id]
        pts = slot.polygon.reshape((-1, 1, 2))
        cv2.polylines(output, [pts], True, (0, 0, 255) if occupied else (0, 200, 0), 2)
        center = tuple(np.mean(slot.polygon, axis=0).astype(int))
        cv2.putText(output, f"{slot.slot_id}: {'OCC' if occupied else 'FREE'}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(output, f"{slot.slot_id}: {'OCC' if occupied else 'FREE'}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    free = sum(not value for value in status.values())
    cv2.rectangle(output, (10, 10), (220, 50), (30, 30, 30), -1)
    cv2.putText(output, f"Free: {free}/{len(slots)}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return output


def process_video(source: str, slots_path: str, model_path: str = "yolov8n.pt", output: str | None = None,
                  confidence: float = 0.35, overlap_threshold: float = 0.20) -> None:
    model = YOLO(model_path)
    slots = load_slots(slots_path)
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=confidence, verbose=False)[0]
        boxes: list[tuple[int, int, int, int]] = []
        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                if int(cls) in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box)
                    boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)

        status = slot_occupancy(slots, boxes, overlap_threshold)
        annotated = draw_status(frame, slots, status)

        if output and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25,
                                     (annotated.shape[1], annotated.shape[0]))
        if writer:
            writer.write(annotated)
        cv2.imshow("AI Parking Space Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time parking occupancy detector using YOLOv8 + OpenCV")
    parser.add_argument("--source", default="0", help="Video file, RTSP URL, or 0 for webcam")
    parser.add_argument("--slots", required=True, help="JSON file containing parking-slot polygons")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--overlap", type=float, default=0.20)
    args = parser.parse_args()
    process_video(args.source, args.slots, args.model, args.output, args.confidence, args.overlap)


if __name__ == "__main__":
    main()
