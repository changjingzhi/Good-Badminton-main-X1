#!/usr/bin/env python
"""Quick test script for the Yolo11sBallDetector (QNN shuttlecock detection).

Usage (from Good-Badminton-main/):
    python test_ball_detection.py
"""

import os
import sys

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR

# Allow importing yolo11_pose utils from the sibling project
YOLO11_PYTHON_DIR = os.path.join(
    os.path.dirname(REPO_ROOT), "yolov11_pose", "code", "python"
)
sys.path.insert(0, YOLO11_PYTHON_DIR)

from badminton_analysis.detection.shuttlecock import Yolo11sBallDetector

# --- Configuration ---
MODEL_PATH = os.path.join(
    REPO_ROOT, "weights", "yolo11s-ball_qcs8550_fp16.qnn236.ctx.bin.aidem"
)
IMAGE_PATH = os.path.join(REPO_ROOT, "templates", "demo.png")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "ball_result.jpg")
CONF_THRES = 0.25
IOU_THRES = 0.45

# --- Detection ---
detector = Yolo11sBallDetector(MODEL_PATH)
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(IMAGE_PATH)

detections = detector.predict(img, conf=CONF_THRES)

print(f"\n检测到 {len(detections)} 个羽毛球:")
for i, det in enumerate(detections, 1):
    print(
        f"  {i}. 中心=({det['center_x']:.0f},{det['center_y']:.0f})  "
        f"宽高=({det['width']:.0f},{det['height']:.0f})  "
        f"置信度={det['confidence']:.3f}"
    )

# --- Visualisation ---
for det in detections:
    x1, y1 = int(det["x1"]), int(det["y1"])
    x2, y2 = int(det["x2"]), int(det["y2"])
    cx, cy = int(det["center_x"]), int(det["center_y"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(
        img,
        f"{det['confidence']:.2f}",
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

cv2.imwrite(OUTPUT_PATH, img)
print(f"\n结果已保存至: {OUTPUT_PATH}")
