import os
import sys

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
YOLO11_PYTHON_DIR = os.path.join(REPO_ROOT, "yolov11_pose", "code", "python")
sys.path.insert(0, YOLO11_PYTHON_DIR)

from yolo11_pose import Yolo11Pose
from utils import draw_pose_res

image_path = os.path.join("/home/aidlux/2026_6_25/Good-Badminton-main/templates/demo.png")
model_path = os.path.join("/home/aidlux/2026_6_25/Good-Badminton-main/weights/yolo11n-pose_qcs8550_fp16.qnn236.ctx.bin.aidem")
output_path = os.path.join(SCRIPT_DIR, "result.jpg")

img = cv2.imread(image_path)
model = Yolo11Pose(model_path, width=640, height=640, model_type="QNN")
detections = model(img, invoke_nums=10, conf_thres=0.1, iou_thres=0.25)
result = draw_pose_res(img, detections)
cv2.imwrite(output_path, result)