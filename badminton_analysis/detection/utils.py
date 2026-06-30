from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


SKELETON: Tuple[Tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


def preprocess_image(image: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    length = max(height, width)
    scale = length / input_size
    canvas = np.zeros((length, length, 3), dtype=np.uint8)
    canvas[:height, :width] = image
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = cv2.resize(canvas, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return (canvas.astype(np.float32) / 255.0)[None, :], scale


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def clip_boxes(boxes: np.ndarray, shape: Sequence[int]) -> None:
    boxes[:, 0].clip(0, shape[1], out=boxes[:, 0])
    boxes[:, 1].clip(0, shape[0], out=boxes[:, 1])
    boxes[:, 2].clip(0, shape[1], out=boxes[:, 2])
    boxes[:, 3].clip(0, shape[0], out=boxes[:, 3])


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-12)
        order = rest[iou <= iou_thres]
    return keep


def pose_postprocess(
    prediction: np.ndarray,
    original_shape: Sequence[int],
    scale: float,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> List[dict]:
    prediction = np.asarray(prediction)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {prediction.shape}")
        prediction = prediction[0]
    if prediction.shape == (56, 8400):
        prediction = prediction.T
    if prediction.ndim != 2 or prediction.shape != (8400, 56):
        raise ValueError(f"Expected prediction shape (8400, 56), got {prediction.shape}")

    scores = prediction[:, 4]
    mask = scores >= conf_thres
    if not np.any(mask):
        return []

    pred = prediction[mask]
    scores = scores[mask]
    boxes = xywh2xyxy(pred[:, :4])
    boxes *= scale
    clip_boxes(boxes, original_shape)
    keypoints = pred[:, 5:56].reshape(-1, 17, 3).copy()
    keypoints[:, :, :2] *= scale
    keypoints[:, :, 0].clip(0, original_shape[1], out=keypoints[:, :, 0])
    keypoints[:, :, 1].clip(0, original_shape[0], out=keypoints[:, :, 1])

    keep = nms(boxes, scores, iou_thres)[:max_det]
    detections: List[dict] = []
    for idx in keep:
        detections.append({"box": boxes[idx].astype(np.float32), "score": float(scores[idx]), "keypoints": keypoints[idx].astype(np.float32)})
    return detections


def draw_pose_res(image: np.ndarray, detections: Iterable[dict], kpt_thres: float = 0.25) -> np.ndarray:
    result = image.astype(np.uint8).copy()
    for i, det in enumerate(detections, start=1):
        box = np.asarray(det["box"], dtype=np.float32)
        score = float(det["score"])
        keypoints = np.asarray(det["keypoints"], dtype=np.float32)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        print(i, [x1, y1, x2, y2], score, "person")
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f"person {score:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for x, y, conf in keypoints:
            if conf >= kpt_thres:
                cv2.circle(result, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
        for a, b in SKELETON:
            if keypoints[a, 2] >= kpt_thres and keypoints[b, 2] >= kpt_thres:
                pa = int(round(keypoints[a, 0])), int(round(keypoints[a, 1]))
                pb = int(round(keypoints[b, 0])), int(round(keypoints[b, 1]))
                cv2.line(result, pa, pb, (255, 0, 0), 2)
    return result
