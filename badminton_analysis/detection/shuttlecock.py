from collections import deque
import os
import sys
import time

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

# Allow importing yolo11_pose from the sibling yolov11_pose project
_YOLO11_POSE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "yolov11_pose", "code", "python",
)
if _YOLO11_POSE_DIR not in sys.path:
    sys.path.insert(0, _YOLO11_POSE_DIR)


class QNNShuttlecockDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.input_shape = (1, 640, 640, 3)
        self.interpreter = None
        self.iou_thres = 0.45
        self.conf_threshold = 0.18
        self._init_model()

    def _init_model(self):
        import aidlite

        model = aidlite.Model.create_instance(self.model_path)
        if model is None:
            raise RuntimeError(f"创建模型失败: {self.model_path}")

        config = aidlite.Config.create_instance()
        if config is None:
            raise RuntimeError("创建配置失败")

        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        config.framework_type = aidlite.FrameworkType.TYPE_QNN236
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 0

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        if self.interpreter is None:
            raise RuntimeError("创建解释器失败")

        input_shapes = [[1, 640, 640, 3]]
        output_shapes = [[1, 5, 8400]]
        model.set_model_properties(input_shapes, aidlite.DataType.TYPE_FLOAT32,
                                   output_shapes, aidlite.DataType.TYPE_FLOAT32)

        result = self.interpreter.init()
        if result != 0:
            raise RuntimeError(f"解释器初始化失败: {result}")

        result = self.interpreter.load_model()
        if result != 0:
            raise RuntimeError(f"加载模型失败: {result}")

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img.flatten().astype(np.float32)

    def _sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

    def predict(self, frame, conf=0.25):
        input_data = self.preprocess(frame)

        result = self.interpreter.set_input_tensor(0, input_data)
        if result != 0:
            raise RuntimeError("设置输入张量失败")

        result = self.interpreter.invoke()
        if result != 0:
            raise RuntimeError("推理失败")

        out_data = self.interpreter.get_output_tensor(0)
        if out_data is None:
            return []

        flat_array = np.array(out_data, dtype=np.float32)
        total_elements = len(flat_array)

        num_channels = 5
        grid_sizes = [80, 40, 20]
        outputs = []

        offset = 0
        for grid_size in grid_sizes:
            grid_elements = num_channels * grid_size * grid_size
            if offset + grid_elements > total_elements:
                break
            output_slice = flat_array[offset:offset + grid_elements]
            output = output_slice.reshape(grid_size, grid_size, num_channels)
            outputs.append(output)
            offset += grid_elements

        if not outputs:
            return []

        return self._postprocess(outputs, frame.shape, conf)

    def _postprocess(self, outputs, frame_shape, conf_threshold=0.25):
        h, w = frame_shape[:2]
        scale_x = w / self.input_shape[2]
        scale_y = h / self.input_shape[1]

        stride = [8, 16, 32]
        all_boxes = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_channels = output.shape
            stride_i = stride[i] if i < len(stride) else 32

            # QNN 模型输出格式：cx, cy, bw, bh 已经是像素值，置信度需要归一化
            cx = output[..., 0] * scale_x
            cy = output[..., 1] * scale_y
            bw = output[..., 2] * scale_x
            bh = output[..., 3] * scale_y
            confidence = output[..., 4] / 640.0

            mask = confidence > conf_threshold
            if np.any(mask):
                for gy in range(grid_h):
                    for gx in range(grid_w):
                        if mask[gy, gx]:
                            center_x = cx[gy, gx]
                            center_y = cy[gy, gx]
                            width = bw[gy, gx]
                            height = bh[gy, gx]
                            conf = float(confidence[gy, gx])

                            all_boxes.append({
                                'center_x': center_x,
                                'center_y': center_y,
                                'width': width,
                                'height': height,
                                'confidence': conf,
                                'x1': center_x - width / 2,
                                'y1': center_y - height / 2,
                                'x2': center_x + width / 2,
                                'y2': center_y + height / 2,
                            })

        if not all_boxes:
            return []

        all_boxes.sort(key=lambda x: x['confidence'], reverse=True)

        selected_indices = []
        for i, box in enumerate(all_boxes):
            keep = True
            for j in selected_indices:
                iou = self._compute_iou(
                    [box['x1'], box['y1'], box['x2'], box['y2']],
                    [all_boxes[j]['x1'], all_boxes[j]['y1'], all_boxes[j]['x2'], all_boxes[j]['y2']]
                )
                if iou > self.iou_thres:
                    keep = False
                    break
            if keep:
                selected_indices.append(i)

        return [all_boxes[i] for i in selected_indices]

    def _compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def __del__(self):
        if hasattr(self, 'interpreter') and self.interpreter is not None:
            try:
                self.interpreter.destory()
            except Exception:
                pass


class Yolo11sBallDetector:
    """QNN shuttlecock detector powered by the same inference pipeline as
    ``yolo11_pose.Yolo11Pose``.

    Wraps the aidlite QNN runtime with padding-aware preprocessing and
    vectorised NMS postprocessing, exposing a ``predict(frame, conf)``
    method compatible with ``ShuttlecockTracker``.
    """

    def __init__(self, model_path, width=640, height=640, model_type="QNN236"):
        import aidlite

        self.model_path = model_path
        self.width = width
        self.height = height
        self.output_shape = [1, 5, 8400]
        self.iou_thres = 0.1

        model = aidlite.Model.create_instance(model_path)
        if model is None:
            raise RuntimeError(f"Create model failed: {model_path}")

        model.set_model_properties(
            [[1, height, width, 3]],
            aidlite.DataType.TYPE_FLOAT32,
            [self.output_shape],
            aidlite.DataType.TYPE_FLOAT32,
        )

        config = aidlite.Config.create_instance()
        if config is None:
            raise RuntimeError("Create config failed")
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        if model_type.upper() in ("QNN236",):
            config.framework_type = aidlite.FrameworkType.TYPE_QNN236
        elif model_type.upper() in ("QNN",):
            config.framework_type = aidlite.FrameworkType.TYPE_QNN
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1

        interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        if interpreter is None:
            raise RuntimeError("Build interpreter failed")
        if interpreter.init() != 0:
            raise RuntimeError("Interpreter init failed")
        if interpreter.load_model() != 0:
            raise RuntimeError("Interpreter load model failed")
        self.interpreter = interpreter

        # Import postprocess helpers from yolov11_pose utils
        from utils import nms, xywh2xyxy, clip_boxes
        self._nms = nms
        self._xywh2xyxy = xywh2xyxy
        self._clip_boxes = clip_boxes

    # ------------------------------------------------------------------
    def _preprocess(self, image):
        """Pad to square, resize to (width, height), normalise to [0,1].
        Returns (NHWC float32 tensor, scale)."""
        h, w = image.shape[:2]
        length = max(h, w)
        scale = length / self.width
        canvas = np.zeros((length, length, 3), dtype=np.uint8)
        canvas[:h, :w] = image
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = cv2.resize(canvas, (self.width, self.height),
                            interpolation=cv2.INTER_LINEAR)
        tensor = (canvas.astype(np.float32) / 255.0)[None, :]
        return tensor, scale

    # ------------------------------------------------------------------
    def predict(self, frame, conf=0.1):
        """Run detection and return a list of dicts, each with keys
        ``center_x, center_y, width, height, confidence, x1, y1, x2, y2``."""
        input_tensor, scale = self._preprocess(frame)

        if self.interpreter.set_input_tensor(0, input_tensor.data) != 0:
            raise RuntimeError("set_input_tensor failed")
        if self.interpreter.invoke() != 0:
            raise RuntimeError("interpreter invoke failed")

        output = self.interpreter.get_output_tensor(0)
        if output is None:
            return []
        output = np.asarray(output, dtype=np.float32).reshape(*self.output_shape)
        return self._postprocess(output, frame.shape, scale, conf)

    # ------------------------------------------------------------------
    def _postprocess(self, prediction, original_shape, scale,
                     conf_thres=0.1):
        prediction = np.asarray(prediction)
        if prediction.ndim == 3:
            prediction = prediction[0]          # [1,5,8400] -> [5,8400]
        if prediction.shape == (5, 8400):
            prediction = prediction.T            # -> [8400, 5]

        if prediction.ndim != 2 or prediction.shape != (8400, 5):
            raise ValueError(
                f"Expected prediction shape (8400, 5), got {prediction.shape}")

        scores = prediction[:, 4]                # raw objectness
        mask = scores >= conf_thres
        if not np.any(mask):
            return []

        pred = prediction[mask]
        scores = scores[mask]

        boxes = self._xywh2xyxy(pred[:, :4])     # cxcywh -> x1y1x2y2
        boxes *= scale                           # back to original image
        self._clip_boxes(boxes, original_shape)

        keep = self._nms(boxes, scores, self.iou_thres)

        detections = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            detections.append({
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(scores[idx]),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            })
        return detections

    # ------------------------------------------------------------------
    def __del__(self):
        interpreter = getattr(self, "interpreter", None)
        if interpreter is not None:
            try:
                interpreter.destory()
            except Exception:
                pass


class ShuttlecockTracker:
    """Detect, filter, track, and draw shuttlecock positions."""

    def __init__(
        self,
        yolo_ball_model,
        trajectory_length=30,
        show_trajectory=True,
        show_performance_stats=False,
        max_jump_pixels=220,
        prediction_gate_pixels=260,
        max_missing_frames=5,
        roi_padding_ratio=0.08,
        max_box_area_ratio=0.004,
        max_aspect_ratio=4.0,
    ):
        self.yolo_ball_model = yolo_ball_model
        self.trajectory_length = trajectory_length
        self.show_trajectory = show_trajectory
        self.show_performance_stats = show_performance_stats
        self.max_jump_pixels = max_jump_pixels
        self.prediction_gate_pixels = prediction_gate_pixels
        self.max_missing_frames = max_missing_frames
        self.roi_padding_ratio = roi_padding_ratio
        self.max_box_area_ratio = max_box_area_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.shuttlecock_trajectory = deque(maxlen=trajectory_length)
        self.last_valid_position = None
        self.last_candidate = None
        self.last_detection = self._empty_detection_state()
        self.missing_frames = 0

        self.is_qnn = isinstance(yolo_ball_model, (QNNShuttlecockDetector, Yolo11sBallDetector))

        if not self.is_qnn:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                self.ultra_device = 0
            else:
                self.ultra_device = "cpu"

    def detect_ball(self, frame, conf=0.1, roi_corners=None):
        t0 = time.time()

        if self.is_qnn:
            qnn_boxes = self.yolo_ball_model.predict(frame, conf=conf)
            candidates = self._extract_candidates_from_qnn(qnn_boxes, frame.shape, roi_corners)
        else:
            try:
                ball_results = self.yolo_ball_model(frame, conf=conf, device=self.ultra_device, verbose=False)[0]
            except TypeError:
                ball_results = self.yolo_ball_model(frame, conf=conf, verbose=False)[0]

            candidates = self._extract_candidates(ball_results, frame.shape, roi_corners)

        if self.show_performance_stats:
            print(f"Shuttlecock inference took {time.time() - t0:.2f} sec")

        selected = self._select_candidate(candidates)
        self.last_candidate = selected
        self.last_detection = {
            "visible": selected is not None,
            "accepted": False,
            "image": list(selected["point"]) if selected else None,
            "confidence": selected["confidence"] if selected else None,
            "candidate_count": len(candidates),
        }
        return list(selected["point"]) if selected else [0, 0]

    def _extract_candidates_from_qnn(self, qnn_boxes, frame_shape, roi_corners):
        frame_area = max(1, frame_shape[0] * frame_shape[1])
        candidates = []

        for box in qnn_boxes:
            center_x, center_y = box['center_x'], box['center_y']
            width, height = box['width'], box['height']
            confidence = box['confidence']

            if width <= 0 or height <= 0:
                continue

            point = (int(center_x), int(center_y))
            area_ratio = (width * height) / frame_area
            aspect_ratio = max(width / height, height / width)

            if area_ratio > self.max_box_area_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            if not self._point_in_roi(point, roi_corners):
                continue

            candidates.append({
                "point": point,
                "confidence": float(confidence),
                "area_ratio": float(area_ratio),
                "aspect_ratio": float(aspect_ratio),
            })

        return candidates

    def update_trajectory(self, ball_position, roi_corners=None):
        if ball_position == [0, 0] or ball_position is None:
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        point = tuple(ball_position)
        if not self._point_in_roi(point, roi_corners):
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        if self._is_outlier(point):
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        self._append_valid_point(point)
        self.last_detection["accepted"] = True
        self.last_detection["image"] = list(point)
        return list(point)

    def _extract_candidates(self, ball_results, frame_shape, roi_corners):
        boxes = ball_results.boxes
        if boxes is None or boxes.xywh.shape[0] < 1:
            return []

        xywh = boxes.xywh.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xywh))
        frame_area = max(1, frame_shape[0] * frame_shape[1])

        candidates = []
        for box, confidence in zip(xywh, confidences):
            center_x, center_y, width, height = [float(value) for value in box]
            if width <= 0 or height <= 0:
                continue

            point = (int(center_x), int(center_y))
            area_ratio = (width * height) / frame_area
            aspect_ratio = max(width / height, height / width)
            if area_ratio > self.max_box_area_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            if not self._point_in_roi(point, roi_corners):
                continue

            candidates.append(
                {
                    "point": point,
                    "confidence": float(confidence),
                    "area_ratio": float(area_ratio),
                    "aspect_ratio": float(aspect_ratio),
                }
            )

        return candidates

    def _select_candidate(self, candidates):
        if not candidates:
            return None

        if not self.shuttlecock_trajectory:
            return max(candidates, key=lambda item: item["confidence"])

        predicted = self._predict_next_position()

        def score(candidate):
            distance = self._distance(candidate["point"], predicted)
            size_penalty = candidate["area_ratio"] * 4000
            return candidate["confidence"] * 1000 - distance * 1.4 - size_penalty

        return max(candidates, key=score)

    def _point_in_roi(self, point, roi_corners):
        if roi_corners is None:
            return True

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        padding = int(max(x2 - x1, y2 - y1) * self.roi_padding_ratio)
        return (x1 - padding) <= point[0] <= (x2 + padding) and (y1 - padding) <= point[1] <= (y2 + padding)

    def _is_outlier(self, point):
        if not self.shuttlecock_trajectory:
            return False

        last_point = self.shuttlecock_trajectory[-1]
        jump_distance = self._distance(point, last_point)
        strict_gate = self.missing_frames <= self.max_missing_frames
        if jump_distance > self.max_jump_pixels and strict_gate:
            return True

        predicted = self._predict_next_position()
        predicted_distance = self._distance(point, predicted)
        if predicted_distance > self.prediction_gate_pixels and strict_gate:
            return True

        return False

    def _predict_next_position(self):
        if len(self.shuttlecock_trajectory) < 2:
            return self.shuttlecock_trajectory[-1]

        prev_x, prev_y = self.shuttlecock_trajectory[-2]
        last_x, last_y = self.shuttlecock_trajectory[-1]
        return (last_x + (last_x - prev_x), last_y + (last_y - prev_y))

    def _append_valid_point(self, point):
        self.shuttlecock_trajectory.append(point)
        self.last_valid_position = point
        self.missing_frames = 0

    def _record_missing_detection(self):
        self.missing_frames += 1
        if self.missing_frames > self.max_missing_frames:
            self.last_valid_position = None

    def _mark_detection_rejected(self):
        self.last_detection["accepted"] = False
        self.last_detection["image"] = None

    def _empty_detection_state(self):
        return {
            "visible": False,
            "accepted": False,
            "image": None,
            "confidence": None,
            "candidate_count": 0,
        }

    def _distance(self, point_a, point_b):
        return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))

    def draw_trajectory(self, frame):
        if not self.shuttlecock_trajectory:
            return

        t0 = time.time()
        color = (87, 108, 255)
        points = list(self.shuttlecock_trajectory)

        for i, point in enumerate(points):
            radius = int(3 + (i / len(points)) * 4)
            cv2.circle(frame, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)

        latest_point = points[-1]
        cv2.circle(frame, latest_point, 6, (0, 165, 255), thickness=-1, lineType=cv2.LINE_AA)

        if self.show_performance_stats:
            print(f"Drawing shuttlecock trajectory took {time.time() - t0:.2f} sec")

    def handle_visualization(self, frame):
        if self.show_trajectory and self.shuttlecock_trajectory:
            self.draw_trajectory(frame)

    def clear_trajectory(self):
        self.shuttlecock_trajectory.clear()
        self.last_valid_position = None
        self.last_candidate = None
        self.last_detection = self._empty_detection_state()
        self.missing_frames = 0

    def get_trajectory(self):
        return list(self.shuttlecock_trajectory)

    def get_last_detection(self):
        return dict(self.last_detection)