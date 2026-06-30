import os
import sys

import cv2
import numpy as np


# Allow importing yolo11_pose from the sibling yolov11_pose project
_YOLO11_POSE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "yolov11_pose", "code", "python",
)
if _YOLO11_POSE_DIR not in sys.path:
    sys.path.insert(0, _YOLO11_POSE_DIR)


class QNNPoseProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.conf = 0.5
        self.iou_thres = 0.45
        self.inference_name = "QNN-Pose"
        self.input_shape = (1, 640, 640, 3)
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
        img = np.expand_dims(img, axis=0)
        return img.flatten().astype(np.float32)

    def process_frame(self, frame):
        input_data = self.preprocess(frame)

        result = self.interpreter.set_input_tensor(0, input_data)
        if result != 0:
            raise RuntimeError("设置输入张量失败")

        result = self.interpreter.invoke()
        if result != 0:
            raise RuntimeError("推理失败")

        out_data = self.interpreter.get_output_tensor(0)
        if out_data is None:
            return None, None

        flat_array = np.array(out_data, dtype=np.float32)
        total_elements = len(flat_array)

        num_channels = 56
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
            return None, None

        return self._postprocess(outputs, frame.shape)

    def _postprocess(self, outputs, frame_shape):
        h, w = frame_shape[:2]
        scale_x = w / self.input_shape[2]
        scale_y = h / self.input_shape[1]

        stride = [8, 16, 32]
        all_detections = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_channels = output.shape
            stride_i = stride[i] if i < len(stride) else 32

            num_keypoints = 17
            for gy in range(grid_h):
                for gx in range(grid_w):
                    confidence = output[gy, gx, 4] / 640.0
                    if confidence < self.conf:
                        continue

                    cx = output[gy, gx, 0] * scale_x
                    cy = output[gy, gx, 1] * scale_y
                    bw = output[gy, gx, 2] * scale_x
                    bh = output[gy, gx, 3] * scale_y

                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2

                    keypoints = []
                    scores = []
                    for k in range(num_keypoints):
                        kx = output[gy, gx, 5 + k * 3] * scale_x
                        ky = output[gy, gx, 5 + k * 3 + 1] * scale_y
                        ks = output[gy, gx, 5 + k * 3 + 2]
                        keypoints.append([kx, ky])
                        scores.append(ks)

                    all_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'keypoints': keypoints,
                        'scores': scores
                    })

        if not all_detections:
            return None, None

        all_detections.sort(key=lambda x: x['confidence'], reverse=True)

        selected_indices = []
        for i, det in enumerate(all_detections):
            keep = True
            for j in selected_indices:
                iou = self._compute_iou(det['bbox'], all_detections[j]['bbox'])
                if iou > self.iou_thres:
                    keep = False
                    break
            if keep:
                selected_indices.append(i)

        all_keypoints = []
        all_scores = []
        for i in selected_indices:
            all_keypoints.append(all_detections[i]['keypoints'])
            all_scores.append(all_detections[i]['scores'])

        if not all_keypoints:
            return None, None

        return np.array(all_keypoints), np.array(all_scores)

    def _sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

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


class YOLOPoseProcessor:
    """Ultralytics YOLO pose processor with COCO 17 keypoint output."""

    def __init__(self, model_path="yolo11n-pose.pt", device="auto", conf=0.25):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf = conf
        self.inference_name = "YOLO-Pose"
        if device in (None, "auto"):
            selected = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    selected = 0
            except Exception:
                selected = "cpu"
            self.device = selected
        else:
            self.device = device

        print(f"Initializing YOLO pose model (model: {self.model_path}, device: {self.device})")
        self.model = YOLO(self.model_path)

    def process_frame(self, frame):
        result = self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return None, None

        keypoints = result.keypoints.xy
        scores = result.keypoints.conf
        if keypoints.shape[0] == 0:
            return None, None

        keypoints = keypoints.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy() if scores is not None else None
        return keypoints, scores


class Yolo11PoseQNNProcessor:
    """QNN pose processor powered by yolov11_pose / aidlite inference pipeline.

    Wraps ``yolo11_pose.Yolo11Pose`` so that it exposes the same
    ``process_frame(frame) -> (keypoints, scores)`` interface used by
    `PlayerPoseVisualizer`, making it a drop-in replacement for
    ``QNNPoseProcessor`` / ``YOLOPoseProcessor``.
    """

    def __init__(self, model_path, width=640, height=640,
                 model_type="QNN", conf=0.1, iou_thres=0.1):
        from yolo11_pose import Yolo11Pose

        self.model_path = model_path
        self.width = width
        self.height = height
        self.model_type = model_type
        self.conf = conf
        self.iou_thres = iou_thres
        self.inference_name = "Yolo11Pose-QNN"

        print(f"Initializing Yolo11Pose-QNN model: {self.model_path}")
        self.model = Yolo11Pose(
            model_path=self.model_path,
            width=self.width,
            height=self.height,
            model_type=self.model_type,
        )

    def process_frame(self, frame):
        """Run pose detection on *frame* (BGR uint8 image).

        Returns
        -------
        keypoints : numpy.ndarray or None
            Shape ``(N, 17, 2)`` where N is the number of detected persons.
        scores : numpy.ndarray or None
            Shape ``(N, 17)`` with per-keypoint confidence values.
        """
        if frame is None:
            return None, None

        detections = self.model(
            frame,
            invoke_nums=1,
            conf_thres=self.conf,
            iou_thres=self.iou_thres,
            verbose=False,
        )

        if not detections:
            return None, None

        all_kpts = []
        all_scores = []
        for det in detections:
            kpts = np.asarray(det["keypoints"])          # (17, 3)  [x, y, conf]
            all_kpts.append(kpts[:, :2])                  # (17, 2)
            all_scores.append(kpts[:, 2])                 # (17,)

        return np.stack(all_kpts, axis=0), np.stack(all_scores, axis=0)

    def __del__(self):
        if hasattr(self, "model") and self.model is not None:
            try:
                del self.model
            except Exception:
                pass