from __future__ import annotations

import time

import aidlite
import numpy as np

from .utils import pose_postprocess, preprocess_image


class Yolo11Pose:
    def __init__(
        self,
        model_path: str,
        width: int = 640,
        height: int = 640,
        model_type: str = "qnn",
    ):
        self.width = width
        self.height = height
        self.output_shape = [1, 56, 8400]

        model = aidlite.Model.create_instance(model_path)
        if model is None:
            raise RuntimeError("Create model failed")
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
        if model_type.lower() == "qnn":
            config.framework_type = aidlite.FrameworkType.TYPE_QNN
        elif model_type.lower() in ("snpe", "snpe2"):
            config.framework_type = aidlite.FrameworkType.TYPE_SNPE2
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

    def __del__(self):
        interpreter = getattr(self, "interpreter", None)
        if interpreter is not None:
            interpreter.destory()

    def __call__(self, image: np.ndarray, invoke_nums: int = 10, conf_thres: float = 0.5, iou_thres: float = 0.45):
        input_tensor, scale = preprocess_image(image, self.width)
        if self.interpreter.set_input_tensor(0, input_tensor.data) != 0:
            raise RuntimeError("set_input_tensor failed")

        invoke_time = []
        for _ in range(invoke_nums):
            t1 = time.time()
            result = self.interpreter.invoke()
            invoke_time.append((time.time() - t1) * 1000)
            if result != 0:
                raise RuntimeError("interpreter invoke failed")

        output = self.interpreter.get_output_tensor(0).reshape(*self.output_shape)
        return pose_postprocess(output, image.shape, scale, conf_thres=conf_thres, iou_thres=iou_thres)
