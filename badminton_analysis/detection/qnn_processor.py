import cv2
import numpy as np


class QNNProcessor:
    def __init__(self, model_path, input_shape, output_shapes, framework_type=None, accelerate_type=None):
        self.model_path = model_path
        self.input_shape = input_shape
        self.output_shapes = output_shapes
        self.framework_type = framework_type
        self.accelerate_type = accelerate_type
        
        self.interpreter = None
        self.input_tensor_size = None
        self._init_model()

    def _init_model(self):
        import aidlite
        
        model = aidlite.Model.create_instance(self.model_path)
        if model is None:
            raise RuntimeError(f"创建模型失败: {self.model_path}")
        
        input_shapes = [self.input_shape]
        output_shapes = self.output_shapes
        
        dtype = aidlite.DataType.TYPE_FLOAT32
        if "fp16" in self.model_path.lower():
            dtype = aidlite.DataType.TYPE_FLOAT16
        
        model.set_model_properties(input_shapes, dtype, output_shapes, dtype)

        config = aidlite.Config.create_instance()
        if config is None:
            raise RuntimeError("创建配置失败")
        
        if self.framework_type is not None:
            config.framework_type = self.framework_type
        else:
            config.framework_type = aidlite.FrameworkType.TYPE_QNN
        
        if self.accelerate_type is not None:
            config.accelerate_type = self.accelerate_type
        else:
            config.accelerate_type = aidlite.AccelerateType.TYPE_NPU

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        if self.interpreter is None:
            raise RuntimeError("创建解释器失败")

        result = self.interpreter.init()
        if result != 0:
            raise RuntimeError(f"解释器初始化失败: {result}")

        result = self.interpreter.load_model()
        if result != 0:
            raise RuntimeError(f"加载模型失败: {result}")

        _, _, _, channels = self.input_shape
        self.input_tensor_size = self.input_shape[1] * self.input_shape[2] * channels

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img.flatten().astype(np.float32)

    def predict(self, frame):
        input_data = self.preprocess(frame)
        
        result = self.interpreter.set_input_tensor(0, input_data)
        if result != 0:
            raise RuntimeError("设置输入张量失败")

        result = self.interpreter.invoke()
        if result != 0:
            raise RuntimeError("推理失败")

        outputs = []
        for i, shape in enumerate(self.output_shapes):
            out_data = self.interpreter.get_output_tensor(i)
            if out_data is None:
                raise RuntimeError(f"获取输出张量 {i} 失败")
            flat_size = np.prod(shape)
            out_array = np.array(out_data[:flat_size], dtype=np.float32).reshape(shape)
            outputs.append(out_array)
        
        return outputs

    def __del__(self):
        if hasattr(self, 'interpreter') and self.interpreter is not None:
            try:
                self.interpreter.destory()
            except Exception:
                pass