"""
YOLO11模型转ONNX格式转换脚本
将羽毛球检测模型和姿态估计模型转换为ONNX格式
"""
from ultralytics import YOLO
import argparse
import os


def convert_to_onnx(model_path, output_path=None, img_size=640, dynamic=False, simplify=True):
    """
    将YOLO11模型转换为ONNX格式

    Args:
        model_path: 输入模型路径(.pt文件)
        output_path: 输出ONNX模型路径(可选，默认在输入路径同目录)
        img_size: 输入图像尺寸，默认640
        dynamic: 是否使用动态输入尺寸，默认False
        simplify: 是否简化ONNX模型，默认True
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)

    # 设置输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.dirname(model_path)
        output_path = os.path.join(output_dir, f"{base_name}.onnx")

    print(f"开始转换到ONNX格式...")
    print(f"  - 输入尺寸: {img_size}")
    print(f"  - 动态输入: {dynamic}")
    print(f"  - 模型简化: {simplify}")

    # 导出为ONNX格式
    model.export(
        format='onnx',
        imgsz=img_size,
        dynamic=dynamic,
        simplify=simplify
    )

    # YOLO会自动在模型目录生成onnx文件，我们需要重命名或移动
    onnx_generated_path = model_path.replace('.pt', '.onnx')

    if os.path.exists(onnx_generated_path):
        if onnx_generated_path != output_path:
            import shutil
            shutil.move(onnx_generated_path, output_path)
        print(f"✓ ONNX模型已成功导出到: {output_path}")
    else:
        print(f"✗ ONNX模型导出失败")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='YOLO11模型转ONNX格式')
    parser.add_argument('--ball-model', type=str, default='weights/yolo11s-ball.pt',
                       help='羽毛球检测模型路径')
    parser.add_argument('--pose-model', type=str, default='weights/yolo11n-pose.pt',
                       help='姿态估计模型路径')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸，默认640')
    parser.add_argument('--dynamic', action='store_true',
                       help='启用动态输入尺寸')
    parser.add_argument('--no-simplify', action='store_true',
                       help='禁用模型简化')

    args = parser.parse_args()

    print("=" * 60)
    print("YOLO11模型转ONNX格式")
    print("=" * 60)

    # 转换羽毛球检测模型
    print("\n1. 转换羽毛球检测模型...")
    try:
        convert_to_onnx(
            model_path=args.ball_model,
            img_size=args.img_size,
            dynamic=args.dynamic,
            simplify=not args.no_simplify
        )
    except Exception as e:
        print(f"✗ 羽毛球检测模型转换失败: {e}")

    # 转换姿态估计模型
    print("\n2. 转换姿态估计模型...")
    try:
        convert_to_onnx(
            model_path=args.pose_model,
            img_size=args.img_size,
            dynamic=args.dynamic,
            simplify=not args.no_simplify
        )
    except Exception as e:
        print(f"✗ 姿态估计模型转换失败: {e}")

    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()