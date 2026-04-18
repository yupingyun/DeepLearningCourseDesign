import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
temp_yolov5_dir = os.path.join(script_dir, 'temp_yolov5')
sys.path.insert(0, temp_yolov5_dir)
os.chdir(temp_yolov5_dir)

def convert_pt_to_onnx():
    model_path = os.path.join(script_dir, "yolo_onnx/model/best.pt")
    output_path = os.path.join(script_dir, "weights/fire.onnx")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return False

    print("=" * 50)
    print("YOLOv5 PT 转 ONNX 转换工具")
    print("=" * 50)
    print(f"\n加载模型: {model_path}")

    try:
        import torch
        from models.experimental import attempt_load

        print("加载 YOLOv5 模型...")

        device = torch.device('cpu')
        model = attempt_load(model_path, device=device, inplace=True, fuse=True)
        model.eval()

        stride = int(model.stride.max())
        print(f"模型 stride: {stride}")

        img_size = 640
        dummy_input = torch.zeros(1, 3, img_size, img_size).to(device)

        print(f"\n开始转换为 ONNX 格式...")
        print(f"输出路径: {output_path}")

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size'}
                }
            )

        print("\n转换成功!")
        print(f"ONNX 模型已保存至: {output_path}")

        file_size = os.path.getsize(output_path)
        print(f"模型文件大小: {file_size / (1024*1024):.2f} MB")

        return True

    except Exception as e:
        import traceback
        print(f"\n转换失败: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = convert_pt_to_onnx()
    sys.exit(0 if success else 1)
