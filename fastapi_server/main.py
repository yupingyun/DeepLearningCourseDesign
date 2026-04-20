import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from yolo_onnx.proccess import YOLODetector, ModelType

def ensure_model_ready():
    """确保模型已准备就绪，如果需要则转换模型"""
    pt_model_path = os.path.join(script_dir, "yolo_onnx/model/best.pt")
    onnx_model_path = os.path.join(script_dir, "weights/fire.onnx")
    
    # 检查ONNX模型是否存在或是否需要更新
    if not os.path.exists(onnx_model_path) or os.path.getmtime(pt_model_path) > os.path.getmtime(onnx_model_path):
        print("正在转换模型...")
        # 运行模型转换脚本
        convert_script = os.path.join(script_dir, "convert_model.py")
        result = os.system(f"{sys.executable} {convert_script}")
        if result != 0:
            print("错误: 模型转换失败")
            return False
    return True

def main():
    # 确保模型已准备就绪
    if not ensure_model_ready():
        return
    
    model_path = os.path.join(script_dir, "weights/fire.onnx")
    input_dir = os.path.join(script_dir, "inputs")
    output_dir = os.path.join(script_dir, "outputs")

    print("=" * 50)
    print("YOLOv5 火灾检测")
    print("=" * 50)

    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在，已自动创建")
        os.makedirs(input_dir)
        print(f"请将待检测的图片或视频放入 {input_dir} 目录后重新运行程序")
        return

    image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_exts)]
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_exts)]

    if not image_files and not video_files:
        print(f"在 {input_dir} 目录中没有找到任何图片或视频文件")
        print(f"支持的图片格式: {', '.join(image_exts)}")
        print(f"支持的视频格式: {', '.join(video_exts)}")
        return

    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        model_type=ModelType.FIRE_SMOKE
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_fire_count = 0
    total_smoke_count = 0
    total_files = len(image_files) + len(video_files)
    processed = 0

    if image_files:
        print(f"\n发现 {len(image_files)} 张图片，开始检测...")
        print("-" * 50)

        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(input_dir, image_file)
            name_without_ext = os.path.splitext(image_file)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}_result.jpg")

            print(f"\n[{i}/{len(image_files)}] 检测图片: {image_file}")
            result = detector.process_image(image_path, output_path)

            fire_count = 0
            smoke_count = 0

            if result['detections']:
                for det in result['detections']:
                    class_id = int(det[5])
                    conf = det[4]
                    class_name = detector.CLASSES.get(class_id, 'unknown')
                    print(f"  - {class_name}: 置信度 {conf:.2f}")
                    if class_id == 0:
                        fire_count += 1
                    elif class_id == 1:
                        smoke_count += 1

            if fire_count > 0 or smoke_count > 0:
                print(f"  [!] 检测到火灾: {fire_count}, 烟雾: {smoke_count}")
            else:
                print(f"  [√] 未检测到火灾或烟雾")

            print(f"  结果已保存至: {output_path}")
            total_fire_count += fire_count
            total_smoke_count += smoke_count
            processed += 1

    if video_files:
        print(f"\n" + "-" * 50)
        print(f"发现 {len(video_files)} 个视频，开始检测...")
        print("-" * 50)

        for i, video_file in enumerate(video_files, 1):
            video_path = os.path.join(input_dir, video_file)
            name_without_ext = os.path.splitext(video_file)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}_result.mp4")

            print(f"\n[{len(image_files) + i}/{total_files}] 检测视频: {video_file}")
            result = detector.process_video(video_path, output_path, skip_frames=0)

            print(f"  处理帧数: {result['processed_frames']}")
            print(f"  检测到目标总数: {result['detection_count']}")
            print(f"  总处理时间: {result['processing_time']:.2f} 秒")

            if result['detection_count'] > 0:
                print(f"  [!] 检测到火灾或烟雾")
            else:
                print(f"  [√] 未检测到火灾或烟雾")

            print(f"  结果已保存至: {output_path}")
            processed += 1

    print("\n" + "=" * 50)
    print("检测完成!")
    print(f"总计检测文件: {total_files}")
    print(f"检测到火灾: {total_fire_count} 处")
    print(f"检测到烟雾: {total_smoke_count} 处")
    print(f"结果保存在: {output_dir} 目录")
    print("=" * 50)


if __name__ == "__main__":
    main()







