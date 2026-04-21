import os
import sys
import subprocess
import yaml
import shutil
from pathlib import Path

def create_dataset_structure(dataset_dir):
    """创建数据集目录结构"""
    os.makedirs(os.path.join(dataset_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test', 'labels'), exist_ok=True)

def create_yolo_config(config_path):
    """创建YOLOv5配置文件"""
    dataset_dir = os.path.join(os.path.dirname(config_path), 'datasets')
    config = {
        'path': dataset_dir,  # 数据集根目录（绝对路径）
        'train': 'train/images',  # 训练集图像目录
        'val': 'val/images',  # 验证集图像目录
        'test': 'test/images',  # 测试集图像目录
        
        'nc': 2,  # 类别数量
        'names': ['fire', 'smoke'],  # 类别名称
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

def clone_yolov5():
    """克隆YOLOv5仓库"""
    yolov5_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
    
    if not os.path.exists(yolov5_dir):
        print("正在克隆YOLOv5仓库...")
        subprocess.run([
            'git', 'clone', 'https://github.com/ultralytics/yolov5.git', 
            yolov5_dir, '--depth', '1'
        ], check=True)
    else:
        print("YOLOv5仓库已存在")
    
    return yolov5_dir

def train_model(yolov5_dir, config_path, epochs=25, batch_size=2, imgsz=64):
    """训练模型"""
    train_cmd = [
        sys.executable,
        os.path.join(yolov5_dir, 'train.py'),
        '--data', config_path,
        '--weights', 'yolov5s.pt',  # 使用预训练权重
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--imgsz', str(imgsz),
        '--cache', 'none',
        '--device', '0',
        '--workers', '0',  # 禁用多线程
        '--project', 'runs/train',
        '--name', 'fire_smoke_detector'
    ]
    
    print(f"开始训练模型，命令: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))

def export_model(yolov5_dir, weights_path):
    """导出ONNX模型"""
    export_cmd = [
        sys.executable,
        os.path.join(yolov5_dir, 'export.py'),
        '--weights', weights_path,
        '--include', 'onnx',
        '--img', '640'
    ]
    
    print(f"导出ONNX模型，命令: {' '.join(export_cmd)}")
    subprocess.run(export_cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'datasets')
    config_path = os.path.join(current_dir, 'fire_smoke.yaml')
    
    # 创建数据集目录结构
    create_dataset_structure(dataset_dir)
    
    # 创建YOLO配置文件
    create_yolo_config(config_path)
    
    # 克隆YOLOv5仓库
    yolov5_dir = clone_yolov5()
    
    # 安装依赖
    print("安装YOLOv5依赖...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-r',
        os.path.join(yolov5_dir, 'requirements.txt')
    ], check=True)
    
    print("=" * 80)
    print("训练准备完成！")
    print("=" * 80)
    
    # 检查是否有训练参数
    if '--train' in sys.argv:
        print("开始训练模型...")
        train_model(yolov5_dir, config_path)
        
        # 导出模型
        best_weights = os.path.join(current_dir, 'runs', 'train', 'fire_smoke_detector', 'weights', 'best.pt')
        if os.path.exists(best_weights):
            export_model(yolov5_dir, best_weights)
            print(f"模型训练完成！")
            print(f"最佳权重: {best_weights}")
            print(f"ONNX模型: {best_weights.replace('.pt', '.onnx')}")
        else:
            print("训练失败，未找到最佳权重文件")

if __name__ == "__main__":
    main()
