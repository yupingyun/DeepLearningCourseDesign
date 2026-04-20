import cv2
import numpy as np
import onnxruntime as rt
import os
import time
from typing import Dict, List, Tuple, Optional
from .YOLO import letterbox, non_max_suppression, scale_boxes
from dataclasses import dataclass
from enum import Enum
import logging
logger = logging.getLogger("yolo_onnx")
logging.basicConfig(level=logging.INFO)

# 本地定义ModelType以避免循环导入
class ModelType(Enum):
    FIRE_SMOKE = 1  # 火灾烟雾检测
    FALL = 2        # 摔倒检测

@dataclass
class DetectionResult:
    model_type: int  # 模型类型
    boxes: List[List[float]]  # [x1, y1, x2, y2, conf, class_id]

class YOLODetector:
    """YOLO目标检测器，用于图像和视频处理"""
    
    # 为每种模型类型定义类别字典
    MODEL_CLASSES = {
        ModelType.FIRE_SMOKE: {
            0: 'fire',  # 火灾
            1: 'smoke'  # 烟雾
        }
    }
    
    @property
    def CLASSES(self):
        """根据模型类型返回相应的类别字典"""
        return self.MODEL_CLASSES.get(self.model_type, {})
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, img_size: Tuple[int, int] = (640, 640),
                 model_type: ModelType = ModelType.FALL):
        """
        初始化YOLO检测器
        
        参数:
            model_path: ONNX模型路径
            conf_threshold: 检测置信度阈值
            iou_threshold: NMS的IoU阈值
            img_size: 模型输入大小 (宽度, 高度)
            model_type: 模型类型 (FIRE_SMOKE等)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model_type = model_type
        
        # 初始化模型会话
        self._initialize_model()
        logger.info(f"{model_type} 模型加载成功")
    
    def _initialize_model(self):
        """使用优化设置初始化ONNX模型"""
        try:
            # 检查可用的执行提供者
            providers = rt.get_available_providers()
            
            # 配置会话选项以进行优化
            session_options = rt.SessionOptions()
            session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            
            provider_options = []
            selected_providers = []
            
            # 尝试使用CUDA（如果可用）
            if 'CUDAExecutionProvider' in providers:
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                selected_providers.append('CUDAExecutionProvider')
                provider_options.append(cuda_options)
            
            # 始终添加CPU作为备选
            selected_providers.append('CPUExecutionProvider')
            provider_options.append({})
            
            # 尝试创建优化的会话
            try:
                self.session = rt.InferenceSession(
                    self.model_path,
                    sess_options=session_options,
                    providers=selected_providers,
                    provider_options=provider_options
                )
                logger.info(f"ONNX Runtime 使用: {self.session.get_providers()}")
                self.has_valid_model = True
                # 获取输入和输出名称
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            except Exception as e:
                logger.warning(f"加载ONNX模型失败: {str(e)}")
                logger.info("使用模拟检测进行演示。")
                self.has_valid_model = False
        except Exception as e:
            logger.warning(f"模型初始化期间出错: {str(e)}")
            logger.info("使用模拟检测进行演示。")
            self.has_valid_model = False
    
    def process_image(self, image_path: str, output_path: Optional[str] = None, 
                     draw_boxes: bool = True) -> Dict:
        """
        处理单个图像以进行目标检测
        
        参数:
            image_path: 输入图像路径
            output_path: 保存结果图像的路径（可选）
            draw_boxes: 是否在输出图像上绘制检测框
            
        返回:
            包含检测结果的字典，包括：
            - detections: 检测信息列表 [x1, y1, x2, y2, confidence, class_id]
            - detection_count: 检测到的目标数量
            - processed_image: 如果draw_boxes为True，则为带有框的OpenCV图像
            - output_path: 如果提供了output_path，则为保存结果的路径
        """
        # 检测目标
        img, detections = self._detect_objects(image_path)
        
        result = {
            'detections': detections,
            'detection_count': 0,
            'output_path': None
        }
        
        # 如果请求，绘制框
        if draw_boxes:
            result_img, detection_count = self._draw_detections(img, detections, self.conf_threshold)
            result['detection_count'] = detection_count
            result['processed_image'] = result_img
            
            # 如果提供了输出路径，保存结果
            if output_path:
                if self._save_result(result_img, output_path):
                    result['output_path'] = os.path.abspath(output_path)
        
        return result
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     skip_frames: int = 0, end_frame: Optional[int] = None,
                     draw_boxes: bool = True) -> Dict:
        """
        处理视频以进行目标检测
        
        参数:
            video_path: 输入视频路径
            output_path: 保存结果视频的路径（可选）
            skip_frames: 为了速度处理每N帧（0表示处理所有帧）
            end_frame: 在这一帧停止处理（可选）
            draw_boxes: 是否在输出视频上绘制检测框
            
        返回:
            包含检测结果的字典，包括：
            - frame_detections: 每个处理帧的检测列表
            - detection_count: 检测到的目标总数
            - processed_frames: 处理的帧数
            - output_path: 如果提供了output_path，则为保存结果的路径
            - processing_time: 总处理时间（秒）
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"错误: 无法在 {video_path} 打开视频")
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 未使用
        
        # 如果提供了输出路径，设置视频写入器
        writer = None
        if output_path and draw_boxes:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 处理视频
        result = {
            'frame_detections': [],
            'detection_count': 0,
            'processed_frames': 0,
            'output_path': None,
            'processing_time': 0
        }
        
        start_time = time.time()
        frame_index = 0
        total_detections = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检查是否到达结束帧
            if end_frame and frame_index >= end_frame:
                break
            
            # 处理每N帧 (skip_frames+1)
            if skip_frames == 0 or frame_index % (skip_frames+1) == 0:
                # 将帧保存到临时文件进行处理
                temp_frame_path = os.path.join(
                    os.path.dirname(output_path) if output_path else "temp",
                    f"temp_frame_{frame_index}.jpg"
                )
                
                # 必要时创建临时目录
                os.makedirs(os.path.dirname(temp_frame_path), exist_ok=True)
                
                cv2.imwrite(temp_frame_path, frame)
                
                try:
                    # 检测帧中的目标
                    _, detections = self._detect_objects(temp_frame_path)
                    
                    # 根据置信度阈值计算有效检测
                    valid_detections = [d for d in detections if float(d[4]) >= self.conf_threshold]
                    total_detections += len(valid_detections)
                    
                    result['frame_detections'].append({
                        'frame': frame_index,
                        'detections': detections
                    })
                    
                    # 如果请求，绘制框
                    if draw_boxes:
                        frame_with_boxes, _ = self._draw_detections(frame, detections, self.conf_threshold)
                        
                        if writer:
                            writer.write(frame_with_boxes)
                    
                    result['processed_frames'] += 1
                
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
            
            frame_index += 1
        
        # 清理
        cap.release()
        if writer:
            writer.release()
        
        # 计算处理时间
        end_time = time.time()
        result['processing_time'] = end_time - start_time
        result['detection_count'] = total_detections
        
        if output_path and os.path.exists(output_path):
            result['output_path'] = os.path.abspath(output_path)
        
        return result
    
    def _detect_objects(self, img_path: str) -> Tuple[np.ndarray, List]:
        """
        使用YOLO模型检测图像中的目标
        
        参数:
            img_path: 输入图像路径
            
        返回:
            img: 调整大小的原始图像
            detections: 检测信息列表 [x1, y1, x2, y2, confidence, class_id]
        """
        # 预处理图像
        img, im, _ = self._preprocess_image(img_path, self.img_size)
        
        # 推理
        pred = self._onnx_inference(im)
        
        # NMS
        pred = self._non_max_suppression(
            pred, 
            self.conf_threshold, 
            self.iou_threshold, 
            max_det=1000
        )
        
        detections = []
        
        for _, det in enumerate(pred):  # 每张图像
            if len(det):
                # 将框从img_size缩放到im0大小
                det[:, :4] = self._scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                detections = det[:, :6].tolist()
        
        return img, detections
    
    def _preprocess_image(self, img_path: str, img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """预处理图像以用于模型输入"""
        # 读取图像
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"错误: 无法读取 {img_path} 处的图像")
        else:
            # 假设img_path已经是一个numpy数组/图像
            img = img_path
        
        original_img = img.copy()
        
        # 此处不调整大小以保持宽高比，从而获得更准确的检测
        # img = cv2.resize(img, (640, 640))
        
        # 预处理 - letterbox保留宽高比，有助于提高检测准确性
        im, ratio, (dw, dh) = letterbox(original_img, img_size, auto=False, stride=32)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255  # 0 - 255 到 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # 扩展为batch维度
            
        return original_img, im, (ratio, (dw, dh))
    
    def _onnx_inference(self, data: np.ndarray) -> np.ndarray:
        """使用ONNX模型运行推理或返回模拟结果"""
        if self.has_valid_model:
            try:
                pred_onnx = self.session.run(
                    [self.output_name], 
                    {self.input_name: data.reshape(1, 3, 640, 640).astype(np.float32)}
                )
                # 在返回前从列表转换为numpy数组
                return pred_onnx[0]  # 提取列表的第一个元素
            except Exception as e:
                logger.warning(f"推理错误: {str(e)}")
                return self._generate_mock_results()
        else:
            return self._generate_mock_results()
            
    def _generate_mock_results(self):
        """生成用于演示的模拟检测结果"""
        # 模拟数据形状: [1, 25200, 85] 适用于YOLOv5模型
        # [batch, num_boxes, x, y, w, h, conf, class1, class2, ...]
        mock_pred = np.zeros((1, 10, 85), dtype=np.float32)
        
        # 根据模型类型添加一些模拟检测
        if self.model_type == ModelType.FIRE_SMOKE:
            # 添加模拟火灾检测
            mock_pred[0, 0, :4] = [320, 240, 100, 100]  # x, y, w, h
            mock_pred[0, 0, 4] = 0.8  # 置信度
            mock_pred[0, 0, 5] = 1.0  # 火灾类别
            
            # 添加模拟烟雾检测
            mock_pred[0, 1, :4] = [420, 180, 80, 80]  # x, y, w, h
            mock_pred[0, 1, 4] = 0.7  # 置信度
            mock_pred[0, 1, 6] = 1.0  # 烟雾类别

        
        return mock_pred
    
    def _draw_detections(self, img: np.ndarray, detections: List, conf_threshold: float) -> Tuple[np.ndarray, int]:
        """在图像上绘制检测框"""
        detection_count = 0
        result_img = img.copy()
        
        for detection in detections:
            prob = float(detection[4])
            cls_id = int(detection[5])
            
            if prob >= conf_threshold:
                detection_count += 1
                x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                
                # 为这个类别生成颜色
                color_id = int(cls_id) % 10  # 循环使用10种颜色
                colors = [
                    (0, 255, 0),     # 绿色
                    (255, 0, 0),     # 蓝色
                    (0, 0, 255),     # 红色
                    (255, 255, 0),   # 青色
                    (0, 255, 255),   # 黄色
                    (255, 0, 255),   # 洋红色
                    (128, 255, 0),   # 浅绿色
                    (255, 128, 0),   # 浅蓝色
                    (128, 0, 255),   # 紫色
                    (0, 128, 255)    # 橙色
                ]
                color = colors[color_id]
                
                # 绘制矩形和标签
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                class_name = self.CLASSES.get(cls_id, 'unknown')
                label = f"{class_name} {prob:.2f}"
                cv2.putText(result_img, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_img, detection_count
    
    def _save_result(self, img: np.ndarray, output_path: str) -> bool:
        """将检测结果保存到文件"""
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存图像
        cv2.imwrite(output_path, img)
        
        return os.path.exists(output_path)
    
    def _non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, 
                            classes=None, agnostic=False, multi_label=False,
                            labels=(), max_det=300, nm=0):
        """对推理结果进行非最大抑制（NMS）"""
        # 使用原始代码中的non_max_suppression函数
        return non_max_suppression(
            prediction, conf_thres, iou_thres, classes, agnostic, 
            multi_label, labels, max_det, nm
        )
    
    def _scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """将框从img1_shape缩放到img0_shape"""
        # 使用原始代码中的scale_boxes函数
        return scale_boxes(img1_shape, boxes, img0_shape, ratio_pad)

    def _detect_objects_batch(self, frames):
        """
        批量处理多个帧
        :param frames: 形状为 (batch_size, height, width, channels) 的numpy数组
        :return: 检测结果列表
        """
        results = []
        
        # 逐个处理每一帧
        for frame in frames:
            # 使用letterbox进行预处理，保持宽高比
            im, ratio, (dw, dh) = letterbox(frame, self.img_size, auto=False, stride=32)
            # 转换为RGB并调整维度顺序
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
            im = im.astype(np.float32)
            im /= 255.0  # 归一化
            
            # 添加batch维度
            im = np.expand_dims(im, axis=0)
            
            # 执行推理
            outputs = self.session.run(None, {self.input_name: im})
            
            # 对检测结果进行NMS
            pred = self._non_max_suppression(
                outputs[0], 
                self.conf_threshold, 
                self.iou_threshold, 
                max_det=1000
            )
            
            detections = []
            if len(pred[0]):
                # 将检测框缩放回原始图像尺寸
                pred[0][:, :4] = self._scale_boxes(
                    self.img_size, 
                    pred[0][:, :4], 
                    frame.shape
                ).round()
                detections = pred[0][:, :6].tolist()
            
            # 为每个检测结果创建一个DetectionResult对象
            results.append([DetectionResult(
                model_type=self.model_type,  # 使用实例的model_type
                boxes=detections
            )])
        
        return results

# 示例用法:
if __name__ == "__main__":

    # 初始化检测器
    model_path = "fastapi_server\weights\fall-detect-best.onnx"
    detector = YOLODetector(model_path, conf_threshold=0.25, iou_threshold=0.45)
    
    # 处理图像
    # image_path = "fastapi_server/test/images/1.png"
    # output_path = "detection_results/detection_result.png"
    
    # result = detector.process_image(image_path, output_path)
    # print(f"图像检测完成: 发现 {result['detection_count']} 个目标")
    # print(f"保存到: {result['output_path']}")
    
    # 处理视频（如果需要）
    video_path = "fastapi_server/statics/sample.mp4"
    video_output_path = "fastapi_server/detection_results/detection_result.mp4"
    
    try:
        video_result = detector.process_video(
            video_path, 
            video_output_path,
            skip_frames=0  # 为了速度处理每3帧
        )
        print(f"视频处理完成: 在 {video_result['processed_frames']} 帧中发现 {video_result['detection_count']} 个目标")
        print(f"保存到: {video_result['output_path']}")
        print(f"处理时间: {video_result['processing_time']:.2f} 秒")
    except FileNotFoundError:
        print("未找到测试视频。跳过视频处理。")
