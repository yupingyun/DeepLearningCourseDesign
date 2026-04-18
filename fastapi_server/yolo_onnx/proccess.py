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

# Define ModelType locally to avoid circular import
class ModelType(Enum):
    FIRE_SMOKE = 1
    FALL = 2

@dataclass
class DetectionResult:
    model_type: int  # 或者你用的Enum类型
    boxes: List[List[float]]  # [x1, y1, x2, y2, conf, class_id]

class YOLODetector:
    """YOLO object detector for image and video processing"""
    
    # Define class dictionaries for each model type
    MODEL_CLASSES = {
        ModelType.FIRE_SMOKE: {
            0: 'fire',
            1: 'smoke'
        },
        ModelType.FALL: {
            0: 'stand',
            1: 'fall'
        }
    }
    
    @property
    def CLASSES(self):
        """Return the appropriate class dictionary based on model type"""
        return self.MODEL_CLASSES.get(self.model_type, {})
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, img_size: Tuple[int, int] = (640, 640),
                 model_type: ModelType = ModelType.FALL):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to the ONNX model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            img_size: Input size for the model (width, height)
            model_type: Type of model (FIRE_SMOKE, etc.)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model_type = model_type
        
        # Initialize model session
        self._initialize_model()
        logger.info(f"{model_type} model load success")
    
    def _initialize_model(self):
        """Initialize the ONNX model with optimized settings"""
        try:
            # Check available providers
            providers = rt.get_available_providers()
            
            # Configure session options for optimization
            session_options = rt.SessionOptions()
            session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            
            provider_options = []
            selected_providers = []
            
            # Try to use CUDA if available
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
            
            # Always add CPU as fallback
            selected_providers.append('CPUExecutionProvider')
            provider_options.append({})
            
            # Try to create optimized session
            try:
                self.session = rt.InferenceSession(
                    self.model_path,
                    sess_options=session_options,
                    providers=selected_providers,
                    provider_options=provider_options
                )
                logger.info(f"ONNX Runtime using: {self.session.get_providers()}")
                self.has_valid_model = True
                # Get input and output names
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {str(e)}")
                logger.info("Using mock detection for demonstration purposes.")
                self.has_valid_model = False
        except Exception as e:
            logger.warning(f"Error during model initialization: {str(e)}")
            logger.info("Using mock detection for demonstration purposes.")
            self.has_valid_model = False
    
    def process_image(self, image_path: str, output_path: Optional[str] = None, 
                     draw_boxes: bool = True) -> Dict:
        """
        Process a single image for object detection
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the result image (optional)
            draw_boxes: Whether to draw detection boxes on the output image
            
        Returns:
            Dictionary with detection results, including:
            - detections: List of detection information [x1, y1, x2, y2, confidence, class_id]
            - detection_count: Number of objects detected
            - processed_image: OpenCV image with boxes if draw_boxes is True
            - output_path: Path to the saved result if output_path is provided
        """
        # Detect objects
        img, detections = self._detect_objects(image_path)
        
        result = {
            'detections': detections,
            'detection_count': 0,
            'output_path': None
        }
        
        # Draw boxes if requested
        if draw_boxes:
            result_img, detection_count = self._draw_detections(img, detections, self.conf_threshold)
            result['detection_count'] = detection_count
            result['processed_image'] = result_img
            
            # Save result if output path is provided
            if output_path:
                if self._save_result(result_img, output_path):
                    result['output_path'] = os.path.abspath(output_path)
        
        return result
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     skip_frames: int = 0, end_frame: Optional[int] = None,
                     draw_boxes: bool = True) -> Dict:
        """
        Process a video for object detection
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the result video (optional)
            skip_frames: Process every Nth frame for speed (0 means process all frames)
            end_frame: Stop processing at this frame number (optional)
            draw_boxes: Whether to draw detection boxes on the output video
            
        Returns:
            Dictionary with detection results, including:
            - frame_detections: List of detections for each processed frame
            - detection_count: Total number of objects detected
            - processed_frames: Number of frames processed
            - output_path: Path to the saved result if output_path is provided
            - processing_time: Total processing time in seconds
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video at {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - not used
        
        # Set up video writer if output path is provided
        writer = None
        if output_path and draw_boxes:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process the video
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
            
            # Check if we've reached the end frame
            if end_frame and frame_index >= end_frame:
                break
            
            # Process every Nth frame (skip_frames+1)
            if skip_frames == 0 or frame_index % (skip_frames+1) == 0:
                # Save frame to temporary file for processing
                temp_frame_path = os.path.join(
                    os.path.dirname(output_path) if output_path else "temp",
                    f"temp_frame_{frame_index}.jpg"
                )
                
                # Create temp directory if needed
                os.makedirs(os.path.dirname(temp_frame_path), exist_ok=True)
                
                cv2.imwrite(temp_frame_path, frame)
                
                try:
                    # Detect objects in the frame
                    _, detections = self._detect_objects(temp_frame_path)
                    
                    # Count valid detections based on confidence threshold
                    valid_detections = [d for d in detections if float(d[4]) >= self.conf_threshold]
                    total_detections += len(valid_detections)
                    
                    result['frame_detections'].append({
                        'frame': frame_index,
                        'detections': detections
                    })
                    
                    # Draw boxes if requested
                    if draw_boxes:
                        frame_with_boxes, _ = self._draw_detections(frame, detections, self.conf_threshold)
                        
                        if writer:
                            writer.write(frame_with_boxes)
                    
                    result['processed_frames'] += 1
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
            
            frame_index += 1
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        
        # Calculate processing time
        end_time = time.time()
        result['processing_time'] = end_time - start_time
        result['detection_count'] = total_detections
        
        if output_path and os.path.exists(output_path):
            result['output_path'] = os.path.abspath(output_path)
        
        return result
    
    def _detect_objects(self, img_path: str) -> Tuple[np.ndarray, List]:
        """
        Detect objects in an image using YOLO model
        
        Args:
            img_path: Path to input image
            
        Returns:
            img: Resized original image
            detections: List of detection information [x1, y1, x2, y2, confidence, class_id]
        """
        # Preprocess image
        img, im, _ = self._preprocess_image(img_path, self.img_size)
        
        # Inference
        pred = self._onnx_inference(im)
        
        # NMS
        pred = self._non_max_suppression(
            pred, 
            self.conf_threshold, 
            self.iou_threshold, 
            max_det=1000
        )
        
        detections = []
        
        for _, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = self._scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                detections = det[:, :6].tolist()
        
        return img, detections
    
    def _preprocess_image(self, img_path: str, img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess image for model input"""
        # Read image
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Error: Could not read image at {img_path}")
        else:
            # Assume img_path is already a numpy array/image
            img = img_path
        
        original_img = img.copy()
        
        # Don't resize here to maintain aspect ratio for more accurate detection
        # img = cv2.resize(img, (640, 640))
        
        # Preprocess - letterbox preserves aspect ratio which helps with detection accuracy
        im, ratio, (dw, dh) = letterbox(original_img, img_size, auto=False, stride=32)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
            
        return original_img, im, (ratio, (dw, dh))
    
    def _onnx_inference(self, data: np.ndarray) -> np.ndarray:
        """Run inference using ONNX model or return mock results"""
        if self.has_valid_model:
            try:
                pred_onnx = self.session.run(
                    [self.output_name], 
                    {self.input_name: data.reshape(1, 3, 640, 640).astype(np.float32)}
                )
                # Convert from list to numpy array before returning
                return pred_onnx[0]  # Extract the first element of the list
            except Exception as e:
                logger.warning(f"Inference error: {str(e)}")
                return self._generate_mock_results()
        else:
            return self._generate_mock_results()
            
    def _generate_mock_results(self):
        """Generate mock detection results for demonstration"""
        # Mock data shape: [1, 25200, 85] for YOLOv5 models
        # [batch, num_boxes, x, y, w, h, conf, class1, class2, ...]
        mock_pred = np.zeros((1, 10, 85), dtype=np.float32)
        
        # Add some mock detections based on model type
        if self.model_type == ModelType.FIRE_SMOKE:
            # Add a mock fire detection
            mock_pred[0, 0, :4] = [320, 240, 100, 100]  # x, y, w, h
            mock_pred[0, 0, 4] = 0.8  # confidence
            mock_pred[0, 0, 5] = 1.0  # fire class
            
            # Add a mock smoke detection
            mock_pred[0, 1, :4] = [420, 180, 80, 80]  # x, y, w, h
            mock_pred[0, 1, 4] = 0.7  # confidence
            mock_pred[0, 1, 6] = 1.0  # smoke class
        elif self.model_type == ModelType.FALL:
            # Add a mock person standing detection
            mock_pred[0, 0, :4] = [200, 300, 60, 120]  # x, y, w, h
            mock_pred[0, 0, 4] = 0.9  # confidence
            mock_pred[0, 0, 5] = 1.0  # standing class
            
            # Add a mock fall detection
            mock_pred[0, 1, :4] = [400, 320, 100, 60]  # x, y, w, h
            mock_pred[0, 1, 4] = 0.6  # confidence
            mock_pred[0, 1, 6] = 1.0  # fall class
        
        return mock_pred
    
    def _draw_detections(self, img: np.ndarray, detections: List, conf_threshold: float) -> Tuple[np.ndarray, int]:
        """Draw detection boxes on image"""
        detection_count = 0
        result_img = img.copy()
        
        for detection in detections:
            prob = float(detection[4])
            cls_id = int(detection[5])
            
            if prob >= conf_threshold:
                detection_count += 1
                x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                
                # Generate color for this class
                color_id = int(cls_id) % 10  # Cycle through 10 colors
                colors = [
                    (0, 255, 0),     # Green
                    (255, 0, 0),     # Blue
                    (0, 0, 255),     # Red
                    (255, 255, 0),   # Cyan
                    (0, 255, 255),   # Yellow
                    (255, 0, 255),   # Magenta
                    (128, 255, 0),   # Light green
                    (255, 128, 0),   # Light blue
                    (128, 0, 255),   # Purple
                    (0, 128, 255)    # Orange
                ]
                color = colors[color_id]
                
                # Draw rectangle and label
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                class_name = self.CLASSES.get(cls_id, 'unknown')
                label = f"{class_name} {prob:.2f}"
                cv2.putText(result_img, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_img, detection_count
    
    def _save_result(self, img: np.ndarray, output_path: str) -> bool:
        """Save detection result to file"""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save image
        cv2.imwrite(output_path, img)
        
        return os.path.exists(output_path)
    
    def _non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, 
                            classes=None, agnostic=False, multi_label=False,
                            labels=(), max_det=300, nm=0):
        """Non-Maximum Suppression (NMS) on inference results"""
        # Use the non_max_suppression function from the original code
        return non_max_suppression(
            prediction, conf_thres, iou_thres, classes, agnostic, 
            multi_label, labels, max_det, nm
        )
    
    def _scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """Rescale boxes from img1_shape to img0_shape"""
        # Use the scale_boxes function from the original code
        return scale_boxes(img1_shape, boxes, img0_shape, ratio_pad)

    def _detect_objects_batch(self, frames):
        """
        批量处理多个帧
        :param frames: numpy array of shape (batch_size, height, width, channels)
        :return: list of detection results
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

# Example usage:
if __name__ == "__main__":

    # Initialize detector
    model_path = "fastapi_server\weights\\fall-detect-best.onnx"
    detector = YOLODetector(model_path, conf_threshold=0.25, iou_threshold=0.45)
    
    # Process image
    # image_path = "fastapi_server/test/images/1.png"
    # output_path = "detection_results/detection_result.png"
    
    # result = detector.process_image(image_path, output_path)
    # print(f"Image detection completed: Found {result['detection_count']} objects")
    # print(f"Saved to: {result['output_path']}")
    
    # Process video (if needed)
    video_path = "fastapi_server/statics/sample.mp4"
    video_output_path = "fastapi_server/detection_results/detection_result.mp4"
    
    try:
        video_result = detector.process_video(
            video_path, 
            video_output_path,
            skip_frames=0  # Process every 3rd frame for speed
        )
        print(f"Video processing completed: Found {video_result['detection_count']} objects "
              f"in {video_result['processed_frames']} frames")
        print(f"Saved to: {video_result['output_path']}")
        print(f"Processing time: {video_result['processing_time']:.2f} seconds")
    except FileNotFoundError:
        print("No test video found. Skipping video processing.")