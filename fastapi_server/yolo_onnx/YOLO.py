import cv2
import numpy as np
import onnxruntime as rt
import os
import time
 
CLASSES = {
    0: 'fire',
    1: 'smoke',
    2: 'car',
    3: 'motorbike',
    4: 'aeroplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'sofa',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tvmonitor',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}
def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
 
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
 
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
 
def onnx_inf(onnxModulePath, data):
    # Use global session to avoid recreating it for every frame
    global onnx_session
    if 'onnx_session' not in globals():
        # Check available providers
        providers = rt.get_available_providers()
        
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
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            selected_providers.append('CUDAExecutionProvider')
            provider_options.append(cuda_options)
        
        # Always add CPU as fallback
        selected_providers.append('CPUExecutionProvider')
        provider_options.append({})
        
        try:
            onnx_session = rt.InferenceSession(
                onnxModulePath, 
                sess_options=session_options,
                providers=selected_providers,
                provider_options=provider_options
            )
            print(f"ONNX Runtime using: {onnx_session.get_providers()}")
        except Exception as e:
            print(f"Warning: Optimized session creation failed ({str(e)}), using default settings")
            onnx_session = rt.InferenceSession(onnxModulePath)
    
    # Get input/output names
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    # Ensure data is properly shaped and in the correct format
    # Avoid unnecessary reshapes if data is already correct
    if data.shape != (1, 3, 640, 640):
        data = data.reshape(1, 3, 640, 640)
    
    # Ensure data is contiguous for better performance
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    
    # Run inference
    start = time.time()
    pred_onnx = onnx_session.run([output_name], {input_name: data.astype(np.float32)})
    inference_time = time.time() - start
    
    # Uncomment to debug performance
    # print(f"Inference time: {inference_time*1000:.2f} ms")
    
    return pred_onnx
 
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # isinstance 用来判断某个变量是否属于某种类型
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
 
def nms_boxes(boxes, scores):
 
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
 
    areas = w * h
    order = scores.argsort()[::-1]
 
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
 
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
 
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.45)[0]
 
        order = order[inds + 1]
    keep = np.array(keep)
    return keep
 
def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
 
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
 
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
 
    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
 
    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm))] * bs
 
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros(len(lb), nc + nm + 5)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)
 
        # If none remain process next image
        if not x.shape[0]:
            continue
 
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
 
        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks
 
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
 
        else:  # best class only
            conf = np.max(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            j = np.argmax(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(box.shape[:1][0]) > conf_thres]
 
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes, device=x.device)).any(1)]
 
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        index = x[:, 4].argsort(axis=0)[:max_nms][::-1]
        x = x[index]
 
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_boxes(boxes, scores)
        i = i[:max_det]  # limit detections
 
        # 用来合并框的
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
 
        output[xi] = x[i]
 
    return output
 
def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
 
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
 
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
 
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
 
def preprocess_image(img_path, img_size=(640, 640)):
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Error: Could not read image at {img_path}")
    
    original_img = img.copy()
    img = cv2.resize(img, (640, 640))
    
    # Preprocess
    im = letterbox(img, img_size, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
        
    return img, im, original_img

def detect_objects(model_path, img_path, img_size=(640, 640), conf_thres=0.25, 
                  iou_thres=0.45, max_det=1000, classes=None, agnostic_nms=False):
    """
    Detect objects in an image using YOLO model
    
    Args:
        model_path: Path to ONNX model
        img_path: Path to input image
        img_size: Input size for the model
        conf_thres: Confidence threshold for detections
        iou_thres: NMS IoU threshold
        max_det: Maximum number of detections
        classes: Filter by class IDs (None means all classes)
        agnostic_nms: Class-agnostic NMS flag
        
    Returns:
        img: Image with detection boxes drawn
        detections: List of detection information [x1, y1, x2, y2, confidence, class_id]
    """
    # Preprocess image
    img, im, original_img = preprocess_image(img_path, img_size)
    
    # Inference
    pred = onnx_inf(model_path, im)
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    detections = []
    
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            detections = det[:, :6].tolist()
    
    return img, detections

def draw_detections(img, detections, conf_threshold=0.4):
    """Draw detection boxes on image"""
    detection_count = 0
    result_img = img.copy()
    
    for detection in detections:
        prob = float(detection[4])
        cls_id = int(detection[5])
        
        if prob >= conf_threshold:
            detection_count += 1
            x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
            
            # Generate color for this detection (using green)
            color = (0, 255, 0)
            
            # Draw rectangle and label
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASSES.get(cls_id, 'unknown')} {prob:.2f}"
            cv2.putText(result_img, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result_img, detection_count

def save_result(img, output_path):
    """Save detection result to file"""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save image
    cv2.imwrite(output_path, img)
    
    return os.path.exists(output_path)

if __name__ == "__main__":
    # Paths for testing
    onnx_model_path = "fastapi_server/test/weight/fire.onnx"
    test_img_path = "fastapi_server/test/images/1.png"
    output_path = "fastapi_server/detection_results/detection_result.png"
    
    try:
        # Perform object detection
        img, detections = detect_objects(onnx_model_path, test_img_path)
        
        # Draw boxes on image
        result_img, detection_count = draw_detections(img, detections)
        
        # Save and display results
        if save_result(result_img, output_path):
            print(f"Detection completed: Found {detection_count} objects")
            print(f"File successfully saved to: {os.path.abspath(output_path)}")
        else:
            print("Error: Failed to save the output image")
            
        # Display the image if possible
        try:
            cv2.imshow("YOLO Detection Result", result_img)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Could not display image (no GUI available)")
            
    except Exception as e:
        print(f"Error during detection: {e}")