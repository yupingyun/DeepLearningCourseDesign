import cv2
import numpy as np
import onnxruntime as rt
import os
import time
 
CLASSES = {
    0: 'fire',           # 火灾
    1: 'smoke',          # 烟雾
    2: 'car',            # 汽车
    3: 'motorbike',      # 摩托车
    4: 'aeroplane',      # 飞机
    5: 'bus',            # 公交车
    6: 'train',          # 火车
    7: 'truck',          # 卡车
    8: 'boat',           # 船
    9: 'traffic light',  # 交通灯
    10: 'fire hydrant',  # 消防栓
    11: 'stop sign',     # 停止标志
    12: 'parking meter', # 停车计时器
    13: 'bench',         # 长椅
    14: 'bird',          # 鸟
    15: 'cat',           # 猫
    16: 'dog',           # 狗
    17: 'horse',         # 马
    18: 'sheep',         # 羊
    19: 'cow',           # 牛
    20: 'elephant',      # 大象
    21: 'bear',          # 熊
    22: 'zebra',         # 斑马
    23: 'giraffe',       # 长颈鹿
    24: 'backpack',      # 背包
    25: 'umbrella',      # 雨伞
    26: 'handbag',       # 手提包
    27: 'tie',           # 领带
    28: 'suitcase',      # 行李箱
    29: 'frisbee',       # 飞盘
    30: 'skis',          # 滑雪板
    31: 'snowboard',     # 滑雪板
    32: 'sports ball',   # 运动球
    33: 'kite',          # 风筝
    34: 'baseball bat',  # 棒球 bat
    35: 'baseball glove', # 棒球手套
    36: 'skateboard',    # 滑板
    37: 'surfboard',     # 冲浪板
    38: 'tennis racket', # 网球拍
    39: 'bottle',        # 瓶子
    40: 'wine glass',    # 酒杯
    41: 'cup',           # 杯子
    42: 'fork',          # 叉子
    43: 'knife',         # 刀
    44: 'spoon',         # 勺子
    45: 'bowl',          # 碗
    46: 'banana',        # 香蕉
    47: 'apple',         # 苹果
    48: 'sandwich',      # 三明治
    49: 'orange',        # 橙子
    50: 'broccoli',      # 西兰花
    51: 'carrot',        # 胡萝卜
    52: 'hot dog',       # 热狗
    53: 'pizza',         # 披萨
    54: 'donut',         # 甜甜圈
    55: 'cake',          # 蛋糕
    56: 'chair',         # 椅子
    57: 'sofa',          # 沙发
    58: 'potted plant',  # 盆栽植物
    59: 'bed',           # 床
    60: 'dining table',  # 餐桌
    61: 'toilet',        # 厕所
    62: 'tvmonitor',     # 电视监视器
    63: 'laptop',        # 笔记本电脑
    64: 'mouse',         # 鼠标
    65: 'remote',        # 遥控器
    66: 'keyboard',      # 键盘
    67: 'cell phone',    # 手机
    68: 'microwave',     # 微波炉
    69: 'oven',          # 烤箱
    70: 'toaster',       # 烤面包机
    71: 'sink',          # 水槽
    72: 'refrigerator',  # 冰箱
    73: 'book',          # 书
    74: 'clock',         # 时钟
    75: 'vase',          # 花瓶
    76: 'scissors',      # 剪刀
    77: 'teddy bear',    # 泰迪熊
    78: 'hair drier',    # 吹风机
    79: 'toothbrush'     # 牙刷
}

def box_iou(box1, box2, eps=1e-7):
    """计算两个边界框的IoU（交并比）"""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
 
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """调整图像大小并填充，同时满足步长倍数约束"""
    shape = im.shape[:2]  # 当前形状 [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # 缩放比例（新/旧）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大（以获得更好的验证mAP）
        r = min(r, 1.0)
 
    # 计算填充
    ratio = r, r  # 宽度，高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh填充
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度，高度比例
 
    dw /= 2  # 将填充分为2边
    dh /= 2
 
    if shape[::-1] != new_unpad:  # 调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return im, ratio, (dw, dh)
 
def onnx_inf(onnxModulePath, data):
    """使用ONNX模型进行推理"""
    # 使用全局会话以避免为每一帧重新创建
    global onnx_session
    if 'onnx_session' not in globals():
        # 检查可用的执行提供者
        providers = rt.get_available_providers()
        
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
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            selected_providers.append('CUDAExecutionProvider')
            provider_options.append(cuda_options)
        
        # 始终添加CPU作为备选
        selected_providers.append('CPUExecutionProvider')
        provider_options.append({})
        
        try:
            onnx_session = rt.InferenceSession(
                onnxModulePath, 
                sess_options=session_options,
                providers=selected_providers,
                provider_options=provider_options
            )
            print(f"ONNX Runtime 使用: {onnx_session.get_providers()}")
        except Exception as e:
            print(f"警告: 优化会话创建失败 ({str(e)}), 使用默认设置")
            onnx_session = rt.InferenceSession(onnxModulePath)
    
    # 获取输入/输出名称
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    # 确保数据形状正确且格式正确
    # 如果数据已经正确，避免不必要的重塑
    if data.shape != (1, 3, 640, 640):
        data = data.reshape(1, 3, 640, 640)
    
    # 确保数据是连续的以获得更好的性能
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    
    # 运行推理
    start = time.time()
    pred_onnx = onnx_session.run([output_name], {input_name: data.astype(np.float32)})
    inference_time = time.time() - start
    
    # 取消注释以调试性能
    # print(f"推理时间: {inference_time*1000:.2f} ms")
    
    return pred_onnx
 
def xywh2xyxy(x):
    """将nx4框从 [x, y, w, h] 转换为 [x1, y1, x2, y2]，其中xy1=左上角，xy2=右下角"""
    # isinstance 用来判断某个变量是否属于某种类型
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # 左上角x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # 左上角y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # 右下角x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # 右下角y
    return y
 
def nms_boxes(boxes, scores):
    """对边界框执行非最大抑制"""
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
        nm=0,  # 掩码数量
):
    """对推理结果进行非最大抑制（NMS）以拒绝重叠检测
    返回:
         检测列表，每张图像(n,6)张量 [xyxy, conf, cls]
    """
 
    # 检查
    assert 0 <= conf_thres <= 1, f'无效的置信度阈值 {conf_thres}, 有效值在0.0到1.0之间'
    assert 0 <= iou_thres <= 1, f'无效的IoU {iou_thres}, 有效值在0.0到1.0之间'
    if isinstance(prediction, (list, tuple)):  # YOLOv5模型在验证模型中，输出 = (inference_out, loss_out)
        prediction = prediction[0]  # 只选择推理输出
 
    bs = prediction.shape[0]  # 批量大小
    nc = prediction.shape[2] - nm - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 候选
 
    # 设置
    max_wh = 7680  # (像素) 最大框宽度和高度
    max_nms = 30000  # torchvision.ops.nms()中的最大框数
    redundant = True  # 需要冗余检测
    multi_label &= nc > 1  # 每个框多个标签（增加0.5ms/图像）
    merge = False  # 使用merge-NMS
 
    mi = 5 + nc  # 掩码开始索引
    output = [np.zeros((0, 6 + nm))] * bs
 
    for xi, x in enumerate(prediction):  # 图像索引，图像推理
        x = x[xc[xi]]  # 置信度
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros(len(lb), nc + nm + 5)
            v[:, :4] = lb[:, 1:5]  # 框
            v[:, 4] = 1.0  # 置信度
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # 类别
            x = np.concatenate((x, v), 0)
 
        # 如果没有剩余，处理下一张图像
        if not x.shape[0]:
            continue
 
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
 
        # 框/掩码
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) 到 (x1, y1, x2, y2)
        mask = x[:, mi:]  # 如果没有掩码，则为零列
 
        # 检测矩阵 nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
 
        else:  # 仅最佳类别
            conf = np.max(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            j = np.argmax(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(box.shape[:1][0]) > conf_thres]
 
        # 按类别过滤
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes, device=x.device)).any(1)]
 
        # 检查形状
        n = x.shape[0]  # 框数
        if not n:  # 无框
            continue
        index = x[:, 4].argsort(axis=0)[:max_nms][::-1]
        x = x[index]
 
        # 批量NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        boxes, scores = x[:, :4] + c, x[:, 4]  # 框（按类别偏移），分数
        i = nms_boxes(boxes, scores)
        i = i[:max_det]  # 限制检测
 
        # 用来合并框的
        if merge and (1 < n < 3E3):  # Merge NMS（使用加权平均合并框）
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou矩阵
            weights = iou * scores[None]  # 框权重
            x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # 合并框
            if redundant:
                i = i[iou.sum(1) > 1]  # 需要冗余
 
        output[xi] = x[i]
 
    return output
 
def clip_boxes(boxes, shape):
    """将框（xyxy）裁剪到图像形状（高度，宽度）"""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
 
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """将框（xyxy）从img1_shape缩放到img0_shape"""
    if ratio_pad is None:  # 从img0_shape计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = 旧 / 新
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
 
    boxes[..., [0, 2]] -= pad[0]  # x填充
    boxes[..., [1, 3]] -= pad[1]  # y填充
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
 
def preprocess_image(img_path, img_size=(640, 640)):
    """预处理图像以用于模型输入"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"错误: 无法读取 {img_path} 处的图像")
    
    original_img = img.copy()
    img = cv2.resize(img, (640, 640))
    
    # 预处理
    im = letterbox(img, img_size, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255  # 0 - 255 到 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # 扩展为batch维度
        
    return img, im, original_img

def detect_objects(model_path, img_path, img_size=(640, 640), conf_thres=0.25, 
                  iou_thres=0.45, max_det=1000, classes=None, agnostic_nms=False):
    """
    使用YOLO模型检测图像中的目标
    
    参数:
        model_path: ONNX模型路径
        img_path: 输入图像路径
        img_size: 模型输入大小
        conf_thres: 检测置信度阈值
        iou_thres: NMS IoU阈值
        max_det: 最大检测数量
        classes: 按类别ID过滤（None表示所有类别）
        agnostic_nms: 类别无关NMS标志
        
    返回:
        img: 绘制了检测框的图像
        detections: 检测信息列表 [x1, y1, x2, y2, confidence, class_id]
    """
    # 预处理图像
    img, im, original_img = preprocess_image(img_path, img_size)
    
    # 推理
    pred = onnx_inf(model_path, im)
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    detections = []
    
    for i, det in enumerate(pred):  # 每张图像
        if len(det):
            # 将框从img_size缩放到im0大小
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            detections = det[:, :6].tolist()
    
    return img, detections

def draw_detections(img, detections, conf_threshold=0.4):
    """在图像上绘制检测框"""
    detection_count = 0
    result_img = img.copy()
    
    for detection in detections:
        prob = float(detection[4])
        cls_id = int(detection[5])
        
        if prob >= conf_threshold:
            detection_count += 1
            x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
            
            # 为这个检测生成颜色（使用绿色）
            color = (0, 255, 0)
            
            # 绘制矩形和标签
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASSES.get(cls_id, 'unknown')} {prob:.2f}"
            cv2.putText(result_img, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result_img, detection_count

def save_result(img, output_path):
    """将检测结果保存到文件"""
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存图像
    cv2.imwrite(output_path, img)
    
    return os.path.exists(output_path)

if __name__ == "__main__":
    # 测试路径
    onnx_model_path = "fastapi_server/test/weight/fire.onnx"
    test_img_path = "fastapi_server/test/images/1.png"
    output_path = "fastapi_server/detection_results/detection_result.png"
    
    try:
        # 执行目标检测
        img, detections = detect_objects(onnx_model_path, test_img_path)
        
        # 在图像上绘制框
        result_img, detection_count = draw_detections(img, detections)
        
        # 保存和显示结果
        if save_result(result_img, output_path):
            print(f"检测完成: 发现 {detection_count} 个目标")
            print(f"文件成功保存到: {os.path.abspath(output_path)}")
        else:
            print("错误: 无法保存输出图像")
            
        # 尽可能显示图像
        try:
            cv2.imshow("YOLO检测结果", result_img)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("无法显示图像（无GUI可用）")
            
    except Exception as e:
        print(f"检测过程中出错: {e}")
