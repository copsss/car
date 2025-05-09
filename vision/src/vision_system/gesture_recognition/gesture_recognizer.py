"""
手势识别模块 - 实现握拳暂停音乐、竖起大拇指确认和摇手表示拒绝等手势识别
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum, auto

class GestureType(Enum):
    """手势类型枚举类"""
    UNKNOWN = auto()      # 未知手势
    FIST = auto()         # 握拳（暂停音乐）
    THUMBS_UP = auto()    # 竖起大拇指（确认）
    WAVE = auto()         # 摇手（拒绝）
    NORMAL = auto()       # 正常状态（无手势）

@dataclass
class GestureResult:
    """手势识别结果数据类"""
    gesture: GestureType  # 识别的手势类型
    confidence: float     # 结果置信度
    position: tuple       # 手势位置坐标 (x, y)
    timestamp: float      # 时间戳

class GestureRecognizer:
    """
    手势识别类 - 实现手势识别相关功能
    
    功能:
    1. 识别握拳手势（暂停音乐）
    2. 识别竖起大拇指手势（确认）
    3. 识别摇手手势（拒绝）
    """
    
    def __init__(self, camera_id=0, confidence_threshold=0.6):
        """
        初始化手势识别器
        
        参数:
        - camera_id: 摄像头ID
        - confidence_threshold: 识别置信度阈值
        """
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.last_result = None
        self.previous_positions = []  # 用于存储手势位置历史，用于判断摇手动作
        self.max_position_history = 10  # 历史位置最大存储数量
        
    def initialize(self):
        """初始化手势识别设备和模型"""
        print("初始化手势识别模块...")
        # 这里应该加载相关的手势识别模型和算法
        # 例如，加载预训练的深度学习模型或进行摄像头初始化
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("无法访问摄像头")
            
        # 在实际项目中，应该加载手部关键点检测模型
        # 例如使用OpenCV的DNN模块、MediaPipe Hands等
        
        return True
        
    def release(self):
        """释放资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            
    def _detect_hands(self, frame):
        """
        检测手部区域和关键点
        
        参数:
        - frame: 输入的视频帧
        
        返回:
        - hand_landmarks: 手部关键点列表
        - hand_position: 手的位置，格式为(x, y)
        """
        # 在实际项目中，应该使用专门的手势识别库
        # 例如MediaPipe Hands或者自定义的深度学习模型
        # 这里提供一个基于颜色的简化检测，仅作为示例
        
        # 转换到HSV色彩空间，以便进行肤色检测
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定义肤色范围（简化版）
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 创建肤色掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作优化掩码
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        # 找到掩码中的轮廓
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 没有检测到手
        if not contours:
            return None, (0, 0)
            
        # 找到最大的轮廓（假设是手）
        hand_contour = max(contours, key=cv2.contourArea)
        
        # 计算手的位置（轮廓的质心）
        M = cv2.moments(hand_contour)
        if M["m00"] == 0:
            return None, (0, 0)
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        hand_position = (cx, cy)
        
        # 近似轮廓以获取关键点
        epsilon = 0.02 * cv2.arcLength(hand_contour, True)
        approx = cv2.approxPolyDP(hand_contour, epsilon, True)
        
        # 计算凸包和凸缺陷
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects = None
        
        try:
            if len(hull) > 3:
                defects = cv2.convexityDefects(hand_contour, hull)
        except:
            pass
            
        return defects, hand_position

    def _recognize_gesture(self, defects, hand_position, frame_shape):
        """
        识别手势类型
        
        参数:
        - defects: 手部轮廓的凸缺陷
        - hand_position: 手的位置坐标
        - frame_shape: 帧的形状(高, 宽, 通道)
        
        返回:
        - gesture_type: 识别的手势类型
        - confidence: 识别的置信度
        """
        # 如果没有检测到手或凸缺陷，返回未知手势
        if defects is None:
            return GestureType.UNKNOWN, 0.0
            
        # 计算帧的中心位置
        frame_height, frame_width = frame_shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)
        
        # 计算凸缺陷的数量，用于手势识别
        finger_count = 0
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 10000:  # 距离阈值，过滤掉小的凸缺陷
                finger_count += 1
                
        # 更新手部位置历史
        self.previous_positions.append(hand_position)
        if len(self.previous_positions) > self.max_position_history:
            self.previous_positions.pop(0)
            
        # 手势识别逻辑
        gesture = GestureType.NORMAL
        confidence = 0.0
        
        # 1. 握拳识别（无或很少凸缺陷）
        if finger_count <= 1:
            gesture = GestureType.FIST
            confidence = 0.7 + (1 - finger_count) * 0.1  # 调整置信度
            
        # 2. 竖起大拇指识别（假设有2-3个凸缺陷，且手部在画面上半部分）
        elif finger_count <= 3 and hand_position[1] < frame_center[1]:
            gesture = GestureType.THUMBS_UP
            confidence = 0.6 + finger_count * 0.05
            
        # 3. 摇手识别（通过分析手部位置历史）
        elif len(self.previous_positions) >= 5:
            # 计算横向移动的方差
            x_positions = [pos[0] for pos in self.previous_positions]
            x_variance = np.var(x_positions)
            
            # 如果横向移动方差大，认为是摇手
            if x_variance > 1000:  # 阈值需要根据实际情况调整
                gesture = GestureType.WAVE
                # 方差越大，置信度越高
                confidence = min(0.9, 0.6 + x_variance / 10000)
                
        # 如果置信度低于阈值，返回正常状态
        if confidence < self.confidence_threshold:
            gesture = GestureType.NORMAL
            confidence = 0.9  # 正常状态的置信度较高
            
        return gesture, confidence

    def process_frame(self, frame=None):
        """
        处理单帧图像，识别手势
        
        参数:
        - frame: 输入的视频帧，如果为None则从摄像头获取
        
        返回:
        - result: GestureResult对象，包含手势识别结果
        """
        # 如果没有提供帧，则从摄像头获取
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                return None
                
        # 检测手部
        defects, hand_position = self._detect_hands(frame)
        
        # 识别手势
        gesture, confidence = self._recognize_gesture(defects, hand_position, frame.shape)
        
        # 创建结果对象
        result = GestureResult(
            gesture=gesture,
            confidence=confidence,
            position=hand_position,
            timestamp=time.time()
        )
        
        self.last_result = result
        return result
        
    def visualize(self, frame, result):
        """
        在图像上可视化手势识别结果
        
        参数:
        - frame: 输入的视频帧
        - result: GestureResult对象
        
        返回:
        - vis_frame: 可视化后的视频帧
        """
        if frame is None or result is None:
            return frame
            
        vis_frame = frame.copy()
        
        # 绘制手部位置
        cv2.circle(vis_frame, result.position, 10, (0, 255, 0), -1)
        
        # 绘制手势类型
        gesture_text = f"手势: {result.gesture.name}"
        confidence_text = f"置信度: {result.confidence:.2f}"
        
        # 根据手势类型选择颜色
        if result.gesture == GestureType.FIST:
            color = (0, 0, 255)  # 红色
        elif result.gesture == GestureType.THUMBS_UP:
            color = (0, 255, 0)  # 绿色
        elif result.gesture == GestureType.WAVE:
            color = (255, 0, 0)  # 蓝色
        else:
            color = (255, 255, 255)  # 白色
            
        # 添加文本
        cv2.putText(vis_frame, gesture_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_frame, confidence_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        # 如果是特定手势，添加提示
        if result.gesture == GestureType.FIST:
            cv2.putText(vis_frame, "暂停音乐", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif result.gesture == GestureType.THUMBS_UP:
            cv2.putText(vis_frame, "确认", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif result.gesture == GestureType.WAVE:
            cv2.putText(vis_frame, "拒绝", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        return vis_frame


# 示例使用代码
def gesture_recognizer_demo():
    """手势识别模块演示函数"""
    recognizer = GestureRecognizer(confidence_threshold=0.6)
    
    try:
        recognizer.initialize()
        
        while True:
            ret, frame = recognizer.cap.read()
            if not ret:
                break
                
            # 处理帧
            result = recognizer.process_frame(frame)
            
            # 可视化
            vis_frame = recognizer.visualize(frame, result)
            
            # 显示结果
            cv2.imshow('Gesture Recognition Demo', vis_frame)
            
            # 根据识别结果打印信息
            if result and result.gesture != GestureType.NORMAL:
                print(f"检测到手势: {result.gesture.name}, 置信度: {result.confidence:.2f}")
                
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        recognizer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    gesture_recognizer_demo()