"""
眼动追踪模块 - 实现用户目光集中区域识别和驾驶员视线偏离检测
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class EyeTrackingResult:
    """眼动追踪结果数据类"""
    is_looking_road: bool  # 是否注视道路
    gaze_coordinates: tuple  # 视线坐标 (x, y)
    distraction_duration: float  # 分心持续时间（秒）
    confidence: float  # 结果置信度
    timestamp: float  # 时间戳

class EyeTracker:
    """
    眼动追踪类 - 实现眼动追踪相关功能
    
    功能:
    1. 识别用户的目光集中区域
    2. 检测驾驶员视线是否偏离道路
    """
    
    def __init__(self, distraction_threshold=3.0, camera_id=0):
        """
        初始化眼动追踪器
        
        参数:
        - distraction_threshold: 分心阈值，单位秒，超过此时间视为分心
        - camera_id: 摄像头ID
        """
        self.distraction_threshold = distraction_threshold
        self.camera_id = camera_id
        self.distraction_start_time = None
        self.last_result = None
        
    def initialize(self):
        """初始化眼动追踪设备和模型"""
        print("初始化眼动追踪模块...")
        # 这里应该加载相关的眼动追踪模型和算法
        # 例如，加载预训练的深度学习模型或进行摄像头初始化
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("无法访问摄像头")
        return True
        
    def release(self):
        """释放资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            
    def _detect_eye_landmarks(self, frame):
        """
        检测眼睛关键点位置
        
        参数:
        - frame: 输入的视频帧
        
        返回:
        - landmarks: 眼睛关键点坐标
        """
        # 在实际项目中，这里应使用专门的眼睛关键点检测算法
        # 例如使用dlib、MediaPipe等库获取眼睛关键点
        # 这里提供一个模拟实现
        
        # 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用Haar级联分类器检测眼睛区域
        # 在实际项目中应替换为更精确的深度学习模型
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        landmarks = []
        for (ex, ey, ew, eh) in eyes:
            # 简单示例：将眼睛区域的中心点作为关键点
            eye_center = (ex + ew//2, ey + eh//2)
            landmarks.append(eye_center)
            
        return landmarks
        
    def _calculate_gaze_direction(self, landmarks):
        """
        根据眼睛关键点计算视线方向
        
        参数:
        - landmarks: 眼睛关键点
        
        返回:
        - gaze_point: 视线注视点坐标
        - is_looking_forward: 是否注视前方
        """
        # 在实际项目中，这里应实现基于眼睛关键点的视线方向估计算法
        # 例如使用瞳孔中心与眼角的几何关系，或使用专门的视线估计模型
        
        if not landmarks or len(landmarks) < 2:
            return (0, 0), False
            
        # 简单示例：使用两眼中点作为视线焦点
        x1, y1 = landmarks[0]
        x2, y2 = landmarks[1] if len(landmarks) > 1 else landmarks[0]
        gaze_point = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # 简单示例：假设画面中心区域为道路区域
        frame_center_x = 320  # 假设图像宽为640
        frame_center_y = 240  # 假设图像高为480
        
        # 计算视线点与中心点的距离
        distance = np.sqrt((gaze_point[0] - frame_center_x)**2 + 
                           (gaze_point[1] - frame_center_y)**2)
        
        # 如果距离小于阈值，认为在看前方道路
        # 实际应用中需要更精确的区域划分和判断逻辑
        is_looking_forward = distance < 100
        
        return gaze_point, is_looking_forward
        
    def _calculate_distraction_time(self, is_looking_road):
        """
        计算分心时间
        
        参数:
        - is_looking_road: 是否注视道路
        
        返回:
        - distraction_time: 分心持续时间（秒）
        """
        current_time = time.time()
        
        if not is_looking_road:
            # 如果没有看路，且之前没有记录分心开始时间，则记录当前时间
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
                return 0.0
            # 如果没有看路，且已有分心开始时间，则计算经过的时间
            else:
                return current_time - self.distraction_start_time
        else:
            # 如果看路，重置分心开始时间
            self.distraction_start_time = None
            return 0.0
    
    def process_frame(self, frame=None):
        """
        处理单帧图像，检测眼动状态
        
        参数:
        - frame: 输入的视频帧，如果为None则从摄像头获取
        
        返回:
        - result: EyeTrackingResult对象，包含眼动追踪结果
        """
        # 如果没有提供帧，则从摄像头获取
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                return None
        
        # 检测眼睛关键点
        landmarks = self._detect_eye_landmarks(frame)
        
        # 计算视线方向和是否看路
        gaze_point, is_looking_road = self._calculate_gaze_direction(landmarks)
        
        # 计算分心时间
        distraction_time = self._calculate_distraction_time(is_looking_road)
        
        # 创建结果对象
        result = EyeTrackingResult(
            is_looking_road=is_looking_road,
            gaze_coordinates=gaze_point,
            distraction_duration=distraction_time,
            confidence=0.8 if landmarks else 0.0,  # 简单示例：有关键点则置信度为0.8
            timestamp=time.time()
        )
        
        self.last_result = result
        return result
    
    def is_distracted(self):
        """
        判断驾驶员是否分心
        
        返回:
        - is_distracted: 是否分心
        - duration: 分心持续时间
        """
        if not self.last_result:
            return False, 0.0
            
        is_distracted = (not self.last_result.is_looking_road and 
                         self.last_result.distraction_duration >= self.distraction_threshold)
                         
        return is_distracted, self.last_result.distraction_duration
        
    def visualize(self, frame, result):
        """
        在图像上可视化眼动追踪结果
        
        参数:
        - frame: 输入的视频帧
        - result: EyeTrackingResult对象
        
        返回:
        - vis_frame: 可视化后的视频帧
        """
        if frame is None or result is None:
            return frame
            
        vis_frame = frame.copy()
        
        # 绘制视线焦点
        cv2.circle(vis_frame, result.gaze_coordinates, 5, (0, 255, 0), -1)
        
        # 绘制分心状态
        is_distracted, duration = self.is_distracted()
        color = (0, 0, 255) if is_distracted else (0, 255, 0)
        status_text = f"分心: {is_distracted}, 时长: {duration:.1f}秒"
        cv2.putText(vis_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        return vis_frame


# 示例使用代码
def eye_tracker_demo():
    """眼动追踪模块演示函数"""
    tracker = EyeTracker(distraction_threshold=3.0)
    
    try:
        tracker.initialize()
        
        while True:
            ret, frame = tracker.cap.read()
            if not ret:
                break
                
            # 处理帧
            result = tracker.process_frame(frame)
            
            # 可视化
            vis_frame = tracker.visualize(frame, result)
            
            # 显示结果
            cv2.imshow('Eye Tracking Demo', vis_frame)
            
            # 检查分心状态
            is_distracted, duration = tracker.is_distracted()
            if is_distracted:
                print(f"警告：驾驶员分心已持续 {duration:.1f} 秒!")
                
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    eye_tracker_demo()