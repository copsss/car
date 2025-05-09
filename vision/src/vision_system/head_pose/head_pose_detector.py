"""
头部姿态检测模块 - 实现头部姿态的检测，包括转向角度和头部位置
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum, auto


class AttentionState(Enum):
    """注意力状态枚举类"""
    UNKNOWN = auto()       # 未知状态
    ATTENTIVE = auto()     # 注意力集中
    DISTRACTED = auto()    # 注意力分散
    DROWSY = auto()        # 困倦状态


@dataclass
class HeadPoseResult:
    """头部姿态检测结果数据类"""
    roll: float            # 头部滚转角度（绕z轴旋转）
    pitch: float           # 头部俯仰角度（绕x轴旋转）
    yaw: float             # 头部偏航角度（绕y轴旋转）
    position: tuple        # 头部位置坐标 (x, y)
    attention: AttentionState  # 注意力状态
    confidence: float      # 结果置信度
    timestamp: float       # 时间戳


class HeadPoseDetector:
    """
    头部姿态检测类 - 实现头部姿态检测相关功能
    
    功能：
    1. 检测头部在3D空间中的姿态（俯仰、偏航、滚转角度）
    2. 评估驾驶员的注意力状态
    3. 提供头部姿态的可视化
    """
    
    def __init__(self, camera_id=0, model_path=None):
        """
        初始化头部姿态检测器
        
        参数:
        - camera_id: 摄像头ID
        - model_path: 模型路径，如果为None则使用默认模型
        """
        self.camera_id = camera_id
        self.model_path = model_path
        self.last_result = None
        self.face_detector = None
        self.landmark_detector = None
        self.attention_history = []  # 存储历史注意力状态用于平滑预测
        self.history_size = 5        # 历史记录大小
        
    def initialize(self):
        """初始化头部姿态检测设备和模型"""
        print("初始化头部姿态检测模块...")
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("无法访问摄像头")
            
        # 初始化面部检测器
        # 使用OpenCV的预训练Haar级联分类器或DNN模型
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 在实际项目中，应该加载更准确的面部特征点检测模型
        # 例如dlib或OpenCV的face landmark detector
        # 这里简化处理，在实际项目中替换为更准确的模型
        
        # 3D模型点，代表人脸关键特征的3D坐标
        # 使用通用面部模型的标准化坐标
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # 鼻尖
            (0.0, -330.0, -65.0),     # 下巴
            (-225.0, 170.0, -135.0),  # 左眼左角
            (225.0, 170.0, -135.0),   # 右眼右角
            (-150.0, -150.0, -125.0), # 左嘴角
            (150.0, -150.0, -125.0)   # 右嘴角
        ])
        
        # 相机参数（这里使用估计值，实际应进行相机标定）
        self.camera_matrix = None
        self.dist_coeffs = None
        
        return True
        
    def release(self):
        """释放资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            
    def _detect_face(self, frame):
        """
        检测人脸区域
        
        参数:
        - frame: 输入的视频帧
        
        返回:
        - face_rect: 人脸区域坐标，格式为(x, y, w, h)
        - confidence: 检测置信度
        """
        if self.face_detector is None:
            return None, 0.0
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, 0.0
            
        # 选择最大的人脸（假设是驾驶员）
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # 由于Haar级联分类器没有提供置信度，这里根据人脸大小估计
        # 人脸越大，置信度越高
        face_area = face[2] * face[3]
        frame_area = frame.shape[0] * frame.shape[1]
        confidence = min(0.9, face_area / frame_area * 10)  # 归一化
        
        return face, confidence
        
    def _get_face_landmarks(self, frame, face_rect):
        """
        获取人脸特征点
        
        参数:
        - frame: 输入的视频帧
        - face_rect: 人脸区域坐标

        返回:
        - landmarks: 人脸特征点坐标列表
        """
        # 在实际项目中，应该使用专门的人脸特征点检测器
        # 例如dlib、OpenCV DNN或MediaPipe Face Mesh
        # 这里提供一个简化版实现，仅用于演示
        
        x, y, w, h = face_rect
        
        # 估计人脸关键点位置（简化版）
        # 在实际项目中，这里应该使用更准确的特征点检测算法
        landmarks = [
            (x + w // 2, y + h // 2),                 # 鼻尖
            (x + w // 2, y + h),                      # 下巴
            (x, y + h // 3),                          # 左眼左角
            (x + w, y + h // 3),                      # 右眼右角
            (x + w // 3, y + 2 * h // 3),            # 左嘴角
            (x + 2 * w // 3, y + 2 * h // 3)         # 右嘴角
        ]
        
        return np.array(landmarks, dtype=np.float32)
        
    def _estimate_head_pose(self, frame, landmarks):
        """
        估计头部姿态角度
        
        参数:
        - frame: 输入的视频帧
        - landmarks: 人脸特征点坐标

        返回:
        - roll, pitch, yaw: 头部的三个旋转角度
        - tvec: 头部位置向量
        """
        if landmarks is None or len(landmarks) < 6:
            return 0.0, 0.0, 0.0, (0, 0, 0)
            
        # 初始化相机矩阵（如果尚未设置）
        if self.camera_matrix is None:
            height, width = frame.shape[:2]
            focal_length = width
            center = (width / 2, height / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float32
            )
            
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1))
            
        # 求解PnP问题，获取旋转和平移向量
        success, rvec, tvec = cv2.solvePnP(
            self.model_points, 
            landmarks, 
            self.camera_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0, (0, 0, 0)
            
        # 将旋转向量转换为旋转矩阵
        rmat, _ = cv2.Rodrigues(rvec)
        
        # 将旋转矩阵转换为欧拉角
        angles = self._rotation_matrix_to_euler_angles(rmat)
        
        # 将弧度转换为度数
        roll = angles[2] * 180 / np.pi
        pitch = angles[0] * 180 / np.pi
        yaw = angles[1] * 180 / np.pi
        
        # 头部位置
        head_position = (int(tvec[0][0]), int(tvec[1][0]))
        
        return roll, pitch, yaw, head_position
        
    def _rotation_matrix_to_euler_angles(self, R):
        """
        将旋转矩阵转换为欧拉角
        
        参数:
        - R: 3x3旋转矩阵
        
        返回:
        - angles: [pitch, yaw, roll]的数组
        """
        # 确保是有效的旋转矩阵
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # pitch
            y = np.arctan2(-R[2, 0], sy)      # yaw
            z = np.arctan2(R[1, 0], R[0, 0])  # roll
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
            
        return np.array([x, y, z])
        
    def _assess_attention(self, roll, pitch, yaw):
        """
        评估驾驶员注意力状态
        
        参数:
        - roll, pitch, yaw: 头部姿态角度
        
        返回:
        - attention_state: 注意力状态
        - confidence: 评估置信度
        """
        # 设置角度阈值（可根据实际情况调整）
        roll_threshold = 20.0
        pitch_threshold = 20.0
        yaw_threshold = 30.0
        
        # 检查是否分心（头部偏转过大）
        is_distracted = (
            abs(roll) > roll_threshold or
            abs(pitch) > pitch_threshold or
            abs(yaw) > yaw_threshold
        )
        
        # 检查是否困倦（头部微微低垂）
        is_drowsy = (pitch > 10.0 and abs(roll) < 10.0 and abs(yaw) < 10.0)
        
        # 确定注意力状态
        if is_drowsy:
            attention = AttentionState.DROWSY
            # 计算置信度（与阈值的距离相关）
            confidence = min(0.9, 0.5 + (pitch - 10.0) / 20.0)
        elif is_distracted:
            attention = AttentionState.DISTRACTED
            # 计算置信度（与阈值的距离相关）
            roll_conf = max(0, (abs(roll) - roll_threshold)) / roll_threshold
            pitch_conf = max(0, (abs(pitch) - pitch_threshold)) / pitch_threshold
            yaw_conf = max(0, (abs(yaw) - yaw_threshold)) / yaw_threshold
            confidence = min(0.9, 0.5 + max(roll_conf, pitch_conf, yaw_conf))
        else:
            attention = AttentionState.ATTENTIVE
            # 计算置信度（与阈值的距离相关）
            roll_factor = 1.0 - abs(roll) / roll_threshold
            pitch_factor = 1.0 - abs(pitch) / pitch_threshold
            yaw_factor = 1.0 - abs(yaw) / yaw_threshold
            confidence = min(0.9, 0.5 + (roll_factor + pitch_factor + yaw_factor) / 6.0)
            
        # 使用历史状态平滑结果
        self.attention_history.append((attention, confidence))
        if len(self.attention_history) > self.history_size:
            self.attention_history.pop(0)
            
        # 如果历史记录中有超过一半的状态与当前状态不同，且它们的置信度较高
        # 则采用历史多数状态
        if len(self.attention_history) >= 3:
            counter = {}
            for att, conf in self.attention_history:
                if att not in counter:
                    counter[att] = 0
                counter[att] += conf
                
            # 找出历史记录中置信度加权最高的状态
            max_att = max(counter.items(), key=lambda x: x[1])
            if max_att[0] != attention and max_att[1] > confidence * 1.5:
                attention = max_att[0]
                confidence = min(0.9, confidence * 0.8 + 0.1)
                
        return attention, confidence

    def process_frame(self, frame=None):
        """
        处理单帧图像，检测头部姿态
        
        参数:
        - frame: 输入的视频帧，如果为None则从摄像头获取
        
        返回:
        - result: HeadPoseResult对象，包含头部姿态检测结果
        """
        # 如果没有提供帧，则从摄像头获取
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                return None
                
        # 检测人脸
        face_rect, face_confidence = self._detect_face(frame)
        
        if face_rect is None:
            # 没有检测到人脸
            result = HeadPoseResult(
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                position=(0, 0),
                attention=AttentionState.UNKNOWN,
                confidence=0.0,
                timestamp=time.time()
            )
            self.last_result = result
            return result
            
        # 获取人脸特征点
        landmarks = self._get_face_landmarks(frame, face_rect)
        
        # 估计头部姿态
        roll, pitch, yaw, head_position = self._estimate_head_pose(frame, landmarks)
        
        # 评估注意力状态
        attention, attention_confidence = self._assess_attention(roll, pitch, yaw)
        
        # 综合置信度（人脸检测和注意力评估的加权平均）
        confidence = 0.6 * face_confidence + 0.4 * attention_confidence
        
        # 创建结果对象
        result = HeadPoseResult(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            position=head_position,
            attention=attention,
            confidence=confidence,
            timestamp=time.time()
        )
        
        self.last_result = result
        return result
        
    def visualize(self, frame, result):
        """
        在图像上可视化头部姿态检测结果
        
        参数:
        - frame: 输入的视频帧
        - result: HeadPoseResult对象
        
        返回:
        - vis_frame: 可视化后的视频帧
        """
        if frame is None or result is None:
            return frame
            
        vis_frame = frame.copy()
        
        # 如果没有有效的头部姿态，返回原始帧
        if result.confidence <= 0.1:
            cv2.putText(vis_frame, "未检测到人脸", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis_frame
            
        # 1. 在图像上绘制头部姿态信息
        roll_text = f"Roll: {result.roll:.1f}"
        pitch_text = f"Pitch: {result.pitch:.1f}"
        yaw_text = f"Yaw: {result.yaw:.1f}"
        
        cv2.putText(vis_frame, roll_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, pitch_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, yaw_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        # 2. 绘制注意力状态
        attention_text = f"注意力: {result.attention.name}"
        conf_text = f"置信度: {result.confidence:.2f}"
        
        # 根据注意力状态选择颜色
        if result.attention == AttentionState.ATTENTIVE:
            color = (0, 255, 0)  # 绿色 - 注意力集中
        elif result.attention == AttentionState.DISTRACTED:
            color = (0, 0, 255)  # 红色 - 注意力分散
        elif result.attention == AttentionState.DROWSY:
            color = (0, 165, 255)  # 橙色 - 困倦
        else:
            color = (255, 255, 255)  # 白色 - 未知状态
            
        cv2.putText(vis_frame, attention_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_frame, conf_text, (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        # 3. 可视化头部姿态轴
        try:
            # 在实际项目中，应该使用更准确的方法绘制头部姿态
            # 这里提供一个简化版本
            face_rect, _ = self._detect_face(frame)
            if face_rect is not None:
                x, y, w, h = face_rect
                
                # 绘制人脸框
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # 绘制头部中心点
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(vis_frame, (center_x, center_y), 3, (0, 255, 0), -1)
                
                # 绘制头部姿态方向（简化版）
                # Roll - 绿色
                roll_rad = result.roll * np.pi / 180.0
                roll_x = int(center_x + 50 * np.sin(roll_rad))
                roll_y = int(center_y + 50 * np.cos(roll_rad))
                cv2.line(vis_frame, (center_x, center_y), (roll_x, roll_y), (0, 255, 0), 2)
                
                # Pitch - 蓝色
                pitch_rad = result.pitch * np.pi / 180.0
                pitch_factor = np.cos(pitch_rad)  # 透视效果
                pitch_y = int(center_y - 50 * np.sin(pitch_rad))
                cv2.line(vis_frame, (center_x, center_y), (center_x, pitch_y), (255, 0, 0), 2)
                
                # Yaw - 红色
                yaw_rad = result.yaw * np.pi / 180.0
                yaw_x = int(center_x + 50 * np.sin(yaw_rad) * pitch_factor)
                cv2.line(vis_frame, (center_x, center_y), (yaw_x, center_y), (0, 0, 255), 2)
        except Exception as e:
            print(f"可视化头部姿态时出错: {e}")
                    
        return vis_frame


# 示例使用代码
def head_pose_detector_demo():
    """头部姿态检测模块演示函数"""
    detector = HeadPoseDetector()
    
    try:
        detector.initialize()
        
        while True:
            ret, frame = detector.cap.read()
            if not ret:
                break
                
            # 处理帧
            result = detector.process_frame(frame)
            
            # 可视化
            vis_frame = detector.visualize(frame, result)
            
            # 显示结果
            cv2.imshow('Head Pose Detection Demo', vis_frame)
            
            # 根据注意力状态打印警告
            if result and result.attention == AttentionState.DISTRACTED:
                print("警告：注意力分散！")
            elif result and result.attention == AttentionState.DROWSY:
                print("警告：驾驶员可能疲劳！")
                
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        detector.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    head_pose_detector_demo()