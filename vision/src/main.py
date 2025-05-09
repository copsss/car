"""
车载多模态视觉交互系统主程序 - 集成眼动追踪、头部姿态识别和手势识别功能
"""

import cv2
import numpy as np
import time
import threading
from datetime import datetime

# 导入视觉系统模块
from vision_system.eye_tracking.eye_tracker import EyeTracker
from vision_system.head_pose.head_pose_detector import HeadPoseDetector, AttentionState
from vision_system.gesture_recognition.gesture_recognizer import GestureRecognizer, GestureType

# 导入中文显示工具
from utils.text_utils import put_chinese_text

class ModalityStatus:
    """模态状态类，用于记录各模态的状态信息"""
    def __init__(self):
        # 眼动追踪状态
        self.is_driver_distracted = False  # 驾驶员是否分心
        self.distraction_duration = 0.0    # 分心持续时间
        self.looking_road = True           # 是否在看路
        
        # 头部姿态状态
        self.head_attention = AttentionState.ATTENTIVE  # 当前头部注意力状态
        self.head_confidence = 0.0                     # 头部姿态置信度
        
        # 手势识别状态
        self.hand_gesture = GestureType.NORMAL      # 当前手势类型
        self.hand_confidence = 0.0                  # 手势置信度
        
        # 警告状态
        self.warning_active = False         # 是否正在警告
        self.warning_level = 0              # 警告级别 (0-3)
        self.warning_start_time = None      # 警告开始时间
        self.warning_acknowledged = False   # 警告是否已确认

class VisionSystem:
    """
    车载多模态视觉交互系统类
    
    集成了眼动追踪、头部姿态识别和手势识别三个模态，
    实现多模态融合的视觉交互功能
    """
    
    def __init__(self, eye_distraction_threshold=3.0, camera_id=0):
        """
        初始化视觉系统
        
        参数:
        - eye_distraction_threshold: 眼动分心阈值（秒）
        - camera_id: 摄像头ID
        """
        # 创建各个模态对象
        self.eye_tracker = EyeTracker(distraction_threshold=eye_distraction_threshold, 
                                     camera_id=camera_id)
        self.head_detector = HeadPoseDetector(camera_id=camera_id)
        self.gesture_recognizer = GestureRecognizer(camera_id=camera_id)
        
        # 系统状态
        self.status = ModalityStatus()
        
        # 系统控制
        self.running = False
        self.camera_lock = threading.Lock()  # 摄像头资源锁
        
        # 日志记录
        self.log_file = None
        self.start_time = None
        
    def initialize(self):
        """初始化系统"""
        print("初始化车载多模态视觉交互系统...")
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法访问摄像头")
            
        # 初始化各个模态
        # 注意：在实际系统中，每个模态可能有独立的摄像头
        # 这里为简化示例，共用一个摄像头
        
        # 由于共用摄像头，这里不直接调用各模块的initialize方法
        # 而是在处理帧时手动传入帧数据
        
        # 初始化日志
        log_filename = f"vision_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_file = open(log_filename, "w", encoding="utf-8")
        self.log_message("系统初始化")
        
        self.start_time = time.time()
        return True
        
    def release(self):
        """释放资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            
        if self.log_file:
            self.log_file.close()
            
    def log_message(self, message):
        """记录日志"""
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_entry = f"[{timestamp}] {message}\n"
            self.log_file.write(log_entry)
            self.log_file.flush()
            
    def process_modalities(self, frame):
        """
        处理所有模态
        
        参数:
        - frame: 输入的视频帧
        
        返回:
        - vis_frame: 可视化后的视频帧
        """
        # 处理眼动追踪
        eye_result = self.eye_tracker.process_frame(frame)
        
        # 处理头部姿态
        head_result = self.head_detector.process_frame(frame)
        
        # 处理手势识别
        gesture_result = self.gesture_recognizer.process_frame(frame)
        
        # 更新系统状态
        self._update_system_status(eye_result, head_result, gesture_result)
        
        # 执行多模态融合逻辑
        self._fusion_logic()
        
        # 可视化结果
        vis_frame = self._visualize_results(frame, eye_result, head_result, gesture_result)
        
        return vis_frame
        
    def _update_system_status(self, eye_result, head_result, gesture_result):
        """
        更新系统状态
        
        参数:
        - eye_result: 眼动追踪结果
        - head_result: 头部姿态结果
        - gesture_result: 手势识别结果
        """
        # 更新眼动状态
        if eye_result:
            is_distracted, duration = self.eye_tracker.is_distracted()
            self.status.is_driver_distracted = is_distracted
            self.status.distraction_duration = duration
            self.status.looking_road = eye_result.is_looking_road
            
        # 更新头部姿态状态
        if head_result:
            self.status.head_attention = head_result.attention
            self.status.head_confidence = head_result.confidence
            
        # 更新手势状态
        if gesture_result:
            self.status.hand_gesture = gesture_result.gesture
            self.status.hand_confidence = gesture_result.confidence
            
    def _fusion_logic(self):
        """
        多模态融合逻辑，处理模态间的关系和交互
        """
        # 检查是否需要触发警告
        if self.status.is_driver_distracted and not self.status.warning_active:
            # 驾驶员分心且当前没有活跃警告，触发新警告
            self.status.warning_active = True
            self.status.warning_level = 1
            self.status.warning_start_time = time.time()
            self.status.warning_acknowledged = False
            
            self.log_message(f"触发分心警告 - 持续时间: {self.status.distraction_duration:.1f}秒")
            
        # 警告级别升级逻辑
        if self.status.warning_active and not self.status.warning_acknowledged:
            warning_duration = time.time() - self.status.warning_start_time
            
            # 根据警告持续时间升级警告级别
            if warning_duration > 5.0 and self.status.warning_level < 2:
                self.status.warning_level = 2
                self.log_message(f"警告升级到级别2 - 持续时间: {warning_duration:.1f}秒")
                
            if warning_duration > 8.0 and self.status.warning_level < 3:
                self.status.warning_level = 3
                self.log_message(f"警告升级到级别3 - 持续时间: {warning_duration:.1f}秒")
                
        # 警告确认逻辑
        if self.status.warning_active:
            # 通过手势确认（竖起大拇指）
            if (self.status.hand_gesture == GestureType.THUMBS_UP and 
                  self.status.hand_confidence > 0.7):
                self._acknowledge_warning("竖起大拇指")
                
            # 通过注意力状态确认
            elif (self.status.head_attention == AttentionState.ATTENTIVE and 
                 self.status.head_confidence > 0.7):
                self._acknowledge_warning("恢复注意力")
                
            # 如果驾驶员重新注视道路，也可以自动确认警告
            elif self.status.looking_road and self.status.warning_level < 3:
                self._acknowledge_warning("重新注视道路")
                
        # 警告取消逻辑
        if self.status.warning_active and self.status.warning_acknowledged:
            # 如果已经确认警告，且持续注视道路超过2秒，取消警告状态
            if self.status.looking_road and not self.status.is_driver_distracted:
                self.status.warning_active = False
                self.log_message("警告状态已解除")
                
        # 警告拒绝逻辑
        if self.status.warning_active:
            # 通过手势拒绝（摇手）
            if (self.status.hand_gesture == GestureType.WAVE and 
                  self.status.hand_confidence > 0.7):
                self._reject_warning("摇手")
                
    def _acknowledge_warning(self, method):
        """
        确认警告
        
        参数:
        - method: 确认方法
        """
        if not self.status.warning_acknowledged:
            self.status.warning_acknowledged = True
            self.log_message(f"警告已确认 - 方法: {method}")
            
    def _reject_warning(self, method):
        """
        拒绝警告 - 在实际系统中，可能需要更严格的处理
        
        参数:
        - method: 拒绝方法
        """
        # 记录拒绝事件
        self.log_message(f"警告被拒绝 - 方法: {method}")
        
        # 在实际系统中，可能不允许拒绝高级别警告，这里简化处理
        if self.status.warning_level < 3:
            self.status.warning_acknowledged = True
            
    def _visualize_results(self, frame, eye_result, head_result, gesture_result):
        """
        可视化所有模态的结果
        
        参数:
        - frame: 输入的视频帧
        - eye_result: 眼动追踪结果
        - head_result: 头部姿态结果
        - gesture_result: 手势识别结果
        
        返回:
        - vis_frame: 可视化后的视频帧
        """
        # 创建可视化画布
        if frame is None:
            return None
            
        # 使用原始帧的副本
        vis_frame = frame.copy()
        
        # 分屏布局 - 将画面分为四个区域
        h, w = frame.shape[:2]
        
        # 1. 绘制主视图（集成视图）
        status_height = 100  # 状态栏高度
        
        # 如果存在警告，绘制警告状态
        if self.status.warning_active:
            # 根据警告级别选择颜色和闪烁频率
            warning_colors = [
                (0, 255, 255),  # 黄色 - 级别0
                (0, 165, 255),  # 橙色 - 级别1
                (0, 0, 255),    # 红色 - 级别2
                (0, 0, 255)     # 红色 - 级别3
            ]
            
            # 闪烁效果
            blink_freq = [0, 1, 2, 4]  # 每秒闪烁次数
            current_time = time.time()
            should_blink = int(current_time * blink_freq[self.status.warning_level]) % 2 == 0
            
            # 如果是高级别警告或处于闪烁期间，绘制警告边框
            if self.status.warning_level >= 2 or should_blink:
                # 绘制警告边框
                cv2.rectangle(vis_frame, (0, 0), (w, h), 
                            warning_colors[self.status.warning_level], 10)
                
            # 警告文本
            warning_text = "警告！请目视前方" if not self.status.warning_acknowledged else "请保持注意力"
            text_color = (255, 255, 255)  # 白色文本
            
            # 绘制警告文本背景
            cv2.rectangle(vis_frame, (0, 0), (w, status_height), 
                        warning_colors[self.status.warning_level], -1)
            
            # 绘制警告文本 - 使用中文文本显示函数
            text_size = 36
            text_x = w // 2 - len(warning_text) * text_size // 4
            vis_frame = put_chinese_text(
                vis_frame, warning_text, (text_x, status_height // 2 + text_size // 2), 
                text_color=text_color, text_size=text_size
            )
        
        # 2. 绘制各模态状态信息
        y_offset = h - 40
        
        # 眼动状态
        eye_status = f"眼动: {'分心' if self.status.is_driver_distracted else '注意'} ({self.status.distraction_duration:.1f}秒)"
        eye_color = (0, 0, 255) if self.status.is_driver_distracted else (0, 255, 0)
        vis_frame = put_chinese_text(vis_frame, eye_status, (10, y_offset), text_color=eye_color, text_size=24)
        
        # 头部姿态状态
        head_status = f"头部: {self.status.head_attention.name} ({self.status.head_confidence:.2f})"
        vis_frame = put_chinese_text(vis_frame, head_status, (w//3, y_offset), text_color=(0, 255, 0), text_size=24)
        
        # 手势状态
        hand_status = f"手势: {self.status.hand_gesture.name} ({self.status.hand_confidence:.2f})"
        vis_frame = put_chinese_text(vis_frame, hand_status, (2*w//3, y_offset), text_color=(0, 255, 0), text_size=24)
        
        # 3. 集成各模态的可视化效果
        # 这里可以根据需要添加更多可视化内容
        
        return vis_frame
        
    def run(self):
        """运行视觉系统主循环"""
        self.running = True
        
        try:
            while self.running:
                # 获取视频帧
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取视频帧")
                    break
                    
                # 处理所有模态并获取可视化结果
                vis_frame = self.process_modalities(frame)
                
                # 显示结果
                cv2.imshow('车载多模态视觉交互系统', vis_frame)
                
                # 检查警告状态并发出声音提示（实际中应使用更好的声音反馈）
                if self.status.warning_active and not self.status.warning_acknowledged:
                    if self.status.warning_level >= 2:
                        print('\a')  # 简单的蜂鸣声
                
                # 按q或ESC退出
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q或ESC
                    break
                    
        finally:
            self.running = False
            self.release()
            cv2.destroyAllWindows()


# 示例使用代码
def main():
    """主函数"""
    vision_system = VisionSystem(eye_distraction_threshold=3.0)
    
    try:
        vision_system.initialize()
        vision_system.run()
    except Exception as e:
        print(f"系统运行错误: {e}")
    finally:
        vision_system.release()


if __name__ == "__main__":
    main()