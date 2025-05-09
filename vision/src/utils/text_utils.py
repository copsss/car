"""
显示工具模块 - 提供中文文本显示等功能
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def put_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=30):
    """
    在图片上显示中文
    
    参数:
    - img: OpenCV格式的图像
    - text: 要显示的文本
    - position: 文本位置，元组(x, y)
    - text_color: 文本颜色，默认绿色
    - text_size: 文本大小
    
    返回:
    - img_pil: 添加文本后的图像
    """
    # 判断图像是否为OpenCV格式，如果是则转换为RGB
    if isinstance(img, np.ndarray):
        # 将OpenCV的BGR格式转换为RGB格式
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img_pil)
    
    # 加载字体，使用系统默认的微软雅黑字体
    try:
        font = ImageFont.truetype("msyh.ttc", text_size)  # 微软雅黑
    except IOError:
        try:
            font = ImageFont.truetype("simsun.ttc", text_size)  # 宋体
        except IOError:
            # 如果找不到上述字体，则使用默认字体
            font = ImageFont.load_default()
    
    # 在图片上绘制文字
    draw.text(position, text, text_color, font=font)
    
    # 将PIL图像格式转换回OpenCV格式
    img_opencv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_opencv