# 基于GazeCapture数据集的眼动追踪模型训练方案

## 1. 引言

本文档详细描述了使用GazeCapture数据集训练眼动追踪模型的完整流程，包括数据获取、预处理、模型设计、训练策略和领域适应等关键步骤。该模型将应用于车载多模态交互系统中，用于实时追踪驾驶员的眼动行为，监测注意力状态。

## 2. GazeCapture数据集概述

### 2.1 数据集简介

GazeCapture是目前最大的用于眼动追踪的公开数据集，由苹果和MIT联合发布：

- **规模**：包含超过2.5M帧的眼动数据
- **多样性**：1450名参与者，不同年龄、性别和种族背景
- **设备**：iOS设备（iPhone和iPad），使用前置摄像头采集
- **标注**：包含精确的注视点坐标（x, y）和面部关键点位置
- **访问方式**：通过[官方网站](http://gazecapture.csail.mit.edu/)申请获取

### 2.2 数据集结构

GazeCapture数据集的组织结构如下：

```
GazeCapture/
├── metadata.mat        # 全局元数据
├── README.txt          # 数据集说明
└── XXX/                # 按参与者ID组织的文件夹
    ├── appleFace.json  # 苹果API检测的面部关键点
    ├── dotInfo.json    # 屏幕上显示的点的信息
    ├── faceGrid.mat    # 面部在图像中的网格位置
    ├── frames/         # 包含摄像头捕获的图像帧
    │   ├── XXXX.jpg    # 帧图像
    ├── info.json       # 会话信息
    └── screen.mat      # 屏幕尺寸和方向信息
```

### 2.3 数据集特点

- **自然场景多样性**：包含不同光照、背景和头部姿态下的数据
- **设备多样性**：不同设备型号和屏幕尺寸
- **校准序列**：包含设备校准过程中的数据
- **注视轨迹**：包含随机点移动的追踪数据

## 3. 数据预处理流程

### 3.1 数据下载与组织

1. 向项目管理员申请数据集访问权限
2. 下载数据集（约120GB）并解压到指定目录
3. 验证数据完整性，确保所有子目录和文件正确

### 3.2 数据清洗

```python
def clean_gazecapture_data(dataset_path):
    """清洗GazeCapture数据集，移除低质量样本"""
    valid_participants = []
    total_frames = 0
    
    # 遍历所有参与者
    for participant_id in os.listdir(dataset_path):
        participant_path = os.path.join(dataset_path, participant_id)
        if not os.path.isdir(participant_path):
            continue
            
        # 加载会话信息
        info_path = os.path.join(participant_path, 'info.json')
        if not os.path.exists(info_path):
            print(f"警告: 参与者 {participant_id} 缺少info.json文件")
            continue
            
        with open(info_path, 'r') as f:
            info = json.load(f)
            
        # 检查数据质量
        if info.get('screenH', 0) < 400 or info.get('screenW', 0) < 400:
            print(f"警告: 参与者 {participant_id} 屏幕分辨率过低")
            continue
            
        # 检查帧数量
        frames_dir = os.path.join(participant_path, 'frames')
        if not os.path.exists(frames_dir):
            print(f"警告: 参与者 {participant_id} 缺少frames目录")
            continue
            
        frame_count = len(os.listdir(frames_dir))
        if frame_count < 100:
            print(f"警告: 参与者 {participant_id} 帧数过少: {frame_count}")
            continue
            
        # 检查面部检测质量
        face_path = os.path.join(participant_path, 'appleFace.json')
        if not os.path.exists(face_path):
            print(f"警告: 参与者 {participant_id} 缺少appleFace.json文件")
            continue
            
        with open(face_path, 'r') as f:
            face_data = json.load(f)
            
        # 计算有效面部检测比例
        valid_faces = sum(1 for f in face_data if f.get('confidence', 0) > 0.5)
        valid_ratio = valid_faces / len(face_data) if face_data else 0
        
        if valid_ratio < 0.8:
            print(f"警告: 参与者 {participant_id} 有效面部检测比例过低: {valid_ratio:.2f}")
            continue
            
        # 添加到有效参与者列表
        valid_participants.append(participant_id)
        total_frames += frame_count
        
    print(f"清洗完成: 保留 {len(valid_participants)}/{len(os.listdir(dataset_path))} 参与者")
    print(f"总有效帧数: {total_frames}")
    
    return valid_participants
```

### 3.3 特征提取

1. **面部对齐与裁剪**:

```python
def extract_eye_patches(frame, face_landmarks, eye_size=(224, 224)):
    """根据面部关键点提取左右眼图像"""
    # 获取眼睛关键点
    left_eye_points = face_landmarks[36:42]  # 左眼关键点索引
    right_eye_points = face_landmarks[42:48]  # 右眼关键点索引
    
    # 计算眼睛中心和边界框
    left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
    right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
    
    # 计算眼睛区域扩展范围
    eye_width = int(eye_size[0] // 2)
    eye_height = int(eye_size[1] // 2)
    
    # 提取左眼图像
    left_eye_patch = frame[
        max(0, left_eye_center[1] - eye_height):min(frame.shape[0], left_eye_center[1] + eye_height),
        max(0, left_eye_center[0] - eye_width):min(frame.shape[1], left_eye_center[0] + eye_width)
    ]
    
    # 提取右眼图像
    right_eye_patch = frame[
        max(0, right_eye_center[1] - eye_height):min(frame.shape[0], right_eye_center[1] + eye_height),
        max(0, right_eye_center[0] - eye_width):min(frame.shape[1], right_eye_center[0] + eye_width)
    ]
    
    # 调整大小
    if left_eye_patch.size > 0:
        left_eye_patch = cv2.resize(left_eye_patch, eye_size)
    else:
        left_eye_patch = np.zeros((eye_size[1], eye_size[0], 3), dtype=np.uint8)
        
    if right_eye_patch.size > 0:
        right_eye_patch = cv2.resize(right_eye_patch, eye_size)
    else:
        right_eye_patch = np.zeros((eye_size[1], eye_size[0], 3), dtype=np.uint8)
    
    return left_eye_patch, right_eye_patch
```

2. **注视点坐标归一化**:

```python
def normalize_gaze_coordinates(gaze_x, gaze_y, screen_w, screen_h):
    """将注视点坐标归一化到 [-1, 1] 范围"""
    norm_x = (gaze_x / screen_w) * 2 - 1
    norm_y = (gaze_y / screen_h) * 2 - 1
    
    return norm_x, norm_y
```

### 3.4 数据增强

为提高模型鲁棒性，我们应用以下数据增强方法：

```python
def augment_eye_patch(eye_patch):
    """对眼部图像进行数据增强"""
    # 随机亮度变化
    brightness = random.uniform(0.8, 1.2)
    eye_patch = cv2.convertScaleAbs(eye_patch, alpha=brightness, beta=0)
    
    # 随机对比度变化
    contrast = random.uniform(0.8, 1.2)
    eye_patch = cv2.convertScaleAbs(eye_patch, alpha=contrast, beta=0)
    
    # 随机翻转（仅水平翻转，避免左右眼混淆）
    if random.random() > 0.5:
        eye_patch = cv2.flip(eye_patch, 1)
    
    # 随机旋转（角度较小，避免眼睛特征丢失）
    angle = random.uniform(-10, 10)
    h, w = eye_patch.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    eye_patch = cv2.warpAffine(eye_patch, M, (w, h))
    
    # 随机噪声
    if random.random() > 0.8:
        noise = np.random.normal(0, 5, eye_patch.shape).astype(np.uint8)
        eye_patch = cv2.add(eye_patch, noise)
    
    return eye_patch
```

### 3.5 数据集构建

我们将数据集分为训练集、验证集和测试集，比例为70%、15%、15%。

```python
def build_dataset(valid_participants, dataset_path, output_path):
    """构建训练集、验证集和测试集"""
    # 随机打乱参与者列表
    random.shuffle(valid_participants)
    
    # 按比例划分
    train_size = int(len(valid_participants) * 0.7)
    val_size = int(len(valid_participants) * 0.15)
    
    train_participants = valid_participants[:train_size]
    val_participants = valid_participants[train_size:train_size+val_size]
    test_participants = valid_participants[train_size+val_size:]
    
    # 创建输出目录
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
    
    # 处理每个参与者的数据
    process_participants(train_participants, dataset_path, os.path.join(output_path, 'train'), augment=True)
    process_participants(val_participants, dataset_path, os.path.join(output_path, 'val'), augment=False)
    process_participants(test_participants, dataset_path, os.path.join(output_path, 'test'), augment=False)
```

## 4. 模型架构设计

### 4.1 iTracker模型概述

我们基于iTracker架构设计眼动追踪模型，该模型能够有效地从眼部和面部图像中提取特征，预测注视点位置。

### 4.2 网络结构

```python
class EyeTrackingModel(nn.Module):
    def __init__(self):
        super(EyeTrackingModel, self).__init__()
        
        # 左眼和右眼共享权重的特征提取网络
        self.eye_model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # 面部特征提取网络
        self.face_model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # 特征融合和回归网络
        self.fusion_model = nn.Sequential(
            nn.Linear(2*64*13*13 + 64*13*13, 1024),  # 两只眼睛 + 面部特征
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 输出x, y坐标
        )
        
    def forward(self, left_eye, right_eye, face):
        # 提取左右眼特征
        left_eye_feat = self.eye_model(left_eye)
        right_eye_feat = self.eye_model(right_eye)
        
        # 提取面部特征
        face_feat = self.face_model(face)
        
        # 特征融合
        features = torch.cat([left_eye_feat, right_eye_feat, face_feat], dim=1)
        
        # 预测注视点坐标
        gaze = self.fusion_model(features)
        
        return gaze
```

### 4.3 损失函数

我们使用欧几里得距离作为主要损失函数，计算预测注视点与真实注视点之间的距离：

```python
def gaze_loss(pred, target):
    """计算注视点预测损失"""
    return torch.norm(pred - target, dim=1).mean()
```

### 4.4 模型变体

为适应车载环境，我们设计了几个模型变体进行比较实验：

1. **基础iTracker**：原始架构，使用眼部和面部图像
2. **轻量化iTracker**：减少卷积层数量和通道数
3. **仅眼部模型**：仅使用眼部图像，降低计算复杂度
4. **注意力增强**：在特征融合层加入注意力机制

## 5. 训练策略

### 5.1 优化器与学习率

```python
def configure_optimizer(model, lr=0.0001):
    """配置优化器和学习率调度器"""
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器：每10轮衰减为原来的0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    return optimizer, scheduler
```

### 5.2 训练循环

```python
def train_model(model, train_loader, val_loader, epochs=50):
    """训练眼动追踪模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 配置优化器和调度器
    optimizer, scheduler = configure_optimizer(model)
    
    # 记录最佳模型
    best_val_loss = float('inf')
    best_model_path = 'best_gaze_model.pth'
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch_idx, (left_eyes, right_eyes, faces, targets) in enumerate(train_loader):
            left_eyes, right_eyes, faces = left_eyes.to(device), right_eyes.to(device), faces.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(left_eyes, right_eyes, faces)
            loss = gaze_loss(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for left_eyes, right_eyes, faces, targets in val_loader:
                left_eyes, right_eyes, faces = left_eyes.to(device), right_eyes.to(device), faces.to(device)
                targets = targets.to(device)
                
                outputs = model(left_eyes, right_eyes, faces)
                loss = gaze_loss(outputs, targets)
                val_loss += loss.item()
        
        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with val_loss: {val_loss:.4f}')
            
    return best_model_path
```

### 5.3 早停策略

为避免过拟合，我们实现了基于验证集性能的早停策略：

```python
def train_with_early_stopping(model, train_loader, val_loader, patience=5, epochs=50):
    """使用早停策略训练模型"""
    # 配置优化器和调度器
    optimizer, scheduler = configure_optimizer(model)
    
    # 早停参数
    counter = 0
    best_val_loss = float('inf')
    best_model_path = 'best_gaze_model.pth'
    
    for epoch in range(epochs):
        # 训练和验证代码（同上）
        # ...
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
                
    return best_model_path
```

## 6. 领域适应

由于GazeCapture数据集主要基于移动设备采集，我们需要将模型适应到车载环境。

### 6.1 合成数据生成

我们使用Unity模拟器生成车载环境下的眼动数据：

```python
def generate_synthetic_car_data(output_path, num_samples=10000):
    """生成合成车载环境眼动数据"""
    # 注意：这里提供的是伪代码，实际实现需要Unity环境
    
    # 配置车载环境参数
    car_environments = [
        {'time': 'day', 'weather': 'sunny'},
        {'time': 'day', 'weather': 'cloudy'},
        {'time': 'night', 'weather': 'clear'},
        {'time': 'night', 'weather': 'rainy'}
    ]
    
    # 生成样本
    for i in range(num_samples):
        # 随机选择环境
        env = random.choice(car_environments)
        
        # 设置Unity场景
        unity_simulator.set_environment(env)
        
        # 随机生成注视点
        gaze_x, gaze_y = random.uniform(-1, 1), random.uniform(-1, 1)
        unity_simulator.set_gaze_target(gaze_x, gaze_y)
        
        # 渲染和捕获图像
        face_img, left_eye_img, right_eye_img = unity_simulator.capture_images()
        
        # 保存数据
        sample_path = os.path.join(output_path, f'synthetic_{i:06d}')
        os.makedirs(sample_path, exist_ok=True)
        
        cv2.imwrite(os.path.join(sample_path, 'face.jpg'), face_img)
        cv2.imwrite(os.path.join(sample_path, 'left_eye.jpg'), left_eye_img)
        cv2.imwrite(os.path.join(sample_path, 'right_eye.jpg'), right_eye_img)
        
        # 保存标签
        with open(os.path.join(sample_path, 'gaze.txt'), 'w') as f:
            f.write(f'{gaze_x} {gaze_y}')
```

### 6.2 迁移学习

使用预训练模型作为起点，在合成数据和少量真实车载数据上进行微调：

```python
def fine_tune_model(pretrained_model_path, car_data_loader, epochs=10):
    """在车载数据上微调预训练模型"""
    # 加载预训练模型
    model = EyeTrackingModel()
    model.load_state_dict(torch.load(pretrained_model_path))
    
    # 冻结特征提取层，只微调融合层
    for param in model.eye_model.parameters():
        param.requires_grad = False
    
    for param in model.face_model.parameters():
        param.requires_grad = False
    
    # 配置优化器（仅优化未冻结参数）
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.00005
    )
    
    # 微调过程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (left_eyes, right_eyes, faces, targets) in enumerate(car_data_loader):
            left_eyes, right_eyes, faces = left_eyes.to(device), right_eyes.to(device), faces.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(left_eyes, right_eyes, faces)
            loss = gaze_loss(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均损失
        train_loss /= len(car_data_loader)
        print(f'Fine-tuning Epoch: {epoch}, Loss: {train_loss:.4f}')
    
    # 保存微调后的模型
    torch.save(model.state_dict(), 'car_adapted_gaze_model.pth')
    
    return 'car_adapted_gaze_model.pth'
```

### 6.3 领域对抗训练

为进一步缩小域差距，我们使用领域对抗训练方法：

```python
class DomainAdversarialModel(nn.Module):
    def __init__(self, base_model):
        super(DomainAdversarialModel, self).__init__()
        self.base_model = base_model
        
        # 添加域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # 二分类：源域 vs 目标域
            nn.Sigmoid()
        )
        
    def forward(self, left_eye, right_eye, face, alpha=1.0):
        # 特征提取（直到倒数第二层）
        left_feat = self.base_model.eye_model(left_eye)
        right_feat = self.base_model.eye_model(right_eye)
        face_feat = self.base_model.face_model(face)
        
        combined_feat = torch.cat([left_feat, right_feat, face_feat], dim=1)
        
        # 向前传播到倒数第二层获取特征
        x = self.base_model.fusion_model[:-1](combined_feat)
        
        # 梯度反转层
        reversed_features = GradientReversalLayer.apply(x, alpha)
        
        # 主任务：注视点预测
        gaze = self.base_model.fusion_model[-1](x)
        
        # 辅助任务：域分类
        domain_preds = self.domain_classifier(reversed_features)
        
        return gaze, domain_preds
```

## 7. 模型评估

### 7.1 评估指标

我们使用以下指标评估模型性能：

```python
def evaluate_model(model, test_loader):
    """评估模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 评估指标
    total_samples = 0
    mse_loss = 0
    angle_errors = []
    
    with torch.no_grad():
        for left_eyes, right_eyes, faces, targets in test_loader:
            left_eyes, right_eyes, faces = left_eyes.to(device), right_eyes.to(device), faces.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(left_eyes, right_eyes, faces)
            
            # 计算MSE
            batch_mse = torch.mean((outputs - targets) ** 2, dim=1).sum()
            mse_loss += batch_mse.item()
            
            # 计算角度误差（假设归一化坐标，需要根据实际设备进行角度转换）
            # 这里使用简化公式，实际应用中需要根据设备和距离进行调整
            batch_angles = torch.atan2(
                torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1)),
                torch.tensor(0.57).to(device)  # 假设平均视距为57cm
            ) * (180 / np.pi)
            
            angle_errors.extend(batch_angles.cpu().numpy())
            total_samples += targets.size(0)
    
    # 计算平均指标
    avg_mse = mse_loss / total_samples
    avg_angle_error = np.mean(angle_errors)
    median_angle_error = np.median(angle_errors)
    
    print(f'Evaluation Results:')
    print(f'  Average MSE: {avg_mse:.4f}')
    print(f'  Average Angular Error: {avg_angle_error:.2f} degrees')
    print(f'  Median Angular Error: {median_angle_error:.2f} degrees')
    
    return {
        'mse': avg_mse,
        'mean_angle_error': avg_angle_error,
        'median_angle_error': median_angle_error
    }
```

### 7.2 分心检测评估

专门评估模型在分心检测任务上的性能：

```python
def evaluate_distraction_detection(model, test_loader, distraction_threshold=0.3):
    """评估分心检测性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 评估指标
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # 定义屏幕中心区域（道路区域）
    road_region = {
        'x_min': -0.2,
        'x_max': 0.2,
        'y_min': -0.2,
        'y_max': 0.1
    }
    
    with torch.no_grad():
        for left_eyes, right_eyes, faces, targets in test_loader:
            left_eyes, right_eyes, faces = left_eyes.to(device), right_eyes.to(device), faces.to(device)
            targets = targets.to(device)
            
            # 前向传播
            pred_gazes = model(left_eyes, right_eyes, faces)
            
            for i in range(len(targets)):
                pred_x, pred_y = pred_gazes[i].cpu().numpy()
                true_x, true_y = targets[i].cpu().numpy()
                
                # 预测分心状态
                pred_distracted = not (
                    road_region['x_min'] <= pred_x <= road_region['x_max'] and
                    road_region['y_min'] <= pred_y <= road_region['y_max']
                )
                
                # 真实分心状态
                true_distracted = not (
                    road_region['x_min'] <= true_x <= road_region['x_max'] and
                    road_region['y_min'] <= true_y <= road_region['y_max']
                )
                
                # 更新混淆矩阵
                if pred_distracted and true_distracted:
                    true_positives += 1
                elif pred_distracted and not true_distracted:
                    false_positives += 1
                elif not pred_distracted and not true_distracted:
                    true_negatives += 1
                else:
                    false_negatives += 1
    
    # 计算评估指标
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f'Distraction Detection Results:')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  F1 Score: {f1:.4f}')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

## 8. 模型部署

### 8.1 模型优化

为在车载环境中高效部署，我们对模型进行以下优化：

```python
def optimize_model_for_deployment(model_path, output_path):
    """优化模型以便在车载环境部署"""
    # 加载训练好的模型
    model = EyeTrackingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 1. 模型量化
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    # 2. 模型剪枝
    # 注意：实际剪枝需要更复杂的实现，这里简化为伪代码
    pruned_model = apply_pruning(model)
    
    # 3. 导出为ONNX格式
    left_eye = torch.randn(1, 3, 224, 224)
    right_eye = torch.randn(1, 3, 224, 224)
    face = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        (left_eye, right_eye, face),
        os.path.join(output_path, "gaze_model.onnx"),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['left_eye', 'right_eye', 'face'],
        output_names=['gaze'],
        dynamic_axes={
            'left_eye': {0: 'batch_size'},
            'right_eye': {0: 'batch_size'},
            'face': {0: 'batch_size'},
            'gaze': {0: 'batch_size'}
        }
    )
    
    # 4. 转换为TensorRT格式（如果车载系统支持）
    # 注意：这需要TensorRT环境，这里提供伪代码
    """
    import tensorrt as trt
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(os.path.join(output_path, "gaze_model.onnx"), 'rb') as model_file:
        parser.parse(model_file.read())
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    engine = builder.build_engine(network, config)
    
    with open(os.path.join(output_path, "gaze_model.trt"), 'wb') as f:
        f.write(engine.serialize())
    """
    
    return os.path.join(output_path, "gaze_model.onnx")
```

### 8.2 集成接口

提供简洁的接口，方便与车载系统集成：

```python
class EyeTrackingService:
    def __init__(self, model_path, device='cuda'):
        """初始化眼动追踪服务"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = EyeTrackingModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化分心状态
        self.distraction_start_time = None
        self.is_looking_road = True
        self.distraction_threshold = 3.0  # 3秒分心阈值
        
    def process_frame(self, frame):
        """处理单帧图像"""
        # 检测面部
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(frame)
        
        if len(faces) == 0:
            return None, True  # 如果没有检测到面部，假设看向道路
        
        face = faces[0]
        
        # 检测面部关键点
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        landmarks = predictor(frame, face)
        
        # 提取眼部图像
        left_eye_img, right_eye_img = extract_eye_patches(frame, landmarks)
        face_img = extract_face_patch(frame, landmarks)
        
        # 图像预处理
        left_eye_tensor = preprocess_image(left_eye_img).unsqueeze(0).to(self.device)
        right_eye_tensor = preprocess_image(right_eye_img).unsqueeze(0).to(self.device)
        face_tensor = preprocess_image(face_img).unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            gaze_prediction = self.model(left_eye_tensor, right_eye_tensor, face_tensor)
        
        # 转换为numpy数组
        gaze_coords = gaze_prediction[0].cpu().numpy()
        
        # 判断是否看向道路区域
        is_looking_road = self._is_looking_road(gaze_coords)
        
        # 更新分心状态
        self._update_distraction_state(is_looking_road)
        
        return gaze_coords, is_looking_road
    
    def _is_looking_road(self, gaze_coords):
        """判断是否看向道路区域"""
        x, y = gaze_coords
        
        # 定义道路区域（归一化坐标）
        road_region = {
            'x_min': -0.2,
            'x_max': 0.2,
            'y_min': -0.2,
            'y_max': 0.1
        }
        
        return (road_region['x_min'] <= x <= road_region['x_max'] and
                road_region['y_min'] <= y <= road_region['y_max'])
    
    def _update_distraction_state(self, is_looking_road):
        """更新分心状态"""
        current_time = time.time()
        
        if not is_looking_road:
            # 如果没有看路，且之前没有记录分心开始时间，则记录当前时间
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
        else:
            # 如果看路，重置分心开始时间
            self.distraction_start_time = None
        
        self.is_looking_road = is_looking_road
    
    def is_distracted(self):
        """判断驾驶员是否分心"""
        if self.distraction_start_time is None:
            return False, 0.0
            
        # 计算分心持续时间
        distraction_duration = time.time() - self.distraction_start_time
        
        # 判断是否超过阈值
        is_distracted = distraction_duration >= self.distraction_threshold
        
        return is_distracted, distraction_duration
```

## 9. 结论与建议

### 9.1 训练建议

1. **数据多样性**：确保训练数据包含不同光照条件、不同人群和不同驾驶环境
2. **领域适应**：先在GazeCapture上预训练，再在车载数据上微调
3. **模型选择**：根据车载硬件性能选择合适复杂度的模型变体
4. **评估重点**：优先考虑分心检测性能，其次是注视点精度

### 9.2 性能优化

1. **批处理推理**：在实际部署中使用批处理模式提高吞吐量
2. **硬件加速**：利用车载GPU/NPU加速模型推理
3. **量化剪枝**：根据具体硬件平台选择最适合的模型压缩策略
4. **异步处理**：图像获取和模型推理异步进行，减少延迟

### 9.3 下一步工作

1. **多模态融合**：将眼动追踪与头部姿态和手势识别融合，提高分心检测鲁棒性
2. **个性化适应**：实现在线学习，适应不同驾驶员的眼动特征
3. **场景适应**：根据驾驶场景（城市、高速等）动态调整分心阈值
4. **临床验证**：在真实驾驶环境中验证系统性能和安全性