# RSNA数据集评估报告 - 病灶攻击 vs 全图攻击

生成时间: 2026-02-23

## 1. 数据集概况

### 基本信息
- **数据集名称**: RSNA Pneumonia Detection Challenge
- **总图像数**: 26,684张训练图像（DICOM格式）
- **总标注数**: 30,227条标注记录
- **数据位置**: `D:\PycharmProjects\HONER_PROJECT\dataset\rsna\`

### 数据分布

#### 按Target分类（二分类）
- **正常样本 (Target=0)**: 20,672 (68.4%)
- **肺炎样本 (Target=1)**: 9,555 (31.6%)

#### 按详细类别分类（三分类）
- **No Lung Opacity / Not Normal**: 11,821 (39.1%)
- **Lung Opacity (肺炎)**: 9,555 (31.6%)
- **Normal**: 8,851 (29.3%)

#### 病灶分布
- **有病灶边界框的样本**: 9,555条
- **有病灶的患者数**: 6,012位
- **病灶数量分布**:
  - 1个病灶: 2,614位患者 (43.5%)
  - 2个病灶: 3,266位患者 (54.3%)
  - 3个病灶: 119位患者 (2.0%)
  - 4个病灶: 13位患者 (0.2%)

### 数据文件结构
```
dataset/rsna/
├── stage_2_train_images/        # 26,684张DICOM图像
├── stage_2_test_images/         # 测试图像
├── stage_2_train_labels.csv     # 包含边界框坐标 (x, y, width, height)
├── stage_2_detailed_class_info.csv  # 详细分类标签
└── stage_2_sample_submission.csv    # 提交样例
```

## 2. 适用性评估 - 病灶攻击 vs 全图攻击

### ✅ 数据集优势

1. **明确的病灶定位**
   - 提供准确的边界框标注 (x, y, width, height)
   - 可以精确定位攻击区域
   - 支持针对性扰动生成

2. **合适的任务设置**
   - 二分类任务（正常/肺炎）或三分类任务
   - 临床意义明确
   - 可评估攻击对诊断系统的影响

3. **数据量充足**
   - 6,012位患者有病灶标注
   - 每位患者平均1.6个病灶
   - 足够进行统计显著性检验

4. **实验对比设计**

   **方案A：针对病灶攻击**
   - 仅对边界框内的病灶区域施加对抗扰动
   - 保持背景和其他区域不变
   - 评估局部扰动的攻击效果

   **方案B：全图攻击**
   - 对整张X光图像施加对抗扰动
   - 传统的对抗攻击方式
   - 作为baseline对比

   **对比维度**:
   - 攻击成功率
   - 扰动大小（L0, L2, L∞范数）
   - 视觉可察觉性
   - 攻击计算开销

### ⚠️ 需要注意的问题

1. **图像格式处理**
   - DICOM格式需要特殊处理（使用pydicom库）
   - 需要转换为模型可接受的格式

2. **边界框处理**
   - 部分患者有多个病灶（最多4个）
   - 需要决定：攻击单个病灶 or 所有病灶

3. **评估基准模型**
   - 需要选择或训练一个肺炎检测模型
   - CheXzero可能不适用（你已提到）
   - 建议：ResNet/DenseNet-based分类器

## 3. CUDA环境评估

### 当前配置 ❌

```
GPU: NVIDIA GeForce RTX 3060 (6GB VRAM)
Driver Version: 560.70
CUDA Version: 12.6 (驱动支持)

PyTorch配置:
- Version: 2.6.0+cpu (⚠️ CPU版本)
- CUDA available: False
- cuDNN: None
```

### ⚠️ 严重问题：PyTorch为CPU版本

当前安装的PyTorch是CPU版本，无法使用GPU加速。这会严重影响DeepFool和CW攻击的运行速度。

### 🔧 解决方案：重新安装CUDA版PyTorch

需要卸载当前版本并安装GPU版本：

```bash
# 卸载当前CPU版本
pip uninstall torch torchvision

# 安装CUDA 12.4版本（推荐，与你的驱动兼容）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 或者安装CUDA 11.8版本（更稳定）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 验证安装

```python
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0)}')
```

预期输出：
```
CUDA available: True
CUDA version: 12.4 (or 11.8)
Device: NVIDIA GeForce RTX 3060
```

## 4. 攻击实现准备

### 已有资源

1. **DeepFool**: `DeepFool-master/DeepFool-master/Python/`
2. **C&W Attack**: `Carlini–Wagner attacks/nn_robust_attacks-master/`
   - l0_attack.py
   - l2_attack.py
   - li_attack.py

### 需要开发的模块

1. **DICOM图像加载器**
   ```python
   import pydicom
   from PIL import Image
   ```

2. **病灶区域提取器**
   - 根据边界框坐标提取病灶
   - 支持多病灶处理

3. **局部攻击实现**
   - 修改现有攻击算法，仅对指定区域施加扰动
   - 保持其他区域不变

4. **评估指标**
   - 攻击成功率
   - 扰动范数 (L0, L2, L∞)
   - SSIM/PSNR (结构相似度/峰值信噪比)

## 5. 推荐实验流程

### Phase 1: 环境配置
1. ✅ 重新安装CUDA版PyTorch
2. 安装依赖：`pip install pydicom pillow pandas scikit-learn`

### Phase 2: 数据准备
1. 编写DICOM加载器
2. 数据划分（训练/验证/测试）
3. 病灶区域统计分析

### Phase 3: 模型训练
1. 训练肺炎分类模型（ResNet/DenseNet）
2. 在RSNA数据集上验证性能

### Phase 4: 攻击实验
1. 实现全图攻击（baseline）
2. 实现病灶攻击
3. 对比分析

### Phase 5: 评估与分析
1. 量化指标对比
2. 视觉分析
3. 撰写论文

## 6. 结论

### ✅ 数据集完全适用

RSNA数据集**非常适合**你的实验需求：
- ✅ 有明确的病灶边界框标注
- ✅ 数据量充足（6K+有病灶的样本）
- ✅ 任务明确（肺炎检测）
- ✅ 支持病灶攻击 vs 全图攻击对比

### ⚠️ 当前最紧急问题

**必须先解决CUDA问题**，否则无法高效运行DeepFool和CW攻击。

建议立即执行：
```bash
pip uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 预估计算资源

- **GPU**: RTX 3060 (6GB) 足够
- **内存**: 建议16GB+
- **存储**: 数据集约10GB，实验结果约20GB

---

如有任何问题，请随时提出。
