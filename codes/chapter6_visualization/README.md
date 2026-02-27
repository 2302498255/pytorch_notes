# 第六章：可视化模块

本章介绍可视化工具，帮助理解模型训练过程和内部机制。

## 📚 学习内容

### 核心概念

- **TensorBoard**：强大的可视化工具，记录标量、图像、分布等
- **卷积核/特征图可视化**：理解 CNN 学到了什么
- **混淆矩阵**：分析分类结果
- **CAM 可视化**：理解模型关注图像的哪个区域
- **模型参数统计**：分析模型复杂度

### 可视化全景

```
训练过程 → TensorBoard（Loss、Accuracy 曲线）
  ↓
模型内部 → 卷积核 / 特征图可视化
  ↓
分类结果 → 混淆矩阵
  ↓
模型决策 → Grad-CAM 热力图
  ↓
模型规模 → torchinfo 参数统计
```

## 📝 Notebook 列表

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_tensorboard.ipynb` | TensorBoard 安装与核心 API | ⭐⭐ |
| `02_cnn_visualization.ipynb` | 卷积核与特征图可视化 | ⭐⭐ |
| `03_confusion_matrix.ipynb` | 混淆矩阵与训练曲线绘制 | ⭐⭐ |
| `04_gradcam.ipynb` | Grad-CAM 热力图实现 | ⭐⭐⭐ |
| `05_model_summary.ipynb` | torchinfo 模型参数统计 | ⭐ |

## 🎯 学习路线

1. 从 `01_tensorboard.ipynb` 开始，掌握 TensorBoard 基本用法
2. 学习 `02_cnn_visualization.ipynb`，理解 CNN 内部工作原理
3. 学习 `03_confusion_matrix.ipynb`，掌握分类结果分析
4. 学习 `04_gradcam.ipynb`，掌握模型可解释性工具
5. 学习 `05_model_summary.ipynb`，快速查看模型信息

## 💡 核心知识点

### 1. TensorBoard 核心 API

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/exp1')

writer.add_scalar('Loss/train', loss, epoch)     # 标量
writer.add_scalars('Loss', {'train': t, 'val': v}, epoch)  # 多曲线
writer.add_histogram('weights', param, epoch)     # 分布
writer.add_image('sample', img_tensor, epoch)     # 图像
writer.add_graph(model, dummy_input)              # 模型结构
writer.close()
```

### 2. 特征图提取（Hook 法）

```python
features = {}
def hook_fn(name):
    def hook(m, i, o):
        features[name] = o.detach()
    return hook
model.conv1.register_forward_hook(hook_fn('conv1'))
```

### 3. 模型参数统计

```python
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))
```

## 🔗 相关资源

- [TensorBoard 官方文档](https://www.tensorflow.org/tensorboard)
- [torchinfo GitHub](https://github.com/TylerYep/torchinfo)
- [Grad-CAM 论文](https://arxiv.org/abs/1610.02391)
