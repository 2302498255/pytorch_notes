# 第三章：数据模块

本章介绍 PyTorch 的数据处理模块，包括 Dataset、DataLoader、Sampler 和 Transforms。

## 📚 学习内容

### 核心概念

- **Dataset**：数据集的抽象类，定义如何读取单个样本
- **DataLoader**：数据加载器，负责批量加载、打乱、并行处理
- **Sampler**：采样器，控制数据的采样策略
- **Transform**：数据变换，用于预处理和数据增强

### 数据加载流程

```
硬盘数据
  ↓
Dataset (定义如何读取)
  ↓
Transform (预处理/增强)
  ↓
Sampler (采样策略，可选)
  ↓
DataLoader (批量加载)
  ↓
模型训练
```

## 📝 Notebook 列表

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_dataset_basics.ipynb` | Dataset 基础概念和使用 | ⭐ |
| `02_custom_dataset.ipynb` | 自定义 Dataset 的三种形式 | ⭐⭐ |
| `03_dataloader.ipynb` | DataLoader 的使用和参数 | ⭐⭐ |
| `04_sampler.ipynb` | Sampler 采样器和不平衡数据处理 | ⭐⭐⭐ |
| `05_transforms.ipynb` | 数据预处理和数据增强 | ⭐⭐ |
| `06_complete_example.ipynb` | 完整的端到端示例 | ⭐⭐⭐ |

## 🎯 学习路线

### 新手入门

1. 从 `01_dataset_basics.ipynb` 开始，理解 Dataset 的基本概念
2. 学习 `02_custom_dataset.ipynb`，掌握如何创建自定义 Dataset
3. 学习 `03_dataloader.ipynb`，了解如何批量加载数据
4. 学习 `05_transforms.ipynb`，掌握数据预处理
5. 最后看 `06_complete_example.ipynb`，整合所有知识

### 进阶学习

- 如果遇到类别不平衡问题，学习 `04_sampler.ipynb`
- 如果需要优化数据加载速度，重点学习 DataLoader 的 `num_workers` 参数
- 如果需要自定义数据增强，学习如何创建自定义 Transform

## 💡 核心知识点

### 1. Dataset 的三个核心方法

```python
class MyDataset(Dataset):
    def __init__(self):
        # 初始化：只读取元信息，不加载数据
        pass

    def __getitem__(self, index):
        # 根据索引返回单个样本
        return data, label

    def __len__(self):
        # 返回数据集大小
        return len(self.data)
```

### 2. DataLoader 的关键参数

| 参数 | 作用 | 训练集 | 验证集 |
|------|------|--------|--------|
| `batch_size` | 批量大小 | 32-128 | 32-128 |
| `shuffle` | 是否打乱 | True | False |
| `num_workers` | 工作进程数 | 4-8 | 4-8 |
| `pin_memory` | 固定内存 | True (GPU) | True (GPU) |
| `drop_last` | 丢弃最后不完整batch | False | False |

### 3. Transform 使用原则

**训练集**：使用数据增强
```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**验证集**：只做基本预处理
```python
valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 4. 处理不平衡数据

使用 `WeightedRandomSampler`：

```python
# 计算类别权重
class_weights = 1.0 / torch.tensor(class_counts)

# 为每个样本分配权重
sample_weights = [class_weights[label] for label in labels]

# 创建 sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# 使用 sampler
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

## 🔧 常见问题

### Q1: 数据加载太慢怎么办？

**解决方案**：
- 增加 `num_workers`（Linux/Mac）
- 使用 `pin_memory=True`（GPU 训练）
- 减小图像分辨率
- 使用更快的存储设备（SSD）

### Q2: 显存不足 (OOM) 怎么办？

**解决方案**：
- 减小 `batch_size`
- 减小图像分辨率
- 减小模型大小
- 使用梯度累积

### Q3: 为什么训练集要 shuffle，验证集不需要？

**原因**：
- 训练时打乱数据可以提高模型泛化能力
- 验证时固定顺序便于对比不同模型的性能

### Q4: shuffle 和 sampler 能同时使用吗？

**答案**：不能！它们是互斥的。

```python
# ❌ 错误
loader = DataLoader(dataset, shuffle=True, sampler=my_sampler)

# ✅ 正确：二选一
loader = DataLoader(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=my_sampler)
```

### Q5: num_workers 应该设置为多少？

**建议**：
- Windows：设为 0（Windows 多进程可能有问题）
- Linux/Mac：设为 4-8
- 根据 CPU 核心数：`min(8, os.cpu_count())`

## 📊 数据集组织形式对比

| 形式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| TXT 文件 | 简单直观 | 需要手动维护 | 小型数据集 |
| 文件夹结构 | 直观易管理 | 不适合多标签 | 单标签分类 |
| CSV 文件 | 灵活，支持元信息 | 需要 pandas | 复杂数据集 |

## 🚀 快速开始

### 最简单的例子

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. 定义 Dataset
class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

# 2. 创建数据集
data = [(torch.randn(3, 64, 64), 0) for _ in range(100)]
dataset = MyDataset(data)

# 3. 创建 DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. 使用
for images, labels in loader:
    print(f"Batch: {images.shape}, {labels.shape}")
    break
```

## 📖 推荐学习顺序

1. **第一天**：学习 `01_dataset_basics.ipynb` 和 `02_custom_dataset.ipynb`
2. **第二天**：学习 `03_dataloader.ipynb` 和 `05_transforms.ipynb`
3. **第三天**：学习 `04_sampler.ipynb`（可选，处理不平衡数据时必学）
4. **第四天**：学习 `06_complete_example.ipynb`，整合所有知识

## 🎓 学习目标

完成本章学习后，你应该能够：

- ✅ 理解 PyTorch 数据加载的整体流程
- ✅ 创建自定义 Dataset 类
- ✅ 使用 DataLoader 批量加载数据
- ✅ 使用 Transform 进行数据预处理和增强
- ✅ 处理类别不平衡的数据集
- ✅ 优化数据加载速度
- ✅ 调试常见的数据加载问题

## 🔗 相关资源

- [PyTorch 官方文档 - Data Loading](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [torchvision.transforms 文档](https://pytorch.org/vision/stable/transforms.html)
- [PyTorch 数据加载最佳实践](https://pytorch.org/docs/stable/data.html)

## 💻 实践项目

完成学习后，可以尝试以下实践项目：

1. **项目 1**：为 CIFAR-10 数据集创建自定义 DataLoader
2. **项目 2**：实现一个处理不平衡数据的完整流程
3. **项目 3**：对比不同数据增强策略对模型性能的影响
4. **项目 4**：优化数据加载速度，测试不同 `num_workers` 的影响

## 📝 小结

数据模块是深度学习训练流程的基础。掌握本章内容后，你将能够：

- 高效地加载和处理数据
- 处理各种复杂的数据场景
- 优化训练流程的数据瓶颈
- 为模型训练提供高质量的数据

祝学习愉快！🎉
