# 第四章：模型模块

本章介绍 PyTorch 模型的核心构建部分，包括 Module、容器、常用网络层、API 函数、Hook 机制和权重初始化。

## 📚 学习内容

### 核心概念

- **Module**：所有神经网络的基类，管理参数和子模块
- **Parameter**：可训练参数，默认开启梯度
- **容器**：Sequential、ModuleList、ModuleDict 等组织网络层的工具
- **网络层**：卷积、池化、标准化、Dropout、激活函数等
- **Hook**：在不修改模型代码的前提下提取中间信息

### 模型构建流程

```
继承 nn.Module
  ↓
__init__：初始化网络层
  ↓
forward：定义层的连接方式
  ↓
实例化模型 → 前向传播
  ↓
Hook（可选）：提取特征图/梯度
```

## 📝 Notebook 列表

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_module_parameter.ipynb` | Module 基类与 Parameter | ⭐ |
| `02_containers.ipynb` | Sequential、ModuleList、ModuleDict | ⭐⭐ |
| `03_common_layers.ipynb` | 卷积、池化、BN、Dropout、激活函数 | ⭐⭐ |
| `04_module_api.ipynb` | train/eval、state_dict、to(device) 等 | ⭐⭐ |
| `05_hook_gradcam.ipynb` | Hook 函数与 Grad-CAM 可视化 | ⭐⭐⭐ |
| `06_weight_init.ipynb` | Xavier、Kaiming 等权重初始化方法 | ⭐⭐ |

## 🎯 学习路线

### 新手入门

1. 从 `01_module_parameter.ipynb` 开始，理解 Module 和 Parameter
2. 学习 `02_containers.ipynb`，掌握如何组织网络层
3. 学习 `03_common_layers.ipynb`，了解各种网络层的用法
4. 学习 `04_module_api.ipynb`，掌握模型管理的常用方法

### 进阶学习

- `05_hook_gradcam.ipynb`：理解 Hook 机制，实现 Grad-CAM
- `06_weight_init.ipynb`：学习权重初始化对训练的影响

## 💡 核心知识点

### 1. 模型创建三步曲

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### 2. 容器对比

| 容器 | 自动 forward | 使用场景 |
|------|-------------|---------|
| `Sequential` | ✅ | 简单线性堆叠 |
| `ModuleList` | ❌ | 需要循环/条件控制 |
| `ModuleDict` | ❌ | 需要按名称动态选择 |

### 3. BN 和 Dropout 的训练/推理差异

```python
model.train()  # 训练模式：BN 用当前 batch 统计，Dropout 生效
model.eval()   # 评估模式：BN 用滑动平均，Dropout 关闭
```

### 4. 权重初始化选择

| 激活函数 | 推荐初始化 |
|---------|----------|
| ReLU | Kaiming |
| Sigmoid / Tanh | Xavier |

## 🔗 相关资源

- [PyTorch nn.Module 文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch nn 层文档](https://pytorch.org/docs/stable/nn.html)
- [Grad-CAM 论文](https://arxiv.org/abs/1610.02391)
