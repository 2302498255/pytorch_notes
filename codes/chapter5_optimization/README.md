# 第五章：优化模块

本章介绍模型优化过程中的三大核心概念：损失函数、优化器和学习率调整。

## 📚 学习内容

### 核心概念

- **损失函数**：衡量预测值与真实值的差距，是优化的目标
- **优化器**：根据梯度更新模型参数
- **学习率调整器**：在训练过程中动态调整学习率

### 优化流程

```
前向传播 → 计算损失（Loss Function）
  ↓
反向传播 → 计算梯度
  ↓
优化器 → 更新参数（Optimizer.step）
  ↓
调整学习率（LR Scheduler.step）
```

## 📝 Notebook 列表

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_loss_functions.ipynb` | 常用损失函数详解与对比 | ⭐⭐ |
| `02_optimizers.ipynb` | SGD、Adam、AdamW 等优化器 | ⭐⭐ |
| `03_lr_schedulers.ipynb` | 学习率调整策略与可视化 | ⭐⭐ |
| `04_optimization_practice.ipynb` | 完整优化流程实战 | ⭐⭐⭐ |

## 🎯 学习路线

1. 从 `01_loss_functions.ipynb` 开始，理解各种损失函数的适用场景
2. 学习 `02_optimizers.ipynb`，掌握 SGD 和 Adam 系列优化器
3. 学习 `03_lr_schedulers.ipynb`，了解学习率调整策略
4. 最后通过 `04_optimization_practice.ipynb` 整合所有知识

## 💡 核心知识点

### 1. 损失函数选择

| 任务 | 推荐损失函数 |
|------|------------|
| 多分类 | `CrossEntropyLoss` |
| 二分类 | `BCEWithLogitsLoss` |
| 回归 | `MSELoss` / `SmoothL1Loss` |
| 度量学习 | `TripletMarginLoss` |

### 2. 优化器选择

```python
# SGD：大多数论文使用，配合 momentum
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)

# AdamW：快速实验的首选
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
```

### 3. 学习率调整

```python
# 余弦退火（最常用）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 指标停滞时衰减
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
```

## 🔗 相关资源

- [PyTorch Loss Functions 文档](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [PyTorch Optimizer 文档](https://pytorch.org/docs/stable/optim.html)
