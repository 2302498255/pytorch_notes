# 第七章：小技巧汇总

本章介绍 PyTorch 开发中常用的实用技巧和工具库。

## 📚 学习内容

### 核心概念

- **模型保存与加载**：序列化、断点续训
- **Finetune 微调**：迁移学习、冻结层、差异学习率
- **GPU 使用**：单 GPU、多 GPU、设备管理
- **训练代码模板**：标准化的训练流程
- **实用工具库**：TorchMetrics、Albumentations、TorchEnsemble

### 实战技能树

```
模型持久化 → 保存 / 加载 / 断点续训
  ↓
迁移学习  → Finetune / 冻结层 / 差异学习率
  ↓
GPU 加速  → 单 GPU / DataParallel / 设备管理
  ↓
工程化    → 训练模板 / 日志 / 指标评估
  ↓
工具库    → TorchMetrics / Albumentations / TorchEnsemble
```

## 📝 Notebook 列表

| 文件 | 内容 | 难度 |
|------|------|------|
| `01_save_load.ipynb` | 模型保存、加载与断点续训 | ⭐⭐ |
| `02_finetune.ipynb` | 迁移学习与模型微调 | ⭐⭐ |
| `03_gpu_usage.ipynb` | GPU 使用与多 GPU 训练 | ⭐⭐ |
| `04_training_template.ipynb` | 完整训练代码模板 | ⭐⭐⭐ |
| `05_useful_libraries.ipynb` | TorchMetrics / Albumentations / TorchEnsemble | ⭐⭐ |

## 🎯 学习路线

### 必学内容

1. `01_save_load.ipynb`：模型保存与加载是基本技能
2. `02_finetune.ipynb`：迁移学习是快速出结果的利器
3. `03_gpu_usage.ipynb`：GPU 加速是深度学习的基础
4. `04_training_template.ipynb`：掌握标准训练流程

### 进阶拓展

- `05_useful_libraries.ipynb`：了解实用工具库

## 💡 核心知识点

### 1. 模型保存（推荐方式）

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
```

### 2. 断点续训

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch,
}
torch.save(checkpoint, 'checkpoint.pth')
```

### 3. 微调两种方式

```python
# 方式一：冻结特征层
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, num_classes)  # 只训练新分类器

# 方式二：差异学习率
optimizer = SGD([
    {'params': model.features.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-2},
])
```

### 4. GPU 使用

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data = data.to(device)
```

## 🔗 相关资源

- [PyTorch 保存加载文档](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [TorchMetrics 文档](https://torchmetrics.readthedocs.io/)
- [Albumentations 文档](https://albumentations.ai/)
