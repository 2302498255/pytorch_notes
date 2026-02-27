# 第六章 可视化模块 - 练习题

---

## 基础题 (Fundamental)

### 题目 1: TensorBoard 基本使用

**题目描述：**
使用 `SummaryWriter` 记录一个模拟训练过程的标量数据（train_loss、val_loss、learning_rate），并保存到 TensorBoard 日志。

**提示：**
- 使用 `add_scalar` 记录单个标量
- 使用 `add_scalars` 在同一图表中记录多条曲线
- 模拟 50 个 epoch 的训练过程

**参考答案：**

```python
import torch
import math
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/tmp/tb_demo/experiment_1')

for epoch in range(50):
    # 模拟训练指标
    train_loss = 1.0 * math.exp(-0.05 * epoch) + 0.1 * torch.randn(1).item()
    val_loss = 1.2 * math.exp(-0.04 * epoch) + 0.15 * torch.randn(1).item()
    lr = 0.1 * (0.95 ** epoch)

    # 记录标量
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('LR', lr, epoch)

    # 同一图表中绘制多条曲线
    writer.add_scalars('Loss_comparison', {
        'train': train_loss,
        'val': val_loss,
    }, epoch)

writer.close()
print("TensorBoard 日志已保存到 /tmp/tb_demo/experiment_1")
print("运行 'tensorboard --logdir=/tmp/tb_demo' 查看")
```

---

### 题目 2: 记录参数分布

**题目描述：**
训练一个简单模型 5 个 epoch，使用 `add_histogram` 记录每个 epoch 后各层权重和梯度的分布变化。

**提示：**
- 遍历 `model.named_parameters()` 获取参数
- 在每个 epoch 结束后记录
- 同时记录权重和梯度

**参考答案：**

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 5),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter('/tmp/tb_demo/histograms')

for epoch in range(5):
    # 模拟训练
    x = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))

    for step in range(20):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    # 记录参数分布
    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param.data, epoch)
        if param.grad is not None:
            writer.add_histogram(f'grads/{name}', param.grad, epoch)

    print(f"Epoch {epoch}: loss={loss.item():.4f}")

writer.close()
print("直方图日志已保存")
```

---

### 题目 3: 混淆矩阵计算

**题目描述：**
给定模型的预测结果和真实标签，计算并打印混淆矩阵。计算每个类别的精确率和召回率。

**提示：**
- 混淆矩阵：行=真实类别，列=预测类别
- 精确率 = TP / (TP + FP)
- 召回率 = TP / (TP + FN)

**参考答案：**

```python
import torch
import numpy as np

def compute_confusion_matrix(preds, targets, num_classes):
    """计算混淆矩阵。"""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm

def compute_metrics(cm):
    """从混淆矩阵计算精确率和召回率。"""
    num_classes = cm.shape[0]
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i].float()
        fp = cm[:, i].sum().float() - tp
        fn = cm[i, :].sum().float() - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

# 模拟预测
torch.manual_seed(42)
targets = torch.randint(0, 4, (100,))
preds = targets.clone()
# 引入一些错误
noise_idx = torch.randperm(100)[:20]
preds[noise_idx] = torch.randint(0, 4, (20,))

cm = compute_confusion_matrix(preds, targets, num_classes=4)
precision, recall = compute_metrics(cm)

print("混淆矩阵：")
print(cm)
print(f"\n{'类别':>4s} {'精确率':>8s} {'召回率':>8s}")
for i in range(4):
    print(f"  {i:>2d}  {precision[i]:>8.2%}  {recall[i]:>8.2%}")

accuracy = cm.diag().sum().float() / cm.sum().float()
print(f"\n总体准确率: {accuracy:.2%}")
```

---

## 进阶题 (Intermediate)

### 题目 4: CNN 卷积核可视化

**题目描述：**
加载预训练的 AlexNet（或任意 CNN 模型），提取第一层卷积核，使用 `make_grid` 拼接成网格图。

**提示：**
- 第一层卷积核形状为 `(out_ch, 3, H, W)`，可直接作为 RGB 图像
- 使用 `torchvision.utils.make_grid` 拼接
- `normalize=True` 映射到 [0, 1]

**参考答案：**

```python
import torch
import torch.nn as nn
from torchvision.utils import make_grid

# 创建一个简单的 CNN（替代预训练模型）
model = nn.Sequential(
    nn.Conv2d(3, 16, 5),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3),
)

# 提取第一层卷积核
first_conv = None
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        first_conv = module
        break

kernels = first_conv.weight.data  # (16, 3, 5, 5)
print(f"卷积核形状: {kernels.shape}")
print(f"  out_channels={kernels.shape[0]}")
print(f"  in_channels={kernels.shape[1]}")
print(f"  kernel_size=({kernels.shape[2]}, {kernels.shape[3]})")

# 拼接为网格图
grid = make_grid(kernels, nrow=8, normalize=True, padding=1)
print(f"\n网格图形状: {grid.shape}")
print(f"值范围: [{grid.min():.4f}, {grid.max():.4f}]")

# 如果要保存或显示：
# import matplotlib.pyplot as plt
# plt.imshow(grid.permute(1, 2, 0))
# plt.savefig('kernels.png')
```

---

### 题目 5: 使用 Hook 提取特征图

**题目描述：**
对一个 CNN 模型注册 forward hook，在前向传播时自动捕获所有卷积层的特征图。打印各层特征图的形状和统计信息。

**提示：**
- 使用闭包（closure）为每层创建独立的 hook 函数
- 在 hook 中使用 `.detach()` 避免影响计算图
- 使用完后移除 hook

**参考答案：**

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 32 → 16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 16 → 8
        x = self.pool(self.relu(self.conv3(x)))             # 8 → 4
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
model.eval()

# 存储特征图
feature_maps = {}
hooks = []

def make_hook(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach()
    return hook

# 注册 hook
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

# 前向传播
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    output = model(x)

# 分析特征图
print(f"{'层名':>8s} {'形状':>20s} {'均值':>8s} {'标准差':>8s} {'最大值':>8s}")
print("-" * 60)
for name, fmap in feature_maps.items():
    print(f"{name:>8s} {str(fmap.shape):>20s} "
          f"{fmap.mean():>8.4f} {fmap.std():>8.4f} {fmap.max():>8.4f}")

# 移除 hook
for h in hooks:
    h.remove()
```

---

### 题目 6: 使用 torchinfo 分析模型

**题目描述：**
使用 `torchinfo.summary` 分析 ResNet-18 和一个自定义模型的参数量、计算量、内存占用。

**提示：**
- 安装 torchinfo：`pip install torchinfo`
- 指定 `col_names` 控制显示的列
- 通过 `depth` 控制显示层级深度

**参考答案：**

```python
import torch
import torch.nn as nn

# 方法一：使用 torchinfo（需要安装）
try:
    from torchinfo import summary

    # 自定义模型
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )

    print("=== 模型摘要 ===")
    summary(model, input_size=(1, 3, 32, 32),
            col_names=["input_size", "output_size", "num_params"],
            depth=2)

except ImportError:
    print("torchinfo 未安装，使用手动统计")

# 方法二：手动统计（总是可用）
print("\n=== 手动统计 ===")
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total:,}")
print(f"可训练参数量: {trainable:,}")
print(f"参数存储（float32）: {total * 4 / 1024 / 1024:.2f} MB")

# 逐层统计
print(f"\n{'层':>30s} {'参数量':>12s}")
print("-" * 45)
for name, param in model.named_parameters():
    print(f"{name:>30s} {param.numel():>12,}")
```

---

## 挑战题 (Challenge)

### 题目 7: 完整的训练可视化系统

**题目描述：**
实现一个 `TrainingVisualizer` 类，集成 TensorBoard 记录、混淆矩阵生成、特征图提取和模型参数统计。在训练过程中自动记录所有信息。

**提示：**
- 封装 SummaryWriter 的常用操作
- 在每个 epoch 结束后记录 loss、accuracy、lr
- 定期记录混淆矩阵和参数分布
- 提供清晰的 API 接口

**参考答案：**

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TrainingVisualizer:
    """训练可视化管理器。"""

    def __init__(self, log_dir, model):
        self.writer = SummaryWriter(log_dir)
        self.model = model
        self.epoch = 0

    def log_scalar(self, tag, value, step=None):
        step = step if step is not None else self.epoch
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, value_dict, step=None):
        step = step if step is not None else self.epoch
        self.writer.add_scalars(tag, value_dict, step)

    def log_epoch(self, train_loss, val_loss, train_acc, val_acc, lr):
        """记录一个 epoch 的指标。"""
        self.log_scalars('Loss', {'train': train_loss, 'val': val_loss})
        self.log_scalars('Accuracy', {'train': train_acc, 'val': val_acc})
        self.log_scalar('LR', lr)
        self.epoch += 1

    def log_parameters(self):
        """记录模型参数分布。"""
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param.data, self.epoch)
            if param.grad is not None:
                self.writer.add_histogram(f'grads/{name}', param.grad, self.epoch)

    def log_confusion_matrix(self, preds, targets, class_names=None):
        """记录混淆矩阵。"""
        num_classes = max(targets.max(), preds.max()) + 1
        cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t, p in zip(targets, preds):
            cm[t, p] += 1

        # 转为文本形式记录
        cm_str = str(cm.numpy())
        self.writer.add_text('ConfusionMatrix', f'```\n{cm_str}\n```', self.epoch)
        return cm

    def log_model_graph(self, input_size):
        """记录模型结构图。"""
        dummy = torch.randn(*input_size)
        self.writer.add_graph(self.model, dummy)

    def close(self):
        self.writer.close()

# 使用示例
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
viz = TrainingVisualizer('/tmp/tb_demo/full_viz', model)

# 记录模型结构
viz.log_model_graph(input_size=(1, 1, 28, 28))

# 模拟训练
for epoch in range(10):
    train_loss = 1.0 * (0.9 ** epoch)
    val_loss = 1.2 * (0.88 ** epoch)
    train_acc = 0.5 + 0.05 * epoch
    val_acc = 0.45 + 0.04 * epoch
    lr = 0.01 * (0.95 ** epoch)

    viz.log_epoch(train_loss, val_loss, train_acc, val_acc, lr)

    if epoch % 3 == 0:
        viz.log_parameters()

viz.close()
print("可视化日志已保存")
```

---

## 总结与学习建议

### 关键概念回顾

1. **TensorBoard**：add_scalar/add_scalars/add_histogram/add_image/add_graph
2. **卷积核可视化**：第一层可作为 RGB 图像，深层逐通道显示
3. **特征图提取**：使用 forward hook 自动捕获
4. **混淆矩阵**：行=真实，列=预测，对角线=正确
5. **模型统计**：torchinfo 查看参数量和计算量

### 常见陷阱

- TensorBoard 日志目录冲突导致曲线混乱
- 忘记调用 `writer.close()` 导致数据未写入
- Hook 使用后忘记 remove 导致内存泄漏
- 混淆矩阵的行列顺序搞反
