# 第五章 优化模块 - 练习题

---

## 基础题 (Fundamental)

### 题目 1: 损失函数对比实验

**题目描述：**
给定相同的预测值和目标值，分别使用 `L1Loss`、`MSELoss`、`SmoothL1Loss` 计算损失，对比 `reduction='mean'`、`'sum'`、`'none'` 三种模式的输出。

**提示：**
- `'none'`：返回每个元素的损失
- `'mean'`：返回所有元素损失的平均值
- `'sum'`：返回所有元素损失的总和

**参考答案：**

```python
import torch
import torch.nn as nn

pred = torch.tensor([1.0, 2.5, 3.0, 4.5])
target = torch.tensor([1.5, 2.0, 3.5, 4.0])

for loss_cls in [nn.L1Loss, nn.MSELoss, nn.SmoothL1Loss]:
    print(f"\n=== {loss_cls.__name__} ===")
    for reduction in ['none', 'mean', 'sum']:
        loss_fn = loss_cls(reduction=reduction)
        loss = loss_fn(pred, target)
        print(f"  reduction='{reduction}': {loss}")
```

---

### 题目 2: CrossEntropyLoss 深入理解

**题目描述：**
手动实现 CrossEntropyLoss 的计算过程（LogSoftmax + NLLLoss），验证与 `nn.CrossEntropyLoss` 的结果一致。

**提示：**
- CrossEntropyLoss = LogSoftmax + NLLLoss
- 手动计算：softmax → log → 取目标类别的负值 → 求平均

**参考答案：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
logits = torch.randn(4, 5)  # 4 个样本，5 个类别
targets = torch.tensor([1, 0, 3, 2])

# 方式一：nn.CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()
loss_auto = loss_fn(logits, targets)

# 方式二：LogSoftmax + NLLLoss
log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()
log_probs = log_softmax(logits)
loss_manual_v1 = nll_loss(log_probs, targets)

# 方式三：完全手动计算
softmax_probs = F.softmax(logits, dim=1)
log_probs_manual = torch.log(softmax_probs)
# 取出每个样本目标类别的 log 概率
loss_per_sample = -log_probs_manual[range(4), targets]
loss_manual_v2 = loss_per_sample.mean()

print(f"nn.CrossEntropyLoss:     {loss_auto.item():.6f}")
print(f"LogSoftmax + NLLLoss:    {loss_manual_v1.item():.6f}")
print(f"完全手动计算:              {loss_manual_v2.item():.6f}")
print(f"三者一致: {torch.allclose(loss_auto, loss_manual_v1) and torch.allclose(loss_auto, loss_manual_v2)}")
```

---

### 题目 3: 创建和使用优化器

**题目描述：**
创建一个简单的线性模型，分别使用 SGD、Adam、AdamW 优化器训练 100 步，对比收敛速度。

**提示：**
- 固定随机种子保证数据一致
- 记录每步的 loss 值
- 观察不同优化器的收敛曲线

**参考答案：**

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(100, 10)
y = X @ torch.randn(10, 1) + 0.5

results = {}

for opt_name, opt_cls, kwargs in [
    ('SGD', torch.optim.SGD, {'lr': 0.01}),
    ('SGD+momentum', torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
    ('Adam', torch.optim.Adam, {'lr': 0.01}),
    ('AdamW', torch.optim.AdamW, {'lr': 0.01, 'weight_decay': 0.01}),
]:
    torch.manual_seed(0)
    model = nn.Linear(10, 1)
    optimizer = opt_cls(model.parameters(), **kwargs)
    criterion = nn.MSELoss()

    losses = []
    for step in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    results[opt_name] = losses
    print(f"{opt_name:15s}: 初始 loss={losses[0]:.4f}, 最终 loss={losses[-1]:.4f}")
```

---

### 题目 4: 参数组与差异学习率

**题目描述：**
创建一个包含特征提取器和分类器的模型，为两部分设置不同的学习率。验证两组参数的学习率确实不同。

**提示：**
- 优化器接受参数组列表
- 每个参数组可以有独立的 lr、weight_decay 等

**参考答案：**

```python
import torch
import torch.nn as nn

class TwoPartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(16, 5)

    def forward(self, x):
        return self.classifier(self.features(x))

model = TwoPartModel()

# 差异学习率
optimizer = torch.optim.SGD([
    {'params': model.features.parameters(), 'lr': 0.001},    # 特征层：小学习率
    {'params': model.classifier.parameters(), 'lr': 0.01},   # 分类器：大学习率
], momentum=0.9)

# 验证参数组
print("参数组信息：")
for i, group in enumerate(optimizer.param_groups):
    n_params = sum(p.numel() for p in group['params'])
    print(f"  组 {i}: lr={group['lr']}, momentum={group['momentum']}, params={n_params}")

# 模拟训练
criterion = nn.CrossEntropyLoss()
x = torch.randn(16, 10)
y = torch.randint(0, 5, (16,))

for step in range(5):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    print(f"Step {step}: loss={loss.item():.4f}")
```

---

## 进阶题 (Intermediate)

### 题目 5: 学习率调整器可视化

**题目描述：**
分别使用 `StepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`、`OneCycleLR` 四种调度器，记录 100 个 epoch 的学习率变化并对比。

**提示：**
- 创建一个虚拟的优化器用于测试
- 每个 epoch 调用 `scheduler.step()`
- 记录 `optimizer.param_groups[0]['lr']`

**参考答案：**

```python
import torch
import torch.nn as nn

def get_lr_curve(scheduler_cls, optimizer, epochs=100, **kwargs):
    """获取学习率变化曲线。"""
    lrs = []
    scheduler = scheduler_cls(optimizer, **kwargs)
    for epoch in range(epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # 模拟 loss 先下降后停滞
            fake_loss = max(0.1, 1.0 - epoch * 0.02) if epoch < 50 else 0.1
            scheduler.step(fake_loss)
        else:
            scheduler.step()
    return lrs

epochs = 100
configs = {}

# StepLR
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
configs['StepLR'] = get_lr_curve(
    torch.optim.lr_scheduler.StepLR, opt, epochs, step_size=30, gamma=0.1
)

# CosineAnnealingLR
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
configs['CosineAnnealing'] = get_lr_curve(
    torch.optim.lr_scheduler.CosineAnnealingLR, opt, epochs, T_max=epochs, eta_min=1e-5
)

# ReduceLROnPlateau
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
configs['ReduceOnPlateau'] = get_lr_curve(
    torch.optim.lr_scheduler.ReduceLROnPlateau, opt, epochs, patience=10, factor=0.5
)

# OneCycleLR
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.001)
configs['OneCycleLR'] = get_lr_curve(
    torch.optim.lr_scheduler.OneCycleLR, opt, epochs, max_lr=0.1, total_steps=epochs
)

# 打印对比
print(f"{'Epoch':>6s}", end="")
for name in configs:
    print(f"  {name:>16s}", end="")
print()

for epoch in [0, 10, 25, 50, 75, 99]:
    print(f"{epoch:>6d}", end="")
    for name, lrs in configs.items():
        print(f"  {lrs[epoch]:>16.6f}", end="")
    print()
```

---

### 题目 6: 带 Warmup 的学习率调度器

**题目描述：**
实现一个自定义的 Warmup + CosineAnnealing 学习率调度器。前 `warmup_epochs` 个 epoch 线性增加学习率，之后使用余弦退火。

**提示：**
- 使用 `LambdaLR` 自定义调度逻辑
- warmup 阶段：`lr = base_lr * epoch / warmup_epochs`
- cosine 阶段：标准余弦退火

**参考答案：**

```python
import torch
import torch.nn as nn
import math

def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """创建 Warmup + Cosine 调度器。"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 测试
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = create_warmup_cosine_scheduler(optimizer, warmup_epochs=10, total_epochs=100)

print("Warmup + Cosine 学习率曲线：")
for epoch in range(100):
    lr = optimizer.param_groups[0]['lr']
    if epoch % 10 == 0 or epoch < 12:
        print(f"  Epoch {epoch:>3d}: lr = {lr:.6f}")
    scheduler.step()
```

---

### 题目 7: 自定义损失函数

**题目描述：**
实现以下自定义损失函数：
1. `FocalLoss`：用于处理类别不平衡的分类任务
2. `LabelSmoothingLoss`：带标签平滑的交叉熵

**提示：**
- Focal Loss: $FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$
- Label Smoothing: 将硬标签变为软标签

**参考答案：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss 用于处理类别不平衡。"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 各类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测正确的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingLoss(nn.Module):
    """标签平滑交叉熵损失。"""

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)

        # 创建平滑标签
        smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = (-smooth_targets * log_probs).sum(dim=1).mean()
        return loss

# 测试
logits = torch.randn(8, 5)
targets = torch.randint(0, 5, (8,))

ce_loss = nn.CrossEntropyLoss()(logits, targets)
focal_loss = FocalLoss(gamma=2.0)(logits, targets)
smooth_loss = LabelSmoothingLoss(num_classes=5, smoothing=0.1)(logits, targets)

print(f"CrossEntropy Loss:     {ce_loss.item():.4f}")
print(f"Focal Loss (gamma=2):  {focal_loss.item():.4f}")
print(f"Label Smoothing Loss:  {smooth_loss.item():.4f}")
```

---

## 挑战题 (Challenge)

### 题目 8: 完整的训练优化流程

**题目描述：**
在 MNIST 数据集上实现一个完整的优化流程：模型定义 → 损失函数选择 → 优化器配置 → 学习率调度 → 训练循环 → 验证。对比不同优化器和学习率策略的效果。

**提示：**
- 使用简单的 MLP 模型
- 对比 SGD vs Adam
- 对比 StepLR vs CosineAnnealing
- 记录训练和验证指标

**参考答案：**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 模拟数据（替代 MNIST）
torch.manual_seed(42)
X_train = torch.randn(2000, 784)
y_train = torch.randint(0, 10, (2000,))
X_val = torch.randn(500, 784)
y_val = torch.randint(0, 10, (500,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(opt_name, scheduler_name, epochs=20):
    """训练并评估。"""
    torch.manual_seed(0)
    model = MLP()
    criterion = nn.CrossEntropyLoss()

    # 选择优化器
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 选择调度器
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc

# 对比实验
print(f"{'Optimizer':>10s} + {'Scheduler':>15s} = {'Best Val Acc':>12s}")
print("-" * 45)
for opt in ['SGD', 'Adam']:
    for sched in ['StepLR', 'CosineAnnealing']:
        acc = train_and_evaluate(opt, sched)
        print(f"{opt:>10s} + {sched:>15s} = {acc:>11.2%}")
```

---

## 总结与学习建议

### 关键概念回顾

1. **损失函数**：分类用 CrossEntropyLoss，回归用 MSELoss/SmoothL1Loss
2. **优化器**：SGD 稳健可靠，Adam/AdamW 收敛快
3. **学习率调度**：CosineAnnealing 最常用，ReduceLROnPlateau 最省心
4. **参数组**：不同层可以设置不同的学习率

### 常见陷阱

- CrossEntropyLoss 的输入是 logits，不需要先做 Softmax
- ReduceLROnPlateau 的 `step()` 需要传入监控指标
- 学习率调度器的 `step()` 应放在 epoch 循环末尾
- 梯度累积时不要每步都 `zero_grad()`
