# 第七章 小技巧汇总 - 练习题

---

## 基础题 (Fundamental)

### 题目 1: 模型保存与加载

**题目描述：**
创建一个模型，训练几步后保存 `state_dict`，再加载到一个新的模型实例中。验证两个模型的输出一致。

**提示：**
- 使用 `torch.save` 和 `torch.load`
- 保存和加载 `state_dict`（推荐方式）
- 验证加载后的模型输出与原模型一致

**参考答案：**

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 创建并训练模型
torch.manual_seed(42)
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for _ in range(10):
    x = torch.randn(16, 10)
    y = torch.randint(0, 5, (16,))
    loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 保存
torch.save(model.state_dict(), '/tmp/model_weights.pth')
print("模型已保存")

# 加载到新实例
model_loaded = SimpleModel()
model_loaded.load_state_dict(torch.load('/tmp/model_weights.pth', map_location='cpu'))
model_loaded.eval()

# 验证一致性
test_input = torch.randn(4, 10)
model.eval()
with torch.no_grad():
    out_original = model(test_input)
    out_loaded = model_loaded(test_input)

print(f"输出一致: {torch.allclose(out_original, out_loaded)}")
print(f"最大差异: {(out_original - out_loaded).abs().max().item()}")

import os
os.remove('/tmp/model_weights.pth')
```

---

### 题目 2: 断点续训 Checkpoint

**题目描述：**
实现完整的 checkpoint 保存和恢复功能。保存模型参数、优化器状态、学习率调度器状态和当前 epoch。从 checkpoint 恢复后验证可以继续训练。

**提示：**
- checkpoint 字典应包含：model、optimizer、scheduler、epoch
- 恢复时所有状态都要加载

**参考答案：**

```python
import torch
import torch.nn as nn
import os

model = nn.Linear(10, 5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

# 训练 5 个 epoch 后保存
print("=== 第一阶段训练 ===")
for epoch in range(5):
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: loss={loss.item():.4f}, lr={lr:.6f}")

# 保存 checkpoint
checkpoint = {
    'epoch': 4,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}
torch.save(checkpoint, '/tmp/checkpoint.pth')

# 恢复
print("\n=== 从 Checkpoint 恢复 ===")
model2 = nn.Linear(10, 5)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.5)

ckpt = torch.load('/tmp/checkpoint.pth', map_location='cpu')
model2.load_state_dict(ckpt['model_state_dict'])
optimizer2.load_state_dict(ckpt['optimizer_state_dict'])
scheduler2.load_state_dict(ckpt['scheduler_state_dict'])
start_epoch = ckpt['epoch'] + 1

print(f"从 epoch {start_epoch} 恢复, lr={optimizer2.param_groups[0]['lr']:.6f}")

# 继续训练
print("\n=== 第二阶段训练 ===")
for epoch in range(start_epoch, start_epoch + 5):
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    loss = criterion(model2(x), y)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    scheduler2.step()
    lr = optimizer2.param_groups[0]['lr']
    print(f"Epoch {epoch}: loss={loss.item():.4f}, lr={lr:.6f}")

os.remove('/tmp/checkpoint.pth')
```

---

### 题目 3: 迁移学习与微调

**题目描述：**
实现两种微调方式：(1) 冻结特征提取层，只训练分类器；(2) 差异学习率微调。对比两种方式下可训练的参数数量。

**提示：**
- 冻结：`param.requires_grad = False`
- 差异学习率：为优化器传入参数组列表
- 替换分类器后新层默认 `requires_grad=True`

**参考答案：**

```python
import torch
import torch.nn as nn

class PretrainedModel(nn.Module):
    """模拟预训练模型。"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 1000)  # 预训练的分类器

    def forward(self, x):
        return self.classifier(self.features(x))

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ===== 方式一：冻结特征层 =====
print("=== 方式一：冻结特征层 ===")
model_v1 = PretrainedModel()

# 冻结所有参数
for param in model_v1.parameters():
    param.requires_grad = False

# 替换分类器
model_v1.classifier = nn.Linear(128, 10)

total, trainable = count_params(model_v1)
print(f"总参数: {total:,}, 可训练: {trainable:,} ({100*trainable/total:.1f}%)")

optimizer_v1 = torch.optim.Adam(model_v1.classifier.parameters(), lr=0.001)

# ===== 方式二：差异学习率 =====
print("\n=== 方式二：差异学习率 ===")
model_v2 = PretrainedModel()
model_v2.classifier = nn.Linear(128, 10)

total, trainable = count_params(model_v2)
print(f"总参数: {total:,}, 可训练: {trainable:,} ({100*trainable/total:.1f}%)")

optimizer_v2 = torch.optim.SGD([
    {'params': model_v2.features.parameters(), 'lr': 0.0001},
    {'params': model_v2.classifier.parameters(), 'lr': 0.001},
], momentum=0.9)

print("\n参数组学习率：")
for i, group in enumerate(optimizer_v2.param_groups):
    n = sum(p.numel() for p in group['params'])
    print(f"  组 {i}: lr={group['lr']}, 参数量={n:,}")
```

---

### 题目 4: GPU 设备管理

**题目描述：**
编写一个设备感知的训练函数，自动检测可用设备（CUDA / MPS / CPU），正确地将模型和数据迁移到目标设备。

**提示：**
- Tensor 的 `.to(device)` 是非原地操作，需要重新赋值
- Model 的 `.to(device)` 是原地操作
- 检查 MPS 可用性：`torch.backends.mps.is_available()`

**参考答案：**

```python
import torch
import torch.nn as nn

def get_device():
    """自动检测最佳设备。"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用 Apple MPS")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device

def train_on_device(model, train_data, train_labels, epochs=5):
    """设备感知的训练函数。"""
    device = get_device()

    # 模型迁移（原地操作）
    model.to(device)

    # 数据迁移（非原地操作，需要重新赋值）
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        # 取回 CPU 用于日志
        loss_val = loss.item()
        print(f"Epoch {epoch}: loss={loss_val:.4f}, device={next(model.parameters()).device}")

    return model

# 测试
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
X = torch.randn(64, 10)
y = torch.randint(0, 5, (64,))

model = train_on_device(model, X, y, epochs=5)
```

---

## 进阶题 (Intermediate)

### 题目 5: DataParallel 多 GPU 模型加载

**题目描述：**
模拟 DataParallel 训练保存的模型（state_dict 键带 `module.` 前缀），实现一个通用的加载函数，自动处理有无前缀的情况。

**提示：**
- DataParallel 保存的 state_dict 键格式：`module.layer.weight`
- 普通保存的格式：`layer.weight`
- 加载函数需要自动适配两种情况

**参考答案：**

```python
import torch
import torch.nn as nn
from collections import OrderedDict

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def load_state_dict_flexible(model, state_dict):
    """灵活加载 state_dict，自动处理 'module.' 前缀。"""
    model_keys = set(model.state_dict().keys())
    saved_keys = set(state_dict.keys())

    # 检查是否需要去除 'module.' 前缀
    if all(k.startswith('module.') for k in saved_keys):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v  # 去掉 'module.'
        state_dict = new_state_dict
        print("检测到 DataParallel 前缀，已自动去除")

    # 检查是否需要添加 'module.' 前缀
    elif all(k.startswith('module.') for k in model_keys):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict['module.' + k] = v
        state_dict = new_state_dict
        print("模型需要 DataParallel 前缀，已自动添加")

    model.load_state_dict(state_dict)
    print("State dict 加载成功")

# 测试：模拟 DataParallel 保存的 state_dict
model_original = MyModel()
# 模拟加 'module.' 前缀
dp_state_dict = OrderedDict()
for k, v in model_original.state_dict().items():
    dp_state_dict['module.' + k] = v

print("保存的键:", list(dp_state_dict.keys()))

# 加载到普通模型
model_new = MyModel()
load_state_dict_flexible(model_new, dp_state_dict)

# 验证
x = torch.randn(4, 10)
with torch.no_grad():
    out1 = model_original(x)
    out2 = model_new(x)
print(f"输出一致: {torch.allclose(out1, out2)}")
```

---

### 题目 6: AverageMeter 工具类

**题目描述：**
实现一个 `AverageMeter` 类用于跟踪训练指标，支持当前值、累计平均值和滑动窗口平均值。在模拟训练中使用它。

**提示：**
- 记录 val（当前值）、sum（累计和）、count（计数）、avg（平均值）
- 滑动窗口平均：只取最近 N 个值的平均
- 提供 `reset()` 方法

**参考答案：**

```python
import torch
from collections import deque

class AverageMeter:
    """跟踪指标的平均值和当前值。"""

    def __init__(self, name='', window_size=20):
        self.name = name
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.window = deque(maxlen=self.window_size)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.window.append(val)

    @property
    def window_avg(self):
        if len(self.window) == 0:
            return 0
        return sum(self.window) / len(self.window)

    def __str__(self):
        return (f"{self.name}: val={self.val:.4f}, "
                f"avg={self.avg:.4f}, "
                f"window_avg={self.window_avg:.4f}")

# 使用示例
loss_meter = AverageMeter('Loss', window_size=10)
acc_meter = AverageMeter('Accuracy', window_size=10)

# 模拟训练
import math
for step in range(50):
    fake_loss = 1.0 * math.exp(-0.05 * step) + 0.1 * torch.randn(1).item()
    fake_acc = min(0.95, 0.3 + 0.015 * step + 0.05 * torch.randn(1).item())

    loss_meter.update(fake_loss)
    acc_meter.update(fake_acc)

    if step % 10 == 0:
        print(f"Step {step:>3d} | {loss_meter} | {acc_meter}")

print(f"\n最终统计:")
print(f"  Loss: 全局平均={loss_meter.avg:.4f}, 最近窗口={loss_meter.window_avg:.4f}")
print(f"  Acc:  全局平均={acc_meter.avg:.4f}, 最近窗口={acc_meter.window_avg:.4f}")
```

---

### 题目 7: 完整训练模板

**题目描述：**
实现一个可复用的训练模板，包含：参数配置、日志记录、训练/验证循环、最佳模型保存、checkpoint 保存和早停机制。

**提示：**
- 使用 dataclass 或字典管理配置
- 实现 early stopping
- 保存最佳模型和定期 checkpoint

**参考答案：**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
import logging
import os

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.01
    patience: int = 5  # 早停的耐心值
    save_dir: str = '/tmp/training_output'

class EarlyStopping:
    """早停机制。"""
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return False
        if score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 触发早停
        else:
            self.best_score = score
            self.counter = 0
        return False

def train(config):
    """完整训练流程。"""
    os.makedirs(config.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    torch.manual_seed(42)
    X_train = torch.randn(500, 10)
    y_train = torch.randint(0, 5, (500,))
    X_val = torch.randn(100, 10)
    y_val = torch.randint(0, 5, (100,))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=config.batch_size)

    # 模型
    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 5),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    early_stopping = EarlyStopping(patience=config.patience)

    best_val_acc = 0

    for epoch in range(config.epochs):
        # 训练
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:>3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"LR: {lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(config.save_dir, 'best_model.pth'))

        # 早停检查
        if early_stopping(val_loss):
            print(f"早停触发于 epoch {epoch}")
            break

    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.2%}")

    # 清理
    best_path = os.path.join(config.save_dir, 'best_model.pth')
    if os.path.exists(best_path):
        os.remove(best_path)

# 运行
config = TrainConfig(epochs=30, lr=0.005, patience=8)
train(config)
```

---

## 挑战题 (Challenge)

### 题目 8: 模型集成实现

**题目描述：**
实现一个模型集成框架：训练 3 个不同初始化的模型，使用投票法和平均法进行集成预测，对比单模型和集成模型的准确率。

**提示：**
- 不同随机种子初始化不同模型
- 投票法：取多数预测结果
- 平均法：平均 logits 后取 argmax

**参考答案：**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

class SimpleClassifier(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 5),
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def ensemble_vote(models, x):
    """投票法集成。"""
    all_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = model(x).argmax(1)
            all_preds.append(preds)

    # 逐样本投票
    stacked = torch.stack(all_preds, dim=0)  # (n_models, batch)
    final_preds = []
    for i in range(x.size(0)):
        votes = stacked[:, i].tolist()
        final_preds.append(Counter(votes).most_common(1)[0][0])
    return torch.tensor(final_preds)

def ensemble_average(models, x):
    """平均法集成。"""
    all_logits = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            all_logits.append(logits)
    avg_logits = torch.stack(all_logits).mean(dim=0)
    return avg_logits.argmax(1)

# 准备数据
torch.manual_seed(0)
X_train = torch.randn(1000, 20)
y_train = (X_train[:, :5].sum(1) > 0).long() + (X_train[:, 5:10].sum(1) > 0).long()
y_train = y_train.clamp(0, 4)
X_test = torch.randn(200, 20)
y_test = (X_test[:, :5].sum(1) > 0).long() + (X_test[:, 5:10].sum(1) > 0).long()
y_test = y_test.clamp(0, 4)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=200)

# 训练 3 个模型
models = []
for i, seed in enumerate([42, 123, 456]):
    model = SimpleClassifier(seed=seed)
    model = train_model(model, train_loader, epochs=30)
    acc = evaluate(model, test_loader)
    models.append(model)
    print(f"模型 {i+1} (seed={seed}): accuracy={acc:.2%}")

# 集成预测
print("\n=== 集成结果 ===")
for x, y in test_loader:
    # 投票法
    vote_preds = ensemble_vote(models, x)
    vote_acc = (vote_preds == y).float().mean()

    # 平均法
    avg_preds = ensemble_average(models, x)
    avg_acc = (avg_preds == y).float().mean()

    print(f"投票法准确率: {vote_acc:.2%}")
    print(f"平均法准确率: {avg_acc:.2%}")
```

---

## 总结与学习建议

### 关键概念回顾

1. **模型保存**：推荐保存 `state_dict`，注意 `map_location` 跨设备加载
2. **断点续训**：保存 model + optimizer + scheduler + epoch
3. **微调**：冻结法适合数据少，差异学习率适合数据多
4. **GPU**：Tensor 的 `to()` 非原地，Model 的 `to()` 原地
5. **多 GPU**：DataParallel 会给键加 `module.` 前缀
6. **集成**：多个模型组合通常能提升性能

### 常见陷阱

- 忘记 `model.eval()` 导致推理时 BN/Dropout 行为错误
- DataParallel 的 `module.` 前缀导致加载失败
- Tensor 的 `to(device)` 忘记重新赋值
- checkpoint 没有保存 scheduler 状态导致恢复后学习率错误
