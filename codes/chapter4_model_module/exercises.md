# 第四章 模型模块 - 练习题

---

## 基础题 (Fundamental)

### 题目 1: 创建简单的全连接网络

**题目描述：**
创建一个简单的三层全连接网络 `SimpleMLP`，接受 784 维输入（28×28 图像展平），经过两个隐藏层（256、128），输出 10 个类别。使用 ReLU 激活函数。

**提示：**
- 继承 `nn.Module`
- 在 `__init__` 中定义三个 `nn.Linear` 层
- 在 `forward` 中连接各层，使用 `F.relu` 激活

**参考答案：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 测试
model = SimpleMLP()
fake_input = torch.randn(4, 1, 28, 28)
output = model(fake_input)
print(f"输入形状: {fake_input.shape}")
print(f"输出形状: {output.shape}")  # torch.Size([4, 10])

# 查看参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
```

---

### 题目 2: 使用 Sequential 构建网络

**题目描述：**
使用 `nn.Sequential` 分别用两种方式（直接传入模块、OrderedDict）构建与题目 1 相同结构的网络，并验证输出一致。

**提示：**
- 方式一：直接传入模块列表
- 方式二：使用 `OrderedDict` 给每层命名
- 使用相同的随机种子验证输出

**参考答案：**

```python
import torch
import torch.nn as nn
from collections import OrderedDict

# 方式一：直接传入
model_v1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 方式二：OrderedDict
model_v2 = nn.Sequential(OrderedDict([
    ('flatten', nn.Flatten()),
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 128)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(128, 10)),
]))

# 测试
fake_input = torch.randn(4, 1, 28, 28)
out_v1 = model_v1(fake_input)
out_v2 = model_v2(fake_input)
print(f"方式一输出形状: {out_v1.shape}")
print(f"方式二输出形状: {out_v2.shape}")

# 可以通过名称访问子模块
print(f"方式二 fc1 层: {model_v2.fc1}")
```

---

### 题目 3: Parameter 与 Tensor 的区别

**题目描述：**
创建一个自定义模块 `CustomLayer`，包含一个 `nn.Parameter` 权重和一个普通 `Tensor` 属性。使用 `model.parameters()` 和 `model.state_dict()` 观察两者的区别。

**提示：**
- `nn.Parameter` 会被 `parameters()` 发现
- 普通 `Tensor` 不会被 `parameters()` 发现
- 使用 `register_buffer` 可以将普通 Tensor 注册到 `state_dict`

**参考答案：**

```python
import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Parameter：会被优化器更新
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # 普通 Tensor：不会被优化器更新
        self.scale = torch.tensor(0.1)
        # register_buffer：不被优化器更新，但会保存到 state_dict
        self.register_buffer('running_mean', torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T * self.scale + self.running_mean

layer = CustomLayer(10, 5)

# 查看 parameters
print("=== Parameters（可训练参数）===")
for name, param in layer.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# 查看 state_dict
print("\n=== State Dict ===")
for name, val in layer.state_dict().items():
    print(f"  {name}: shape={val.shape}")

# 注意：scale 不在 parameters 中，也不在 state_dict 中
# running_mean 不在 parameters 中，但在 state_dict 中
```

---

### 题目 4: 探索 Module 的结构

**题目描述：**
创建一个包含多层嵌套的模型，使用 `named_modules()`、`named_children()`、`named_parameters()` 分别遍历，理解它们的区别。

**提示：**
- `modules()`：递归遍历所有模块（包含自身）
- `children()`：只遍历直接子模块
- `parameters()`：遍历所有可训练参数

**参考答案：**

```python
import torch.nn as nn

class SubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.block = SubBlock()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x

model = MyModel()

# named_modules：递归遍历所有模块
print("=== named_modules（全部模块）===")
for name, module in model.named_modules():
    print(f"  '{name}': {module.__class__.__name__}")

# named_children：只遍历直接子模块
print("\n=== named_children（直接子模块）===")
for name, child in model.named_children():
    print(f"  '{name}': {child.__class__.__name__}")

# named_parameters：遍历所有参数
print("\n=== named_parameters（所有参数）===")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

---

## 进阶题 (Intermediate)

### 题目 5: 使用 ModuleList 实现可变深度网络

**题目描述：**
创建一个 `FlexibleMLP` 类，通过参数控制隐藏层的数量和大小。使用 `nn.ModuleList` 管理可变数量的线性层。

**提示：**
- 使用列表推导式创建 `nn.ModuleList`
- 每层后加 ReLU 和 Dropout
- 必须使用 `nn.ModuleList` 而非普通 Python list

**参考答案：**

```python
import torch
import torch.nn as nn

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        """
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表，如 [256, 128, 64]
            output_dim: 输出维度
            dropout: Dropout 概率
        """
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(len(hidden_dims))
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = dropout(self.relu(layer(x)))
        x = self.output_layer(x)
        return x

# 测试不同深度
for hidden_dims in [[128], [256, 128], [512, 256, 128, 64]]:
    model = FlexibleMLP(784, hidden_dims, 10)
    x = torch.randn(4, 784)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"hidden_dims={hidden_dims}, output={out.shape}, params={n_params:,}")
```

---

### 题目 6: 使用 ModuleDict 实现可选择的网络

**题目描述：**
创建一个 `SelectableModel` 类，支持在前向传播时动态选择特征提取器（CNN 或 MLP）和激活函数（ReLU 或 GELU）。

**提示：**
- 使用 `nn.ModuleDict` 存储多个可选模块
- 在 `forward` 中通过字符串参数选择使用哪个模块

**参考答案：**

```python
import torch
import torch.nn as nn

class SelectableModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.extractors = nn.ModuleDict({
            'cnn': nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            ),
            'mlp': nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 16),
            ),
        })
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
        })
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x, extractor='cnn', activation='relu'):
        x = self.extractors[extractor](x)
        x = self.activations[activation](x)
        x = self.fc(x)
        return x

model = SelectableModel()
x = torch.randn(4, 1, 28, 28)

# 不同组合
for ext in ['cnn', 'mlp']:
    for act in ['relu', 'gelu']:
        out = model(x, extractor=ext, activation=act)
        print(f"extractor={ext}, activation={act} → output={out.shape}")
```

---

### 题目 7: 实现简单 CNN 并观察各层输出

**题目描述：**
创建一个包含 3 个卷积层的 CNN，使用 `register_forward_hook` 提取每一层的特征图，打印各层输出的形状。

**提示：**
- 注册 forward hook 到每个卷积层
- hook 函数签名：`hook(module, input, output)`
- 使用完后移除 hook

**参考答案：**

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN()

# 用字典存储各层特征图
feature_maps = {}

def make_hook(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach()
    return hook

# 注册 hook
handles = []
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

# 前向传播
x = torch.randn(1, 3, 32, 32)
output = model(x)

# 查看各层特征图
print("各层特征图形状：")
for name, fmap in feature_maps.items():
    print(f"  {name}: {fmap.shape}")

# 移除 hook
for h in handles:
    h.remove()
```

---

### 题目 8: 实现权重初始化函数

**题目描述：**
为一个 CNN 模型实现自定义的权重初始化函数，对不同类型的层使用不同的初始化策略。验证初始化前后权重分布的变化。

**提示：**
- Conv2d 使用 Kaiming 初始化
- BatchNorm2d 的 weight 初始化为 1，bias 初始化为 0
- Linear 使用 Xavier 初始化
- 使用 `model.apply()` 应用初始化

**参考答案：**

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        x = self.classifier(x)
        return x

def init_weights(m):
    """自定义权重初始化。"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

model = CNN()

# 初始化前
print("=== 初始化前 ===")
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")

# 应用初始化
model.apply(init_weights)

# 初始化后
print("\n=== 初始化后 ===")
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
```

---

## 挑战题 (Challenge)

### 题目 9: 实现 ResNet 残差块

**题目描述：**
实现一个 `BasicBlock`（残差块），包含两个 3×3 卷积、BN 和 ReLU，以及 shortcut 连接。当输入输出通道数不同时，使用 1×1 卷积进行 downsample。用残差块搭建一个小型 ResNet。

**提示：**
- 残差连接：`output = F(x) + x`
- 当通道数变化或空间尺寸变化时，需要 downsample
- downsample 使用 1×1 卷积 + BN

**参考答案：**

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 当维度不匹配时使用 downsample
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)
        return out

class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 16, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 测试
model = TinyResNet(num_classes=10)
x = torch.randn(4, 3, 32, 32)
out = model(x)
print(f"输出形状: {out.shape}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

### 题目 10: 实现 Grad-CAM

**题目描述：**
对一个预训练的模型实现 Grad-CAM，使用 Hook 函数提取最后一层卷积的特征图和梯度，生成类激活热力图。

**提示：**
- 使用 `register_forward_hook` 捕获特征图
- 使用 `register_full_backward_hook` 捕获梯度
- 权重 = 梯度的全局平均池化
- 热力图 = ReLU(权重加权特征图求和)

**参考答案：**

```python
import torch
import torch.nn as nn
import numpy as np

class GradCAM:
    """Grad-CAM 实现。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.fmap = None
        self.grad = None

        # 注册 hook
        self._hook_fwd = target_layer.register_forward_hook(self._forward_hook)
        self._hook_bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.fmap = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """生成 Grad-CAM 热力图。"""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 构造 one-hot 并反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot)

        # 计算权重：全局平均池化梯度
        weights = self.grad.mean(dim=[2, 3], keepdim=True)

        # 加权求和
        cam = (weights * self.fmap).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().numpy()

        # 归一化到 [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, target_class

    def remove_hooks(self):
        self._hook_fwd.remove()
        self._hook_bwd.remove()

# 使用示例
model = TinyResNet(num_classes=10)  # 使用上一题的模型
grad_cam = GradCAM(model, model.layer3[-1].conv2)

x = torch.randn(1, 3, 32, 32)
cam, pred_class = grad_cam.generate(x)
print(f"预测类别: {pred_class}")
print(f"CAM 形状: {cam.shape}")
print(f"CAM 值范围: [{cam.min():.4f}, {cam.max():.4f}]")

grad_cam.remove_hooks()
```

---

### 题目 11: 模型状态管理综合练习

**题目描述：**
实现一个完整的模型管理工作流：创建模型 → 训练（模拟） → 保存 checkpoint → 加载恢复 → 冻结部分层 → 继续训练。

**提示：**
- 使用 `state_dict()` 和 `load_state_dict()` 管理参数
- 使用 `requires_grad = False` 冻结层
- checkpoint 需要保存模型、优化器、epoch 等状态

**参考答案：**

```python
import torch
import torch.nn as nn
import os

class SimpleNet(nn.Module):
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
        x = self.features(x)
        x = self.classifier(x)
        return x

# 1. 创建模型和优化器
model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 2. 模拟训练几个 epoch
print("=== 训练阶段 ===")
for epoch in range(3):
    model.train()
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")

# 3. 保存 checkpoint
checkpoint = {
    'epoch': 2,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, '/tmp/checkpoint.pth')
print("\nCheckpoint 已保存")

# 4. 加载恢复
model_new = SimpleNet()
optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.001)

ckpt = torch.load('/tmp/checkpoint.pth', map_location='cpu')
model_new.load_state_dict(ckpt['model_state_dict'])
optimizer_new.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch'] + 1
print(f"从 epoch {start_epoch} 恢复")

# 5. 冻结特征提取层，只训练分类器
for param in model_new.features.parameters():
    param.requires_grad = False

# 验证冻结效果
print("\n=== 冻结后参数状态 ===")
for name, param in model_new.named_parameters():
    print(f"  {name}: requires_grad={param.requires_grad}")

# 6. 只优化未冻结的参数
trainable_params = filter(lambda p: p.requires_grad, model_new.parameters())
optimizer_ft = torch.optim.Adam(trainable_params, lr=0.0001)

print("\n=== 微调训练 ===")
for epoch in range(start_epoch, start_epoch + 3):
    model_new.train()
    x = torch.randn(32, 10)
    y = torch.randint(0, 5, (32,))
    loss = criterion(model_new(x), y)
    optimizer_ft.zero_grad()
    loss.backward()
    optimizer_ft.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")

# 清理
os.remove('/tmp/checkpoint.pth')
```

---

## 总结与学习建议

### 关键概念回顾

1. **Module**：所有网络的基类，通过 8 个有序字典管理属性
2. **Parameter**：可训练参数，会被 `parameters()` 发现和优化器更新
3. **容器**：Sequential（自动执行）、ModuleList/Dict（手动控制）
4. **Hook**：不修改代码即可提取中间信息
5. **初始化**：ReLU 用 Kaiming，Sigmoid/Tanh 用 Xavier

### 常见陷阱

- 使用普通 Python list 而非 `nn.ModuleList` 导致参数丢失
- 忘记切换 `train()`/`eval()` 导致 BN 和 Dropout 行为错误
- `load_state_dict` 时键名不匹配
- Hook 使用完毕后忘记 `remove()`
