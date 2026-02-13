# Chapter 2 核心模块 - 练习题

本文档包含三个难度级别的练习题，涵盖张量创建、基本操作、形状变换、广播机制、scatter/gather 和自动微分等核心内容。

---

## 基础题（5题）

### 练习 1: 张量创建方式对比

**题目：**

创建以下张量，并比较它们的特性：
1. 使用 `torch.zeros()` 创建一个 `3x4` 的零张量
2. 使用 `torch.ones()` 创建一个 `2x3x4` 的全1张量
3. 使用 `torch.arange()` 创建从0到9的张量
4. 使用 `torch.linspace()` 创建10个均匀分布的值（0到1之间）
5. 使用 `torch.randint()` 创建一个 `2x3` 的随机整数张量（范围0-10）

然后打印每个张量的：
- 形状（shape）
- 数据类型（dtype）
- 设备位置（device）

**提示：**

- 查看 PyTorch 官方文档了解各个创建函数的参数
- 使用 `.shape`, `.dtype`, `.device` 属性获取相关信息

**参考答案：**

```python
import torch

# 1. 零张量
z = torch.zeros(3, 4)
print(f"Zeros: shape={z.shape}, dtype={z.dtype}, device={z.device}")

# 2. 全1张量
o = torch.ones(2, 3, 4)
print(f"Ones: shape={o.shape}, dtype={o.dtype}, device={o.device}")

# 3. arange张量
a = torch.arange(0, 10)
print(f"Arange: shape={a.shape}, dtype={a.dtype}, device={a.device}")

# 4. linspace张量
l = torch.linspace(0, 1, 10)
print(f"Linspace: shape={l.shape}, dtype={l.dtype}, device={l.device}")

# 5. 随机整数张量
r = torch.randint(0, 10, (2, 3))
print(f"Randint: shape={r.shape}, dtype={r.dtype}, device={r.device}")
```

---

### 练习 2: 基本张量操作

**题目：**

给定两个张量 `a` 和 `b`：
```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
b = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
```

执行以下操作并输出结果：
1. 元素级加法（`a + b`）
2. 元素级乘法（`a * b`）
3. 矩阵乘法（`a @ a.T`）
4. 逐元素平方根（`torch.sqrt(a)`）
5. 张量求和（全部、按行、按列）

**提示：**

- 元素级操作按对应位置执行
- 矩阵乘法可用 `@` 操作符或 `torch.matmul()`
- 求和时注意 `dim` 参数的用法

**参考答案：**

```python
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
b = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

# 1. 元素级加法
add_result = a + b
print(f"加法:\n{add_result}")

# 2. 元素级乘法
mul_result = a * b
print(f"乘法:\n{mul_result}")

# 3. 矩阵乘法
matmul_result = a @ a.T
print(f"矩阵乘法 (a @ a.T):\n{matmul_result}")

# 4. 逐元素平方根
sqrt_result = torch.sqrt(a)
print(f"平方根:\n{sqrt_result}")

# 5. 求和
sum_all = torch.sum(a)
sum_row = torch.sum(a, dim=0)
sum_col = torch.sum(a, dim=1)
print(f"全部求和: {sum_all}")
print(f"按行求和: {sum_row}")
print(f"按列求和: {sum_col}")
```

---

### 练习 3: 张量形状变换

**题目：**

给定张量 `x = torch.arange(24).reshape(2, 3, 4)`，执行以下形状变换：

1. 使用 `view()` 将其变换为 `(6, 4)` 的形状
2. 使用 `reshape()` 将其变换为 `(4, 6)` 的形状
3. 使用 `unsqueeze()` 在第0维添加一个维度
4. 使用 `squeeze()` 移除单维度（如果存在）
5. 使用 `permute()` 交换维度顺序为 `(3, 2, 0, 1)` 的4D张量

在每个操作后打印张量的形状。

**提示：**

- `view()` 要求张量连续，`reshape()` 更灵活
- `unsqueeze()` 和 `squeeze()` 用于增删维度
- `permute()` 用于重排维度顺序

**参考答案：**

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)
print(f"原始形状: {x.shape}")

# 1. view 变换
x_view = x.view(6, 4)
print(f"view(6, 4): {x_view.shape}")

# 2. reshape 变换
x_reshape = x.reshape(4, 6)
print(f"reshape(4, 6): {x_reshape.shape}")

# 3. unsqueeze 增加维度
x_unsqueeze = x.unsqueeze(0)
print(f"unsqueeze(0): {x_unsqueeze.shape}")

# 4. squeeze 移除单维度
x_squeezed = x_unsqueeze.squeeze(0)
print(f"squeeze(0): {x_squeezed.shape}")

# 5. permute 重排维度
x_4d = x.unsqueeze(0)  # 转为 (1, 2, 3, 4)
x_permute = x_4d.permute(2, 3, 0, 1)  # 重排为 (3, 4, 1, 2)
print(f"permute后: {x_permute.shape}")
```

---

### 练习 4: 张量索引和切片

**题目：**

给定张量：
```python
x = torch.arange(12).reshape(3, 4)
```

执行以下索引和切片操作：

1. 获取第一行（`x[0]`）
2. 获取第二列（`x[:, 1]`）
3. 获取前两行和前三列（`x[:2, :3]`）
4. 使用高级索引获取特定元素：行索引 `[0, 2, 1]`，列索引 `[0, 2, 1]`
5. 使用布尔索引获取所有大于6的元素

**提示：**

- 标准索引和切片使用位置
- 高级索引用于非连续元素选择
- 布尔索引根据条件过滤元素

**参考答案：**

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(f"原始张量:\n{x}")

# 1. 第一行
row_0 = x[0]
print(f"第一行: {row_0}")

# 2. 第二列
col_1 = x[:, 1]
print(f"第二列: {col_1}")

# 3. 前两行和前三列
slice_result = x[:2, :3]
print(f"前两行和前三列:\n{slice_result}")

# 4. 高级索引
row_indices = torch.tensor([0, 2, 1])
col_indices = torch.tensor([0, 2, 1])
advanced_index = x[row_indices, col_indices]
print(f"高级索引结果: {advanced_index}")

# 5. 布尔索引
mask = x > 6
bool_result = x[mask]
print(f"大于6的元素: {bool_result}")
```

---

### 练习 5: 数据类型转换

**题目：**

创建以下张量并进行类型转换：

1. 创建整数张量 `x = torch.tensor([1, 2, 3, 4, 5])`
2. 转换为浮点数（float32）
3. 转换为 int64
4. 转换为布尔值
5. 创建一个 double 类型的张量并转换为 float16

对每个转换后的张量打印其 dtype。

**提示：**

- 使用 `.to()` 或 `.type()` 方法进行类型转换
- 使用 `torch.float32`, `torch.int64` 等指定类型
- 注意不同类型之间的精度差异

**参考答案：**

```python
import torch

# 1. 整数张量
x = torch.tensor([1, 2, 3, 4, 5])
print(f"原始dtype: {x.dtype}")

# 2. 转换为 float32
x_float = x.to(torch.float32)
print(f"float32: {x_float.dtype}")

# 3. 转换为 int64
x_int64 = x.to(torch.int64)
print(f"int64: {x_int64.dtype}")

# 4. 转换为布尔值
x_bool = x.to(torch.bool)
print(f"bool: {x_bool.dtype}")

# 5. double 转 float16
x_double = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
x_float16 = x_double.to(torch.float16)
print(f"float64 -> float16: {x_float16.dtype}")

# 使用 type() 方法的替代方案
x_alt = x.type(torch.float32)
print(f"使用type()转换: {x_alt.dtype}")
```

---

## 进阶题（5题）

### 练习 6: 广播机制

**题目：**

理解 PyTorch 的广播规则，执行以下操作：

1. 张量形状 `(3, 1)` 与 `(1, 4)` 相加 → 结果形状应为 `(3, 4)`
2. 张量形状 `(2, 3, 4)` 与 `(4,)` 相乘 → 结果形状应为 `(2, 3, 4)`
3. 张量形状 `(5, 1, 3)` 与 `(1, 4, 3)` 相加 → 结果形状应为 `(5, 4, 3)`
4. 解释为什么形状 `(3, 4)` 与 `(2, 4)` 无法广播
5. 创建一个例子说明广播如何减少内存使用

**提示：**

- 广播规则：从右往左对齐维度，逐个比较
- 对齐后，两维要么相同，要么其中一个为1
- 广播不会创建实际的副本，只是逻辑上的扩展

**参考答案：**

```python
import torch

# 1. (3, 1) + (1, 4) -> (3, 4)
a1 = torch.randn(3, 1)
b1 = torch.randn(1, 4)
result1 = a1 + b1
print(f"(3, 1) + (1, 4) = {result1.shape}")

# 2. (2, 3, 4) * (4,) -> (2, 3, 4)
a2 = torch.randn(2, 3, 4)
b2 = torch.randn(4)
result2 = a2 * b2
print(f"(2, 3, 4) * (4,) = {result2.shape}")

# 3. (5, 1, 3) + (1, 4, 3) -> (5, 4, 3)
a3 = torch.randn(5, 1, 3)
b3 = torch.randn(1, 4, 3)
result3 = a3 + b3
print(f"(5, 1, 3) + (1, 4, 3) = {result3.shape}")

# 4. (3, 4) 与 (2, 4) 无法广播
# 解释：从右对齐，第一维3和2不相等，且都不为1，所以无法广播
try:
    a4 = torch.randn(3, 4)
    b4 = torch.randn(2, 4)
    result4 = a4 + b4
except RuntimeError as e:
    print(f"无法广播错误: {e}")

# 5. 广播减少内存使用
# 不广播（显式复制）
a5 = torch.randn(1000, 1)
b5 = torch.randn(1, 5000)
# 如果复制：需要1000*5000 = 500万个元素
result5_copy = a5.expand(1000, 5000) + b5.expand(1000, 5000)

# 广播（隐式）
result5 = a5 + b5  # 不创建额外副本，只是逻辑扩展
print(f"广播结果形状: {result5.shape}")
print(f"实际存储元素数: a5={a5.numel()}, b5={b5.numel()}, 总共={a5.numel() + b5.numel()}")
```

---

### 练习 7: 张量的 scatter 和 gather 操作

**题目：**

掌握 `torch.scatter()` 和 `torch.gather()` 操作：

1. 创建一个源张量和索引张量，使用 `scatter_()` 将值分散到目标张量
2. 使用 `gather()` 从张量中收集指定索引的值
3. 实现一个"一热编码"（one-hot encoding）的反向操作：从概率向量中获取最大值索引
4. 使用 scatter 实现向量的"分箱"（binning）操作
5. 比较 scatter 和 gather 的关系

**提示：**

- `scatter_(dim, index, src)` 在指定维度按索引分散值
- `gather(dim, index)` 在指定维度按索引收集值
- 这两个操作在维度上是互补的
- 索引张量的形状决定了输出的形状

**参考答案：**

```python
import torch

# 1. scatter 操作
src = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
index = torch.tensor([[0, 1, 2], [2, 0, 1]])  # 2x3
output = torch.zeros(2, 3)
output.scatter_(1, index, src)
print(f"Scatter结果:\n{output}")

# 2. gather 操作
data = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
index = torch.tensor([[0, 2], [1, 2]])
gathered = torch.gather(data, 1, index)
print(f"Gather结果:\n{gathered}")

# 3. 从概率向量中获取最大值索引（one-hot反向）
probs = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
max_indices = torch.argmax(probs, dim=1, keepdim=True)
print(f"最大值索引:\n{max_indices}")

# 4. 分箱操作
values = torch.randn(10)
bins = torch.tensor([0, 1, 2])  # 3个箱子
bin_indices = torch.bucketize(values, bins)
print(f"原始值: {values}")
print(f"箱子索引: {bin_indices}")

# 5. scatter 和 gather 的关系
# 如果 y = x.gather(dim, index)
# 则 z.scatter_(dim, index, y) 会将y放回原来的位置
x = torch.arange(12).reshape(3, 4).float()
index = torch.tensor([[0, 2], [1, 3], [0, 3]])
gathered = x.gather(1, index)
print(f"原始张量:\n{x}")
print(f"Gather结果:\n{gathered}")

# 反向 scatter
scatter_back = torch.zeros_like(x)
scatter_back.scatter_(1, index, gathered)
print(f"Scatter回来的结果:\n{scatter_back}")
```

---

### 练习 8: 自动微分基础

**题目：**

使用 PyTorch 的自动微分机制计算导数：

1. 计算 `f(x) = x^2 + 2x + 1` 在 `x=3` 处的导数
2. 计算向量函数 `f(x) = [x₁², x₁*x₂, x₂²]` 在 `x=[2, 3]` 处的雅可比矩阵的第一行
3. 实现一个简单的标量函数，计算其对多个变量的梯度
4. 使用 `backward()` 计算复合函数的导数
5. 使用 `grad=True` 创建张量并进行多次梯度计算

**提示：**

- 需要设置 `requires_grad=True` 才能记录计算图
- 使用 `.backward()` 计算梯度
- 梯度存储在 `.grad` 属性中
- 使用 `.detach()` 分离张量

**参考答案：**

```python
import torch

# 1. 计算 f(x) = x^2 + 2x + 1 在 x=3 处的导数
x = torch.tensor(3.0, requires_grad=True)
f = x**2 + 2*x + 1
f.backward()
print(f"f'(3) = {x.grad}")  # 应该是 2*3 + 2 = 8

# 2. 计算向量函数的雅可比矩阵
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

# f(x) = [x1^2, x1*x2, x2^2]
f1 = x1**2
f1.backward()
df1_dx1 = x1.grad.clone()
df1_dx2_val = 0.0

x1.grad = None
x2.grad = None

# 对 f2 = x1*x2 求导
f2 = x1 * x2
f2.backward()
df2_dx1 = x1.grad.clone()
df2_dx2 = x2.grad.clone()

print(f"df1/dx1 = {df1_dx1}, df2/dx1 = {df2_dx1}, df2/dx2 = {df2_dx2}")

# 3. 标量函数的多变量梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x**2).sum() + (x * 2).sum()
y.backward()
print(f"多变量梯度: {x.grad}")

# 4. 复合函数导数
x = torch.tensor(2.0, requires_grad=True)
y = torch.sin(x)
z = y**2 + torch.exp(y)
z.backward()
print(f"复合函数导数: {x.grad}")

# 5. 多次梯度计算
x = torch.tensor(1.0, requires_grad=True)
# 第一次
y1 = x**3
y1.backward()
grad1 = x.grad.clone()
print(f"第一次梯度: {grad1}")

# 第二次（需要重置）
x.grad = None
y2 = (x**2) * 4
y2.backward()
grad2 = x.grad.clone()
print(f"第二次梯度: {grad2}")
```

---

### 练习 9: 自动微分高阶应用

**题目：**

使用自动微分实现更高级的应用：

1. 计算二阶导数（Hessian矩阵的对角元素）
2. 实现一个简单的神经网络前向传播并计算梯度
3. 使用 `torch.autograd.grad()` 手动计算梯度
4. 实现数值梯度检验（numerical gradient checking）
5. 使用 `retain_graph=True` 实现多次 backward 调用

**提示：**

- 二阶导数需要两次 backward
- `torch.autograd.grad()` 是 `.backward()` 的更灵活替代
- 数值梯度检验用于验证自动微分的正确性
- 默认情况下图在 backward 后被释放

**参考答案：**

```python
import torch

# 1. 计算二阶导数
x = torch.tensor(2.0, requires_grad=True)
y = x**3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"一阶导数: {dy_dx}, 二阶导数: {d2y_dx2}")

# 2. 简单神经网络梯度
x = torch.randn(1, 3, requires_grad=True)
W1 = torch.randn(3, 4, requires_grad=True)
b1 = torch.randn(4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)

# 前向传播
h = torch.relu(x @ W1 + b1)
y = h @ W2
loss = y.sum()
loss.backward()

print(f"W1的梯度形状: {W1.grad.shape}")
print(f"b1的梯度形状: {b1.grad.shape}")

# 3. 使用 torch.autograd.grad()
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x**2).sum() + torch.sin(x).sum()
grads = torch.autograd.grad(y, x, create_graph=True)
print(f"自动微分梯度: {grads[0]}")

# 4. 数值梯度检验
def numerical_gradient(f, x, eps=1e-5):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone()
        x_plus.view(-1)[i] += eps
        x_minus = x.clone()
        x_minus.view(-1)[i] -= eps
        grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

x = torch.tensor([1.0, 2.0], requires_grad=True)
f = lambda t: (t**2).sum()

# 自动微分梯度
y = f(x)
y.backward()
auto_grad = x.grad.clone()

# 数值梯度
x.grad = None
x.requires_grad = False
num_grad = numerical_gradient(f, x)
print(f"自动微分梯度: {auto_grad}")
print(f"数值梯度: {num_grad}")
print(f"误差: {(auto_grad - num_grad).abs().max()}")

# 5. 多次 backward 调用
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x**2).sum()
z = (x**3).sum()

# 计算两个loss的梯度
loss = y + z
loss.backward(retain_graph=True)
grad_first = x.grad.clone()
print(f"第一次梯度: {grad_first}")

x.grad = None
# 再次计算不同的梯度
y.backward(retain_graph=True)
grad_second = x.grad.clone()
print(f"第二次梯度: {grad_second}")
```

---

### 练习 10: 张量操作的综合应用

**题目：**

结合前面的知识解决以下综合问题：

1. 实现批量矩阵乘法（batch matrix multiplication）
2. 实现一个简单的注意力机制中的 softmax 操作
3. 使用张量操作实现 one-hot 编码
4. 计算两组向量之间的余弦相似度
5. 实现一个简单的线性回归，包括前向传播、损失计算和梯度更新

**提示：**

- 批量操作需要理解张量的批量维度
- Softmax 需要使用数值稳定的版本
- 余弦相似度使用 L2 范数
- 线性回归需要计算梯度并更新参数

**参考答案：**

```python
import torch
import torch.nn.functional as F

# 1. 批量矩阵乘法
batch_size, n, m, k = 2, 3, 4, 5
A = torch.randn(batch_size, n, m)
B = torch.randn(batch_size, m, k)
C = torch.bmm(A, B)
print(f"批量矩阵乘法: {A.shape} @ {B.shape} = {C.shape}")

# 2. 注意力机制中的 softmax
scores = torch.randn(2, 3, 4)  # batch_size=2, seq_len=3, num_heads=4
# 数值稳定的 softmax
attention_weights = F.softmax(scores, dim=-1)
print(f"Softmax输出和为1: {attention_weights.sum(dim=-1)[0]}")

# 3. One-hot 编码
class_indices = torch.tensor([0, 2, 1, 0])
num_classes = 3
one_hot = F.one_hot(class_indices, num_classes).float()
print(f"One-hot编码:\n{one_hot}")

# 4. 余弦相似度
v1 = torch.randn(10, 5)  # 10个向量，每个5维
v2 = torch.randn(10, 5)

# L2 范数归一化
v1_norm = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
v2_norm = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-8)

# 余弦相似度 = v1 · v2 / (||v1|| ||v2||)
cosine_sim = (v1_norm * v2_norm).sum(dim=1)
print(f"余弦相似度: {cosine_sim}")

# 5. 简单线性回归
torch.manual_seed(42)

# 创建数据
n_samples = 100
X = torch.randn(n_samples, 1)
true_w = torch.tensor([[2.0]])
true_b = torch.tensor([0.5])
y = X @ true_w.T + true_b + torch.randn(n_samples, 1) * 0.1

# 初始化参数
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
learning_rate = 0.01

# 训练循环
for epoch in range(100):
    # 前向传播
    y_pred = X @ w.T + b
    loss = ((y_pred - y)**2).mean()

    # 反向传播
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()
    loss.backward()

    # 梯度下降
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

print(f"真实参数: w={true_w.item()}, b={true_b.item()}")
print(f"学习到的参数: w={w.item():.4f}, b={b.item():.4f}")
```

---

## 挑战题（3题）

### 练习 11: 自定义梯度函数

**题目：**

使用 `torch.autograd.Function` 实现自定义梯度函数：

1. 实现一个自定义的 ReLU 函数，包括前向和反向传播
2. 实现一个约束函数：硬阈值（hard threshold）- 绝对值小于阈值的设为0
3. 实现一个双向 ReLU（参数化 ReLU），允许负值有非零斜率
4. 验证自定义函数的梯度与数值梯度相符
5. 在一个小神经网络中使用自定义函数并进行端到端训练

**提示：**

- 继承 `torch.autograd.Function`
- 实现 `forward()` 静态方法和 `backward()` 静态方法
- 使用 `apply()` 调用自定义函数
- 梯度检验用于验证正确性

**参考答案：**

```python
import torch
import torch.nn as nn
from torch.autograd import Function

# 1. 自定义 ReLU
class CustomReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

# 使用自定义 ReLU
x = torch.randn(3, 4, requires_grad=True)
y = CustomReLU.apply(x)
loss = y.sum()
loss.backward()
print(f"自定义ReLU梯度:\n{x.grad}")

# 2. 硬阈值函数
class HardThreshold(Function):
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        ctx.threshold = threshold
        ctx.save_for_backward(x)
        return torch.where(torch.abs(x) < threshold, torch.zeros_like(x), x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        threshold = ctx.threshold
        grad_input = torch.where(torch.abs(x) < threshold, torch.zeros_like(grad_output), grad_output)
        return grad_input, None

# 使用硬阈值
x = torch.randn(3, 4, requires_grad=True)
y = HardThreshold.apply(x, 0.5)
loss = y.sum()
loss.backward()
print(f"硬阈值梯度:\n{x.grad}")

# 3. 参数化 ReLU（PReLU）
class ParametricReLU(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.where(x >= 0, x, alpha * x)

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_x = torch.where(x >= 0, grad_output, alpha * grad_output)
        grad_alpha = torch.where(x >= 0, torch.zeros_like(grad_output), x * grad_output).sum()
        return grad_x, grad_alpha

# 使用 PReLU
x = torch.randn(3, 4, requires_grad=True)
alpha = torch.tensor(0.1, requires_grad=True)
y = ParametricReLU.apply(x, alpha)
loss = y.sum()
loss.backward()
print(f"PReLU x梯度形状: {x.grad.shape}")
print(f"PReLU alpha梯度: {alpha.grad}")

# 4. 梯度检验
def check_gradient(x, func, eps=1e-5):
    x_np = x.detach().numpy()

    # 自动微分梯度
    x_auto = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    y_auto = func(x_auto)
    y_auto.backward()
    grad_auto = x_auto.grad.numpy()

    # 数值梯度
    grad_num = np.zeros_like(x_np)
    for i in range(x_np.size):
        x_plus = x_np.copy()
        x_plus.flat[i] += eps
        x_minus = x_np.copy()
        x_minus.flat[i] -= eps

        y_plus = func(torch.from_numpy(x_plus))
        y_minus = func(torch.from_numpy(x_minus))
        grad_num.flat[i] = (y_plus.sum() - y_minus.sum()) / (2 * eps)

    error = np.abs(grad_auto - grad_num).max()
    print(f"梯度检验误差: {error}")
    return error < 1e-3

import numpy as np
x = torch.randn(2, 3, dtype=torch.float64)
check_gradient(x, lambda t: CustomReLU.apply(t))

# 5. 在神经网络中使用
class CustomNetworkReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = CustomReLU.apply(x)
        x = self.fc2(x)
        return x

# 训练
model = CustomNetworkReLU()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)

for epoch in range(10):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = nn.MSELoss()(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

### 练习 12: 复杂张量操作

**题目：**

实现以下复杂的张量操作：

1. 实现一个"滑动窗口"提取操作，从张量中提取所有大小为 `k` 的滑动窗口
2. 实现张量的"展开"（unfold）和"折回"（fold）操作
3. 实现一个矩阵转置的变种：对多维张量进行指定维度的转置
4. 实现一个"Kronecker积"（张量积）操作
5. 实现一个高效的"非极大值抑制"（NMS）操作用于边界框去重

**提示：**

- 滑动窗口可以使用 `unfold()` 或手动切片实现
- 张量积涉及多维操作
- NMS 需要利用张量操作进行高效计算

**参考答案：**

```python
import torch

# 1. 滑动窗口提取
def sliding_window(x, window_size):
    """
    x: (batch, length)
    返回: (batch, num_windows, window_size)
    """
    batch, length = x.shape
    num_windows = length - window_size + 1
    windows = x.unfold(1, window_size, 1)
    return windows

x = torch.arange(20).reshape(2, 10).float()
windows = sliding_window(x, 3)
print(f"原始形状: {x.shape}")
print(f"滑动窗口形状: {windows.shape}")
print(f"第一个样本的第一个窗口: {windows[0, 0, :]}")

# 2. Unfold 和 Fold 操作
# unfold: 从张量中提取滑动窗口
x = torch.arange(16).reshape(1, 4, 4).float()
unfolded = x.unfold(1, 2, 1).unfold(2, 2, 1)
print(f"Unfold前: {x.shape}")
print(f"Unfold后: {unfolded.shape}")

# fold: 将展开的张量折回
# 这通常用于反向操作，例如在反卷积中
patches = torch.randn(1, 4, 3, 3)
output_size = (4, 4)
folded = torch.nn.functional.fold(
    patches.view(1, 4*3*3, -1),
    output_size=output_size,
    kernel_size=(3, 3)
)
print(f"Fold结果: {folded.shape}")

# 3. 多维张量转置
def transpose_multiple_dims(x, dims):
    """
    对张量进行多个维度的转置
    dims: 要转置的两个维度
    """
    return x.transpose(*dims)

x = torch.randn(2, 3, 4, 5)
# 转置第0和第2维
x_transposed = transpose_multiple_dims(x, (0, 2))
print(f"原始形状: {x.shape}")
print(f"转置后形状: {x_transposed.shape}")

# 4. Kronecker 积
def kronecker_product(a, b):
    """
    计算 Kronecker 积: C[i*m:(i+1)*m, j*n:(j+1)*n] = a[i,j] * b
    """
    m, n = a.shape
    p, q = b.shape
    c = torch.zeros(m*p, n*q, device=a.device, dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            c[i*p:(i+1)*p, j*q:(j+1)*q] = a[i, j] * b
    return c

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[0, 5], [6, 7]], dtype=torch.float32)
kron_prod = kronecker_product(a, b)
print(f"Kronecker积:")
print(kron_prod)

# 验证与 torch.kron() 的一致性
kron_prod_torch = torch.kron(a, b)
print(f"torch.kron() 结果与自定义一致: {torch.allclose(kron_prod, kron_prod_torch)}")

# 5. 非极大值抑制（NMS）
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4), 格式为 [x1, y1, x2, y2]
    scores: (N,)
    返回: 保留的索引
    """
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long)

    x1, y1, x2, y2 = boxes.unbind(dim=1)

    # 计算面积
    area = (x2 - x1) * (y2 - y1)

    # 按分数排序
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep.append(current.item())

        if len(sorted_indices) == 1:
            break

        # 计算与当前框的IoU
        x1_current = x1[current]
        y1_current = y1[current]
        x2_current = x2[current]
        y2_current = y2[current]

        x1_inter = torch.max(x1_current, x1[sorted_indices[1:]])
        y1_inter = torch.max(y1_current, y1[sorted_indices[1:]])
        x2_inter = torch.min(x2_current, x2[sorted_indices[1:]])
        y2_inter = torch.min(y2_current, y2[sorted_indices[1:]])

        inter_width = torch.clamp(x2_inter - x1_inter, min=0)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)
        inter_area = inter_width * inter_height

        union_area = area[current] + area[sorted_indices[1:]] - inter_area
        iou = inter_area / union_area

        sorted_indices = sorted_indices[1:][iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)

# 测试 NMS
boxes = torch.tensor([
    [10, 10, 20, 20],
    [12, 12, 22, 22],
    [100, 100, 110, 110],
    [105, 105, 115, 115]
], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

keep_indices = nms(boxes, scores, iou_threshold=0.3)
print(f"NMS保留的框索引: {keep_indices}")
print(f"保留的框:\n{boxes[keep_indices]}")
```

---

### 练习 13: 性能优化与最佳实践

**题目：**

理解和应用 PyTorch 的性能优化技巧：

1. 比较不同张量操作的性能：原地操作 vs 非原地操作
2. 演示梯度累积（gradient accumulation）及其内存优势
3. 实现混合精度训练（mixed precision training）并比较性能
4. 使用 `torch.jit.script()` 编译代码以提高性能
5. 演示张量缓存和内存管理最佳实践

**提示：**

- 使用 `time.time()` 或 `torch.cuda.Event()` 测量性能
- 原地操作（_）节省内存但可能影响梯度计算
- 混合精度需要使用 `torch.cuda.amp`
- JIT 编译适用于确定性的纯函数

**参考答案：**

```python
import torch
import time

# 1. 原地操作 vs 非原地操作
def benchmark_inplace_vs_regular():
    x = torch.randn(10000, 10000)
    y = torch.randn(10000, 10000)

    # 非原地操作
    start = time.time()
    for _ in range(100):
        z = x + y
    time_regular = time.time() - start

    # 原地操作
    x = torch.randn(10000, 10000)
    start = time.time()
    for _ in range(100):
        x += y
    time_inplace = time.time() - start

    print(f"非原地操作时间: {time_regular:.4f}s")
    print(f"原地操作时间: {time_inplace:.4f}s")
    print(f"加速比: {time_regular / time_inplace:.2f}x")

# benchmark_inplace_vs_regular()

# 2. 梯度累积
def gradient_accumulation_demo():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)

    batch_size = 32
    accumulation_steps = 4
    effective_batch_size = batch_size * accumulation_steps

    total_loss = 0
    for i in range(0, len(X), batch_size):
        # 前向传播
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        y_pred = model(batch_X)
        loss = torch.nn.functional.mse_loss(y_pred, batch_y)

        # 反向传播（不更新参数）
        loss.backward()

        # 每 accumulation_steps 步更新参数
        if (i // batch_size + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"更新在有效批大小 {effective_batch_size} 后")

        total_loss += loss.item()

# 3. 混合精度训练
def mixed_precision_training():
    try:
        # 检查是否支持混合精度
        from torch.cuda.amp import autocast, GradScaler

        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()

        X = torch.randn(1000, 100)
        y = torch.randint(0, 10, (1000,))

        # 混合精度训练
        for epoch in range(2):
            optimizer.zero_grad()

            # 使用 autocast 自动混合精度
            with autocast(dtype=torch.float16):
                y_pred = model(X)
                loss = torch.nn.functional.cross_entropy(y_pred, y)

            # 反向传播（使用缩放）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch % 1 == 0:
                print(f"混合精度训练 - Epoch {epoch}, Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"混合精度训练不可用: {e}")

# 4. JIT 编译
def jit_compilation_demo():
    # 使用 script() 编译 PyTorch 代码
    @torch.jit.script
    def custom_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * torch.cos(y) + x * y

    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)

    # 编译后的版本
    start = time.time()
    for _ in range(100):
        result_jit = custom_function(x, y)
    time_jit = time.time() - start

    # 普通版本
    def custom_function_normal(x, y):
        return torch.sin(x) * torch.cos(y) + x * y

    start = time.time()
    for _ in range(100):
        result_normal = custom_function_normal(x, y)
    time_normal = time.time() - start

    print(f"JIT编译时间: {time_jit:.4f}s")
    print(f"普通函数时间: {time_normal:.4f}s")
    print(f"加速比: {time_normal / time_jit:.2f}x")

# 5. 张量缓存和内存管理
def memory_management_best_practices():
    print("内存管理最佳实践:")
    print("1. 使用 detach() 分离不需要梯度的张量")
    print("2. 使用 del 删除不需要的变量")
    print("3. 使用 torch.cuda.empty_cache() 清空缓存")
    print("4. 使用 with torch.no_grad(): 避免不必要的梯度记录")

    # 示例
    x = torch.randn(1000, 1000, requires_grad=True)

    # 不推荐：x 保留计算图
    # y = (x ** 2).sum()

    # 推荐：分离张量
    y = (x ** 2).detach().sum()

    # 清理内存
    del x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 运行演示
print("=" * 50)
print("性能优化演示")
print("=" * 50)

gradient_accumulation_demo()
print()
mixed_precision_training()
print()
jit_compilation_demo()
print()
memory_management_best_practices()
```

---

## 总结

本练习集涵盖了 PyTorch 第二章核心模块的主要内容：

- **基础题**：掌握张量的创建、基本操作和形状变换
- **进阶题**：理解广播机制、scatter/gather 操作和自动微分
- **挑战题**：实现自定义梯度、复杂张量操作和性能优化

建议按难度顺序逐步完成，确保对每个概念有深入的理解。

