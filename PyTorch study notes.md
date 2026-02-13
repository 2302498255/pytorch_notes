# PyTorch 学习笔记

> [参考书籍](https://tingsongyu.github.io/PyTorch-Tutorial-2nd/chapter-1/1.6-JupyterNotebook-install.html) | [参考仓库](https://github.com/TingsongYu/PyTorch-Tutorial-2nd) | [官方文档](https://docs.pytorch.org/docs/stable/torch.html)

---

## 第一章：环境配置

本章介绍 PyTorch 的安装和环境配置，包括如何检查和使用不同的计算设备（CPU、CUDA、MPS）。

### 1.1 安装 PyTorch

#### 1.1.1 选择合适的版本

访问 PyTorch 官网：https://pytorch.org/get-started/locally/

根据您的系统选择：

- **操作系统**：Linux / Mac / Windows
- **包管理器**：Conda（推荐）/ Pip
- **Python 版本**：推荐 3.8-3.11
- **计算平台**：CPU / CUDA（NVIDIA GPU）

#### 1.1.2 安装步骤

**方式1：使用 Conda（推荐）**

```bash
# 创建虚拟环境
conda create -n ml python=3.11

# 激活环境
conda activate ml

# CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU 版本（CUDA 11.8）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# GPU 版本（CUDA 12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Mac (Apple Silicon) - 支持 MPS 加速
pip3 install torch torchvision
```

**方式2：使用 Pip**

```bash
# CPU 版本
pip3 install torch torchvision torchaudio

# GPU 版本（CUDA 11.8）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 版本（CUDA 12.1）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**安装 Jupyter（可选）**

```bash
pip install jupyter
python -m ipykernel install --user --name ml
```

#### 1.1.3 验证安装

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

# Mac Apple Silicon
if hasattr(torch.backends, 'mps'):
    print(f"MPS 可用: {torch.backends.mps.is_available()}")
```

#### 1.1.4 常见问题

**Q1: 如何查看我的 CUDA 版本？**

```bash
# 查看驱动支持的最高 CUDA 版本
nvidia-smi

# 查看已安装的 CUDA toolkit 版本
nvcc --version
```

**Q2: 安装后 import torch 报错？**

```bash
# 卸载后重新安装
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

**Q3: 如何在 Jupyter 中使用虚拟环境？**

```bash
# 安装 ipykernel
pip install ipykernel

# 将环境添加到 Jupyter
python -m ipykernel install --user --name=ml --display-name="Python (ml)"

# 启动 Jupyter
jupyter notebook
# 然后在 Kernel 菜单中选择 "Python (ml)"
```

### 1.2 检查可用设备

在不同平台上，PyTorch 支持的设备不同：

- **Windows/Linux**: 通常使用 CUDA (NVIDIA GPU) 或 CPU
- **Mac (Apple Silicon)**: 使用 MPS (Metal Performance Shaders) 或 CPU
- **Mac (Intel)**: 仅支持 CPU

#### 设备检查 API 对照表

| 设备类型       | 检查方法                              | 参数说明                          | 返回值                                      | 用法示例                                                               |
| -------------- | ------------------------------------- | --------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------- |
| **CPU**  | 无需检查                              | -                                 | 始终可用                                    | `device = torch.device("cpu")`                                       |
| **CUDA** | `torch.cuda.is_available()`         | 无参数                            | `bool`: True 表示 CUDA 可用               | `if torch.cuda.is_available(): device = torch.device("cuda")`        |
| **CUDA** | `torch.cuda.device_count()`         | 无参数                            | `int`: 返回可用 GPU 数量                  | `print(f"GPU 数量: {torch.cuda.device_count()}")`                    |
| **CUDA** | `torch.cuda.get_device_name(index)` | `index (int)`: GPU 索引，默认 0 | `str`: GPU 名称                           | `print(f"GPU 名称: {torch.cuda.get_device_name(0)}")`                |
| **MPS**  | `torch.backends.mps.is_available()` | 无参数                            | `bool`: True 表示 MPS 可用                | `if torch.backends.mps.is_available(): device = torch.device("mps")` |
| **MPS**  | `torch.backends.mps.is_built()`     | 无参数                            | `bool`: True 表示 PyTorch 已构建 MPS 支持 | `print(f"MPS 已构建: {torch.backends.mps.is_built()}")`              |

#### 通用设备检查代码

```python
import torch

def get_device():
    """
    自动检测并返回最佳可用设备
  
    Returns:
        torch.device: 可用的设备对象
    """
    if torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# 使用示例
device = get_device()
print(f"使用设备: {device}")
```

#### 完整设备信息检查代码

```python
import torch

print("=" * 60)
print("PyTorch 设备检查")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")

# CPU 信息
print(f"\n【CPU】")
print(f"  可用: True (始终可用)")
print(f"  设备: {torch.device('cpu')}")

# CUDA 信息
print(f"\n【CUDA】")
print(f"  可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  当前设备: cuda:{torch.cuda.current_device()}")

# MPS 信息 (Mac Apple Silicon)
print(f"\n【MPS】")
mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
mps_built = torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False
print(f"  可用: {mps_available}")
print(f"  已构建: {mps_built}")
if mps_available:
    print(f"  设备: {torch.device('mps')}")

# 推荐设备
print(f"\n【推荐设备】")
if mps_available:
    recommended = torch.device("mps")
    print(f"  {recommended} (Apple Silicon GPU)")
elif torch.cuda.is_available():
    recommended = torch.device("cuda")
    print(f"  {recommended} (NVIDIA GPU)")
else:
    recommended = torch.device("cpu")
    print(f"  {recommended} (CPU)")

print("=" * 60)
```

#### 使用设备创建和移动 Tensor

| 操作                      | 方法                                  | 参数说明                     | 用法示例                                                 |
| ------------------------- | ------------------------------------- | ---------------------------- | -------------------------------------------------------- |
| **创建时指定设备**  | `torch.tensor(data, device=device)` | `device`: 设备对象或字符串 | `t = torch.tensor([1, 2, 3], device="mps")`            |
| **创建时指定设备**  | `torch.zeros(size, device=device)`  | `device`: 设备对象或字符串 | `t = torch.zeros(3, 3, device=device)`                 |
| **移动现有 Tensor** | `tensor.to(device)`                 | `device`: 设备对象或字符串 | `t = t.to("mps")` 或 `t = t.to(torch.device("mps"))` |
| **移动现有 Tensor** | `tensor.cuda()`                     | 无参数                       | `t = t.cuda()` (移动到默认 CUDA 设备)                  |
| **移动现有 Tensor** | `tensor.cpu()`                      | 无参数                       | `t = t.cpu()` (移动到 CPU)                             |
| **移动现有 Tensor** | `t = t.to('mps')`                   | 无参数                       | `t = t.to('mps')` (移动到 MPS 设备，仅 Mac)            |
| **查看设备**        | `tensor.device`                     | 属性，非方法                 | `print(t.device)` 输出: `device(type='mps')`         |

**完整示例：**

```python
import torch

# 自动获取最佳设备
if torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 方法1: 创建时指定设备
t1 = torch.tensor([1, 2, 3], device=device)
print(f"t1 设备: {t1.device}")

# 方法2: 创建后移动到设备
t2 = torch.tensor([4, 5, 6])
t2 = t2.to(device)
print(f"t2 设备: {t2.device}")

# 方法3: 使用字符串指定设备
t3 = torch.zeros(2, 3, device="mps" if torch.backends.mps.is_available() else "cpu")
print(f"t3 设备: {t3.device}")

# 方法4: 移动回 CPU
t4 = t3.cpu()
print(f"t4 设备: {t4.device}")
```

---

## 第二章：核心模块

本章介绍 PyTorch 的核心模块，包括张量（Tensor）和自动微分（Autograd）系统。

### 2.1 核心数据结构：Tensor

- Tensor(张量), 多维矩阵, 是pytorch中最核心的数据结构，用于表达各类数据，如输入数据、模型的参数、模型的特征图、模型的输出等。这里边有一个很重要的数据，就是模型的参数。对于模型的参数，我们需要更新它们，而更新操作需要记录梯度，梯度的记录功能正是被张量所实现的（求梯度是autograd实现的）.

**参考文档：** [PyTorch Tensor 官方文档](https://docs.pytorch.org/docs/stable/torch.html)

**什么是 Tensor？**

Tensor（张量）是 PyTorch 中最核心的数据结构，可以理解为多维数组。在深度学习中，Tensor 用于表达各类数据：

- **输入数据**：如图像、文本等
- **模型参数**：如权重矩阵、偏置向量等
- **特征图**：神经网络中间层的输出
- **模型输出**：如分类概率、回归值等

对于模型的参数，我们需要更新它们，而更新操作需要记录梯度。梯度的记录功能正是被张量所实现的（求梯度是 autograd 实现的）。

#### 2.1.1 张量的结构

![1770743463582](image/study_notes/1770743463582.png)

Tensor 主要有以下八个**主要属性**：

| 属性                    | 类型         | 说明                                                                                                   |
| ----------------------- | ------------ | ------------------------------------------------------------------------------------------------------ |
| **data**          | Tensor       | 多维数组，最核心的属性，其他属性都是为其服务的                                                         |
| **dtype**         | torch.dtype  | 多维数组的数据类型，如 torch.float32、torch.int64 等                                                   |
| **shape**         | torch.Size   | 多维数组的形状，如 (3, 4) 表示 3 行 4 列                                                               |
| **device**        | torch.device | tensor 所在的设备，如 cpu、cuda、mps                                                                   |
| **grad**          | Tensor       | 对应于 data 的梯度，形状与 data 一致                                                                   |
| **grad_fn**       | Function     | 记录创建该 Tensor 时用到的 Function，该 Function 在反向传播计算中使用，是自动求导的关键                |
| **requires_grad** | bool         | 指示是否计算梯度，默认为 False                                                                         |
| **is_leaf**       | bool         | 指示节点是否为叶子节点。为叶子节点时，反向传播结束，其梯度仍会保存；非叶子节点的梯度被释放，以节省内存 |

#### 2.1.2 张量的创建

张量的创建有多种方式，包括直接创建、依据数值创建、依概率分布创建等。本节提供完整的创建方法参考和实用示例。

##### 快速参考表

| 创建方式             | 函数                   | 主要用途                      | 示例                                     |
| -------------------- | ---------------------- | ----------------------------- | ---------------------------------------- |
| **从数据创建** | `torch.tensor()`     | 从列表、数组等创建            | `torch.tensor([1, 2, 3])`              |
| **从 NumPy**   | `torch.from_numpy()` | 从 NumPy 数组创建（共享内存） | `torch.from_numpy(np_arr)`             |
| **未初始化**   | `torch.empty()`      | 创建未初始化的张量（最快）    | `torch.empty(3, 4)`                    |
| **全零张量**   | `torch.zeros()`      | 创建全零张量                  | `torch.zeros(3, 4)`                    |
| **全一张量**   | `torch.ones()`       | 创建全一张量                  | `torch.ones(2, 3)`                     |
| **填充值**     | `torch.full()`       | 创建指定值的张量              | `torch.full((2, 3), 5.0)`              |
| **单位矩阵**   | `torch.eye()`        | 创建单位矩阵                  | `torch.eye(3)`                         |
| **等差数列**   | `torch.arange()`     | 创建等差数列                  | `torch.arange(0, 10, 2)`               |
| **等分序列**   | `torch.linspace()`   | 创建等分序列                  | `torch.linspace(0, 1, 5)`              |
| **对数序列**   | `torch.logspace()`   | 创建对数等分序列              | `torch.logspace(0, 2, 5)`              |
| **均匀分布**   | `torch.rand()`       | [0, 1) 均匀分布               | `torch.rand(3, 4)`                     |
| **标准正态**   | `torch.randn()`      | 标准正态分布                  | `torch.randn(3, 4)`                    |
| **整数随机**   | `torch.randint()`    | 整数均匀分布                  | `torch.randint(0, 10, (3, 4))`         |
| **正态分布**   | `torch.normal()`     | 自定义正态分布                | `torch.normal(0, 1, (3, 4))`           |
| **随机排列**   | `torch.randperm()`   | 随机排列                      | `torch.randperm(10)`                   |
| **伯努利**     | `torch.bernoulli()`  | 伯努利分布                    | `torch.bernoulli(torch.tensor([0.3]))` |

##### 一、直接创建（从已有数据）

###### 1. torch.tensor()

**功能：** 从 Python 列表、元组、NumPy 数组等创建张量。

**函数签名：**

```python
torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
```

**主要参数：**

| 参数                    | 类型         | 说明                                                              | 默认值   |
| ----------------------- | ------------ | ----------------------------------------------------------------- | -------- |
| **data**          | array_like   | tensor 的初始数据，可以是 list, tuple, numpy array, scalar 等     | 必需     |
| **dtype**         | torch.dtype  | tensor 的数据类型，如 torch.float32, torch.int64 等（见下方详解） | 自动推断 |
| **device**        | torch.device | tensor 所在的设备，如 cpu, cuda, mps                              | cpu      |
| **requires_grad** | bool         | 是否需要计算梯度                                                  | False    |
| **pin_memory**    | bool         | 是否将 tensor 存于锁页内存（用于 GPU 加速）                       | False    |

**dtype 参数详解：**

PyTorch 支持多种数据类型，选择合适的 dtype 对性能和精度都有重要影响。

| dtype              | 别名             | 说明               | 字节数 | 数值范围        | 使用场景                       |
| ------------------ | ---------------- | ------------------ | ------ | --------------- | ------------------------------ |
| `torch.float32`  | `torch.float`  | 单精度浮点（默认） | 4      | ±3.4e38        | **深度学习训练（推荐）** |
| `torch.float64`  | `torch.double` | 双精度浮点         | 8      | ±1.7e308       | 科学计算、高精度要求           |
| `torch.float16`  | `torch.half`   | 半精度浮点         | 2      | ±65504         | 混合精度训练、推理加速         |
| `torch.bfloat16` | -                | Brain Float16      | 2      | 同 float32 范围 | 混合精度训练（更稳定）         |
| `torch.int64`    | `torch.long`   | 64位整数           | 8      | -2^63 ~ 2^63-1  | **标签、索引（推荐）**   |
| `torch.int32`    | `torch.int`    | 32位整数           | 4      | -2^31 ~ 2^31-1  | 标签、索引                     |
| `torch.int16`    | `torch.short`  | 16位整数           | 2      | -32768 ~ 32767  | 节省内存                       |
| `torch.int8`     | -                | 8位整数            | 1      | -128 ~ 127      | 量化模型                       |
| `torch.uint8`    | -                | 8位无符号整数      | 1      | 0 ~ 255         | **图像数据（推荐）**     |
| `torch.bool`     | -                | 布尔型             | 1      | True / False    | 掩码、条件判断                 |

**dtype 使用建议：**

```python
import torch

# 深度学习训练：使用 float32（默认，平衡精度和速度）
weights = torch.randn(10, 10, dtype=torch.float32)

# 标签/索引：使用 int64 或 long
labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

# 图像数据：使用 uint8（0-255 范围）
image = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)

# 混合精度训练：使用 float16 或 bfloat16
with torch.cuda.amp.autocast():  # 自动混合精度
    output = model(input.half())  # half() 转换为 float16

# 布尔掩码：使用 bool
mask = torch.tensor([True, False, True, False], dtype=torch.bool)
```

**dtype 自动推断规则：**

```python
# 整数 → int64
t1 = torch.tensor([1, 2, 3])
print(t1.dtype)  # torch.int64

# 浮点数 → float32
t2 = torch.tensor([1.0, 2.0, 3.0])
print(t2.dtype)  # torch.float32

# 布尔值 → bool
t3 = torch.tensor([True, False])
print(t3.dtype)  # torch.bool

# 混合类型 → 提升到更高精度
t4 = torch.tensor([1, 2.0])  # int 和 float 混合
print(t4.dtype)  # torch.float32
```

**示例：**

```python
import torch
import numpy as np

# 从列表创建
t1 = torch.tensor([1, 2, 3])
print(f"从列表: {t1}, dtype: {t1.dtype}")  # tensor([1, 2, 3]), dtype: torch.int64

# 从嵌套列表创建（2D）
t2 = torch.tensor([[1., -1.], [1., -1.]])
print(f"2D张量: {t2}, shape: {t2.shape}")  # shape: torch.Size([2, 2])

# 从 NumPy 数组创建（会复制数据）
arr = np.array([[1, 2, 3], [4, 5, 6]])
t3 = torch.tensor(arr)
print(f"从NumPy: {t3}, dtype: {t3.dtype}")

# 指定数据类型
t4 = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"指定dtype: {t4}, dtype: {t4.dtype}")  # dtype: torch.float32

# 指定设备
t5 = torch.tensor([1, 2, 3], device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"指定device: {t5.device}")

# 需要梯度
t6 = torch.tensor([1., 2., 3.], requires_grad=True)
print(f"requires_grad: {t6.requires_grad}")  # True
```

**注意事项：**

- `torch.tensor()` 会**复制数据**，修改原数据不会影响张量
- 数据类型会自动推断，但可以通过 `dtype` 参数指定
- 对于大数组，建议使用 `torch.from_numpy()` 以避免复制

###### 2. torch.from_numpy()

**功能：** 从 NumPy 数组创建张量，**共享内存**（不复制数据）。

**函数签名：**

```python
torch.from_numpy(ndarray) → Tensor
```

**重要特性：**

- ✅ **共享内存**：修改 NumPy 数组会影响张量，反之亦然
- ✅ **零拷贝**：性能更好，适合大数组
- ⚠️ **仅支持 CPU**：创建的张量只能在 CPU 上
- ⚠️ **仅支持连续数组**：NumPy 数组必须是 C-contiguous

**示例：**

```python
import torch
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
t = torch.from_numpy(arr)

print("初始状态:")
print(f"NumPy array:\n{arr}")
print(f"Tensor:\n{t}")

# 修改 NumPy 数组，Tensor 也会改变
print("\n修改 NumPy 数组:")
arr[0, 0] = 999
print(f"NumPy array:\n{arr}")
print(f"Tensor:\n{t}")  # Tensor 也被修改了

# 修改 Tensor，NumPy 数组也会改变
print("\n修改 Tensor:")
t[0, 1] = -999
print(f"NumPy array:\n{arr}")  # NumPy 数组也被修改了
print(f"Tensor:\n{t}")
```

**对比：torch.tensor() vs torch.from_numpy()**

| 特性               | `torch.tensor()`   | `torch.from_numpy()` |
| ------------------ | -------------------- | ---------------------- |
| **内存**     | 复制数据             | 共享内存               |
| **性能**     | 较慢（需要复制）     | 较快（零拷贝）         |
| **设备**     | 可指定 CPU/GPU/MPS   | 仅 CPU                 |
| **修改影响** | 独立，互不影响       | 相互影响               |
| **适用场景** | 小数据、需要独立副本 | 大数据、需要高性能     |

##### 二、依据数值创建（固定值或序列）

这类函数用于创建具有特定数值模式的张量。

###### 0. torch.empty() / torch.empty_like() ⚡ 性能最优

**功能：** 创建**未初始化**的张量（值未定义，可能是任意值）。这是**最快的创建方式**，适合后续会完全填充的场景。

**函数签名：**

```python
torch.empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.empty_like(input, dtype=None, layout=None, device=None, requires_grad=False)
```

**主要参数：**

| 参数             | 类型         | 说明                                  |
| ---------------- | ------------ | ------------------------------------- |
| **size**   | int...       | 张量的形状                            |
| **input**  | Tensor       | 参考张量，创建的张量与 input 形状相同 |
| **dtype**  | torch.dtype  | 数据类型，默认 torch.float32          |
| **device** | torch.device | 设备位置                              |

**重要特性：**

- ⚡ **性能最优**：不初始化内存，只分配空间，速度最快
- ⚠️ **值未定义**：张量中的值是未初始化的，可能是任意值（包括 NaN、Inf）
- ✅ **适合场景**：后续会完全填充数据的场景，如批量创建后统一赋值

**示例：**

```python
import torch

# 创建未初始化的 3x4 矩阵（值未定义）
t1 = torch.empty(3, 4)
print(f"empty(3, 4):\n{t1}")  # 值可能是任意数

# 创建后立即填充（推荐用法）
t2 = torch.empty(3, 4)
t2.fill_(5.0)  # 或 t2[:] = 5.0
print(f"填充后:\n{t2}")  # 全为 5.0

# 批量创建时性能对比
import time

# 方法1：使用 zeros（较慢，需要初始化）
start = time.time()
for _ in range(1000):
    t = torch.zeros(1000, 1000)
time1 = time.time() - start

# 方法2：使用 empty + fill（较快）
start = time.time()
for _ in range(1000):
    t = torch.empty(1000, 1000)
    t.fill_(0)
time2 = time.time() - start

print(f"zeros 耗时: {time1:.4f}s")
print(f"empty+fill 耗时: {time2:.4f}s")
print(f"性能提升: {time1/time2:.2f}x")

# 使用 empty_like
x = torch.randn(2, 3)
t3 = torch.empty_like(x)
t3.fill_(0)
print(f"empty_like shape: {t3.shape}")
```

**性能对比：**

| 方法              | 速度        | 内存初始化        | 适用场景         |
| ----------------- | ----------- | ----------------- | ---------------- |
| `torch.empty()` | ⚡⚡⚡ 最快 | ❌ 不初始化       | 后续会完全填充   |
| `torch.zeros()` | ⚡⚡ 较快   | ✅ 初始化为0      | 需要零初始化     |
| `torch.ones()`  | ⚡⚡ 较快   | ✅ 初始化为1      | 需要1初始化      |
| `torch.full()`  | ⚡ 较慢     | ✅ 初始化为指定值 | 需要特定值初始化 |

**最佳实践：**

- 如果后续会完全覆盖数据，使用 `torch.empty()` + 填充
- 如果需要零初始化，直接使用 `torch.zeros()`
- 批量创建时，`torch.empty()` 性能优势明显

###### 1. torch.zeros() / torch.zeros_like()

**功能：** 创建全零张量。

**函数签名：**

```python
torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False)
```

**主要参数：**

| 参数             | 类型         | 说明                                  |
| ---------------- | ------------ | ------------------------------------- |
| **size**   | int...       | 张量的形状，可以是多个整数或元组      |
| **input**  | Tensor       | 参考张量，创建的张量与 input 形状相同 |
| **dtype**  | torch.dtype  | 数据类型，默认 torch.float32          |
| **device** | torch.device | 设备位置                              |
| **out**    | Tensor       | 输出张量（可选，用于原地操作）        |

**示例：**

```python
import torch

# 创建 3x4 的全零矩阵
t1 = torch.zeros(3, 4)
print(f"zeros(3, 4):\n{t1}")

# 创建 2x3x4 的全零张量
t2 = torch.zeros(2, 3, 4)
print(f"zeros(2, 3, 4) shape: {t2.shape}")

# 指定数据类型
t3 = torch.zeros(3, 4, dtype=torch.int64)
print(f"zeros with int64 dtype: {t3.dtype}")

# 使用 zeros_like
x = torch.randn(2, 3)
t4 = torch.zeros_like(x)
print(f"zeros_like shape: {t4.shape}, dtype: {t4.dtype}")

# 使用 out 参数（原地操作）
o_t = torch.tensor([1])
t5 = torch.zeros((3, 3), out=o_t)
print(f"t5 和 o_t 是同一个对象: {id(t5) == id(o_t)}")  # True
```

###### 2. torch.ones() / torch.ones_like()

**功能：** 创建全一张量。

**用法与 `torch.zeros()` 完全相同，只是填充值为 1。**

```python
import torch

t1 = torch.ones(3, 4)
t2 = torch.ones_like(torch.randn(2, 3))
print(f"ones(3, 4):\n{t1}")
```

###### 3. torch.full() / torch.full_like()

**功能：** 创建填充指定值的张量。

**函数签名：**

```python
torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.full_like(input, fill_value, out=None, dtype=None, layout=None, device=None, requires_grad=False)
```

**示例：**

```python
import torch

# 创建填充值为 3.141592 的 2x3 矩阵
t1 = torch.full((2, 3), 3.141592)
print(f"full((2, 3), 3.141592):\n{t1}")

# 创建填充值为 5 的整数张量
t2 = torch.full((3, 4), 5, dtype=torch.int32)
print(f"full with int32:\n{t2}")

# 使用 full_like
x = torch.randn(2, 3)
t3 = torch.full_like(x, 10.0)
print(f"full_like: {t3}")
```

###### 4. torch.eye()

**功能：** 创建单位矩阵（对角线为 1，其他为 0）。

**函数签名：**

```python
torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**示例：**

```python
import torch

# 创建 3x3 单位矩阵
t1 = torch.eye(3)
print(f"eye(3):\n{t1}")

# 创建 3x4 矩阵（前3列是单位矩阵）
t2 = torch.eye(3, 4)
print(f"eye(3, 4):\n{t2}")

# 创建 4x3 矩阵（前3行是单位矩阵）
t3 = torch.eye(4, 3)
print(f"eye(4, 3):\n{t3}")
```

###### 5. torch.arange()

**功能：** 创建等差数列的一维张量。

**函数签名：**

```python
torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**重要：** 区间为 `[start, end)`，**右开区间**。

**主要参数：**

| 参数            | 类型   | 说明             | 默认值 |
| --------------- | ------ | ---------------- | ------ |
| **start** | Number | 起始值           | 0      |
| **end**   | Number | 结束值（不包含） | 必需   |
| **step**  | Number | 步长             | 1      |

**示例：**

```python
import torch

# 从 0 到 9（不包含 10）
t1 = torch.arange(10)
print(f"arange(10): {t1}")  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 从 1 到 10，步长为 2
t2 = torch.arange(1, 11, 2)
print(f"arange(1, 11, 2): {t2}")  # tensor([1, 3, 5, 7, 9])

# 浮点数
t3 = torch.arange(1, 2.51, 0.5)
print(f"arange(1, 2.51, 0.5): {t3}")  # tensor([1.0000, 1.5000, 2.0000, 2.5000])

# 指定数据类型
t4 = torch.arange(0, 5, dtype=torch.float32)
print(f"arange with float32: {t4}")
```

###### 6. torch.linspace()

**功能：** 创建等分的一维张量（闭区间）。

**函数签名：**

```python
torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**重要：** 区间为 `[start, end]`，**闭区间**。

**示例：**

```python
import torch

# 从 3 到 10，分成 5 等份
t1 = torch.linspace(3, 10, steps=5)
print(f"linspace(3, 10, 5): {t1}")  # tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])

# 从 1 到 5，分成 3 等份
t2 = torch.linspace(1, 5, steps=3)
print(f"linspace(1, 5, 3): {t2}")  # tensor([1., 3., 5.])

# 对比 arange 和 linspace
print(f"arange(0, 5): {torch.arange(0, 5)}")      # [0, 1, 2, 3, 4] - 5个元素
print(f"linspace(0, 4, 5): {torch.linspace(0, 4, 5)}")  # [0., 1., 2., 3., 4.] - 5个元素，但包含4
```

**对比：arange vs linspace**

| 特性               | `arange()`              | `linspace()`          |
| ------------------ | ------------------------- | ----------------------- |
| **区间**     | `[start, end)` 左闭右开 | `[start, end]` 闭区间 |
| **参数**     | `start, end, step`      | `start, end, steps`   |
| **控制方式** | 通过步长控制              | 通过数量控制            |
| **适用场景** | 固定步长的序列            | 固定数量的等分序列      |

###### 7. torch.logspace()

**功能：** 创建对数等分的一维张量。

**函数签名：**

```python
torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**说明：** 创建从 `base^start` 到 `base^end` 的对数等分序列。

**示例：**

```python
import torch

# 从 10^0.1 到 10^1.0，分成 5 等份
t1 = torch.logspace(start=0.1, end=1.0, steps=5)
print(f"logspace(0.1, 1.0, 5): {t1}")
# tensor([ 1.2589,  2.1135,  3.5481,  5.9566, 10.0000])

# 使用底数为 2
t2 = torch.logspace(start=2, end=2, steps=1, base=2)
print(f"logspace(2, 2, 1, base=2): {t2}")  # tensor([4.]) = 2^2
```

##### 三、依概率分布创建（随机张量）

这类函数用于创建随机数张量，常用于模型初始化、数据增强等场景。

###### 1. torch.rand() / torch.rand_like()

**功能：** 在区间 `[0, 1)` 上生成均匀分布的随机数。

**函数签名：**

```python
torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.rand_like(input, dtype=None, layout=None, device=None, requires_grad=False)
```

**示例：**

```python
import torch

# 创建 3x4 的随机矩阵（值在 [0, 1) 之间）
t1 = torch.rand(3, 4)
print(f"rand(3, 4):\n{t1}")

# 创建 2x3x4 的随机张量
t2 = torch.rand(2, 3, 4)
print(f"rand(2, 3, 4) shape: {t2.shape}")

# 使用 rand_like
x = torch.zeros(2, 3)
t3 = torch.rand_like(x)
print(f"rand_like shape: {t3.shape}")
```

**常见用途：**

- 模型权重初始化
- 数据增强（随机噪声）
- 随机采样

###### 2. torch.randn() / torch.randn_like()

**功能：** 生成标准正态分布（均值为 0，标准差为 1）的随机数。

**函数签名：**

```python
torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.randn_like(input, dtype=None, layout=None, device=None, requires_grad=False)
```

**示例：**

```python
import torch

# 创建标准正态分布的 3x4 矩阵
t1 = torch.randn(3, 4)
print(f"randn(3, 4):\n{t1}")

# 常用于权重初始化
weight = torch.randn(10, 20) * 0.01  # 缩放以控制方差
```

**常见用途：**

- **神经网络权重初始化**（最常用）
- 生成随机噪声
- 模拟正态分布数据

###### 3. torch.randint() / torch.randint_like()

**功能：** 在区间 `[low, high)` 上生成整数的均匀分布。

**函数签名：**

```python
torch.randint(low=0, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.randint_like(input, low=0, high, dtype=None, layout=None, device=None, requires_grad=False)
```

**示例：**

```python
import torch

# 生成 [0, 10) 之间的随机整数，形状为 (3, 4)
t1 = torch.randint(0, 10, (3, 4))
print(f"randint(0, 10, (3, 4)):\n{t1}")

# 生成 [5, 20) 之间的随机整数
t2 = torch.randint(5, 20, (2, 3))
print(f"randint(5, 20, (2, 3)):\n{t2}")

# 生成单个随机整数（需要 size 参数）
t3 = torch.randint(0, 100, (1,)).item()  # .item() 获取标量值
print(f"单个随机整数: {t3}")
```

**常见用途：**

- 随机索引选择
- 数据采样
- 随机打乱

###### 4. torch.normal()

**功能：** 生成自定义均值和标准差的正态分布随机数。

**函数签名：**

```python
torch.normal(mean, std, size=None, *, out=None)
```

**重要：** `mean` 和 `std` 的组合有 4 种情况，行为不同：

| mean 类型        | std 类型         | 行为说明                                                 |
| ---------------- | ---------------- | -------------------------------------------------------- |
| **Tensor** | **Tensor** | 每个元素从不同的高斯分布采样，均值和标准差由对应位置确定 |
| **Tensor** | **标量**   | 每个元素采用相同的标准差，不同的均值                     |
| **标量**   | **Tensor** | 每个元素采用相同的均值，不同的标准差                     |
| **标量**   | **标量**   | 从一个高斯分布中生成大小为 `size` 的张量               |

**示例：**

```python
import torch

# 情况1：标量 mean 和 std，指定 size
t1 = torch.normal(mean=0, std=1, size=(3, 4))
print(f"normal(0, 1, (3, 4)):\n{t1}")

# 情况2：Tensor mean 和 std（对应位置）
mean = torch.tensor([1., 2., 3.])
std = torch.tensor([0.1, 0.2, 0.3])
t2 = torch.normal(mean, std)
print(f"normal with tensor mean/std: {t2}")
# 第0个元素从 N(1, 0.1) 采样，第1个从 N(2, 0.2) 采样，第2个从 N(3, 0.3) 采样

# 情况3：Tensor mean，标量 std
mean = torch.arange(1., 4.)
t3 = torch.normal(mean, std=0.5)
print(f"normal with tensor mean, scalar std: {t3}")

# 情况4：标量 mean，Tensor std
std = torch.tensor([0.1, 0.2, 0.3])
t4 = torch.normal(mean=0.0, std=std)
print(f"normal with scalar mean, tensor std: {t4}")
```

**常见用途：**

- 自定义权重初始化
- 添加高斯噪声
- 模拟特定分布的数据

###### 5. torch.randperm()

**功能：** 生成 `[0, n)` 的随机排列。

**函数签名：**

```python
torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
```

**示例：**

```python
import torch

# 生成 0 到 9 的随机排列
t1 = torch.randperm(10)
print(f"randperm(10): {t1}")  # 例如: tensor([3, 7, 1, 9, 0, 4, 8, 2, 6, 5])

# 常用于数据打乱
data = torch.arange(10)
indices = torch.randperm(10)
shuffled_data = data[indices]
print(f"原始数据: {data}")
print(f"打乱后: {shuffled_data}")
```

**常见用途：**

- 数据打乱（shuffle）
- 随机采样
- 索引随机化

###### 6. torch.bernoulli()

**功能：** 生成伯努利分布（0/1 二项分布）的随机数。

**函数签名：**

```python
torch.bernoulli(input, *, generator=None, out=None) → Tensor
```

**说明：** `input` 中的每个元素表示该位置为 1 的概率。

**示例：**

```python
import torch

# 每个位置以 0.5 的概率为 1
t1 = torch.bernoulli(torch.tensor([0.5, 0.1, 1.0]))
print(f"bernoulli([0.5, 0.1, 1.0]): {t1}")  # 例如: tensor([1., 0., 1.])

# 从概率矩阵生成
probs = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
t2 = torch.bernoulli(probs)
print(f"bernoulli from matrix:\n{t2}")
```

**常见用途：**

- Dropout 层实现
- 二分类采样
- 随机掩码生成

##### 常见使用场景总结

| 场景                           | 推荐函数                 | 示例                                               | 说明                     |
| ------------------------------ | ------------------------ | -------------------------------------------------- | ------------------------ |
| **模型权重初始化**       | `torch.randn()`        | `weight = torch.randn(10, 20) * 0.01`            | 标准正态分布，常用于权重 |
| **偏置初始化**           | `torch.zeros()`        | `bias = torch.zeros(10)`                         | 偏置通常初始化为0        |
| **批量创建（性能优化）** | `torch.empty()` + 填充 | `t = torch.empty(1000, 1000); t.fill_(0)`        | 最快，适合后续完全填充   |
| **数据打乱**             | `torch.randperm()`     | `indices = torch.randperm(len(data))`            | 生成随机排列索引         |
| **批量索引**             | `torch.randint()`      | `batch_idx = torch.randint(0, len(data), (32,))` | 随机采样索引             |
| **添加噪声**             | `torch.randn()`        | `noisy = data + torch.randn_like(data) * 0.1`    | 数据增强常用             |
| **单位矩阵**             | `torch.eye()`          | `I = torch.eye(3)`                               | 线性代数运算             |
| **等分序列**             | `torch.linspace()`     | `x = torch.linspace(0, 1, 100)`                  | 绘图、评估常用           |
| **临时缓冲区**           | `torch.empty()`        | `buffer = torch.empty(batch_size, hidden_dim)`   | 不关心初始值，后续会填充 |

##### 注意事项和最佳实践

1. **数据类型选择**

   - 默认浮点类型是 `torch.float32`
   - 整数类型默认是 `torch.int64`
   - 可以通过 `dtype` 参数指定
2. **设备选择**

   - 默认在 CPU 上创建
   - 创建后可以用 `.to(device)` 移动到 GPU/MPS
   - 或创建时直接指定 `device` 参数
3. **内存共享**

   - `torch.from_numpy()` 共享内存，修改会影响原数组
   - `torch.tensor()` 复制数据，互不影响
4. **随机种子**

   - 使用 `torch.manual_seed()` 设置随机种子以保证可复现性
   - 不同设备（CPU/CUDA/MPS）需要分别设置
5. **性能建议**

   - 大数组优先使用 `torch.from_numpy()`（零拷贝）
   - 批量创建时考虑使用 `torch.empty()` + 填充（更快）
   - GPU 操作时注意数据在正确的设备上

---

#### 2.1.3 张量的操作

张量提供了丰富的操作函数，包括拼接、切分、索引、形状变换等。这些操作是构建深度学习模型的基础。

| 函数名                   | 描述                                                                                                   | 主要参数                                                                                                                                                                          | 用法示例                                                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **cat**            | 将多个张量拼接在一起，例如多个特征图的融合可用。                                                       | `tensors (sequence)`: 要拼接的张量序列 `<br>dim (int)`: 拼接的维度，默认0                                                                                                     | `torch.cat([t1, t2, t3], dim=0)<br>``torch.cat([t1, t2], dim=1)`                                                                                                                                         |
| **concat**         | 同cat, 是cat()的别名。                                                                                 | 同cat                                                                                                                                                                             | `torch.concat([t1, t2], dim=0)`                                                                                                                                                                          |
| **chunk**          | 将tensor在某个维度上分成n份。                                                                          | `input (Tensor)`: 输入张量 `<br>chunks (int)`: 分割的份数 `<br>dim (int)`: 分割的维度，默认0                                                                                | `torch.chunk(t, chunks=3, dim=0)<br>`返回3个张量的元组                                                                                                                                                   |
| **stack**          | 在新的轴上拼接张量。与hstack\vstack不同，它是新增一个轴。默认从第0个轴插入新轴。                       | `tensors (sequence)`: 要堆叠的张量序列 `<br>dim (int)`: 插入新轴的维度，默认0                                                                                                 | `torch.stack([t1, t2, t3], dim=0)`, 固定列, 遍历每一行, 将两个张量在该列的数据堆叠在一起 `<br>torch.stack([t1, t2], dim=1)`, 固定行（第0维），遍历每一行; 对于每一行，将两个张量在该行的数据堆叠在一起 |
| **hstack**         | 水平堆叠张量。即第二个维度上增加，等同于torch.column_stack。                                           | `tensors (sequence)`: 要堆叠的张量序列                                                                                                                                          | `torch.hstack([t1, t2])<br>`要求除dim=1外其他维度相同                                                                                                                                                    |
| **vstack**         | 垂直堆叠。按行堆叠张量，即第一个维度上增加。                                                           | `tensors (sequence)`: 要堆叠的张量序列                                                                                                                                          | `torch.vstack([t1, t2])<br>`要求除dim=0外其他维度相同                                                                                                                                                    |
| **column_stack**   | 水平堆叠张量。即第二个维度上增加，等同于torch.hstack。                                                 | `tensors (sequence)`: 要堆叠的张量序列                                                                                                                                          | `torch.column_stack([t1, t2])`                                                                                                                                                                           |
| **row_stack**      | 按行堆叠张量。即第一个维度上增加，等同于torch.vstack。                                                 | `tensors (sequence)`: 要堆叠的张量序列                                                                                                                                          | `torch.row_stack([t1, t2])`                                                                                                                                                                              |
| **dstack**         | 沿第三个轴进行逐像素（depthwise）拼接。                                                                | `tensors (sequence)`: 要堆叠的张量序列                                                                                                                                          | `torch.dstack([t1, t2])<br>`在dim=2上堆叠                                                                                                                                                                |
| **split**          | 按给定的大小切分出多个张量。                                                                           | `tensor (Tensor)`: 输入张量 `<br>split_size_or_sections (int/list)`: 每份大小或分割点列表 `<br>dim (int)`: 分割维度，默认0                                                  | `torch.split(t, split_size_or_sections=2, dim=0)<br>``torch.split(t, [1, 4], dim=0)`                                                                                                                     |
| **tensor_split**   | 切分张量，核心看indices_or_sections变量如何设置。                                                      | `tensor (Tensor)`: 输入张量 `<br>indices_or_sections (int/list)`: 分割份数或索引列表 `<br>dim (int)`: 分割维度，默认0                                                       | `torch.tensor_split(t, 3, dim=0)<br>``torch.tensor_split(t, [2, 5], dim=0)`                                                                                                                              |
| **hsplit**         | 类似numpy.hsplit()，将张量按列进行切分。若传入整数，则按等分划分。若传入list，则按list中元素进行索引。 | `input (Tensor)`: 输入张量 `<br>indices_or_sections (int/list)`: 分割份数或索引列表                                                                                           | `torch.hsplit(t, 2)<br>``torch.hsplit(t, [2, 3])`                                                                                                                                                        |
| **vsplit**         | 垂直切分。将张量按行进行切分。                                                                         | `input (Tensor)`: 输入张量 `<br>indices_or_sections (int/list)`: 分割份数或索引列表                                                                                           | `torch.vsplit(t, 2)<br>``torch.vsplit(t, [1, 3])`                                                                                                                                                        |
| **dsplit**         | 类似numpy.dsplit()，将张量按索引或指定的份数进行切分。                                                 | `input (Tensor)`: 输入张量 `<br>indices_or_sections (int/list)`: 分割份数或索引列表                                                                                           | `torch.dsplit(t, 2)<br>`在dim=2上切分                                                                                                                                                                    |
| **gather**         | 高级索引方法，目标检测中常用于索引bbox。在指定的轴上，根据给定的index进行索引。                        | `input (Tensor)`: 输入张量 `<br>dim (int)`: 索引的维度 `<br>index (LongTensor)`: 索引张量                                                                                   | `torch.gather(input, dim=1, index=index)<br>`index形状需与input相同（除dim维度）                                                                                                                         |
| **index_select**   | 在指定的维度上，按索引进行选择数据，然后拼接成新张量。新张量的指定维度上长度是index的长度。            | `input (Tensor)`: 输入张量 `<br>dim (int)`: 选择的维度 `<br>index (LongTensor)`: 索引张量（1D）                                                                             | `torch.index_select(t, dim=0, index=torch.tensor([0, 2]))`                                                                                                                                               |
| **masked_select**  | 根据mask（0/1, False/True 形式的mask）索引数据，返回1-D张量。                                          | `input (Tensor)`: 输入张量 `<br>mask (BoolTensor)`: 布尔掩码，形状需与input相同                                                                                               | `torch.masked_select(t, mask)<br>`返回1D张量                                                                                                                                                             |
| **take**           | 取张量中的某些元素，返回的是1D张量。                                                                   | `input (Tensor)`: 输入张量 `<br>index (LongTensor)`: 索引张量（1D）                                                                                                           | `torch.take(t, torch.tensor([0, 2, 5]))<br>`将张量展平后按索引取值                                                                                                                                       |
| **take_along_dim** | 取张量中的某些元素，返回的张量与index维度保持一致。可搭配torch.argmax和torch.argsort使用。             | `input (Tensor)`: 输入张量 `<br>indices (LongTensor)`: 索引张量 `<br>dim (int)`: 操作的维度                                                                                 | `torch.take_along_dim(t, indices, dim=1)<br>`保持index的形状                                                                                                                                             |
| **nonzero**        | 返回非零元素的index。                                                                                  | `input (Tensor)`: 输入张量 `<br>as_tuple (bool)`: 是否返回元组形式，默认False                                                                                                 | `torch.nonzero(t)<br>``torch.nonzero(t, as_tuple=True)`                                                                                                                                                  |
| **where**          | 根据一个是非条件，选择x的元素还是y的元素，拼接成新张量。                                               | `condition (BoolTensor)`: 条件张量 `<br>x (Tensor)`: True时选择的元素 `<br>y (Tensor)`: False时选择的元素                                                                   | `torch.where(condition, x, y)<br>``torch.where(condition)` 返回满足条件的索引                                                                                                                            |
| **scatter**        | 将src中数据根据index中的索引按照dim的方向填进input中。index告诉哪些位置需要变，src告诉要变的值是什么。 | `input (Tensor)`: 输入张量 `<br>dim (int)`: 操作的维度 `<br>index (LongTensor)`: 索引张量 `<br>src (Tensor)`: 源数据 `<br>reduce (str)`: 归约方式，可选'multiply'/'add' | `t.scatter_(dim=1, index=index, src=src)<br>`原地操作，返回修改后的t                                                                                                                                     |
| **scatter_add**    | 同scatter一样，对input进行元素修改，这里是 +=，而scatter是直接替换。                                   | `input (Tensor)`: 输入张量 `<br>dim (int)`: 操作的维度 `<br>index (LongTensor)`: 索引张量 `<br>src (Tensor)`: 源数据                                                      | `t.scatter_add_(dim=1, index=index, src=src)<br>`执行加法操作                                                                                                                                            |
| **reshape**        | 变换形状。返回具有相同数据但不同形状的新张量（可能复制）。                                             | `input (Tensor)`: 输入张量 `<br>shape (tuple/int...)`: 目标形状                                                                                                               | `torch.reshape(t, (2, 3))<br>``t.reshape(2, 3)`                                                                                                                                                          |
| **view**           | 变换形状（要求内存连续）。返回共享内存的视图，不复制数据。                                             | `*shape (int...)`: 目标形状，-1表示自动推断                                                                                                                                     | `t.view(2, 3)<br>``t.view(-1, 3)` 自动推断第一维                                                                                                                                                         |
| **flatten**        | 展平张量。将多维张量展平为1D或部分展平。                                                               | `start_dim (int)`: 开始维度，默认0 `<br>end_dim (int)`: 结束维度，默认-1                                                                                                      | `t.flatten()<br>``t.flatten(1)` 从第1维开始展平                                                                                                                                                          |
| **permute**        | 交换轴。重新排列张量的维度。                                                                           | `input (Tensor)`: 输入张量 `<br>*dims (int...)`: 新的维度顺序                                                                                                                 | `torch.permute(t, 2, 0, 1)<br>``t.permute(2, 0, 1)`                                                                                                                                                      |
| **transpose**      | 交换轴。交换两个指定的维度。                                                                           | `input (Tensor)`: 输入张量 `<br>dim0 (int)`: 第一个维度 `<br>dim1 (int)`: 第二个维度                                                                                        | `torch.transpose(t, 0, 1)<br>``t.transpose(0, 1)`                                                                                                                                                        |
| **swapaxes**       | Alias for torch.transpose()。交换轴。                                                                  | 同transpose                                                                                                                                                                       | `torch.swapaxes(t, 0, 1)`                                                                                                                                                                                |
| **swapdims**       | Alias for torch.transpose()。交换轴。                                                                  | 同transpose                                                                                                                                                                       | `torch.swapdims(t, 0, 1)`                                                                                                                                                                                |
| **t**              | 转置。仅适用于2D张量，等价于transpose(0, 1)。                                                          | `input (Tensor)`: 输入张量（2D）                                                                                                                                                | `torch.t(t)<br>``t.t()`                                                                                                                                                                                  |
| **movedim**        | 移动轴。将指定的维度移动到新位置。                                                                     | `input (Tensor)`: 输入张量 `<br>source (int/tuple)`: 源维度 `<br>destination (int/tuple)`: 目标位置                                                                         | `torch.movedim(t, 1, 0)<br>``torch.movedim(t, (0, 1), (1, 0))`                                                                                                                                           |
| **moveaxis**       | 同movedim。Alias for torch.movedim()。                                                                 | 同movedim                                                                                                                                                                         | `torch.moveaxis(t, 1, 0)`                                                                                                                                                                                |
| **narrow**         | 在指定轴上，设置起始和长度进行索引。                                                                   | `input (Tensor)`: 输入张量 `<br>dim (int)`: 操作的维度 `<br>start (int)`: 起始位置 `<br>length (int)`: 长度                                                               | `torch.narrow(t, dim=0, start=0, length=2)<br>`等价于 `t[0:2, ...]`                                                                                                                                    |
| **squeeze**        | 移除张量为1的轴。                                                                                      | `input (Tensor)`: 输入张量 `<br>dim (int, optional)`: 指定要移除的维度，默认移除所有size=1的维度                                                                              | `torch.squeeze(t)<br>``torch.squeeze(t, dim=0)`                                                                                                                                                          |
| **unsqueeze**      | 增加一个轴，常用于匹配数据维度。                                                                       | `input (Tensor)`: 输入张量 `<br>dim (int)`: 插入新轴的位置                                                                                                                    | `torch.unsqueeze(t, dim=0)<br>``t.unsqueeze(0)`                                                                                                                                                          |
| **unbind**         | 移除张量的某个轴，并返回一串张量。                                                                     | `input (Tensor)`: 输入张量 `<br>dim (int)`: 要移除的维度，默认0                                                                                                               | `torch.unbind(t, dim=0)<br>`返回元组，如 `([1], [2], [3])`                                                                                                                                             |
| **tile**           | 将张量重复X遍，X遍表示可按多个维度进行重复。                                                           | `input (Tensor)`: 输入张量 `<br>dims (tuple/int...)`: 每个维度重复的次数                                                                                                      | `torch.tile(t, (2, 2))<br>``torch.tile(t, 3)`                                                                                                                                                            |
| **conj**           | 返回共轭复数。                                                                                         | `input (Tensor)`: 输入张量                                                                                                                                                      | `torch.conj(t)<br>`仅对复数张量有效                                                                                                                                                                      |

##### view() vs reshape() vs flatten() 详解

**重要概念：内存连续性（Contiguous Memory）**

在理解 `view()` 和 `reshape()` 之前，需要先了解 PyTorch 中的内存连续性概念。

**什么是内存连续？**

张量在内存中是以一维数组的形式存储的。"内存连续"指的是张量元素在内存中的存储顺序与逻辑顺序一致。

**C-contiguous（行优先）vs F-contiguous（列优先）：**

- **C-contiguous（C风格，行优先）**：最后一个维度变化最快，这是 PyTorch 和 NumPy 的默认方式
- **F-contiguous（Fortran风格，列优先）**：第一个维度变化最快

**可视化理解：**

```python
import torch

# 2x3 矩阵的逻辑视图
# [[1, 2, 3],
#  [4, 5, 6]]

# C-contiguous 内存布局（行优先）
# [1, 2, 3, 4, 5, 6]
# 先存储第一行，再存储第二行

# F-contiguous 内存布局（列优先）
# [1, 4, 2, 5, 3, 6]
# 先存储第一列，再存储第二列
```

**详细示例：**

```python
import torch

# 创建张量（默认 C-contiguous）
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(f"is_contiguous: {x.is_contiguous()}")  # True
print(f"stride: {x.stride()}")  # (3, 1) - 行跨度3，列跨度1

# 内存中的实际存储：[1, 2, 3, 4, 5, 6]
# x[0,0] → 内存位置 0
# x[0,1] → 内存位置 1 (跨度 1)
# x[1,0] → 内存位置 3 (跨度 3)

# 转置后内存不连续
x_t = x.transpose(0, 1)  # [[1, 4], [2, 5], [3, 6]]
print(f"转置后 is_contiguous: {x_t.is_contiguous()}")  # False
print(f"转置后 stride: {x_t.stride()}")  # (1, 3) - 行跨度1，列跨度3

# 内存中的存储仍然是：[1, 2, 3, 4, 5, 6]
# 但逻辑顺序变了：
# x_t[0,0] → 内存位置 0 (值 1)
# x_t[0,1] → 内存位置 3 (值 4, 跨度 3)
# x_t[1,0] → 内存位置 1 (值 2, 跨度 1)
```

**stride（步幅）的含义：**

stride 表示在每个维度上移动一个位置需要跨越多少个元素。

```python
import torch

x = torch.randn(2, 3, 4)
print(f"shape: {x.shape}")    # [2, 3, 4]
print(f"stride: {x.stride()}")  # (12, 4, 1)

# 解释：
# - 在第0维移动1步，需要跨越 12 个元素（3*4）
# - 在第1维移动1步，需要跨越 4 个元素
# - 在第2维移动1步，需要跨越 1 个元素

# 访问 x[i, j, k] 的内存位置：
# offset = i * 12 + j * 4 + k * 1
```

**三个函数的对比：**

| 函数          | 是否要求连续内存 | 是否总是返回视图        | 是否可能复制数据 | 使用建议                     |
| ------------- | ---------------- | ----------------------- | ---------------- | ---------------------------- |
| `view()`    | ✅ 必须连续      | ✅ 总是视图（共享内存） | ❌ 从不复制      | 确定内存连续时使用，性能最优 |
| `reshape()` | ❌ 不要求        | ⚠️  尽量返回视图      | ✅ 必要时复制    | 不确定时使用（更安全）       |
| `flatten()` | ❌ 不要求        | ⚠️  尽量返回视图      | ✅ 必要时复制    | 专门用于展平为1D             |

**详细示例：**

```python
import torch

# 示例1：view() - 要求内存连续
x = torch.randn(2, 3, 4)
print(f"原始张量 shape: {x.shape}")  # [2, 3, 4]

# view() 成功：内存连续
y = x.view(2, 12)
print(f"view() 成功: {y.shape}")  # [2, 12]

# 修改 y 会影响 x（共享内存）
y[0, 0] = 999
print(f"修改 y 后，x[0,0,0] = {x[0,0,0]}")  # 999

# 转置后内存不连续
x_t = x.transpose(0, 1)
print(f"转置后 is_contiguous: {x_t.is_contiguous()}")  # False

# view() 失败：内存不连续
try:
    y = x_t.view(3, 8)
except RuntimeError as e:
    print(f"view() 错误: {e}")
    # RuntimeError: view size is not compatible with input tensor's size and stride

# 解决方案1：使用 reshape()（推荐）
y = x_t.reshape(3, 8)  # 成功！会自动复制数据
print(f"reshape() 成功: {y.shape}")  # [3, 8]

# 解决方案2：先 contiguous() 再 view()
y = x_t.contiguous().view(3, 8)  # 成功！
print(f"contiguous().view() 成功: {y.shape}")  # [3, 8]
```

**reshape() 如何尽量返回视图？**

`reshape()` 的智能之处在于：它会先尝试返回视图（不复制数据），只有在无法返回视图时才复制数据。

**什么时候 reshape() 能返回视图？**

当满足以下条件时，`reshape()` 可以返回视图：

1. **张量内存是连续的（C-contiguous）**
2. **新形状可以通过调整 stride 实现**

```python
import torch

# 情况1：连续张量 reshape - 返回视图
x = torch.randn(2, 3, 4)
y = x.reshape(2, 12)
print(f"x 是否连续: {x.is_contiguous()}")  # True
print(f"y 是否连续: {y.is_contiguous()}")  # True
print(f"共享内存: {x.data_ptr() == y.data_ptr()}")  # True - 返回视图！

# 验证：修改 y 会影响 x
y[0, 0] = 999
print(f"x[0,0,0] = {x[0,0,0]}")  # 999

# 情况2：非连续张量 reshape - 必须复制
x = torch.randn(2, 3, 4)
x_t = x.transpose(0, 1)  # 转置后不连续
print(f"x_t 是否连续: {x_t.is_contiguous()}")  # False

y = x_t.reshape(3, 8)
print(f"y 是否连续: {y.is_contiguous()}")  # True
print(f"共享内存: {x_t.data_ptr() == y.data_ptr()}")  # False - 复制了数据！

# 验证：修改 y 不会影响 x_t
y[0, 0] = 999
print(f"x_t[0,0,0] = {x_t[0,0,0]}")  # 不是 999
```

**性能对比：**

```python
import torch
import time

x = torch.randn(1000, 1000, 100)

# 测试1：连续张量 reshape（返回视图，非常快）
start = time.time()
for _ in range(1000):
    y = x.reshape(1000, 100000)
print(f"连续 reshape: {time.time() - start:.4f}s")  # ~0.0001s

# 测试2：非连续张量 reshape（复制数据，较慢）
x_t = x.transpose(0, 1)
start = time.time()
for _ in range(1000):
    y = x_t.reshape(1000, 100000)
print(f"非连续 reshape: {time.time() - start:.4f}s")  # ~0.5s
```

**最佳实践：**

- ✅ 优先使用 `reshape()`：安全且智能
- ✅ 如果确定内存连续，用 `view()` 可以强制返回视图（更明确）
- ⚠️  避免频繁对非连续张量 reshape（会触发大量复制）
- ✅ 如果需要多次 reshape 非连续张量，先调用 `.contiguous()` 一次

```python
# ❌ 不好：每次 reshape 都复制
x_t = x.transpose(0, 1)
for _ in range(100):
    y = x_t.reshape(new_shape)  # 每次都复制！

# ✅ 好：只复制一次
x_t = x.transpose(0, 1).contiguous()  # 一次性复制
for _ in range(100):
    y = x_t.reshape(new_shape)  # 返回视图，快！
```

**示例2：flatten() - 展平张量**

```python
import torch

x = torch.randn(2, 3, 4)
print(f"原始 shape: {x.shape}")  # [2, 3, 4]

# flatten() - 展平为1D
y1 = x.flatten()
print(f"flatten(): {y1.shape}")  # [24]

# flatten(start_dim) - 从指定维度开始展平
y2 = x.flatten(start_dim=1)
print(f"flatten(1): {y2.shape}")  # [2, 12]

# flatten(start_dim, end_dim) - 展平指定范围
y3 = x.flatten(start_dim=0, end_dim=1)
print(f"flatten(0, 1): {y3.shape}")  # [6, 4]
```

**示例3：squeeze() 和 unsqueeze() 详解**

`squeeze()` 和 `unsqueeze()` 是用于调整张量维度的重要函数，在深度学习中经常用于维度匹配。

##### squeeze() - 压缩维度

**作用：** 移除所有大小为 1 的维度，或移除指定的大小为 1 的维度。

**语法：**

```python
torch.squeeze(input, dim=None)
# 或
input.squeeze(dim=None)
```

**参数：**

- `dim` (int, optional): 如果指定，只移除该维度（前提是该维度大小为 1）
- 如果不指定 `dim`，移除所有大小为 1 的维度

**示例：**

```python
import torch

# 创建一个包含多个 size=1 维度的张量
x = torch.randn(1, 3, 1, 4, 1)
print(f"原始 shape: {x.shape}")  # [1, 3, 1, 4, 1]

# 情况1：不指定 dim - 移除所有 size=1 的维度
y1 = x.squeeze()
print(f"squeeze(): {y1.shape}")  # [3, 4]

# 情况2：指定 dim=0 - 只移除第0维（size=1）
y2 = x.squeeze(dim=0)
print(f"squeeze(0): {y2.shape}")  # [3, 1, 4, 1]

# 情况3：指定 dim=2 - 只移除第2维（size=1）
y3 = x.squeeze(dim=2)
print(f"squeeze(2): {y3.shape}")  # [1, 3, 4, 1]

# 情况4：指定的维度 size≠1 - 不会报错，返回原张量
y4 = x.squeeze(dim=1)  # 第1维 size=3，不是1
print(f"squeeze(1): {y4.shape}")  # [1, 3, 1, 4, 1] - 没有变化
```

**常见使用场景：**

```python
# 场景1：批量大小为1时，移除 batch 维度
batch_data = torch.randn(1, 3, 224, 224)  # [1, C, H, W]
single_image = batch_data.squeeze(0)       # [C, H, W]

# 场景2：移除多余的维度（如某些操作产生的单维度）
x = torch.randn(10, 1, 5)
y = x.squeeze(1)  # [10, 5]

# 场景3：处理标签维度
labels = torch.tensor([[1], [2], [3]])  # [3, 1]
labels = labels.squeeze(1)              # [3]
```

##### unsqueeze() - 扩展维度

**作用：** 在指定位置插入一个大小为 1 的新维度。

**语法：**

```python
torch.unsqueeze(input, dim)
# 或
input.unsqueeze(dim)
```

**参数：**

- `dim` (int): 插入新维度的位置（必须指定）
  - 正数：从前往后数（0 表示最前面）
  - 负数：从后往前数（-1 表示最后面）

**示例：**

```python
import torch

x = torch.randn(3, 4)
print(f"原始 shape: {x.shape}")  # [3, 4]

# 在不同位置插入维度
y1 = x.unsqueeze(0)   # 在最前面插入
print(f"unsqueeze(0): {y1.shape}")   # [1, 3, 4]

y2 = x.unsqueeze(1)   # 在中间插入
print(f"unsqueeze(1): {y2.shape}")   # [3, 1, 4]

y3 = x.unsqueeze(2)   # 在最后插入
print(f"unsqueeze(2): {y3.shape}")   # [3, 4, 1]

y4 = x.unsqueeze(-1)  # 负数索引：在最后插入
print(f"unsqueeze(-1): {y4.shape}")  # [3, 4, 1]

y5 = x.unsqueeze(-2)  # 负数索引：在倒数第二个位置插入
print(f"unsqueeze(-2): {y5.shape}")  # [3, 1, 4]
```

**常见使用场景：**

```python
import torch

# 场景1：添加 batch 维度
image = torch.randn(3, 224, 224)      # [C, H, W]
batch_image = image.unsqueeze(0)      # [1, C, H, W] - 添加 batch 维度

# 场景2：广播运算（Broadcasting）
a = torch.randn(4, 3)     # [4, 3]
b = torch.randn(3)        # [3]
# 直接相加会报错或广播不符合预期

# 方法1：在 b 的前面添加维度
b = b.unsqueeze(0)        # [1, 3]
c = a + b                 # [4, 3] + [1, 3] → [4, 3]

# 方法2：在 b 的后面添加维度（用于列广播）
a = torch.randn(4, 3)
b = torch.randn(4)
b = b.unsqueeze(1)        # [4, 1]
c = a + b                 # [4, 3] + [4, 1] → [4, 3]

# 场景3：scatter/gather 操作需要匹配维度
labels = torch.tensor([1, 2, 0])      # [3]
labels = labels.unsqueeze(1)          # [3, 1] - scatter 需要这个形状

# 场景4：卷积操作需要 4D 输入
x = torch.randn(3, 224, 224)          # [C, H, W]
x = x.unsqueeze(0)                    # [1, C, H, W] - 添加 batch 维度
output = conv2d(x)
```

##### squeeze() 和 unsqueeze() 的互逆关系

```python
import torch

x = torch.randn(3, 4)
print(f"原始: {x.shape}")  # [3, 4]

# unsqueeze 后再 squeeze，恢复原状
y = x.unsqueeze(0)         # [1, 3, 4]
z = y.squeeze(0)           # [3, 4]
print(f"恢复: {z.shape}")  # [3, 4]

# 验证：是否相等
print(f"相等: {torch.equal(x, z)}")  # True
```

##### 常见错误和注意事项

```python
import torch

# 错误1：squeeze 指定的维度 size 不为 1
x = torch.randn(2, 3, 4)
# y = x.squeeze(1)  # 不会报错，但没有效果（第1维 size=3）
y = x.squeeze(1)
print(f"shape: {y.shape}")  # [2, 3, 4] - 没有变化

# 错误2：unsqueeze 的 dim 超出范围
x = torch.randn(3, 4)  # 2维张量
# y = x.unsqueeze(3)   # 错误！dim 范围应该是 [-3, 2]
# 正确范围：[-len(x.shape)-1, len(x.shape)]

# 正确用法
y1 = x.unsqueeze(0)   # OK: dim=0
y2 = x.unsqueeze(1)   # OK: dim=1
y3 = x.unsqueeze(2)   # OK: dim=2
y4 = x.unsqueeze(-1)  # OK: dim=-1
y5 = x.unsqueeze(-2)  # OK: dim=-2
# y6 = x.unsqueeze(-3) # OK: dim=-3
```

**最佳实践：**

- ✅ 使用 `unsqueeze()` 而不是 `view()` 来添加维度（更清晰）
- ✅ 使用 `squeeze()` 移除已知的单维度（如 batch=1）
- ⚠️  小心使用 `squeeze()` 不带参数：可能移除意外的维度
- ✅ 优先使用 `squeeze(dim)` 指定维度（更安全）

```python
# ❌ 不推荐：可能移除意外维度
x = torch.randn(1, 3, 1, 4)
y = x.squeeze()  # [3, 4] - 移除了两个维度！

# ✅ 推荐：明确指定维度
y = x.squeeze(0).squeeze(1)  # 或 x.squeeze(0).squeeze(2)
```

**示例4：-1 参数（自动推断维度）**

```python
import torch

x = torch.randn(2, 3, 4)  # 24 个元素

# 使用 -1 自动推断维度
y1 = x.view(-1)          # [24]
y2 = x.view(-1, 4)       # [6, 4]
y3 = x.view(2, -1)       # [2, 12]
y4 = x.view(2, 3, -1)    # [2, 3, 4]

print(f"view(-1): {y1.shape}")
print(f"view(-1, 4): {y2.shape}")
print(f"view(2, -1): {y2.shape}")

# 错误：最多只能有一个 -1
# y = x.view(-1, -1)  # RuntimeError
```

**性能对比：**

```python
import torch
import time

x = torch.randn(1000, 1000)

# view() - 最快（不复制数据）
start = time.time()
for _ in range(10000):
    y = x.view(-1)
print(f"view(): {time.time() - start:.4f}s")

# reshape() - 稍慢（可能复制）
start = time.time()
for _ in range(10000):
    y = x.reshape(-1)
print(f"reshape(): {time.time() - start:.4f}s")

# flatten() - 与 reshape() 类似
start = time.time()
for _ in range(10000):
    y = x.flatten()
print(f"flatten(): {time.time() - start:.4f}s")
```

**使用建议：**

1. **优先使用 `reshape()`**：更安全，适用范围广
2. **性能关键时使用 `view()`**：确保内存连续，避免意外复制
3. **展平为1D时使用 `flatten()`**：语义更清晰
4. **检查内存连续性**：使用 `tensor.is_contiguous()`
5. **强制连续**：使用 `tensor.contiguous()` 后再 `view()`

**常见陷阱：**

```python
import torch

# 陷阱1：转置后直接 view()
x = torch.randn(2, 3)
x_t = x.t()  # 转置
# y = x_t.view(6)  # 错误！

# 正确做法
y = x_t.reshape(6)  # 或 x_t.contiguous().view(6)

# 陷阱2：以为 reshape() 总是复制
x = torch.randn(2, 3)
y = x.reshape(6)
y[0] = 999
print(x[0, 0])  # 999 - 也被修改了！reshape() 尽量返回视图

# 陷阱3：忘记 -1 只能用一次
x = torch.randn(2, 3, 4)
# y = x.view(-1, -1)  # 错误！
y = x.view(-1, 4)  # 正确
```

##### torch.stack 函数详解

为了更清晰地理解 `torch.stack` 函数的工作原理，特别是不同 `dim` 参数的效果，这里提供详细的解释和 3×4 张量的示例。

`dim`是选择器, dim 所指定的那个维度的下标, 决定张量来自于哪一个元素
eg.
`s = stack(a,b,dim=0)`, a,b 都是2*3的tensor,
那么 s[0] 全部来自于a, s[0][0] 表示a[0][:], s[1][i][j] = b[i][j]

`s = stack(a,b,dim=1)`,
s[i][0][j] = a[i][j]
s[i][1][j] = b[i][j]

###### 基础概念

`torch.stack` 函数在**新的维度**上拼接张量，与 `torch.cat` 在**已有维度**上拼接不同。关键理解点：

- **输入**: 多个形状相同的张量
- **输出**: 新增一个维度，结果维度数 = 输入维度数 + 1
- **dim 参数**: 指定在哪个位置插入新维度

###### 使用 3×4 张量示例

使用更大的维度（3×4）来清晰展示不同 `dim` 的效果：

```python
import torch

# 两个 3×4 的张量（3行4列）
a = torch.tensor([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])  # shape: [3, 4]

b = torch.tensor([[101, 102, 103, 104], 
                  [105, 106, 107, 108], 
                  [109, 110, 111, 112]])  # shape: [3, 4]
```

###### 不同 dim 值的结果对比

| 操作                    | 结果形状      | 形状分析                   | 关键理解        |
| ----------------------- | ------------- | -------------------------- | --------------- |
| `stack([a,b], dim=0)` | `[2, 3, 4]` | `[堆叠数量, 行数, 列数]` | 第0维是堆叠数量 |
| `stack([a,b], dim=1)` | `[3, 2, 4]` | `[行数, 堆叠数量, 列数]` | 第1维是堆叠数量 |
| `stack([a,b], dim=2)` | `[3, 4, 2]` | `[行数, 列数, 堆叠数量]` | 第2维是堆叠数量 |

###### 详细分析每个 dim

**1. stack dim=0:**

```python
result = torch.stack([a, b], dim=0)
# 结果形状: [2, 3, 4]
# - result[0] = a (第0个张量)
# - result[1] = b (第1个张量)
```

- **理解**: 将整个张量作为整体堆叠
- **访问方式**: `result[哪个张量, 哪一行, 哪一列]`
- **特点**: 第0维表示"哪个张量"

**2. stack dim=1:**

```python
result = torch.stack([a, b], dim=1)
# 结果形状: [3, 2, 4]
# - result[:, 0] = a (第0个张量的所有行)
# - result[:, 1] = b (第1个张量的所有行)
```

- **理解**: 按行分别堆叠
- **访问方式**: `result[哪一行, 哪个张量, 哪一列]`
- **特点**: 第1维表示"哪个张量"，数据按行组织

**3. stack dim=2:**

```python
result = torch.stack([a, b], dim=2)
# 结果形状: [3, 4, 2]
# - result[:, :, 0] = a (第0个张量的所有元素)
# - result[:, :, 1] = b (第1个张量的所有元素)
```

- **理解**: 按元素分别堆叠
- **访问方式**: `result[哪一行, 哪一列, 哪个张量]`
- **特点**: 第2维表示"哪个张量"，数据按元素组织

###### 通用规则

- `stack(tensors, dim=n)` 在结果的第 `n` 维插入"堆叠数量"
- 结果形状：`input_shape[:n] + [len(tensors)] + input_shape[n:]`
- 新增维度的位置 = `dim` 参数值

###### 记忆技巧

记住这个口诀："**在第几维插入，第几维就是堆叠数量**"

- `dim=0`: 在最外层插入 → `[2, 原始形状...]`
- `dim=1`: 在第二层插入 → `[原始形状[0], 2, 原始形状[1:]...]`
- `dim=2`: 在第三层插入 → `[原始形状[0], 原始形状[1], 2, ...]`

###### 与 cat 的区别

1. **维度变化**: stack 增加维度，cat 不增加维度
2. **操作方式**: stack 在新维度上堆叠，cat 在指定维度上拼接
3. **形状**: stack 结果是 `(N+1)` 维，cat 结果是 `N` 维（N 是输入维度）

##### scatter 函数详解

`scatter` 函数是 PyTorch 中一个功能强大但较难理解的函数，常用于根据索引将源张量的值填充到目标张量的指定位置。

###### 函数签名

```python
torch.scatter(input, dim, index, src, reduce=None) → Tensor
# 或使用原地操作版本
tensor.scatter_(dim, index, src, reduce=None) → Tensor
```

###### 参数说明

| 参数             | 类型          | 说明                                                                                                                                                                                                                         |
| ---------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **input**  | Tensor        | 目标张量，数据将被填充到这个张量中                                                                                                                                                                                           |
| **dim**    | int           | 操作的维度，指定在哪个维度上进行 scatter 操作。`<br>`对于2D张量：`dim=0`表示行维度，`dim=1`表示列维度 `<br>`对于3D张量：`dim=0`表示批次，`dim=1`表示高度，`dim=2`表示宽度 `<br>`详见下方"dim 参数的含义"表格 |
| **index**  | LongTensor    | 索引张量，指定在 `dim` 维度上的哪些位置需要被填充。形状必须与 `input` 相同（除了 `dim` 维度可以不同）                                                                                                                  |
| **src**    | Tensor        | 源张量，提供要填充的值。可以是标量、与 `input` 同形状的张量，或可广播到 `input` 形状的张量                                                                                                                               |
| **reduce** | str, optional | 归约方式，可选值：`None`（默认，直接替换）、`'add'`（相加）、`'multiply'`（相乘）                                                                                                                                      |

###### 核心理解

**dim 参数的含义：**

`dim` 参数指定了在哪个维度上进行 scatter 操作。理解不同 `dim` 值对应的维度是掌握 `scatter` 函数的关键：

| dim 值          | 维度名称       | 操作方向                   | 说明                          | 示例（对于形状 [B, H, W] 的张量）                             |
| --------------- | -------------- | -------------------------- | ----------------------------- | ------------------------------------------------------------- |
| **dim=0** | 第0维/批次维度 | 在**行维度**上操作   | 固定其他维度，改变第0维的位置 | 对于形状 `[3, 4]`：固定列，改变行位置                       |
| **dim=1** | 第1维/高度维度 | 在**列维度**上操作   | 固定其他维度，改变第1维的位置 | 对于形状 `[3, 4]`：固定行，改变列位置                       |
| **dim=2** | 第2维/宽度维度 | 在**深度维度**上操作 | 固定其他维度，改变第2维的位置 | 对于形状 `[B, H, W]`：固定批次和高度，改变宽度位置          |
| **dim=3** | 第3维/通道维度 | 在**通道维度**上操作 | 固定其他维度，改变第3维的位置 | 对于形状 `[B, C, H, W]`：固定批次、高度、宽度，改变通道位置 |

**记忆技巧：**

- 对于2D张量 `[行, 列]`：
  - `dim=0` → 操作**行**（垂直方向）
  - `dim=1` → 操作**列**（水平方向）
- 对于3D张量 `[批次, 高度, 宽度]`：
  - `dim=0` → 操作**批次**维度
  - `dim=1` → 操作**高度**维度（行）
  - `dim=2` → 操作**宽度**维度（列）
- 对于4D张量 `[批次, 通道, 高度, 宽度]`：
  - `dim=0` → 操作**批次**维度
  - `dim=1` → 操作**通道**维度
  - `dim=2` → 操作**高度**维度（行）
  - `dim=3` → 操作**宽度**维度（列）

**scatter 的工作原理：**

1. `index` 告诉你在 `dim` 维度上的哪些位置需要修改: 例如, dim = 0, 表示行号按照 index的值走, 其他按照index的索引走. 因此, 如果此时是一个2维张量, 可以固定index的列, 然后根据列的值去改变input中对应列的不同行的取值.
2. `src` 告诉你要填充的值是什么
3. 对于 `input` 中的每个元素，根据 `index` 在 `dim` 维度上的值，决定是否用 `src` 中对应位置的值来替换（或相加/相乘）

**关键点：**

- `index` 的形状必须与 `input` 相同（除了 `dim` 维度）
- `index` 中的值表示在 `dim` 维度上的索引位置
- `src` 可以是标量、张量，只要能广播到 `input` 的形状
- **重要**：`index[i, j, k, ...]` 中的值表示在 `dim` 维度上的目标位置，其他维度保持不变

###### 基础示例

**示例 1：基本用法（替换模式）**

```python
import torch

# 创建一个目标张量
input_tensor = torch.zeros(3, 5)
print("初始 input:")
print(input_tensor)

# 创建索引张量，指定在 dim=1 上的位置
index = torch.tensor([[0, 1, 2, 0, 0],
                       [2, 0, 0, 1, 2],
                       [0, 1, 2, 2, 1]])

# 创建源数据
src = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0, 9.0, 10.0],
                     [11.0, 12.0, 13.0, 14.0, 15.0]])

# 执行 scatter 操作
result = input_tensor.scatter_(dim=1, index=index, src=src)
print("\n执行 scatter(dim=1, index=index, src=src) 后:")
print(result)
```

**输出：**

```
初始 input:
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])

执行 scatter(dim=1, index=index, src=src) 后:
tensor([[ 5.,  2.,  3.,  0.,  0.],
        [ 8.,  9., 10.,  0.,  0.],
        [11., 15., 14.,  0.,  0.]])
```

**解释：**

- `dim = 1 `, 列号跟着 index走, 其他根据index的索引走
- 对于第0行：`index[0] = [0, 1, 2, 0, 0]`，`src[0] = [1.0, 2.0, 3.0, 4.0, 5.0]`
  - 位置0：`index[0,0]=0`，将 `src[0,0]=1.0` 放到 `result[0,0]`
  - 位置1：`index[0,1]=1`，将 `src[0,1]=2.0` 放到 `result[0,1]`
  - 位置2：`index[0,2]=2`，将 `src[0,2]=3.0` 放到 `result[0,2]`
  - 位置3：`index[0,3]=0`，将 `src[0,3]=4.0` 放到 `result[0,0]`（覆盖之前的1.0）
  - 位置4：`index[0,4]=0`，将 `src[0,4]=5.0` 放到 `result[0,0]`（覆盖之前的4.0）
  - 最终第0行：`[5.0, 2.0, 3.0, 0.0, 0.0]`
- 对于第1行:
  - 位置0: `index[1,0]=2`, 将 `src[1,0]=6` 放到 `result[1,2]`
  - 位置4: `index[1,4]=2`, 将 `src[1,4]=10.0` 放到 `result[1,2]`

**示例 1.5：dim=0 的情况（在行维度上操作）**

理解不同 `dim` 值的关键是：`dim` 指定了在哪个维度上进行索引操作。

```python
import torch

# 创建一个目标张量 (4行3列)
input_tensor = torch.zeros(4, 3)
print("初始 input (4行3列):")
print(input_tensor)

# 创建索引张量，指定在 dim=0（行维度）上的位置
# index 的形状必须与 input 相同（除了 dim 维度）
# 这里 index[i, j] 表示：对于第 j 列，将 src[i, j] 放到第 index[i, j] 行
index = torch.tensor([[0, 1, 2],    # 第0行：第0列放到第0行，第1列放到第1行，第2列放到第2行
                      [1, 2, 0],    # 第1行：第0列放到第1行，第1列放到第2行，第2列放到第0行
                      [2, 0, 1],    # 第2行：第0列放到第2行，第1列放到第0行，第2列放到第1行
                      [0, 0, 0]])    # 第3行：所有列都放到第0行

# 创建源数据
src = torch.tensor([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0]])

# 执行 scatter 操作，在 dim=0（行维度）上操作
result = input_tensor.scatter_(dim=0, index=index, src=src)
print("\n执行 scatter(dim=0, index=index, src=src) 后:")
print(result)
print("\n详细解释:")
print("对于每一列，根据 index 在行维度上填充值")
```

**输出：**

```
初始 input (4行3列):
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])

执行 scatter(dim=0, index=index, src=src) 后:
tensor([[10., 11., 12.],
        [ 4.,  2.,  9.],
        [ 7.,  5.,  3.],
        [ 0.,  0.,  0.]])

详细解释:
对于每一列，根据 index 在行维度上填充值
```

**详细解释：**

当 `dim=0` 时，操作在**行维度**上进行：行号根据索引值来, 其他根据index的索引决定

**逐列分析：**

1. **第0列（column 0）**：

   - `index[0, 0] = 0` → 将 `src[0, 0] = 1.0` 放到 `result[0, 0]`
   - `index[1, 0] = 1` → 将 `src[1, 0] = 4.0` 放到 `result[1, 0]`
   - `index[2, 0] = 2` → 将 `src[2, 0] = 7.0` 放到 `result[2, 0]`
   - `index[3, 0] = 0` → 将 `src[3, 0] = 10.0` 放到 `result[0, 0]`（覆盖之前的 1.0）
   - 最终第0列：`[10.0, 4.0, 7.0]`（第0行是 1.0+10.0=11.0）
2. **第1列（column 1）**：

   - `index[0, 1] = 1` → 将 `src[0, 1] = 2.0` 放到 `result[1, 1]`
   - `index[1, 1] = 2` → 将 `src[1, 1] = 5.0` 放到 `result[2, 1]`
   - `index[2, 1] = 0` → 将 `src[2, 1] = 8.0` 放到 `result[0, 1]`
   - `index[3, 1] = 0` → 将 `src[3, 1] = 11.0` 放到 `result[0, 1]`（覆盖之前的 8.0）
   - 最终第1列：`[11.0, 2.0, 5.0]`
3. **第2列（column 2）**：

   - `index[0, 2] = 2` → 将 `src[0, 2] = 3.0` 放到 `result[2, 2]`
   - `index[1, 2] = 0` → 将 `src[1, 2] = 6.0` 放到 `result[0, 2]`
   - `index[2, 2] = 1` → 将 `src[2, 2] = 9.0` 放到 `result[1, 2]`
   - `index[3, 2] = 0` → 将 `src[3, 2] = 12.0` 放到 `result[0, 2]`（覆盖之前的 6.0）
   - 最终第2列：`[12.0, 9.0, 3.0]`

**对比 dim=0 和 dim=1：**

- `dim=1`（列维度）：在每一**行**内，根据 `index` 在列维度上填充值
- `dim=0`（行维度）：在每一**列**内，根据 `index` 在行维度上填充值

**示例 1.6：dim=2 的情况（3D张量，在深度维度上操作）**

对于3D张量，`dim=2` 表示在第三个维度（深度/通道维度）上操作。

```python
import torch

# 创建一个3D目标张量 (2个样本, 3行, 4列)
# 形状: [batch_size, height, width] = [2, 3, 4]
input_tensor = torch.zeros(2, 3, 4)
print("初始 input 形状:", input_tensor.shape)
print("初始 input[0]:")
print(input_tensor[0])
print("\n初始 input[1]:")
print(input_tensor[1])

# 创建索引张量，指定在 dim=2（列/宽度维度）上的位置
# index 的形状: [2, 3, 4]，与 input 相同
# index[i, j, k] 表示：对于第 i 个样本的第 j 行，将 src[i, j, k] 放到第 index[i, j, k] 列
index = torch.tensor([
    # 样本0
    [[0, 1, 2, 3],    # 第0行：按顺序放到0,1,2,3列
     [3, 2, 1, 0],    # 第1行：倒序放到3,2,1,0列
     [1, 1, 2, 2]],   # 第2行：放到1,1,2,2列（会有覆盖）
    # 样本1
    [[2, 0, 1, 3],    # 第0行
     [0, 0, 1, 1],    # 第1行：0,0列和1,1列会有覆盖
     [3, 2, 1, 0]]    # 第2行：倒序
])

# 创建源数据
src = torch.arange(1, 25).float().reshape(2, 3, 4)
print("\n源数据 src 形状:", src.shape)
print("src[0]:")
print(src[0])
print("\nsrc[1]:")
print(src[1])

# 执行 scatter 操作，在 dim=2（列维度）上操作
result = input_tensor.scatter_(dim=2, index=index, src=src)
print("\n执行 scatter(dim=2, index=index, src=src) 后:")
print("result[0]:")
print(result[0])
print("\nresult[1]:")
print(result[1])
```

**输出：**

```
初始 input 形状: torch.Size([2, 3, 4])
初始 input[0]:
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])

初始 input[1]:
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])

源数据 src 形状: torch.Size([2, 3, 4])
src[0]:
tensor([[ 1.,  2.,  3.,  4.],
        [ 5.,  6.,  7.,  8.],
        [ 9., 10., 11., 12.]])

src[1]:
tensor([[13., 14., 15., 16.],
        [17., 18., 19., 20.],
        [21., 22., 23., 24.]])

执行 scatter(dim=2, index=index, src=src) 后:
result[0]:
tensor([[ 1.,  2.,  3.,  4.],   # 第0行：按顺序，无覆盖
        [ 8.,  7.,  6.,  5.],   # 第1行：倒序，无覆盖
        [ 0., 10., 12.,  0.]])   # 第2行：10和11都放到列1和2，9和12被覆盖为0

result[1]:
tensor([[14., 15., 13., 16.],   # 第0行：13→列2, 14→列0, 15→列1, 16→列3
        [ 18., 20.,  0.,  0.]])   # 第1行：17和18→列0（覆盖），19和20→列1（覆盖）
        [24., 23., 22., 21.]])   # 第2行：倒序
```

**详细解释：**

当 `dim=2` 时，操作在**第三个维度**（列/宽度维度）上进行：

- `index[i, j, k]` 表示：对于第 `i` 个样本的第 `j` 行，将 `src[i, j, k]` 的值放到第 `index[i, j, k]` 列

**逐样本、逐行分析：**

**样本0：**

- **第0行**：`index[0, 0] = [0, 1, 2, 3]`，`src[0, 0] = [1, 2, 3, 4]`

  - 按顺序填充：`result[0, 0, 0]=1`, `result[0, 0, 1]=2`, `result[0, 0, 2]=3`, `result[0, 0, 3]=4`
- **第1行**：`index[0, 1] = [3, 2, 1, 0]`，`src[0, 1] = [5, 6, 7, 8]`

  - 倒序填充：`result[0, 1, 3]=5`, `result[0, 1, 2]=6`, `result[0, 1, 1]=7`, `result[0, 1, 0]=8`
- **第2行**：`index[0, 2] = [1, 1, 2, 2]`，`src[0, 2] = [9, 10, 11, 12]`

  - `result[0, 2, 1]=9` → 被覆盖
  - `result[0, 2, 1]=10`（覆盖9）
  - `result[0, 2, 2]=11`
  - `result[0, 2, 2]=12`（覆盖11）
  - 最终：`[0, 10, 12, 0]`

**样本1：**

- **第0行**：`index[1, 0] = [2, 0, 1, 3]`，`src[1, 0] = [13, 14, 15, 16]`

  - `result[1, 0, 2]=13`, `result[1, 0, 0]=14`, `result[1, 0, 1]=15`, `result[1, 0, 3]=16`
- **第1行**：`index[1, 1] = [0, 0, 1, 1]`，`src[1, 1] = [17, 18, 19, 20]`

  - `result[1, 1, 0]=17` → 被覆盖
  - `result[1, 1, 0]=18`（覆盖17）
  - `result[1, 1, 1]=19` → 被覆盖
  - `result[1, 1, 1]=20`（覆盖19）
  - 最终：`[18, 20, 0, 0]`

**总结不同 dim 值的操作方向：**

- `dim=0`：在**行维度**上操作，固定列，改变行位置
- `dim=1`：在**列维度**上操作，固定行，改变列位置
- `dim=2`：在**深度/通道维度**上操作，固定前两个维度，改变第三个维度的位置
- 更高维度：依此类推

**示例 1.7：dim=0 与 dim=1 的对比示例**

通过同一个数据在不同 dim 下的操作，直观理解区别：

```python
import torch

# 相同的输入数据
input_base = torch.zeros(3, 4)
index = torch.tensor([[0, 1, 2, 0],
                      [1, 2, 0, 1],
                      [2, 0, 1, 2]])
src = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0]])

# dim=0：在行维度上操作
input_dim0 = input_base.clone()
result_dim0 = input_dim0.scatter_(dim=0, index=index, src=src)
print("dim=0 (行维度操作):")
print(result_dim0)
print("\n解释：对于每一列，根据 index 在行维度上填充")
print("例如第0列：index[:,0]=[0,1,2]，src[:,0]=[1,5,9]")
print("→ result[0,0]=1, result[1,0]=5, result[2,0]=9")

# dim=1：在列维度上操作
input_dim1 = input_base.clone()
result_dim1 = input_dim1.scatter_(dim=1, index=index, src=src)
print("\n" + "="*50)
print("dim=1 (列维度操作):")
print(result_dim1)
print("\n解释：对于每一行，根据 index 在列维度上填充")
print("例如第0行：index[0]=[0,1,2,0]，src[0]=[1,2,3,4]")
print("→ result[0,0]=4(最后覆盖), result[0,1]=2, result[0,2]=3")
```

**输出：**

```
dim=0 (行维度操作):
tensor([[ 1., 10.,  7.,  4.],
        [ 5.,  2., 11.,  8.],
        [ 9.,  6.,  3., 12.]])

解释：对于每一列，根据 index 在行维度上填充
例如第0列：index[:,0]=[0,1,2]，src[:,0]=[1,5,9]
→ result[0,0]=1, result[1,0]=5, result[2,0]=9

==================================================
dim=1 (列维度操作):
tensor([[ 4.,  2.,  3.,  0.],
        [ 7.,  8.,  6.,  0.],
        [10., 11., 12.,  0.]])

解释：对于每一行，根据 index 在列维度上填充
例如第0行：index[0]=[0,1,2,0]，src[0]=[1,2,3,4]
→ result[0,0]=4(最后覆盖), result[0,1]=2, result[0,2]=3
```

**示例 2：使用标量作为 src**

```python
import torch

input_tensor = torch.zeros(3, 5)
index = torch.tensor([[0, 1, 2, 0, 0],
                       [2, 0, 0, 1, 2],
                       [0, 1, 2, 2, 1]])

# 使用标量
result = input_tensor.scatter_(dim=1, index=index, src=1.0)
print("使用标量 src=1.0:")
print(result)
```

**输出：**

```
tensor([[1., 1., 1., 0., 0.],
        [1., 1., 1., 0., 0.],
        [1., 1., 1., 0., 0.]])
```

**示例 3：使用 reduce='add'（相加模式）**

```python
import torch

input_tensor = torch.zeros(3, 5)
index = torch.tensor([[0, 1, 2, 0, 0],
                       [2, 0, 0, 1, 2],
                       [0, 1, 2, 2, 1]])

src = torch.ones(3, 5)

# 使用 add 模式
result = input_tensor.scatter_(dim=1, index=index, src=src, reduce='add')
print("使用 reduce='add':")
print(result)
```

**输出：**

```
tensor([[3., 1., 1., 0., 0.],  # 位置0有3个1相加=3
        [2., 1., 2., 0., 0.],  # 位置0有2个1相加=2，位置2有2个1相加=2
        [1., 2., 2., 0., 0.]])  # 位置1有2个1相加=2，位置2有2个1相加=2
```

**解释：**

- 在 `reduce='add'` 模式下，如果多个 `src` 值映射到同一个位置，它们会被相加
- 第0行位置0：有3个值（来自index的3个0）都映射到这里，所以 `1+1+1=3`

**示例 4：使用 reduce='multiply'（相乘模式）**

```python
import torch

input_tensor = torch.ones(3, 5) * 2  # 初始值为2
index = torch.tensor([[0, 1, 2, 0, 0],
                       [2, 0, 0, 1, 2],
                       [0, 1, 2, 2, 1]])

src = torch.ones(3, 5) * 3  # 源值为3

result = input_tensor.scatter_(dim=1, index=index, src=src, reduce='multiply')
print("使用 reduce='multiply' (初始值为2，src为3):")
print(result)
```

**输出：**

```
tensor([[54.,  3.,  3.,  2.,  2.],  # 位置0: 2*3*3*3=54
        [ 9.,  3.,  9.,  2.,  2.],  # 位置0: 2*3*3=9，位置2: 2*3*3=9
        [ 3.,  9.,  9.,  2.,  2.]])  # 位置1: 2*3*3=9，位置2: 2*3*3=9
```

###### 实际应用场景

**场景 1：One-hot 编码**

```python
import torch

# 将类别索引转换为 one-hot 编码
num_classes = 5
labels = torch.tensor([2, 0, 4, 1, 3])  # 类别索引

# 创建零张量
one_hot = torch.zeros(labels.size(0), num_classes)

# 使用 scatter 填充
one_hot.scatter_(dim=1, index=labels.unsqueeze(1), src=1.0)
print("One-hot 编码:")
print(one_hot)
```

**输出：**

```
tensor([[0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0.]])
```

**场景 2：目标检测中的 bbox 索引**

```python
import torch

# 假设有3个样本，每个样本有5个候选框，我们想选择特定的框
batch_size = 3
num_boxes = 5
box_dim = 4  # [x, y, w, h]

# 所有候选框
all_boxes = torch.randn(batch_size, num_boxes, box_dim)
print("所有候选框形状:", all_boxes.shape)

# 每个样本选择的框索引
selected_indices = torch.tensor([[0], [2], [4]])  # 第0个样本选第0个框，第1个样本选第2个框，第2个样本选第4个框

# 创建输出张量
selected_boxes = torch.zeros(batch_size, 1, box_dim)

# 使用 gather 更合适，但这里演示 scatter 的逆操作思路
# 实际上应该用 gather，但 scatter 可以用于反向操作
```

**场景 3：稀疏矩阵填充**

使用 `scatter` 可以高效地将稀疏数据填充到密集矩阵中。

```python
import torch

# 创建一个稀疏矩阵，只在特定位置有值
rows = 4
cols = 5

# 稀疏数据：(行索引, 列索引, 值)
row_indices = torch.tensor([0, 1, 2, 0, 3])  # 5个位置的行索引
col_indices = torch.tensor([0, 1, 2, 3, 4])  # 5个位置的列索引
sparse_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # 对应的值

# 方法1：使用 scatter（按行填充）
# 将每行的数据单独 scatter
dense_matrix = torch.zeros(rows, cols)
for row in range(rows):
    # 找到属于这一行的所有列索引和值
    mask = row_indices == row
    if mask.any():
        cols_in_row = col_indices[mask]
        values_in_row = sparse_values[mask]
        # 在这一行上 scatter
        dense_matrix[row].scatter_(0, cols_in_row, values_in_row)

print("方法1 - 按行 scatter:")
print(dense_matrix)
# tensor([[1., 0., 0., 4., 0.],
#         [0., 2., 0., 0., 0.],
#         [0., 0., 3., 0., 0.],
#         [0., 0., 0., 0., 5.]])

# 方法2：使用 scatter（更简洁，按列填充）
dense_matrix2 = torch.zeros(rows, cols)
# 将列索引扩展为2D，匹配 dense_matrix 的形状
for col in range(cols):
    mask = col_indices == col
    if mask.any():
        rows_in_col = row_indices[mask]
        values_in_col = sparse_values[mask]
        # 在这一列上 scatter
        dense_matrix2[:, col].scatter_(0, rows_in_col, values_in_col)

print("\n方法2 - 按列 scatter:")
print(dense_matrix2)

# 方法3：使用高级索引（最直接，推荐用于稀疏矩阵）
dense_matrix3 = torch.zeros(rows, cols)
dense_matrix3[row_indices, col_indices] = sparse_values

print("\n方法3 - 高级索引（推荐）:")
print(dense_matrix3)

# 方法4：使用 scatter_add（如果有重复位置需要累加）
dense_matrix4 = torch.zeros(rows, cols)
# 假设有重复位置
row_indices_dup = torch.tensor([0, 1, 2, 0, 3, 0])  # 位置 (0, 0) 重复
col_indices_dup = torch.tensor([0, 1, 2, 3, 4, 0])
sparse_values_dup = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])

# 使用 scatter_ 会覆盖
dense_matrix4[row_indices_dup, col_indices_dup] = sparse_values_dup
print("\n方法4a - 直接索引（覆盖）:")
print(dense_matrix4)  # (0,0) 位置是 10.0（被覆盖）

# 使用 index_put_ 的 accumulate=True 可以累加
dense_matrix5 = torch.zeros(rows, cols)
dense_matrix5.index_put_((row_indices_dup, col_indices_dup),
                         sparse_values_dup, accumulate=True)
print("\n方法4b - index_put 累加模式:")
print(dense_matrix5)  # (0,0) 位置是 11.0（1.0 + 10.0）
```

**输出：**

```
方法1 - 按行 scatter:
tensor([[1., 0., 0., 4., 0.],
        [0., 2., 0., 0., 0.],
        [0., 0., 3., 0., 0.],
        [0., 0., 0., 0., 5.]])

方法2 - 按列 scatter:
tensor([[1., 0., 0., 4., 0.],
        [0., 2., 0., 0., 0.],
        [0., 0., 3., 0., 0.],
        [0., 0., 0., 0., 5.]])

方法3 - 高级索引（推荐）:
tensor([[1., 0., 0., 4., 0.],
        [0., 2., 0., 0., 0.],
        [0., 0., 3., 0., 0.],
        [0., 0., 0., 0., 5.]])

方法4a - 直接索引（覆盖）:
tensor([[10., 0., 0., 4., 0.],
        [ 0., 2., 0., 0., 0.],
        [ 0., 0., 3., 0., 0.],
        [ 0., 0., 0., 0., 5.]])

方法4b - index_put 累加模式:
tensor([[11., 0., 0., 4., 0.],
        [ 0., 2., 0., 0., 0.],
        [ 0., 0., 3., 0., 0.],
        [ 0., 0., 0., 0., 5.]])
```

**总结：**

- 对于简单的稀疏矩阵填充，**方法3（高级索引）最简洁直观**
- 如果需要逐行/逐列处理，可以使用 **方法1/2（scatter）**
- 如果有重复位置需要累加，使用 **index_put_ 的 accumulate=True**

```

###### 注意事项

1. **索引越界**：`index` 中的值必须在 `[0, input.size(dim))` 范围内，否则会报错
2. **形状匹配**：`index` 的形状必须与 `input` 相同（除了 `dim` 维度）
3. **原地操作**：`scatter_` 是原地操作，会修改原张量；`scatter` 返回新张量
4. **覆盖问题**：在默认模式下，如果多个值映射到同一位置，后面的值会覆盖前面的值
5. **reduce 模式**：使用 `reduce='add'` 或 `reduce='multiply'` 时，多个值会进行归约操作

###### 与 gather 的对比

- **gather**：从源张量中根据索引收集值 → `output[i][j] = input[i][index[i][j]]`
- **scatter**：根据索引将值分散到目标张量 → `output[i][index[i][j]] = src[i][j]`

两者是相反的操作，`gather` 用于"收集"，`scatter` 用于"分散"。

###### 完整示例代码

```python
import torch

def demonstrate_scatter():
    """演示 scatter 函数的各种用法"""
  
    print("=" * 60)
    print("scatter 函数完整演示")
    print("=" * 60)
  
    # 示例 1: 基本替换
    print("\n【示例1】基本替换操作")
    input_tensor = torch.zeros(2, 4)
    index = torch.tensor([[0, 1, 2, 0],
                          [1, 2, 0, 3]])
    src = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]])
    result1 = input_tensor.scatter_(dim=1, index=index, src=src)
    print("Input shape:", input_tensor.shape)
    print("Index shape:", index.shape)
    print("Src shape:", src.shape)
    print("Result:\n", result1)
  
    # 示例 2: 使用 add 模式
    print("\n【示例2】使用 reduce='add'")
    input_tensor2 = torch.zeros(2, 4)
    index2 = torch.tensor([[0, 1, 0, 0],
                           [1, 1, 2, 2]])
    src2 = torch.ones(2, 4)
    result2 = input_tensor2.scatter_(dim=1, index=index2, src=src2, reduce='add')
    print("Result:\n", result2)
    print("说明：位置[0,0]有3个1相加=3，位置[1,1]有2个1相加=2")
  
    # 示例 3: One-hot 编码
    print("\n【示例3】One-hot 编码")
    num_classes = 5
    labels = torch.tensor([2, 0, 4, 1])
    one_hot = torch.zeros(4, num_classes)
    one_hot.scatter_(dim=1, index=labels.unsqueeze(1), src=1.0)
    print("Labels:", labels)
    print("One-hot:\n", one_hot)
  
    print("\n" + "=" * 60)

# 运行演示
demonstrate_scatter()
```

###### 总结

`scatter` 函数是一个强大的索引和赋值工具，特别适用于：

- One-hot 编码转换
- 稀疏数据的批量填充
- 根据条件进行选择性赋值
- 需要根据索引进行归约操作的场景

理解 `scatter` 的关键是理解 `index` 如何指定位置，`src` 如何提供值，以及它们如何在 `dim` 维度上协同工作。

#### 2.1.4 张量的随机种子

随机种子（random seed）是编程语言中的基础概念，主要用于实验的复现。PyTorch 针对不同设备（CPU、CUDA、MPS）提供了相应的随机种子设置函数。
随机种子（random seed）是编程语言中基础的概念，大多数编程语言都有随机种子的概念，它主要用于实验的复现。针对随机种子pytorch也有一些设置函数。

| 函数名                  | 功能描述             | 说明                                                                                                     |
| ----------------------- | -------------------- | -------------------------------------------------------------------------------------------------------- |
| **manual_seed**   | 手动设置随机种子     | 建议设置为42，这是近期一个玄学研究。说42有效的提高模型精度。当然大家可以设置为你喜欢的，只要保持一致即可 |
| **initial_seed**  | 返回初始种子         | 返回当前使用的随机种子值                                                                                 |
| **get_rng_state** | 获取随机数生成器状态 | Returns the random number generator state as a torch.ByteTensor                                          |
| **set_rng_state** | 设定随机数生成器状态 | 这两怎么用暂时未知。Sets the random number generator state                                               |

以上均是设置 CPU 上的张量随机种子。不同设备需要分别设置随机种子：

##### CPU 随机种子

```python
import torch

# 设置随机种子
torch.manual_seed(42)

# 获取初始种子
print(torch.initial_seed())  # 输出: 42

# 获取随机数生成器状态
state = torch.get_rng_state()
print(state.shape)  # 输出: torch.Size([...])
```

##### CUDA 随机种子

```python
import torch

# 设置单个 GPU 的随机种子
torch.cuda.manual_seed(42)

# 设置所有 GPU 的随机种子
torch.cuda.manual_seed_all(42)

# 获取 CUDA 初始种子
print(torch.cuda.initial_seed())  # 输出: 42

# 获取 CUDA 随机数生成器状态
cuda_state = torch.cuda.get_rng_state()
print(cuda_state.shape)
```

##### MPS 随机种子（Mac Apple Silicon）

在 MPS 设备上，使用 `torch.mps` 模块来管理随机种子：

```python
import torch

# 检查 MPS 是否可用
if torch.backends.mps.is_available():
    # 设置 MPS 随机种子
    torch.mps.manual_seed(42)
  
    # 获取 MPS 随机数生成器状态
    mps_state = torch.mps.get_rng_state()
    print(mps_state.shape)  # 输出: torch.Size([...])
    print(mps_state.dtype)  # 输出: torch.uint8
  
    # 设置 MPS 随机数生成器状态（恢复状态）
    torch.mps.set_rng_state(mps_state)
```

**MPS 随机种子函数说明：**

| 函数名                             | 功能描述                  | 说明                                              |
| ---------------------------------- | ------------------------- | ------------------------------------------------- |
| `torch.mps.manual_seed(seed)`    | 手动设置 MPS 随机种子     | 设置 MPS 设备的随机种子值                         |
| `torch.mps.get_rng_state()`      | 获取 MPS 随机数生成器状态 | 返回 MPS 设备的完整随机数生成器状态（ByteTensor） |
| `torch.mps.set_rng_state(state)` | 设定 MPS 随机数生成器状态 | 恢复 MPS 设备的随机数生成器状态                   |

**注意：**

- MPS 模块**没有** `initial_seed()` 和 `seed()` 函数，只有 `manual_seed()`, `get_rng_state()`, `set_rng_state()` 三个函数
- `torch.mps.get_rng_state()` 返回完整的随机数生成器状态（ByteTensor），可以用于保存和恢复状态
- `torch.mps.set_rng_state(state)` 用于恢复之前保存的随机数生成器状态
- MPS 的随机种子与 CPU 和 CUDA 的随机种子是独立的，需要分别设置
- 如果需要获取当前使用的种子值，需要在设置时自己记录，MPS 模块不提供查询函数

**完整示例：**

```python
import torch

# 设置所有设备的随机种子
seed = 42
torch.manual_seed(seed)  # CPU

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # CUDA

if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)  # MPS

# 获取各设备的初始种子
print(f"CPU seed: {torch.initial_seed()}")
if torch.cuda.is_available():
    print(f"CUDA seed: {torch.cuda.initial_seed()}")
if torch.backends.mps.is_available():
    # MPS 没有 initial_seed() 函数，需要自己记录种子值
    print(f"MPS seed: 42 (需要自己记录)")

```

#### 2.1.4 广播机制（Broadcasting）

**什么是广播？**

广播（Broadcasting）是 PyTorch 中非常重要的概念，它允许不同形状的张量进行运算，而无需手动扩展维度。

**广播规则：**

1. **从右向左对齐维度**
2. **维度大小为 1 的可以扩展到任意大小**
3. **缺失的维度视为大小 1**
4. **两个张量在某个维度上的大小必须相等，或其中一个为 1**

**示例：**

```python
import torch

# 示例1：标量广播
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # [2, 3]
b = 10
c = a + b  # b 广播为 [[10, 10, 10], [10, 10, 10]]

# 示例2：向量广播
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # [2, 3]
b = torch.tensor([10, 20, 30])  # [3]
c = a + b  # b 广播为 [[10, 20, 30], [10, 20, 30]]

# 示例3：列向量广播
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # [2, 3]
b = torch.tensor([[10], [20]])  # [2, 1]
c = a + b  # b 广播为 [[10, 10, 10], [20, 20, 20]]

# 示例4：不兼容的形状
a = torch.randn(4, 3)
b = torch.randn(4)
# c = a + b  # 错误！[4, 3] + [4] 无法广播
# 解决：手动添加维度
b = b.unsqueeze(1)  # [4] → [4, 1]
c = a + b  # [4, 3] + [4, 1] → [4, 3]
```

**常见应用：**

```python
# 批量归一化
data = torch.randn(64, 3, 224, 224)  # [batch, channel, H, W]
mean = data.mean(dim=[0, 2, 3], keepdim=True)  # [1, 3, 1, 1]
std = data.std(dim=[0, 2, 3], keepdim=True)
normalized = (data - mean) / std  # 广播到 [64, 3, 224, 224]
```

#### 2.1.5 张量的数学操作

张量提供了非常丰富的数学操作，包括逐元素操作、聚合操作、比较操作、线性代数操作等。这里按**类别 + 常用函数表**的形式整理，方便查阅。

##### 1. Pointwise Ops（逐元素操作）

这类操作不会改变张量形状，只是对每个元素做相同的数学运算。

| 函数名                                                        | 功能                           | 主要参数                 | 使用示例                           |
| ------------------------------------------------------------- | ------------------------------ | ------------------------ | ---------------------------------- |
| `torch.abs` / `tensor.abs()`                              | 取绝对值                       | `input`：张量          | `torch.abs(x)`                   |
| `torch.relu` / `tensor.relu()`                            | ReLU 激活：`max(x, 0)`       | `input`：张量          | `torch.relu(x)`                  |
| `torch.clamp` / `tensor.clamp()`                          | 截断到区间 `[min, max]`      | `min`、`max`：上下界 | `torch.clamp(x, min=0., max=1.)` |
| `torch.round` / `torch.floor` / `torch.ceil`            | 四舍五入 / 向下取整 / 向上取整 | `input`：张量          | `torch.floor(x)`                 |
| `torch.exp` / `torch.log` / `torch.sqrt`                | 指数 / 对数 / 平方根           | `input`：张量          | `torch.log(x + 1e-6)`            |
| `torch.sin` / `torch.cos` / `torch.tan`                 | 三角函数                       | `input`：张量          | `torch.sin(x)`                   |
| `torch.pow` / `tensor.pow()`                              | 幂运算                         | `exponent`：标量或张量 | `torch.pow(x, 2)`                |
| `torch.add` / `torch.sub` / `torch.mul` / `torch.div` | 加/减/乘/除（支持广播）        | `input`, `other`     | `torch.add(x, y)` 或 `x + y`   |

**重要区分：**

- **`torch.mul(a, b)` / `a * b`**：逐元素乘，结果形状与广播后的形状一致。
- **`torch.mm(a, b)` / `a @ b`**：矩阵乘法，要求 `(m,n) @ (n,p) → (m,p)`，用于线性层、矩阵运算。

**示例：逐元素乘（mul）与标量、张量**

```python
import torch

# 张量 × 标量
x = torch.tensor([1., 2., 3.])
print(torch.mul(x, 2))   # tensor([2., 4., 6.])
print(x * 2)             # 与上面等价

# 张量 × 张量（逐元素，同形状）
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[2., 2.], [2., 2.]])
print(torch.mul(a, b))   # tensor([[ 2.,  4.],
                         #         [ 6.,  8.]])
print(a * b)             # 与上面等价

# 广播： (3,1) * (1,3) → (3,3)
c = torch.tensor([[1.], [2.], [3.]])  # (3, 1)
d = torch.tensor([[10., 20., 30.]])   # (1, 3)
print(torch.mul(c, d))   # 形状 (3, 3)
```

**示例：add / sub / div 用法**

```python
import torch

x = torch.tensor([1., 2., 3.])
y = torch.tensor([10., 20., 30.])

print(torch.add(x, y))   # tensor([11., 22., 33.])
print(torch.sub(y, x))   # tensor([ 9., 18., 27.])
print(torch.div(y, x))   # tensor([10., 10., 10.])
# 等价于 x + y, y - x, y / x
```

**简单示例：**

```python
import torch

x = torch.tensor([-1.0, 0.5, 2.0])
print("x       :", x)
print("abs    :", torch.abs(x))
print("relu   :", torch.relu(x))
print("clamp0-1:", torch.clamp(x, 0., 1.))
print("square :", torch.pow(x, 2))


x       : tensor([-1.0000,  0.5000,  2.0000])
abs    : tensor([1.0000, 0.5000, 2.0000])
relu   : tensor([0.0000, 0.5000, 2.0000])
clamp0-1: tensor([0.0000, 0.5000, 1.0000])
square : tensor([1.0000, 0.2500, 4.0000])
```

##### 2. Reduction Ops（聚合 / 降维操作）

这类操作通常会**在某个维度上聚合**（求和、求平均等），从而减少该维度的长度。

| 函数名                              | 功能                  | 主要参数                                                                                                         | 使用示例                         |
| ----------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `torch.sum` / `tensor.sum()`    | 求和                  | `dim`：在哪个维度聚合, dim=0, 在行上处理, 其他维度保持, dim = 1, 对列求和, 其他不变；`keepdim`：是否保留维度 | `x.sum(dim=1, keepdim=True)`   |
| `torch.mean` / `tensor.mean()`  | 均值                  | 同上                                                                                                             | `x.mean(dim=0)`                |
| `torch.max` / `torch.min`       | 返回最大/最小值及索引 | `dim`：维度；返回 `(values, indices)`                                                                        | `values, idx = x.max(dim=1)`   |
| `torch.argmax` / `torch.argmin` | 返回最大/最小值索引   | `dim`：维度                                                                                                    | `idx = x.argmax(dim=1)`        |
| `torch.prod`                      | 所有元素连乘          | `dim`：按维度连乘                                                                                              | `x.prod(dim=1)`                |
| `torch.std` / `torch.var`       | 标准差 / 方差         | `unbiased`：是否使用无偏估计                                                                                   | `x.std(dim=0, unbiased=False)` |
| `torch.all` / `torch.any`       | 逻辑与 / 或 聚合      | `dim`：维度                                                                                                    | `(x > 0).all(dim=1)`           |
| `torch.norm` / `tensor.norm()`  | 范数（L2/L1等）       | `p`：范数类型，`dim`：在哪个维度                                                                             | `x.norm(p=2, dim=1)`           |

**简单示例：**

```python
import torch

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

print("sum_all        :", x.sum())
print("sum_dim0       :", x.sum(dim=0))      # 列方向求和
print("sum_dim1       :", x.sum(dim=1, keepdim=True))      # 行方向求和
print("max_dim0       :", x.max(dim=0))      # 列方向最大值
print("mean_dim1      :", x.mean(dim=1))     # 每行的平均值
values, idx = x.max(dim=1)
print("row_max values :", values)
print("row_max indices:", idx)

sum_all        : tensor(21.)
sum_dim0       : tensor([5., 7., 9.])
sum_dim1       : tensor([[ 6.],
        [15.]])
max_dim0       : torch.return_types.max(
values=tensor([4., 5., 6.]),
indices=tensor([1, 1, 1]))
mean_dim1      : tensor([2., 5.])
row_max values : tensor([3., 6.])
row_max indices: tensor([2, 2])
```

##### 3. Comparison Ops（比较 / 排序相关）

| 函数名                                           | 功能                                     | 主要参数                    | 使用示例                                    |
| ------------------------------------------------ | ---------------------------------------- | --------------------------- | ------------------------------------------- |
| `torch.eq` / `gt` / `ge` / `lt` / `le` | 等于 / 大于 / 大于等于 / 小于 / 小于等于 | `input`, `other`        | `(x > 0)`、`torch.eq(x, y)`             |
| `torch.isnan` / `torch.isinf`                | 判断 NaN / 无穷                          | `input`：张量             | `torch.isnan(x)`                          |
| `torch.where`                                  | 条件选择                                 | `condition`, `x`, `y` | `torch.where(x > 0, x, 0.)`               |
| `torch.sort`                                   | 排序，返回值和索引                       | `dim`, `descending`     | `values, idx = torch.sort(x, dim=1)`      |
| `torch.argsort`                                | 仅返回排序索引                           | 同上                        | `idx = torch.argsort(x, dim=1)`           |
| `torch.topk`                                   | 取前 k 大/小元素及索引                   | `k`, `dim`, `largest` | `values, idx = torch.topk(x, k=3, dim=1)` |

**简单示例：**

```python
import torch

x = torch.tensor([[1.0, -2.0, 3.0],
                  [0.5,  4.0, -1.0]])

mask = x > 0
print("mask:\n", mask)
print("where >0 keep, else 0:\n", torch.where(mask, x, torch.zeros_like(x)))

values, idx = torch.topk(x, k=2, dim=1)
print("top2 values:\n", values)
print("top2 indices:\n", idx)

mask:
 tensor([[ True, False,  True],
        [ True,  True, False]])
where >0 keep, else 0:
 tensor([[1.0000, 0.0000, 3.0000],
        [0.5000, 4.0000, 0.0000]])
top2 values:
 tensor([[3.0000, 1.0000],
        [4.0000, 0.5000]])
top2 indices:
 tensor([[2, 0],
        [1, 0]])
```

##### 4. 线性代数 / BLAS & LAPACK Operations

这类操作用于矩阵乘法、向量内积、分解等，是深度学习中最常用的一类。

| 函数名                                   | 功能                                     | 主要参数                                           | 使用示例                                |
| ---------------------------------------- | ---------------------------------------- | -------------------------------------------------- | --------------------------------------- |
| `torch.matmul` / `@`                 | 通用矩阵乘法（支持 1D/2D/3D 批量）       | `input`, `other`                               | `x @ w` 或 `torch.matmul(x, w)`     |
| `torch.mm`                             | 2D 矩阵乘法                              | 两个二维张量                                       | `torch.mm(A, B)`                      |
| `torch.bmm`                            | 批量矩阵乘法 (3D)                        | `[B, N, M] @ [B, M, K]`                          | `torch.bmm(batch_A, batch_B)`         |
| `torch.addmm`                          | `beta * input + alpha * (mat1 @ mat2)` | `input`, `mat1`, `mat2`, `beta`, `alpha` | 线性层内部常用                          |
| `torch.mv` / `torch.dot`             | 矩阵-向量乘 / 向量内积                   | 向量/矩阵                                          | `torch.mv(A, x)`, `torch.dot(x, y)` |
| `torch.norm`                           | 向量或矩阵范数                           | `p`，`dim`                                     | `x.norm(p=2)`                         |
| `torch.inverse` / `torch.linalg.inv` | 矩阵求逆                                 | 方阵                                               | `torch.linalg.inv(A)`                 |
| `torch.svd` / `torch.linalg.svd`     | 奇异值分解                               | `full_matrices` 等                               | `U, S, Vh = torch.linalg.svd(A)`      |

**简单示例（矩阵乘法）：**

```python
import torch

x = torch.tensor([[1., 2.],
                  [3., 4.]])      # 形状 [2, 2]
w = torch.tensor([[5., 6.],
                  [7., 8.]])      # 形状 [2, 2]

print("x @ w =\n", x @ w)         # 等价于 torch.matmul(x, w)
```

##### 5. 其他常用操作（部分）

| 函数名                               | 功能                               | 使用示例                            |
| ------------------------------------ | ---------------------------------- | ----------------------------------- |
| `torch.clone` / `tensor.clone()` | 拷贝一个张量（与原张量不共享存储） | `y = x.clone()`                   |
| `torch.flip`                       | 按指定维度翻转                     | `torch.flip(x, dims=[0])`         |
| `torch.diag` / `torch.diagonal`  | 从向量构造对角矩阵 / 取对角线      | `torch.diag(v)`，`x.diagonal()` |
| `torch.tril` / `torch.triu`      | 取下三角 / 上三角部分              | `torch.tril(x)`                   |
| `torch.cumsum` / `torch.cumprod` | 累积和 / 累积乘                    | `x.cumsum(dim=0)`                 |

> **建议**：不用强行记住所有 API，大致知道有哪几大类即可，用到时再查官方文档或本笔记。

### 2.2 自动微分（Autograd）

自动微分（Automatic Differentiation，简称 Autograd）是 PyTorch 的核心功能之一，它能够自动计算梯度，是深度学习训练的基础。通过计算图（DAG）机制，PyTorch 可以自动追踪所有操作并计算梯度。

#### 2.2.1 Autograd 官方定义

根据 PyTorch 官方文档，autograd 的核心概念如下：

> Conceptually, autograd keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG) consisting of Function objects. In this DAG, leaves are the input tensors, roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.

**核心要点：**

- 自动求导机制通过**有向无环图（DAG）**实现
- 在 DAG 中，记录数据（对应 `tensor.data`）以及操作（对应 `tensor.grad_fn`）
- 操作在 PyTorch 中统称为 **Function**，如加法、减法、乘法、ReLU、卷积、池化等，都是 Function
- **叶子节点（leaves）**：输入张量（通常是需要求梯度的参数）
- **根节点（roots）**：输出张量（通常是损失函数）

**前向传播（Forward Pass）时，autograd 同时做两件事：**

1. 执行请求的操作来计算结果张量
2. 在 DAG 中维护操作的梯度函数（gradient function）

**反向传播（Backward Pass）时，当在根节点调用 `.backward()` 后：**

1. 从每个 `.grad_fn` 计算梯度
2. 将梯度累积到相应张量的 `.grad` 属性中
3. 使用链式法则，一直传播到叶子张量

#### 2.2.2 Autograd 的使用

autograd 的使用主要有三种方法，分别适用于不同的场景：

##### 1. torch.autograd.backward

`backward()` 函数是使用频率最高的自动求导函数，99% 的训练代码中都会用它进行梯度求导。

**函数签名：**

```python
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, 
                        create_graph=False, grad_variables=None, inputs=None)
```

**主要参数：**

| 参数                   | 类型                                 | 说明                                                           |
| ---------------------- | ------------------------------------ | -------------------------------------------------------------- |
| **tensors**      | Tensor 或 Sequence[Tensor]           | 用于求导的张量，通常是损失函数 loss                            |
| **grad_tensors** | Tensor 或 Sequence[Tensor], optional | 雅可比向量积中使用，用于加权梯度                               |
| **retain_graph** | bool, optional                       | 是否需要保留计算图。默认 False，反向传播后释放计算图以节省内存 |
| **create_graph** | bool, optional                       | 是否创建计算图，用于高阶求导                                   |
| **inputs**       | Tensor 或 Sequence[Tensor], optional | 指定要对哪些输入求梯度。如果未提供，梯度会累积到所有叶子张量   |

**注意：** `Tensor.backward()` 接口内部调用了 `torch.autograd.backward()`。

**示例 1：基本用法**

```python
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)      # a = w + x = 3
b = torch.add(w, 1)      # b = w + 1 = 2
y = torch.mul(a, b)      # y = a * b = 6

y.backward()             # 反向传播
print(f"w.grad: {w.grad}")  # dy/dw = b + a = 2 + 3 = 5
print(f"x.grad: {x.grad}")  # dy/dx = b = 2
```

**示例 2：retain_graph 参数**

默认情况下，反向传播后计算图会被释放。如果需要多次求导，需要设置 `retain_graph=True`：

```python
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

# 第一次反向传播
y.backward(retain_graph=True)
print(f"第一次 w.grad: {w.grad}")  # tensor([5.])

# 第二次反向传播（需要 retain_graph=True）
y.backward()
print(f"第二次 w.grad: {w.grad}")  # tensor([10.]) - 梯度会累加
```

**示例 3：grad_tensors 参数（雅可比向量积）**

当有多个输出时，可以使用 `grad_tensors` 来加权不同输出的梯度：

```python
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)      # a = w + x = 3
b = torch.add(w, 1)      # b = w + 1 = 2

y0 = torch.mul(a, b)     # y0 = (x+w) * (w+1) = 6, dy0/dw = 2w + x + 1 = 5
y1 = torch.add(a, b)     # y1 = (x+w) + (w+1) = 5, dy1/dw = 2

loss = torch.cat([y0, y1], dim=0)  # [y0, y1]

# 使用 grad_tensors 加权
grad_tensors = torch.tensor([1., 2.])
loss.backward(gradient=grad_tensors)

# w.grad = 1 * (dy0/dw) + 2 * (dy1/dw) = 1 * 5 + 2 * 2 = 9
print(f"w.grad: {w.grad}")  # tensor([9.])
```

##### 2. torch.autograd.grad

`torch.autograd.grad` 用于计算 `outputs` 对 `inputs` 的导数，返回梯度而不是累积到 `.grad` 属性中。

**函数签名：**

```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, 
                    create_graph=False, only_inputs=True, allow_unused=False)
```

**主要参数：**

| 参数                   | 类型                                 | 说明                         |
| ---------------------- | ------------------------------------ | ---------------------------- |
| **outputs**      | Tensor 或 Sequence[Tensor]           | 用于求导的张量，如 loss      |
| **inputs**       | Tensor 或 Sequence[Tensor]           | 所要计算导数的张量           |
| **grad_outputs** | Tensor 或 Sequence[Tensor], optional | 雅可比向量积中使用           |
| **retain_graph** | bool, optional                       | 是否需要保留计算图           |
| **create_graph** | bool, optional                       | 是否创建计算图，用于高阶求导 |
| **allow_unused** | bool, optional                       | 是否允许未使用的输入张量     |

**示例：一阶和二阶导数**

```python
import torch

x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)     # y = x^2

# 一阶导数：dy/dx = 2x = 6
grad_1 = torch.autograd.grad(y, x, create_graph=True)
print(f"一阶导数: {grad_1[0]}")  # tensor([6.], grad_fn=<...>)

# 二阶导数：d(dy/dx)/dx = d(2x)/dx = 2
grad_2 = torch.autograd.grad(grad_1[0], x)
print(f"二阶导数: {grad_2[0]}")  # tensor([2.])
```

##### 3. torch.autograd.Function

`torch.autograd.Function` 用于自定义操作（op），当你需要实现特殊的数学函数或 PyTorch 中没有的网络层时，可以自定义 Function。

**实现步骤：**

1. 继承 `torch.autograd.Function`
2. 实现 `forward` 方法：定义前向传播的计算公式
3. 实现 `backward` 方法：定义反向传播的梯度计算公式

**示例：自定义 Exp 函数**

```python
import torch
from torch.autograd.function import Function

class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        """
        前向传播：计算 e^x
        ctx: 上下文对象，用于保存反向传播需要的信息
        """
        result = i.exp()
        ctx.save_for_backward(result)  # 保存结果用于反向传播
        return result
  
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：计算梯度
        grad_output: 来自上一层的梯度
        """
        result, = ctx.saved_tensors  # 取出保存的结果
        grad_input = grad_output * result  # d(e^x)/dx = e^x
        return grad_input

# 使用自定义 Function
x = torch.tensor([1.], requires_grad=True)
y = Exp.apply(x)  # 需要使用 apply 方法调用
print(f"y = e^x = {y}")  # tensor([2.7183], grad_fn=<ExpBackward>)

y.backward()
print(f"dy/dx = {x.grad}")  # tensor([2.7183]) = e^1
```

**注意事项：**

- `forward` 和 `backward` 的第一个参数都是 `ctx`（上下文对象）
- `backward` 返回的参数个数必须与 `forward` 的输入参数个数相同
- 不需要梯度的参数，`backward` 中返回 `None`
- 使用 `ctx.save_for_backward()` 保存反向传播需要的数据
- 调用时使用 `Function.apply()` 方法

#### 2.2.3 Autograd 相关的重要知识点

在使用 autograd 时，有几个重要的知识点需要注意，这些知识点对于正确使用 PyTorch 进行训练至关重要。

##### 知识点 1：梯度不会自动清零

PyTorch 的梯度会**累积**，不会自动清零。每次调用 `backward()` 时，梯度会累加到 `.grad` 属性中。

```python
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

for i in range(4):
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)
  
    y.backward()
    print(f"第{i+1}次 w.grad: {w.grad}")  # 梯度会累加
  
    # 通常需要手动清零
    # w.grad.zero_()
```

**输出：**

```
第1次 w.grad: tensor([5.])
第2次 w.grad: tensor([10.])
第3次 w.grad: tensor([15.])
第4次 w.grad: tensor([20.])
```

**解决方案：** 在训练循环中，每次反向传播前调用 `optimizer.zero_grad()` 清零梯度。

##### 知识点 2：依赖于叶子节点的节点，requires_grad 默认为 True

如果节点的运算依赖于叶子节点（`requires_grad=True`），那么该节点的 `requires_grad` 会自动设置为 `True`。

```python
import torch

w = torch.tensor([1.], requires_grad=True)  # 叶子节点
x = torch.tensor([2.], requires_grad=True)  # 叶子节点

a = torch.add(w, x)  # 依赖于叶子节点
b = torch.add(w, 1)  # 依赖于叶子节点
y = torch.mul(a, b)  # 依赖于叶子节点

print(f"a.requires_grad: {a.requires_grad}")  # True
print(f"b.requires_grad: {b.requires_grad}")  # True
print(f"y.requires_grad: {y.requires_grad}")  # True

print(f"a.is_leaf: {a.is_leaf}")  # False
print(f"b.is_leaf: {b.is_leaf}")  # False
print(f"y.is_leaf: {y.is_leaf}")  # False
```

##### 知识点 3：叶子节点不可以执行 in-place 操作

叶子节点（`is_leaf=True` 且 `requires_grad=True`）不允许执行 in-place 操作（如 `+=`、`add_()` 等），因为计算图的 backward 过程依赖于叶子节点的值。

```python
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

# 错误：叶子节点不能执行 in-place 操作
# w.add_(1)  # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.

# 正确：创建新张量
w = w + 1  # 或 w = torch.add(w, 1)
```

##### 知识点 4：detach 的作用

`detach()` 可以从计算图中剥离出数据，返回一个新张量，新张量与旧张量**共享数据**，但不再参与梯度计算。

```python
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()

# detach 后，新张量与旧张量共享数据
w_detach = w.detach()
w_detach.data[0] = 999
print(f"w: {w}")  # tensor([999.], requires_grad=True) - w 也被修改了

# detach 后的张量不参与梯度计算
w_detach.requires_grad = False
```

##### 知识点 5：with torch.no_grad() 的作用

`torch.no_grad()` 是一个上下文管理器，用于禁用梯度计算，可以加快速度并节省内存。在推理（inference）时特别有用。

```python
import torch

x = torch.tensor([1.], requires_grad=True)
y = x ** 2

# 不使用 no_grad：会构建计算图
z1 = y * 2
print(f"z1.requires_grad: {z1.requires_grad}")  # True

# 使用 no_grad：不构建计算图
with torch.no_grad():
    z2 = y * 2
    print(f"z2.requires_grad: {z2.requires_grad}")  # False
```

**使用场景：**

- 模型推理时
- 更新参数时（如 `optimizer.step()`）
- 评估模型性能时

#### 总结

Autograd 是 PyTorch 自动微分的核心机制，通过计算图（DAG）实现：

- **前向传播**：执行操作并记录梯度函数
- **反向传播**：使用链式法则计算梯度
- **关键函数**：`backward()`、`grad()`、`Function`
- **重要注意**：梯度累积、叶子节点限制、detach 和 no_grad 的使用

掌握这些知识点对于理解 PyTorch 的训练流程和调试梯度问题非常重要。

> 参考：[PyTorch Autograd 官方文档](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

---

## 第三章：数据模块

本章介绍 PyTorch 的数据处理模块，包括 Dataset（数据集）、DataLoader（数据加载器）和 Sampler（采样器）。这些是深度学习训练流程中的重要组成部分。

### 为什么需要数据模块？

在深度学习中，我们通常需要处理大量数据（成千上万甚至百万级别的图片、文本等）。如果直接将所有数据加载到内存中会导致：

- **内存溢出**：数据集太大，内存放不下
- **训练效率低**：无法利用 GPU 的并行计算能力
- **代码复杂**：需要手动处理批量加载、数据打乱、多进程等操作

PyTorch 的数据模块提供了一套优雅的解决方案：

```
硬盘中的数据
    ↓
Dataset（定义如何读取单个样本）
    ↓
Sampler（定义采样策略，可选）
    ↓
DataLoader（批量加载、多进程、打乱）
    ↓
模型训练
```

**数据加载的核心思想：**

1. **延迟加载**：初始化时只读取数据路径，使用时才加载数据
2. **批量处理**：将多个样本组成一个 batch，提高 GPU 利用率
3. **并行加载**：使用多进程预先加载数据，避免 GPU 等待

### 3.1 Dataset：数据集的抽象类

#### 3.1.1 Dataset 的基本概念

**什么是 Dataset？**

`torch.utils.data.Dataset` 是 PyTorch 中表示数据集的抽象类。你可以把它理解为一个"数据清单"，它告诉 PyTorch：

- 我有多少个样本（`__len__` 方法）
- 如何获取第 i 个样本（`__getitem__` 方法）

**必须实现的三个方法：**

| 方法                   | 作用                                   | 返回值        |
| ---------------------- | -------------------------------------- | ------------- |
| `__init__()`         | 初始化数据集，设置数据路径、转换方法等 | 无            |
| `__getitem__(index)` | 根据索引获取一个样本                   | (data, label) |
| `__len__()`          | 返回数据集的大小                       | int           |

**为什么这样设计？**

这种设计的好处是：

- **节省内存**：初始化时不加载数据，只记录路径
- **灵活性**：可以动态地读取和处理数据
- **支持大数据集**：即使数据集有百万张图片，也不会占用太多内存

**简单类比：**

把 Dataset 想象成一个图书馆的目录卡片系统：

- `__init__`：建立目录索引（记录每本书在哪个书架）
- `__len__`：告诉你图书馆有多少本书
- `__getitem__`：根据编号去书架上取书

所有自定义数据集都应该继承它并重写以下方法：

- `__init__()`: 初始化数据集，设置数据路径、转换方法等
- `__getitem__(index)`: 根据索引获取一个样本
- `__len__()`: 返回数据集的大小

**基本结构：**

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """初始化数据集"""
        self.data_dir = data_dir
        self.transform = transform  # 数据预处理/转换函数
        # 读取数据信息（路径、标签等）

    def __getitem__(self, index):
        """根据索引返回一个样本"""
        # 1. 根据 index 读取数据
        # 2. 应用 transform（如果有的话）
        # 3. 返回数据和标签
        return data, label

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
```

**什么是 transform？**

`transform` 是一个**数据预处理函数**

，用于对原始数据进行转换。在图像任务中，常见的转换包括：

- 调整图片大小（Resize）
- 转换为 Tensor
- 数据标准化（Normalization）
- 数据增强（随机翻转、裁剪等）

**为什么需要 transform？**

1. **统一尺寸**：神经网络要求输入尺寸一致，但原始图片大小可能不同
2. **转换格式**：PIL Image → Tensor，方便 PyTorch 处理
3. **数据增强**：通过随机变换增加数据多样性，提高模型泛化能力
4. **标准化**：将数据缩放到合适的范围，加速训练

**transform 的使用示例：**

```python
from torchvision import transforms
from PIL import Image

# 定义 transform：将图片转换为 224x224 的 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),    # 调整大小
    transforms.ToTensor(),            # PIL Image → Tensor
    transforms.Normalize(             # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 使用 transform
img = Image.open('cat.jpg')           # PIL Image, 可能是 (400, 300)
img_tensor = transform(img)           # Tensor, shape: [3, 224, 224]

print(type(img))         # <class 'PIL.Image.Image'>
print(type(img_tensor))  # <class 'torch.Tensor'>
print(img_tensor.shape)  # torch.Size([3, 224, 224])
```

**在 Dataset 中使用 transform：**

```python
class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform  # 保存 transform

    def __getitem__(self, index):
        # 1. 读取原始图片
        img = Image.open(self.img_paths[index])  # PIL Image

        # 2. 应用 transform（如果提供了）
        if self.transform is not None:
            img = self.transform(img)  # 转换为 Tensor

        # 3. 返回数据和标签
        return img, label

# 使用时传入 transform
dataset = MyDataset(
    data_dir='./data',
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)
```

**关键理解：**

- `transform=None` 表示这是一个**可选参数**
- 如果不传 transform，就返回原始数据（PIL Image）
- 如果传了 transform，就返回处理后的数据（Tensor）
- transform 在 `__getitem__` 中调用，每次读取数据时都会应用

#### 3.1.2 自定义 Dataset

**Dataset 的设计思路：**

1. **初始化阶段**：读取数据的元信息（路径、标签），而不是读取数据本身
2. **获取阶段**：在 `__getitem__` 中才真正从硬盘读取数据
3. **好处**：节省内存，支持大规模数据集

**新手入门示例：最简单的 Dataset**

让我们从一个最简单的例子开始，理解 Dataset 的工作原理：

```python
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """最简单的 Dataset 示例"""

    def __init__(self, data_list):
        """
        data_list: 数据列表，例如 [(x1, y1), (x2, y2), ...]
        """
        self.data = data_list

    def __getitem__(self, index):
        """返回第 index 个样本"""
        return self.data[index]

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

# 使用示例
data = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
dataset = SimpleDataset(data)

print(f"数据集大小: {len(dataset)}")  # 5
print(f"第0个样本: {dataset[0]}")      # (1, 10)
print(f"第2个样本: {dataset[2]}")      # (3, 30)

# 可以像列表一样遍历
for i, (x, y) in enumerate(dataset):
    print(f"样本 {i}: x={x}, y={y}")
```

**输出：**

```
数据集大小: 5
第0个样本: (1, 10)
第2个样本: (3, 30)
样本 0: x=1, y=10
样本 1: x=2, y=20
样本 2: x=3, y=30
样本 3: x=4, y=40
样本 4: x=5, y=50
```

**关键理解：**

- `dataset[0]` 会自动调用 `__getitem__(0)`
- `len(dataset)` 会自动调用 `__len__()`
- 可以像操作列表一样操作 Dataset

**示例：COVID-19 数据集（形式2 - 标签在文件夹中）**

```python
import os
from torch.utils.data import Dataset
from PIL import Image

class COVID19Dataset_2(Dataset):
    """
    数据集形式-2：数据的划分及标签在文件夹中体现
    目录结构：
    train/
        no-finding/
            img1.png
            img2.png
        covid-19/
            img3.png
            img4.png
    """

    def __init__(self, root_dir, transform=None):
        """
        获取数据集的路径、预处理的方法
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        # 标签映射字典
        self.str_2_int = {"no-finding": 0, "covid-19": 1}

        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量 index，从硬盘中读取数据，并预处理，转为 Tensor
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception(f"\ndata_dir:{self.root_dir} is empty!")
        return len(self.img_info)

    def _get_img_info(self):
        """
        读取数据集信息，将硬盘中的数据路径、标签读取进来
        """
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpeg"):
                    path_img = os.path.join(root, file)
                    sub_dir = os.path.basename(root)
                    label_int = self.str_2_int[sub_dir]
                    self.img_info.append((path_img, label_int))
```

**示例：COVID-19 数据集（形式3 - 标签在 CSV 中）**

```python
import pandas as pd

class COVID19Dataset_3(Dataset):
    """
    数据集形式-3：数据的划分及标签在 CSV 中
    CSV 格式：
    img-name, label, set-type
    img1.png, 0, train
    img2.png, 1, valid
    """

    def __init__(self, root_dir, path_csv, mode, transform=None):
        """
        mode: str, 'train' 或 'valid'
        """
        self.root_dir = root_dir
        self.path_csv = path_csv
        self.mode = mode
        self.transform = transform
        self.img_info = []

        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception(f"\ndata_dir:{self.root_dir} is empty!")
        return len(self.img_info)

    def _get_img_info(self):
        """
        从 CSV 中读取数据信息
        """
        df = pd.read_csv(self.path_csv)
        # 只保留对应模式的数据
        df.drop(df[df["set-type"] != self.mode].index, inplace=True)
        df.reset_index(inplace=True)  # 非常重要！pandas 的 drop 不会改变 index

        # 遍历表格，获取每张样本信息
        for idx in range(len(df)):
            path_img = os.path.join(self.root_dir, df.loc[idx, "img-name"])
            label_int = int(df.loc[idx, "label"])
            self.img_info.append((path_img, label_int))
```

#### 3.1.3 常见数据集组织形式

在实际项目中，数据集的组织方式多种多样。以下是三种最常见的形式：

| 形式                    | 描述                        | 目录结构示例                                                                                    | 优点                                      | 缺点                                  | 适用场景                           |
| ----------------------- | --------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------- | ---------------------------------- |
| **形式1：TXT**    | 数据路径和标签在 txt 文件中 | `train.txt<br>``img1.jpg 0<br>``img2.jpg 1`                                                   | 简单直观，易于编辑                        | 需要手动维护 txt 文件                 | 小型数据集，快速原型               |
| **形式2：文件夹** | 标签信息体现在文件夹名称中  | `train/<br>``├── cat/<br>``│   ├── img1.jpg<br>``└── dog/<br>``    └── img2.jpg` | 组织清晰，易于管理 `<br>`符合直觉       | 不适合多标签任务 `<br>`移动文件麻烦 | 单标签分类任务 `<br>`ImageNet 等 |
| **形式3：CSV**    | 数据信息在 CSV 表格中       | `data.csv<br>``img,label,split<br>``img1.jpg,0,train`                                         | 灵活，支持多种元信息 `<br>`便于数据分析 | 需要 pandas 库 `<br>`相对复杂       | 复杂数据集 `<br>`多标签任务      |

**形式1：TXT 文件示例**

```
# train.txt 内容
data/images/cat_001.jpg 0
data/images/dog_001.jpg 1
data/images/cat_002.jpg 0
data/images/dog_002.jpg 1
```

```python
class TXTDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.transform = transform
        self.img_info = []

        # 读取 txt 文件
        with open(txt_path, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                self.img_info.append((img_path, int(label)))

    def __getitem__(self, index):
        img_path, label = self.img_info[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_info)
```

**形式2：文件夹结构（最常用）**

这是最直观的组织方式，PyTorch 提供了 `ImageFolder` 类来直接读取这种结构：

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

# 使用 PyTorch 内置的 ImageFolder
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 自动识别文件夹名称作为标签
dataset = ImageFolder(root='data/train', transform=transform)

# ImageFolder 会自动：
# 1. 遍历所有子文件夹
# 2. 将文件夹名称映射为类别索引
# 3. 读取所有图片

print(dataset.classes)       # ['cat', 'dog']
print(dataset.class_to_idx)  # {'cat': 0, 'dog': 1}
print(len(dataset))          # 样本总数
```

**形式3：CSV 文件（最灵活）**

CSV 格式可以存储更多元信息，适合复杂场景：

```csv
img_name,label,set_type,age,gender
img_001.jpg,0,train,25,male
img_002.jpg,1,valid,30,female
img_003.jpg,0,train,28,male
```

**选择建议：**

- **新手学习**：推荐形式2（文件夹），使用 `ImageFolder`
- **快速原型**：推荐形式1（TXT），简单直接
- **生产环境**：推荐形式3（CSV），便于管理和扩展
- **大型数据集**：考虑使用数据库或专门的数据格式（如 LMDB、HDF5）

### 3.2 DataLoader：数据加载器

#### 3.2.1 DataLoader 的作用

**为什么需要 DataLoader？**

Dataset 只能一次返回一个样本，但在训练神经网络时，我们需要：

- 一次处理多个样本（batch）以提高 GPU 利用率
- 每个 epoch 打乱数据顺序，避免模型记住数据顺序
- 使用多进程预加载数据，避免 GPU 等待数据

DataLoader 就是为了解决这些问题而设计的。

**DataLoader 做了什么？**

`torch.utils.data.DataLoader` 是 PyTorch 的数据加载器，负责：

1. **批量加载**：将数据组织成 batch（例如：32 张图片打包成一个 batch）
2. **打乱数据**：训练时打乱数据顺序（shuffle）
3. **并行加载**：多进程加载数据，提高效率
4. **自动拼接**：将多个样本自动拼接成一个 batch tensor

**可视化理解：**

```
Dataset: [样本0, 样本1, 样本2, 样本3, 样本4, 样本5, 样本6, 样本7, 样本8, 样本9]
         ↓ (batch_size=3, shuffle=False)
DataLoader:
  Batch 0: [样本0, 样本1, 样本2]  → shape: [3, ...]
  Batch 1: [样本3, 样本4, 样本5]  → shape: [3, ...]
  Batch 2: [样本6, 样本7, 样本8]  → shape: [3, ...]
  Batch 3: [样本9]                → shape: [1, ...] (最后一个batch不完整)
```

**简单示例：理解 DataLoader 的作用**

```python
from torch.utils.data import Dataset, DataLoader

# 创建一个简单的数据集
class NumberDataset(Dataset):
    def __init__(self, n):
        self.data = list(range(n))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

# 创建数据集：包含 0-9 共 10 个数字
dataset = NumberDataset(10)

# 创建 DataLoader
loader = DataLoader(dataset, batch_size=3, shuffle=False)

# 查看每个 batch
for batch_idx, batch_data in enumerate(loader):
    print(f"Batch {batch_idx}: {batch_data}")

# 输出：
# Batch 0: tensor([0, 1, 2])
# Batch 1: tensor([3, 4, 5])
# Batch 2: tensor([6, 7, 8])
# Batch 3: tensor([9])
```

**关键观察：**

- DataLoader 自动将数据分成多个 batch
- 每个 batch 是一个 tensor（自动转换）
- 最后一个 batch 可能不完整（只有 1 个元素）

**基本使用：**

```python
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# 定义数据预处理
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# 创建数据集
train_set = AntsBeesDataset(root_dir, transform=transforms_train)

# 创建数据加载器
train_loader = DataLoader(
    dataset=train_set,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=False
)

# 迭代数据
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # inputs: [batch_size, channels, height, width]
        # labels: [batch_size]
        print(inputs.shape, labels.shape)
```

#### 3.2.2 DataLoader 的重要参数

| 参数               | 类型     | 说明                              | 默认值   |
| ------------------ | -------- | --------------------------------- | -------- |
| `dataset`        | Dataset  | 数据集对象                        | 必需     |
| `batch_size`     | int      | 每个 batch 的样本数量             | 1        |
| `shuffle`        | bool     | 是否在每个 epoch 开始时打乱数据   | False    |
| `sampler`        | Sampler  | 自定义采样策略（与 shuffle 互斥） | None     |
| `batch_sampler`  | Sampler  | 批量采样器                        | None     |
| `num_workers`    | int      | 多进程加载数据的进程数            | 0        |
| `collate_fn`     | callable | 如何将多个样本拼接成一个 batch    | 默认拼接 |
| `pin_memory`     | bool     | 是否将数据拷贝到 CUDA 固定内存中  | False    |
| `drop_last`      | bool     | 是否丢弃最后不足一个 batch 的数据 | False    |
| `timeout`        | numeric  | 数���读取超时时间              | 0        |
| `worker_init_fn` | callable | 每个 worker 的初始化函数          | None     |

**重要参数详解：**

**1. batch_size：批量大小**

```python
# batch_size=2：每次返回 2 个样本
train_loader_bs2 = DataLoader(dataset=train_set, batch_size=2)
# 输出：[2, 3, 224, 224]

# batch_size=3：每次返回 3 个样本
train_loader_bs3 = DataLoader(dataset=train_set, batch_size=3)
# 输出：[3, 3, 224, 224]
```

**2. drop_last：是否丢弃最后不完整的 batch**

```python
# 假设数据集有 10 个样本，batch_size=3
# drop_last=False（默认）：返回 4 个 batch，最后一个 batch 只有 1 个样本
train_loader = DataLoader(dataset=train_set, batch_size=3, drop_last=False)
# batch 0: [3, ...], batch 1: [3, ...], batch 2: [3, ...], batch 3: [1, ...]

# drop_last=True：丢弃最后不足的 batch，只返回 3 个完整的 batch
train_loader_drop = DataLoader(dataset=train_set, batch_size=3, drop_last=True)
# batch 0: [3, ...], batch 1: [3, ...], batch 2: [3, ...]
```

**3. shuffle：是否打乱数据**

- `shuffle=True`：每个 epoch 开始时重新打乱数据顺序（训练集常用）
- `shuffle=False`：保持数据原有顺序（验证集/测试集常用）

**注意**：`shuffle` 和 `sampler` 参数互斥，不能同时使用。

**4. num_workers：多进程加载**

- `num_workers=0`：主进程加载数据（默认）
- `num_workers>0`：使用多进程加载数据，加快速度
- 建议值：`num_workers = 4 * num_GPU`

**5. collate_fn：自定义批量拼接**

`collate_fn` 用于自定义如何将多个样本合并成一个 batch。默认行为是简单地堆叠张量。

**默认 collate_fn 的行为：**

```python
# 默认情况下，DataLoader 做了什么？
samples = [(img1, label1), (img2, label2), (img3, label3)]
# ↓ 默认 collate_fn
batch_imgs = torch.stack([img1, img2, img3])      # [3, C, H, W]
batch_labels = torch.tensor([label1, label2, label3])  # [3]
```

**何时需要自定义 collate_fn？**

1. **处理变长序列**（如文本、时间序列）
2. **过滤无效样本**（返回 None 的样本）
3. **返回字典格式**而非元组
4. **复杂的数据结构**（多个输入、元数据等）

**示例1：处理变长文本序列**

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn_text(batch):
    """
    处理变长文本序列
    batch: [(seq1, label1), (seq2, label2), ...]
    其中 seq 的长度可能不同
    """
    sequences, labels = zip(*batch)

    # 填充到相同长度
    sequences_padded = pad_sequence(
        sequences,
        batch_first=True,  # [batch, seq_len]
        padding_value=0    # 填充值
    )
    labels = torch.tensor(labels)

    return sequences_padded, labels

# 使用
train_loader = DataLoader(
    dataset=text_dataset,
    batch_size=32,
    collate_fn=collate_fn_text
)

# 示例数据
for sequences, labels in train_loader:
    print(sequences.shape)  # [32, max_seq_len]
    print(labels.shape)     # [32]
    break
```

**示例2：过滤无效样本**

```python
def collate_fn_filter(batch):
    """过滤掉 None 样本"""
    # 过滤 None
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # 使用默认的 collate
    return torch.utils.data.dataloader.default_collate(batch)

# 在 Dataset 中返回 None 表示无效样本
class MyDataset(Dataset):
    def __getitem__(self, index):
        try:
            img = Image.open(self.paths[index])
            return transform(img), self.labels[index]
        except:
            return None  # 损坏的图片返回 None
```

**示例3：返回字典格式**

```python
def collate_fn_dict(batch):
    """返回字典格式的 batch"""
    images, labels, paths = zip(*batch)

    return {
        'images': torch.stack(images),
        'labels': torch.tensor(labels),
        'paths': paths  # 保持为 tuple
    }

# 使用
for batch in train_loader:
    images = batch['images']
    labels = batch['labels']
    paths = batch['paths']
```

**示例4：多输入多输出**

```python
def collate_fn_multi(batch):
    """
    处理多输入多输出
    batch: [(img, mask, label, metadata), ...]
    """
    images, masks, labels, metadata = zip(*batch)

    return {
        'images': torch.stack(images),
        'masks': torch.stack(masks),
        'labels': torch.tensor(labels),
        'metadata': metadata
    }
```

**示例5：动态填充（保留序列长度信息）**

```python
def collate_fn_with_lengths(batch):
    """
    填充序列并返回原始长度
    用于 RNN 的 pack_padded_sequence
    """
    sequences, labels = zip(*batch)

    # 获取原始长度
    lengths = torch.tensor([len(seq) for seq in sequences])

    # 填充
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)

    return sequences_padded, labels, lengths

# 使用
for sequences, labels, lengths in train_loader:
    # 使用 pack_padded_sequence 处理
    from torch.nn.utils.rnn import pack_padded_sequence
    packed = pack_padded_sequence(
        sequences,
        lengths,
        batch_first=True,
        enforce_sorted=False
    )
```

**常见陷阱：**

```python
# 陷阱1：忘记处理空 batch
def collate_fn_bad(batch):
    batch = [item for item in batch if item is not None]
    # 如果 batch 为空会出错！
    return torch.stack([item[0] for item in batch])

# 正确做法
def collate_fn_good(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# 陷阱2：返回格式不一致
def collate_fn_inconsistent(batch):
    if len(batch) > 10:
        return torch.stack([item[0] for item in batch])
    else:
        return [item[0] for item in batch]  # 错误！格式不一致
```

### 3.3 Sampler：数据采样器

#### 3.3.1 Sampler 的作用

Sampler（采样器）用于控制 DataLoader 如何从数据集中采样数据。PyTorch 提供了多种采样器：

| 采样器                    | 说明         |
| ------------------------- | ------------ |
| `SequentialSampler`     | 顺序采样     |
| `RandomSampler`         | 随机采样     |
| `WeightedRandomSampler` | 加权随机采样 |
| `SubsetRandomSampler`   | 子集随机采样 |
| `BatchSampler`          | 批量采样     |

#### 3.3.2 WeightedRandomSampler：加权随机采样

`WeightedRandomSampler` 用于对数据集进行加权采样，常用于处理类别不平衡问题。

**主要参数：**

```python
torch.utils.data.WeightedRandomSampler(
    weights,           # 每个样本的采样权重
    num_samples,       # 采样的样本数量
    replacement=True   # 是否有放回采样
)
```

**使用步骤：**

```python
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

# 第一步：定义每个类别的采样权重
# 假设类别 0 和类别 1，希望类别 1 的采样概率是类别 0 的 5 倍
weights = torch.tensor([1, 5], dtype=torch.float)

# 第二步：生成每个样本的采样权重
train_targets = [sample[1] for sample in train_data.img_info]  # 获取所有标签
samples_weights = weights[train_targets]  # 根据标签分配权重

# 第三步：实例化 WeightedRandomSampler
sampler_w = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True  # 有放回采样
)

# 第四步：创建 DataLoader（注意：不能同时使用 shuffle）
train_loader = DataLoader(
    dataset=train_data,
    batch_size=2,
    sampler=sampler_w  # 使用自定义采样器
)

# 迭代数据
for epoch in range(10):
    for i, (inputs, target) in enumerate(train_loader):
        print(target)  # 可以看到类别 1 出现的频率更高
```

**输出示例：**

```
tensor([1, 1])
tensor([1, 0])
tensor([1, 1])
tensor([1, 1])
tensor([0, 1])
...
# 由于类别 1 的权重是类别 0 的 5 倍，可以看到 [1, 1] 出现的频率很高
```

#### 3.3.3 处理不平衡数据集

在实际应用中，数据集经常存在类别不平衡问题（某些类别样本很多，某些很少）。使用 `WeightedRandomSampler` 可以平衡采样。

**示例：CIFAR-10 不平衡数据集**

假设有 10 个类别，每个类别的样本数量不同：

```python
import collections
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

# 第一步：计算各类别的采样权重
train_targets = [sample[1] for sample in train_data.img_info]
label_counter = collections.Counter(train_targets)

# 统计每个类别的样本数量
class_sample_counts = [label_counter[k] for k in sorted(label_counter)]
print(f"各类别样本数量: {class_sample_counts}")
# 输出：[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# 计算权重（使用倒数，样本少的类别权重高）
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
print(f"各类别权重: {weights}")
# 输出：[0.0100, 0.0050, 0.0033, 0.0025, 0.0020, 0.0017, 0.0014, 0.0013, 0.0011, 0.0010]

# 第二步：生成每个样本的采样权重
samples_weights = weights[train_targets]

# 第三步：实例化 WeightedRandomSampler
sampler_w = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True
)

# 第四步：创建 DataLoader
train_loader_sampler = DataLoader(
    dataset=train_data,
    batch_size=16,
    sampler=sampler_w
)

# 验证采样效果
for epoch in range(10):
    label_count = []
    for i, (inputs, target) in enumerate(train_loader_sampler):
        label_count.extend(target.tolist())
    print(collections.Counter(label_count))

# 输出示例（使用 sampler 后，各类别采样数量趋于平衡）：
# Counter({0: 520, 1: 515, 2: 498, 3: 505, 4: 512, 5: 490, 6: 508, 7: 495, 8: 502, 9: 510})
```

**对比：不使用 sampler**

```python
train_loader = DataLoader(dataset=train_data, batch_size=16)

for epoch in range(10):
    label_count = []
    for i, (inputs, target) in enumerate(train_loader):
        label_count.extend(target.tolist())
    print(collections.Counter(label_count))

# 输出示例（不使用 sampler，采样数量与原始分布一致）：
# Counter({9: 1000, 8: 900, 7: 800, 6: 700, 5: 600, 4: 500, 3: 400, 2: 300, 1: 200, 0: 100})
```

### 3.4 Dataset 的常用操作

PyTorch 提供了一些工具函数来操作数据集。

#### 3.4.1 ConcatDataset：拼接数据集

`ConcatDataset` 用于将多个数据集拼接成一个大数据集。

```python
from torch.utils.data import ConcatDataset

# 创建多个数据集
train_data_1 = COVID19Dataset(root_dir_1, txt_path_1)
train_data_2 = COVID19Dataset_2(root_dir_2)
train_data_3 = COVID19Dataset_3(root_dir_3, csv_path, "train")

# 拼接数据集
train_set_all = ConcatDataset([train_data_1, train_data_2, train_data_3])

print(len(train_data_1))  # 2
print(len(train_data_2))  # 2
print(len(train_data_3))  # 2
print(len(train_set_all)) # 6 = 2 + 2 + 2
```

**使用场景：**

- 合并多个来源的数据集
- 合并不同格式的数据集

#### 3.4.2 Subset：子数据集

`Subset` 用于从数据集中抽取指定索引的样本，构成子数据集。

```python
from torch.utils.data import Subset

# 从 train_set_all 中抽取索引为 0, 1, 2, 5 的样本
train_sub_set = Subset(train_set_all, [0, 1, 2, 5])

print(len(train_sub_set))  # 4
```

**使用场景：**

- 快速验证代码（使用小数据集）
- 抽取特定样本进行分析

#### 3.4.3 random_split：随机划分数据集

`random_split` 用于将数据集随机划分成多个子集，常用于划分训练集和验证集。

```python
from torch.utils.data import random_split

# 将 train_set_all（共 6 个样本）随机划分为 4 和 2
set_split_1, set_split_2 = random_split(train_set_all, [4, 2])

print(len(set_split_1))  # 4
print(len(set_split_2))  # 2
```

**常用场景：**

```python
# 将训练集按 8:2 划分为训练集和验证集
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
```

**注意事项：**

- `random_split` 是随机划分，每次运行���果可能不同
- 如果需要固定划分结果，可以设置随机种子：

```python
import torch

# 设置随机种子
torch.manual_seed(42)
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
```

### 3.5 完整示例：数据加载流程

下面是一个完整的数据加载流程示例：

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

# 1. 定义 Dataset
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_info)

    def _get_img_info(self):
        # 读取数据信息
        pass

# 2. 定义数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 创建数据集
full_dataset = MyDataset(root_dir="data/", transform=train_transform)

# 4. 划分训练集和验证集
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# 5. 创建 DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 6. 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 计算验证指标
```

### 3.6 数据预处理与增强：transforms

在深度学习中，原始数据通常需要经过预处理才能输入模型。PyTorch 提供了 `torchvision.transforms` 模块来实现各种数据变换操作。

#### 3.6.1 transforms 概述

**什么是 transforms？**

transforms 是对图像数据进行预处理和增强的工具集，主要包括：

1. **数据预处理**：将数据转换为模型可以接受的格式（如 ToTensor、Normalize）
2. **数据增强**：通过随机变换增加数据多样性，提高模型泛化能力（如随机翻转、旋转）

**为什么需要 transforms？**

```python
from PIL import Image
import torch

# 原始图像数据
img = Image.open("cat.jpg")  # PIL Image 对象
print(type(img))  # <class 'PIL.Image.Image'>

# 模型需要的是 Tensor
# ❌ 不能直接输入模型
# output = model(img)  # 会报错

# ✅ 需要先转换为 Tensor
from torchvision import transforms
transform = transforms.ToTensor()
img_tensor = transform(img)
print(type(img_tensor))  # <class 'torch.Tensor'>
print(img_tensor.shape)  # torch.Size([3, 224, 224])
```

**transforms 的两大作用：**

1. **格式转换**：PIL Image → Tensor
2. **数值处理**：归一化、标准化等
3. **数据增强**：随机变换增加训练数据多样性

#### 3.6.2 Compose：组合多个 transforms

在实际应用中，我们通常需要对数据进行多步变换。`Compose` 可以将多个 transform 操作组合成一个。

**基本用法：**

```python
from torchvision import transforms
from PIL import Image

# 定义一系列变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # 1. 调整大小
    transforms.ToTensor(),                   # 2. 转为 Tensor
    transforms.Normalize(                     # 3. 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 应用变换
img = Image.open("cat.jpg")
img_transformed = transform(img)
print(img_transformed.shape)  # torch.Size([3, 224, 224])
```

**Compose 的工作原理：**

```python
# Compose 的内部实现（简化版）
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        # 依次执行每个 transform
        for t in self.transforms:
            img = t(img)  # 上一个的输出是下一个的输入
        return img
```

**重要提示：**

- transforms 是**按顺序**执行的
- 每个 transform 的输出是下一个 transform 的输入
- 顺序错误会导致报错（如在 ToTensor 之前使用 Normalize）

#### 3.6.3 常用数据预处理 transforms

这些 transforms 用于将数据转换为模型可接受的格式。

##### 1. ToTensor：转换为 Tensor

`ToTensor` 将 PIL Image 或 NumPy 数组转换为 Tensor。

```python
from torchvision import transforms
from PIL import Image

transform = transforms.ToTensor()
img = Image.open("cat.jpg")  # PIL Image, shape: (H, W, C), range: [0, 255]
img_tensor = transform(img)   # Tensor, shape: (C, H, W), range: [0.0, 1.0]

print(f"类型: {type(img_tensor)}")        # torch.Tensor
print(f"形状: {img_tensor.shape}")        # torch.Size([3, 224, 224])
print(f"数值范围: [{img_tensor.min()}, {img_tensor.max()}]")  # [0.0, 1.0]
```

**ToTensor 做了三件事：**

1. **格式转换**：PIL Image → Tensor
2. **维度调整**：(H, W, C) → (C, H, W)
3. **数值归一化**：[0, 255] → [0.0, 1.0]（除以 255）

##### 2. Normalize：标准化

`Normalize` 对 Tensor 进行标准化处理，公式为：`output = (input - mean) / std`

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                        std=[0.229, 0.224, 0.225])     # ImageNet 标准差
])

img = Image.open("cat.jpg")
img_normalized = transform(img)
```

**常用的 mean 和 std 值：**

```python
# ImageNet 数据集的统计值（RGB 三通道）
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 灰度图（单通道）
mean = [0.5]
std = [0.5]

# 自定义：根据自己的数据集计算
# 计算方法见后文
```

**注意事项：**

- `Normalize` 必须在 `ToTensor` **之后**使用
- `mean` 和 `std` 的长度必须与通道数匹配

##### 3. Resize：调整大小

`Resize` 将图像调整到指定大小。

```python
# 方式 1：指定 (height, width)
transform = transforms.Resize((256, 256))  # 调整为 256x256
img = transform(img)
print(img.size)  # (256, 256)

# 方式 2：指定短边长度，长边等比例缩放
transform = transforms.Resize(256)  # 短边调整为 256，长边等比例缩放
img = transform(Image.open("cat.jpg"))  # 假设原图是 400x300
print(img.size)  # (341, 256) - 保持宽高比
```

**使用建议：**

```python
# ✅ 推荐：Resize + CenterCrop 组合使用
# 这样可以保证所有图像大小一致，适合批处理
transform = transforms.Compose([
    transforms.Resize(256),        # 短边缩放到 256
    transforms.CenterCrop(224),    # 中心裁剪 224x224
    transforms.ToTensor()
])
```

**为什么 Resize(int) 可能导致错误？**

```python
# 问题示例
transform = transforms.Compose([
    transforms.Resize(5),  # 只指定一个 int
    transforms.ToTensor()
])

dataset = COVID19Dataset(img_dir, txt_path, transform=transform)
loader = DataLoader(dataset, batch_size=2)

# ❌ 可能报错：torch.stack() 期望所有 Tensor 形状相同
# 原因：不同图像的宽高比不同，Resize(5) 后形状可能是 (5, 3) 或 (5, 7)
# DataLoader 无法将不同形状的 Tensor 组合成 batch
```

**解决方法：**

```python
# 方法 1：使用 Resize((h, w)) 固定尺寸
transform = transforms.Compose([
    transforms.Resize((5, 5)),  # 固定大小
    transforms.ToTensor()
])

# 方法 2：Resize + CenterCrop
transform = transforms.Compose([
    transforms.Resize(5),
    transforms.CenterCrop(5),   # 裁剪成固定大小
    transforms.ToTensor()
])
```

##### 4. CenterCrop：中心裁剪

从图像中心裁剪指定大小的区域。

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # 从中心裁剪 224x224
    transforms.ToTensor()
])
```

##### 5. Pad：填充

在图像边缘填充指定像素。

```python
# 四边填充相同像素
transform = transforms.Pad(10)  # 四边各填充 10 像素

# 分别指定四边填充
transform = transforms.Pad((10, 20, 30, 40))  # (left, top, right, bottom)

# 指定填充颜色
transform = transforms.Pad(10, fill=255, padding_mode='constant')  # 白色填充
```

#### 3.6.4 常用数据增强 transforms

数据增强通过随机变换增加训练数据的多样性，提高模型泛化能力。

##### 1. RandomHorizontalFlip：随机水平翻转

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
    transforms.ToTensor()
])

# 训练时每次调用可能翻转也可能不翻转
for epoch in range(3):
    img_transformed = transform(img)
    # 每次结果可能不同
```

**使用场景：**

- 物体识别（猫、狗的方向不影响类别）
- 不适用于文字识别（翻转后文字变成镜像）

##### 2. RandomVerticalFlip：随机垂直翻转

```python
transform = transforms.RandomVerticalFlip(p=0.5)  # 50% 概率垂直翻转
```

**使用场景：**

- 卫星图像、医学影像
- 不适用于自然场景（天空通常在上方）

##### 3. RandomRotation：随机旋转

```python
# 随机旋转 -10 到 10 度
transform = transforms.RandomRotation(degrees=10)

# 指定旋转角度范围
transform = transforms.RandomRotation(degrees=(-30, 30))

# 填充旋转后的空白区域
transform = transforms.RandomRotation(degrees=10, fill=255)  # 白色填充
```

##### 4. RandomCrop：随机裁剪

```python
# 随机裁剪 224x224 区域
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),  # 随机位置裁剪
    transforms.ToTensor()
])

# 如果图像太小，可以先填充
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224, padding=4),  # 先填充 4 像素再裁剪
    transforms.ToTensor()
])
```

**RandomCrop vs CenterCrop：**

- `CenterCrop`：总是从中心裁剪，用于**验证集**
- `RandomCrop`：随机位置裁剪，用于**训练集**（数据增强）

##### 5. ColorJitter：颜色抖动

随机改变图像的亮度、对比度、饱和度和色调。

```python
transform = transforms.ColorJitter(
    brightness=0.2,    # 亮度变化范围 [0.8, 1.2]
    contrast=0.2,      # 对比度变化范围 [0.8, 1.2]
    saturation=0.2,    # 饱和度变化范围 [0.8, 1.2]
    hue=0.1            # 色调变化范围 [-0.1, 0.1]
)
```

**使用场景：**

- 提高模型对光照变化的鲁棒性
- 模拟不同拍摄条件

##### 6. GaussianBlur：高斯模糊

```python
# 高斯模糊，kernel_size 必须是奇数
transform = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
```

##### 7. RandomPerspective：随机透视变换

```python
# 随机透视变换（模拟不同视角）
transform = transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
```

#### 3.6.5 高级 transforms

##### 1. FiveCrop 和 TenCrop

`FiveCrop` 将图像裁剪为 5 个区域（四角 + 中心），`TenCrop` 额外包含翻转后的 5 个区域。

**FiveCrop 的使用：**

```python
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch

# ❌ 错误用法：FiveCrop 返回的是 tuple，不能直接传给 ToTensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.FiveCrop(224),
    transforms.ToTensor(),  # 报错！ToTensor 不接受 tuple
    transforms.Normalize([0.5], [0.5])
])
```

**为什么会报错？**

```python
# FiveCrop 返回一个包含 5 个图像的 tuple
img = Image.open("cat.jpg")
cropper = transforms.FiveCrop(224)
crops = cropper(img)
print(type(crops))  # <class 'tuple'>
print(len(crops))   # 5

# ToTensor 期望输入是单个图像，不能处理 tuple
# 所以会报错
```

**✅ 正确用法：使用 Lambda 处理**

```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.FiveCrop(224),
    # 使用 Lambda 将 5 个图像分别转换并堆叠
    transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
    transforms.Normalize([0.5], [0.5])
])

dataset = COVID19Dataset(img_dir, txt_path, transform=transform)
loader = DataLoader(dataset, batch_size=2)

for data, labels in loader:
    print(data.shape)  # torch.Size([2, 5, 1, 224, 224])
    # batch_size=2, ncrops=5, channels=1, height=224, width=224
```

**在模型中使用 FiveCrop：**

```python
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, cls_num=2):
        super(TinyCNN, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3)
        self.fc = nn.Linear(36, cls_num)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TinyCNN(2)

# 训练/推理
for data, labels in loader:
    bs, ncrops, c, h, w = data.size()
    # torch.Size([2, 5, 1, 224, 224])

    # 将 batch 和 crops 维度合并
    result = model(data.view(-1, c, h, w))  # [10, 2]

    # 将结果重新分组并平均
    result_avg = result.view(bs, ncrops, -1).mean(1)  # [2, 2]
    # 或者取最大值：result.view(bs, ncrops, -1).max(1)[0]

    print(f"每个 crop 的结果: {result.shape}")      # [10, 2]
    print(f"平均后的结果: {result_avg.shape}")       # [2, 2]
```

**FiveCrop 的应用场景：**

- **测试阶段**：对一张图像的 5 个裁剪分别预测，然后平均结果
- **提高预测准确性**：比单次预测更稳定
- **不推荐训练时使用**：会增加 5 倍计算量

**TenCrop 用法：**

```python
# TenCrop = FiveCrop + 翻转后的 FiveCrop
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
    transforms.Normalize([0.5], [0.5])
])

# 使用方法与 FiveCrop 相同，只是 ncrops=10
```

##### 2. RandomChoice：随机选择一个 transform

从多个 transforms 中随机选择一个执行。

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomChoice([
        transforms.Pad(10),                      # 或者填充 10 像素
        transforms.RandomVerticalFlip(p=1),      # 或者垂直翻转
        transforms.ColorJitter(brightness=0.5)   # 或者调整亮度
    ]),
    transforms.ToTensor()
])

# 每次调用只会执行其中一个变换
for i in range(3):
    img_transformed = transform(img)
    # 三次结果不同，每次只应用一个变换
```

##### 3. RandomOrder：随机打乱执行顺序

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomOrder([
        transforms.Pad((0, 100, 0, 0)),          # 顶部填充 100 像素
        transforms.RandomVerticalFlip(p=1)        # 垂直翻转
    ]),
    transforms.ToTensor()
])

# 执行顺序随机：
# - 有时先填充再翻转（黑边在底部）
# - 有时先翻转再填充（黑边在顶部）
```

**RandomOrder 的效果：**

- 顺序 1：Pad → Flip → 黑边在图像底部
- 顺序 2：Flip → Pad → 黑边在图像顶部

##### 4. RandomApply：随机应用一组 transforms

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.Pad((0, 100, 0, 0)),
        transforms.RandomVerticalFlip(p=1)
    ], p=0.5),  # 50% 概率应用这组变换
    transforms.ToTensor()
])

# 结果：
# - 50% 概率：同时执行 Pad 和 Flip
# - 50% 概率：不执行任何变换
```

**RandomChoice vs RandomApply：**

- `RandomChoice`：从多个变换中选一个
- `RandomApply`：一组变换要么全部执行，要么都不执行

##### 5. AutoAugment：自动数据增强

AutoAugment 是 Google 提出的自动数据增强策略，针对不同数据集优化。

```python
# 选择预定义的策略
policy = transforms.AutoAugmentPolicy.CIFAR10   # CIFAR-10 数据集
# policy = transforms.AutoAugmentPolicy.IMAGENET  # ImageNet 数据集
# policy = transforms.AutoAugmentPolicy.SVHN      # SVHN 数据集

transform = transforms.Compose([
    transforms.AutoAugment(policy),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# AutoAugment 会自动应用一系列优化的变换组合
```

**AutoAugment 的特点：**

- 包含 25 种变换操作（旋转、平移、颜色变换等）
- 针对特定数据集优化的策略
- 可以显著提升模型性能

##### 6. RandAugment：简化的自动增强

RandAugment 是 AutoAugment 的简化版本，只需要调整两个参数。

```python
transform = transforms.Compose([
    transforms.RandAugment(
        num_ops=2,           # 随机选择 2 个操作
        magnitude=9,         # 操作强度（0-30）
        num_magnitude_bins=31
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```

**RandAugment 的优势：**

- 只需调整 2 个超参数（num_ops, magnitude）
- AutoAugment 需要针对每个数据集搜索策略
- 性能接近 AutoAugment，但更简单

##### 7. TrivialAugmentWide：无需调参的增强

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(),  # 无需任何参数
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```

**TrivialAugmentWide 的特点：**

- 完全无需调参
- 从 14 种变换中随机选择一个，随机强度
- 适合快速实验

**三种自动增强对比：**

| 方法 | 超参数数量 | 计算成本 | 适用场景 |
|------|-----------|---------|---------|
| AutoAugment | 需要搜索策略 | 高 | 追求最佳性能 |
| RandAugment | 2 个（num_ops, magnitude） | 中 | 平衡性能和简单性 |
| TrivialAugmentWide | 0 个 | 低 | 快速实验 |

#### 3.6.6 自定义 transforms

有时内置的 transforms 无法满足需求，需要自定义变换。

**方式 1：使用 Lambda**

```python
# 简单的自定义变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2),  # 将像素值乘以 2
    transforms.Lambda(lambda x: x.clamp(0, 1))  # 限制在 [0, 1]
])
```

**方式 2：定义类**

```python
class AddGaussianNoise:
    """添加高斯噪声"""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): 输入的 Tensor 图像
        Returns:
            Tensor: 添加噪声后的 Tensor
        """
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# 使用自定义 transform
transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(mean=0, std=0.1),
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
])
```

**方式 3：更复杂的自定义**

```python
class RandomErasing:
    """随机擦除：随机选择图像中的矩形区域并擦除"""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img

        # 计算擦除区域
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for _ in range(100):
            target_area = torch.empty(1).uniform_(*self.scale) * area
            aspect_ratio = torch.empty(1).uniform_(*self.ratio)

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            if w < img_w and h < img_h:
                i = torch.randint(0, img_h - h + 1, (1,)).item()
                j = torch.randint(0, img_w - w + 1, (1,)).item()
                img[:, i:i+h, j:j+w] = self.value
                return img

        return img

# 使用
transform = transforms.Compose([
    transforms.ToTensor(),
    RandomErasing(p=0.5)
])
```

#### 3.6.7 训练集 vs 验证集的 transforms 策略

**核心原则：**

- **训练集**：使用数据增强，增加数据多样性
- **验证集/测试集**：只做基本预处理，不做随机增强

**标准范式：**

```python
from torchvision import transforms

# 训练集 transforms：包含数据增强
train_transform = transforms.Compose([
    # 1. 调整大小
    transforms.Resize(256),

    # 2. 数据增强（随机变换）
    transforms.RandomCrop(224),              # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ColorJitter(                   # 颜色抖动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomRotation(15),            # 随机旋转

    # 3. 转换为 Tensor
    transforms.ToTensor(),

    # 4. 标准化
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 验证集 transforms：只做基本预处理
valid_transform = transforms.Compose([
    # 1. 调整大小
    transforms.Resize(256),

    # 2. 中心裁剪（不是随机裁剪）
    transforms.CenterCrop(224),

    # 3. 转换为 Tensor
    transforms.ToTensor(),

    # 4. 标准化（与训练集相同）
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 创建数据集
train_dataset = MyDataset(train_dir, transform=train_transform)
valid_dataset = MyDataset(valid_dir, transform=valid_transform)
```

**为什么验证集不用数据增强？**

1. **保持一致性**：验证集应该反映真实数据的分布
2. **可复现性**：每次验证结果应该相同，便于对比模型
3. **公平比较**：所有模型在相同的验证集上评估

**不同任务的 transforms 策略：**

```python
# 1. 图像分类（如 ImageNet）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. 物体检测（需要同时变换图像和边界框）
# 注意：物体检测通常使用 albumentations 库，见下节

# 3. 医学影像（不能随意翻转旋转）
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.2),  # 只调整亮度
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 4. 人脸识别（保持人脸方向）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),       # 只水平翻转
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```

#### 3.6.8 最佳实践和常见错误

##### 最佳实践

**1. 计算数据集的 mean 和 std**

```python
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

def compute_mean_std(dataset):
    """计算数据集的均值和标准差"""
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

# 使用
dataset = MyDataset(data_dir, transform=transforms.ToTensor())
mean, std = compute_mean_std(dataset)
print(f"Mean: {mean}")
print(f"Std: {std}")
```

**2. 可视化 transforms 效果**

```python
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def show_transform_effects(img, transform, num_samples=5):
    """可视化 transform 的效果"""
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))

    # 显示原图
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # 显示变换后的图像
    for i in range(1, num_samples + 1):
        img_transformed = transform(img)

        # 如果是 Tensor，转回 PIL Image
        if isinstance(img_transformed, torch.Tensor):
            img_transformed = transforms.ToPILImage()(img_transformed)

        axes[i].imshow(img_transformed)
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# 使用
img = Image.open("cat.jpg")
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3),
    transforms.RandomRotation(15)
])

show_transform_effects(img, transform, num_samples=5)
```

**3. transforms 的顺序很重要**

```python
# ✅ 正确顺序
transform = transforms.Compose([
    transforms.Resize(256),              # 1. PIL Image 操作
    transforms.RandomCrop(224),          # 2. PIL Image 操作
    transforms.RandomHorizontalFlip(),   # 3. PIL Image 操作
    transforms.ToTensor(),               # 4. PIL → Tensor
    transforms.Normalize([0.5], [0.5])   # 5. Tensor 操作
])

# ❌ 错误顺序
transform = transforms.Compose([
    transforms.ToTensor(),               # 先转 Tensor
    transforms.RandomCrop(224),          # ❌ RandomCrop 不接受 Tensor
    transforms.Normalize([0.5], [0.5])
])
```

**4. 使用 transforms 的 __repr__ 调试**

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(transform)
# 输出：
# Compose(
#     Resize(size=256, interpolation=bilinear)
#     RandomCrop(size=(224, 224), padding=None)
#     ToTensor()
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# )
```

##### 常见错误

**错误 1：ToTensor 之前使用 Normalize**

```python
# ❌ 错误
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.Normalize([0.5], [0.5]),  # ❌ Normalize 期望 Tensor
    transforms.ToTensor()
])

# ✅ 正确
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),               # 先转 Tensor
    transforms.Normalize([0.5], [0.5])   # 再 Normalize
])
```

**错误 2：通道数不匹配**

```python
# ❌ 错误：灰度图使用 RGB 的 mean/std
img = Image.open("gray.jpg").convert('L')  # 灰度图，1 通道
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 3 通道
])
# 报错：expected 3 channels, got 1

# ✅ 正确
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   # 1 通道
])
```

**错误 3：FiveCrop 后忘记使用 Lambda**

```python
# ❌ 错误
transform = transforms.Compose([
    transforms.FiveCrop(224),
    transforms.ToTensor(),  # ❌ ToTensor 不接受 tuple
])

# ✅ 正确
from torchvision.transforms import ToTensor
transform = transforms.Compose([
    transforms.FiveCrop(224),
    transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
])
```

**错误 4：验证集使用了随机增强**

```python
# ❌ 错误：验证集不应该随机增强
valid_transform = transforms.Compose([
    transforms.RandomCrop(224),          # ❌ 应该用 CenterCrop
    transforms.RandomHorizontalFlip(),   # ❌ 不应该随机翻转
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ 正确
valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),          # 固定裁剪
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```

#### 3.6.9 transforms 实战示例

下面是一个完整的使用 transforms 的示例：

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 1. 定义 Dataset
class COVID19Dataset(Dataset):
    def __init__(self, root_dir, txt_path, transform=None):
        self.root_dir = root_dir
        self.txt_path = txt_path
        self.transform = transform
        self.img_info = []
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('L')  # 灰度图

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_info)

    def _get_img_info(self):
        with open(self.txt_path, "r") as f:
            txt_data = f.read().strip().split("\n")

        self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                         for i in txt_data]

# 2. 定义 transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 3. 创建数据集和 DataLoader
train_dataset = COVID19Dataset(img_dir, train_txt, transform=train_transform)
valid_dataset = COVID19Dataset(img_dir, valid_txt, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 4. 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 计算验证指标
```

#### 3.6.10 总结

**transforms 的核心要点：**

1. **基本概念**
   - transforms 用于数据预处理和增强
   - 使用 Compose 组合多个变换
   - 变换按顺序执行

2. **常用预处理**
   - `ToTensor`：PIL Image → Tensor，自动归一化到 [0, 1]
   - `Normalize`：标准化，(x - mean) / std
   - `Resize`：调整大小
   - `CenterCrop`：中心裁剪

3. **常用增强**
   - `RandomHorizontalFlip`：随机水平翻转
   - `RandomCrop`：随机裁剪
   - `ColorJitter`：颜色抖动
   - `RandomRotation`：随机旋转

4. **高级技巧**
   - `FiveCrop`/`TenCrop`：多区域裁剪（测试时使用）
   - `AutoAugment`/`RandAugment`：自动数据增强
   - 自定义 transforms：使用 Lambda 或定义类

5. **最佳实践**
   - 训练集使用数据增强，验证集不使用
   - ToTensor 必须在 Normalize 之前
   - 注意通道数匹配
   - 根据任务选择合适的增强策略

> 参考：[torchvision.transforms 官方文档](https://pytorch.org/vision/stable/transforms.html)

---

### 3.7 常见问题与最佳实践

#### 3.7.1 常见问题 FAQ

**Q1: num_workers 应该设置为多少？**

```python
# 推荐设置
num_workers = 4  # 一般设置为 4-8

# 在 Windows 上可能需要设置为 0
num_workers = 0  # Windows 多进程可能有问题

# 根据 CPU 核心数设置
import os
num_workers = min(8, os.cpu_count())
```

**注意事项：**

- `num_workers=0`：主进程加载数据（最稳定，但慢）
- `num_workers>0`：使用多进程加载（更快，但在 Windows 上可能有问题）
- 过大的 `num_workers` 会占用太多内存

**Q2: 什么时候使用 pin_memory？**

```python
# 使用 GPU 训练时，建议开启
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,
    pin_memory=True  # 加快数据传输到 GPU 的速度
)

# 使用 CPU 训练时，不需要开启
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,
    pin_memory=False
)
```

**Q3: shuffle 和 sampler 能同时使用吗？**

不能！`shuffle` 和 `sampler` 参数是互斥的。

```python
# 错误：不能同时使用
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,
    shuffle=True,        # ❌
    sampler=my_sampler   # ❌
)

# 正确：二选一
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,
    shuffle=True  # 使用 shuffle
)

# 或者
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,
    sampler=my_sampler  # 使用自定义 sampler
)
```

**Q4: 为什么训练集要 shuffle，验证集不需要？**

```python
# 训练集：需要打乱，避免模型记住数据顺序
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# 验证集：不需要打乱，保持固定顺序便于复现结果
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
```

**原因：**

- 训练时打乱数据可以提高模型泛化能力
- 验证时固定顺序便于对比不同模型的性能

**Q5: drop_last 什么时候需要设置为 True？**

```python
# 一般情况：保留所有数据
train_loader = DataLoader(train_set, batch_size=32, drop_last=False)

# 使用 Batch Normalization 时：建议 drop_last=True
# 因为 BN 在 batch_size=1 时会出问题
train_loader = DataLoader(train_set, batch_size=32, drop_last=True)
```

#### 3.7.2 最佳实践

**1. 数据预处理的位置**

```python
# ✅ 推荐：在 Dataset 的 __getitem__ 中进行预处理
class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        # 只读取路径，不加载数据

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if self.transform:
            img = self.transform(img)  # 在这里进行预处理
        return img, label

# ❌ 不推荐：在 __init__ 中预处理所有数据
class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        for path in paths:
            img = Image.open(path)
            if transform:
                img = transform(img)  # 占用大量内存
            self.data.append(img)
```

**2. 训练集和验证集使用不同的 transform**

```python
from torchvision import transforms

# 训练集：使用数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),  # 随机翻转
    transforms.ColorJitter(),            # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 验证集：只做基本预处理，不做数据增强
valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = MyDataset(train_dir, transform=train_transform)
valid_set = MyDataset(valid_dir, transform=valid_transform)
```

**3. 合理设置 batch_size**

```python
# batch_size 的选择需要权衡：
# - 太小：训练慢，梯度估计不准确
# - 太大：占用显存，可能导致 OOM（Out of Memory）

# 常见的 batch_size 设置
batch_sizes = {
    "小模型 + 小图片": 128,
    "中等模型": 64,
    "大模型（ResNet50）": 32,
    "超大模型（Transformer）": 16,
}

# 如果遇到 OOM 错误，尝试减小 batch_size
try:
    train_loader = DataLoader(train_set, batch_size=64)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("显存不足，减小 batch_size")
        train_loader = DataLoader(train_set, batch_size=32)
```

**4. 使用 collate_fn 处理变长数据**

```python
# 对于变长序列（如文本），需要自定义 collate_fn
def collate_fn(batch):
    """
    batch: list of (data, label)
    """
    data_list, label_list = zip(*batch)

    # 填充到相同长度
    from torch.nn.utils.rnn import pad_sequence
    data_padded = pad_sequence(data_list, batch_first=True)
    labels = torch.tensor(label_list)

    return data_padded, labels

# 使用自定义 collate_fn
train_loader = DataLoader(
    train_set,
    batch_size=32,
    collate_fn=collate_fn
)
```

**5. 调试技巧：先用小数据集测试**

```python
# 使用 Subset 创建小数据集，快速验证代码
from torch.utils.data import Subset

# 只使用前 100 个样本进行调试
debug_set = Subset(train_set, range(100))
debug_loader = DataLoader(debug_set, batch_size=16)

# 快速测试一个 epoch
for inputs, labels in debug_loader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    print(f"Loss: {loss.item()}")
    break  # 只测试一个 batch
```

**6. 数据加载性能优化**

```python
# 性能优化建议
train_loader = DataLoader(
    dataset=train_set,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # 使用多进程
    pin_memory=True,         # 使用固定内存（GPU训练时）
    prefetch_factor=2,       # 每个 worker 预取的 batch 数量
    persistent_workers=True  # 保持 worker 进程（PyTorch 1.7+）
)
```

### 3.8 总结

本章介绍了 PyTorch 数据模块的核心组件：

1. **Dataset**：数据集的抽象类

   - 自定义 Dataset 需要实现 `__init__`、`__getitem__`、`__len__`
   - 支持多种数据组织形式（txt、文件夹、CSV）
2. **DataLoader**：数据加载器

   - 批量加载、打乱数据、并行加载
   - 重要参数：`batch_size`、`shuffle`、`num_workers`、`drop_last`
3. **Sampler**：采样器

   - `WeightedRandomSampler` 用于处理类别不平衡问题
   - 通过设置权重控制采样概率
4. **常用操作**

   - `ConcatDataset`：拼接多个数据集
   - `Subset`：抽取子数据集
   - `random_split`：随机划分数据集
5. **transforms**：数据预处理与增强

   - 数据预处理：ToTensor、Normalize、Resize 等
   - 数据增强：RandomCrop、RandomFlip、ColorJitter 等
   - 高级变换：FiveCrop、AutoAugment、RandAugment 等
   - 训练集使用数据增强，验证集只做基本预处理

掌握这些工具是进行深度学习训练的基础。

> 参考：[PyTorch 数据加载官方文档](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---
