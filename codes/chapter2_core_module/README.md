# 第二章：核心模块 - 代码示例

本目录包含第二章"核心模块"的所有代码示例，每个 notebook 对应一个核心概念。

## 文件说明

### 1. `01_tensor_basics.ipynb`
**主题：** 张量基础（结构、属性）

**内容：**
- 张量的基本概念（0维、1维、2维、高维）
- 张量的八个主要属性：data, dtype, shape, device, grad, grad_fn, requires_grad, is_leaf
- 设备检查和设备转换
- 叶子节点和非叶子节点的区别
- 常用数据类型（int8, int32, float32, float64, bool等）
- 综合示例：查看所有属性

### 2. `02_tensor_creation.ipynb`
**主题：** 张量的创建方法

**内容：**
- **直接创建**：
  - torch.tensor - 从数据创建
  - torch.from_numpy - 从numpy创建（共享内存）
  - torch.as_tensor - 尽可能共享内存
- **依据数值创建**：
  - torch.zeros, torch.ones, torch.full
  - torch.zeros_like, torch.ones_like, torch.full_like
  - torch.arange, torch.linspace, torch.logspace
  - torch.eye - 单位矩阵
- **依概率分布创建**：
  - torch.rand - 均匀分布
  - torch.randn - 标准正态分布
  - torch.normal - 指定参数的正态分布
  - torch.randint - 整数均匀分布
  - torch.randperm - 随机排列
  - torch.bernoulli - 伯努利分布
- 随机种子的设置和重要性

### 3. `03_tensor_operations.ipynb`
**主题：** 张量的操作（拼接、切分、索引）

**内容：**
- **拼接操作**：
  - torch.cat - 在现有维度上拼接
  - torch.stack - 在新维度上堆叠
  - torch.hstack, torch.vstack - 水平/垂直堆叠
- **切分操作**：
  - torch.split - 按大小切分
  - torch.chunk - 分成n份
  - torch.hsplit, torch.vsplit - 水平/垂直切分
- **索引操作**：
  - torch.gather - 高级索引
  - torch.index_select - 按索引选择
  - torch.masked_select - 根据掩码选择
  - 基本索引和切片

### 4. `04_tensor_shape.ipynb`
**主题：** 形状变换（view/reshape/squeeze/unsqueeze）

**内容：**
- **改变形状**：
  - reshape - 改变形状（可能复制数据）
  - view - 改变形状（共享内存，要求连续）
  - flatten - 展平张量
  - reshape vs view 的区别
- **增减维度**：
  - squeeze - 移除大小为1的维度
  - unsqueeze - 增加大小为1的维度
- **转置维度**：
  - transpose - 交换两个维度
  - permute - 重新排列所有维度
  - t() - 2D张量转置
- **实际应用**：
  - 批处理图像的形状变换
  - 添加批次维度
  - 连续性检查（is_contiguous）

### 5. `05_broadcasting.ipynb`
**主题：** 广播机制详解

**内容：**
- 什么是广播机制
- **广播规则**：
  - 从右向左对齐维度
  - 每个维度必须相同、为1或不存在
- 广播过程详解
- **常见广播模式**：
  - 标量与张量
  - 向量与矩阵（每行/每列操作）
  - 批处理操作
- 使用 unsqueeze 辅助广播
- **实际应用**：
  - 批归一化
  - 注意力机制中的加权
  - 特征归一化
- 性能优势对比（广播 vs 循环）

### 6. `06_scatter_gather.ipynb`
**主题：** scatter 和 gather 详解

**内容：**
- **gather**：
  - 基本用法和规则
  - dim=0 vs dim=1 的区别
  - 可视化理解
  - 提取对角线元素
- **scatter**：
  - 基本用法和规则（gather的逆操作）
  - dim=0 vs dim=1 对比
  - 3D张量上的应用
  - scatter_add - 累加模式
- **实际应用**：
  - One-hot 编码
  - 标签平滑（Label Smoothing）
  - 提取 top-k 值
- scatter 和 gather 的互逆关系
- 记忆技巧

### 7. `07_math_operations.ipynb`
**主题：** 数学运算

**内容：**
- **逐元素操作**：
  - 基本运算：abs, sqrt, exp, log, sin, cos
  - 激活函数：relu, clamp, pow
- **算术运算**：
  - 加减乘除：+, -, *, /, //, %, **
- **重要区分**：
  - mul - 逐元素乘法
  - mm - 矩阵乘法
  - matmul - 通用矩阵乘法（支持广播）
  - bmm - 批量矩阵乘法
- **聚合操作**：
  - sum, mean, max, min
  - argmax, argmin
  - keepdim 参数的作用
- **比较操作**：
  - 逐元素比较：>, <, ==
  - where - 条件选择
  - topk, sort - 排序和选择
- **线性代数**：
  - dot - 向量点积
  - mv - 矩阵-向量乘法
  - mm - 矩阵-矩阵乘法
  - bmm - 批量矩阵乘法
  - norm - 范数计算
- **实际应用**：
  - Softmax 实现
  - L2 正则化
  - 余弦相似度

### 8. `08_autograd.ipynb`
**主题：** 自动微分（Autograd）

**内容：**
- **基本概念**：
  - backward() - 反向传播
  - requires_grad - 是否需要梯度
  - grad - 梯度值
  - grad_fn - 创建张量的函数
- **计算图**：
  - 计算图可视化
  - 叶子节点 vs 非叶子节点
- **重要方法**：
  - retain_graph - 保留计算图
  - gradient 参数 - 雅可比向量积
  - torch.autograd.grad - 计算梯度并返回
- **梯度管理**：
  - 梯度累加（不会自动清零）
  - grad.zero_() - 手动清零
- **梯度控制**：
  - detach - 切断梯度传播
  - torch.no_grad() - 禁用梯度计算
  - torch.set_grad_enabled() - 动态控制
- **实际应用**：
  - 简单神经网络的前向和反向传播
  - 高阶导数计算
  - 梯度下降更新
- **常见错误和注意事项**

## 使用建议

1. **按顺序学习**：建议按照文档章节顺序运行这些 notebook
2. **交互式学习**：可以修改代码参数，观察不同的输出结果
3. **理解概念**：每个 cell 都有注释说明，帮助理解概念
4. **实践应用**：尝试将学到的知识应用到实际问题中
5. **重点关注**：
   - scatter/gather 的 dim 参数理解
   - 广播机制的规则
   - mul vs mm 的区别
   - autograd 的叶子节点概念

## 注意事项

- 确保已安装 PyTorch 和相关依赖
- MPS 相关代码仅在 Mac Apple Silicon 设备上可用
- CUDA 相关代码仅在支持 CUDA 的设备上可用
- 运行前建议先运行设备检查代码确认可用设备

## 核心要点总结

### 张量操作
- **形状变换**: view（共享内存）vs reshape（可能复制）
- **拼接切分**: cat（现有维度）vs stack（新维度）
- **索引操作**: gather/scatter 是互逆操作

### 广播机制
- 从右向左对齐维度
- 每个维度要么相同，要么为1，要么不存在
- 使用 keepdim=True 保持维度便于广播

### 数学运算
- `*` 是逐元素乘法，矩阵乘法用 `mm` 或 `matmul`
- 聚合操作的 dim 参数：指定在哪个维度上操作
- 线性代数：dot, mv, mm, bmm, matmul

### 自动微分
- 梯度不会自动清零，需要手动 `grad.zero_()`
- 只有叶子节点保留梯度
- 推理时使用 `torch.no_grad()` 节省内存
- detach 创建共享数据但不参与梯度计算的张量

## 学习路线

```
01_tensor_basics     → 了解张量的基本概念和属性
    ↓
02_tensor_creation   → 掌握各种创建张量的方法
    ↓
03_tensor_operations → 学习拼接、切分、索引操作
    ↓
04_tensor_shape      → 理解形状变换操作
    ↓
05_broadcasting      → 掌握广播机制（重要）
    ↓
06_scatter_gather    → 深入理解 scatter/gather（难点）
    ↓
07_math_operations   → 学习数学运算（重要）
    ↓
08_autograd          → 理解自动微分（核心）
```

## 参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- 第二章文档：详细的理论说明和概念讲解

## 版本信息

- 建议 PyTorch 版本: >= 2.0
- Python 版本: >= 3.8
- 其他依赖: numpy

## 反馈

如有问题或建议，欢迎提出！
