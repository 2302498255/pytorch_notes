# PyTorch 学习笔记 📚

[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一份全面、系统、新手友好的 PyTorch 学习资料，包含详细的学习笔记和配套的实践代码。

## 📖 项目简介

本项目是一份完整的 PyTorch 学习教程，适合零基础入门和进阶学习。内容涵盖从环境配置到核心模块，再到数据处理的完整知识体系。

### ✨ 特色

- **📝 详细笔记**：5,600+ 行的完整学习笔记，涵盖所有核心概念
- **💻 实践代码**：17 个 Jupyter Notebook，包含可运行的示例代码
- **🎯 练习题**：32 道精心设计的练习题（基础→进阶→挑战）
- **🔄 持续更新**：基于 PyTorch 2.10.0，紧跟最新版本
- **🌟 新手友好**：详细的中文注释，循序渐进的学习路径

## 📂 项目结构

```
pytorch_notes/
├── PyTorch 学习笔记.md          # 完整的学习笔记（5,660 行）
├── codes/                        # 实践代码
│   ├── chapter1_environment_setup/    # 第一章：环境配置
│   │   ├── 01_installation_guide.ipynb
│   │   ├── 02_device_check.ipynb
│   │   ├── 03_first_tensor.ipynb
│   │   ├── exercises.md          # 练习题（8 题）
│   │   └── README.md
│   ├── chapter2_core_module/          # 第二章：核心模块
│   │   ├── 01_tensor_basics.ipynb
│   │   ├── 02_tensor_creation.ipynb
│   │   ├── 03_tensor_operations.ipynb
│   │   ├── 04_tensor_shape.ipynb
│   │   ├── 05_broadcasting.ipynb
│   │   ├── 06_scatter_gather.ipynb
│   │   ├── 07_math_operations.ipynb
│   │   ├── 08_autograd.ipynb
│   │   ├── exercises.md          # 练习题（13 题）
│   │   └── README.md
│   └── chapter3_data_module/          # 第三章：数据模块
│       ├── 01_dataset_basics.ipynb
│       ├── 02_custom_dataset.ipynb
│       ├── 03_dataloader.ipynb
│       ├── 04_sampler.ipynb
│       ├── 05_transforms.ipynb
│       ├── 06_complete_example.ipynb
│       ├── exercises.md          # 练习题（11 题）
│       └── README.md
├── chapter2_core_modular/        # 早期测试代码（参考）
└── test01_setup/                 # 环境测试代码
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- PyTorch 2.10.0+
- Jupyter Notebook 或 JupyterLab

### 2. 安装 PyTorch

**使用 Conda（推荐）：**
```bash
# CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU 版本（CUDA 12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**使用 Pip：**
```bash
# CPU 版本
pip3 install torch torchvision torchaudio

# GPU 版本（CUDA 12.1）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 克隆仓库

```bash
git clone https://github.com/2302498255/pytorch_notes.git
cd pytorch_notes
```

### 4. 启动 Jupyter Notebook

```bash
cd codes
jupyter notebook
```

## 📚 学习路径

### 第一章：环境配置（1-2 天）
- [x] PyTorch 安装和验证
- [x] 设备检查（CPU/CUDA/MPS）
- [x] 创建第一个张量
- [x] 完成 8 道练习题

### 第二章：核心模块（3-5 天）
- [x] 张量基础和创建
- [x] 张量操作（拼接、切分、索引）
- [x] 形状变换（view/reshape/squeeze/unsqueeze）
- [x] 广播机制
- [x] scatter/gather 详解
- [x] 数学运算
- [x] 自动微分（Autograd）
- [x] 完成 13 道练习题

### 第三章：数据模块（2-3 天）
- [x] Dataset 基础
- [x] 自定义 Dataset
- [x] DataLoader 使用
- [x] Sampler 采样器
- [x] Transforms 数据增强
- [x] 完整示例
- [x] 完成 11 道练习题

## 🎯 核心内容

### 张量操作
- 创建、索引、切片
- 形状变换（view、reshape、transpose、permute）
- 广播机制
- 数学运算（逐元素、聚合、线性代数）

### 自动微分
- 计算图原理
- 梯度计算和管理
- detach 和 no_grad
- 自定义梯度函数

### 数据处理
- Dataset 和 DataLoader
- 数据预处理和增强
- 处理不平衡数据
- 自定义 collate_fn

### Transforms（重点补充）
- 基础预处理：Resize、ToTensor、Normalize
- 数据增强：RandomFlip、RandomRotation、ColorJitter
- 高级技术：FiveCrop、AutoAugment、RandAugment
- 自定义 transforms

## 📝 练习题

每章都配有精心设计的练习题，分为三个难度等级：

- **基础题**：巩固核心概念
- **进阶题**：深入掌握技巧
- **挑战题**：锻炼综合应用能力

**总计 32 道练习题**，每题都包含：
- 题目描述
- 解题提示
- 完整的参考答案
- 使用示例

## 🛠️ 技术栈

- **深度学习框架**：PyTorch 2.10.0
- **计算机视觉**：torchvision
- **数值计算**：NumPy
- **图像处理**：PIL
- **可视化**：Matplotlib

## 📖 参考资料

本项目参考了以下优秀资源：

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch Tutorial 2nd Edition](https://tingsongyu.github.io/PyTorch-Tutorial-2nd/)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

如果您发现任何问题或有改进建议，请：
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：

- GitHub Issues: [提交 Issue](https://github.com/2302498255/pytorch_notes/issues)

## ⭐ Star History

如果这个项目对您有帮助，请给一个 ⭐️ Star！

---

**最后更新**: 2026-02-13
**PyTorch 版本**: 2.10.0
**文档版本**: v1.0
