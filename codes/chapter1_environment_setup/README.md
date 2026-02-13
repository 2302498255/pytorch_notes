# 第一章：环境配置 - 实践代码

本目录包含第一章"环境配置"的所有实践代码，每个 notebook 对应文档中的一个章节。

## 📚 文件说明

### 1. `01_installation_guide.ipynb`
**对应章节：** 1.1 安装 PyTorch

**内容：**
- PyTorch 安装前的准备工作
- 不同平台的安装命令（Conda/Pip）
- CPU 版本 vs GPU 版本选择
- CUDA 版本选择指南
- 安装验证代码

**学习目标：**
- 了解如何选择合适的 PyTorch 版本
- 掌握不同平台的安装方法
- 学会验证安装是否成功

### 2. `02_device_check.ipynb`
**对应章节：** 1.2 设备检查

**内容：**
- CPU 设备检查
- CUDA（NVIDIA GPU）设备检查
- MPS（Apple Silicon GPU）设备检查
- 设备性能对比
- 多 GPU 检查和管理
- 设备选择最佳实践

**学习目标：**
- 掌握如何检查可用的计算设备
- 了解不同设备的特点和性能
- 学会在代码中正确选择和使用设备

### 3. `03_first_tensor.ipynb`
**对应章节：** 综合实践

**内容：**
- 创建第一个 PyTorch 张量
- 在不同设备上运行张量操作
- 简单的张量运算
- 设备间数据迁移
- 性能测试对比

**学习目标：**
- 快速上手 PyTorch 基本操作
- 理解张量和设备的关系
- 建立对 PyTorch 的初步认识

## 🎯 使用建议

1. **按顺序学习**：建议按照文件编号顺序运行这些 notebook
2. **动手实践**：每个 cell 都可以运行，建议修改参数观察结果
3. **理解概念**：每个 notebook 都有详细注释和说明
4. **解决问题**：遇到安装问题可以参考常见问题部分

## ⚠️ 注意事项

### 系统要求
- **Python 版本**：推荐 3.8-3.11
- **操作系统**：Windows/Linux/macOS
- **GPU（可选）**：NVIDIA GPU（CUDA）或 Apple Silicon（MPS）

### 常见问题
1. **CUDA 版本不匹配**：使用 `nvidia-smi` 检查驱动支持的 CUDA 版本
2. **MPS 不可用**：确保是 Apple Silicon Mac 且系统版本 >= 12.3
3. **导入错误**：卸载后重新安装，确保版本匹配

### 学习路径
```
01_installation_guide.ipynb (安装)
    ↓
02_device_check.ipynb (设备检查)
    ↓
03_first_tensor.ipynb (第一个张量)
    ↓
进入第二章：核心模块
```

## 📖 参考资源

- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- [CUDA 下载页面](https://developer.nvidia.com/cuda-downloads)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)

## 💡 学习提示

- 第一章是基础，确保环境配置正确后再进入后续章节
- 如果没有 GPU，使用 CPU 也完全可以学习
- 记住常用的设备检查代码，后续章节会频繁使用
- 建议将设备检查代码保存为代码片段，方便复用

---

**准备好了吗？让我们开始 PyTorch 的学习之旅！** 🚀
