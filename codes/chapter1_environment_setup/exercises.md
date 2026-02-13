# Chapter 1: Environment Setup - Exercises

## Basic Level (基础题)

### Exercise 1.1: Device Check and Information

**题目 (Problem):**
编写代码检查你的计算机是否有可用的GPU，并打印以下信息：
- 是否可用CUDA
- GPU数量（如果有的话）
- GPU设备名称

**提示 (Hint):**
- 使用 `torch.cuda.is_available()` 检查CUDA可用性
- 使用 `torch.cuda.device_count()` 获取GPU数量
- 使用 `torch.cuda.get_device_name()` 获取设备名称

**参考答案 (Reference Solution):**
```python
import torch

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    # Get name of each GPU
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("No GPU available, using CPU")
```

---

### Exercise 1.2: PyTorch Version Query

**题目 (Problem):**
编写一个函数来获取并显示以下版本信息：
- PyTorch版本
- CUDA版本（如果支持的话）
- cuDNN版本（如果支持的话）
- Python版本

**提示 (Hint):**
- 使用 `torch.__version__` 获取PyTorch版本
- 使用 `torch.version.cuda` 获取编译时CUDA版本
- 使用 `torch.backends.cudnn.version()` 获取cuDNN版本
- 使用 `sys.version` 获取Python版本

**参考答案 (Reference Solution):**
```python
import torch
import sys

def print_version_info():
    """Print version information of PyTorch and dependencies"""
    print("=" * 50)
    print("Environment Information")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA Version (at compile time): {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("CUDA: Not available")

    print("=" * 50)

print_version_info()
```

---

### Exercise 1.3: Device Type Detection

**题目 (Problem):**
编写代码创建一个张量，并将其放在适当的设备上（优先GPU，其次CPU）。打印张量的设备位置和数据类型。

**提示 (Hint):**
- 使用条件语句选择设备
- 使用 `.to(device)` 或在创建时指定 `device` 参数
- 使用 `.device` 和 `.dtype` 属性查询张量信息

**参考答案 (Reference Solution):**
```python
import torch

# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a tensor on the selected device
tensor = torch.randn(3, 4, device=device)

# Print tensor information
print(f"Tensor device: {tensor.device}")
print(f"Tensor dtype: {tensor.dtype}")
print(f"Tensor shape: {tensor.shape}")
print(f"\nTensor:\n{tensor}")
```

---

## Advanced Level (进阶题)

### Exercise 2.1: GPU Memory Information

**题目 (Problem):**
编写函数获取并显示GPU内存使用情况：
- 总GPU内存大小
- 已用内存
- 可用内存
- 内存使用百分比

**提示 (Hint):**
- 使用 `torch.cuda.get_device_properties()` 获取GPU属性
- 使用 `torch.cuda.memory_allocated()` 获取已分配内存
- 使用 `torch.cuda.get_device_properties(device).total_memory` 获取总内存
- 考虑使用单位转换（bytes to MB/GB）

**参考答案 (Reference Solution):**
```python
import torch

def print_gpu_memory_info():
    """Print GPU memory usage information"""
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda")

    # Get device properties
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory / (1024 ** 3)  # Convert to GB

    # Get current memory usage
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)

    print("=" * 60)
    print("GPU Memory Information")
    print("=" * 60)
    print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Allocated Memory: {allocated:.2f} GB")
    print(f"Reserved Memory: {reserved:.2f} GB")
    print(f"Free Memory: {(total_memory - allocated):.2f} GB")
    print(f"Memory Usage: {(allocated / total_memory * 100):.2f}%")
    print("=" * 60)

print_gpu_memory_info()
```

---

### Exercise 2.2: Device-to-Device Tensor Migration

**题目 (Problem):**
编写一个函数，可以将张量在CPU和GPU之间迁移，并比较迁移前后的内存占用变化。

**提示 (Hint):**
- 使用 `.cpu()` 和 `.cuda()` 或 `.to(device)` 进行设备迁移
- 使用 `torch.cuda.memory_allocated()` 检查GPU内存
- 创建不同大小的张量进行测试

**参考答案 (Reference Solution):**
```python
import torch

def migrate_tensor_and_report(tensor_size=(1000, 1000)):
    """Migrate tensor between CPU and GPU, report memory changes"""

    print("=" * 60)
    print("Tensor Migration Demo")
    print("=" * 60)

    # Create tensor on CPU
    tensor_cpu = torch.randn(tensor_size, device='cpu')
    cpu_memory = tensor_cpu.element_size() * tensor_cpu.nelement() / (1024 ** 2)
    print(f"Tensor shape: {tensor_size}")
    print(f"Tensor size: {cpu_memory:.2f} MB")
    print(f"Location: CPU")

    if torch.cuda.is_available():
        # Check GPU memory before migration
        gpu_memory_before = torch.cuda.memory_allocated() / (1024 ** 2)

        # Migrate to GPU
        tensor_gpu = tensor_cpu.to('cuda')
        gpu_memory_after = torch.cuda.memory_allocated() / (1024 ** 2)

        print(f"\nAfter moving to GPU:")
        print(f"GPU memory before: {gpu_memory_before:.2f} MB")
        print(f"GPU memory after: {gpu_memory_after:.2f} MB")
        print(f"GPU memory increase: {(gpu_memory_after - gpu_memory_before):.2f} MB")
        print(f"Tensor location: {tensor_gpu.device}")

        # Migrate back to CPU
        tensor_cpu_again = tensor_gpu.to('cpu')
        print(f"\nAfter moving back to CPU:")
        print(f"Tensor location: {tensor_cpu_again.device}")
    else:
        print("GPU not available, skipping migration test")

    print("=" * 60)

migrate_tensor_and_report(tensor_size=(5000, 5000))
```

---

### Exercise 2.3: Device Compatibility Check

**题目 (Problem):**
编写一个函数，检查多个操作是否兼容（例如两个张量的设备类型是否一致）。实现以下检查：
- 两个张量是否在同一设备上
- 两个张量的数据类型是否相同
- 张量操作是否会导致设备冲突

**提示 (Hint):**
- 比较 `.device` 属性
- 比较 `.dtype` 属性
- 尝试进行张量操作并捕获异常

**参考答案 (Reference Solution):**
```python
import torch

def check_device_compatibility(tensor1, tensor2):
    """Check if two tensors are compatible for operations"""

    print("=" * 60)
    print("Device Compatibility Check")
    print("=" * 60)

    # Check device
    same_device = tensor1.device == tensor2.device
    print(f"Tensor 1 device: {tensor1.device}")
    print(f"Tensor 2 device: {tensor2.device}")
    print(f"Same device: {same_device}")

    # Check dtype
    same_dtype = tensor1.dtype == tensor2.dtype
    print(f"\nTensor 1 dtype: {tensor1.dtype}")
    print(f"Tensor 2 dtype: {tensor2.dtype}")
    print(f"Same dtype: {same_dtype}")

    # Try operation and report
    print(f"\nAttempting element-wise addition:")
    try:
        result = tensor1 + tensor2
        print(f"Success! Result device: {result.device}, dtype: {result.dtype}")
    except RuntimeError as e:
        print(f"Failed: {e}")

    print("=" * 60)

# Test cases
if torch.cuda.is_available():
    t1 = torch.randn(2, 3, device='cuda')
    t2 = torch.randn(2, 3, device='cpu')
    print("Test 1: Different devices")
    check_device_compatibility(t1, t2)

    print("\nTest 2: Different dtypes")
    t3 = torch.randn(2, 3, device='cuda', dtype=torch.float32)
    t4 = torch.randn(2, 3, device='cuda', dtype=torch.float64)
    check_device_compatibility(t3, t4)
else:
    print("GPU not available for testing")
```

---

## Challenge Level (挑战题)

### Exercise 3.1: Multi-GPU Management

**题目 (Problem):**
如果你有多个GPU，编写代码实现以下功能：
- 列出所有可用的GPU及其编号
- 计算在不同GPU上创建的张量
- 创建一个简单的数据并行模型（使用DataParallel或类似机制）
- 显示每个GPU上的内存占用

**提示 (Hint):**
- 使用 `torch.cuda.device_count()` 获取GPU数量
- 在循环中使用 `torch.cuda.set_device()` 在不同GPU间切换
- 使用 `nn.DataParallel()` 进行数据并行
- 使用 `torch.cuda.memory_allocated(device)` 检查每个GPU的内存

**参考答案 (Reference Solution):**
```python
import torch
import torch.nn as nn

def demo_multi_gpu_management():
    """Demonstrate multi-GPU management"""

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print("=" * 60)
    print("Multi-GPU Management Demo")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # List all GPUs
    print("\nAvailable GPUs:")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Total Memory: {props.total_memory / (1024**3):.2f} GB")

    if num_gpus > 1:
        print("\n--- Distributing tensors across GPUs ---")

        # Create tensors on different GPUs
        tensors = []
        for i in range(num_gpus):
            with torch.cuda.device(i):
                tensor = torch.randn(100, 100, device=f'cuda:{i}')
                tensors.append(tensor)
                allocated = torch.cuda.memory_allocated(i) / (1024**2)
                print(f"GPU {i} - Created tensor, allocated memory: {allocated:.2f} MB")

        print("\n--- Memory allocation per GPU ---")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / (1024**2)
            reserved = torch.cuda.memory_reserved(i) / (1024**2)
            print(f"GPU {i}: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")
    else:
        print("\nOnly one GPU available, skipping multi-GPU demo")

    print("=" * 60)

# Run demo
if torch.cuda.device_count() > 0:
    demo_multi_gpu_management()
```

---

### Exercise 3.2: Intelligent Device Selection with Performance Metrics

**题目 (Problem):**
编写一个智能设备选择函数，根据以下条件自动选择最佳计算设备：
1. 如果有GPU且GPU内存充足（大于阈值），选择GPU
2. 如果GPU内存不足，回退到CPU
3. 如果有多个GPU，选择内存最充足的GPU
4. 记录选择过程和性能指标

**提示 (Hint):**
- 使用内存查询API获取各GPU的可用内存
- 比较不同GPU的内存
- 可选：使用简单的计时来比较性能
- 返回选择的设备和原因

**参考答案 (Reference Solution):**
```python
import torch
import time

def intelligent_device_selection(required_memory_gb=0.5):
    """
    Intelligently select the best device based on available resources

    Args:
        required_memory_gb: Required GPU memory in GB (default: 0.5 GB)

    Returns:
        device: Selected torch.device
        info: Dictionary with selection information
    """

    print("=" * 60)
    print("Intelligent Device Selection")
    print("=" * 60)

    info = {
        'device': None,
        'reason': '',
        'available_memory_gb': 0,
    }

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        info['device'] = torch.device('cpu')
        info['reason'] = 'CUDA not available'
        print(f"Selected: {info['device']}")
        return info['device'], info

    # Find GPU with most available memory
    best_gpu = -1
    max_available_memory = 0

    print(f"\nChecking GPUs (required memory: {required_memory_gb:.2f} GB):")
    print("-" * 60)

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        available = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)

        print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Available: {available:.2f} GB")

        if available >= required_memory_gb:
            if available > max_available_memory:
                max_available_memory = available
                best_gpu = i

    print("-" * 60)

    if best_gpu >= 0:
        device = torch.device(f'cuda:{best_gpu}')
        info['device'] = device
        info['reason'] = f'GPU {best_gpu} selected (available: {max_available_memory:.2f} GB)'
        print(f"Selected: GPU {best_gpu}")
    else:
        device = torch.device('cpu')
        info['device'] = device
        info['reason'] = f'GPU memory insufficient, falling back to CPU'
        print(f"Selected: CPU (GPU memory < {required_memory_gb:.2f} GB)")

    info['available_memory_gb'] = max_available_memory

    print("=" * 60)
    return device, info

# Test the function
device, info = intelligent_device_selection(required_memory_gb=0.1)
print(f"\nSelection Info:")
print(f"  Device: {info['device']}")
print(f"  Reason: {info['reason']}")
print(f"  Available Memory: {info['available_memory_gb']:.2f} GB")

# Verify by creating a tensor on the selected device
test_tensor = torch.randn(100, 100, device=device)
print(f"\nTest tensor created on: {test_tensor.device}")
```

---

## Summary

This exercise set covers:

### Basic Level 基础题
- Device detection and GPU information
- Version information queries
- Device type selection

### Advanced Level 进阶题
- GPU memory analysis
- Device migration and memory tracking
- Device compatibility checking

### Challenge Level 挑战题
- Multi-GPU management and memory monitoring
- Intelligent device selection with performance metrics

Each exercise includes:
- Problem statement (English and Chinese)
- Helpful hints
- Complete reference solution
- Practical applications

## Tips for Success

1. **Run incrementally**: Test code line by line to understand each step
2. **Modify parameters**: Try different tensor sizes and GPU memory thresholds
3. **Combine exercises**: Use concepts from basic exercises in advanced ones
4. **Performance testing**: Add timing measurements to understand computational costs
5. **Error handling**: Add try-except blocks to handle edge cases gracefully

## Next Steps

After completing these exercises:
- Explore tensor operations on different devices
- Benchmark CPU vs GPU performance
- Investigate mixed-precision computing (float16 vs float32)
- Study data parallel and distributed training setups
