# 第三章 数据模块 - 练习题

## 基础题 (Fundamental)

### 题目 1: 创建简单 Dataset 类

**题目描述：**
创建一个简单的 `SimpleDataset` 类，继承自 `torch.utils.data.Dataset`，实现一个包含 100 个样本的数据集。每个样本是一个形状为 (28, 28) 的随机图像和一个 0-9 的标签。

**提示：**
- 需要实现 `__len__()` 和 `__getitem__()` 方法
- 使用 `torch.randn()` 生成随机图像
- 使用 `torch.randint()` 生成随机标签

**参考答案：**

```python
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 28, 28)
        self.labels = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# 使用示例
dataset = SimpleDataset(100)
print(f"数据集大小: {len(dataset)}")
image, label = dataset[0]
print(f"样本形状: {image.shape}, 标签: {label}")
```

---

### 题目 2: 使用 DataLoader 加载数据

**题目描述：**
使用上一个练习的 `SimpleDataset`，创建一个 `DataLoader`，设置批次大小为 32，启用 shuffle 和多进程加载。遍历一个完整的 epoch，打印每个批次的数据形状和标签形状。

**提示：**
- 使用 `torch.utils.data.DataLoader`
- 设置 `batch_size=32`、`shuffle=True`、`num_workers=2`
- 使用 for 循环遍历 DataLoader

**参考答案：**

```python
from torch.utils.data import DataLoader

# 创建数据集和DataLoader
dataset = SimpleDataset(100)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# 遍历一个 epoch
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"批次 {batch_idx}:")
    print(f"  图像形状: {images.shape}")  # torch.Size([32, 28, 28])
    print(f"  标签形状: {labels.shape}")  # torch.Size([32])

    if batch_idx == 2:  # 只显示前3个批次
        break
```

**输出示例：**
```
批次 0:
  图像形状: torch.Size([32, 28, 28])
  标签形状: torch.Size([32])
批次 1:
  图像形状: torch.Size([32, 28, 28])
  标签形状: torch.Size([32])
批次 2:
  图像形状: torch.Size([4, 28, 28])
  标签形状: torch.Size([4])
```

---

### 题目 3: 数据集划分（Train/Valid/Test）

**题目描述：**
给定一个包含 1000 个样本的数据集，将其按照 7:2:1 的比例划分为训练集、验证集和测试集。为三个子集分别创建 DataLoader（训练集 shuffle=True，其他 shuffle=False）。

**提示：**
- 使用 `torch.utils.data.random_split()`
- 需要计算每个子集的样本数量
- 验证集和测试集应该不 shuffle

**参考答案：**

```python
from torch.utils.data import random_split, DataLoader

# 创建原始数据集
dataset = SimpleDataset(1000)

# 计算划分大小
train_size = int(0.7 * len(dataset))  # 700
valid_size = int(0.2 * len(dataset))  # 200
test_size = len(dataset) - train_size - valid_size  # 100

# 划分数据集
train_dataset, valid_dataset, test_dataset = random_split(
    dataset,
    [train_size, valid_size, test_size]
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 验证
print(f"训练集大小: {len(train_dataset)}")  # 700
print(f"验证集大小: {len(valid_dataset)}")  # 200
print(f"测试集大小: {len(test_dataset)}")   # 100

# 检查数据加载
for images, labels in train_loader:
    print(f"批次形状: {images.shape}, {labels.shape}")
    break
```

---

### 题目 4: 使用内置数据集（MNIST）

**题目描述：**
使用 PyTorch 的 MNIST 数据集，下载训练集和测试集，创建相应的 DataLoader。打印数据集信息（数据集大小、单个样本形状）。

**提示：**
- 使用 `torchvision.datasets.MNIST`
- 需要指定 `download=True` 来自动下载
- 使用 `torchvision.transforms.ToTensor()` 转换图像

**参考答案：**

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义转换
transform = transforms.ToTensor()

# 下载并加载 MNIST 数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 打印信息
print(f"训练集大小: {len(train_dataset)}")  # 60000
print(f"测试集大小: {len(test_dataset)}")   # 10000

# 获取单个样本
image, label = train_dataset[0]
print(f"单个样本形状: {image.shape}")       # torch.Size([1, 28, 28])
print(f"标签: {label}")

# 获取批次
for images, labels in train_loader:
    print(f"批次形状: {images.shape}, {labels.shape}")
    break
```

---

## 进阶题 (Intermediate)

### 题目 5: 自定义 Dataset 与数据增强

**题目描述：**
创建一个自定义的 `AugmentedImageDataset` 类，加载任意图像文件夹中的图像。实现两套转换管道：基础转换（ToTensor + Normalize）和增强转换（随机旋转、随机裁剪、随机水平翻转、亮度调整等）。支持在初始化时选择使用哪套管道。

**提示：**
- 使用 `os.listdir()` 遍历文件夹
- 使用 `PIL.Image.open()` 加载图像
- 使用 `torchvision.transforms.Compose()` 组合多个转换
- 包含 `RandomRotation`、`RandomCrop`、`RandomHorizontalFlip`、`ColorJitter` 等

**参考答案：**

```python
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AugmentedImageDataset(Dataset):
    def __init__(self, img_dir, augment=False, img_size=224):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir)
                          if f.endswith(('.jpg', '.png', '.jpeg'))]

        # 基础转换
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]

        # 增强转换
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomCrop(img_size, padding=10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
            ] + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# 使用示例
# dataset_basic = AugmentedImageDataset('./images', augment=False)
# dataset_augmented = AugmentedImageDataset('./images', augment=True)
```

---

### 题目 6: 自定义 Sampler（类平衡采样）

**题目描述：**
创建一个自定义的 `BalancedSampler` 类，继承自 `torch.utils.data.Sampler`。实现按类别平衡的采样方法，确保每个批次中各类别的样本数量大致相等。假设数据集有多个类别，需要能够处理不平衡的数据分布。

**提示：**
- 继承 `torch.utils.data.Sampler`
- 需要实现 `__iter__()` 和 `__len__()` 方法
- 统计每个类别的样本索引
- 按比例采样不同类别的样本

**参考答案：**

```python
import torch
from torch.utils.data import Sampler
from collections import defaultdict
import random

class BalancedSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        """
        Args:
            dataset: 数据集，需要有一个 get_labels() 方法返回所有标签
            shuffle: 是否打乱
        """
        self.dataset = dataset
        self.shuffle = shuffle

        # 获取所有标签
        self.labels = dataset.labels if hasattr(dataset, 'labels') else \
                      [dataset[i][1] for i in range(len(dataset))]

        # 按类别分组索引
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)

        # 计算最大类别的样本数
        self.max_count = max(len(indices) for indices in self.class_indices.values())

    def __iter__(self):
        indices = []

        # 对每个类别进行采样（使用 replace 实现过采样）
        for label in self.class_indices:
            class_indices = self.class_indices[label]
            # 使用替换采样达到最大计数
            sampled_indices = torch.multinomial(
                torch.ones(len(class_indices)),
                self.max_count,
                replacement=True
            ).tolist()
            indices.extend([class_indices[i] for i in sampled_indices])

        if self.shuffle:
            random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return len(self.class_indices) * self.max_count

# 使用示例
# sampler = BalancedSampler(dataset)
# dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

### 题目 7: 使用 Collate Function 处理可变长度序列

**题目描述：**
创建一个处理可变长度文本序列的数据集和自定义 collate_fn。数据集包含不同长度的整数序列（表示单词索引）和对应的标签。自定义 collate_fn 需要将可变长度的序列填充到同一长度，并返回填充掩码。

**提示：**
- 创建包含可变长度序列的数据集
- 使用 `torch.nn.utils.rnn.pad_sequence()` 进行填充
- collate_fn 应返回序列、掩码和标签的元组
- 掩码应指示哪些位置是真实数据，哪些是填充

**参考答案：**

```python
import torch
from torch.utils.data import Dataset, DataLoader
import random

class VariableLengthDataset(Dataset):
    def __init__(self, num_samples=100, vocab_size=1000):
        self.samples = []
        for _ in range(num_samples):
            # 随机生成可变长度序列
            length = random.randint(5, 50)
            sequence = torch.randint(0, vocab_size, (length,))
            label = torch.tensor(random.randint(0, 4))  # 0-4 5个类别
            self.samples.append((sequence, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_variable_length(batch):
    """
    自定义 collate_fn 处理可变长度序列

    Args:
        batch: 一个批次的样本列表，每个样本是 (sequence, label) 元组

    Returns:
        padded_sequences: 填充后的序列 (batch_size, max_length)
        masks: 填充掩码 (batch_size, max_length)
        labels: 标签 (batch_size,)
    """
    sequences, labels = zip(*batch)

    # 获取最大长度
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_length = lengths.max().item()
    batch_size = len(sequences)

    # 创建填充后的序列和掩码
    padded_sequences = torch.zeros(batch_size, max_length, dtype=torch.long)
    masks = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, sequence in enumerate(sequences):
        seq_len = len(sequence)
        padded_sequences[i, :seq_len] = sequence
        masks[i, :seq_len] = True  # True 表示真实数据

    labels = torch.stack(labels)

    return padded_sequences, masks, labels

# 使用示例
dataset = VariableLengthDataset(100)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=collate_variable_length
)

for sequences, masks, labels in dataloader:
    print(f"序列形状: {sequences.shape}")      # (16, max_length)
    print(f"掩码形状: {masks.shape}")          # (16, max_length)
    print(f"标签形状: {labels.shape}")         # (16,)
    print(f"掩码示例: {masks[0]}")
    break
```

---

### 题目 8: 多数据源加载与关联

**题目描述：**
创建一个多模态数据集类 `MultimodalDataset`，同时加载图像和文本数据。每个样本包含一张图像、对应的文本描述和一个类别标签。实现一个方法可以动态改变文本编码方式（如：字符级、词级或使用预训练的分词器）。

**提示：**
- 需要同时处理两种不同类型的数据
- 创建配对的图像和文本文件
- 支持不同的文本编码方式
- 返回图像张量、编码文本和标签

**参考答案：**

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class MultimodalDataset(Dataset):
    def __init__(self, img_dir, text_dir, img_transform=None,
                 text_encoding='word', vocab_size=1000):
        """
        Args:
            img_dir: 图像文件夹路径
            text_dir: 文本文件夹路径
            img_transform: 图像转换管道
            text_encoding: 文本编码方式 ('char', 'word', 'pretrained')
            vocab_size: 词汇表大小
        """
        self.img_dir = img_dir
        self.text_dir = text_dir
        self.img_transform = img_transform
        self.text_encoding = text_encoding
        self.vocab_size = vocab_size

        # 获取所有文件对
        self.filenames = [f.replace('.jpg', '')
                          for f in os.listdir(img_dir)
                          if f.endswith('.jpg')]

        # 创建简单的词汇表
        self.word2idx = {str(i): i for i in range(vocab_size)}
        self.word2idx['<UNK>'] = vocab_size - 1

    def __len__(self):
        return len(self.filenames)

    def _encode_text(self, text):
        """将文本编码为数字序列"""
        if self.text_encoding == 'char':
            # 字符级编码
            encoded = [ord(c) % self.vocab_size for c in text[:100]]
        elif self.text_encoding == 'word':
            # 词级编码
            words = text.lower().split()
            encoded = [int(self.word2idx.get(w, self.vocab_size - 1))
                      for w in words[:50]]
        else:  # pretrained
            # 预训练模型编码（这里简化处理）
            encoded = [hash(text) % self.vocab_size]

        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # 加载图像
        img_path = os.path.join(self.img_dir, filename + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)

        # 加载文本
        text_path = os.path.join(self.text_dir, filename + '.txt')
        with open(text_path, 'r') as f:
            text = f.read()

        # 编码文本
        text_encoded = self._encode_text(text)

        # 提取标签（从文件名中）
        label = torch.tensor(int(filename.split('_')[0]), dtype=torch.long)

        return image, text_encoded, label

    def set_text_encoding(self, encoding):
        """动态改变文本编码方式"""
        if encoding in ['char', 'word', 'pretrained']:
            self.text_encoding = encoding
        else:
            raise ValueError(f"不支持的编码方式: {encoding}")

# 使用示例
# dataset = MultimodalDataset('./images', './texts')
# dataset.set_text_encoding('word')  # 改变为词级编码
#
# for image, text, label in dataset:
#     print(f"图像: {image.shape}, 文本: {text.shape}, 标签: {label}")
```

---

## 挑战题 (Challenge)

### 题目 9: 处理类不平衡数据的完整解决方案

**题目描述：**
实现一个完整的解决方案处理高度不平衡的分类数据集。该方案应包括：
1. 自定义的 WeightedRandomSampler 实现类权重采样
2. 过采样少数类和欠采样多数类的混合策略
3. 在 DataLoader 中集成该采样器
4. 验证采样后的批次中各类别的分布

**提示：**
- 使用 `torch.utils.data.WeightedRandomSampler`
- 计算每个类别的权重（反比例）
- 实现过采样和欠采样策略
- 验证类别分布的改变

**参考答案：**

```python
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np

class ImbalancedDataset(Dataset):
    """
    模拟不平衡数据集
    类别分布: [0: 1000, 1: 100, 2: 10]
    """
    def __init__(self, num_samples=1110):
        # 创建不平衡标签分布
        labels = []
        labels.extend([0] * 1000)   # 类别0: 1000个样本
        labels.extend([1] * 100)    # 类别1: 100个样本
        labels.extend([2] * 10)     # 类别2: 10个样本

        self.data = torch.randn(num_samples, 10)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ImbalancedDataLoader:
    """处理不平衡数据的完整解决方案"""

    def __init__(self, dataset, batch_size=32, strategy='weighted'):
        """
        Args:
            dataset: 数据集实例
            batch_size: 批次大小
            strategy: 采样策略 ('weighted', 'oversample', 'undersample', 'combined')
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.dataloader = self._create_dataloader()

    def _create_dataloader(self):
        if self.strategy == 'weighted':
            sampler = self._create_weighted_sampler()
        elif self.strategy == 'oversample':
            sampler = self._create_oversample_sampler()
        elif self.strategy == 'undersample':
            sampler = self._create_undersample_sampler()
        elif self.strategy == 'combined':
            sampler = self._create_combined_sampler()
        else:
            raise ValueError(f"未知策略: {self.strategy}")

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler
        )

    def _create_weighted_sampler(self):
        """
        使用加权采样器：根据类别频率反向加权
        """
        labels = self.dataset.labels
        class_counts = torch.bincount(labels)

        # 计算权重：样本数越少，权重越大
        weights = 1.0 / class_counts.float()
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.dataset),
            replacement=True
        )
        return sampler

    def _create_oversample_sampler(self):
        """
        过采样少数类：将所有类的样本数提升到最多类的数量
        """
        labels = self.dataset.labels
        class_counts = torch.bincount(labels)
        max_count = class_counts.max().item()

        # 对每个类计算其采样权重
        weights = max_count / class_counts.float()
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.dataset) * max_count // class_counts.sum().item(),
            replacement=True
        )
        return sampler

    def _create_undersample_sampler(self):
        """
        欠采样多数类：将所有类的样本数降低到最少类的数量
        """
        labels = self.dataset.labels
        class_counts = torch.bincount(labels)
        min_count = class_counts.min().item()

        # 对每个类计算其采样权重
        weights = min_count / class_counts.float()
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.dataset) * min_count // class_counts.sum().item(),
            replacement=False
        )
        return sampler

    def _create_combined_sampler(self):
        """
        混合策略：过采样少数类50%，欠采样多数类50%
        """
        labels = self.dataset.labels
        class_counts = torch.bincount(labels)

        # 目标分布：介于最少和最多之间
        min_count = class_counts.min().item()
        max_count = class_counts.max().item()
        target_count = int((min_count + max_count) / 2)

        weights = target_count / class_counts.float()
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.dataset) * target_count // class_counts.sum().item(),
            replacement=True
        )
        return sampler

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def analyze_distribution(dataloader, name=""):
        """分析批次中的类别分布"""
        all_labels = []
        for _, labels in dataloader:
            all_labels.extend(labels.numpy())

        counter = Counter(all_labels)
        total = len(all_labels)

        print(f"\n{name} 类别分布:")
        for class_id in sorted(counter.keys()):
            count = counter[class_id]
            percentage = 100 * count / total
            print(f"  类别 {class_id}: {count} 个样本 ({percentage:.2f}%)")


# 使用示例
if __name__ == "__main__":
    # 创建不平衡数据集
    dataset = ImbalancedDataset(1110)

    print("原始数据集类别分布:")
    original_counter = Counter(dataset.labels.numpy())
    for class_id in sorted(original_counter.keys()):
        print(f"  类别 {class_id}: {original_counter[class_id]} 个样本")

    # 尝试不同的策略
    strategies = ['weighted', 'oversample', 'undersample', 'combined']

    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"策略: {strategy.upper()}")
        print('='*50)

        loader = ImbalancedDataLoader(dataset, batch_size=32, strategy=strategy)
        dataloader = loader.get_dataloader()

        # 分析前5个批次
        for i in range(5):
            for images, labels in dataloader:
                counter = Counter(labels.numpy())
                print(f"批次 {i}: {dict(counter)}")
                break
```

---

### 题目 10: 自定义高级 Collate Function

**题目描述：**
实现一个高级的 collate_fn，用于处理复杂的混合数据类型场景。该函数需要：
1. 处理可变长度的图像序列（动态视频帧）
2. 处理对应的文本注解（可变长度）
3. 处理元数据（JSON 格式的额外信息）
4. 生成适当的索引和掩码张量
5. 支持在 GPU 上的自动转移

**提示：**
- 定义包含混合数据类型的数据样本结构
- 在 collate_fn 中分别处理不同的数据类型
- 使用 .to(device) 进行设备转移
- 返回结构化的字典包含所有处理后的数据

**参考答案：**

```python
import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Tuple
import random

class ComplexMultimodalSample:
    """表示复杂的多模态样本"""
    def __init__(self, frames, text, metadata, label):
        self.frames = frames  # 图像序列
        self.text = text      # 文本列表
        self.metadata = metadata  # 字典
        self.label = label

class ComplexMultimodalDataset(Dataset):
    def __init__(self, num_samples=100):
        self.samples = []
        for i in range(num_samples):
            # 可变数量的视频帧
            num_frames = random.randint(5, 20)
            frames = [torch.randn(3, 224, 224) for _ in range(num_frames)]

            # 可变长度的文本注解
            num_texts = random.randint(1, 5)
            texts = [f"caption_{j}_{random.randint(0, 100)}"
                    for j in range(num_texts)]

            # 元数据
            metadata = {
                'video_id': f'video_{i}',
                'duration': random.randint(10, 60),
                'fps': random.choice([24, 30, 60]),
                'resolution': random.choice(['720p', '1080p', '4K'])
            }

            label = torch.tensor(random.randint(0, 9), dtype=torch.long)

            sample = ComplexMultimodalSample(frames, texts, metadata, label)
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def advanced_collate_fn(batch: List[ComplexMultimodalSample], device='cpu'):
    """
    高级 collate 函数处理复杂的多模态数据

    Args:
        batch: 一个批次的 ComplexMultimodalSample 列表
        device: 目标设备 ('cpu' 或 'cuda:0' 等)

    Returns:
        一个包含处理后数据的字典
    """
    batch_size = len(batch)

    # 1. 处理视频帧序列
    # 找最大帧数
    max_frames = max(len(sample.frames) for sample in batch)

    # 创建填充后的帧张量
    padded_frames = torch.zeros(batch_size, max_frames, 3, 224, 224)
    frame_masks = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    frame_lengths = []

    for i, sample in enumerate(batch):
        num_frames = len(sample.frames)
        frame_lengths.append(num_frames)
        # 栈叠帧
        frames_stack = torch.stack(sample.frames)
        padded_frames[i, :num_frames] = frames_stack
        frame_masks[i, :num_frames] = True

    # 2. 处理文本注解
    # 找最大文本数和最大文本长度
    max_texts = max(len(sample.text) for sample in batch)
    max_text_length = max(
        max(len(text) for text in sample.text)
        for sample in batch
    )

    # 将文本转换为索引张量（使用 ASCII 值或简化处理）
    text_indices = torch.zeros(batch_size, max_texts, max_text_length, dtype=torch.long)
    text_masks = torch.zeros(batch_size, max_texts, dtype=torch.bool)

    for i, sample in enumerate(batch):
        for j, text in enumerate(sample.text):
            text_masks[i, j] = True
            for k, char in enumerate(text[:max_text_length]):
                text_indices[i, j, k] = ord(char)

    # 3. 处理元数据
    # 提取公共元数据字段
    video_ids = [sample.metadata['video_id'] for sample in batch]
    durations = torch.tensor(
        [sample.metadata['duration'] for sample in batch],
        dtype=torch.float32
    )
    fpses = torch.tensor(
        [sample.metadata['fps'] for sample in batch],
        dtype=torch.long
    )

    # 分辨率编码
    resolution_map = {'720p': 0, '1080p': 1, '4K': 2}
    resolutions = torch.tensor(
        [resolution_map[sample.metadata['resolution']] for sample in batch],
        dtype=torch.long
    )

    # 4. 收集标签
    labels = torch.stack([sample.label for sample in batch])

    # 5. 创建输出字典
    output = {
        # 视频帧
        'frames': padded_frames.to(device),
        'frame_lengths': torch.tensor(frame_lengths, dtype=torch.long).to(device),
        'frame_masks': frame_masks.to(device),

        # 文本
        'text_indices': text_indices.to(device),
        'text_masks': text_masks.to(device),

        # 元数据
        'video_ids': video_ids,
        'durations': durations.to(device),
        'fpses': fpses.to(device),
        'resolutions': resolutions.to(device),

        # 标签
        'labels': labels.to(device),

        # 批次信息
        'batch_size': batch_size,
        'max_frames': max_frames,
        'max_texts': max_texts,
    }

    return output


# 使用示例
if __name__ == "__main__":
    # 创建数据集
    dataset = ComplexMultimodalDataset(32)

    # 创建 DataLoader，使用自定义 collate_fn
    device = 'cpu'  # 如果有 GPU 可改为 'cuda:0'
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=lambda batch: advanced_collate_fn(batch, device=device)
    )

    # 遍历一个批次
    for batch_data in dataloader:
        print("\n批次数据结构:")
        print(f"- frames 形状: {batch_data['frames'].shape}")
        print(f"  (batch_size, max_frames, channels, height, width)")

        print(f"- frame_lengths 形状: {batch_data['frame_lengths'].shape}")
        print(f"  (batch_size,)")

        print(f"- frame_masks 形状: {batch_data['frame_masks'].shape}")
        print(f"  (batch_size, max_frames)")

        print(f"- text_indices 形状: {batch_data['text_indices'].shape}")
        print(f"  (batch_size, max_texts, max_text_length)")

        print(f"- text_masks 形状: {batch_data['text_masks'].shape}")
        print(f"  (batch_size, max_texts)")

        print(f"- durations 形状: {batch_data['durations'].shape}")
        print(f"- fpses 形状: {batch_data['fpses'].shape}")
        print(f"- resolutions 形状: {batch_data['resolutions'].shape}")

        print(f"- labels 形状: {batch_data['labels'].shape}")

        print(f"\n视频 IDs: {batch_data['video_ids']}")
        print(f"批次大小: {batch_data['batch_size']}")

        break
```

---

### 题目 11: 分布式数据加载与同步

**题目描述：**
实现一个支持分布式训练的数据加载系统。该系统应该：
1. 支持多进程/多 GPU 环境下的数据分片
2. 使用 `DistributedSampler` 确保不同进程加载不同的数据
3. 实现数据加载的同步机制
4. 支持动态调整数据加载策略（如改变 epoch 号以改变数据顺序）
5. 提供性能监控和日志记录

**提示：**
- 使用 `torch.utils.data.distributed.DistributedSampler`
- 需要处理多进程和多 GPU 的场景
- 实现 epoch 设置机制以改变随机顺序
- 添加性能指标统计

**参考答案：**

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import logging
from typing import Optional
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistributedDataset(Dataset):
    """用于分布式训练的简单数据集"""
    def __init__(self, num_samples=10000):
        self.data = torch.randn(num_samples, 100)
        self.labels = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DistributedDataLoaderWrapper:
    """
    分布式数据加载器包装类

    特性：
    - 自动处理数据分片
    - 支持多个 epoch
    - 性能监控
    - 日志记录
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False
    ):
        """
        Args:
            dataset: 数据集实例
            batch_size: 批次大小
            num_workers: 数据加载进程数
            rank: 当前进程的排名
            world_size: 总进程数
            shuffle: 是否打乱数据
            seed: 随机种子
            drop_last: 是否丢弃最后一个不完整的批次
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.epoch = 0
        self.dataloader = None
        self.metrics = {
            'epoch': 0,
            'batches_processed': 0,
            'total_time': 0,
            'avg_batch_time': 0
        }

        logger.info(
            f"初始化分布式数据加载器 - "
            f"Rank: {rank}/{world_size}, "
            f"数据集大小: {len(dataset)}"
        )

        self._create_dataloader()

    def _create_dataloader(self):
        """创建数据加载器和采样器"""
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed + self.epoch,
            drop_last=self.drop_last
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

        logger.debug(f"数据加载器创建成功 (Rank {self.rank})")

    def set_epoch(self, epoch: int):
        """
        设置当前 epoch

        这是必要的，因为采样器需要知道 epoch 号来改变随机顺序
        """
        self.epoch = epoch
        self.metrics['epoch'] = epoch

        if hasattr(self.dataloader.sampler, 'set_epoch'):
            self.dataloader.sampler.set_epoch(epoch)
            logger.info(f"设置 epoch = {epoch} (Rank {self.rank})")

    def get_dataloader(self):
        """获取当前的 DataLoader"""
        return self.dataloader

    def __iter__(self):
        """迭代数据加载器"""
        start_time = time.time()
        batch_count = 0
        batch_times = []

        for batch_idx, (data, labels) in enumerate(self.dataloader):
            batch_start = time.time()

            yield data, labels

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            batch_count += 1

            if batch_idx % 100 == 0 and self.rank == 0:
                logger.debug(
                    f"处理批次 {batch_idx}/{len(self.dataloader)} - "
                    f"批次时间: {batch_time:.4f}s"
                )

        total_time = time.time() - start_time

        # 更新指标
        self.metrics['batches_processed'] += batch_count
        self.metrics['total_time'] += total_time
        if batch_count > 0:
            self.metrics['avg_batch_time'] = (
                self.metrics['total_time'] / self.metrics['batches_processed']
            )

        if self.rank == 0:
            logger.info(
                f"Epoch {self.epoch} 完成 - "
                f"批次数: {batch_count}, "
                f"总耗时: {total_time:.2f}s, "
                f"平均批次时间: {self.metrics['avg_batch_time']:.4f}s"
            )

    def __len__(self):
        """返回 epoch 中的批次数"""
        return len(self.dataloader)

    def get_metrics(self) -> dict:
        """获取性能指标"""
        return self.metrics.copy()

    def log_metrics(self):
        """记录性能指标"""
        if self.rank == 0:
            logger.info(f"性能指标: {self.metrics}")


# 分布式训练模拟函数
def simulate_distributed_training(
    num_epochs: int = 3,
    rank: int = 0,
    world_size: int = 1
):
    """
    模拟分布式训练

    Args:
        num_epochs: 训练 epoch 数
        rank: 当前进程的排名
        world_size: 总进程数
    """
    # 创建数据集和加载器
    dataset = DistributedDataset(num_samples=10000)

    dataloader_wrapper = DistributedDataLoaderWrapper(
        dataset=dataset,
        batch_size=32,
        num_workers=2,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=True
    )

    # 模拟训练循环
    for epoch in range(num_epochs):
        dataloader_wrapper.set_epoch(epoch)

        batch_count = 0
        for data, labels in dataloader_wrapper:
            # 模拟训练步骤
            batch_count += 1

            if batch_count % 50 == 0 and rank == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_count}: "
                    f"数据形状 {data.shape}, 标签形状 {labels.shape}"
                )

        if rank == 0:
            logger.info(f"Epoch {epoch} 完成")

    # 输出最终指标
    if rank == 0:
        dataloader_wrapper.log_metrics()


# 使用示例
if __name__ == "__main__":
    # 单进程模拟（rank=0, world_size=1）
    logger.info("开始模拟分布式训练 (1 个进程)")
    simulate_distributed_training(num_epochs=3, rank=0, world_size=1)

    logger.info("\n" + "="*50 + "\n")

    # 多进程模拟（这里只展示逻辑，实际多进程需要使用 torch.multiprocessing）
    logger.info("多进程分布式训练配置示例:")
    logger.info("使用 torch.distributed.launch 或 torchrun")
    logger.info("命令: torchrun --nproc_per_node=2 train.py")
```

---

## 总结与学习建议

### 关键概念回顾

1. **Dataset**: 定义数据如何加载和预处理
2. **DataLoader**: 定义数据如何批量化和并行化加载
3. **Sampler**: 控制数据采样顺序和策略
4. **Transforms**: 对数据进行预处理和增强
5. **Collate Function**: 自定义批次的组织方式

### 进阶技巧

- 使用 `.to(device)` 在加载时移动数据到 GPU
- 使用 `pin_memory=True` 加速 CPU 到 GPU 的数据传输
- 使用 `num_workers > 0` 实现多进程数据加载
- 使用 `persistent_workers=True` 保持工作进程活动以减少开销

### 性能优化建议

1. 根据数据类型选择合适的 `num_workers`
2. 使用 `persistent_workers=True` 减少进程创建开销
3. 合理设置 `prefetch_factor` 预加载数据
4. 监控数据加载时间，确保不成为训练的瓶颈
5. 对大规模数据集使用 `DistributedSampler` 和多 GPU

### 常见陷阱

- 忘记在循环中调用 `.set_epoch()` 导致数据分布不同步
- 在 collate_fn 中进行过于复杂的操作导致性能下降
- 不正确地处理可变长度数据导致形状不匹配
- 在数据增强中应用过度变换导致数据分布改变过大
