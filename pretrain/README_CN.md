# Kronos 预训练指南

## 概述

预训练（Pretraining）是在大规模多股票/市场数据上从头开始训练 Kronos 模型的过程。与微调（Finetuning）不同，预训练不加载任何预训练权重，而是随机初始化所有参数，在大量数据上进行训练以获得通用的金融市场表示能力。

## 预训练 vs 微调

| 特性 | 预训练 (Pretraining) | 微调 (Finetuning) |
|------|---------------------|------------------|
| **初始化** | 随机初始化 | 加载预训练权重 |
| **数据集** | 多股票/市场的大规模数据 | 单只股票或特定数据集 |
| **训练轮次** | 较多（50-100+ epochs） | 较少（20-30 epochs） |
| **学习率** | 较低（稳定训练） | 较高（快速适应） |
| **目标** | 学习通用金融模式 | 适应特定数据分布 |
| **计算资源** | 需要更多 GPU 和时间 | 相对较少资源 |

## 目录结构

```
pretrain/
├── README.md                 # 本文件
├── README_CN.md             # 中文版本
├── pretrain_kronos.py       # 预训练主脚本
├── config.yaml              # 预训练配置文件
└── configs/                 # 其他配置示例
    └── pretrain_example.yaml
```

## 快速开始

### 1. 准备数据

将多个股票的 CSV 数据文件放在同一目录下：

```
data/
├── stock_001.csv
├── stock_002.csv
├── stock_003.csv
└── ...
```

每个 CSV 文件应包含以下列：
- `timestamps`: 时间戳
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `amount`: 成交额（可选）

### 2. 修改配置文件

编辑 [`config.yaml`](config.yaml)：

```yaml
data:
  data_dir: "path/to/your/csv/files/"  # 数据目录
  max_files: -1  # -1=全部加载，或指定文件数量
  sample_per_file: 1000  # 每个文件的样本数
  
training:
  basemodel_epochs: 100  # 预训练轮次
  batch_size: 64  # 根据 GPU 内存调整
  
model_paths:
  exp_name: "my_pretrain_experiment"  # 实验名称

experiment:
  pre_trained_tokenizer: false  # 必须为 false（随机初始化）
  pre_trained_predictor: false  # 必须为 false（随机初始化）
```

### 3. 启动预训练

#### 单机单卡训练

```bash
cd pretrain
python pretrain_kronos.py --config config.yaml
```

#### 多 GPU 分布式训练（推荐）

```bash
cd pretrain
torchrun --nproc_per_node=4 pretrain_kronos.py --config config.yaml
```

#### Windows PowerShell

```powershell
cd pretrain
python pretrain_kronos.py --config config.yaml
```

## 配置说明

### 数据配置

```yaml
data:
  data_dir: "D:/dev/Kronos/download/data/"  # CSV 文件目录
  lookback_window: 512  # 回溯窗口大小
  predict_window: 48    # 预测窗口大小
  clip: 5.0            # 梯度截断值
  
  # 数据集划分
  train_ratio: 0.9     # 训练集比例
  val_ratio: 0.1       # 验证集比例
  test_ratio: 0.0      # 测试集比例
  
  # 数据加载控制
  max_files: -1        # 最大文件数（-1=全部）
  sample_per_file: 1000 # 每文件采样样本数
```

### 训练配置

```yaml
training:
  tokenizer_epochs: 50      # Tokenizer 训练轮次
  basemodel_epochs: 100     # Predictor 训练轮次
  batch_size: 64           # 批次大小
  log_interval: 100        # 日志间隔
  
  # 学习率设置
  tokenizer_learning_rate: 0.0002   # Tokenizer 学习率
  predictor_learning_rate: 0.00001  # Predictor 学习率（较低）
  
  # Adam 优化器参数
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  
  accumulation_steps: 2  # 梯度累积步数
```

### 模型配置

```yaml
model_paths:
  # 参考模型路径（仅用于获取架构配置）
  pretrained_tokenizer: "../pretrained/Kronos-Tokenizer-2k"
  pretrained_predictor: "../pretrained/Kronos-small"
  
  exp_name: "kronos_pretrain_multi_stocks"  # 实验名称
  base_path: "../output/pretrain/"          # 输出基础路径
  
  # 以下路径自动生成
  base_save_path: ""
  finetuned_tokenizer: ""
```

### 实验配置

```yaml
experiment:
  name: "kronos_pretrain_multi_stocks"
  description: "Pretraining Kronos on multiple stocks"
  
  train_tokenizer: true       # 训练 Tokenizer
  train_basemodel: true       # 训练 Predictor
  skip_existing: false        # 不跳过已有模型
  
  # 关键配置：预训练必须为 false
  pre_trained_tokenizer: false   # 不加载预训练 Tokenizer
  pre_trained_predictor: false   # 不加载预训练 Predictor
```

## 输出结果

训练完成后，输出目录结构：

```
output/pretrain/kronos_pretrain_multi_stocks/
├── logs/                           # 训练日志
│   ├── pretrain_training_rank_0.log
│   └── ...
├── tokenizer/                      # 训练后的 Tokenizer
│   └── best_model/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer_config.json
└── basemodel/                      # 训练后的 Predictor
    └── best_model/
        ├── config.json
        ├── model.safetensors
        └── special_tokens_map.json
```

## 分布式训练

### 多 GPU 训练（单机）

```bash
torchrun --nproc_per_node=4 pretrain_kronos.py --config config.yaml
```

### 多机多卡训练

```bash
# Node 0
torchrun --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.1.1" \
         --master_port="1234" \
         --nproc_per_node=4 \
         pretrain_kronos.py --config config.yaml

# Node 1
torchrun --nnodes=2 \
         --node_rank=1 \
         --master_addr="192.168.1.1" \
         --master_port="1234" \
         --nproc_per_node=4 \
         pretrain_kronos.py --config config.yaml
```

## 性能优化建议

### 1. 数据加载优化

```yaml
training:
  num_workers: 8  # 增加数据加载进程数
  batch_size: 64  # 根据 GPU 内存调整
```

### 2. 混合精度训练（未来支持）

```yaml
training:
  use_amp: true  # TODO: 自动混合精度
```

### 3. 梯度累积

当 GPU 内存有限时：

```yaml
training:
  batch_size: 32       # 减小批次
  accumulation_steps: 4  # 累积 4 步 = 有效 batch 128
```

### 4. 数据采样策略

```yaml
data:
  max_files: 100        # 限制文件数量进行测试
  sample_per_file: 500  # 减少样本数加速
```

## 监控训练进度

### 查看日志

```bash
tail -f output/pretrain/kronos_pretrain_multi_stocks/logs/pretrain_training_rank_0.log
```

### 关键指标

- **Training Loss**: 训练损失，应逐渐下降
- **Validation Loss**: 验证损失，用于早停和模型选择
- **LR**: 学习率，按 OneCycleLR 策略变化

### 使用 Comet.ml（可选）

```yaml
experiment:
  use_comet: true
```

## 常见问题

### Q1: 预训练需要多少数据？

A: 建议至少 10+ 只股票，每只股票 1 年 + 的日线数据，或更多。数据量越大，模型学习效果越好。

### Q2: 预训练需要多长时间？

A: 取决于：
- 数据量（股票数量 × 时间跨度）
- GPU 数量
- 批次大小和训练轮次

示例：100 只股票 × 250 天/年 × 2 年，使用 4×V100 GPU，约需 2-3 天。

### Q3: 可以中途停止并恢复训练吗？

A: 当前版本不支持断点续训。如需此功能，请提交 Issue。

### Q4: 预训练后如何使用模型？

A: 预训练完成的模型可以作为新的预训练权重，用于后续微调：

```yaml
# 微调配置中
pretrained_predictor: "path/to/pretrain/best_model"
pre_trained_predictor: true
```

### Q5: OOM（显存不足）怎么办？

A: 尝试以下方法：
1. 减小 `batch_size`
2. 增加 `accumulation_steps`
3. 减小 `lookback_window`
4. 使用更小的模型（如 Kronos-mini）

## 与微调的配合

预训练完成后，可以在微调中使用：

```yaml
# finetune/config.yaml
model_paths:
  pretrained_predictor: "../output/pretrain/kronos_pretrain_multi_stocks/basemodel/best_model"
  
experiment:
  pre_trained_predictor: true  # 加载预训练权重
```

## 代码结构

### PretrainKlineDataset

核心数据集类，特点：
- 支持多 CSV 文件批量加载
- 跨文件随机采样
- 高效的数据预处理和标准化
- 每个 epoch 动态种子确保多样性

### 训练流程

```python
main()
  ├─ 加载配置
  ├─ 随机初始化模型（Tokenizer + Predictor）
  ├─ 创建 DataLoader（多文件采样）
  ├─ train_model()
  │   ├─ for epoch in epochs:
  │   │   ├─ Training loop
  │   │   ├─ Validation loop
  │   │   └─ Save best model
  │   └─ return best_val_loss
  └─ 保存最终模型
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

与 Kronos 项目主许可证一致。

## 引用

如果您使用了预训练脚本，请引用 Kronos 论文：

```bibtex
@article{kronos2024,
  title={Kronos: A Foundation Model for Financial K-line Time Series},
  author={...},
  journal={...},
  year={2024}
}
```
