# 快速开始指南

## 1. 环境准备

确保已安装以下依赖：

```bash
pip install torch>=2.0.0 pandas numpy pyyaml tqdm matplotlib scikit-learn
```

## 2. 数据准备

将多个股票的 CSV 数据文件放在同一目录下，例如：

```
D:/dev/Kronos/download/data/
├── stock_001.csv
├── stock_002.csv
├── stock_003.csv
└── ...
```

每个 CSV 文件应包含：
- timestamps（时间戳）
- open, high, low, close, volume, amount

## 3. 配置修改

编辑 `config.yaml` 文件：

```yaml
data:
  data_path: "你的数据目录"  # 修改为实际路径
  max_total_records: 300000  # 限制总记录数，防止内存溢出
  
training:
  basemodel_epochs: 50      # CPU 建议 50，GPU 建议 100
  batch_size: 32            # CPU 建议 16-32，GPU 建议 64
  
model_paths:
  exp_name: "你的实验名称"  # 自定义实验名称

experiment:
  pre_trained_tokenizer: false   # 必须为 false
  pre_trained_predictor: false   # 必须为 false
```

## 4. 启动预训练

### Windows 用户（推荐）

双击运行：
```
start_pretrain.bat
```

或使用命令行：
```powershell
.\start_pretrain.bat --config config.yaml
```

### Linux/Mac 用户

单机单卡：
```bash
python pretrain_kronos.py --config config.yaml
```

多 GPU 分布式训练：
```bash
torchrun --nproc_per_node=4 pretrain_kronos.py --config config.yaml
```

## 5. 使用示例脚本

运行示例脚本自动检查和启动训练：

```bash
python pretrain_example.py
```

## 6. 查看结果

训练完成后，模型保存在：

```
output/pretrain/<exp_name>/
├── logs/              # 训练日志
├── tokenizer/         # Tokenizer 模型
│   └── best_model/
└── basemodel/         # Predictor 模型
    └── best_model/
```

## 7. 性能优化

### CPU 训练推荐配置

```yaml
training:
  batch_size: 16
  num_workers: 4
  
data:
  max_total_records: 200000
```

### GPU 训练推荐配置

```yaml
training:
  batch_size: 64
  num_workers: 8
  
data:
  max_total_records: 500000
```

### 内存不足时

减小以下参数：
- `max_total_records: 100000`
- `batch_size: 8`
- `max_files: 20`

## 8. 常见问题

**Q: 显存/内存不足怎么办？**  
A: 减小 `batch_size`、`max_total_records` 或 `max_files`

**Q: 如何中断训练？**  
A: 按 Ctrl+C 安全中断

**Q: 训练多久？**  
A: CPU: 数小时到数天；GPU: 数十分钟到数小时（取决于数据量）

**Q: 某些文件加载失败？**  
A: 检查 CSV 格式，确保时间戳列名为 `timestamps`

详细文档请查看 [README_CN.md](README_CN.md) 或 [README.md](README.md)  
问题修复说明请查看 [FIXES.md](FIXES.md)
