# 预训练脚本修复说明

## 问题诊断

在初次运行时遇到了以下问题：

### 1. 退出码 3221225477 (内存访问违规)
**原因**: 
- 数据量过大（245,221 条记录）导致 CPU 内存溢出
- Pandas 弃用警告未处理
- 时间戳解析错误导致数据加载失败

### 2. Pandas 弃用警告
```
FutureWarning: DataFrame.fillna with 'method' is deprecated
FutureWarning: parsing datetimes with mixed time zones will raise an error
```

## 已修复的问题

### ✅ 修复 1: 内存保护机制
添加了 `max_total_records` 参数限制最大数据加载量：
- **训练集**: 默认 500,000 条记录
- **验证集**: 默认 200,000 条记录

```python
# config.yaml
data:
  max_total_records: 300000  # 可根据实际内存调整
```

### ✅ 修复 2: 时间戳解析优化
改进了时间戳处理方式，支持混合时区：

```python
# 优先使用 UTC 解析，失败时使用容错模式
try:
    df['timestamps'] = pd.to_datetime(df['timestamps'], utc=True)
except Exception as e:
    df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
```

### ✅ 修复 3: Pandas 弃用方法更新
将弃用的 `fillna(method='ffill')` 替换为 `ffill()`：

```python
# 旧代码
file_data = file_data.fillna(method='ffill')

# 新代码
file_data = file_data.ffill()
```

### ✅ 修复 4: 数据采样优化
改进了 `__getitem__` 方法的健壮性：
- 添加空数据检查
- 自动跳过过短的文件
- 确保不会访问无效索引

### ✅ 修复 5: 验证集大小控制
验证集样本数现在受 `sample_per_file` 参数控制，避免过多的验证样本。

## 修改的文件

1. **pretrain_kronos.py**
   - 添加 `max_total_records` 参数到 `PretrainKlineDataset`
   - 改进时间戳解析逻辑
   - 更新缺失值处理方法
   - 优化数据采样策略
   - 添加内存保护检查

2. **config.yaml**
   - 添加 `max_total_records: 300000` 配置项
   - 添加中英文注释说明

3. **create_dataloaders 函数**
   - 传递 `max_total_records` 参数到数据集

## 使用方法

### 推荐配置（CPU/单 GPU）

```yaml
data:
  data_path: "D:/dev/Kronos/download/data/"
  max_files: -1              # 加载所有文件
  max_total_records: 300000  # 限制总记录数
  
training:
  batch_size: 32            # CPU 建议减小批次
  basemodel_epochs: 50      # 预训练轮次
```

### 多 GPU 分布式训练

```yaml
data:
  max_total_records: 500000  # GPU 充足可增加限制
  
training:
  batch_size: 64
  num_workers: 8
```

## 运行测试

### 快速测试（小数据集）

```bash
cd pretrain
python pretrain_kronos.py --config configs/pretrain_example.yaml
```

该配置仅加载前 50 个文件，每个文件 500 个样本，适合快速验证。

### 完整训练

```bash
python pretrain_kronos.py --config config.yaml
```

## 性能优化建议

### CPU 训练
- 减小 `batch_size` 到 16-32
- 设置 `max_total_records: 200000`
- 增加 `num_workers: 4`

### GPU 训练
- 使用 `batch_size: 64`
- 设置 `max_total_records: 500000`
- 使用 `torchrun` 进行分布式训练

### 内存不足时的解决方案

1. **立即生效**: 减小 `max_total_records`
   ```yaml
   max_total_records: 100000
   ```

2. **减少文件数**: 设置 `max_files`
   ```yaml
   max_files: 20  # 只加载前 20 个文件
   ```

3. **减小批次**: 降低 `batch_size`
   ```yaml
   batch_size: 16
   ```

## 已知限制

1. **AAPL 数据文件问题**: 某些 AAPL 数据文件的时间戳格式可能不兼容，会被自动跳过
2. **CPU 训练速度慢**: 建议使用 GPU 或减少数据量
3. **混合时区警告**: 已通过 `utc=True` 参数处理

## 下一步计划

- [ ] 添加断点续训功能
- [ ] 支持自动混合精度（AMP）训练
- [ ] 添加 Comet.ml 实验跟踪
- [ ] 优化大数据集加载性能

## 反馈与支持

如遇到其他问题，请查看日志文件：
```
output/pretrain/kronos_pretrain_multi_stocks/logs/pretrain_training_rank_0.log
```

或提交 Issue 到项目仓库。
