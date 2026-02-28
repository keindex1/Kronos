# 股票数据下载工具使用指南

## 概述

这是一个功能强大的股票数据下载工具套件，支持多种数据源和格式，专为金融数据分析和机器学习模型训练设计。

## 功能特性

- 🌍 **多数据源**: 支持 akshare (中国股市)、yfinance (全球股市)、tushare (中国股市)
- ⏰ **多周期**: 支持1分钟到月线的各种K线数据
- 📊 **标准格式**: 统一的CSV输出格式 (timestamps, open, high, low, close, volume, amount)
- 🚀 **批量下载**: 支持批量下载多只股票数据
- 🔧 **自动清理**: 自动数据清理和格式标准化

## 文件说明

### 核心文件
- `download_stock_data.py`: 主下载脚本
- `batch_download.py`: 批量下载脚本
- `stock_config.py`: 股票配置文件
- `plot_kline.py`: K线图绘制工具

## 使用方法

### 1. 单股票下载

#### 基本用法
```bash
# 下载600977最近30天的5分钟数据
python download_stock_data.py --symbol 600977

# 下载特定时间范围的日线数据
python download_stock_data.py --symbol 000001 --period 1day --start 2024-01-01 --end 2024-12-31

# 下载美股数据 (使用yfinance)
python download_stock_data.py --symbol AAPL --source yfinance --period 1day --days 365
```

#### 参数说明
- `--symbol`: 股票代码 (必须)
- `--source`: 数据源 (akshare/yfinance/tushare，默认akshare)
- `--period`: 数据周期 (1min/5min/15min/30min/60min/1day/1week/1month，默认5min)
- `--days`: 最近N天数据
- `--start`: 开始日期 (YYYY-MM-DD)
- `--end`: 结束日期 (YYYY-MM-DD)
- `--output`: 输出文件名 (默认自动生成)

### 2. 批量下载

#### 基本用法
```bash
# 下载多只股票
python batch_download.py --stocks 600000,000001,600519 --period 5min --days 30

# 使用预定义股票组合
python batch_download.py --config a_stocks --period 1day --days 365

# 下载美股组合
python batch_download.py --config us_stocks --source yfinance --period 1day --days 180
```

#### 预定义组合
- `a_stocks`: 热门A股 (浦发银行、招商银行、贵州茅台等)
- `us_stocks`: 热门美股 (苹果、微软、特斯拉等)
- `hk_stocks`: 热门港股 (阿里巴巴、腾讯、美团等)

### 3. 数据格式

输出的CSV文件包含以下标准格式的字段：

```csv
timestamps,open,high,low,close,volume,amount
2024-06-18 09:30:00,11.27,11.28,11.26,11.27,379.0,427161.0
2024-06-18 09:35:00,11.27,11.28,11.27,11.27,277.0,312192.0
```

字段说明：
- `timestamps`: 时间戳
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量 (股)
- `amount`: 成交额 (元)

## 使用示例

### 示例1：下载分析用数据
```bash
# 下载600977最近一个月的5分钟数据用于分析
python download_stock_data.py --symbol 600977 --period 5min --days 30 --output analysis/600977_5min.csv
```

### 示例2：训练数据准备
```bash
# 批量下载A股数据用于模型训练
python batch_download.py --config a_stocks --period 5min --days 90 --output-dir training_data
```

### 示例3：历史回测数据
```bash
# 下载贵州茅台2024年全年日线数据
python download_stock_data.py --symbol 600519 --period 1day --start 2024-01-01 --end 2024-12-31 --output backtest/moutai_2024.csv
```

### 示例4：实时监控数据
```bash
# 下载最近3天的分钟级数据
python download_stock_data.py --symbol 000001 --period 1min --days 3 --output monitor/realtime.csv
```

## 数据源说明

### akshare (推荐中国股市)
- 支持A股、港股、期货等
- 数据质量高，更新及时
- 免费使用，无需注册

### yfinance (推荐国际股市)  
- 支持全球主要股市
- 包含美股、欧股等
- 免费使用，数据来源Yahoo Finance

### tushare (中国股市专业版)
- 需要注册获取token
- 数据更全面，包含财务数据
- 有免费和付费版本

## 常见问题

### Q: 如何获取股票代码？
A: 
- A股: 6位数字 (如600000, 000001)
- 美股: 股票简称 (如AAPL, TSLA)
- 港股: 5位数字 (如00700, 09988)

### Q: 数据下载失败怎么办？
A:
1. 检查网络连接
2. 确认股票代码正确
3. 尝试更换数据源
4. 检查时间范围是否合理

### Q: 如何处理数据缺失？
A: 脚本会自动清理无效数据，但建议:
1. 检查交易时间 (避免非交易时段)
2. 使用较长的时间范围
3. 对比多个数据源

### Q: 可以自定义输出格式吗？
A: 目前输出标准CSV格式，可以通过pandas进一步处理:

```python
import pandas as pd
df = pd.read_csv('your_data.csv')
# 自定义处理...
```

## 性能优化

### 大批量下载建议
```bash
# 分批下载，避免请求过频
python batch_download.py --config a_stocks --period 1day --days 30

# 使用更长的时间间隔
time.sleep(1)  # 在代码中添加延迟
```

### 存储优化
- 按周期分目录存储: `data/5min/`, `data/1day/`
- 定期压缩历史数据
- 使用数据库存储大量数据

## 与Kronos模型集成

下载的数据可以直接用于Kronos模型训练:

```bash
# 1. 下载训练数据
python download_stock_data.py --symbol 600977 --period 5min --days 90 --output finetune_csv/data/600977_train.csv

# 2. 使用快速训练脚本
cd finetune_csv
python fast_finetune_tokenizer.py --config configs/quick_test.yaml

# 3. 进行预测
python examples/prediction_cn_markets_min.py --symbol 600977
```

## 技术支持

如有问题或建议，请参考项目文档或提交issue。

## 版权声明

本工具基于开源库开发，请遵守相关数据源的使用条款。