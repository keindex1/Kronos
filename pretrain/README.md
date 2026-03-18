# Kronos Pretraining Guide

## Overview

Pretraining is the process of training the Kronos model from scratch on large-scale multi-stock/market data. Unlike finetuning, pretraining does not load any pretrained weights but instead randomly initializes all parameters and trains on massive data to acquire general financial market representation capabilities.

## Pretraining vs Finetuning

| Feature | Pretraining | Finetuning |
|---------|-------------|------------|
| **Initialization** | Random initialization | Load pretrained weights |
| **Dataset** | Large-scale multi-stock/market data | Single stock or specific dataset |
| **Training Epochs** | More (50-100+ epochs) | Less (20-30 epochs) |
| **Learning Rate** | Lower (stable training) | Higher (fast adaptation) |
| **Objective** | Learn general financial patterns | Adapt to specific data distribution |
| **Compute Resources** | Requires more GPUs and time | Relatively fewer resources |

## Directory Structure

```
pretrain/
├── README.md                 # This file
├── README_CN.md             # Chinese version
├── pretrain_kronos.py       # Pretraining main script
├── config.yaml              # Pretraining configuration
└── configs/                 # Additional config examples
    └── pretrain_example.yaml
```

## Quick Start

### 1. Prepare Data

Place multiple stock CSV data files in the same directory:

```
data/
├── stock_001.csv
├── stock_002.csv
├── stock_003.csv
└── ...
```

Each CSV file should contain the following columns:
- `timestamps`: Timestamp
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume
- `amount`: Amount (optional)

### 2. Modify Configuration File

Edit [`config.yaml`](config.yaml):

```yaml
data:
  data_dir: "path/to/your/csv/files/"  # Data directory
  max_files: -1  # -1=load all, or specify number
  sample_per_file: 1000  # Samples per file
  
training:
  basemodel_epochs: 100  # Pretraining epochs
  batch_size: 64  # Adjust based on GPU memory
  
model_paths:
  exp_name: "my_pretrain_experiment"  # Experiment name

experiment:
  pre_trained_tokenizer: false  # Must be false (random initialization)
  pre_trained_predictor: false  # Must be false (random initialization)
```

### 3. Start Pretraining

#### Single GPU Training

```bash
cd pretrain
python pretrain_kronos.py --config config.yaml
```

#### Multi-GPU Distributed Training (Recommended)

```bash
cd pretrain
torchrun --nproc_per_node=4 pretrain_kronos.py --config config.yaml
```

#### Windows PowerShell

```powershell
cd pretrain
python pretrain_kronos.py --config config.yaml
```

## Configuration Details

### Data Configuration

```yaml
data:
  data_dir: "D:/dev/Kronos/download/data/"  # CSV files directory
  lookback_window: 512  # Lookback window size
  predict_window: 48    # Predict window size
  clip: 5.0            # Gradient clipping value
  
  # Dataset split
  train_ratio: 0.9     # Training set ratio
  val_ratio: 0.1       # Validation set ratio
  test_ratio: 0.0      # Test set ratio
  
  # Data loading control
  max_files: -1        # Max files (-1=all)
  sample_per_file: 1000 # Samples per file
```

### Training Configuration

```yaml
training:
  tokenizer_epochs: 50      # Tokenizer training epochs
  basemodel_epochs: 100     # Predictor training epochs
  batch_size: 64           # Batch size
  log_interval: 100        # Log interval
  
  # Learning rate settings
  tokenizer_learning_rate: 0.0002   # Tokenizer learning rate
  predictor_learning_rate: 0.00001  # Predictor learning rate (lower)
  
  # Adam optimizer parameters
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  
  accumulation_steps: 2  # Gradient accumulation steps
```

### Model Configuration

```yaml
model_paths:
  # Reference model paths (for architecture config only)
  pretrained_tokenizer: "../pretrained/Kronos-Tokenizer-2k"
  pretrained_predictor: "../pretrained/Kronos-small"
  
  exp_name: "kronos_pretrain_multi_stocks"  # Experiment name
  base_path: "../output/pretrain/"          # Output base path
  
  # These paths are auto-generated
  base_save_path: ""
  finetuned_tokenizer: ""
```

### Experiment Configuration

```yaml
experiment:
  name: "kronos_pretrain_multi_stocks"
  description: "Pretraining Kronos on multiple stocks"
  
  train_tokenizer: true       # Train Tokenizer
  train_basemodel: true       # Train Predictor
  skip_existing: false        # Don't skip existing models
  
  # Critical: Must be false for pretraining
  pre_trained_tokenizer: false   # Don't load pretrained Tokenizer
  pre_trained_predictor: false   # Don't load pretrained Predictor
```

## Output Results

After training completes, output directory structure:

```
output/pretrain/kronos_pretrain_multi_stocks/
├── logs/                           # Training logs
│   ├── pretrain_training_rank_0.log
│   └── ...
├── tokenizer/                      # Trained Tokenizer
│   └── best_model/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer_config.json
└── basemodel/                      # Trained Predictor
    └── best_model/
        ├── config.json
        ├── model.safetensors
        └── special_tokens_map.json
```

## Distributed Training

### Multi-GPU Training (Single Machine)

```bash
torchrun --nproc_per_node=4 pretrain_kronos.py --config config.yaml
```

### Multi-Machine Multi-GPU Training

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

## Performance Optimization Tips

### 1. Data Loading Optimization

```yaml
training:
  num_workers: 8  # Increase data loading processes
  batch_size: 64  # Adjust based on GPU memory
```

### 2. Mixed Precision Training (Future)

```yaml
training:
  use_amp: true  # TODO: Automatic Mixed Precision
```

### 3. Gradient Accumulation

When GPU memory is limited:

```yaml
training:
  batch_size: 32       # Reduce batch size
  accumulation_steps: 4  # Accumulate 4 steps = effective batch 128
```

### 4. Data Sampling Strategy

```yaml
data:
  max_files: 100        # Limit file count for testing
  sample_per_file: 500  # Reduce samples to accelerate
```

## Monitoring Training Progress

### View Logs

```bash
tail -f output/pretrain/kronos_pretrain_multi_stocks/logs/pretrain_training_rank_0.log
```

### Key Metrics

- **Training Loss**: Should gradually decrease
- **Validation Loss**: Used for early stopping and model selection
- **LR**: Learning rate, changes according to OneCycleLR schedule

### Using Comet.ml (Optional)

```yaml
experiment:
  use_comet: true
```

## FAQ

### Q1: How much data is needed for pretraining?

A: Recommended at least 10+ stocks, each with 1+ year of daily data, or more. The larger the dataset, the better the model learns.

### Q2: How long does pretraining take?

A: Depends on:
- Data volume (number of stocks × time span)
- Number of GPUs
- Batch size and training epochs

Example: 100 stocks × 250 days/year × 2 years, using 4×V100 GPUs, approximately 2-3 days.

### Q3: Can I stop and resume training midway?

A: Current version doesn't support checkpoint resumption. Please submit an Issue if you need this feature.

### Q4: How to use the model after pretraining?

A: The pretrained model can serve as new pretrained weights for subsequent finetuning:

```yaml
# In finetune config
pretrained_predictor: "path/to/pretrain/best_model"
pre_trained_predictor: true
```

### Q5: What to do if OOM (Out Of Memory)?

A: Try these methods:
1. Reduce `batch_size`
2. Increase `accumulation_steps`
3. Reduce `lookback_window`
4. Use a smaller model (e.g., Kronos-mini)

## Integration with Finetuning

After pretraining completes, you can use it in finetuning:

```yaml
# finetune/config.yaml
model_paths:
  pretrained_predictor: "../output/pretrain/kronos_pretrain_multi_stocks/basemodel/best_model"
  
experiment:
  pre_trained_predictor: true  # Load pretrained weights
```

## Code Structure

### PretrainKlineDataset

Core dataset class features:
- Supports multi-CSV file batch loading
- Cross-file random sampling
- Efficient data preprocessing and normalization
- Dynamic seed per epoch for diversity

### Training Flow

```python
main()
  ├─ Load configuration
  ├─ Randomly initialize models (Tokenizer + Predictor)
  ├─ Create DataLoader (multi-file sampling)
  ├─ train_model()
  │   ├─ for epoch in epochs:
  │   │   ├─ Training loop
  │   │   ├─ Validation loop
  │   │   └─ Save best model
  │   └─ return best_val_loss
  └─ Save final model
```

## Contributing

Issues and Pull Requests are welcome!

## License

Consistent with the main Kronos project license.

## Citation

If you use the pretraining scripts, please cite the Kronos paper:

```bibtex
@article{kronos2024,
  title={Kronos: A Foundation Model for Financial K-line Time Series},
  author={...},
  journal={...},
  year={2024}
}
```
