import os
import sys
import json
import time
import pickle
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import logging
from logging.handlers import RotatingFileHandler
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

sys.path.append('../')
sys.path.append('../finetune/')
from model import Kronos, KronosTokenizer, KronosPredictor
from config_loader import CustomFinetuneConfig


class PretrainKlineDataset(Dataset):
    """
    预训练数据集类，支持从多个 CSV 文件加载数据
    与微调数据集的主要区别：
    1. 支持多文件批量加载
    2. 数据量更大，需要更高效的采样策略
    3. 跨股票/市场的随机采样
    """
    
    def __init__(self, data_dir, data_type='train', lookback_window=512, predict_window=48, 
                 clip=5.0, seed=100, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0,
                 max_files=-1, sample_per_file=1000, max_total_records=500000):
        self.data_dir = data_dir
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_files = max_files  # -1 表示加载所有文件
        self.sample_per_file = sample_per_file  # 每个文件的样本数
        self.max_total_records = max_total_records  # 最大总记录数，防止内存溢出
        
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        
        self.py_rng = random.Random(seed)
        
        self.all_data = []  # 存储所有文件的数据
        self.file_boundaries = []  # 每个文件的起始和结束索引
        self.n_samples = 0
            
        self._load_and_preprocess_data()
        self._compute_total_samples()
        
        print(f"[{data_type.upper()}] Total files loaded: {len(self.file_boundaries)}")
        print(f"[{data_type.upper()}] Total records: {sum([b[1]-b[0] for b in self.file_boundaries])}")
        print(f"[{data_type.upper()}] Available samples: {self.n_samples}")
    
    def _load_and_preprocess_data(self):
        """加载目录下的所有 CSV 文件"""
        data_path = Path(self.data_dir)
        csv_files = list(data_path.glob('*.csv'))
        
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        if self.max_files > 0:
            csv_files = csv_files[:self.max_files]
        
        print(f"Found {len(csv_files)} CSV files, loading...")
        
        cumulative_length = 0
        total_records = 0
        
        for i, csv_file in enumerate(csv_files):
            try:
                df = pd.read_csv(str(csv_file))
                
                # 数据预处理
                try:
                    df['timestamps'] = pd.to_datetime(df['timestamps'], utc=True)
                except Exception as e:
                    print(f"Warning: Error parsing timestamps in {csv_file.name}, trying alternative method")
                    df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
                
                df = df.sort_values('timestamps').reset_index(drop=True)
                
                # 添加时间特征
                df['minute'] = df['timestamps'].dt.minute
                df['hour'] = df['timestamps'].dt.hour
                df['weekday'] = df['timestamps'].dt.weekday
                df['day'] = df['timestamps'].dt.day
                df['month'] = df['timestamps'].dt.month
                
                # 提取特征
                file_data = df[self.feature_list + self.time_feature_list].copy()
                
                # 缺失值处理 - 使用 ffill 替代 fillna(method='ffill')
                if file_data.isnull().any().any():
                    print(f"Warning: Missing values in {csv_file.name}, performing forward fill")
                    file_data = file_data.ffill()
                
                # 记录文件边界
                file_length = len(file_data)
                if file_length >= self.window:
                    # 检查是否超过最大记录数
                    if total_records + file_length > self.max_total_records:
                        print(f"Warning: Reached max total records limit ({self.max_total_records}), stopping load")
                        break
                    
                    self.all_data.append(file_data)
                    self.file_boundaries.append((cumulative_length, cumulative_length + file_length))
                    cumulative_length += file_length
                    total_records += file_length
                    
                    if (i + 1) % 10 == 0:
                        print(f"Loaded {i+1}/{len(csv_files)} files...")
                        
            except Exception as e:
                print(f"Error loading {csv_file.name}: {str(e)}, skipping...")
                continue
        
        if len(self.all_data) == 0:
            raise ValueError("No valid data loaded from any file")
        
        print(f"Successfully loaded {len(self.all_data)} files ({total_records} total records)")
    
    def _compute_total_samples(self):
        """计算总样本数"""
        # 简化计算：直接基于所有文件的可用样本数
        total_samples = 0
        for file_data in self.all_data:
            n_samples = max(0, len(file_data) - self.window + 1)
            if n_samples > 0:
                # 训练集使用全部样本，验证/测试集按比例减少
                if self.data_type == 'train':
                    total_samples += n_samples
                else:
                    # 验证集和测试集只取部分样本（避免过多）
                    total_samples += min(n_samples, self.sample_per_file)
        
        self.n_samples = total_samples
    
    def set_epoch_seed(self, epoch):
        """设置每个 epoch 的随机种子"""
        epoch_seed = self.seed + epoch
        self.py_rng.seed(epoch_seed)
        self.current_epoch = epoch
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """随机选择一个文件并返回样本"""
        if len(self.all_data) == 0:
            raise ValueError("No data loaded")
        
        # 随机选择一个文件
        file_idx = self.py_rng.randint(0, len(self.all_data) - 1)
        file_data = self.all_data[file_idx]
        
        max_start = len(file_data) - self.window
        if max_start <= 0:
            # 如果当前文件太短，尝试下一个文件
            for i in range(len(self.all_data)):
                test_file_data = self.all_data[i]
                if len(test_file_data) >= self.window:
                    file_data = test_file_data
                    max_start = len(file_data) - self.window
                    break
            
            # 如果所有文件都太短，抛出错误
            if max_start <= 0:
                raise ValueError(f"No file has enough data (window={self.window})")
        
        # 训练时随机采样，验证/测试时顺序采样
        if self.data_type == 'train':
            epoch = getattr(self, 'current_epoch', 0)
            start_idx = (idx * 9973 + (epoch + 1) * 104729) % (max_start + 1)
        else:
            start_idx = idx % (max_start + 1)
        
        end_idx = start_idx + self.window
        
        window_data = file_data.iloc[start_idx:end_idx]
        
        x = window_data[self.feature_list].values.astype(np.float32)
        x_stamp = window_data[self.time_feature_list].values.astype(np.float32)
        
        # 标准化和截断
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)
        
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        
        return x_tensor, x_stamp_tensor


def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"pretrain_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    log_file = os.path.join(log_dir, f"pretrain_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)
    
    logger.info(f"=== Pretrain Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def create_dataloaders(config):
    """创建数据加载器"""
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating data loaders...")
    
    train_dataset = PretrainKlineDataset(
        data_dir=config.data_path,
        data_type='train',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        max_files=getattr(config, 'max_files', -1),
        sample_per_file=getattr(config, 'sample_per_file', 1000),
        max_total_records=getattr(config, 'max_total_records', 500000)
    )
    
    val_dataset = PretrainKlineDataset(
        data_dir=config.data_path,
        data_type='val',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        max_files=getattr(config, 'max_files', 10),  # 验证集只使用前 10 个文件
        sample_per_file=getattr(config, 'sample_per_file', 500),
        max_total_records=getattr(config, 'max_total_records', 200000)
    )
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        sampler=val_sampler
    )
    
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def train_model(model, tokenizer, device, config, save_dir, logger):
    """训练模型"""
    logger.info("Starting training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.basemodel_epochs,
        pct_start=0.03,
        div_factor=10
    )
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    best_val_loss = float('inf')
    batch_idx_global = 0
    
    for epoch in range(config.basemodel_epochs):
        epoch_start_time = time.time()
        model.train()
        
        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            
            logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            train_batches += 1
            
            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_msg = (f"[Epoch {epoch+1}/{config.basemodel_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {loss.item():.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)
            
            batch_idx_global += 1
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                
                logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                
                val_loss += loss.item()
                val_batches += 1
        
        if use_ddp:
            tensor_sum = torch.tensor([epoch_train_loss, train_batches, val_loss, val_batches], dtype=torch.float64, device=device)
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            epoch_train_loss_all = tensor_sum[0].item()
            train_batches_all = int(tensor_sum[1].item())
            val_loss_all = tensor_sum[2].item()
            val_batches_all = int(tensor_sum[3].item())
            avg_train_loss = (epoch_train_loss_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_val_loss = (val_loss_all / val_batches_all) if val_batches_all > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
                       f"Training Loss: {avg_train_loss:.4f}\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n"
                       f"Epoch Time: {epoch_time:.2f} seconds\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = f"Best model saved to: {model_save_path} (validation loss: {best_val_loss:.4f})"
                logger.info(save_msg)
                print(save_msg)
    
    return best_val_loss


def main():
    import argparse,os
    
    parser = argparse.ArgumentParser(description='Kronos Pretraining Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = CustomFinetuneConfig(args.config)
    
    os.makedirs(config.basemodel_save_path, exist_ok=True)
    
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # 预训练模式下，不加载预训练权重，随机初始化
    logger.info("Random initialization for pretraining (not loading pretrained weights)...")
    print("Random initialization for pretraining (not loading pretrained weights)...")
    
    # 加载或初始化 Tokenizer（随机初始化）
    cfg_path_tok = os.path.join(config.pretrained_tokenizer_path, 'config.json')
    if os.path.exists(cfg_path_tok):
        with open(cfg_path_tok, 'r') as f:
            arch_t = json.load(f)
    else:
        # 默认配置
        arch_t = {
            'd_in': 6,
            'd_model': 256,
            'n_heads': 4,
            'ff_dim': 512,
            'n_enc_layers': 4,
            'n_dec_layers': 4,
            'ffn_dropout_p': 0.0,
            'attn_dropout_p': 0.0,
            'resid_dropout_p': 0.0,
            's1_bits': 10,
            's2_bits': 10,
            'beta': 0.05,
            'gamma0': 1.0,
            'gamma': 1.1,
            'zeta': 0.05,
            'group_size': 4
        }
    
    tokenizer = KronosTokenizer(
        d_in=arch_t.get('d_in', 6),
        d_model=arch_t.get('d_model', 256),
        n_heads=arch_t.get('n_heads', 4),
        ff_dim=arch_t.get('ff_dim', 512),
        n_enc_layers=arch_t.get('n_enc_layers', 4),
        n_dec_layers=arch_t.get('n_dec_layers', 4),
        ffn_dropout_p=arch_t.get('ffn_dropout_p', 0.0),
        attn_dropout_p=arch_t.get('attn_dropout_p', 0.0),
        resid_dropout_p=arch_t.get('resid_dropout_p', 0.0),
        s1_bits=arch_t.get('s1_bits', 10),
        s2_bits=arch_t.get('s2_bits', 10),
        beta=arch_t.get('beta', 0.05),
        gamma0=arch_t.get('gamma0', 1.0),
        gamma=arch_t.get('gamma', 1.1),
        zeta=arch_t.get('zeta', 0.05),
        group_size=arch_t.get('group_size', 4)
    )

    # 加载或初始化 Predictor（随机初始化）
    cfg_path = os.path.join(config.pretrained_predictor_path, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            arch = json.load(f)
    else:
        # 默认配置（Kronos-small）
        arch = {
            's1_bits': 10,
            's2_bits': 10,
            'n_layers': 12,
            'd_model': 832,
            'n_heads': 16,
            'ff_dim': 2048,
            'ffn_dropout_p': 0.2,
            'attn_dropout_p': 0.0,
            'resid_dropout_p': 0.2,
            'token_dropout_p': 0.0,
            'learn_te': True
        }
    
    model = Kronos(
        s1_bits=arch.get('s1_bits', 10),
        s2_bits=arch.get('s2_bits', 10),
        n_layers=arch.get('n_layers', 12),
        d_model=arch.get('d_model', 832),
        n_heads=arch.get('n_heads', 16),
        ff_dim=arch.get('ff_dim', 2048),
        ffn_dropout_p=arch.get('ffn_dropout_p', 0.2),
        attn_dropout_p=arch.get('attn_dropout_p', 0.0),
        resid_dropout_p=arch.get('resid_dropout_p', 0.2),
        token_dropout_p=arch.get('token_dropout_p', 0.0),
        learn_te=arch.get('learn_te', True)
    )
    
    tokenizer = tokenizer.to(device)
    model = model.to(device)
    
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_size:,}")
    print(f"Model parameters: {model_size:,}")
    
    logger.info("=== Pretraining Configuration ===")
    logger.info(f"Data directory: {config.data_path}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.predictor_learning_rate}")
    logger.info(f"Training epochs: {config.basemodel_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Max files to load: {getattr(config, 'max_files', -1)}")
    logger.info(f"Samples per file: {getattr(config, 'sample_per_file', 1000)}")
    
    logger.info("Starting pretraining...")
    print("Starting pretraining...")
    best_val_loss = train_model(model, tokenizer, device, config, config.basemodel_save_path, logger)
    
    final_msg = f"Pretraining completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.basemodel_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
