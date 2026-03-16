import os
import sys
import time
import random
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

sys.path.append('../')
from model import Kronos, KronosTokenizer

from config_loader import CustomFinetuneConfig
import finetune_base_model as basemodel_module
import finetune_tokenizer as tokenizer_module
from finetune_tokenizer import set_seed, setup_logging as setup_tokenizer_logging
from finetune_base_model import train_model, setup_logging as setup_basemodel_logging


class MultiStockKlineDataset(Dataset):
    DEFAULT_SYMBOL_COLUMN = None
    DEFAULT_SYMBOL_CANDIDATES = ['symbol', 'ts_code', 'code', 'ticker', 'stock_code', 'sec_code']

    def __init__(
        self,
        data_path,
        data_type='train',
        lookback_window=90,
        predict_window=10,
        clip=5.0,
        seed=100,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    ):
        self.data_path = data_path
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        self.py_rng = random.Random(seed)
        self.current_epoch = 0

        self.symbol_column = self._resolve_symbol_column()
        self.segments = []
        self.index_map = []
        self.epoch_index_map = []

        self._load_and_prepare_segments()
        self.n_samples = len(self.index_map)

        print(
            f"[{data_type.upper()}] Multi-stock segments: {len(self.segments)}, "
            f"Available samples: {self.n_samples}"
        )

    def _resolve_symbol_column(self):
        if self.DEFAULT_SYMBOL_COLUMN:
            return self.DEFAULT_SYMBOL_COLUMN

        env_symbol_column = os.environ.get('KRONOS_SYMBOL_COLUMN')
        if env_symbol_column:
            return env_symbol_column

        return None

    def _pick_symbol_column(self, df: pd.DataFrame):
        if self.symbol_column:
            if self.symbol_column not in df.columns:
                raise ValueError(
                    f"Configured symbol column '{self.symbol_column}' not found in CSV columns: {list(df.columns)}"
                )
            return self.symbol_column

        for col in self.DEFAULT_SYMBOL_CANDIDATES:
            if col in df.columns:
                return col

        return None

    def _validate_required_columns(self, df: pd.DataFrame, symbol_col: str):
        required = ['timestamps'] + self.feature_list
        if symbol_col is not None:
            required = [symbol_col] + required

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

    def _resolve_data_files(self):
        raw_path = str(self.data_path).strip()
        if not raw_path:
            raise ValueError("data_path is empty")

        if os.path.exists(raw_path) or any(ch in raw_path for ch in ['*', '?', '[']):
            candidates = [raw_path]
        else:
            candidates = [p.strip() for p in raw_path.replace('\n', ';').replace(',', ';').split(';') if p.strip()]

        data_files = []
        for candidate in candidates:
            if os.path.isdir(candidate):
                data_files.extend(sorted(glob.glob(os.path.join(candidate, '*.csv'))))
            elif any(ch in candidate for ch in ['*', '?', '[']):
                data_files.extend(sorted(glob.glob(candidate)))
            elif os.path.isfile(candidate):
                data_files.append(candidate)

        deduped = []
        seen = set()
        for file_path in data_files:
            full_path = os.path.abspath(file_path)
            if full_path not in seen and full_path.lower().endswith('.csv'):
                deduped.append(full_path)
                seen.add(full_path)

        if not deduped:
            raise ValueError(
                "No CSV files found from data_path. "
                "Use a CSV file path, folder path, glob pattern, or comma/semicolon separated list."
            )

        return deduped

    def _load_and_prepare_segments(self):
        data_files = self._resolve_data_files()
        print(f"[{self.data_type.upper()}] Loading {len(data_files)} CSV file(s)")

        symbol_frames = {}
        detected_symbol_column = None

        for file_path in data_files:
            df = pd.read_csv(file_path)
            symbol_col = self._pick_symbol_column(df)

            self._validate_required_columns(df, symbol_col)

            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.sort_values('timestamps').reset_index(drop=True)
            df['minute'] = df['timestamps'].dt.minute
            df['hour'] = df['timestamps'].dt.hour
            df['weekday'] = df['timestamps'].dt.weekday
            df['day'] = df['timestamps'].dt.day
            df['month'] = df['timestamps'].dt.month

            if symbol_col is None:
                file_symbol = os.path.splitext(os.path.basename(file_path))[0]
                symbol_col = '__kronos_file_symbol__'
                df[symbol_col] = file_symbol
            else:
                detected_symbol_column = symbol_col

            for symbol, group_df in df.groupby(symbol_col, sort=False):
                symbol_frames.setdefault(str(symbol), []).append(group_df.copy())

        if detected_symbol_column is not None:
            print(f"Detected symbol column: {detected_symbol_column}")
        else:
            print("No symbol column found; using each CSV filename as stock symbol")

        usable_symbols = 0
        total_symbols = len(symbol_frames)
        dropped_symbols = []

        for symbol, group_list in symbol_frames.items():
            group_df = pd.concat(group_list, axis=0, ignore_index=True)
            group_df = group_df.sort_values('timestamps').reset_index(drop=True)

            group_data = group_df[self.feature_list + self.time_feature_list].copy()
            if group_data.isnull().any().any():
                group_data = group_data.ffill().bfill()

            split_data = self._split_single_symbol(group_data)
            if split_data is None or len(split_data) < self.window:
                dropped_symbols.append((symbol, len(group_data)))
                continue

            feat_arr = split_data[self.feature_list].values.astype(np.float32)
            time_arr = split_data[self.time_feature_list].values.astype(np.float32)

            segment_idx = len(self.segments)
            self.segments.append((feat_arr, time_arr, symbol))

            max_start = len(split_data) - self.window
            for start_idx in range(max_start + 1):
                self.index_map.append((segment_idx, start_idx))

            usable_symbols += 1

        if self.index_map:
            self.epoch_index_map = list(self.index_map)

        print(f"[{self.data_type.upper()}] Symbols total/usable: {total_symbols}/{usable_symbols}")
        if dropped_symbols:
            print(
                f"[{self.data_type.upper()}] Dropped {len(dropped_symbols)} symbols for insufficient "
                f"split length (<{self.window})"
            )

        if len(self.index_map) == 0:
            raise ValueError(
                "No training windows generated from the multi-stock CSV. "
                "Check lookback/predict windows and split ratios."
            )

    def _split_single_symbol(self, group_data: pd.DataFrame):
        total_length = len(group_data)
        if total_length <= 0:
            return None

        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))

        if self.data_type == 'train':
            return group_data.iloc[:train_end].copy()
        if self.data_type == 'val':
            return group_data.iloc[train_end:val_end].copy()
        return group_data.iloc[val_end:].copy()

    def set_epoch_seed(self, epoch):
        self.current_epoch = epoch
        self.py_rng.seed(self.seed + epoch)

        if self.data_type == 'train':
            self.epoch_index_map = list(self.index_map)
            self.py_rng.shuffle(self.epoch_index_map)
        else:
            self.epoch_index_map = self.index_map

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.n_samples <= 0:
            raise ValueError("No samples available in dataset")

        if not self.epoch_index_map:
            self.epoch_index_map = self.index_map

        segment_idx, start_idx = self.epoch_index_map[idx]
        feat_arr, time_arr, _ = self.segments[segment_idx]

        end_idx = start_idx + self.window
        x = feat_arr[start_idx:end_idx].copy()
        x_stamp = time_arr[start_idx:end_idx].copy()

        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)


class SequentialTrainerV2:
    def __init__(self, config_path: str = None, symbol_column: str = None, data_path_override: str = None):
        self.config = CustomFinetuneConfig(config_path)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                str(self.config.device_id if hasattr(self.config, 'device_id') else 0)
            )
        )

        if data_path_override:
            self.config.data_path = data_path_override

        if symbol_column:
            MultiStockKlineDataset.DEFAULT_SYMBOL_COLUMN = symbol_column

        self.device = self._setup_device()
        self._patch_multi_stock_dataset()

        self.config.print_config_summary()

    def _patch_multi_stock_dataset(self):
        basemodel_module.CustomKlineDataset = MultiStockKlineDataset
        tokenizer_module.CustomKlineDataset = MultiStockKlineDataset
        if self.rank == 0:
            print("Patched tokenizer/basemodel dataset to MultiStockKlineDataset")

    def _setup_device(self):
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")

        if self.rank == 0:
            print(
                f"Using device: {device} "
                f"(rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank})"
            )
        return device

    def _setup_distributed(self):
        if self.world_size > 1 and torch.cuda.is_available():
            backend = os.environ.get("DIST_BACKEND", "nccl").lower()
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            if self.rank == 0:
                print(f"Distributed training initialized: backend={backend}, world_size={self.world_size}")
        else:
            if self.rank == 0:
                print("Distributed training not enabled, using single GPU/CPU training")

    def _check_existing_models(self):
        tokenizer_exists = os.path.exists(self.config.tokenizer_best_model_path)
        basemodel_exists = os.path.exists(self.config.basemodel_best_model_path)

        print(f"Tokenizer model exists: {tokenizer_exists}")
        print(f"Basemodel model exists: {basemodel_exists}")

        return tokenizer_exists, basemodel_exists

    def _create_directories(self):
        os.makedirs(self.config.tokenizer_save_path, exist_ok=True)
        os.makedirs(self.config.basemodel_save_path, exist_ok=True)
        print(f"Created directory: {self.config.tokenizer_save_path}")
        print(f"Created directory: {self.config.basemodel_save_path}")

    def train_tokenizer_phase(self):
        print("\n" + "=" * 60)
        print("Starting Tokenizer Fine-tuning Phase (Multi-stock V2)")
        print("=" * 60)

        tokenizer_exists, _ = self._check_existing_models()
        if tokenizer_exists and self.config.skip_existing:
            print("Tokenizer model already exists, skipping training")
            return True

        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_tokenizer_logging(self.config.exp_name, log_dir, self.rank)

        set_seed(self.config.seed)

        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading pretrained tokenizer...")
            if self.rank == 0:
                print("Loading pretrained tokenizer...")
            tokenizer = KronosTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
        else:
            if self.rank == 0:
                print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture")
            cfg_path = os.path.join(self.config.pretrained_tokenizer_path, 'config.json')
            with open(cfg_path, 'r', encoding='utf-8') as f:
                arch = __import__('json').load(f)
            tokenizer = KronosTokenizer(
                d_in=arch.get('d_in', 6),
                d_model=arch.get('d_model', 256),
                n_heads=arch.get('n_heads', 4),
                ff_dim=arch.get('ff_dim', 512),
                n_enc_layers=arch.get('n_enc_layers', 4),
                n_dec_layers=arch.get('n_dec_layers', 4),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.0),
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                beta=arch.get('beta', 0.05),
                gamma0=arch.get('gamma0', 1.0),
                gamma=arch.get('gamma', 1.1),
                zeta=arch.get('zeta', 0.05),
                group_size=arch.get('group_size', 4),
            )

        tokenizer = tokenizer.to(self.device)
        model_size = sum(p.numel() for p in tokenizer.parameters())
        logger.info(f"Tokenizer parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Tokenizer parameters: {model_size:,}")

        logger.info("=== Training Configuration (V2) ===")
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"Lookback window: {self.config.lookback_window}")
        logger.info(f"Predict window: {self.config.predict_window}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.tokenizer_learning_rate}")
        logger.info(f"Training epochs: {self.config.tokenizer_epochs}")
        logger.info(f"Device: {self.device}")

        logger.info("Starting tokenizer fine-tuning training...")
        if self.rank == 0:
            print("Starting tokenizer fine-tuning training...")

        start_time = time.time()
        best_val_loss = tokenizer_module.train_tokenizer(
            tokenizer,
            self.device,
            self.config,
            self.config.tokenizer_save_path,
            logger,
        )
        training_time = time.time() - start_time

        final_msg = (
            f"Tokenizer training completed! Best validation loss: {best_val_loss:.4f}\n"
            f"Training time: {training_time / 60:.2f} minutes\n"
            f"Model saved to: {self.config.tokenizer_save_path}"
        )
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")

        return True

    def train_basemodel_phase(self):
        print("\n" + "=" * 60)
        print("Starting Basemodel Fine-tuning Phase (Multi-stock V2)")
        print("=" * 60)

        if getattr(self.config, 'pre_trained_tokenizer', True):
            if not os.path.exists(self.config.finetuned_tokenizer_path):
                raise FileNotFoundError(
                    f"Fine-tuned tokenizer does not exist: {self.config.finetuned_tokenizer_path}"
                )

        _, basemodel_exists = self._check_existing_models()
        if basemodel_exists and self.config.skip_existing:
            print("Basemodel model already exists, skipping training")
            return True

        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_basemodel_logging(self.config.exp_name, log_dir, self.rank)

        set_seed(self.config.seed)

        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading fine-tuned tokenizer...")
            if self.rank == 0:
                print("Loading fine-tuned tokenizer...")
            tokenizer = KronosTokenizer.from_pretrained(self.config.finetuned_tokenizer_path)
        else:
            if self.rank == 0:
                print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture")
            cfg_path = os.path.join(self.config.pretrained_tokenizer_path, 'config.json')
            with open(cfg_path, 'r', encoding='utf-8') as f:
                arch = __import__('json').load(f)
            tokenizer = KronosTokenizer(
                d_in=arch.get('d_in', 6),
                d_model=arch.get('d_model', 256),
                n_heads=arch.get('n_heads', 4),
                ff_dim=arch.get('ff_dim', 512),
                n_enc_layers=arch.get('n_enc_layers', 4),
                n_dec_layers=arch.get('n_dec_layers', 4),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.0),
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                beta=arch.get('beta', 0.05),
                gamma0=arch.get('gamma0', 1.0),
                gamma=arch.get('gamma', 1.1),
                zeta=arch.get('zeta', 0.05),
                group_size=arch.get('group_size', 4),
            )

        tokenizer = tokenizer.to(self.device)

        if getattr(self.config, 'pre_trained_predictor', True):
            logger.info("Loading pretrained predictor...")
            if self.rank == 0:
                print("Loading pretrained predictor...")
            model = Kronos.from_pretrained(self.config.pretrained_predictor_path)
        else:
            if self.rank == 0:
                print("pre_trained_predictor=False, randomly initializing Predictor architecture")
            cfg_path = os.path.join(self.config.pretrained_predictor_path, 'config.json')
            with open(cfg_path, 'r', encoding='utf-8') as f:
                arch = __import__('json').load(f)
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
                learn_te=arch.get('learn_te', True),
            )

        model = model.to(self.device)

        model_size = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Model parameters: {model_size:,}")

        logger.info("=== Training Configuration (V2) ===")
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"Lookback window: {self.config.lookback_window}")
        logger.info(f"Predict window: {self.config.predict_window}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.predictor_learning_rate}")
        logger.info(f"Training epochs: {self.config.basemodel_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Tokenizer path: {self.config.finetuned_tokenizer_path}")
        logger.info(f"Pretrained model path: {self.config.pretrained_predictor_path}")

        logger.info("Starting fine-tuning training...")
        if self.rank == 0:
            print("Starting fine-tuning training...")

        start_time = time.time()
        best_val_loss = train_model(
            model,
            tokenizer,
            self.device,
            self.config,
            self.config.basemodel_save_path,
            logger,
        )
        training_time = time.time() - start_time

        final_msg = (
            f"Basemodel training completed! Best validation loss: {best_val_loss:.4f}\n"
            f"Training time: {training_time / 60:.2f} minutes\n"
            f"Model saved to: {self.config.basemodel_save_path}"
        )
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")

        return True

    def run_training(self):
        if self.rank == 0:
            print("Starting Kronos model sequential fine-tuning training (V2 multi-stock)")
            print(f"Experiment name: {self.config.experiment_name}")
            print(f"Experiment description: {self.config.experiment_description}")

        self._setup_distributed()
        self._create_directories()
        self._check_existing_models()

        total_start_time = time.time()

        try:
            if self.config.train_tokenizer:
                success = self.train_tokenizer_phase()
                if not success:
                    print("Tokenizer training failed, terminating training")
                    return False
            else:
                print("Skipping Tokenizer training phase")

            if self.config.train_basemodel:
                success = self.train_basemodel_phase()
                if not success:
                    print("Basemodel training failed, terminating training")
                    return False
            else:
                print("Skipping Basemodel training phase")

            total_time = time.time() - total_start_time

            if self.rank == 0:
                print("\n" + "=" * 60)
                print("Training completed!")
                print("=" * 60)
                print(f"Total training time: {total_time / 60:.2f} minutes")
                print(f"Tokenizer model: {self.config.tokenizer_best_model_path}")
                print(f"Basemodel model: {self.config.basemodel_best_model_path}")
                print("=" * 60)

            return True

        except Exception as e:
            if self.rank == 0:
                print(f"Error occurred during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Kronos Model Sequential Fine-tuning Training V2 (Multi-stock CSV)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path (default: config.yaml)',
    )
    parser.add_argument(
        '--skip-tokenizer',
        action='store_true',
        help='Skip tokenizer training phase',
    )
    parser.add_argument(
        '--skip-basemodel',
        action='store_true',
        help='Skip basemodel training phase',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip training for existing models',
    )
    parser.add_argument(
        '--symbol-column',
        type=str,
        default=None,
        help='Symbol column name in CSV (e.g., symbol, ts_code). If omitted, auto-detect is used.',
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Training data source: CSV path, folder path, glob pattern, or comma/semicolon-separated CSV list.',
    )

    args = parser.parse_args()

    trainer = SequentialTrainerV2(
        args.config,
        symbol_column=args.symbol_column,
        data_path_override=args.data_path,
    )

    if args.skip_tokenizer:
        trainer.config.train_tokenizer = False
    if args.skip_basemodel:
        trainer.config.train_basemodel = False
    if args.skip_existing:
        trainer.config.skip_existing = True

    success = trainer.run_training()

    if success:
        print("Training completed successfully!")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)
    else:
        print("Training failed!")
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(1)


if __name__ == '__main__':
    main()
