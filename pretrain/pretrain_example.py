"""
Example script for Kronos pretraining
Kronos 预训练示例脚本

This script demonstrates how to use the pretrain_kronos.py script
本脚本演示如何使用 pretrain_kronos.py 脚本

Usage / 用法:
    python pretrain_example.py
"""

import os
import subprocess
import sys
from pathlib import Path


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("Checking environment... / 检查环境...")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("ERROR: PyTorch not installed. Please install PyTorch >= 2.0.0")
        return False
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'yaml']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} installed")
        except ImportError:
            print(f"✗ {pkg} NOT installed")
            return False
    
    print()
    return True


def prepare_data_directory():
    """准备数据目录"""
    print("=" * 60)
    print("Preparing data directory... / 准备数据目录...")
    print("=" * 60)
    
    # Example data directory
    data_dir = Path("../download/data")
    
    if not data_dir.exists():
        print(f"Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Count CSV files
    csv_files = list(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    if len(csv_files) == 0:
        print("\nWARNING: No CSV files found!")
        print("Please download stock data first using download/ scripts")
        print("警告：未找到 CSV 文件！")
        print("请先使用 download/ 目录下的脚本下载股票数据")
        return False
    
    # Show first few files
    print("\nFirst 5 files:")
    for f in csv_files[:5]:
        print(f"  - {f.name}")
    
    print()
    return True


def modify_config():
    """修改配置文件"""
    print("=" * 60)
    print("Modifying configuration... / 修改配置...")
    print("=" * 60)
    
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    print(f"Configuration file: {config_path.absolute()}")
    print("\nKey settings to check:")
    print("  - data.data_dir: Should point to your CSV data directory")
    print("  - experiment.pre_trained_tokenizer: Must be false")
    print("  - experiment.pre_trained_predictor: Must be false")
    print("  - training.basemodel_epochs: Recommended 50-100 for pretraining")
    print()
    
    return True


def run_pretraining(config_name="config.yaml", distributed=False, nproc_per_node=1):
    """运行预训练
    
    Args:
        config_name: Configuration file name
        distributed: Whether to use distributed training
        nproc_per_node: Number of GPUs per node
    """
    print("=" * 60)
    print("Starting pretraining... / 开始预训练...")
    print("=" * 60)
    
    if distributed:
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            "pretrain_kronos.py",
            "--config", config_name
        ]
        print(f"Using distributed training with {nproc_per_node} GPUs")
    else:
        cmd = [
            "python",
            "pretrain_kronos.py",
            "--config", config_name
        ]
        print("Using single GPU/CPU training")
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("Pretraining completed successfully! / 预训练成功完成！")
        print("=" * 60)
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print(f"ERROR: Pretraining failed with exit code {e.returncode}")
        print("=" * 60)
        return False
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user / 训练被用户中断")
        print("=" * 60)
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Kronos Pretraining Example / Kronos 预训练示例")
    print("=" * 60 + "\n")
    
    # Step 1: Check environment
    if not check_environment():
        print("\nEnvironment check failed. Please install required dependencies.")
        return
    
    # Step 2: Prepare data
    if not prepare_data_directory():
        print("\nData preparation failed.")
        return
    
    # Step 3: Modify config
    if not modify_config():
        print("\nConfiguration modification failed.")
        return
    
    # Step 4: Run pretraining
    # Choose training mode based on available GPUs
    import torch
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if gpu_count >= 4:
        print(f"\nDetected {gpu_count} GPUs, using distributed training...")
        success = run_pretraining(distributed=True, nproc_per_node=4)
    elif gpu_count > 0:
        print(f"\nDetected {gpu_count} GPU(s), using single GPU training...")
        success = run_pretraining(distributed=False)
    else:
        print("\nNo GPU detected, using CPU training (slow)...")
        print("Recommendation: Use GPU for faster training")
        success = run_pretraining(distributed=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary / 总结")
    print("=" * 60)
    if success:
        print("✓ Pretraining completed successfully")
        print("✓ Model saved to: output/pretrain/<exp_name>/basemodel/best_model")
        print("✓ Tokenizer saved to: output/pretrain/<exp_name>/tokenizer/best_model")
    else:
        print("✗ Pretraining failed or was interrupted")
        print("Please check logs for details")
    
    print("\nNext steps:")
    print("1. Check training logs in output/pretrain/<exp_name>/logs/")
    print("2. Evaluate the pretrained model")
    print("3. Use the model for finetuning on specific stocks")
    print()


if __name__ == "__main__":
    main()
