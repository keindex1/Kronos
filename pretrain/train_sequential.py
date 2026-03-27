import os
import sys
import argparse
import subprocess
import torch
import shutil
from pathlib import Path

def run_command(cmd, desc):
    """运行命令并显示状态"""
    print(f"Running {desc}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        print(f"{desc} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {desc}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def load_yaml_config(config_path):
    """加载 YAML 配置文件"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("PyYAML not available")
        return None
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def update_config_path(config_path, key, new_value):
    """
    更新配置文件中的某个路径
    支持嵌套字典中的键更新
    """
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 查找键在嵌套字典中的位置
        def update_nested_dict(d, target_key, value):
            for k, v in d.items():
                if k == target_key:
                    d[k] = value
                    return True
                elif isinstance(v, dict):
                    if update_nested_dict(v, target_key, value):
                        return True
            return False
        
        if update_nested_dict(config, key, new_value):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"Updated {key} to {new_value} in config")
        else:
            print(f"Key {key} not found in config file")
    except ImportError:
        print("PyYAML not available, skipping config update")
    except Exception as e:
        print(f"Error updating config: {e}")

def main():
    parser = argparse.ArgumentParser(description='Sequentially pretrain Kronos tokenizer and basemodel')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Configuration file path (default: config.yaml)')
    parser.add_argument('--skip-tokenizer', action='store_true', 
                        help='Skip tokenizer pretraining (only run basemodel)')
    parser.add_argument('--skip-basemodel', action='store_true', 
                        help='Skip basemodel pretraining (only run tokenizer)')
    parser.add_argument('--tokenizer-output-path', type=str, default=None,
                        help='Custom output path for tokenizer (default: from config)')
    parser.add_argument('--basemodel-output-path', type=str, default=None,
                        help='Custom output path for basemodel (default: from config)')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在，如果不存在尝试在configs子目录中查找
    config_path = Path(args.config)
    if not config_path.exists():
        # 尝试在configs子目录中查找
        alt_config_path = Path("configs") / args.config
        if alt_config_path.exists():
            print(f"Using config from: {alt_config_path}")
            args.config = str(alt_config_path)
        else:
            print(f"Configuration file {args.config} does not exist!")
            print(f"Tried paths: {config_path} and {alt_config_path}")
            return
    
    print(f"Starting sequential pretraining with config: {args.config}")
    
    # 获取当前工作目录
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # 步骤1: 预训练tokenizer
        if not args.skip_tokenizer:
            print("\n" + "="*60)
            print("STEP 1: Pretraining Tokenizer")
            print("="*60)
            
            tokenizer_cmd = [
                sys.executable, "pretrain_tokenizer.py", 
                "--config", args.config
            ]
            
            if not run_command(tokenizer_cmd, "Tokenizer Pretraining"):
                print("Tokenizer pretraining failed. Stopping.")
                return
            
            # 更新配置文件，指向新的tokenizer路径
            if args.tokenizer_output_path:
                update_config_path(args.config, 'pretrained_tokenizer_path', args.tokenizer_output_path)
        
        # 步骤2: 预训练basemodel
        if not args.skip_basemodel:
            print("\n" + "="*60)
            print("STEP 2: Pretraining Basemodel")
            print("="*60)
            
            # 需要确保配置文件中的tokenizer路径是最新的
            # 这里假设tokenizer已保存到config中指定的路径
            basemodel_cmd = [
                sys.executable, "pretrain_kronos.py", 
                "--config", args.config
            ]
            
            if not run_command(basemodel_cmd, "Basemodel Pretraining"):
                print("Basemodel pretraining failed.")
                return
        
        print("\n" + "="*60)
        print("SEQUENTIAL PRETRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)

def update_config_path(config_path, key, new_value):
    """
    更新配置文件中的某个路径
    这是一个简化版本，实际实现可能需要根据配置文件格式调整
    """
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if key in config:
            config[key] = new_value
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            print(f"Updated {key} to {new_value} in config")
        else:
            print(f"Key {key} not found in config file")
    except ImportError:
        print("PyYAML not available, skipping config update")
    except Exception as e:
        print(f"Error updating config: {e}")

if __name__ == "__main__":
    main()