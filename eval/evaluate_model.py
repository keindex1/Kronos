import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模型和评估器
from model.kronos import KronosPredictor, Kronos, KronosTokenizer
from eval.evaluate_torch_optimized import evaluate_predictions


def load_and_predict(model_path, data_path):
    """
    加载模型并进行预测
    """
    print(f"Loading model from: {model_path}")
    
    # 构建tokenizer和model的实际路径
    tokenizer_path = os.path.join(model_path, "tokenizer", "best_model")
    model_path_sub = os.path.join(model_path, "basemodel", "best_model")
    
    # 检查tokenizer路径是否存在
    if not os.path.exists(tokenizer_path):
        print(f"Error: Could not find tokenizer at {tokenizer_path}")
        return None, None
    
    # 检查model路径是否存在
    if not os.path.exists(model_path_sub):
        print(f"Error: Could not find model at {model_path_sub}")
        return None, None
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    print(f"Loading model from: {model_path_sub}")
    
    # 使用from_pretrained加载预训练的tokenizer和model
    try:
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        print(f"Successfully loaded tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    try:
        model = Kronos.from_pretrained(model_path_sub)
        print(f"Successfully loaded model")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # 创建predictor
    predictor = KronosPredictor(model, tokenizer)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 检查列名
    print(f"Available columns: {list(df.columns)}")
    
    # 假设数据包含 OHLCV 列
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 如果列名是大写的，转换为小写
    if 'Open' in df.columns:
        df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }, inplace=True)
    
    # 检查是否包含必需的列
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        # 尝试使用常见的列名变体
        col_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
            'VOLUME': 'volume', 'vol': 'volume'
        }
        for old_col, new_col in col_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
    
    # 再次检查必需的列
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Still missing required columns: {missing_cols}")
        return None, None
    
    # 选择必需的列
    df = df[required_columns].copy()
    
    # 确保数据类型正确
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除任何NaN值
    df.dropna(inplace=True)
    
    print(f"Data shape after cleaning: {df.shape}")
    
    # 进行预测 - 使用最近的数据作为历史观测
    lookback = min(100, len(df) - 10)  # 确保有足够的数据进行预测
    pred_len = 10
    
    if len(df) < lookback + pred_len:
        print(f"Not enough data: need {lookback + pred_len}, got {len(df)}")
        # 尝试减少预测长度
        pred_len = max(5, len(df) - lookback)
        if pred_len <= 0:
            pred_len = len(df) // 2
        print(f"Adjusting pred_len to {pred_len}")
    
    # 使用最后的lookback数据作为输入，预测接下来的pred_len数据
    history_df = df.iloc[-(lookback + pred_len):-pred_len].copy() if pred_len > 0 else df.iloc[:-5].copy()
    actual_future = df.iloc[-pred_len:].copy() if pred_len > 0 else df.iloc[-5:].copy()
    
    print(f"History data shape: {history_df.shape}")
    print(f"Actual future shape: {actual_future.shape}")
    
    # 生成时间戳
    import datetime
    # 创建时间戳序列，使用正确的频率标识符
    base_time = datetime.datetime.now() - datetime.timedelta(minutes=len(df)*5)  # 假设数据是5分钟间隔
    all_times = pd.date_range(start=base_time, periods=len(df), freq='5min')  # 修复频率字符串
    history_times = all_times[:len(history_df)]
    future_times = all_times[len(history_df):len(history_df)+pred_len]
    
    print(f"Making prediction with lookback={len(history_df)}, pred_len={len(actual_future)}")
    
    # 预测未来数据
    try:
        # 确保传入Series而不是DatetimeIndex
        predicted_future = predictor.predict(
            df=history_df.reset_index(drop=True),  # 重置索引
            x_timestamp=pd.Series(history_times),  # 使用Series
            y_timestamp=pd.Series(future_times),  # 使用Series
            pred_len=len(actual_future)
        )
        print(f"Prediction successful, shape: {predicted_future.shape}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    return actual_future, predicted_future


def extract_values_for_evaluation(actual_df, pred_df):
    """
    从预测结果中提取数值用于评估
    """
    # 我们关注收盘价预测的准确性
    actual_close = actual_df['close'].values
    pred_close = pred_df['close'].values
    
    # 确保长度一致
    min_len = min(len(actual_close), len(pred_close))
    actual_close = actual_close[:min_len]
    pred_close = pred_close[:min_len]
    
    return actual_close, pred_close


def evaluate_model_performance(model_path, data_path):
    """
    评估模型性能
    """
    print("Loading model and predicting...")
    result = load_and_predict(model_path, data_path)
    
    # 检查是否成功加载模型
    if result[0] is None or result[1] is None:
        print("Failed to load model or make prediction.")
        return None
    
    actual_df, pred_df = result
    
    print("Extracting values for evaluation...")
    actual_vals, pred_vals = extract_values_for_evaluation(actual_df, pred_df)
    
    # 确保数据有效
    if len(actual_vals) == 0 or len(pred_vals) == 0:
        print("No valid data for evaluation")
        return None
    
    # 使用PyTorch评估器
    print("Evaluating with PyTorch-based evaluator...")
    actual_tensor = torch.from_numpy(actual_vals).float()
    pred_tensor = torch.from_numpy(pred_vals).float()
    torch_results = evaluate_predictions(
        y_true=actual_tensor,
        y_pred=pred_tensor,
        print_results=True
    )
    
    # 输出评估摘要
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    reg_metrics = torch_results['Regression_Metrics']
    
    print("\nRegression Performance:")
    print(f"  MAE:              {reg_metrics['MAE'].item():.6f}")
    print(f"  MSE:              {reg_metrics['MSE'].item():.6f}")
    print(f"  RMSE:             {reg_metrics['RMSE'].item():.6f}")
    print(f"  MAPE:             {reg_metrics['MAPE'].item():.6f}%")
    print(f"  R²:               {reg_metrics['R²'].item():.6f}")
    print(f"  IC:               {reg_metrics['IC'].item():.6f}")
    print(f"  Rank IC:          {reg_metrics['Rank IC'].item():.6f}")
    print(f"  Max Deviation:    {reg_metrics['Max Deviation'].item():.6f}")
    print(f"  Direction Acc.:   {reg_metrics['Direction Accuracy'].item():.6f}")
    print(f"  Shape Similarity: {reg_metrics['Shape Similarity'].item():.6f}")
    print(f"  Tracking Error:   {reg_metrics['Tracking Error'].item():.6f}")
    
    print(f"\nPrediction Quality Assessment:")
    print(f"  Based on {len(actual_vals)} prediction samples")
    
    if reg_metrics['R²'].item() > 0.7:
        print("  Overall fit: Excellent")
    elif reg_metrics['R²'].item() > 0.5:
        print("  Overall fit: Good")
    elif reg_metrics['R²'].item() > 0.3:
        print("  Overall fit: Fair")
    else:
        print("  Overall fit: Poor")
        
    if abs(reg_metrics['IC'].item()) > 0.1:
        print("  Predictive power: Strong")
    elif abs(reg_metrics['IC'].item()) > 0.05:
        print("  Predictive power: Moderate")
    elif abs(reg_metrics['IC'].item()) > 0.02:
        print("  Predictive power: Weak")
    else:
        print("  Predictive power: Very weak")
    
    if reg_metrics['Direction Accuracy'].item() > 0.7:
        print("  Direction prediction: Excellent")
    elif reg_metrics['Direction Accuracy'].item() > 0.6:
        print("  Direction prediction: Good")
    elif reg_metrics['Direction Accuracy'].item() > 0.5:
        print("  Direction prediction: Fair")
    else:
        print("  Direction prediction: Poor")
        
    return torch_results


if __name__ == "__main__":
    model_path = r"D:\dev\Kronos\output\002119_5min_20260226"
    data_path = r"D:\dev\Kronos\download\data\600821_5min_20260325.csv"
    
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"Data path does not exist: {data_path}")
        sys.exit(1)
    
    print("Starting model evaluation...")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    
    try:
        results = evaluate_model_performance(model_path, data_path)
        if results is not None:
            print("\nModel evaluation completed successfully!")
        else:
            print("\nModel evaluation failed.")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()