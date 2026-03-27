import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模型
from model.kronos import KronosPredictor, Kronos, KronosTokenizer


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


def display_predictions(actual_df, predicted_df):
    """
    显示预测结果，包含新增指标
    """
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    print("\nActual vs Predicted Close Prices:")
    print("-" * 50)
    
    # 确保两个数据框长度一致
    min_len = min(len(actual_df), len(predicted_df))
    
    for i in range(min_len):
        actual_close = actual_df.iloc[i]['close']
        predicted_close = predicted_df.iloc[i]['close']
        diff = abs(actual_close - predicted_close)
        
        print(f"Step {i+1:2d}: Actual={actual_close:8.4f}, Predicted={predicted_close:8.4f}, Diff={diff:6.4f}")
    
    # 计算总体指标
    actual_closes = actual_df['close'].values[:min_len]
    predicted_closes = predicted_df['close'].values[:min_len]
    
    # 使用新的评估器计算所有指标
    from eval.evaluate_torch_optimized import evaluate_predictions as torch_evaluate
    actual_tensor = torch.from_numpy(actual_closes).float()
    pred_tensor = torch.from_numpy(predicted_closes).float()
    
    results = torch_evaluate(actual_tensor, pred_tensor, print_results=False)
    reg_metrics = results['Regression_Metrics']
    
    print("\nOverall Metrics:")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE): {reg_metrics['MAE'].item():.6f}")
    print(f"Mean Squared Error (MSE): {reg_metrics['MSE'].item():.6f}")
    print(f"Root Mean Squared Error (RMSE): {reg_metrics['RMSE'].item():.6f}")
    print(f"Max Deviation: {reg_metrics['Max Deviation'].item():.6f}")
    print(f"Direction Accuracy: {reg_metrics['Direction Accuracy'].item():.6f}")
    print(f"Shape Similarity: {reg_metrics['Shape Similarity'].item():.6f}")
    print(f"Tracking Error: {reg_metrics['Tracking Error'].item():.6f}")
    
    print("\nActual Close Prices:")
    print(actual_closes)
    print("\nPredicted Close Prices:")
    print(predicted_closes)
    
    return {
        'mae': reg_metrics['MAE'].item(),
        'mse': reg_metrics['MSE'].item(),
        'rmse': reg_metrics['RMSE'].item(),
        'max_deviation': reg_metrics['Max Deviation'].item(),
        'direction_accuracy': reg_metrics['Direction Accuracy'].item(),
        'shape_similarity': reg_metrics['Shape Similarity'].item(),
        'tracking_error': reg_metrics['Tracking Error'].item(),
        'actual_prices': actual_closes,
        'predicted_prices': predicted_closes
    }


def predict_with_model(model_path, data_path):
    """
    使用模型进行预测
    """
    print("Loading model and predicting...")
    result = load_and_predict(model_path, data_path)
    
    # 检查是否成功加载模型
    if result[0] is None or result[1] is None:
        print("Failed to load model or make prediction.")
        return None
    
    actual_df, predicted_df = result
    
    # 显示预测结果
    metrics = display_predictions(actual_df, predicted_df)
    
    return metrics


if __name__ == "__main__":
    model_path = r"D:\dev\Kronos\output\002119_5min_20260226"
    data_path = r"D:\dev\Kronos\download\data\600821_5min_20260325.csv"
    
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"Data path does not exist: {data_path}")
        sys.exit(1)
     
    print("Starting prediction with model...")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    
    try:
        results = predict_with_model(model_path, data_path)
        if results is not None:
            print("\nPrediction completed successfully!")
        else:
            print("\nPrediction failed.")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()                                                                                                                                                                                                                                                                                                                                                                                                                              nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn                                    nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn                              n1nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb n'/