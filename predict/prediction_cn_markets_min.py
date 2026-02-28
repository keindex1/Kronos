# -*- coding: utf-8 -*-
"""
prediction_cn_markets_min.py

Description:
    Predicts future 5-minute K-line data for A-share markets using Kronos model and baostock.
    The script automatically downloads the latest historical data, cleans it, and runs model inference.

Usage:
    python prediction_cn_markets_min.py --symbol 002947 --check_date 2026-02-10

Arguments:
    --symbol       Stock code (e.g. 002947, 600000)
    --check_date   Cutoff datetime for prediction (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)

Output:
    - Saves the prediction results to ./outputs/pred_<symbol>_min_5_data.csv
    - Logs and progress are printed to console

Example:
    python prediction_cn_markets_min.py --symbol 002947 --check_date "2026-02-10 14:30:00"
    python prediction_cn_markets_min.py --symbol 600000
"""

import os
import argparse
import time
import pandas as pd
import baostock as bs
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)

# Setting
# TOKENIZER_PRETRAINED = "../finetune/finetuned/hmd_002947_kline_5min/tokenizer/best_model"
# MODEL_PRETRAINED = "../finetune/finetuned/hmd_002947_kline_5min/basemodel/best_model"
TOKENIZER_PRETRAINED = "D:/dev/Kronos/output/600740_5min_20260225/tokenizer/best_model"
MODEL_PRETRAINED = "D:/dev/Kronos/output/600740_5min_20260225/basemodel/best_model"
DEVICE = "cpu"  # "cuda:0"
MAX_CONTEXT = 512
LOOKBACK = 512
PRED_LEN = 48  # 预测48个5分钟K线 (4小时)
T = 0.5
TOP_P = 0.9
SAMPLE_COUNT = 1

def format_stock_code(symbol: str) -> str:
    """Convert stock code to baostock format (e.g., 002947 -> sz.002947)"""
    if '.' in symbol:
        return symbol
    if symbol.startswith('6'):
        return f"sh.{symbol}"
    else:
        return f"sz.{symbol}"

def load_data(symbol: str, end_date: str = None, lookback: int = 600) -> pd.DataFrame:
    """
    Load 5-minute K-line data from baostock
    
    Args:
        symbol: Stock code
        end_date: End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        lookback: Number of 5-min bars to fetch (default 600 = about 12.5 trading days)
    """
    print(f"📥 Fetching {symbol} 5-minute data from baostock ...")

    # Format stock code for baostock
    bs_code = format_stock_code(symbol)
    
    # Set date range
    if end_date:
        dt_end = pd.to_datetime(end_date)
        end_date_str = dt_end.strftime("%Y-%m-%d")
        
        # Estimate start date: fetch more days to ensure we get enough 5-min bars
        # Trading hours: 4h/day * 12 bars/h = 48 bars/day
        # So we need lookback/48 days, add buffer
        days_needed = int(lookback / 48 * 3)  # 3x buffer for weekends/holidays
        dt_start = dt_end - pd.Timedelta(days=days_needed)
        start_date_str = dt_start.strftime("%Y-%m-%d")
        print(f"   Date range: {start_date_str} - {end_date_str} (Estimated from LOOKBACK={lookback})")
    else:
        # Default: Recent data
        dt_end = pd.to_datetime("today")
        end_date_str = dt_end.strftime("%Y-%m-%d")
        days_needed = int(lookback / 48 * 3)
        dt_start = dt_end - pd.Timedelta(days=days_needed)
        start_date_str = dt_start.strftime("%Y-%m-%d")
        print(f"   Date range: {start_date_str} - {end_date_str} (Recent data)")

    # Login to baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ Baostock login failed: {lg.error_msg}")
        sys.exit(1)
    
    max_retries = 3
    df = None

    # Retry mechanism
    for attempt in range(1, max_retries + 1):
        try:
            # Query 5-minute k-line data
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,time,open,high,low,close,volume,amount",
                start_date=start_date_str,
                end_date=end_date_str,
                frequency="5",  # 5-minute frequency
                adjustflag="3"  # 不复权
            )
            
            if rs.error_code == '0':
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                df = pd.DataFrame(data_list, columns=rs.fields)
                if not df.empty:
                    break
            else:
                print(f"⚠️ Attempt {attempt}/{max_retries} failed: {rs.error_msg}")
        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
        time.sleep(1.5)

    # Logout from baostock
    bs.logout()

    # If still empty after retries
    if df is None or df.empty:
        print(f"❌ Failed to fetch data for {symbol} after {max_retries} attempts. Exiting.")
        sys.exit(1)

    # Data cleaning - parse time field (format: YYYYMMDDHHMMSSmmm)
    # Baostock returns time in format like "20260109093500000"
    df["date"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
    df = df.drop(columns=["time"])
    df = df.sort_values("date").reset_index(drop=True)

    # Convert numeric columns (baostock returns strings)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"--": None, "": None, "": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix invalid open values
    open_bad = (df["open"] == 0) | (df["open"].isna())
    if open_bad.any():
        print(f"⚠️  Fixed {open_bad.sum()} invalid open values.")
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"].fillna(df["close"], inplace=True)

    # Fix missing amount
    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    print(f"✅ Data loaded: {len(df)} rows, range: {df['date'].min()} ~ {df['date'].max()}")

    return df


def prepare_inputs(df, check_date=None):
    """
    Prepare input data for prediction
    
    Args:
        df: DataFrame with historical data
        check_date: Cutoff datetime (if provided, use data up to this point)
    """
    if check_date:
        cutoff = pd.to_datetime(check_date)
        df_input = df[df["date"] <= cutoff]
        if df_input.empty:
            raise ValueError(f"No data found up to check_date {check_date}")
        if len(df_input) < LOOKBACK:
             print(f"⚠️ Warning: Data length ({len(df_input)}) < LOOKBACK ({LOOKBACK}). Using all available data.")
    else:
        df_input = df

    x_df = df_input.iloc[-LOOKBACK:][["open","high","low","close","volume","amount"]]
    x_timestamp = df_input.iloc[-LOOKBACK:]["date"]
    
    # Generate future timestamps (5-minute intervals during trading hours)
    # Trading hours: 9:30-11:30 (24 bars) and 13:00-15:00 (24 bars) = 48 bars/day
    last_time = df_input["date"].iloc[-1]
    y_timestamp = generate_trading_timestamps(last_time, PRED_LEN)
    
    return x_df, pd.Series(x_timestamp.values), pd.Series(y_timestamp), df_input


def generate_trading_timestamps(start_time, num_periods):
    """
    Generate trading hour timestamps for 5-minute K-lines
    Trading hours: 9:30-11:30, 13:00-15:00
    """
    timestamps = []
    current = start_time
    
    while len(timestamps) < num_periods:
        current = current + pd.Timedelta(minutes=5)
        
        # Skip weekends
        if current.weekday() >= 5:
            # Jump to next Monday 9:30
            days_to_monday = 7 - current.weekday()
            current = current + pd.Timedelta(days=days_to_monday)
            current = current.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Adjust to trading hours
        if current.time() < pd.Timestamp("09:30").time():
            current = current.replace(hour=9, minute=30, second=0, microsecond=0)
        elif pd.Timestamp("11:30").time() < current.time() < pd.Timestamp("13:00").time():
            current = current.replace(hour=13, minute=0, second=0, microsecond=0)
        elif current.time() >= pd.Timestamp("15:00").time():
            # Jump to next trading day 9:30
            current = current + pd.Timedelta(days=1)
            current = current.replace(hour=9, minute=30, second=0, microsecond=0)
            # Skip weekends again
            if current.weekday() >= 5:
                days_to_monday = 7 - current.weekday()
                current = current + pd.Timedelta(days=days_to_monday)
        
        timestamps.append(current)
    
    return timestamps


def apply_price_limits(pred_df, last_close, limit_rate=0.1):
    """Apply daily price limits (±10% for most stocks)"""
    print(f"🔒 Applying ±{limit_rate*100:.0f}% price limit ...")

    # Ensure integer index
    pred_df = pred_df.reset_index(drop=True)

    # Ensure float64 dtype for safe assignment
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")

    for i in range(len(pred_df)):
        limit_up = last_close * (1 + limit_rate)
        limit_down = last_close * (1 - limit_rate)

        for col in cols:
            value = pred_df.at[i, col]
            if pd.notna(value):
                clipped = max(min(value, limit_up), limit_down)
                pred_df.at[i, col] = float(clipped)

        last_close = float(pred_df.at[i, "close"])  # ensure float type

    return pred_df


def predict_future(symbol, check_date=None):
    print(f"🚀 Loading Kronos tokenizer:{TOKENIZER_PRETRAINED} model:{MODEL_PRETRAINED} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
    model = Kronos.from_pretrained(MODEL_PRETRAINED)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    df = load_data(symbol, end_date=check_date, lookback=LOOKBACK)
    x_df, x_timestamp, y_timestamp, df_input = prepare_inputs(df, check_date=check_date)

    print("🔮 Generating predictions ...")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
    )

    # Assign timestamps to predicted data (use .values to avoid index mismatch)
    pred_df["date"] = list(y_timestamp) if isinstance(y_timestamp, pd.Series) else y_timestamp

    # Apply ±10% price limit
    last_close = df_input["close"].iloc[-1]
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

    # Merge historical and predicted data
    input_slice = df_input.iloc[-LOOKBACK:]
    
    df_out = pd.concat([
        input_slice[["date", "open", "high", "low", "close", "volume", "amount"]],
        pred_df[["date", "open", "high", "low", "close", "volume", "amount"]]
    ]).reset_index(drop=True)

    # Save CSV
    out_file = os.path.join(save_dir, f"pred_{symbol.replace('.', '_')}_min_5_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"✅ Prediction completed and saved: {out_file} (Rows: {len(df_out)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos 5-min stock prediction script")
    parser.add_argument("--symbol", type=str, default="002947", help="Stock code (e.g., 002947, 600000)")
    parser.add_argument("--check_date", type=str, default=None, help="Cutoff datetime for prediction input (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    args = parser.parse_args()

    predict_future(
        symbol=args.symbol,
        check_date=args.check_date
    )
