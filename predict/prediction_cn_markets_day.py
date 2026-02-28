# -*- coding: utf-8 -*-
"""
prediction_cn_markets_day.py

Description:
    Predicts future daily K-line (1D) data for A-share markets using Kronos model and baostock.
    The script automatically downloads the latest historical data, cleans it, and runs model inference.

Usage:
    python prediction_cn_markets_day.py --symbol 002947

Arguments:
    --symbol     Stock code (e.g. 002947, 600000)

Output:
    - Saves the prediction results to ./outputs/pred_<symbol>_data.csv
    - Logs and progress are printed to console

Example:
    bash> python prediction_cn_markets_day.py --symbol 002947
    python3 prediction_cn_markets_day.py --symbol 600000
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
TOKENIZER_PRETRAINED = "../finetune/finetuned/hmd_002947_kline_1day/tokenizer/best_model"
MODEL_PRETRAINED = "../finetune/finetuned/hmd_002947_kline_1day/basemodel/best_model"
DEVICE = "cpu"  # "cuda:0"
MAX_CONTEXT = 512
LOOKBACK = 30
PRED_LEN = 2
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

def load_data(symbol: str, end_date: str = None, lookback: int = 1000) -> pd.DataFrame:
    print(f"📥 Fetching {symbol} daily data from baostock ...")

    # Format stock code for baostock
    bs_code = format_stock_code(symbol)
    
    # Set date range
    if end_date:
        # If end_date is provided (YYYY-MM-DD), keep format as YYYY-MM-DD for baostock
        dt_end = pd.to_datetime(end_date)
        end_date_str = dt_end.strftime("%Y-%m-%d")
        
        # Estimate start date: take 2x lookback to account for weekends/holidays
        dt_start = dt_end - pd.Timedelta(days=lookback * 2) 
        start_date_str = dt_start.strftime("%Y-%m-%d")
        print(f"   Date range: {start_date_str} - {end_date_str} (Estimated from LOOKBACK={lookback})")
    else:
        # Default: Recent data
        dt_end = pd.to_datetime("today")
        end_date_str = dt_end.strftime("%Y-%m-%d")
        dt_start = dt_end - pd.Timedelta(days=lookback * 2)
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
            # Query daily k-line data
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date_str,
                end_date=end_date_str,
                frequency="d",
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

    # Data cleaning
    # Rename 'date' column from baostock to 'timestamps' for consistency
    df.rename(columns={"date": "timestamps"}, inplace=True)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    # Convert numeric columns (baostock returns strings)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"--": None, "": None})
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

    # Keep only the requested lookback amount logic handled in caller or here?
    # Caller asked to "download only LOOKBACK". We fetched ~2*LOOKBACK. 
    # Let's trim strictly to what "prepare_inputs" might need plus a bit buffer?
    # Actually prepare_inputs slices strict LOOKBACK. 
    # But if we did fetch 2*LOOKBACK, keeping all of it helps seeing a bit more history.
    # But user asked "download only LOOKBACK". 
    # Let's keep at most slightly more than lookback or just return what we fetched.
    # Generally better to let prepare_inputs handle precise slicing for model, 
    # but here we ensure we didn't download 20 years of data.
    
    print(f"✅ Data loaded: {len(df)} rows, range: {df['timestamps'].min()} ~ {df['timestamps'].max()}")

    # print("Data Head:")
    # print(df.head())

    return df


def prepare_inputs(df, check_date=None):
    if check_date:
        cutoff = pd.to_datetime(check_date)
        df_input = df[df["timestamps"] <= cutoff]
        if df_input.empty:
            raise ValueError(f"No data found up to check_date {check_date}")
        if len(df_input) < LOOKBACK:
             print(f"⚠️ Warning: Data length ({len(df_input)}) < LOOKBACK ({LOOKBACK}). Using all available data.")
    else:
        df_input = df

    x_df = df_input.iloc[-LOOKBACK:][["open","high","low","close","volume","amount"]]
    x_timestamp = df_input.iloc[-LOOKBACK:]["timestamps"]
    
    # Start prediction from the next day after the input ends
    start_pred = df_input["timestamps"].iloc[-1] + pd.Timedelta(days=1)
    y_timestamp = pd.bdate_range(start=start_pred, periods=PRED_LEN)
    
    return x_df, pd.Series(x_timestamp), pd.Series(y_timestamp), df_input


def apply_price_limits(pred_df, last_close, limit_rate=0.1):
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

    pred_df["timestamps"] = y_timestamp.values

    # Apply ±10% price limit
    last_close = df_input["close"].iloc[-1]
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

    # Merge historical and predicted data
    # Requirement: Save only LOOKBACK + PRED_LEN items. 
    # Current df_input might be larger because load_data fetched 2*LOOKBACK buffer.
    # We take exactly the input window used + predictions.
    
    # df_input slice used for input was: df_input.iloc[-LOOKBACK:]
    input_slice = df_input.iloc[-LOOKBACK:]
    
    df_out = pd.concat([
        input_slice[["timestamps", "open", "high", "low", "close", "volume", "amount"]],
        pred_df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    ]).reset_index(drop=True)

    # Save CSV
    out_file = os.path.join(save_dir, f"pred_{symbol.replace('.', '_')}_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"✅ Prediction completed and saved: {out_file} (Rows: {len(df_out)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos stock prediction script")
    parser.add_argument("--symbol", type=str, default="002947", help="Stock code (e.g., 002947, 600000)")
    parser.add_argument("--check_date", type=str, default=None, help="Cutoff date for prediction input (YYYY-MM-DD)")
    args = parser.parse_args()

    predict_future(
        symbol=args.symbol,
        check_date=args.check_date
    )
