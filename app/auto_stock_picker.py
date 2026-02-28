#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动选股脚本（基于 predict/prediction_cn_markets_min.py）

用法示例:
    python app/auto_stock_picker.py --check_date 2026-02-27 --T 0.5 --samples 5 --top_n 20

说明:
 - 遍历主板（上海 600/601/603 开头，深圳 000 开头）A 股代码
 - 使用 Kronos 模型进行多次采样预测，计算未来预测期内的最大涨幅
 - 输出涨幅最高的前 N 支股票
"""
import os
import sys
import time
import argparse
import traceback
from importlib import util

import baostock as bs
import pandas as pd


def load_prediction_module():
    # 动态加载 predict 脚本模块以复用其函数与配置
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mod_path = os.path.join(root, "predict", "prediction_cn_markets_min.py")
    spec = util.spec_from_file_location("pred_min", mod_path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_main_board_codes():
    """从 baostock 获取所有股票，并筛选出主板代码（600/601/603/000 开头）。"""
    codes = []
    try:
        from download import stock_config
        codes = list(stock_config.A_STOCKS.keys())
    except Exception:
        codes = []
    codes = sorted(list(set(codes)))
    return codes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auto stock picker based on Kronos 5-min predictor")
    parser.add_argument("--check_date", type=str, default=None, help="Cutoff datetime for prediction input (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--T", type=float, default=0.5, help="Sampling temperature (default 0.5)")
    parser.add_argument("--samples", type=int, default=5, help="采样次数/sample_count (default 5)")
    parser.add_argument("--top_n", type=int, default=20, help="打印前 N 支股票 (default 20)")
    parser.add_argument("--limit", type=int, default=0, help="可选: 只处理前 M 支代码（0 表示全部）")
    args = parser.parse_args(argv)

    print("🔎 加载预测模块和模型（只加载一次）...")
    pred = load_prediction_module()

    # Load tokenizer & model once
    tokenizer = pred.KronosTokenizer.from_pretrained(pred.TOKENIZER_PRETRAINED)
    model = pred.Kronos.from_pretrained(pred.MODEL_PRETRAINED)
    predictor = pred.KronosPredictor(model, tokenizer, device=pred.DEVICE, max_context=pred.MAX_CONTEXT)

    print("📡 获取主板 A 股代码列表（通过 baostock）...")
    codes = fetch_main_board_codes()
    if args.limit > 0:
        codes = codes[: args.limit]

    print(f"🧾 共找到 {len(codes)} 支主板候选股票，开始遍历预测（T={args.T}, samples={args.samples}）...")

    results = []
    for i, code in enumerate(codes, 1):
        try:
            print(f"[{i}/{len(codes)}] 处理 {code} ...")
            df = pred.load_data(code, end_date=args.check_date, lookback=pred.LOOKBACK)

            x_df, x_ts, y_ts, df_input = pred.prepare_inputs(df, check_date=args.check_date)

            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=pred.PRED_LEN,
                T=args.T,
                top_p=pred.TOP_P,
                sample_count=args.samples,
            )

            # 预测期内最大收盘价相对于当前收盘价的涨幅
            last_close = float(df_input["close"].iloc[-1])
            max_pred_close = float(pred_df["close"].max())
            pct = (max_pred_close - last_close) / last_close * 100.0

            results.append({"code": code, "last_close": last_close, "max_pred_close": max_pred_close, "pred_pct": pct})
        except Exception as e:
            print(f"⚠️ 跳过 {code}，发生错误: {e}")
            traceback.print_exc()
        # 小间隔，避免请求过快
        time.sleep(0.5)

    if not results:
        print("❌ 未产生有效预测结果。")
        return

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("pred_pct", ascending=False).reset_index(drop=True)

    top_n = min(args.top_n, len(df_res))
    print("\n🔝 预测涨幅排名（前 {}）:" .format(top_n))
    for idx in range(top_n):
        r = df_res.iloc[idx]
        print(f"{idx+1:2d}. {r['code']}  预测涨幅(max)={r['pred_pct']:.2f}%  last_close={r['last_close']:.3f}  max_pred={r['max_pred_close']:.3f}")


if __name__ == '__main__':
    main()
