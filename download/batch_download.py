# -*- coding: utf-8 -*-
"""
batch_download.py

批量下载股票数据脚本

用法:
    python batch_download.py --stocks 600000,000001,600519 --period 5min --days 30
    python batch_download.py --config a_stocks --period 1day --days 365
"""

import argparse
import sys
import os
from datetime import datetime
import sys
sys.path.append("../")
from download_stock_data import StockDataDownloader
from stock_config import A_STOCKS, US_STOCKS, HK_STOCKS


def batch_download(stock_list, period="5min", days=30, source="akshare", output_dir="data"):
    """批量下载股票数据"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建下载器
    downloader = StockDataDownloader(source=source)
    
    print(f"🚀 开始批量下载 {len(stock_list)} 只股票")
    print(f"   数据周期: {period}")
    print(f"   天数: {days}")
    print(f"   输出目录: {output_dir}")
    print("=" * 60)
    
    success_count = 0
    failed_list = []
    
    for i, symbol in enumerate(stock_list, 1):
        print(f"\n[{i}/{len(stock_list)}] 处理: {symbol}")
        
        try:
            # 下载数据
            df = downloader.download(
                symbol=symbol,
                period=period,
                days=days
            )
            
            if not df.empty:
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d")
                filename = f"{output_dir}/{symbol}_{period}_{timestamp}.csv"
                
                # 保存数据
                downloader.save_to_csv(df, filename)
                success_count += 1
                
            else:
                print(f"❌ {symbol}: 无数据")
                failed_list.append(symbol)
                
        except Exception as e:
            print(f"❌ {symbol}: 下载失败 - {e}")
            failed_list.append(symbol)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print(f"📊 下载完成统计:")
    print(f"   成功: {success_count}/{len(stock_list)}")
    print(f"   失败: {len(failed_list)}")
    
    if failed_list:
        print(f"   失败列表: {', '.join(failed_list)}")


def main():
    parser = argparse.ArgumentParser(description="批量股票数据下载工具")
    
    # 股票选择 (二选一)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stocks", type=str, help="股票代码列表 (逗号分隔)")
    group.add_argument("--config", type=str, choices=["a_stocks", "us_stocks", "hk_stocks"],
                      help="预定义股票组合")
    
    # 其他参数
    parser.add_argument("--period", type=str, default="5min",
                       choices=["1min", "5min", "15min", "30min", "60min", "1day", "1week", "1month"],
                       help="数据周期 (默认: 5min)")
    
    parser.add_argument("--days", type=int, default=30, help="最近N天数据 (默认: 30)")
    
    parser.add_argument("--source", type=str, default="akshare",
                       choices=["akshare", "yfinance", "tushare"],
                       help="数据源 (默认: akshare)")
    
    parser.add_argument("--output-dir", type=str, default="data", help="输出目录 (默认: data)")
    
    args = parser.parse_args()
    
    # 确定股票列表
    if args.stocks:
        stock_list = [s.strip() for s in args.stocks.split(",") if s.strip()]
    elif args.config:
        config_map = {
            "a_stocks": list(A_STOCKS.keys()),
            "us_stocks": list(US_STOCKS.keys()), 
            "hk_stocks": list(HK_STOCKS.keys())
        }
        stock_list = config_map[args.config]
    
    # 执行批量下载
    batch_download(
        stock_list=stock_list,
        period=args.period,
        days=args.days,
        source=args.source,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()