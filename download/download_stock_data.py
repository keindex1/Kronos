# -*- coding: utf-8 -*-
"""
download_stock_data.py

股票数据下载脚本 - 支持多种数据源和格式

用法:
    python download_stock_data.py --symbol 600977 --period 5min --days 30 --output data.csv
    python download_stock_data.py --symbol 000001 --period 1day --start 2024-01-01 --end 2024-12-31
    python download_stock_data.py --symbol AAPL --source yfinance --period 1day --days 365

支持的数据源:
    - akshare: 中国A股、港股数据
    - yfinance: 国际股票数据
    - tushare: 中国股票数据（需要token）

支持的周期:
    - 1min, 5min, 15min, 30min, 60min (分钟级)
    - 1day (日线)
    - 1week (周线)
    - 1month (月线)
"""

import os
import argparse
import pandas as pd
import time
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple


class StockDataDownloader:
    def __init__(self, source: str = "akshare"):
        """
        初始化数据下载器
        
        Args:
            source: 数据源 ("akshare", "yfinance", "tushare")
        """
        self.source = source.lower()
        self.setup_dependencies()
    
    def setup_dependencies(self):
        """设置依赖库"""
        try:
            if self.source == "akshare":
                import akshare as ak
                self.ak = ak
            elif self.source == "yfinance":
                import yfinance as yf
                self.yf = yf
            elif self.source == "baostock":
                import baostock as bs
                self.bs = bs
        except ImportError as e:
            print(f"❌ 缺少依赖库: {e}")
            print(f"请安装: pip install {self.source if self.source != 'baoshare' else 'baostock'}")
            sys.exit(1)
    
    def download_akshare_data(self, symbol: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用akshare下载中国股票数据"""
        try:
            if period.endswith("min"):
                # 分钟级数据
                period_num = period.replace("min", "")
                df = self.ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=period_num,
                    start_date=f"{start_date} 09:30:00",
                    end_date=f"{end_date} 15:00:00",
                    adjust=""
                )
                # 重命名列
                df.rename(columns={
                    "时间": "timestamps",
                    "开盘": "open", 
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                }, inplace=True)
            else:
                # 日线数据
                period_map = {"1day": "daily", "1week": "weekly", "1month": "monthly"}
                period_str = period_map.get(period, "daily")
                
                df = self.ak.stock_zh_a_hist(
                    symbol=symbol,
                    period=period_str,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust=""
                )
                # 重命名列
                df.rename(columns={
                    "日期": "timestamps",
                    "开盘": "open",
                    "收盘": "close", 
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                }, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ akshare数据下载失败: {e}")
            return pd.DataFrame()
    
    def download_yfinance_data(self, symbol: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用yfinance下载国际股票数据"""
        try:
            # yfinance周期映射
            interval_map = {
                "1min": "1m", "5min": "5m", "15min": "15m", 
                "30min": "30m", "60min": "1h",
                "1day": "1d", "1week": "1wk", "1month": "1mo"
            }
            interval = interval_map.get(period, "1d")
            
            ticker = self.yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                return df
            
            # 重命名和重排列
            df.reset_index(inplace=True)
            df.rename(columns={
                # "Datetime": "timestamps" if "Datetime" in df.columns else "timestamps",
                "Date": "timestamps" if "Date" in df.columns else "timestamps", 
                "Open": "open",
                "High": "high", 
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            
            # yfinance没有amount字段，计算近似值
            df["amount"] = df["close"] * df["volume"]

            # 将2025-02-18 00:00:00-05:00 转换为2025-02-18 00:00:00
            df["timestamps"] = df["timestamps"].apply(lambda x: x.replace(tzinfo=None))
            
            return df
            
        except Exception as e:
            print(f"❌ yfinance数据下载失败: {e}")
            return pd.DataFrame()
    
    def download_baostock_data(self, symbol: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        使用baostock下载中国股票数据
        支持周期: 1day, 1week, 1month, 5min, 15min, 30min, 60min
        """
        try:
            # 登录系统
            lg = self.bs.login()
            if lg.error_code != '0':
                print(f"❌ baostock登录失败: {lg.error_msg}")
                return pd.DataFrame()
            
            # 处理代码前缀
            code = symbol
            if not ("." in code):
                if code.startswith("6"):
                    code = f"sh.{code}"
                elif code.startswith("0") or code.startswith("3") or code.startswith("4") or code.startswith("8"):
                    code = f"sz.{code}"
                elif code.startswith("5") or code.startswith("9"): # 基金等可能不完全准确，这里主要针对股票
                     code = f"sh.{code}"
            
            # 处理周期
            # d, w, m, 5, 15, 30, 60
            freq_map = {
                "1day": "d", "daily": "d",
                "1week": "w", "weekly": "w",
                "1month": "m", "monthly": "m",
                "5min": "5", "15min": "15", "30min": "30", "60min": "60"
            }
            
            frequency = freq_map.get(period, "d")
            # baostock不支持1min
            if period == "1min":
                print("⚠️ baostock不支持1min数据，最小支持5min")
                self.bs.logout()
                return pd.DataFrame()
                
            adjustflag = "3" # 默认不复权，为了保持原始价格一致性。如果需要复权可改为 "1"(后复权) 或 "2"(前复权)
            # 注意: 如果需要和 akshare 对齐，akshare 的默认一般也是不复权(adjust="")
            
            fields = "date,open,high,low,close,volume,amount"
            if "min" in period:
                fields = "date,time,open,high,low,close,volume,amount"
            
            rs = self.bs.query_history_k_data_plus(code,
                fields,
                start_date=start_date, end_date=end_date,
                frequency=frequency, adjustflag=adjustflag)
            
            if rs.error_code != '0':
                print(f"❌ baostock查询失败: {rs.error_msg}")
                self.bs.logout()
                return pd.DataFrame()
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
                
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 登出
            self.bs.logout()
            
            if df.empty:
                return df
                
            # 数据类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # 处理时间戳
            # 分钟线 baostock 返回 date(2023-01-01) 和 time(20230101103500000)
            if "time" in df.columns:
                # time 格式是 YYYYMMDDHHMMSSsss
                # 有时候 baostock 分钟线 time 格式可能是 YYYYMMDDHHMMSS
                # 让我们标准化一下
                df['timestamps'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S000') # 常见格式
                # 如果这个转换失败，可能需要更健壮的处理，这里先假设是标准格式
            else:
                df['timestamps'] = pd.to_datetime(df['date'])
                
            return df
            
        except Exception as e:
            print(f"❌ baostock下载异常: {e}")
            try:
                self.bs.logout()
            except:
                pass
            return pd.DataFrame()
    
    def download(self, symbol: str, period: str = "5min", 
                days: Optional[int] = None,
                start_date: Optional[str] = None, 
                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        下载股票数据
        
        Args:
            symbol: 股票代码
            period: 数据周期
            days: 最近N天数据 (与start_date/end_date互斥)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        
        # 处理日期
        if days:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        elif not start_date or not end_date:
            # 默认最近30天
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        print(f"📥 开始下载数据...")
        print(f"   股票代码: {symbol}")
        print(f"   数据周期: {period}")
        print(f"   时间范围: {start_date} ~ {end_date}")
        print(f"   数据源: {self.source}")
        
        # 根据数据源下载
        if self.source == "akshare":
            df = self.download_akshare_data(symbol, period, start_date, end_date)
        elif self.source == "yfinance":
            df = self.download_yfinance_data(symbol, period, start_date, end_date)
        elif self.source == "baostock":
            df = self.download_baostock_data(symbol, period, start_date, end_date)
        else:
            print(f"❌ 不支持的数据源: {self.source}")
            return pd.DataFrame()
        
        if df.empty:
            print(f"❌ 未获取到数据")
            return df
        
        # 数据清理和标准化
        df = self.clean_data(df)
        
        print(f"✅ 下载完成: {len(df)} 条记录")
        if len(df) > 0:
            print(f"   时间范围: {df['timestamps'].min()} ~ {df['timestamps'].max()}")
            print(f"   价格范围: {df['low'].min():.2f} ~ {df['high'].max():.2f}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清理和标准化"""
        if df.empty:
            return df
        
        try:
            # 转换时间格式
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            # 数值列转换
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 移除无效数据
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            # 按时间排序
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            # 确保列顺序
            column_order = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
            available_cols = [col for col in column_order if col in df.columns]
            df = df[available_cols]
            
            return df
            
        except Exception as e:
            print(f"⚠️ 数据清理警告: {e}")
            return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """保存数据到CSV"""
        if df.empty:
            print("❌ 没有数据可保存")
            return
        
        try:
            # 创建输出目录
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
            
            # 保存为CSV
            df.to_csv(filename, index=False)
            print(f"💾 数据已保存: {filename}")
            print(f"   文件大小: {os.path.getsize(filename) / 1024:.1f} KB")
            
            # 显示前几行作为示例
            print("\n📋 数据预览:")
            print(df.head().to_string(index=False))
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="股票数据下载工具")
    
    # 必须参数
    parser.add_argument("--symbol", type=str, required=True, help="股票代码 (如: 600977, 000001, AAPL)")
    
    # 可选参数
    parser.add_argument("--source", type=str, default="akshare", 
                       choices=["akshare", "yfinance", "baostock"],
                       help="数据源 (默认: akshare, 支持 baostock)")
    
    parser.add_argument("--period", type=str, default="5min",
                       choices=["1min", "5min", "15min", "30min", "60min", "1day", "1week", "1month"],
                       help="数据周期 (默认: 5min)")
    
    parser.add_argument("--days", type=int, help="最近N天数据 (与--start/--end互斥)")
    parser.add_argument("--start", type=str, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="结束日期 YYYY-MM-DD")
    
    parser.add_argument("--output", type=str, help="输出文件名 (默认: 自动生成)")
    
    args = parser.parse_args()
    
    # 生成默认输出文件名
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d")
        args.output = f"data/{args.symbol}_{args.period}_{timestamp}.csv"
    
    # 创建下载器并下载数据
    downloader = StockDataDownloader(source=args.source)
    df = downloader.download(
        symbol=args.symbol,
        period=args.period,
        days=args.days,
        start_date=args.start,
        end_date=args.end
    )
    
    # 保存数据
    if not df.empty:
        downloader.save_to_csv(df, args.output)
    else:
        print("❌ 未获取到有效数据")


if __name__ == "__main__":
    main()