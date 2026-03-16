import akshare as ak
import pandas as pd

def get_top_100_dragon_tiger_stocks():
    """
    获取龙虎榜中前100支唯一的股票代码和名称（兼容所有akshare版本）
    返回格式：字典 {股票代码: 股票名称}
    """
    try:
        # 改用东方财富网龙虎榜接口（兼容性更好）
        # stock_dragon_tiger_list_em 是akshare长期维护的核心接口
        dragon_tiger_df = ak.stock_dragon_tiger_list_em()
        
        # 2. 数据预处理：清理空值、去重
        # 东方财富接口的字段名是"代码"和"名称"，和原代码兼容
        dragon_tiger_df = dragon_tiger_df.dropna(subset=['代码', '名称'])
        # 按股票代码去重，保留第一次出现的记录
        unique_stocks_df = dragon_tiger_df.drop_duplicates(subset=['代码'], keep='first')
        
        # 3. 提取前100支股票（如果不足100支则取全部）
        top_100_stocks = unique_stocks_df.head(100)
        
        # 4. 转换为字典格式 {代码: 名称}
        stock_dict = dict(zip(top_100_stocks['代码'], top_100_stocks['名称']))
        
        # 5. 导出为CSV文件（方便查看和使用）
        top_100_stocks[['代码', '名称']].to_csv(
            'top_100_dragon_tiger_stocks.csv', 
            index=False, 
            encoding='utf-8-sig'
        )
        
        # 打印结果示例
        print(f"成功获取 {len(stock_dict)} 支龙虎榜股票（目标100支）")
        print("\n前10支股票示例：")
        for code, name in list(stock_dict.items())[:10]:
            print(f'"{code}": "{name}"')
        
        return stock_dict
    
    except Exception as e:
        print(f"获取龙虎榜数据失败：{str(e)}")
        # 打印可用的龙虎榜接口，方便排查
        print("\n当前akshare支持的龙虎榜接口：")
        for attr in dir(ak):
            if "dragon_tiger" in attr:
                print(f"- ak.{attr}")
        return {}

# 执行函数
if __name__ == "__main__":
    dragon_tiger_stocks = get_top_100_dragon_tiger_stocks()