import akshare as ak
import json

def get_main_board_stocks():
    """
    获取所有主板股票的代码和名称，返回字典格式并导出为JSON文件
    """
    try:
        # 获取A股全市场股票列表
        stock_info_df = ak.stock_info_a_code_name()
        
        # 定义主板股票代码前缀（沪市主板：600/601/603/605/609；深市主板：000/001）
        main_board_prefixes = ('600', '601', '603', '605', '609', '000', '001')
        
        # 筛选主板股票，并转换为指定的字典格式
        main_board_stocks = {}
        for _, row in stock_info_df.iterrows():
            stock_code = row['code']
            stock_name = row['name']
            # 判断是否为主板股票
            if stock_code.startswith(main_board_prefixes):
                main_board_stocks[stock_code] = stock_name
        
        # 导出为JSON文件（方便查看和后续使用）
        with open('main_board_stocks.json', 'w', encoding='utf-8') as f:
            json.dump(main_board_stocks, f, ensure_ascii=False, indent=4)
        
        print(f"成功获取并导出 {len(main_board_stocks)} 只主板股票数据")
        return main_board_stocks
    
    except Exception as e:
        print(f"获取股票数据失败：{e}")
        return {}

# 执行函数并查看结果
if __name__ == "__main__":
    # 获取主板股票字典
    main_board_stocks_dict = get_main_board_stocks()
    
    # 打印前5个示例数据，验证格式
    print("\n示例数据：")
    for code, name in list(main_board_stocks_dict.items())[:5]:
        print(f'"{code}": "{name}"')