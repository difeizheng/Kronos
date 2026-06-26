import akshare as ak
import pandas as pd
from datetime import datetime
import os

def fetch_stock_data(stock_code: str, start_date: str = "20240101", save_dir: str = "./data"):
    """获取A股日线数据并保存为Kronos格式"""
    print(f"获取 {stock_code} 股票数据...")
    
    end_date = datetime.now().strftime("%Y%m%d")
    
    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                            start_date=start_date, end_date=end_date, adjust="qfq")
    
    if df.empty:
        print(f"未获取到数据")
        return None
    
    df = df.rename(columns={
        "日期": "timestamps",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount"
    })
    
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    df = df.sort_values("timestamps").reset_index(drop=True)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{stock_code}_daily.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    print(f"保存成功: {save_path}")
    print(f"数据范围: {df['timestamps'].min()} ~ {df['timestamps'].max()}")
    print(f"数据条数: {len(df)}")
    
    return df

if __name__ == "__main__":
    import sys
    stock = sys.argv[1] if len(sys.argv) > 1 else "000001"
    fetch_stock_data(stock)