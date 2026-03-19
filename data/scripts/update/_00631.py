import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import logging
import io
from contextlib import redirect_stderr
import config

# 設定忽略警告
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

def yf_download(ticker_code: str, start_d=None, end_d=None, period=None) -> pd.DataFrame:
    df = pd.DataFrame()
    # 正二 00631L 必須有 .TW 
    full_ticker = f"{ticker_code}.TW"
    
    try:
        with redirect_stderr(io.StringIO()):
            df = yf.download(
                full_ticker, 
                start=start_d, 
                end=end_d, 
                period=period,
                auto_adjust=False, 
                progress=False,
                threads=False
            )
        if not df.empty:
            # 處理 yfinance 可能產生的多重索引欄位
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.columns = df.columns.str.lower()
    except Exception as e:
        print(f"yfinance 抓取錯誤: {e}")
    return df

def update_00631(specific_date=None):
    file_path = config.DATABASE_DIR / "00631.csv"

    if not os.path.exists(file_path):
        print("檢測到資料庫不存在...")

    if specific_date:
        now_str = specific_date 
        stock_df = yf_download('00631L', start_d=specific_date, end_d=None)
    else:
        now_str = datetime.now().strftime('%Y-%m-%d')
        stock_df = yf_download('00631L', period='1d')

    if not stock_df.empty:
        current_price = round(float(stock_df['close'].iloc[-1]), 2)
        df = pd.read_csv(file_path)
        
        # 刪除重複日期 (避免重複執行時資料堆疊)
        df['date'] = df['date'].astype(str) # 確保格式一致
        df = df[df['date'] != now_str]
        
        new_row = pd.DataFrame({'date': [now_str], 'price': [current_price]})
        df = pd.concat([df, new_row], ignore_index=True)
        
        # 5. 全域數值修正與存檔
        df['price'] = df['price'].astype(float).round(2)
        df = df.sort_values('date').reset_index(drop=True) # 確保日期排序正確
        
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"正二資料庫更新成功! 日期: {now_str} 現價: {current_price}")
        print(df.tail(3))
    else:
        print(f"無法取得 {now_str} 的價格 (可能非開盤日)。")

if __name__ == "__main__":
    update_00631()