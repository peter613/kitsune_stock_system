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
                auto_adjust=True, 
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

def update_00631(specific_date="2025-01-01"):
    file_path = config.DATABASE_DIR / "00631.csv"

    # 1. 抓取全量全新資料
    stock_df = yf_download('00631L', start_d=specific_date, end_d=None)

    if not stock_df.empty:

        # 直接將全新的 stock_df 取代舊有的 df 邏輯
        df = stock_df[['date', 'close']].copy()

        # --- 新增手動修正邏輯 ---(之後yf正常後再刪掉)
        # 針對 2026/03/23 之前的歷史價格進行 1:22 的分割還原
        #df.loc[df['date'] < '2026-03-23', 'close'] = df['close'] / 22
        # -----------------------
        
        # 4. 全域數值修正與排序
        df['close'] = df['close'].astype(float).round(2)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('date').reset_index(drop=True)
        
        # 5. 強制全量覆寫存檔 (Overwrite)
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        current_price = df['close'].iloc[-1]
        print(f"正二資料庫同步成功! 現價: {current_price}")
        print(df.tail(2))
    else:
        print("無法取得價格 (可能非開盤日)。")

if __name__ == "__main__":
    update_00631()