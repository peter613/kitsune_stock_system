"""
etf.py — ETF 標竿資料更新器
職責：從 Yahoo Finance 下載 ETF 收盤價，輸出寬表 etf.csv
新增 ETF 只需修改 ETF_LIST 清單即可。
"""
import pandas as pd
import yfinance as yf
import warnings
import logging
import io
from contextlib import redirect_stderr
import config

# 設定忽略警告
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ── 設定區 ──
ETF_LIST = ['0050', '00631L', '00919', '0056', '00878']   # ← 未來只要在這裡加代號
START_DATE = '2015-01-01'


def update_etf():
    """
    增量下載 ETF_LIST 中所有 ETF 的收盤價，
    若有舊資料則只抓最後一日起算，
    輸出為寬表格式 (date, 0050, 00631L, ...) 至 etf.csv。
    """
    import os
    file_path = config.MARKET_DATA_DIR / "etf.csv"
    
    tickers = [f"{t}.TW" for t in ETF_LIST]

    try:
        with redirect_stderr(io.StringIO()):
            raw = yf.download(
                tickers,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
    except Exception as e:
        print(f"yfinance 抓取錯誤: {e}")
        return

    if raw.empty:
        print("無法取得價格 (可能非開盤日)。")
        return

    # 取 Close 層級
    if isinstance(raw.columns, pd.MultiIndex):
        df_close = raw['Close'].copy()
    else:
        df_close = raw[['Close']].copy()
        df_close.columns = [f"{ETF_LIST[0]}.TW"]

    df_close = df_close.reset_index()
    df_close.columns = df_close.columns.str.strip()

    # 重新命名欄位：去掉 .TW 後綴
    rename_map = {f"{t}.TW": t for t in ETF_LIST}
    rename_map['Date'] = 'date'
    df_close = df_close.rename(columns=rename_map)

    # 確保欄位順序
    etf_cols = [t for t in ETF_LIST if t in df_close.columns]
    df_close = df_close[['date'] + etf_cols]

    # 數值處理
    df_close['date'] = pd.to_datetime(df_close['date']).dt.strftime('%Y-%m-%d')
    for col in etf_cols:
        df_close[col] = df_close[col].astype(float).round(2)

    # ffill 補齊缺值
    df_close[etf_cols] = df_close[etf_cols].ffill()

    df_close = df_close.sort_values('date').reset_index(drop=True)

    # 輸出
    df_close.to_csv(file_path, index=False, encoding='utf-8')

    print(f"ETF 資料庫同步成功! 最新日期 {df_close['date'].max()}，共{len(df_close)} 筆資料") 

if __name__ == "__main__":
    update_etf()
