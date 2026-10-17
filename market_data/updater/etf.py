"""
etf.py — ETF 標竿資料更新器
職責：從 Yahoo Finance 下載 ETF 收盤價，輸出寬表 etf.csv
具備 TWSE 官方備援機制：當 Yahoo Finance 漏抓/缺值 (如 10/15) 時，自動調用證交所官方 API 補齊。
新增 ETF 只需修改 ETF_LIST 清單即可。
"""
import pandas as pd
import yfinance as yf
import requests
import warnings
import logging
import io
import os
from contextlib import redirect_stderr
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

# 設定忽略警告
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ── 設定區 ──
ETF_LIST = ['0050', '00631L'] # 那斯達克：00662 道瓊：00668 標普500：00646
START_DATE = '2015-01-01'


def fetch_twse_stock_month(stock_no, year, month):
    """
    從臺灣證券交易所官方 API 獲取指定月份的每日收盤價 (備援機制)
    """
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={year}{month:02d}01&stockNo={stock_no}&response=json"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        data = r.json()
        res = {}
        if 'data' in data:
            for row in data['data']:
                parts = row[0].split('/')
                # 民國年轉西元年
                dt_str = f"{int(parts[0])+1911}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                try:
                    res[dt_str] = float(row[6].replace(',', ''))
                except Exception:
                    pass
        return res
    except Exception as e:
        print(f"TWSE 備援連線警示 ({stock_no}): {e}")
        return {}


def update_etf():
    """
    下載 ETF_LIST 中所有 ETF 的收盤價，
    優先使用 Yahoo Finance，若遇到缺漏交易日 (如 YF 未收錄)，自動以 TWSE 官方資料補齊。
    輸出為寬表格式 (date, 0050, 00631L, ...) 至 etf.csv。
    """
    file_path = config.MARKET_DATA_DIR / "etf.csv"
    tickers = [f"{t}.TW" for t in ETF_LIST]

    # 設定自訂 Session 避免 Yahoo Finance 429 封鎖
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    try:
        with redirect_stderr(io.StringIO()):
            raw = yf.download(
                tickers,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
                threads=False,
                session=session
            )
    except Exception as e:
        print(f"yfinance 抓取錯誤: {e}")
        raw = pd.DataFrame()

    if raw.empty:
        # 若 yf.download 遭遇 429 / 封鎖，嘗試以 direct chart API 下載
        print("Yahoo Finance 下載為空，切換為直接 API 下載模式...")
        records = {}
        for t in ETF_LIST:
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}.TW?range=10y&interval=1d"
                r = session.get(url, timeout=10)
                if r.status_code == 200:
                    d = r.json()['chart']['result'][0]
                    ts = pd.to_datetime(d['timestamp'], unit='s', utc=True).tz_convert('Asia/Taipei').strftime('%Y-%m-%d')
                    closes = d['indicators']['quote'][0]['close']
                    records[t] = pd.Series(closes, index=ts, name=t)
            except Exception as ex:
                print(f"Direct API 獲取 {t} 失敗: {ex}")
        if records:
            df_close = pd.DataFrame(records).reset_index().rename(columns={'index': 'date'})
        else:
            print("無法自 Yahoo Finance 取得數據。")
            return
    else:
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

    # 確保欄位順序與日期格式
    etf_cols = [t for t in ETF_LIST if t in df_close.columns]
    df_close = df_close[['date'] + etf_cols].copy()
    df_close['date'] = pd.to_datetime(df_close['date']).dt.strftime('%Y-%m-%d')

    # 檢查對齊市場交易日曆 (比對 market.parquet 最近日期，防止 YF 漏抓交易日)
    market_parquet = config.MARKET_DATA_DIR / "market.parquet"
    if market_parquet.exists():
        try:
            m_df = pd.read_parquet(market_parquet, columns=['date'])
            all_trading_days = m_df['date'].drop_duplicates().dt.strftime('%Y-%m-%d').sort_values().tolist()
            # 建立完整交易日曆基底
            base_calendar = pd.DataFrame({'date': [d for d in all_trading_days if d >= START_DATE]})
            df_close = pd.merge(base_calendar, df_close, on='date', how='left')
        except Exception as e:
            print(f"對齊交易日曆提示: {e}")

    # 針對缺漏值 (NaN/None)，自動啟動 TWSE 官方補齊機制
    for col in etf_cols:
        df_close[col] = pd.to_numeric(df_close[col], errors='coerce')
        nan_dates = df_close.loc[df_close[col].isna(), 'date'].tolist()
        
        # 只補齊最近 60 天內的缺漏 (避免歷史過多無謂請求)
        recent_nan_dates = [d for d in nan_dates if d >= (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime('%Y-%m-%d')]
        if recent_nan_dates:
            print(f"偵測到 {col} 在 {len(recent_nan_dates)} 個交易日缺少報價，啟動 TWSE 官方補齊機制...")
            # 按月份分組補齊
            months_needed = set((int(d.split('-')[0]), int(d.split('-')[1])) for d in recent_nan_dates)
            for y, m in months_needed:
                twse_data = fetch_twse_stock_month(col, y, m)
                for d_str in recent_nan_dates:
                    if d_str in twse_data:
                        df_close.loc[df_close['date'] == d_str, col] = twse_data[d_str]

    # 數值處理與 ffill 補齊
    for col in etf_cols:
        df_close[col] = df_close[col].astype(float).round(2)
        df_close[col] = df_close[col].ffill()

    df_close = df_close.dropna(subset=etf_cols).sort_values('date').reset_index(drop=True)

    # 輸出
    df_close.to_csv(file_path, index=False, encoding='utf-8')
    print(f"ETF 資料庫同步成功! 最新日期 {df_close['date'].max()}，共 {len(df_close)} 筆資料")


if __name__ == "__main__":
    update_etf()
