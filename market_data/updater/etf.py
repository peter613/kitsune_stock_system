"""
etf.py — ETF 標竿資料更新器
職責：直連 Yahoo Finance REST API 下載 ETF 還原收盤價，輸出寬表 etf.csv。
特色：純 API 架構、極速（< 1 秒）、0 多餘依賴、具備歷史數據持久化防護。
新增 ETF 只需修改 ETF_LIST 清單即可。
"""
import sys
from pathlib import Path
import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

# ── 設定區 ──
ETF_LIST = ['0050', '00631L']
START_DATE = '2015-01-01'


def fetch_etf_from_api(ticker: str, session: requests.Session) -> pd.Series:
    """
    直連 Yahoo Finance Chart API 獲取指定 ETF 之長週期每日還原收盤價 (adjclose)。
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.TW?range=12y&interval=1d"
    r = session.get(url, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Yahoo API 回應異常 (HTTP {r.status_code})")

    res = r.json().get('chart', {}).get('result')
    if not res:
        raise ValueError(f"Yahoo API 未回傳有效的圖表數據: {ticker}")

    d = res[0]
    timestamps = d.get('timestamp', [])
    indicators = d.get('indicators', {})

    # 優先取用還原收盤價 (adjclose)，若無則取用收盤價 (close)
    adjclose_list = indicators.get('adjclose', [{}])[0].get('adjclose')
    if not adjclose_list:
        adjclose_list = indicators.get('quote', [{}])[0].get('close')

    if not timestamps or not adjclose_list:
        raise ValueError(f"無法從 API 解析出報價序列: {ticker}")

    dates = pd.to_datetime(timestamps, unit='s', utc=True).tz_convert('Asia/Taipei').strftime('%Y-%m-%d')
    s = pd.Series(adjclose_list, index=dates, name=ticker).dropna()
    # 避免同日重複資料，保留最後一筆
    return s.groupby(s.index).last()


def update_etf():
    """
    下載 ETF_LIST 中所有 ETF 的還原收盤價，輸出為寬表格式 (date, 0050, 00631L, ...) 至 etf.csv。
    具備本地歷史資料持久化保護（避免外部 API 縮短週期導致舊資料遺失）。
    """
    file_path = config.MARKET_DATA_DIR / "etf.csv"
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    print("正在透過直連 API 同步 ETF 資料...")
    records = {}
    for t in ETF_LIST:
        try:
            records[t] = fetch_etf_from_api(t, session)
            print(f"[OK] {t} 下載成功，獲取 {len(records[t])} 筆交易日資料")
        except Exception as e:
            print(f"[WARN] 獲取 {t} 失敗: {e}")

    if not records:
        print("[ERROR] 無法自 Yahoo Finance API 取得任何 ETF 數據。")
        return

    df_new = pd.DataFrame(records).reset_index().rename(columns={'index': 'date'})

    # ── 本地資料持久化合併（防護：若本地已有更早資料，不會被 API 覆蓋洗掉）──
    if file_path.exists():
        try:
            df_old = pd.read_csv(file_path, dtype={'date': str})
            # 以 date 為索引進行合併，最新資料優先，舊有歷史保留
            df_merged = pd.concat([df_old, df_new], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=['date'], keep='last')
            df_close = df_merged
        except Exception as e:
            print(f"[WARN] 本地 etf.csv 合併提示: {e}")
            df_close = df_new
    else:
        df_close = df_new

    # 確保欄位順序與日期格式
    etf_cols = [t for t in ETF_LIST if t in df_close.columns]
    df_close = df_close[['date'] + etf_cols].copy()
    df_close['date'] = pd.to_datetime(df_close['date']).dt.strftime('%Y-%m-%d')

    # 檢查對齊市場交易日曆 (比對 market.parquet 最近日期，採聯集確保最新報價不被截斷)
    market_parquet = config.MARKET_DATA_DIR / "market.parquet"
    if market_parquet.exists():
        try:
            m_df = pd.read_parquet(market_parquet, columns=['date'])
            all_trading_days = m_df['date'].drop_duplicates().dt.strftime('%Y-%m-%d').sort_values().tolist()
            calendar_dates = sorted(set([d for d in all_trading_days if d >= START_DATE] + df_close['date'].tolist()))
            base_calendar = pd.DataFrame({'date': calendar_dates})
            df_close = pd.merge(base_calendar, df_close, on='date', how='left')
        except Exception as e:
            print(f"[INFO] 對齊交易日曆提示: {e}")

    # 數值處理與向後補齊 (ffill)
    for col in etf_cols:
        df_close[col] = pd.to_numeric(df_close[col], errors='coerce')
        df_close[col] = df_close[col].ffill().round(2)

    df_close = df_close.dropna(subset=etf_cols).sort_values('date').reset_index(drop=True)

    # 輸出至 etf.csv
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_close.to_csv(file_path, index=False, encoding='utf-8')
    print(f"ETF 資料庫同步成功！涵蓋範圍: {df_close['date'].min()} ~ {df_close['date'].max()}，共 {len(df_close)} 筆交易日。")


if __name__ == "__main__":
    update_etf()
