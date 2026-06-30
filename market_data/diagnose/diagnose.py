import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# 將專案根目錄加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# 驗收專用白名單 (老AI與優質上櫃股)
VALIDATION_POOL = [
    '2330', '2454', '2317', '3231', '2382', # 上市
    '5347', '1815', '8299', '3293', '5483'  # 上櫃
]

def diagnose_market():
    """診斷市場資料，若完美無缺漏回傳 True，否則回傳 False"""
    end_date = pd.Timestamp(datetime.now().date())
    start_date = pd.Timestamp('2021-01-01')
    
    market_path = config.MARKET_DATA_DIR / 'market.parquet'
    if not market_path.exists():
        print("警告：找不到 market.parquet")
        return False
        
    df_m = pd.read_parquet(market_path)
    df_m['date'] = pd.to_datetime(df_m['date'])
    df_m = df_m[(df_m['date'] >= start_date) & (df_m['date'] <= end_date)].copy()
    df_m = df_m[df_m['id'].isin(VALIDATION_POOL)].copy()
    
    anomalies = []
    
    # 1. NaN 分析
    nan_cols = ['open', 'high', 'low', 'close', 'volume', 'Foreign_Net', 'Trust_Net']
    total_nans = df_m[nan_cols].isna().sum().sum()
    if total_nans > 0:
        df_m['nan_count'] = df_m[nan_cols].isna().sum(axis=1)
        nan_by_id = df_m.groupby('id')['nan_count'].sum()
        nan_by_id = nan_by_id[nan_by_id > 0].sort_values(ascending=False).head(10)
        top_nans_str = ", ".join([f"{idx} ({val}個)" for idx, val in nan_by_id.items()])
        anomalies.append(f"發現 {total_nans} 個缺失值。前十大缺失值股票：{top_nans_str}")
        
    # 2. 全局日期連續性分析
    all_dates = sorted(df_m['date'].unique())
    weekdays = ['一', '二', '三', '四', '五', '六', '日']
    
    holidays_path = os.path.join(os.path.dirname(__file__), 'holidays.csv')
    holiday_dates = set()
    if os.path.exists(holidays_path):
        df_h = pd.read_csv(holidays_path)
        df_h['date'] = pd.to_datetime(df_h['date'])
        holiday_dates = set(df_h['date'].dt.date)
    else:
        anomalies.append("警告：尚未產生 holidays.csv，無法進行官方休市比對。")
        
    for i in range(len(all_dates) - 1):
        d1, d2 = all_dates[i], all_dates[i+1]
        b_dates = pd.bdate_range(d1, d2)
        missing_dates = b_dates[1:-1]
        
        if len(missing_dates) > 0:
            d1_str = f"{d1.strftime('%Y-%m-%d')}({weekdays[d1.weekday()]})"
            d2_str = f"{d2.strftime('%Y-%m-%d')}({weekdays[d2.weekday()]})"
            
            all_holidays = all(d.date() in holiday_dates for d in missing_dates)
            if not all_holidays:
                real_missing = [d for d in missing_dates if d.date() not in holiday_dates]
                missing_str = ", ".join([f"{d.strftime('%Y-%m-%d')}({weekdays[d.weekday()]})" for d in real_missing])
                anomalies.append(f"大盤資料缺漏: {d1_str} ~ {d2_str} -> 證交所有開市但未抓取: {missing_str}")

    # 3. 個股連續性分析
    first_date_by_id = df_m.groupby('id')['date'].min()
    actual_count_by_id = df_m.groupby('id').size()
    missing_days_list = []
    for stock_id, first_date in first_date_by_id.items():
        expected_days = sum(1 for d in all_dates if d >= first_date)
        actual_days = actual_count_by_id.get(stock_id, 0)
        missing = expected_days - actual_days
        if missing > 5:
            missing_days_list.append(f"{stock_id} (缺 {missing} 天)")
            
    if missing_days_list:
        msg = ", ".join(missing_days_list)
        msg = msg[:120] + "..." if len(msg) > 120 else msg
        anomalies.append(f"個股資料缺漏超過 5 天：{msg}")
        
    # 4. 零值分析
    zero_close = df_m[df_m['close'] == 0]['id'].unique()
    if len(zero_close) > 0:
        anomalies.append(f"收盤價為 0 的股票：{', '.join(map(str, zero_close))}")
        
    zero_vol_by_id = df_m[df_m['volume'] == 0].groupby('id').size()
    zero_vol_flag = zero_vol_by_id[zero_vol_by_id > 10]
    if not zero_vol_flag.empty:
        zv_str = ", ".join([f"{idx} ({val}天)" for idx, val in zero_vol_flag.items()])
        zv_str = zv_str[:120] + "..." if len(zv_str) > 120 else zv_str
        anomalies.append(f"成交量為 0 超過 10 天的股票：{zv_str}")
        
    # 5. 籌碼為 0 分析
    chips_zero = (df_m['Foreign_Net'] == 0) & (df_m['Trust_Net'] == 0)
    chips_zero_rate = chips_zero.groupby(df_m['id']).mean()
    chips_zero_flag = chips_zero_rate[chips_zero_rate > 0.95].index.tolist()
    if chips_zero_flag:
        msg = ", ".join(map(str, chips_zero_flag))
        msg = msg[:120] + "..." if len(msg) > 120 else msg
        anomalies.append(f"籌碼為 0 超過 95% 天數的股票：{msg}")
        
    # 判斷結果
    if not anomalies:
        return True
    else:
        print("[異常] 市場資料發現以下問題：")
        for a in anomalies:
            print(f"  - {a}")
        return False

def diagnose_revenue():
    """診斷營收資料，若完美無缺漏回傳 True，否則回傳 False"""
    end_date = pd.Timestamp(datetime.now().date())
    start_date = pd.Timestamp('2021-01-01')
    
    rev_path = config.MARKET_DATA_DIR / 'revenue.csv'
    if not rev_path.exists():
        print("警告：找不到 revenue.csv")
        return False
        
    df_r = pd.read_csv(rev_path)
    df_r['date'] = pd.to_datetime(df_r['date'])
    df_r['id'] = df_r['id'].astype(str)
    df_r = df_r[(df_r['date'] >= start_date) & (df_r['date'] <= end_date)].copy()
    df_r = df_r[df_r['id'].isin(VALIDATION_POOL)].copy()
    
    anomalies = []
    
    # NaN 分析
    nan_cols_r = ['month_revenue', 'yoy', 'mom']
    nan_cols_r = [c for c in nan_cols_r if c in df_r.columns]
    total_nans_r = df_r[nan_cols_r].isna().sum().sum()
    if total_nans_r > 0:
        anomalies.append(f"發現 {total_nans_r} 個缺失值。")
        
    # 連續性分析
    expected_months = len(df_r['date'].unique())
    actual_months_by_id = df_r.groupby('id').size()
    missing_months = actual_months_by_id[actual_months_by_id < expected_months - 1]
    if not missing_months.empty:
        mm_str = ", ".join([f"{idx} (缺 {expected_months - val} 月)" for idx, val in missing_months.items()])
        mm_str = mm_str[:120] + "..." if len(mm_str) > 120 else mm_str
        anomalies.append(f"缺漏超過 2 個月的股票：{mm_str}")
        
    # 零值分析
    if 'month_revenue' in df_r.columns:
        zero_rev = df_r[df_r['month_revenue'] == 0]['id'].unique()
        if len(zero_rev) > 0:
            anomalies.append(f"營收為 0 的股票：{', '.join(map(str, zero_rev))}")
            
    # 判斷結果
    if not anomalies:
        return True
    else:
        print("[異常] 營收資料發現以下問題：")
        for a in anomalies:
            print(f"  - {a}")
        return False

def diagnose_all():
    print("[系統通知] 資料庫診斷報告 (專屬驗收池模式)")
    market_ok = diagnose_market()
    revenue_ok = diagnose_revenue()
    
    if market_ok:
        print("[1] 市場資料 (market.parquet): 完美無缺漏！")
    if revenue_ok:
        print("[2] 營收資料 (revenue.csv): 完美無缺漏！")
        
    return market_ok and revenue_ok

if __name__ == '__main__':
    diagnose_all()
