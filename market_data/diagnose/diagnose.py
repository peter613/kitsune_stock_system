import pandas as pd
import numpy as np
from datetime import datetime
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
    
    # 2. 全局日期連續性分析
    all_dates = sorted(df_m['date'].unique())
    weekdays = ['一', '二', '三', '四', '五', '六', '日']
    
    holidays_path = config.MARKET_DIAGNOSE_DIR / 'holidays.csv'
    holiday_dates = set()
    if holidays_path.exists():
        df_h = pd.read_csv(holidays_path)
        df_h['date'] = pd.to_datetime(df_h['date'])
        holiday_dates = set(df_h['date'].dt.date)
    else:
        anomalies.append("警告：尚未產生 holidays.csv，無法進行官方休市比對。")
        
    # 建立絕對標準：從 2021-01-01 到今天的「所有平日」扣除「官方休假日」
    all_bdates = pd.bdate_range(start=start_date, end=end_date)
    expected_trading_dates = pd.DatetimeIndex([d for d in all_bdates if d.date() not in holiday_dates])
    
    # 2. 全市場交易日缺漏分析 (直接比對絕對標準)
    actual_market_dates = pd.DatetimeIndex(df_m['date'].unique())
    market_missing = expected_trading_dates.difference(actual_market_dates)
    
    if len(market_missing) > 0:
        missing_str = ", ".join([f"{d.strftime('%Y-%m-%d')}({weekdays[d.weekday()]})" for d in market_missing])
        anomalies.append(f"全市場交易日缺漏 (應開市但無資料): {missing_str}")

    # 判斷結果
    if not anomalies:
        return True
    else:
        for a in anomalies:
            print(f"{a}")
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
    market_status = diagnose_market()
    revenue_status = diagnose_revenue()
    
    if market_status:
        print("[1] 市場資料 (market.parquet): 完美無缺漏！")
    if revenue_status:
        print("[2] 營收資料 (revenue.csv): 完美無缺漏！")
        
    return market_status and revenue_status

if __name__ == '__main__':
    print(diagnose_market())
