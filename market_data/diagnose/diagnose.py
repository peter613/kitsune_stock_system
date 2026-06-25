import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# 將專案根目錄加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# 驗收專用白名單 (老AI與優質上櫃股)
VALIDATION_POOL = [
    '2330', '2454', '2317', '3231', '2382', # 上市
    '5347', '1815', '8299', '3293', '5483'  # 上櫃
]

def diagnose():
    print("[系統通知] 資料庫診斷報告 (專屬驗收池模式)")
    
    end_date = pd.Timestamp(datetime.now().date())
    start_date = pd.Timestamp('2021-01-01')
    
    print(f"診斷區間：{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}\n")
    
    # ---------------------------------------------------------
    # 1. 診斷 market.parquet
    # ---------------------------------------------------------
    market_path = config.MARKET_DATA_DIR / 'market.parquet'
    if not market_path.exists():
        print("警告：找不到 market.parquet\n")
    else:
        print("[1] 市場資料 (market.parquet)")
        df_m = pd.read_parquet(market_path)
        df_m['date'] = pd.to_datetime(df_m['date'])
        
        # 過濾近一年
        df_m = df_m[(df_m['date'] >= start_date) & (df_m['date'] <= end_date)].copy()
        
        # 只保留白名單內的股票
        df_m = df_m[df_m['id'].isin(VALIDATION_POOL)].copy()
        
        # NaN 分析
        nan_cols = ['open', 'high', 'low', 'close', 'volume', 'Foreign_Net', 'Trust_Net']
        total_nans = df_m[nan_cols].isna().sum().sum()
        print("- 缺失值 (NaN) 分析：")
        print(f"  總缺失值數量：{total_nans}")
        
        if total_nans > 0:
            # 找出缺失值最多的股票
            df_m['nan_count'] = df_m[nan_cols].isna().sum(axis=1)
            nan_by_id = df_m.groupby('id')['nan_count'].sum()
            nan_by_id = nan_by_id[nan_by_id > 0].sort_values(ascending=False).head(10)
            if not nan_by_id.empty:
                top_nans_str = ", ".join([f"{idx} ({val}個)" for idx, val in nan_by_id.items()])
                print(f"  前十大缺失值股票：{top_nans_str}")
        
        # 全局日期連續性分析 (尋找大盤休市或缺漏)
        print("- 全局日期連續性分析 (大盤交易日)：")
        all_dates = sorted(df_m['date'].unique())
        total_days = len(all_dates)
        print(f"  總交易日數：{total_days}")
        
        weekdays = ['一', '二', '三', '四', '五', '六', '日']
        
        # 讀取官方休市名單 (與 diagnose.py 同一層目錄)
        holidays_path = os.path.join(os.path.dirname(__file__), 'holidays.csv')
        holiday_dates = set()
        holiday_dict = {}
        if os.path.exists(holidays_path):
            df_h = pd.read_csv(holidays_path)
            df_h['date'] = pd.to_datetime(df_h['date'])
            holiday_dates = set(df_h['date'].dt.date)
            # 建立對應字典 {date: '清明節'}
            for _, row in df_h.iterrows():
                holiday_dict[row['date'].date()] = row['name']
        else:
            print("  ⚠️ 尚未產生 holidays.csv，無法進行官方休市比對，將全部標示為異常。")
        
        abnormal_gaps = []
        holiday_gaps_count = 0
        
        for i in range(len(all_dates) - 1):
            d1, d2 = all_dates[i], all_dates[i+1]
            b_dates = pd.bdate_range(d1, d2)
            missing_dates = b_dates[1:-1]
            missing_bdays = len(missing_dates)
            
            if missing_bdays > 0:
                d1_str = f"{d1.strftime('%Y-%m-%d')}({weekdays[d1.weekday()]})"
                d2_str = f"{d2.strftime('%Y-%m-%d')}({weekdays[d2.weekday()]})"
                
                # 檢查這批缺漏日期，是否全部都在休市名單內
                all_holidays = all(d.date() in holiday_dates for d in missing_dates)
                
                if all_holidays:
                    # 合法休市，不印出來增加雜訊，只紀錄數量
                    holiday_gaps_count += 1
                else:
                    # 有任何一天不在名單內，就是爬蟲異常
                    # 只列出「非休市」的異常日期
                    real_missing = [d for d in missing_dates if d.date() not in holiday_dates]
                    missing_str = ", ".join([f"{d.strftime('%Y-%m-%d')}({weekdays[d.weekday()]})" for d in real_missing])
                    abnormal_gaps.append(f"{d1_str} ~ {d2_str}\n      -> [異常缺漏] 證交所有開市但未抓取: {missing_str}")
                    
        if abnormal_gaps:
            print(f"  [警告] 發現大盤資料缺漏 (已過濾 {holiday_gaps_count} 個合法休市)：")
            for gap in abnormal_gaps:
                print(f"    -> {gap}")
        else:
            print(f"  [完美] 日期連續性：完美 (自動隱藏了 {holiday_gaps_count} 個合法休市斷層，無任何異常缺漏)")
            
        print("\n- 個股連續性分析：")
        # 計算每檔股票的預期天數 (從它的第一筆資料開始算到最後一天，防呆 IPO 新股)
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
            print(f"  缺漏超過 5 天的股票：{msg[:120]}..." if len(msg) > 120 else f"  缺漏超過 5 天的股票：{msg}")
        else:
            print("  缺漏超過 5 天的股票：無")
            
        # 零值分析
        print("- 異常零值分析：")
        zero_close = df_m[df_m['close'] == 0]['id'].unique()
        print(f"  收盤價為 0：{', '.join(map(str, zero_close)) if len(zero_close) > 0 else '無'}")
        
        zero_vol_by_id = df_m[df_m['volume'] == 0].groupby('id').size()
        zero_vol_flag = zero_vol_by_id[zero_vol_by_id > 10]
        if not zero_vol_flag.empty:
            zv_str = ", ".join([f"{idx} ({val}天)" for idx, val in zero_vol_flag.items()])
            print(f"  成交量為 0 超過 10 天：{zv_str[:120]}..." if len(zv_str) > 120 else f"  成交量為 0 超過 10 天：{zv_str}")
        else:
            print("  成交量為 0 超過 10 天：無")
            
        # 籌碼為 0 分析 (超過 95% 天數)
        chips_zero = (df_m['Foreign_Net'] == 0) & (df_m['Trust_Net'] == 0)
        chips_zero_rate = chips_zero.groupby(df_m['id']).mean()
        chips_zero_flag = chips_zero_rate[chips_zero_rate > 0.95].index.tolist()
        if chips_zero_flag:
            msg = ", ".join(map(str, chips_zero_flag))
            print(f"  籌碼為 0 超過 95% 天數：{msg[:120]}..." if len(msg) > 120 else f"  籌碼為 0 超過 95% 天數：{msg}")
        else:
            print("  籌碼為 0 超過 95% 天數：無")
            
        print("")
        
    # ---------------------------------------------------------
    # 2. 診斷 revenue.csv
    # ---------------------------------------------------------
    rev_path = config.MARKET_DATA_DIR / 'revenue.csv'
    if not rev_path.exists():
        print("警告：找不到 revenue.csv\n")
    else:
        print("[2] 營收資料 (revenue.csv)")
        df_r = pd.read_csv(rev_path)
        df_r['date'] = pd.to_datetime(df_r['date'])
        
        # 確保 id 為字串格式以供比對
        df_r['id'] = df_r['id'].astype(str)
        
        # 過濾近一年
        df_r = df_r[(df_r['date'] >= start_date) & (df_r['date'] <= end_date)].copy()
        
        # 只保留白名單內的股票
        df_r = df_r[df_r['id'].isin(VALIDATION_POOL)].copy()
        
        # NaN 分析
        nan_cols_r = ['month_revenue', 'yoy', 'mom']
        nan_cols_r = [c for c in nan_cols_r if c in df_r.columns]
        total_nans_r = df_r[nan_cols_r].isna().sum().sum()
        print("- 缺失值 (NaN) 分析：")
        print(f"  總缺失值數量：{total_nans_r}")
        
        # 連續性分析
        print("- 連續性分析：")
        expected_months = len(df_r['date'].unique())
        print(f"  總營收公布月份數：{expected_months}")
        actual_months_by_id = df_r.groupby('id').size()
        missing_months = actual_months_by_id[actual_months_by_id < expected_months - 1]
        
        if not missing_months.empty:
            mm_str = ", ".join([f"{idx} (缺 {expected_months - val} 月)" for idx, val in missing_months.items()])
            print(f"  缺漏超過 2 個月的股票：{mm_str[:120]}..." if len(mm_str) > 120 else f"  缺漏超過 2 個月的股票：{mm_str}")
        else:
            print("  缺漏超過 2 個月的股票：無")
            
        # 零值分析
        print("- 異常零值分析：")
        if 'month_revenue' in df_r.columns:
            zero_rev = df_r[df_r['month_revenue'] == 0]['id'].unique()
            print(f"  營收為 0：{', '.join(map(str, zero_rev)) if len(zero_rev) > 0 else '無'}")
            
if __name__ == '__main__':
    diagnose()
