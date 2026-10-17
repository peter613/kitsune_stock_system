import os
import sys

# 把專案根目錄加入 sys.path，確保直接執行時能找到 config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import time
import random
import requests
import pandas as pd
from datetime import datetime
import config

def fetch_holidays():
    print("[INFO] 開始抓取 TWSE 官方休市名單 (2021 至今)...")
    
    # 1. 讀取現有的休市名單
    save_path = config.MARKET_DATA_DIR / 'holidays.csv'
    
    existing_years = set()
    existing_df = None
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        existing_years = set(existing_df['date'].dt.year)
        print(f"[INFO] 發現現有休市名單，已涵蓋年份: {sorted(list(existing_years))}")
        
    current_year = datetime.now().year
    
    # 決定要抓取的年份 (2021~今年，如果沒抓過才抓)
    all_target_years = range(2021, current_year + 1)
    years_to_fetch = [y for y in all_target_years if y not in existing_years]
    
    if not years_to_fetch:
        print("[INFO] 所有年份的休市名單皆已最新，無需重新抓取。")
        return
        
    all_data = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.twse.com.tw/zh/trading/schedule.html'
    }
    
    for year in years_to_fetch:
        print(f"  -> 抓取 {year} 年...")
        url = f"https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule?date={year}0101&response=json"
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            if 'data' in data:
                for row in data['data']:
                    if len(row) >= 2:
                        date_str = row[0]
                        name = row[1]
                        desc = row[2] if len(row) > 2 else ""
                        all_data.append({
                            'date': date_str,
                            'name': name,
                            'description': desc
                        })
            else:
                print(f"  -> {year} 年無資料或格式變更。")
                
        except Exception as e:
            print(f"  -> 抓取 {year} 年時發生錯誤: {e}")
            
        time.sleep(random.uniform(2, 4))
        
    # 合併新舊資料
    new_df = pd.DataFrame(all_data)
    if not new_df.empty:
        new_df['date'] = pd.to_datetime(new_df['date'])
        if existing_df is not None and not existing_df.empty:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df
    else:
        final_df = existing_df
        
    if final_df is not None and not final_df.empty:
        final_df = final_df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"[SUCCESS] 休市名單已更新並儲存至 {save_path} (共 {len(final_df)} 筆)")

if __name__ == "__main__":
    fetch_holidays()
