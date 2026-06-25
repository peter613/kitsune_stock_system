import pandas as pd
import numpy as np
from datetime import datetime
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os
import time
import config
import random
session = requests.Session()
# 設定檔案路徑
FILE_NAME = config.MARKET_DATA_DIR/'market.parquet'

HEADERS_TWSE= {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://www.twse.com.tw/',
        'Connection': 'keep-alive'
    }
HEADERS_TPEX = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://www.tpex.org.tw/',
        'Connection': 'keep-alive'
    }

# ==========================================
# 共通共用函式
# ==========================================
def retry_crwler(url, headers, timeout=10, max_retries=2, sleep_range=(2, 5)):
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(*sleep_range))
            res = session.get(url, headers=headers, timeout=timeout, verify=False)
            res.raise_for_status() 
            return res
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"連線重試 {max_retries} 次均失敗: {e}")
                return None
            print(f"  > 遭遇連線異常，5 秒後進行重試 ({attempt+2}/{max_retries})...")
            time.sleep(5)
    return None

# ==========================================
# 上市 (TWSE)
# ==========================================
def basic_twse(date_str):
    print(f"[上市] 下載收盤行情...")
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date_str}&type=ALLBUT0999&response=json"

    res = retry_crwler(url, HEADERS_TWSE, sleep_range=(2, 5))
    if not res: return None

    try:
        json_data = res.json()
    except Exception as e:
        print(f"[解析錯誤]: {e}")
        return None

    if not json_data.get('tables'):
        return None

    try:
        target = next(t for t in json_data['tables'] if '每日收盤行情' in t.get('title', ''))
    except StopIteration:
        return None

    df = pd.DataFrame(target['data'], columns=target['fields'])
    df = df.replace({r'<[^>]+>': '', ',': '', '--': ''}, regex=True)

    # [0]代號 [1]名稱 [2]成交股數 [4]成交金額 [5]開盤 [6]最高 [7]最低 [8]收盤
    df = df.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8]].copy()
    df.columns = ['id', 'name', 'volume', 'amount', 'open', 'high', 'low', 'close']

    df['id'] = df['id'].astype(str).str.strip()
    df = df[~df['id'].str.startswith('00')]
    df = df[df['id'].str.len() == 4]

    df[['volume', 'amount', 'open', 'high', 'low', 'close']] = df[['volume', 'amount', 'open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')

    df['volume'] /= 1000
    df['amount'] /= 1000
    df.insert(0, 'date', date_str)

    return df

def chips_twse(date_str):
    print(f"[上市] 下載外資/投信籌碼...")
    url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALL&response=json"

    res = retry_crwler(url, HEADERS_TWSE, sleep_range=(2, 5))
    if not res: return None

    try:
        json_data = res.json()
    except Exception as e:
        print(f"[解析錯誤] : {e}")
        return None

    if not json_data.get('data'):
        return None

    df = pd.DataFrame(json_data['data'], columns=json_data['fields'])
    df = df.replace({r'<[^>]+>': '', ',': ''}, regex=True)

    df = df.iloc[:, [0, 1, 4, 10]].copy()
    df.columns = ['id', 'name', 'Foreign_Net', 'Trust_Net']

    df['id'] = df['id'].astype(str).str.strip()
    df = df[~df['id'].str.startswith('00')]
    df = df[df['id'].str.len() == 4]

    if df.empty:
        return None

    df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].apply(pd.to_numeric, errors='coerce')
    df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].fillna(0)

    df['Foreign_Net'] /= 1000
    df['Trust_Net'] /= 1000
    return df

# ==========================================
# 上櫃 (tpex)
# ==========================================
def basic_tpex(date_str):
    print(f"[上櫃] 下載收盤行情...")
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    url = f'https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?date={year}%2F{month}%2F{day}&type=EW&id=&response=json&order=0&sort=asc'
    res = retry_crwler(url, HEADERS_TPEX, sleep_range=(3, 7))
    if not res: return None

    try:
        json_data = res.json()
    except Exception as e:
        print(f"[解析錯誤] : {e}")
        return None

    if not json_data.get('tables') or not json_data['tables'][0].get('data'):
        return None

    table = json_data['tables'][0]
    df = pd.DataFrame(table['data'], columns=table['fields'])
    df = df.replace({r'<[^>]+>': '', ',': ''}, regex=True)

    # [0]代號 [1]名稱 [2]收盤 [4]開盤 [5]最高 [6]最低 [7]成交股數 [8]成交金額
    df = df.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8]].copy()
    df.columns = ['id', 'name', 'close', 'open', 'high', 'low', 'volume', 'amount']

    df['id'] = df['id'].astype(str).str.strip()
    df = df[~df['id'].str.startswith('00')]
    df = df[df['id'].str.len() == 4]

    df[['close', 'open', 'high', 'low', 'volume', 'amount']] = df[['close', 'open', 'high', 'low', 'volume', 'amount']].apply(pd.to_numeric, errors='coerce')
    df[['close', 'open', 'high', 'low', 'volume', 'amount']] = df[['close', 'open', 'high', 'low', 'volume', 'amount']].fillna(0)

    df['volume'] /= 1000
    df['amount'] /= 1000
    df.insert(0, 'date', date_str)

    return df

def chips_tpex(date_str):
    print(f"[上櫃] 下載外資/投信籌碼...")
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    url = f'https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=AL&date={year}%2F{month}%2F{day}&id=&response=json'

    res = retry_crwler(url, HEADERS_TPEX, sleep_range=(3, 7))
    if not res: return None

    try:
        json_data = res.json()
    except Exception as e:
        print(f"[解析錯誤] : {e}")
        return None

    if not json_data.get('tables') or not json_data['tables'][0].get('data'):
        return None

    table = json_data['tables'][0]
    df = pd.DataFrame(table['data'], columns=table['fields'])
    df = df.replace({r'<[^>]+>': '', ',': ''}, regex=True)

    # [0]代號 [1]名稱 [4]外資買賣超 [13]投信買賣超
    df = df.iloc[:, [0, 1, 4, 13]].copy()
    df.columns = ['id', 'name', 'Foreign_Net', 'Trust_Net']

    df['id'] = df['id'].astype(str).str.strip()
    df = df[~df['id'].str.startswith('00')]
    df = df[df['id'].str.len() == 4]

    if df.empty:
        return None

    df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].apply(pd.to_numeric, errors='coerce')
    df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].fillna(0)

    df['Foreign_Net'] /= 1000
    df['Trust_Net'] /= 1000
    return df

# ==========================================
# getdata
# ==========================================

def get_twse_data(date_str):
    
    df_b = basic_twse(date_str)
    if df_b is None or df_b.empty: 
        print(f"[{date_str}] 上市收盤行情抓取失敗")
        return None
        
    df_c = chips_twse(date_str)
    if df_c is None or df_c.empty: 
        print(f"[{date_str}] 上市籌碼抓取失敗(外資與投信將全數補 0)")
        df_b['Foreign_Net'] = 0.0
        df_b['Trust_Net'] = 0.0
        return df_b

    df_c_clean = df_c[['id', 'Foreign_Net', 'Trust_Net']]
    
    df = pd.merge(df_b, df_c_clean, on='id', how='left')
    
    df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].fillna(0)
    return df

def get_tpex_data(date_str):
    
    df_b = basic_tpex(date_str)
    if df_b is None or df_b.empty: 
        print(f"[{date_str}] 上櫃收盤行情抓取失敗")
        return None
        
    df_c = chips_tpex(date_str)
    if df_c is None or df_c.empty: 
        print(f"[{date_str}] 上櫃籌碼抓取失敗(外資與投信將全數補 0)")
        df_b['Foreign_Net'] = 0.0
        df_b['Trust_Net'] = 0.0
        return df_b

    df_c_clean = df_c[['id', 'Foreign_Net', 'Trust_Net']]
    
    df = pd.merge(df_b, df_c_clean, on='id', how='left')
    
    df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].fillna(0)
    return df

def get_data(date_str):
    twse_df = get_twse_data(date_str)
    tpex_df = get_tpex_data(date_str)
    
    valid_dfs = [df for df in [twse_df, tpex_df] if df is not None and not df.empty]
    
    if not valid_dfs:
        return None
        
    final_df = pd.concat(valid_dfs, ignore_index=True)
    return final_df

def update_market_data(specific_date=None):
    # 1. Date setup
    if specific_date:
        clean_date = specific_date.replace('-', '')
        target_dt = datetime.strptime(clean_date, '%Y%m%d')
    else:
        target_dt = datetime.now()

    if target_dt.weekday() >= 5:
        return {"status": "略過", "error": f"今日為週末 (週{['六','日'][target_dt.weekday()-5]})，市場休市。"}

    today_str = target_dt.strftime('%Y%m%d') 
    standard_date_str = target_dt.strftime('%Y-%m-%d') 

    # 2. Fetch Data
    df_new = get_data(today_str)
    if df_new is None or df_new.empty:
        return {"status": "失敗", "error": "無資料 (市場休市或爬蟲異常)。"}

    df_new['date'] = pd.to_datetime(standard_date_str)
    
    # 3. Data Validation Metrics
    new_count = len(df_new)
    nan_count = int(df_new.isna().sum().sum())
    
    # 4. Merge with Database
    if os.path.exists(FILE_NAME):
        try:
            df_db = pd.read_parquet(FILE_NAME)
            df_db['date'] = pd.to_datetime(df_db['date'])
            df_db = df_db[df_db['date'] != df_new['date'].iloc[0]] # 去重
            df_final = pd.concat([df_db, df_new], ignore_index=True)
        except Exception as e:
            return {"status": "失敗", "error": f"本地資料庫讀取失敗: {str(e)}"}
    else:
        df_final = df_new

    df_final = df_final.sort_values(by=['date','id'], ascending=[True, True])

    # 5. Save and Upload unconditionally
    df_final.to_parquet(FILE_NAME, index=False, engine='pyarrow')
    
    # 因為 backup_db.py 已經被移動到根目錄，這裡的路徑需要修正
    from backup_db import backup_to_supabase
    current_year = target_dt.year
    df_year = df_final[df_final['date'].dt.year == current_year]
    temp_file = config.MARKET_DATA_DIR / f"market_{current_year}.parquet"
    df_year.to_parquet(temp_file, index=False, engine='pyarrow')
    
    success = backup_to_supabase('stock_data_backup', str(temp_file), f"market/market_{current_year}.parquet")
    if os.path.exists(temp_file): os.remove(temp_file)
        
    action_msg = "已上傳覆寫至 Supabase 雲端" if success else "上傳 Supabase 失敗"

    return {
        "status": "成功",
        "date": standard_date_str,
        "rows": new_count,
        "nans": nan_count,
        "action": action_msg
    }

if __name__ == "__main__":
    def backfill(dates_to_fix):
        for d_str in dates_to_fix:
            # 直接把 'YYYY-MM-DD' 傳給主程式
            print(f"\n準備回補: {d_str}")
            update_market_data(specific_date=d_str)
            
            # 延遲避免被證交所 Ban IP
            print("等待 5 秒後進行下一個日期的回補...\n")
            time.sleep(5)
        print("指定日期之歷史資料修補全數完成！")

    #===========
    #每日更新程式
    #===========
    update_market_data()

    #===========
    #回補資料
    #===========
    # dates_to_fix = [
    #         '20260302', '20260303',
    #         '20260304', '20260305', 
    #         '20260306', '20260309',
    #     ]
    # backfill(dates_to_fix)
        