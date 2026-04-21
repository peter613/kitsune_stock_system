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
import io
import lxml
session = requests.Session()
# 設定檔案路徑
FILE_NAME = config.DATABASE_DIR/'market.parquet'

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
    if not res: return pd.DataFrame()
    
    try:
        json_data = res.json()
    except Exception as e:
        print(f"[解析錯誤]: {e}")
        return pd.DataFrame()

    required_cols = ['date', 'id', 'name', 'volume', 'amount', 'open', 'high', 'low', 'close']
    if not json_data.get('tables'): return pd.DataFrame(columns=required_cols)

    try:
        target = next(t for t in json_data['tables'] if '每日收盤行情' in t.get('title', ''))
        df = pd.DataFrame(target['data'], columns=target['fields'])
    except: 
        return pd.DataFrame(columns=required_cols)

    df = df.replace({r'<[^>]+>': '', ',': ''}, regex=True)
    df.replace('--', pd.NA, inplace=True)
    
    rename_map = {'證券代號':'id', '證券名稱':'name', '成交股數':'volume', '成交金額':'amount', 
                  '開盤價':'open', '最高價':'high', '最低價':'low', '收盤價':'close'}
    df = df.rename(columns=rename_map)
    
    df['id'] = df['id'].astype(str).str.strip()
    df = df[~df['id'].str.startswith('00')]
    df = df[df['id'].str.len() == 4]
    
    df.insert(0, 'date', date_str)
    num_cols = ['volume', 'amount', 'open', 'high', 'low', 'close']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df['volume'] /= 1000
    df['amount'] /= 1000
    return df[['date','id', 'name', 'volume', 'amount', 'open', 'high', 'low', 'close']]

def chips_twse(date_str):
    print(f"[上市] 下載外資/投信籌碼...")
    url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALL&response=html"

    res = retry_crwler(url, HEADERS_TWSE, sleep_range=(2, 5))
    if not res: return None
    res.encoding = 'utf-8' 
            
    try:
        dfs = pd.read_html(io.StringIO(res.text), thousands=',') 
        df_raw = dfs[0] 
        
        df = df_raw.iloc[:, [0, 1, 4, 10]].copy() 
        df.columns = ['id', 'name', 'Foreign_Net','Trust_Net'] 
        
        df['id'] = df['id'].astype(str).str.strip()
        df = df[~df['id'].str.startswith('00')]
        df = df[df['id'].str.len() == 4]
        
        df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].apply(pd.to_numeric, errors='coerce')
        df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].fillna(0)

        df['Foreign_Net']/=1000
        df['Trust_Net']/=1000
        return df
    
    except Exception as e:
        print(f"[解析錯誤] 無法取得/解析 HTML: {e}")
        return None

# ==========================================
# 上櫃 (tpex)
# ==========================================
def basic_tpex(date_str):
    print(f"[上櫃] 下載收盤行情...")
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    url = f'https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?date={year}%2F{month}%2F{day}&type=EW&id=&response=html&order=0&sort=asc'
    res = retry_crwler(url, HEADERS_TPEX, sleep_range=(3, 7))
    if not res: return None
    res.encoding = 'utf-8' 

    try:
        dfs = pd.read_html(io.StringIO(res.text), thousands=',') 
    except Exception as e:
        print(f"無法解析 HTML 表格: {e}")
        return None
        
    df_raw = dfs[0] 
    
    df = df_raw.iloc[:, [0, 1, 2, 4, 5, 6, 7,8]].copy() 
    df.columns = ['id', 'name', 'close','open','high','low','volume','amount'] 
    
    df['id'] = df['id'].astype(str).str.strip()
    df = df[~df['id'].str.startswith('00')]
    df = df[df['id'].str.len() == 4]
    
    df[['close','open','high','low','volume','amount']] = df[['close','open','high','low','volume','amount']].apply(pd.to_numeric, errors='coerce')
    df[['close','open','high','low','volume','amount']] = df[['close','open','high','low','volume','amount']].fillna(0)

    df['volume'] /= 1000
    df['amount'] /= 1000
    df.insert(0, 'date', date_str)

    return df

def chips_tpex(date_str):
    print(f"[上櫃] 下載外資/投信籌碼...")
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    url = f'https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=AL&date={year}%2F{month}%2F{day}&id=&response=html'

    res = retry_crwler(url, HEADERS_TPEX, sleep_range=(3, 7))
    if not res: return None
    res.encoding = 'utf-8' 

    try:
        dfs = pd.read_html(io.StringIO(res.text), thousands=',') 
    except Exception as e:
        print(f"無法解析 HTML 表格: {e}")
        return None
        
    df_raw = dfs[0] 
    
    df = df_raw.iloc[:, [0, 1, 4, 13]].copy() 
    df.columns = ['id', 'name', 'Foreign_Net','Trust_Net'] 
    
    df['id'] = df['id'].astype(str).str.strip()
    df = df[~df['id'].str.startswith('00')]
    df = df[df['id'].str.len() == 4]
    
    df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].apply(pd.to_numeric, errors='coerce')
    df[['Foreign_Net', 'Trust_Net']] = df[['Foreign_Net', 'Trust_Net']].fillna(0)

    df['Foreign_Net']/=1000
    df['Trust_Net']/=1000
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

# ==========================================
# Part 3: 主程式
# ==========================================
def update_market_data(specific_date=None):
    # 1. 統一狀態賦值：取得字串與 datetime 物件
    if specific_date:
        # 將可能存在的 '-' 移除，統一以 '%Y%m%d' 格式解析
        clean_date = specific_date.replace('-', '')
        target_dt = datetime.strptime(clean_date, '%Y%m%d')
        print(f"回補指定日期資料...")
    else:
        # 每日更新模式
        target_dt = datetime.now()

    # 週末防呆
    if target_dt.weekday() >= 5:
        print(f"指定日期為週{['六','日'][target_dt.weekday()-5]}，市場休市，自動取消更新。")
        return

    # 轉換成爬蟲需要的 YYYYMMDD 格式
    today_str = target_dt.strftime('%Y%m%d') 
    # 轉換回 YYYY-MM-DD 供最後存檔比對使用
    standard_date_str = target_dt.strftime('%Y-%m-%d') 

    print(f"執行Kitsune crawler，日期: {today_str}...")
    
    # 1. 啟動雙引擎抓取
    df_new = get_data(today_str)

    if df_new is None or df_new.empty:
        print('抓取失敗或無新資料，程式中斷(請確認當日是否開市)。')
        return

    # 統一將 date 轉為 YYYY-MM-DD 格式，方便閱讀與排序
    df_new['date'] = pd.to_datetime(standard_date_str) # 直接賦值標準格式
    new_count = len(df_new)

    # 每日爬蟲 NaN 檢查
    print(f"\n今天nan值分布:")
    nan_dist = df_new.isna().sum()
    print(nan_dist[nan_dist > 0].to_string() if nan_dist.sum() > 0 else "無缺失值")

    # 2. 資料庫讀取與更新邏輯
    if os.path.exists(FILE_NAME):
        try:
            df_db = pd.read_parquet(FILE_NAME)
            df_db['date'] = pd.to_datetime(df_db['date'])
            
            # 紀錄合併前的總筆數
            old_total_count = len(df_db)
            
            # 剔除舊資料中「與這次抓取同日期」的數據 (防止重複寫入)
            df_db = df_db[df_db['date'] != df_new['date'].iloc[0]]

            # 上筆資料大小
            last_update_count = 0
            if not df_db.empty:
                last_date = df_db['date'].max()
                last_update_count = len(df_db[df_db['date'] == last_date])

            # 合併新舊資料
            df_final = pd.concat([df_db, df_new], ignore_index=True)
            
            # 計算差額並顯示稽核資訊
            diff = new_count - last_update_count
            print(f"\n本次成功獲取資料: {new_count} 筆")
            print(f"資料合併成功，目前總筆數: {len(df_final)} (與上筆資料差額: {diff})")
            
        except Exception as e:
            # 防覆蓋機制：讀取失敗直接中斷，絕對不覆蓋原檔案
            print(f"讀取舊檔失敗 ({e})。為保護歷史資料，停止覆蓋並強制中斷程式。")
            return
    else:
        print("找不到舊檔案，建立全新資料庫。")
        df_final = df_new

    # 3. 排序 (以日期為主，代號為輔，確保時序正確)
    df_final = df_final.sort_values(by=['date','id'], ascending=[True, True])

    # 4. 驗證與存檔
    print('\n==================== [資料庫抽樣檢驗] ====================')
    print("[2330 台積電] 最新動態:")
    print(df_final[df_final['id'] == '2330'].tail(3).to_string(index=False))
    print('\n[5347 世界先進] 最新動態:')
    print(df_final[df_final['id'] == '5347'].tail(3).to_string(index=False))
    print('==========================================================\n')

    # 自動化模式判定：如果有指定 specific_date，為了方便批次處理，直接存檔不詢問
    if specific_date:
         df_final.to_parquet(FILE_NAME, index=False, engine='pyarrow')
         print(f"{standard_date_str} 資料庫回補成功!")
    else:
        user_input = input('確認資料無誤？ [按 Y 存檔 / 其他鍵取消]: ')
        if user_input.lower() == 'y':
            df_final.to_parquet(FILE_NAME, index=False, engine='pyarrow')
            print("資料庫更新成功!")
        else:
            print("資料未寫入，記憶體已清空。")

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
    #     backfill(dates_to_fix)
    # backfill(dates_to_fix)
        