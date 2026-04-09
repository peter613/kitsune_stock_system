import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
import time
from io import StringIO
import urllib3
import config

# 禁用不安全請求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_115_1_0.html
# https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_115_1_1.html
# ==========================================
# 全域配置與路徑
# ==========================================
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
REVENUE_FILE = config.DATABASE_DIR / 'revenue.csv'
LOOKUP_FILE = config.LOOKUP_DIR / 'stock_lookup.csv'

# 自動計算前一個月的年份與月份
# now = '2022-01-01'
# now = datetime.strptime(now, '%Y-%m-%d')
now = datetime.now()
if now.month == 1:
    year = now.year - 1911 - 1
    month = 12
else:
    year = now.year - 1911
    month = now.month - 1

# ==========================================
# 爬蟲核心函式
# ==========================================
def revenue_crawer(name,is_foreign):
    """
    抓取公開資訊觀測站營收資料
    name: 'sii' (上市) 或 'otc' (上櫃)
    is_foreign: '0' (本國) 或 '1' (外國/KY)
    """
    url_twse = f'https://mopsov.twse.com.tw/nas/t21/{name}/t21sc03_{year}_{month}_{is_foreign}.html'
    try:
        r = requests.get(url_twse, headers=headers, verify=False)
        r.encoding = 'cp950' 
        
        if r.status_code != 200:
            print(f"[{name}] 下載失敗，網站可能維護中或路徑錯誤")
            return None

        dfs = pd.read_html(StringIO(r.text))
        target_df = pd.DataFrame()
        
        for df in dfs:
            df_str = str(df.columns) + str(df.iloc[0:1]) # 同時檢查標題與第一列
            if df.shape[1] > 5 and ('當月營收' in df_str or '公司代號' in df_str or '公司 代號' in df_str):
                target_df = pd.concat([target_df, df], axis=0, ignore_index=True)

        if target_df.empty:
            print(f"[{name}] 找不到表格資料")
            return None

        # 清洗 MultiIndex 並過濾雜質
        target_df.columns = target_df.columns.get_level_values(-1)
        target_df = target_df[~target_df['公司 代號'].astype(str).str.contains('合計')]
        target_df = target_df[target_df['公司 代號'] != '公司 代號']
        
        rename_map = {
            '公司 代號': 'id',
            '公司名稱': 'name',
            '當月營收': 'month_revenue',
            '上月比較 增減(%)': 'mom',
            '去年同月 增減(%)': 'yoy'
        }
        
        cols_to_keep = [c for c in rename_map.keys() if c in target_df.columns]
        final_df = target_df[cols_to_keep].rename(columns=rename_map)
        
        # 轉為數值格式
        numeric_cols = ['month_revenue', 'mom', 'yoy']
        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        final_df['date'] = datetime(year + 1911, month, 1)

        # 新增市場類型標籤：上市(sii)=twse, 上櫃(otc)=tpex
        final_df['market_type'] = 'twse' if name == 'sii' else 'tpex'

        print(f"[{name}{'-ky' if is_foreign == 1 else ''}] 成功抓取 {len(final_df)} 筆資料")
        return final_df

    except Exception as e:
        print(f"⚠️ [{name}{'-ky' if is_foreign == 1 else ''}] 發生錯誤: {e}")
        return None

# ==========================================
# 主要執行邏輯
# ==========================================
def update_revenue():
    """
    供外部調用的主要接口，包含爬取、合併、存檔
    """
    print(f">> 啟動營收更新程序 (目標月份: {year}/{month})...")
    
    # 在函式內才執行爬蟲，避免導入副作用
    twse_df = revenue_crawer('sii',0)
    time.sleep(2)
    twse_ky_df = revenue_crawer('sii',1) 
    time.sleep(2)
    otc_df = revenue_crawer('otc',0)
    time.sleep(2) 
    otc_ky_df = revenue_crawer('otc',1)

    # 1. 合併本次抓取的所有資料
    dfs = [df for df in [twse_df, otc_df, twse_ky_df, otc_ky_df] if df is not None]
    if not dfs:
        print("沒有抓到任何資料，結束程序。")
        return
    new_data = pd.concat(dfs, ignore_index=True)
    new_data['id'] = new_data['id'].astype(str)
    # -----------------------------------------
    # 處理 1: 更新 stock_lookup.csv (處理新上市櫃股票)
    # -----------------------------------------
    new_lookup = new_data[['id', 'name', 'market_type']].drop_duplicates(subset=['id'])
    
    if os.path.exists(LOOKUP_FILE):
        old_lookup = pd.read_csv(LOOKUP_FILE, dtype={'id': str})
        all_lookup = pd.concat([old_lookup, new_lookup], ignore_index=True)
    else:
        all_lookup = new_lookup
        
    # 去重，保留最新抓到的名稱與市場狀態
    all_lookup.drop_duplicates(subset=['id'], keep='last', inplace=True)
    all_lookup.sort_values('id', inplace=True)

    # -----------------------------------------
    # 處理 2: 更新 revenue.csv (純淨長表)
    # -----------------------------------------
    new_revenue = new_data[['date', 'id', 'month_revenue', 'yoy', 'mom']]
    
    if os.path.exists(REVENUE_FILE):
        print(f"讀取現有資料庫: {REVENUE_FILE}")
        old_revenue = pd.read_csv(REVENUE_FILE, dtype={'id': str})
        old_revenue['date'] = pd.to_datetime(old_revenue['date'])
    else:
        print("建立新營收資料庫...")
        old_revenue = pd.DataFrame()

    all_revenue = pd.concat([old_revenue, new_revenue], ignore_index=True)
    all_revenue['date'] = pd.to_datetime(all_revenue['date'])
    all_revenue.drop_duplicates(subset=['date', 'id'], keep='last', inplace=True)
    all_revenue.sort_values(by=['date', 'id'], ascending=[True, True], inplace=True)

    # 4. 交互確認與存檔

    last_date = old_revenue['date'].max()
    last_month_count = len(old_revenue[old_revenue['date'] == last_date])
    print(f"本次新增 {len(new_revenue)} 筆 (上個月筆數: {last_month_count})")
    print(f"\n預計更新後營收總筆數: {len(all_revenue)}")
    print(f"預計更新後股票總檔數: {len(all_lookup)}")
    user_input = input('確認資料無誤？ [按 Y 存檔 / 其他鍵取消]: ').strip().lower()
    
    if user_input == 'y':
        all_lookup.to_csv(LOOKUP_FILE, index=False, encoding='utf-8-sig')
        all_revenue.to_csv(REVENUE_FILE, index=False, encoding='utf-8-sig')
        print("雙檔案(revenue+lookup)存檔完成。")
    else:
        print("操作已取消。")
if __name__ == "__main__":
    # 只有直接執行此檔案時才會跑這裡
    update_revenue()