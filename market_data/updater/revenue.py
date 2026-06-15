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
REVENUE_FILE = config.MARKET_DATA_DIR / 'revenue.csv'
WHITE_LIST_FILE = config.STOCK_WHITE_LIST_DIR / 'white_list.csv'


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
# 爬蟲核心函式 (含重試機制)
# ==========================================
def revenue_crawer(name, is_foreign):
    """
    抓取公開資訊觀測站營收資料
    name: 'sii' (上市) 或 'otc' (上櫃)
    is_foreign: '0' (本國) 或 '1' (外國/KY)
    """
    url_twse = f'https://mopsov.twse.com.tw/nas/t21/{name}/t21sc03_{year}_{month}_{is_foreign}.html'
    
    for attempt in range(2): # 嘗試 2 次 (1次原始 + 1次重試)
        try:
            r = requests.get(url_twse, headers=headers, verify=False, timeout=15)
            r.encoding = 'cp950' 
            
            if r.status_code != 200:
                raise Exception(f"HTTP {r.status_code}")

            dfs = pd.read_html(StringIO(r.text))
            target_df = pd.DataFrame()
            
            for df in dfs:
                df_str = str(df.columns) + str(df.iloc[0:1])
                if df.shape[1] > 5 and ('當月營收' in df_str or '公司代號' in df_str or '公司 代號' in df_str):
                    target_df = pd.concat([target_df, df], axis=0, ignore_index=True)

            if target_df.empty:
                raise Exception("Empty Table")

            # 清洗資料邏輯
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
            
            numeric_cols = ['month_revenue', 'mom', 'yoy']
            for col in numeric_cols:
                if col in final_df.columns:
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            
            final_df['date'] = datetime(year + 1911, month, 1)
            final_df['market_type'] = 'twse' if name == 'sii' else 'tpex'

            print(f"[{name}{'-ky' if is_foreign == '1' else ''}] 成功抓取 {len(final_df)} 筆資料")
            return final_df

        except Exception as e:
            if attempt < 1:
                print(f"[{name}] 第 {attempt+1} 次重試... 原因: {e}")
                time.sleep(5) # 避免過快請求被封鎖
                continue
            else:
                print(f"[{name}] 最終錯誤: {e}")
                return None

# ==========================================
# 主要執行邏輯
# ==========================================
def update_revenue():
    """
    供外部調用的主要接口，包含爬取、合併、存檔
    """
    print(f"啟動營收更新程序 (目標月份: {year}/{month})...")
    
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
    # 更新 revenue.csv (純淨長表)
    # -----------------------------------------
    new_revenue = new_data[['date', 'id', 'month_revenue', 'yoy', 'mom']]
    
    if os.path.exists(REVENUE_FILE):
        print(f"讀取營收資料庫: {REVENUE_FILE}")
        print(f'讀取白名單: {WHITE_LIST_FILE}')
        old_revenue = pd.read_csv(REVENUE_FILE, dtype={'id': str})
        old_revenue['date'] = pd.to_datetime(old_revenue['date'])
        white_list_data = pd.read_csv(WHITE_LIST_FILE, dtype={'id': str})
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
    white_list_exsist = white_list_data[~(white_list_data['market']=='下市櫃')]
    white_list_revenue = new_revenue[new_revenue['id'].isin(white_list_exsist['id'])]
    print(f'\n白名單 : {len(white_list_revenue)}/{len(white_list_exsist)}筆')
    print(f"本次新增 {len(new_revenue)} 筆 (上個月筆數: {last_month_count})")
    print(f"預計更新後營收總筆數: {len(all_revenue)}")

    user_input = input('確認資料無誤？ [按 Y 存檔 / 其他鍵取消]: ').strip().lower()
    
    if user_input == 'y':
        all_revenue.to_csv(REVENUE_FILE, index=False, encoding='utf-8-sig')
        print("revenue.csv存檔完成。")
    else:
        print("操作已取消。")
if __name__ == "__main__":
    # 只有直接執行此檔案時才會跑這裡
    update_revenue()