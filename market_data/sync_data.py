import config
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from datetime import datetime

env_path = config.BASE_DIR/'.env'
load_dotenv(dotenv_path=env_path)

URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("找不到 SUPABASE_URL 或 SUPABASE_SECRET_KEY，請確認 .env 檔案設定")

supabase: Client = create_client(URL, KEY)

def download_from_supabase(bucket_name: str, supabase_file_path: str, local_file_path: str):
    """從 Supabase 下載單一檔案到本地端"""
    label = os.path.basename(local_file_path)
    try:
        # 下載檔案資料 (會回傳 bytes)
        res = supabase.storage.from_(bucket_name).download(supabase_file_path)
        
        # 確保本地資料夾存在，解決 GitHub Actions 因為資料夾是空的而報錯的問題
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # 寫入本地檔案
        with open(local_file_path, "wb") as f:
            f.write(res)
        print(f"{label} 同步下載成功")
        return True
    except Exception as e:
        print(f"{label} 同步下載失敗: {e}")
        return False

def sync():
    """執行每日同步作業"""
    current_year = datetime.now().year
    print(f"開始從 Supabase 同步 {current_year} 年的市場資料...\n")
    
    # 1. 下載當年的市場資料 (Parquet) 到暫存檔
    remote_market = f"market/market_{current_year}.parquet"
    temp_local_market = f"{config.MARKET_DATA_DIR}/market_{current_year}_temp.parquet"
    
    success_market = download_from_supabase('stock_data_backup', remote_market, temp_local_market)
    
    if success_market:
        print(f"正在將 {current_year} 年資料與本地歷史資料合併...")
        try:
            # 讀取下載下來的當年最新資料
            df_new = pd.read_parquet(temp_local_market)
            df_new['date'] = pd.to_datetime(df_new['date'])
            
            # 讀取本地完整的歷史資料
            local_market_path = f"{config.MARKET_DATA_DIR}/market.parquet"
            if os.path.exists(local_market_path):
                df_local = pd.read_parquet(local_market_path)
                df_local['date'] = pd.to_datetime(df_local['date'])
                
                # 剔除本地檔案中的當年資料，保留歷史資料 (小於 current_year 的都是歷史)
                df_history = df_local[df_local['date'].dt.year < current_year]
                
                # 合併: 歷史資料 + 下載的最新的當年資料
                df_final = pd.concat([df_history, df_new], ignore_index=True)
                df_final = df_final.sort_values(by=['date', 'id'], ascending=[True, True])
                
                # 覆寫回本地完整的 market.parquet
                df_final.to_parquet(local_market_path, index=False, engine='pyarrow')
                print(f"合併完成！目前本地端總資料筆數: {len(df_final)}")
            # else:
            #     # 如果本地根本沒有檔案，直接存成 market.parquet
            #     df_new.to_parquet(local_market_path, index=False, engine='pyarrow')
            #     print(f"找不到舊檔案，已直接建立 market.parquet")
                
        except Exception as e:
            print(f"資料合併失敗: {e}")
        finally:
            # 清除下載的暫存檔
            if os.path.exists(temp_local_market):
                os.remove(temp_local_market)

    # 2. 下載最新基金淨值資料 (navpu.csv)
    remote_navpu = "navpu.csv"
    local_navpu = str(config.FUND_STOCK_ADMINISTRATION_DIR / "navpu.csv")
    print("\n開始從 Supabase 同步最新基金淨值 (navpu.csv)...")
    download_from_supabase('stock_data_backup', remote_navpu, local_navpu)

if __name__ == '__main__':
    sync()
