import os
import pandas as pd
import numpy as np
from talib import *
import gc
import config
from scipy.stats import entropy

# --- 設定區 ---
def create_prediction(stock_filter=True,amount_on=True):
    # 1. 檔案路徑更新
    file_name = config.DATABASE_DIR / 'market.parquet'
    file_name_revenue = config.DATABASE_DIR / 'revenue.csv'
    file_save_name = config.FEATURES_DIR / f'pred.csv'

    print("正在讀取資料庫...")
    # 2. 讀取邏輯更新：使用 read_parquet，並在讀取 revenue 時鎖定型態
    df = pd.read_parquet(file_name)
    df_revenue = pd.read_csv(file_name_revenue, encoding="utf-8", dtype={'id': str})
    
    # 3. 強制時間格式化 (與訓練集同步)
    df["date"] = pd.to_datetime(df["date"])
    df_revenue["date"] = pd.to_datetime(df_revenue["date"])

    # 4. [防禦前置] 在進入任何排序或檢查前，強制將 id 轉為字串並補零
    df['id'] = df['id'].astype(str).str.zfill(4)
    df_revenue['id'] = df_revenue['id'].astype(str).str.zfill(4)

    def check_data_order(df):
        # 1. 檢查 ID 是否為單調遞增
        id_monotonic = df['id'].is_monotonic_increasing
        # 2. 檢查在相同 ID 內，日期是否單調遞增
        mask = df['id'] == df['id'].shift(1)
        date_error = (df['date'][mask] < df['date'].shift(1)[mask]).any()
        return id_monotonic and not date_error

    if not check_data_order(df):
        print("檢測到資料順序錯亂，執行排序...")
        df = df.sort_values(['id', 'date']).reset_index(drop=True)
    else:
        print("資料序列正確，跳過排序步驟。")

    # [優化 1] 確保有足夠的 SMA240 暖機時間 (保留 600 天)
    latest_date_in_db = df["date"].max()
    cutoff_date = latest_date_in_db - pd.DateOffset(days=600)
    print(f"預測模式：裁切歷史資料 (保留 {cutoff_date.date()} 之後)...")
    df = df[df["date"] >= cutoff_date].copy().reset_index(drop=True)

    if stock_filter:
        print("正在保留白名單...")
        # ---------------------------------------------------------
        # 白名單
        # ---------------------------------------------------------
        get_white_list = pd.read_csv(config.WHITE_LIST_DIR / 'white_list.csv', encoding='utf-8-sig')
        white_ids = get_white_list['id'].astype(str)

        df = df[df['id'].isin(white_ids)].copy()
        df_revenue = df_revenue[df_revenue['id'].isin(white_ids)].copy()
        # ---------------------------------------------------------
        # 黑名單(電信三雄)
        # ---------------------------------------------------------
        black_ids = ['2412', '3045', '4904']
        df = df[~df['id'].isin(black_ids)].copy()
        df_revenue = df_revenue[~df_revenue['id'].isin(black_ids)].copy()
    
    # ---------------------------------------------------------
    # 【新增】3.5 上市天數過濾 (IPO 隔離防禦)
    # ---------------------------------------------------------
    # 計算每個 id 的總資料筆數
    id_counts = df['id'].value_counts()
    # 找出大於等於 240 天的合格名單
    valid_ids = id_counts[id_counts >= 240].index
    print(f"剔除上市未滿 240 天的股票，數量: {len(id_counts) - len(valid_ids)}")
    # 直接用 isin 進行矩陣級別的過濾
    df = df[df['id'].isin(valid_ids)].copy().reset_index(drop=True)

    def clean_outliers(df, feature_cols):
        for col in feature_cols:
            if col in df.columns:
                if col in ['mom', 'yoy', 'yoy_ma5', 'yoy_ma5_bias']:
                    df[col] = df[col].clip(-100, 100)
                else:
                    lower = df[col].quantile(0.01)
                    upper = df[col].quantile(0.99)
                    df[col] = df[col].clip(lower, upper)
        return df

    # 特徵計算 V11 Sniper
    def compute_features(group):
        n = len(group)
        close = group["close"].values.astype(float)
        high = group["high"].values.astype(float)
        low = group["low"].values.astype(float)
        volume = group["volume"].values.astype(float)
        amount = group["amount"].values.astype(float)
        amount_ln = np.log1p(group['amount'].values.astype(float))
        
        # [優化 2] 處理法人可能存在的空值，避免 NaN 蔓延
        vol_10 = SMA(volume, 10) * 10 + 1  # 加 1 避免分母為零
        vol_3 = SMA(volume, 3) * 3 + 1
        t_net_10 = SMA(group['Trust_Net'].values.astype(float), 10) * 10
        t_net_3 = SMA(group['Trust_Net'].values.astype(float), 3) * 3
        
        #投信方向盤 (保留正負號，判斷是作帳還是結帳)
        trust_in_10 = t_net_10 / vol_10
        trust_in_3 = t_net_3 / vol_3

        amount20 = SMA(amount_ln, 20)
        amount_breakout_20  = amount / (SMA(amount,20) + 1e-9)
        amount_ema_20 = EMA(amount_ln, 20)

        time_idx = np.arange(n, dtype=float)
        r_20 = CORREL(close, time_idx, timeperiod=20)
        r_20_abs = np.abs(r_20)

        # ── 突破 / 支撐  (範圍 0~1)───────────────────────────────────────
        # 突破 : 1代表在最高點
        breakout_20 = close / (MAX(close, timeperiod=20) + 1e-9)
        breakout_200 = close / (MAX(close, timeperiod=200) + 1e-9)
        # 築底 : 1代表無窮遠離底部
        support_20 = 1 - (MIN(close, timeperiod=20) / (close + 1e-9))
        support_200 = 1 - (MIN(close, timeperiod=200) / (close + 1e-9))

        sma200 = close / (SMA(close, timeperiod=200) + 1e-9)
        sma20 = close / (SMA(close, timeperiod=20) + 1e-9)

        adx = ADX(high, low, close, timeperiod=14)
        
        try:
            kama = close / (KAMA(close, timeperiod=30) + 1e-9)
        except:
            kama = np.full(n, np.nan)

        #atr = ATR(high, low, close, timeperiod=14) / (close + 1e-9)

        # ── 字典一次性寫回 DataFrame ──────────────────────────────
        group = group.copy()          # 避免 SettingWithCopyWarning
        cols = {
            'trust_in_10': trust_in_10,
            'trust_in_3': trust_in_3,
            'amount20': amount20,
            'amount_ema_20': amount_ema_20,
            'a_breakout_20':amount_breakout_20,
            'r_20_abs': r_20_abs,
            'breakout_20': breakout_20,
            'breakout_200': breakout_200,
            'support_20': support_20,
            'support_200': support_200,
            'sma200': sma200,
            'sma20': sma20,
            'adx': adx,
            'kama': kama,
        }
        for col, arr in cols.items():
            group[col] = arr

        return group

    print("正在執行特徵計算 (Progressing)...")
    
    # [備份防禦] 進入 groupby 之前，建立副本
    df['safe_id'] = df['id'].astype(str)
    
    df = df.groupby("id", group_keys=False).apply(compute_features)

    # [奪回 ID 補丁] 直接攤平 Index 並還原備份 ID
    df = df.reset_index(drop=True)
    df['id'] = df['safe_id']

    df_revenue["date"] = pd.to_datetime(df_revenue["date"])
    df_revenue = df_revenue.sort_values(['id', 'date'])
    df_revenue['yoy_ma5'] = df_revenue.groupby('id')['yoy'].transform(lambda x: x.rolling(window=5).mean())
    df_revenue["merge_key"] = df_revenue["date"].dt.to_period("M").apply(lambda x: x + 1)
    df["merge_key"] = df["date"].dt.to_period("M")

    df['id'] = df['id'].astype(str)
    df_revenue['id'] = df_revenue['id'].astype(str)

    print("正在執行主表與營收表合併...")
    df = pd.merge(
        df,
        df_revenue[["id", "merge_key", "mom", "yoy", "yoy_ma5"]],
        on=["id", "merge_key"],
        how="left",
    )

    df["yoy"] = df.groupby("id")["yoy"].ffill()
    df["mom"] = df.groupby("id")["mom"].ffill()
    df["yoy_ma5"] = df.groupby("id")["yoy_ma5"].ffill()
    df['yoy_ma5_bias'] = df['yoy'] - df['yoy_ma5']

    # [優化 4] 防禦機制：調降閾值以適應純科技股池
    date_counts = df["date"].value_counts()
    valid_dates = date_counts[date_counts > 250].index # 修改為 250 檔
    if len(valid_dates) == 0:
        print("錯誤：找不到完整的交易日資料。")
        return
    latest_date = valid_dates.max()
    print(f"提取最新營收月份: {df_revenue['date'].dt.strftime('%Y-%m').max()}")
    print(f"提取最新股票交易日: {latest_date.date()} (共 {date_counts[latest_date]} 檔)")
    
    final_predict_df = df[df["date"] == latest_date].copy()
    
    del df
    gc.collect()

    
    print("執行EMA成交值流動性過濾 (保留Top 1/3)...")
    final_predict_df["day_count"] = len(final_predict_df)
    final_predict_df["liq_rank"] = final_predict_df["amount_ema_20"].rank(ascending=False)
    final_predict_df = final_predict_df[final_predict_df["liq_rank"] <= (final_predict_df["day_count"] / 3)]

    if amount_on:
        AMOUNT_THRESHOLD = 50000 #當天成交0.5億
        final_predict_df = final_predict_df[final_predict_df['amount'] > AMOUNT_THRESHOLD].reset_index(drop=True)
        print(f'執行成交值流動性過濾> {AMOUNT_THRESHOLD}(千)...')

    features = [ 
        #刪掉60日特徵 ("amount60","sma60","r2_60")

        # 均線與技術指標
        "sma20",  "sma200", "kama", "adx",
        

        # 流動性 (Log化)
        "amount20",
        
        # 籌碼與法人動能
        "trust_in_10", "trust_in_3",
        
        # 基本面
        "yoy_ma5_bias","yoy",

        #突破與築底
        "breakout_20", "breakout_200",
        "support_20","support_200",
    ]

    print("正在執行特徵去極值處理...")
    final_predict_df = clean_outliers(final_predict_df, features)

    final_output = final_predict_df[["id", "name", "date", "close"] + features].dropna()
    final_output.to_csv(file_save_name, index=False, encoding="utf-8-sig")

    print(f"預測資料已生成: {file_save_name}\n成功保留 {len(final_output)} 檔標的。")

if __name__ == "__main__":
    create_prediction()