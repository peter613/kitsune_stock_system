import os
import pandas as pd
import numpy as np
from talib import *
import gc
import config

# --- 設定區 ---
def create_prediction(stock_filter=True):
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
        print("正在過濾低成長股票...")
        # 傳產: 11(水泥), 12(食品), 13(塑膠), 14(紡織), 19(造紙), 21(橡膠), 22(汽車)
        # 營建航運觀光百貨: 25(營建), 26(航運), 27(觀光), 29(百貨), 55(營建), 59(百貨)
        # 金融證券: 28(金融), 58(金融), 60(證券)
        # 生技與其他傳統: 41(生技), 47(生技化工), 99(其他傳統產業)
        # 其他冷門傳產與化工: 15, 16, 17, 20, 45, 84, 89
        black_prefixes = (
            # 1. 原物料與基礎工業
            '11', # 水泥工業
            '13', # 塑膠工業
            '14', # 紡織纖維
            '18', # 玻璃陶瓷
            '19', # 造紙工業
            '20', # 鋼鐵工業
            '21', # 橡膠工業

            # 2. 民生、消費與觀光
            '12', # 食品工業
            '22', # 汽車工業
            '27', # 觀光事業 (上市/上櫃)
            '29', # 貿易百貨 (上市)
            '59', # 貿易百貨 (上櫃)

            # 3. 化工、生技與醫療 (投信偶爾著墨，但估值邏輯與電子股完全不同)
            '17', # 化學工業 / 傳統生技
            '41', # 生技醫療業
            '47', # 化學工業 (上櫃)
            '65', # 燃氣油電 (如6505台塑化) / 新興生技醫療

            # 4. 傳統機電與電纜
            '15', # 電機機械 (上市)
            '16', # 電器電纜
            '45', # 電機機械 (上櫃，如4532瑞智)

            # 5. 營建與航運
            '25', # 建材營造 (上市)
            '55', # 建材營造 (上櫃)
            '26', # 航運業

            # 6. 金融保險與證券 (股本極大、波動極低，會干擾機器學習的特徵權重)
            '28', # 金融保險 (上市，如各大金控)
            '58', # 金融保險 (上櫃/特定銀行)
            '60', # 證券業

            # 7. 其他綜合類股 (包含許多冷門股或控股公司)
            '84', # 其他業 / 觀光生技 (如8454富邦媒)
            '89', # 其他業 (上櫃冷門股)
            '99'  # 其他業 (上市綜合類，如9904寶成)
        )
        black_ids = ('2412', '3045', '4904', '8454', '2353', '2357','6757')
        
        # 確保 id 是字串再過濾
        df['id'] = df['id'].astype(str)
        df_revenue['id'] = df_revenue['id'].astype(str)
        df = df[~df['id'].str.startswith(black_prefixes) & ~df['id'].isin(black_ids)].copy().reset_index(drop=True)
        df_revenue = df_revenue[~df_revenue['id'].str.startswith(black_prefixes) & ~df_revenue['id'].isin(black_ids)].copy().reset_index(drop=True)

    def clean_outliers(df, feature_cols):
        for col in feature_cols:
            if col in df.columns:
                if col in ['mom', 'yoy', 'yoy_ma3', 'yoy_ma3_bias']:
                    df[col] = df[col].clip(-200, 200)
                else:
                    lower = df[col].quantile(0.01)
                    upper = df[col].quantile(0.99)
                    df[col] = df[col].clip(lower, upper)
        return df

    # 特徵計算 V11 Sniper
    def compute_features(group):
        close = group["close"].values.astype(float)
        high = group["high"].values.astype(float)
        low = group["low"].values.astype(float)
        volume = group["volume"].values.astype(float)
        amount_ln = np.log1p(group['amount'].values.astype(float))
        
        # [優化 2] 處理法人可能存在的空值，避免 NaN 蔓延
        vol_5 = SMA(volume, 5) * 5 + 1  # 加 1 避免分母為零
        t_net_5 = SMA(group['Trust_Net'].values.astype(float), 5) * 5
        f_net_5 = SMA(group['Foreign_Net'].values.astype(float), 5) * 5
        #投信方向盤 (保留正負號，判斷是作帳還是結帳)
        group['trust_in_5'] = t_net_5 / vol_5
        group['foreign_in_5'] = f_net_5 / vol_5
        group['t_f_sync'] = group['trust_in_5'] * group['foreign_in_5']

        # [優化 3] 將流動性移到這裡算，保證排序正確且效能最高
        group["amount_ma60"] = SMA(amount_ln, timeperiod=60)

        group["sma240"] = close / (SMA(close, timeperiod=240) + 1e-9)
        group["sma60"] = close / (SMA(close, timeperiod=60) + 1e-9)
        group["sma20"] = close / (SMA(close, timeperiod=20) + 1e-9)
        group["sma10"] = close / (SMA(close, timeperiod=10) + 1e-9)

        group["adx"] = ADX(high, low, close, timeperiod=14)
        
        try:
            group["kama"] = close / (KAMA(close, timeperiod=30) + 1e-9)
        except:
            group["kama"] = np.nan

        # _, _, macd_hist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        # group["macd_hist"] = macd_hist / (close + 1e-9)

        # upper, middle, lower = BBANDS(close, timeperiod=20)
        # group["bb_width"] = (upper - lower) / (middle + 1e-9)

        group["atr"] = ATR(high, low, close, timeperiod=14) / (close + 1e-9)

        group["amount20"] =SMA(amount_ln, 20)
        group["amount60"] =SMA(amount_ln, 60)

        time_idx = np.arange(len(close), dtype=float)
        group["r2_60"] = CORREL(close, time_idx, timeperiod=60) ** 2
        group["r2_20"] = CORREL(close, time_idx, timeperiod=20) ** 2

        group["breakout_20"] = close / (MAX(high, timeperiod=20) + 1e-9)
        group["breakout_240"] = close / (MAX(high, timeperiod=240) + 1e-9)

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
    df_revenue['yoy_ma3'] = df_revenue.groupby('id')['yoy'].transform(lambda x: x.rolling(window=3).mean())
    df_revenue["merge_key"] = df_revenue["date"].dt.to_period("M").apply(lambda x: x + 1)
    df["merge_key"] = df["date"].dt.to_period("M")

    df['id'] = df['id'].astype(str)
    df_revenue['id'] = df_revenue['id'].astype(str)

    print("正在執行主表與營收表合併...")
    df = pd.merge(
        df,
        df_revenue[["id", "merge_key", "mom", "yoy", "yoy_ma3"]],
        on=["id", "merge_key"],
        how="left",
    )

    df["yoy"] = df.groupby("id")["yoy"].ffill()
    df["mom"] = df.groupby("id")["mom"].ffill()
    df["yoy_ma3"] = df.groupby("id")["yoy_ma3"].ffill()
    df['yoy_ma3_bias'] = df['yoy'] - df['yoy_ma3']

    # [優化 4] 防禦機制：調降閾值以適應純科技股池
    date_counts = df["date"].value_counts()
    valid_dates = date_counts[date_counts > 250].index # 修改為 250 檔
    if len(valid_dates) == 0:
        print("錯誤：找不到完整的交易日資料。")
        return
    latest_date = valid_dates.max()
    print(f"提取最新「完整」交易日: {latest_date.date()} (共 {date_counts[latest_date]} 檔)")
    
    final_predict_df = df[df["date"] == latest_date].copy()
    
    del df
    gc.collect()

    final_predict_df["day_count"] = len(final_predict_df)
    final_predict_df["liq_rank"] = final_predict_df["amount_ma60"].rank(ascending=False)
    
    # [流動性]
    final_predict_df = final_predict_df[final_predict_df["liq_rank"] <= (final_predict_df["day_count"] / 3)]

    features = ["sma240","sma60","sma20","sma10",
                    "adx", "kama",
                    "breakout_20", "breakout_240",
                    "trust_in_5",
                    "r2_60","r2_20",
                    "mom", "yoy", "yoy_ma3_bias",
                    'amount20',"amount60","t_f_sync",
                    'atr'#v12要
                    ]

    print("正在執行特徵去極值處理...")
    final_predict_df = clean_outliers(final_predict_df, features)

    final_output = final_predict_df[["id", "name", "date", "close"] + features].dropna()
    final_output.to_csv(file_save_name, index=False, encoding="utf-8-sig")

    print(f"預測資料已生成: {file_save_name}，成功保留 {len(final_output)} 檔標的。")

if __name__ == "__main__":
    create_prediction()