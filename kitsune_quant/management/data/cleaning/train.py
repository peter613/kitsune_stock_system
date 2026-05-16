import os
from click import group
import pandas as pd
import numpy as np
from talib import *
from datetime import datetime
import gc
import config
from scipy.stats import entropy

def create_training(version='v14', years_10=True, stock_filter=True):
    file_save_name = config.FEATURES_DIR / f'train_{version}.csv'

    # ---------------------------------------------------------
    # 1. 讀取資料與預處理
    # ---------------------------------------------------------
    print("正在載入資料庫並執行型態淨化...")
    
    # Market 資料 (Parquet 格式)
    df = pd.read_parquet(config.DATABASE_DIR / 'market.parquet')
    
    # Revenue 與 Capital (CSV 格式，讀取時即指定型態防錯)
    df_revenue = pd.read_csv(config.DATABASE_DIR / 'revenue.csv', encoding='utf-8', dtype={'id': str})


    # 強制執行時間格式化 (解決 .dt 報錯)
    df['date'] = pd.to_datetime(df['date'])
    df_revenue['date'] = pd.to_datetime(df_revenue['date'])


    # 強制執行 ID 格式化 (解決 Merge 衝突與補零)
    for data in [df, df_revenue]:
        data['id'] = data['id'].astype(str).str.zfill(4)

    # ---------------------------------------------------------
    # 2. 預先排除 (過濾)
    # ---------------------------------------------------------
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
    # 3. 時間切割 (10年長度 + 緩衝期)
    # ---------------------------------------------------------
    if years_10:
        cutoff_date = df['date'].max() - pd.DateOffset(years=10)
        buffer_date = cutoff_date - pd.DateOffset(days=300)
        print(f"時間篩選：保留 {buffer_date.date()} 之後的數據以計算技術指標")
        df = df[df['date'] >= buffer_date].copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # 3.5 上市天數過濾 (IPO 隔離防禦)
    # ---------------------------------------------------------
    # 計算每個 id 的總資料筆數
    id_counts = df['id'].value_counts()
    # 找出大於等於 240 天的合格名單
    valid_ids = id_counts[id_counts >= 240].index
    print(f"剔除上市未滿 240 天的股票，數量: {len(id_counts) - len(valid_ids)}")
    # 直接用 isin 進行矩陣級別的過濾
    df = df[df['id'].isin(valid_ids)].copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. 特徵工程核心 (compute_features)
    # ---------------------------------------------------------
    def compute_features(group):
        close      = group['close'].values.astype(float)
        high       = group['high'].values.astype(float)
        low        = group['low'].values.astype(float)
        volume     = group['volume'].values.astype(float)
        amount     = group['amount'].values.astype(float)
        trust_net  = group['Trust_Net'].values.astype(float)
        foreign_net= group['Foreign_Net'].values.astype(float)

        # ── 均線 ──────────────────────────────────────────────
        sma200 = close / (SMA(close, timeperiod=200) + 1e-9)
        sma20  = close / (SMA(close, timeperiod=20)  + 1e-9)

        try:
            kama = close / (KAMA(close, timeperiod=30) + 1e-9)
        except Exception:
            kama = np.full(len(close), np.nan)

        # ── 技術指標 ──────────────────────────────────────────
        adx = ADX(high, low, close, timeperiod=14)
        #atr = ATR(high, low, close, timeperiod=14) / (close + 1e-9)

        # ── 成交值均線 ────────────────────────────────────────
        amount_ln  = np.log1p(amount)
        amount20 = SMA(amount_ln, 20)
        amount_ema_20 = EMA(amount_ln, 20)

        # ── 突破 / 支撐  (範圍 0~1)───────────────────────────────────────
        # 成交值突破
        amount_breakout_20  = amount / (SMA(amount,20) + 1e-9)
        # 價格突破 : 1代表在最高點
        breakout_20 = close / (MAX(close, timeperiod=20) + 1e-9)
        breakout_200 = close / (MAX(close, timeperiod=200) + 1e-9)
        # 價格築底 : 1代表無窮遠離底部
        support_20 = 1 - (MIN(close, timeperiod=20) / (close + 1e-9))
        support_200 = 1 - (MIN(close, timeperiod=200) / (close + 1e-9))

        # ── 法人籌碼 ──────────────────────────────────────────
        vol_5   = SMA(volume, 5) * 5 + 1
        vol_3   = SMA(volume, 3) * 3 + 1
        t_net_5 = SMA(trust_net,   5) * 5
        t_net_3 = SMA(trust_net,   3) * 3
        f_net_5 = SMA(foreign_net, 5) * 5

        trust_in_5   = t_net_5 / vol_5
        trust_in_3   = t_net_3 / vol_3
        #foreign_in_5 = f_net_5 / vol_5
        #t_f_sync_5     = trust_in_5 * foreign_in_5

        # ── 趨勢相關 ──────────────────────────────────────────
        time_idx = np.arange(len(close), dtype=float)
        r_20     = CORREL(close, time_idx, timeperiod=20)
        r_20_abs = np.abs(r_20)


        # ── 標籤 ──────────────────────────────────────────────
        close_shifted = np.empty(len(close))
        close_shifted[:-20] = close[20:]
        close_shifted[-20:] = np.nan
        return_20d = (close_shifted / (close + 1e-9)) - 1

        # ── 一次性寫回 DataFrame ──────────────────────────────
        group = group.copy()          # 避免 SettingWithCopyWarning
        cols = {
            'sma200'        : sma200,
            'sma20'         : sma20,
            'kama'          : kama,
            'adx'           : adx,
            'amount20'      : amount20,
            'amount_ema_20' : amount_ema_20,
            'a_breakout_20' : amount_breakout_20,
            'breakout_20'   : breakout_20,
            'breakout_200'  : breakout_200,
            'support_20'    : support_20,
            'support_200'   : support_200,
            'trust_in_5'    : trust_in_5,
            'trust_in_3'    : trust_in_3,
            'r_20_abs'          : r_20_abs,
            'return_20d'    : return_20d,
        }
        for col, arr in cols.items():
            group[col] = arr

        return group

    print("開始計算技術指標特徵...")
    df['safe_id'] = df['id'].copy() # 備份 ID
    #=====
    #超重要
    #=====
    df = df.groupby('id', group_keys=False).apply(compute_features)
    df['id'] = df['safe_id'] # 還原 ID
    df = df.reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. 多表併軌 (Revenue & Capital)
    # ---------------------------------------------------------
    print("正在合併營收...")
    
    # 營收 YOY 均線
    df_revenue = df_revenue.sort_values(['id', 'date'])
    df_revenue['yoy_ma3'] = df_revenue.groupby('id')['yoy'].transform(lambda x: x.rolling(5).mean())
    
    # 建立 Merge Key (遞延一個月對齊)
    df_revenue["merge_key"] = df_revenue["date"].dt.to_period("M").apply(lambda x: x + 1)
    df["merge_key"] = df["date"].dt.to_period("M")

    # 合併營收
    df = pd.merge(df, df_revenue[["id", "merge_key", "mom", "yoy", "yoy_ma3"]], on=["id", "merge_key"], how="left")

    # 數值填補與市值對數化
    for col in ["yoy", "mom", "yoy_ma3"]:
        df[col] = df.groupby("id")[col].ffill()
        
    df['yoy_ma3_bias'] = df['yoy'] - df['yoy_ma3']
    
    if years_10:
        df = df[df['date'] >= cutoff_date].copy()

    # ---------------------------------------------------------
    # 6. 換股日篩選與流動性過濾 (MS+9日邏輯)
    # ---------------------------------------------------------
    all_trading_days = pd.DataFrame({'actual_date': sorted(df['date'].unique())})
    target_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS') + pd.Timedelta(days=9)
    
    trade_dates_map = pd.merge_asof(pd.DataFrame({'target_date': target_dates}),
                                   all_trading_days,
                                   left_on='target_date', right_on='actual_date', direction='forward')
    
    final_df = df[df['date'].isin(trade_dates_map['actual_date'].unique())].copy()
    
    print("執行EMA成交值流動性過濾 (保留Top 1/3)...")
    final_df['liq_rank'] = final_df.groupby('date')['amount_ema_20'].rank(ascending=False, method='min')
    final_df['day_count'] = final_df.groupby('date')['id'].transform('count')
    final_df = final_df[final_df['liq_rank'] <= (final_df['day_count'] / 3)].copy()

    # ---------------------------------------------------------
    # 7. 去極值與標籤排名 (Labeling)
    # ---------------------------------------------------------
    print("進行特徵淨化 (Clipping) 與標籤生成...")
    growth_cols = ['mom', 'yoy', 'yoy_ma3', 'yoy_ma3_bias']
    for col in growth_cols:
        final_df[col] = final_df[col].clip(lower=-100.0, upper=100.0)

    final_df = final_df.dropna().reset_index(drop=True)

    # 截面排名標籤 (0~1)
    final_df['label'] = final_df.groupby('date')['return_20d'].rank(pct=True)

    # ---------------------------------------------------------
    # 8. 儲存結果
    # ---------------------------------------------------------
    final_df.to_csv(file_save_name, index=False, encoding='utf-8-sig')
    print(f"訓練集建置成功！檔案位置: {file_save_name}")
    print(f"最新日期: {final_df['date'].max().date()} | 股票數: {final_df['id'].nunique()} | 樣本總數: {len(final_df)}")

if __name__ == "__main__":
    create_training(version='v28', years_10=True, stock_filter=True)
