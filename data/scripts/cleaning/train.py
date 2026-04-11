import os
import pandas as pd
import numpy as np
from talib import *
from datetime import datetime
import gc
import config
from data.scripts.cleaning.black_list import *

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
    # 2. 預先排除 (黑名單過濾)
    # ---------------------------------------------------------
    if stock_filter:
        print("正在過濾低成長/非電子產業股票...")
        # ---------------------------------------------------------
        # 非電子/傳統產業黑名單 (排除低成長、缺乏科技動能與投信作帳特性的類股)
        # ---------------------------------------------------------
        get_black_prefixes = black_prefixes()
        get_black_ids = black_ids()

        df = df[~df['id'].str.startswith(get_black_prefixes) & ~df['id'].isin(get_black_ids)].copy()
        df_revenue = df_revenue[~df_revenue['id'].str.startswith(get_black_prefixes) & ~df_revenue['id'].isin(get_black_ids)].copy()

    # ---------------------------------------------------------
    # 3. 時間切割 (10年長度 + 緩衝期)
    # ---------------------------------------------------------
    if years_10:
        cutoff_date = df['date'].max() - pd.DateOffset(years=10)
        buffer_date = cutoff_date - pd.DateOffset(days=300)
        print(f"時間篩選：保留 {buffer_date.date()} 之後的數據以計算技術指標")
        df = df[df['date'] >= buffer_date].copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. 特徵工程核心 (compute_features)
    # ---------------------------------------------------------
    def compute_features(group):
        close = group['close'].values.astype(float)
        high = group['high'].values.astype(float)
        low = group['low'].values.astype(float)
        volume = group['volume'].values.astype(float)
        
        # 使用 log1p 防止零成交額錯誤
        amount_ln = np.log1p(group['amount'].values.astype(float))

        group['sma240'] = close / (SMA(close, timeperiod=240) + 1e-9)
        group['sma60']  = close / (SMA(close, timeperiod=60) + 1e-9)
        group['sma20']  = close / (SMA(close, timeperiod=20) + 1e-9)
        group['sma10']  = close / (SMA(close, timeperiod=10) + 1e-9)

        try:
            group['kama'] = close / (KAMA(close, timeperiod=30) + 1e-9)
        except:
            group['kama'] = np.nan

        group['adx'] = ADX(high, low, close, timeperiod=14)
        
        # macd, _, macd_hist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        # group['macd_hist'] = macd_hist / (close + 1e-9)
        group['atr'] = ATR(high, low, close, timeperiod=14) / (close + 1e-9)
        
        # 成交量
        group['amount20'] =  SMA(amount_ln, 20)
        group['amount60'] =  SMA(amount_ln, 60)

        group['breakout_20'] = close / (MAX(high, timeperiod=20) + 1e-9)
        group['breakout_60'] = close / (MAX(high, timeperiod=60) + 1e-9)
        group['breakout_240'] = close / (MAX(high, timeperiod=240) + 1e-9)

        vol_5 = SMA(volume, 5) * 5 + 1  # 加 1 避免分母為零
        t_net_5 = SMA(group['Trust_Net'].values.astype(float), 5) * 5
        f_net_5 = SMA(group['Foreign_Net'].values.astype(float), 5) * 5
        #投信方向盤 (保留正負號，判斷是作帳還是結帳)
        group['trust_in_5'] = t_net_5 / vol_5
        group['foreign_in_5'] = f_net_5 / vol_5
        group['t_f_sync'] = group['trust_in_5'] * group['foreign_in_5']
        
        time_idx = np.arange(len(close), dtype=float)
        group['r2_60'] = CORREL(close, time_idx, timeperiod=60) ** 2
        group['r2_20'] = CORREL(close, time_idx, timeperiod=20) ** 2

        # upper, middle, lower = BBANDS(close, timeperiod=20)
        # group['bb_width'] = (upper - lower) / (middle + 1e-9)

        # Label: 20日後報酬率 (Ranker 訓練目標)
        group['return_20d'] = (group['close'].shift(-20) / (group['close'] + 1e-9)) - 1

        #連續漲停次數
        # group['is_limit_up'] = np.where(group['close'] >= group['close'].shift(1) * 1.095, 1, 0)
        # group['limit_up_count_5d'] = group['is_limit_up'].rolling(window=5, min_periods=1).sum()
        
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
    df_revenue['yoy_ma3'] = df_revenue.groupby('id')['yoy'].transform(lambda x: x.rolling(3).mean())
    
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
    
    print("執行流動性排名過濾 (Top 1/3)...")
    final_df['liq_rank'] = final_df.groupby('date')['amount20'].rank(ascending=False, method='min')
    final_df['day_count'] = final_df.groupby('date')['id'].transform('count')
    final_df = final_df[final_df['liq_rank'] <= (final_df['day_count'] / 3)].copy()

    # ---------------------------------------------------------
    # 7. 去極值與標籤排名 (Labeling)
    # ---------------------------------------------------------
    print("進行特徵淨化 (Clipping) 與標籤生成...")
    growth_cols = ['mom', 'yoy', 'yoy_ma3', 'yoy_ma3_bias']
    for col in growth_cols:
        final_df[col] = final_df[col].clip(lower=-100.0, upper=100.0)

    other_features = [ 'trust_in_5', 'amount60', 'breakout_20', 'breakout_60']
    for col in other_features:
        lower = final_df.groupby('date')[col].transform(lambda x: x.quantile(0.01))
        upper = final_df.groupby('date')[col].transform(lambda x: x.quantile(0.99))
        final_df[col] = final_df[col].clip(lower=lower, upper=upper)

    final_df = final_df.dropna().reset_index(drop=True)
    # 截面排名標籤 (0~1)
    final_df['label'] = final_df.groupby('date')['return_20d'].rank(pct=True)

    # ---------------------------------------------------------
    # 8. 儲存結果
    # ---------------------------------------------------------
    final_df.to_csv(file_save_name, index=False, encoding='utf-8-sig')
    print(f"訓練集建置成功！檔案位置: {file_save_name}")
    print(f"樣本總數: {len(final_df)} | 最新日期: {final_df['date'].max().date()}")

if __name__ == "__main__":
    create_training(version='v18', years_10=True, stock_filter=True)