import os
from click import group
import pandas as pd
import numpy as np
from talib import *
from datetime import datetime
import gc
import config

def create_training(version='v14', years_10=True, stock_filter=True):
    file_save_name = config.STOCK_FEATURES_DIR / f'train_{version}.csv'

    # ---------------------------------------------------------
    # 1. 讀取資料與預處理
    # ---------------------------------------------------------
    print("正在載入資料庫並執行型態淨化...")
    
    # Market 資料 (Parquet 格式)
    df = pd.read_parquet(config.MARKET_DATA_DIR / 'market.parquet')
    
    # Revenue 與 Capital (CSV 格式，讀取時即指定型態防錯)
    df_revenue = pd.read_csv(config.MARKET_DATA_DIR / 'revenue.csv', encoding='utf-8', dtype={'id': str})


    # 強制執行時間格式化 (解決 .dt 報錯)
    df['date'] = pd.to_datetime(df['date'])
    df_revenue['date'] = pd.to_datetime(df_revenue['date'])

    # 讀取 ETF 大盤資料 (0050)
    etf_path = config.MARKET_DATA_DIR / 'etf.csv'
    if os.path.exists(etf_path):
        df_etf = pd.read_csv(etf_path)
        if '0050' in df_etf.columns:
            df_etf = df_etf[['date', '0050']].rename(columns={'0050': 'etf_close'})
            df_etf['date'] = pd.to_datetime(df_etf['date'])
            
            # 確保大盤資料對齊所有的交易日，並且 ffill 避免 NaN 毒化 TA-Lib
            all_dates = pd.DataFrame({'date': df['date'].unique()}).sort_values('date')
            all_dates = pd.merge(all_dates, df_etf, on='date', how='left')
            all_dates['etf_close'] = all_dates['etf_close'].ffill().bfill()
            
            df = pd.merge(df, all_dates, on='date', how='left')
        else:
            df['etf_close'] = np.nan
    else:
        df['etf_close'] = np.nan


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
        get_white_list = pd.read_csv(config.STOCK_WHITE_LIST_DIR / 'white_list.csv', encoding='utf-8-sig')
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
        etf_close  = group['etf_close'].values.astype(float) if 'etf_close' in group else np.full(len(close), np.nan)

        # ── 均線 ──────────────────────────────────────────────
        sma200 = close / (SMA(close, timeperiod=200) + 1e-9)
        sma20  = close / (SMA(close, timeperiod=20)  + 1e-9)
        #sma5  = close / (SMA(close, timeperiod=5)  + 1e-9)

        # ── 計算個股kama ──────────
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
        # 價格突破 : 1代表在最高點
        breakout_20 = close / (MAX(close, timeperiod=20) + 1e-9)
        breakout_200 = close / (MAX(close, timeperiod=200) + 1e-9)
        # 價格築底 : 1代表無窮遠離底部
        support_20 = 1 - (MIN(close, timeperiod=20) / (close + 1e-9))
        support_200 = 1 - (MIN(close, timeperiod=200) / (close + 1e-9))

        # ── 法人籌碼 ──────────────────────────────────────────
        trust_period = 10
        vol_period   = SMA(volume, trust_period) * trust_period + 1
        t_net_period = SMA(trust_net, trust_period) * trust_period
        trust_in_10 = t_net_period / vol_period

        close_pct = np.zeros_like(close)
        close_pct[0] = np.nan
        close_pct[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)

        # ── 趨勢相關 ──────────────────────────────────────────
        time_idx = np.arange(len(close), dtype=float)
        r_20     = CORREL(close, time_idx, timeperiod=20)
        r_20_abs = np.abs(r_20)

        # ── 1. 計算個股每日報酬率 ──────────────────────────────
        daily_return = np.empty(len(close))
        daily_return[0] = 0
        daily_return[1:] = (close[1:] / (close[:-1] + 1e-9)) - 1

        # ── 計算 20 日實際報酬 ──────────────────
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
            'breakout_20'   : breakout_20,
            'breakout_200'  : breakout_200,
            'support_20'    : support_20,
            'support_200'   : support_200,
            'trust_in_10'   : trust_in_10,
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
    
    df_revenue = df_revenue.sort_values(['id', 'date'])

    # 營收突破/創高
    rolling_max_12 = df_revenue.groupby('id')['month_revenue'].transform(lambda x: x.rolling(window=12, min_periods=12).max())
    df_revenue['rev_breakout_12'] = df_revenue['month_revenue'] / rolling_max_12

    # 低熵營收動能解法：單月絕對金額 / 過去N個月平均金額
    df_revenue['rev_bias_3'] = df_revenue['month_revenue'] / (df_revenue.groupby('id')['month_revenue'].transform(lambda x: x.rolling(window=3).mean()) + 1e-9)
    df_revenue['rev_bias_6'] = df_revenue['month_revenue'] / (df_revenue.groupby('id')['month_revenue'].transform(lambda x: x.rolling(window=6).mean()) + 1e-9)
    df_revenue['rev_bias_12'] = df_revenue['month_revenue'] / (df_revenue.groupby('id')['month_revenue'].transform(lambda x: x.rolling(window=12).mean()) + 1e-9)

    # 合併營收
    df_revenue["merge_key"] = df_revenue["date"].dt.to_period("M").apply(lambda x: x + 1)
    df["merge_key"] = df["date"].dt.to_period("M")
    
    df = pd.merge(
        df,
        df_revenue[["id", "merge_key", "mom", "yoy", "month_revenue", 'rev_breakout_12', 'rev_bias_3', 'rev_bias_6', 'rev_bias_12']],
        on=["id", "merge_key"],
        how="left",
    )


    # 數值填補與市值對數化
    for col in ["yoy", "mom", "month_revenue", 'rev_breakout_12', 'rev_bias_3', 'rev_bias_6', 'rev_bias_12']:
        df[col] = df.groupby("id")[col].ffill()

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
    growth_cols = ['mom', 'yoy']
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
    create_training(version='v36', years_10=True, stock_filter=True)
