import os
import pandas as pd
import numpy as np
from talib import *
import gc
import config
from scipy.stats import entropy

# --- 設定區 ---
def create_prediction(stock_filter=True, specified_date=None):
    # 1. 檔案路徑更新
    file_name = config.MARKET_DATA_DIR / 'market.parquet'
    file_name_revenue = config.MARKET_DATA_DIR / 'revenue.csv'
    file_save_name = config.STOCK_FEATURES_DIR / f'pred.csv'

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

    # 讀取 ETF 大盤資料 (0050)
    etf_path = config.MARKET_DATA_DIR / 'etf.csv'
    if os.path.exists(etf_path):
        df_etf = pd.read_csv(etf_path)
        if '0050' in df_etf.columns:
            df_etf = df_etf[['date', '0050']].rename(columns={'0050': 'etf_close'})
            df_etf['date'] = pd.to_datetime(df_etf['date'])
            all_dates = pd.DataFrame({'date': df['date'].unique()}).sort_values('date')
            all_dates = pd.merge(all_dates, df_etf, on='date', how='left')
            all_dates['etf_close'] = all_dates['etf_close'].ffill().bfill()
            df = pd.merge(df, all_dates, on='date', how='left')
        else:
            df['etf_close'] = np.nan
    else:
        df['etf_close'] = np.nan

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
                    df[col] = df[col].clip(-100.0, 100.0)
                # else:
                #     lower = df[col].quantile(0.01)
                #     upper = df[col].quantile(0.99)
                #     df[col] = df[col].clip(lower, upper)
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
        trust_net = group['Trust_Net'].values.astype(float)

        try:
            kama = close / (KAMA(close, timeperiod=30) + 1e-9)
        except Exception:
            kama = np.full(len(close), np.nan)

        adx = ADX(high, low, close, timeperiod=14)

        amount_20 = SMA(amount_ln, timeperiod=20)
        amount_ema_20 = EMA(amount_ln, timeperiod=20)

        # ── 突破 / 支撐 / 區間箱型 (Box) ──────────────────
        breakout_200 = close / (MAX(close, timeperiod=200) + 1e-9)
        
        # 近期天花板 (20日最高收盤價)
        box_res_20 = MAX(close, timeperiod=20)
        # 近期地板 (20日最低收盤價)
        box_sup_20 = MIN(close, timeperiod=20)
        
        # 距離近期天花板多遠？ (>1 代表真突破)
        resistance_20 = close / (box_res_20 + 1e-9)
        # 距離近期地板多遠？ (接近1 代表回踩支撐)
        support_20 = close / (box_sup_20 + 1e-9)

        # ── 法人籌碼 ──────────────────────────────────────────
        trust_period = 10
        is_zero_array = np.where(trust_net == 0, 1.0, 0.0)
        zero_days_in_period = SMA(is_zero_array, trust_period) * trust_period
        trust_in = np.where(zero_days_in_period >= 7, 0.0, 1.0)

        # ── 波動與量能 ──────────────────────────────
        vol_sma_20 = SMA(volume, timeperiod=20)
        vol_surge = volume / (vol_sma_20 + 1e-9)
        vol_surge_signed = volume / (vol_sma_20 + 1e-9) * np.where((trust_net + foreign_net) > 0, 1.0, -1.0)

        close_past = np.empty(len(close))
        close_past[20:] = close[:-20]
        close_past[:20] = np.nan
        hist_return_20 = (close / (close_past + 1e-9)) - 1

        close_past_60 = np.empty(len(close))
        close_past_60[60:] = close[:-60]
        close_past_60[:60] = np.nan
        hist_return_60 = (close / (close_past_60 + 1e-9)) - 1

        group = group.copy()
        cols = {
            'kama'          : kama,
            'adx'           : adx,
            'amount_20'     : amount_20,
            'amount_ema_20' : amount_ema_20,
            'breakout_200'  : breakout_200,
            'resistance_20' : resistance_20,
            'support_20'    : support_20,
            'trust_in'      : trust_in,
            'hist_return_20': hist_return_20,
            'hist_return_60': hist_return_60,
            'vol_surge'     : vol_surge,
            'vol_surge_signed': vol_surge_signed
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
    
    # 防禦：避免負數/零值營收造成除法符號反轉或爆炸
    # 台股有 338 筆負數營收（銷貨退回）與 1542 筆零值，必須鎖定下限
    df_revenue['month_revenue'] = df_revenue['month_revenue'].clip(lower=1.0)

    rolling_max_12 = df_revenue.groupby('id')['month_revenue'].transform(lambda x: x.rolling(window=12, min_periods=12).max())
    df_revenue['rev_breakout_12'] = df_revenue['month_revenue'] / rolling_max_12
    
    df_revenue['rev_bias_3'] = df_revenue['month_revenue'] / (df_revenue.groupby('id')['month_revenue'].transform(lambda x: x.rolling(window=3).mean()) + 1e-9)
    df_revenue["merge_key"] = df_revenue["date"].dt.to_period("M").apply(lambda x: x + 1)
    df["merge_key"] = df["date"].dt.to_period("M")

    df['id'] = df['id'].astype(str)
    df_revenue['id'] = df_revenue['id'].astype(str)

    print("正在執行主表與營收表合併...")
    df = pd.merge(
        df,
        df_revenue[["id", "merge_key", "mom", "yoy", "month_revenue", "rev_breakout_12", "rev_bias_3"]],
        on=["id", "merge_key"],
        how="left",
    )

    df["yoy"] = df.groupby("id")["yoy"].ffill()
    df["mom"] = df.groupby("id")["mom"].ffill()
    df["month_revenue"] = df.groupby("id")["month_revenue"].ffill()
    df["rev_breakout_12"] = df.groupby("id")["rev_breakout_12"].ffill()
    df["rev_bias_3"] = df.groupby("id")["rev_bias_3"].ffill()

    # [優化 4] 防禦機制：調降閾值以適應純科技股池
    date_counts = df["date"].value_counts()
    valid_dates = date_counts[date_counts > 250].index # 修改為 250 檔
    if len(valid_dates) == 0:
        print("錯誤：找不到完整的交易日資料。")
        return

    if specified_date is not None:
        specified_date = pd.to_datetime(specified_date).normalize()
        if specified_date not in valid_dates:
            raise ValueError(f"指定日期 {specified_date.date()} 不存在於有效交易日資料中。")
        selected_date = specified_date
        print(f"使用指定日期進行預測：{selected_date.date()} (共 {date_counts[selected_date]} 檔)")
    else:
        selected_date = valid_dates.max()
        print(f"提取最新股票交易日: {selected_date.date()} (共 {date_counts[selected_date]} 檔)")

    print(f"提取最新營收月份: {df_revenue['date'].dt.strftime('%Y-%m').max()}")
    final_predict_df = df[df["date"] == selected_date].copy()
    
    del df
    gc.collect()

    
    print("處理截面特徵 (Cross-Sectional)...")
    def zero_centered_rank(x):
        res = pd.Series(0.0, index=x.index)
        pos = x > 0
        neg = x < 0
        if pos.sum() > 0:
            res[pos] = x[pos].rank(pct=True)
        if neg.sum() > 0:
            res[neg] = x[neg].rank(ascending=False, pct=True) * -1.0
        return res

    final_predict_df["amount_rank"] = final_predict_df["amount_20"].rank(pct=True)
    final_predict_df["rs_rank_20"] = final_predict_df["hist_return_20"].rank(pct=True)
    final_predict_df["rs_rank_60"] = final_predict_df["hist_return_60"].rank(pct=True)

    print("執行EMA成交值流動性過濾 (保留Top 1/3)...")
    final_predict_df["day_count"] = len(final_predict_df)
    final_predict_df["liq_rank"] = final_predict_df["amount_ema_20"].rank(ascending=False, method='min')
    final_predict_df = final_predict_df[final_predict_df["liq_rank"] <= (final_predict_df["day_count"] / 3)]

    print("計算高流動性科技股專屬廣度 (Tech Market Breadth)...")
    final_predict_df['market_kama_mean'] = final_predict_df['kama'].mean()
    final_predict_df['market_adx_mean'] = final_predict_df['adx'].mean()
    final_predict_df['market_res_20_mean'] = final_predict_df['resistance_20'].mean()

    AMOUNT_THRESHOLD = 100000 # 當天成交值 > 1.0億 (設為0則不過濾)
    REVENUE_THRESHOLD = 100000 # 當月營收 > 1.0億 (設為0則不過濾)
    
    if AMOUNT_THRESHOLD > 0:
        final_predict_df = final_predict_df[final_predict_df['amount'] >= AMOUNT_THRESHOLD].reset_index(drop=True)
        print(f'執行成交值流動性過濾> {AMOUNT_THRESHOLD}(千)...')

    if REVENUE_THRESHOLD > 0 and 'month_revenue' in final_predict_df.columns:
        final_predict_df = final_predict_df[final_predict_df['month_revenue'] >= REVENUE_THRESHOLD].reset_index(drop=True)
        print(f'執行營收過濾> {REVENUE_THRESHOLD}(千)...')

    features = [
        # 趨勢強度
        "kama", "adx",

        # 宏觀環境
        "market_kama_mean", "market_adx_mean", "market_res_20_mean",

        # 流動性與動能 
        'hist_return_20', 'hist_return_60', 'vol_surge',
        'vol_surge_signed',
        
        # 籌碼0與1
        "trust_in",

        # 基本面
        "yoy", "rev_bias_3", 'rev_breakout_12',

        # 位階
        "breakout_200",
        'resistance_20', 'support_20', 
    ]

    print("正在執行特徵去極值處理...")
    final_predict_df = clean_outliers(final_predict_df, features)

    final_output = final_predict_df[["id", "name", "date", "close"] + features].dropna()
    final_output.to_csv(file_save_name, index=False, encoding="utf-8-sig")

    print(f"預測資料已生成: {file_save_name}\n成功保留 {len(final_output)} 檔標的。")

if __name__ == "__main__":
    create_prediction(specified_date='20260713')