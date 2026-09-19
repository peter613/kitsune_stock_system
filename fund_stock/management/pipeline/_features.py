import sys
from pathlib import Path

# 確保定位專案根目錄
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import pandas as pd
import numpy as np
from talib import *
import gc
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import config

# =====================================================================
# 單一真實來源 (Single Source of Truth)
# 全專案唯一特徵清單：增減特徵只要在此修改，模型自動同步！
# =====================================================================
FEATURE_COLS = [
    # 趨勢強度
    "kama", "adx",

    # 大盤濾網與生態 (Tech Market Breadth)
    "market_kama_mean", "market_adx_mean",

    # 動能排名與量能
    "vol_surge_signed", "hist_return_60",

    # 基本面
    "yoy", "rev_breakout_12", "rev_bias_3",

    # 位階
    "breakout_200", "resistance_20", "support_20",

    # K 線形態結構 (當沖拋壓 vs 主力護盤)
    "upper_ratio_20",
    #"lower_ratio_20",
]


def compute_technical_features(group):
    """
    純個股技術指標與報酬計算算子 (向量化加速)
    """
    open_p      = group['open'].values.astype(float)
    close       = group['close'].values.astype(float)
    high        = group['high'].values.astype(float)
    low         = group['low'].values.astype(float)
    volume      = group['volume'].values.astype(float)
    amount      = group['amount'].values.astype(float)
    trust_net   = group['Trust_Net'].values.astype(float)
    foreign_net = group['Foreign_Net'].values.astype(float)
    etf_close   = group['etf_close'].values.astype(float) if 'etf_close' in group else np.full(len(close), np.nan)

    # ── 均線 ──────────────────────────────────────────────
    sma60 = close / (SMA(close, timeperiod=60) + 1e-9)
    sma120 = close / (SMA(close, timeperiod=120) + 1e-9)

    # ── 個股 KAMA ─────────────────────────────────────────
    try:
        kama = close / (KAMA(close, timeperiod=30) + 1e-9)
    except Exception:
        kama = np.full(len(close), np.nan)

    # ── 技術指標 ──────────────────────────────────────────
    adx = ADX(high, low, close, timeperiod=14)

    # ── 成交值 ───────────────────────────────────────────
    amount_ln = np.log1p(amount)
    amount_20 = SMA(amount_ln, timeperiod=20)
    amount_ema_20 = EMA(amount_ln, timeperiod=20)

    # ── 突破 / 支撐 / 區間箱型 (Box) ─────────────────────
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

    # ── 波動與量能 ───────────────────────────────────────
    vol_sma_20 = SMA(volume, timeperiod=20)
    vol_surge = volume / (vol_sma_20 + 1e-9)
    vol_surge_signed = volume / (vol_sma_20 + 1e-9) * np.where((trust_net + foreign_net) > 0, 1.0, -1.0)

    # ── 計算 20 日實際報酬 (預測標籤用) ───────────────────
    close_shifted = np.empty(len(close))
    close_shifted[:-20] = close[20:]
    close_shifted[-20:] = np.nan
    return_20d = (close_shifted / (close + 1e-9)) - 1

    # ── 計算過去 20 日與 60 日歷史報酬 ───────────────────
    close_past = np.empty(len(close))
    close_past[20:] = close[:-20]
    close_past[:20] = np.nan
    hist_return_20 = (close / (close_past + 1e-9)) - 1

    close_past_60 = np.empty(len(close))
    close_past_60[60:] = close[:-60]
    close_past_60[:60] = np.nan
    hist_return_60 = (close / (close_past_60 + 1e-9)) - 1

    # ── 20日 K 線影線結構 (當沖拋壓 vs 主力護盤吸籌) ─────────
    rng = (high - low) + 1e-9
    # 上影線佔比 (當沖客獲利倒貨壓力，衝高但守不住): high - max(open, close)
    upper_shadow = high - np.maximum(open_p, close)
    upper_ratio_20 = SMA(upper_shadow / rng, timeperiod=20)

    # 下影線佔比 (主力盤中下檔承接力，殺下去強拉回): min(open, close) - low
    lower_shadow = np.minimum(open_p, close) - low
    #lower_ratio_20 = SMA(lower_shadow / rng, timeperiod=20)

    # ── 一次性寫回 DataFrame ──────────────────────────────
    group = group.copy()
    cols = {
        'kama': kama,
        'adx': adx,
        'sma60': sma60,
        'sma120': sma120,
        'amount_20': amount_20,
        'amount_ema_20': amount_ema_20,
        'breakout_200': breakout_200,
        'resistance_20': resistance_20,
        'support_20': support_20,
        'trust_in': trust_in,
        'hist_return_20': hist_return_20,
        'hist_return_60': hist_return_60,
        'vol_surge': vol_surge,
        'vol_surge_signed': vol_surge_signed,
        'upper_ratio_20': upper_ratio_20,
        #'lower_ratio_20': lower_ratio_20,
        'return_20d': return_20d,
    }
    for col, arr in cols.items():
        group[col] = arr

    return group


def process_and_merge_revenue(df, df_revenue):
    """
    營收特徵計算與跨月對齊合併算子
    """
    df_revenue = df_revenue.sort_values(['id', 'date']).copy()
    # 避免相除問題，將月營收設定下限為 1 元（台股有負數與零值營收）
    df_revenue['month_revenue'] = df_revenue['month_revenue'].clip(lower=1.0)

    # 營收突破/創高 (12個月創高比率)
    rolling_max_12 = df_revenue.groupby('id')['month_revenue'].transform(lambda x: x.rolling(window=12, min_periods=12).max())
    df_revenue['rev_breakout_12'] = df_revenue['month_revenue'] / rolling_max_12

    # 低熵營收動能解法：單月絕對金額 / 過去 3 個月平均金額
    df_revenue['rev_bias_3'] = df_revenue['month_revenue'] / (df_revenue.groupby('id')['month_revenue'].transform(lambda x: x.rolling(window=3).mean()) + 1e-9)

    # 近 3 個月平均營收成長 YoY
    df_revenue['yoy_mean_3'] = df_revenue.groupby('id')['yoy'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # 合併營收 (營收公布月份為次月，merge_key = date + 1 Month)
    df_revenue["merge_key"] = df_revenue["date"].dt.to_period("M").apply(lambda x: x + 1)
    df["merge_key"] = df["date"].dt.to_period("M")

    df = pd.merge(
        df,
        df_revenue[["id", "merge_key", "mom", "yoy", "yoy_mean_3", "month_revenue", 'rev_breakout_12', 'rev_bias_3']],
        on=["id", "merge_key"],
        how="left",
    )

    # 數值向前填補
    for col in ["yoy", "yoy_mean_3", "mom", "month_revenue", 'rev_breakout_12', 'rev_bias_3']:
        df[col] = df.groupby("id")[col].ffill()

    return df


def build_dataset(mode='train', version='v51', stock_filter=True, years_10=True, specified_date=None):
    """
    全系統統一資料集建置管線
    :param mode: 'train' | 'bt' | 'pred'
    :param version: 模型/特徵版本 (如 'v51')
    :param stock_filter: 是否套用白名單與電信黑名單過濾
    :param years_10: 是否限制過去 10 年資料 (train/bt 適用)
    :param specified_date: pred 模式專用指定預測日期 (如 '20260910')
    """
    if mode == 'train':
        file_save_name = config.STOCK_TRAIN_FEATURES_DIR / f'train_{version}.csv'
        desc = "訓練特徵集"
    elif mode == 'bt':
        file_save_name = config.STOCK_BT_FEATURES_DIR / f'bt_{version}.csv'
        desc = "回測特徵集"
    elif mode == 'pred':
        file_save_name = config.STOCK_FEATURES_DIR / 'pred.csv'
        desc = "預測特徵集"
    else:
        raise ValueError(f"未知的管線模式: {mode} (僅支援 'train', 'bt', 'pred')")

    print(f"[{mode.upper()}] 正在載入資料庫並執行型態淨化 ({desc})...")
    df = pd.read_parquet(config.MARKET_DATA_DIR / 'market.parquet')
    df_revenue = pd.read_csv(config.MARKET_DATA_DIR / 'revenue.csv', encoding='utf-8', dtype={'id': str})

    # 時間與 ID 格式標準化
    df['date'] = pd.to_datetime(df['date'])
    df_revenue['date'] = pd.to_datetime(df_revenue['date'])
    df['id'] = df['id'].astype(str).str.zfill(4)
    df_revenue['id'] = df_revenue['id'].astype(str).str.zfill(4)

    # 讀取 0050 ETF 基準資料 (供 Beta 與抗跌因子使用)
    etf_path = config.MARKET_DATA_DIR / 'etf.csv'
    if os.path.exists(etf_path):
        df_etf = pd.read_csv(etf_path)
        if '0050' in df_etf.columns:
            df_etf = df_etf[['date', '0050']].rename(columns={'0050': 'etf_close'})
            df_etf['date'] = pd.to_datetime(df_etf['date'])
            all_dates = pd.DataFrame({'date': sorted(df['date'].unique())})
            all_dates = pd.merge(all_dates, df_etf, on='date', how='left')
            all_dates['etf_close'] = all_dates['etf_close'].ffill().bfill()
            df = pd.merge(df, all_dates, on='date', how='left')
        else:
            df['etf_close'] = np.nan
    else:
        df['etf_close'] = np.nan

    # ---------------------------------------------------------
    # 1. 白名單與黑名單過濾
    # ---------------------------------------------------------
    if stock_filter:
        print("正在保留白名單與剔除黑名單...")
        white_list_path = config.STOCK_WHITE_LIST_DIR / 'white_list.csv'
        if not white_list_path.exists():
            white_list_path = config.FUND_STOCK_MANAGEMENT_DIR / 'data' / 'white_list.csv'
        get_white_list = pd.read_csv(white_list_path, encoding='utf-8-sig')
        white_ids = get_white_list['id'].astype(str).str.zfill(4)

        df = df[df['id'].isin(white_ids)].copy()
        df_revenue = df_revenue[df_revenue['id'].isin(white_ids)].copy()

        black_ids = ['2412', '3045', '4904']
        df = df[~df['id'].isin(black_ids)].copy()
        df_revenue = df_revenue[~df_revenue['id'].isin(black_ids)].copy()

    # ---------------------------------------------------------
    # 2. 時間區間裁切 (歷史暖機緩衝)
    # ---------------------------------------------------------
    if mode == 'pred':
        latest_date_in_db = df["date"].max()
        cutoff_date = latest_date_in_db - pd.DateOffset(days=600)
        print(f"預測模式：裁切歷史資料 (保留 {cutoff_date.date()} 之後以供指標暖機)...")
        df = df[df["date"] >= cutoff_date].copy().reset_index(drop=True)
    else:
        if years_10:
            cutoff_date = df['date'].max() - pd.DateOffset(years=10)
            buffer_date = cutoff_date - pd.DateOffset(days=300)
            print(f"時間篩選：保留 {buffer_date.date()} 之後的數據以計算技術指標")
            df = df[df['date'] >= buffer_date].copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. 上市天數過濾 (IPO 隔離防禦 >= 240天)
    # ---------------------------------------------------------
    id_counts = df['id'].value_counts()
    valid_ids = id_counts[id_counts >= 240].index
    print(f"剔除上市未滿 240 天的股票，數量: {len(id_counts) - len(valid_ids)}")
    df = df[df['id'].isin(valid_ids)].copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. 技術指標計算 (compute_technical_features)
    # ---------------------------------------------------------
    print("開始計算技術指標特徵...")
    df['safe_id'] = df['id'].copy()
    df = df.groupby('id', group_keys=False).apply(compute_technical_features)
    df['id'] = df['safe_id']
    df = df.reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. 合併營收
    # ---------------------------------------------------------
    print("正在合併營收...")
    df = process_and_merge_revenue(df, df_revenue)

    if mode in ('train', 'bt') and years_10:
        df = df[df['date'] >= cutoff_date].copy()

    # ---------------------------------------------------------
    # 6. 模式分流處理
    # ---------------------------------------------------------
    if mode == 'pred':
        date_counts = df["date"].value_counts()
        valid_dates = date_counts[date_counts > 250].index
        if len(valid_dates) == 0:
            raise RuntimeError("錯誤：找不到足夠股票數量的有效交易日資料。")

        if specified_date is not None:
            selected_date = pd.to_datetime(specified_date).normalize()
            if selected_date not in valid_dates:
                raise ValueError(f"指定日期 {selected_date.date()} 不存在於有效交易日資料中。")
            print(f"使用指定日期進行預測：{selected_date.date()} (共 {date_counts[selected_date]} 檔)")
        else:
            selected_date = valid_dates.max()
            print(f"提取最新股票交易日: {selected_date.date()} (共 {date_counts[selected_date]} 檔)")

        final_predict_df = df[df["date"] == selected_date].copy()
        del df
        gc.collect()

        print("處理截面特徵 (Cross-Sectional)...")
        final_predict_df["amount_rank"] = final_predict_df["amount_20"].rank(pct=True)
        final_predict_df["rs_rank_20"] = final_predict_df["hist_return_20"].rank(pct=True)
        final_predict_df["rs_rank_60"] = final_predict_df["hist_return_60"].rank(pct=True)

        for col in ['mom', 'yoy', 'yoy_mean_3']:
            if col in final_predict_df.columns:
                final_predict_df[col] = final_predict_df[col].clip(lower=-100.0, upper=100.0)

        print("執行EMA成交值流動性過濾 (保留Top 1/3)...")
        final_predict_df["day_count"] = len(final_predict_df)
        final_predict_df["liq_rank"] = final_predict_df["amount_ema_20"].rank(ascending=False, method='min')
        final_predict_df = final_predict_df[final_predict_df["liq_rank"] <= (final_predict_df["day_count"] / 3)].copy()

        print("計算高流動性科技股專屬廣度 (Tech Market Breadth)...")
        final_predict_df['market_kama_mean'] = final_predict_df['kama'].mean()
        final_predict_df['market_adx_mean'] = final_predict_df['adx'].mean()
        final_predict_df['market_resistance_mean'] = final_predict_df['resistance_20'].mean()

        AMOUNT_THRESHOLD = 100000  # 1.0 億
        REVENUE_THRESHOLD = 100000 # 1.0 億
        if AMOUNT_THRESHOLD > 0:
            final_predict_df = final_predict_df[final_predict_df['amount'] >= AMOUNT_THRESHOLD].reset_index(drop=True)
            print(f"執行成交值流動性過濾 >= {AMOUNT_THRESHOLD}(千)...")

        if REVENUE_THRESHOLD > 0 and 'month_revenue' in final_predict_df.columns:
            final_predict_df = final_predict_df[final_predict_df['month_revenue'] >= REVENUE_THRESHOLD].reset_index(drop=True)
            print(f"執行營收過濾 >= {REVENUE_THRESHOLD}(千)...")

        output_cols = ["id", "name", "date", "close"] + FEATURE_COLS
        final_output = final_predict_df[[c for c in output_cols if c in final_predict_df.columns]].dropna().reset_index(drop=True)
        final_output.to_csv(file_save_name, index=False, encoding="utf-8-sig")
        print(f"預測資料已生成: {file_save_name} | 成功保留 {len(final_output)} 檔標的。")
        return final_output

    else:
        # train / bt 模式：換股日篩選與流動性過濾 (MS+9日邏輯)
        all_trading_days = pd.DataFrame({'actual_date': sorted(df['date'].unique())})
        target_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS') + pd.Timedelta(days=9)
        trade_dates_map = pd.merge_asof(
            pd.DataFrame({'target_date': target_dates}),
            all_trading_days,
            left_on='target_date', right_on='actual_date', direction='forward'
        )

        final_df = df[df['date'].isin(trade_dates_map['actual_date'].dropna().unique())].copy()

        print("執行EMA成交值流動性過濾 (保留Top 1/3)...")
        final_df['liq_rank'] = final_df.groupby('date')['amount_ema_20'].rank(ascending=False, method='min')
        final_df['day_count'] = final_df.groupby('date')['id'].transform('count')
        final_df = final_df[final_df['liq_rank'] <= (final_df['day_count'] / 3)].copy()

        print("進行特徵淨化 (Clipping)、截面排名生成與標籤生成...")
        final_df['amount_rank'] = final_df.groupby('date')['amount_20'].rank(pct=True)
        final_df['rs_rank_20'] = final_df.groupby('date')['hist_return_20'].rank(pct=True)
        final_df['rs_rank_60'] = final_df.groupby('date')['hist_return_60'].rank(pct=True)

        growth_cols = ['mom', 'yoy', 'yoy_mean_3']
        for col in growth_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].clip(lower=-100.0, upper=100.0)

        if mode == 'train':
            # train 模式：dropna 排除未到期/未來標籤為 NaN 的期數
            final_df = final_df.dropna().reset_index(drop=True)
        else:
            # bt 模式：不因 return_20d 為 NaN 捨棄最新調倉期
            feature_cols_to_check = [col for col in final_df.columns if col not in ['return_20d', 'label']]
            final_df = final_df.dropna(subset=feature_cols_to_check).reset_index(drop=True)

        print("計算高流動性科技股專屬廣度 (Tech Market Breadth)...")
        final_df['market_kama_mean'] = final_df.groupby('date')['kama'].transform('mean')
        final_df['market_adx_mean'] = final_df.groupby('date')['adx'].transform('mean')
        final_df['market_resistance_mean'] = final_df.groupby('date')['resistance_20'].transform('mean')

        # 截面排名標籤 (0~1)
        final_df['label'] = final_df.groupby('date')['return_20d'].rank(pct=True)

        final_df.to_csv(file_save_name, index=False, encoding='utf-8-sig')
        print(f"[{mode.upper()}] 資料建置成功！檔案位置: {file_save_name}")
        print(f"最新日期: {final_df['date'].max().date()} | 股票數: {final_df['id'].nunique()} | 樣本總數: {len(final_df)}")
        return final_df
