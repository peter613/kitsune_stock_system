"""
github_action/risk_control.py
小狐狸量化 — 風控紀律巡檢中心

四大風控紀律：
  1. 微觀個股第一階段移動減碼 (高點回撤 10% 砍半 50%)
  2. 微觀個股第二階段移動停損 (高點回撤 20% 全清倉)
  3. 宏觀帳戶高點移動停損 (總資產自高點跌 10% 全清倉)
  4. 宏觀帳戶%數停利 (總資產獲利達 10% 砍 1/4 入袋降波動率)

通報原則 (極致低熵，無 Emoji，結構比照資料庫報告)：
  - 觸發時：發送結構化純文字報告至風控頻道，指明確切動作與股數。
  - 安全時：完全靜默 (return None)，零雜訊、零打擾。
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import config

env_path = BASE_DIR / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


def send_risk_notify(message: str) -> bool:
    """發送低熵風控警報至指定頻道 (平時安靜，觸發才響)"""
    webhook_url = (
        os.environ.get("DC_RISK_WEBHOOK_URL")
        or os.environ.get("DC_WEBHOOK_URL")
        or os.environ.get("DC_DATABASE_WEBHOOK_URL")
    )
    if not webhook_url:
        print("[WARN] 未設定 DC_RISK_WEBHOOK_URL / DC_WEBHOOK_URL")
        return False

    try:
        res = requests.post(webhook_url, json={"content": message})
        return res.status_code in (200, 204)
    except Exception as e:
        print(f"[ERROR] 發送風控警報失敗: {e}")
        return False


def run_risk_monitor():
    print("[INFO] Starting Risk Control Evaluation...")

    from market_data.sync_data import download_from_supabase

    # 1. 防呆：確保 config 所定義的本地目錄絕對存在 (解決 GitHub Actions 空目錄問題)
    config.STOCK_JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    config.FUND_STOCK_ADMINISTRATION_DIR.mkdir(parents=True, exist_ok=True)
    config.MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    journal_path = config.STOCK_JOURNAL_DIR / "journal.csv"
    navpu_path = config.FUND_STOCK_ADMINISTRATION_DIR / "navpu.csv"
    market_db_path = config.MARKET_DATA_DIR / "market.parquet"

    # 2. 若本地無 navpu.csv，直接從 Supabase 下載最新備份
    if not navpu_path.exists():
        print("[INFO] 本地無 navpu.csv，正在從 Supabase 下載最新版本...")
        download_from_supabase('stock_data_backup', 'navpu.csv', str(navpu_path))

    # 3. 若本地無 market.parquet，從 Supabase 下載當年市場資料
    if not market_db_path.exists():
        current_year = datetime.now().year
        print(f"[INFO] 本地無 market.parquet，正在從 Supabase 下載 {current_year} 年資料...")
        download_from_supabase('stock_data_backup', f"market/market_{current_year}.parquet", str(market_db_path))

    if not journal_path.exists() or not navpu_path.exists() or not market_db_path.exists():
        print("[WARN] 缺少必要檔案 (journal / navpu / market)，略過風控檢查。")
        return None

    try:
        df_record = pd.read_csv(journal_path, dtype={'id': str})
        df_record.columns = df_record.columns.str.strip().str.lower()
        df_record['date'] = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
        df_record['price'] = pd.to_numeric(df_record.get('price', 0), errors='coerce').fillna(0)
        df_record['share'] = pd.to_numeric(df_record.get('share', 0), errors='coerce').fillna(0)
        df_record = df_record.sort_values('date')

        df_history = pd.read_csv(navpu_path)
        df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce').dt.normalize()
        df_history = df_history.sort_values('date').reset_index(drop=True)

        df_price = pd.read_parquet(market_db_path)
        df_price.columns = df_price.columns.str.strip().str.lower()
        if 'price' in df_price.columns and 'close' not in df_price.columns:
            df_price = df_price.rename(columns={'price': 'close'})
        df_price['date'] = pd.to_datetime(df_price['date'], errors='coerce').dt.normalize()
    except Exception as e:
        print(f"[WARN] 讀取資料失敗: {e}")
        return None

    # 1. 解析當前持倉與 Stage 1 減碼狀態
    active_holdings = []
    for aid in df_record['id'].dropna().unique():
        sub = df_record[df_record['id'] == aid]
        stock_name = sub['name'].iloc[0] if 'name' in sub.columns and pd.notnull(sub['name'].iloc[0]) else aid

        cum_shares = 0.0
        total_cost = 0.0
        cycle_start_date = None
        has_trimmed = False

        for _, row in sub.iterrows():
            trade_val = row['price'] * row['share']
            action = str(row['action']).strip().lower()
            if action == 'buy':
                if cum_shares <= 0.0001:
                    cycle_start_date = row['date']
                    has_trimmed = False
                cum_shares += row['share']
                total_cost += trade_val + int(trade_val * 0.001425)
            elif action == 'sell':
                avg_cost = total_cost / cum_shares if cum_shares > 0 else 0
                cum_shares -= row['share']
                total_cost -= row['share'] * avg_cost
                if cum_shares <= 0.0001:
                    cycle_start_date = None
                    has_trimmed = False
                else:
                    has_trimmed = True

        if cum_shares > 0.0001:
            active_holdings.append({
                "code": aid,
                "name": stock_name,
                "shares": int(cum_shares),
                "cost": total_cost / cum_shares,
                "entry_date": cycle_start_date,
                "has_trimmed": has_trimmed,
            })

    # 防呆機制：若目前為 100% 現金空倉狀態，代表清倉/停損指令皆已執行，不需發出警報
    if not active_holdings:
        print("[INFO] 目前為 100% 現金空倉狀態，安全不打擾。")
        return None

    today_str = datetime.now().strftime("%Y-%m-%d")
    latest_px_map = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict()

    # 2. 宏觀帳戶總資產回撤與獲利計算
    recent_buys = df_record[(df_record['action'] == 'buy') & (~df_record['id'].str.startswith('cash'))]
    latest_month = recent_buys['date'].max().to_period('M')
    entry_date = recent_buys[recent_buys['date'].dt.to_period('M') == latest_month]['date'].min()
    entry_date_str = entry_date.strftime('%m/%d') if pd.notnull(entry_date) else ""

    period_df = df_history[df_history['date'] >= entry_date]
    if not period_df.empty:
        peak_nav = period_df['total_nav'].max()
        entry_nav = period_df['total_nav'].iloc[0]
        cur_nav = df_history['total_nav'].iloc[-1]
        macro_drawdown = ((cur_nav / peak_nav) - 1.0) * 100 if peak_nav > 0 else 0.0
        macro_roi = ((cur_nav / entry_nav) - 1.0) * 100 if entry_nav > 0 else 0.0
    else:
        macro_drawdown = 0.0
        macro_roi = 0.0

    # 檢查本週期是否已有 1/4 停利紀錄
    cycle_records = df_record[df_record['date'] >= entry_date]
    cycle_sells = cycle_records[cycle_records['action'] == 'sell']
    has_taken_macro_profit = False
    for _, row in cycle_sells.iterrows():
        l_str = str(row.get('label', '')).lower()
        r_str = str(row.get('reason', '')).lower()
        if 'take_profit' in l_str or '停利' in r_str or '1/4' in r_str:
            has_taken_macro_profit = True
            break

    # 3. 個股微觀回撤與停損計算
    stage2_stocks = []
    stage1_stocks = []

    for h in active_holdings:
        code = h["code"]
        name = h["name"]
        cost = h["cost"]
        shares = h["shares"]
        start_d = h["entry_date"]
        has_trimmed = h["has_trimmed"]

        cur_price = latest_px_map.get(code, cost)
        stock_hist = df_price[(df_price['id'] == code) & (df_price['date'] >= start_d)]
        hist_high = stock_hist['close'].max() if not stock_hist.empty else cur_price
        peak_price = max(cost, hist_high, cur_price)

        roi_pct = ((cur_price / cost) - 1.0) * 100 if cost > 0 else 0.0
        dd_pct = min(0.0, ((cur_price / peak_price) - 1.0) * 100) if peak_price > 0 else 0.0
        max_loss = min(dd_pct, roi_pct)

        if max_loss <= -20.0:
            stage2_stocks.append({
                "code": code, "name": name, "shares": shares, "cur_price": cur_price,
                "cost": cost, "dd_pct": dd_pct, "peak_price": peak_price
            })
        elif max_loss <= -10.0 and not has_trimmed:
            stage1_stocks.append({
                "code": code, "name": name, "shares": shares, "cur_price": cur_price,
                "cost": cost, "dd_pct": dd_pct, "peak_price": peak_price
            })

    # ──────────────────────────────────────────────────────────────────────────
    # 四大風控優先權階層判定 (模仿資料庫報告格式，純文字無 emoji，觸發才發)
    # ──────────────────────────────────────────────────────────────────────────

    # 1. 宏觀帳戶高點移動停損 (總資產自高點跌 10% 全清倉)
    if macro_drawdown <= -10.0:
        clean_lines = [
            "台股風控紀律報告",
            f"日期：{today_str}",
            "狀態：觸發風控",
            "",
            "‧ 觸發項目：宏觀帳戶高點移動停損",
            f"‧ 數據指標：本週期({entry_date_str})總資產自高點回撤已達 {macro_drawdown:.1f}% (門檻 -10.0%)，累積報酬 {macro_roi:+.1f}%",
            "‧ 執行動作：全面清倉 (所有持倉股票 100% 賣出清空)",
        ]
        for h in active_holdings:
            clean_lines.append(f"  - {h['code']} {h['name']}：全數賣出 {h['shares']} 股")

        send_risk_notify("\n".join(clean_lines))
        return "MACRO_STOPLOSS"

    # 2. 微觀個股第二階段移動停損 (高點回撤 20% 全清倉)
    elif stage2_stocks:
        clean_lines = [
            "台股風控紀律報告",
            f"日期：{today_str}",
            "狀態：觸發風控",
            "",
            "‧ 觸發項目：微觀個股第二階段移動停損",
        ]
        for s in stage2_stocks:
            clean_lines.append(
                f"‧ 標的：{s['code']} {s['name']}\n"
                f"  - 數據：現價 {s['cur_price']:.2f} (成本 {s['cost']:.2f})，高點回撤 {s['dd_pct']:.1f}% (門檻 -20.0%)\n"
                f"  - 動作：全數賣出 {s['shares']} 股 (清倉 100%)"
            )
        send_risk_notify("\n".join(clean_lines))
        return "STAGE2_STOPLOSS"

    # 3. 微觀個股第一階段移動減碼 (高點回撤 10% 砍半 50%)
    elif stage1_stocks:
        clean_lines = [
            "台股風控紀律報告",
            f"日期：{today_str}",
            "狀態：觸發風控",
            "",
            "‧ 觸發項目：微觀個股第一階段移動減碼",
        ]
        for s in stage1_stocks:
            trim_shares = int(s['shares'] * 0.5)
            clean_lines.append(
                f"‧ 標的：{s['code']} {s['name']}\n"
                f"  - 數據：現價 {s['cur_price']:.2f} (成本 {s['cost']:.2f})，高點回撤 {s['dd_pct']:.1f}% (門檻 -10.0%)\n"
                f"  - 動作：賣出 {trim_shares} 股 (砍半 50%)"
            )
        send_risk_notify("\n".join(clean_lines))
        return "STAGE1_STOPLOSS"

    # 4. 宏觀帳戶%數停利 (總資產獲利達 10% 砍 1/4 入袋)
    elif macro_roi >= 10.0 and not has_taken_macro_profit:
        clean_lines = [
            "台股風控紀律報告",
            f"日期：{today_str}",
            "狀態：觸發風控",
            "",
            "‧ 觸發項目：宏觀帳戶%數停利",
            f"‧ 數據指標：本週期({entry_date_str})總資產獲利已達 {macro_roi:+.1f}% (門檻 +10.0%)",
            "‧ 執行動作：各檔持股砍 1/4 (25%) 獲利入袋降低波動率",
        ]
        for h in active_holdings:
            quarter = int(h['shares'] * 0.25)
            remain = h['shares'] - quarter
            clean_lines.append(f"  - {h['code']} {h['name']}：目前 {h['shares']} 股 -> 賣出 {quarter} 股 (剩餘 {remain} 股)")

        send_risk_notify("\n".join(clean_lines))
        return "MACRO_TAKE_PROFIT"

    else:
        print("[INFO] 風控巡檢完成：無觸發任何風控門檻，完全不發送訊息打擾。")
        return None


if __name__ == "__main__":
    run_risk_monitor()
