"""
github_action/risk_control.py
小狐狸量化 — 風控紀律巡檢中心

四大風控紀律 (100% 忠實對齊 simulator.py)：
  1. 微觀個股第一階段移動減碼 (高點回撤 10% 砍半 50%)
  2. 微觀個股第二階段移動停損 (高點回撤 20% 全清倉)
  3. 宏觀帳戶高點移動停損 (總資產/單位淨值自高點跌 10% 全清倉)
  4. 宏觀帳戶%數停利 (總資產/單位淨值獲利達 10% 砍 1/4 入袋降波動率)

通報原則 (極致低熵，無 Emoji，結構比照資料庫報告)：
  - 觸發時：發送結構化純文字報告至風控頻道，指明確切動作與股數。
  - 多檔同時觸發時：合併於同一則報告通報，絕不互斥遺漏。
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
    webhook_url = os.environ.get("DC_RISK_WEBHOOK_URL")
    if not webhook_url:
        print("[WARN] 未設定 DC_RISK_WEBHOOK_URL")
        return False

    try:
        res = requests.post(webhook_url, json={"content": message})
        return res.status_code in (200, 204)
    except Exception as e:
        print(f"[ERROR] 發送風控警報失敗: {e}")
        return False


def run_risk_monitor():
    print("[INFO] Starting Risk Control Evaluation...")

    journal_path = config.STOCK_JOURNAL_DIR / "journal.csv"
    navpu_path = config.FUND_STOCK_ADMINISTRATION_DIR / "navpu.csv"
    market_db_path = config.MARKET_DATA_DIR / "market.parquet"

    try:
        df_record = pd.read_csv(journal_path, dtype={'id': str}, encoding='utf-8-sig')
        df_record.columns = df_record.columns.str.strip().str.lower()
        if 'cycle' in df_record.columns:
            df_record['cycle'] = pd.to_numeric(df_record['cycle'], errors='coerce')
        df_record['id'] = df_record['id'].astype(str).str.strip().str.replace('.0', '', regex=False)
        df_record['date'] = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
        df_record['price'] = pd.to_numeric(df_record.get('price', 0), errors='coerce').fillna(0)
        df_record['share'] = pd.to_numeric(df_record.get('share', 0), errors='coerce').fillna(0)
        df_record = df_record.sort_values('date')

        # 補充股票名稱（從 white_list.csv 依 id 對照）
        wl_path = config.MARKET_DATA_DIR / "white_list.csv"
        if wl_path.exists():
            df_wl = pd.read_csv(wl_path, dtype={'id': str}, encoding='utf-8-sig')
            name_col = 'nmae' if 'nmae' in df_wl.columns else ('name' if 'name' in df_wl.columns else None)
            if name_col:
                name_map = dict(zip(
                    df_wl['id'].astype(str).str.strip().str.replace('.0', '', regex=False),
                    df_wl[name_col].astype(str).str.strip()
                ))
                df_record['name'] = df_record['id'].map(name_map).fillna('')
        if 'name' not in df_record.columns:
            df_record['name'] = ''

        df_history = pd.read_csv(navpu_path)
        df_history.columns = df_history.columns.str.strip().str.lower()
        df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce').dt.normalize()
        df_history = df_history.sort_values('date').reset_index(drop=True)

        df_price = pd.read_parquet(market_db_path)
        df_price.columns = df_price.columns.str.strip().str.lower()
        if 'price' in df_price.columns and 'close' not in df_price.columns:
            df_price = df_price.rename(columns={'price': 'close'})
        df_price['date'] = pd.to_datetime(df_price['date'], errors='coerce').dt.normalize()
        df_price['id'] = df_price['id'].astype(str).str.strip().str.replace('.0', '', regex=False)
        # 確保依日期排序，後續 groupby('id').tail(1) 才能準確取得最新收盤價
        df_price = df_price.sort_values('date')
    except Exception as e:
        print(f"[WARN] 讀取資料失敗: {e}")
        return None

    # 1. 解析當前持倉與 Stage 1 減碼狀態
    active_holdings = []
    for aid in df_record['id'].dropna().unique():
        if str(aid).lower().startswith('cash'):
            continue

        sub = df_record[df_record['id'] == aid]
        stock_name = sub['name'].iloc[0] if 'name' in sub.columns and pd.notnull(sub['name'].iloc[0]) and sub['name'].iloc[0] != '' else aid

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
                elif str(row.get('reason', '')).strip().upper() == 'SL_HALF':
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

    # 2. 宏觀帳戶回撤與獲利計算（使用 nav_per_unit 排除出入金干擾，忠實反映真實策略績效）
    curr = df_record[df_record['cycle'] == df_record['cycle'].max()]
    buys = curr[(curr['action'] == 'buy') & (~curr['id'].str.startswith('cash'))]
    entry_date = buys['date'].min() if not buys.empty else df_record['date'].max()
    entry_date_str = entry_date.strftime('%m/%d') if pd.notnull(entry_date) else ""
    has_taken_macro_profit = (curr['reason'] == 'TP_QUARTER').any()

    nav_col = 'nav_per_unit' if 'nav_per_unit' in df_history.columns else 'total_nav'
    period_df = df_history[df_history['date'] >= entry_date]
    if not period_df.empty:
        peak_nav = period_df[nav_col].max()
        entry_nav = period_df[nav_col].iloc[0]
        cur_nav = df_history[nav_col].iloc[-1]
        macro_drawdown = min(0.0, ((cur_nav / peak_nav) - 1.0) * 100) if peak_nav > 0 else 0.0
        macro_roi = ((cur_nav / entry_nav) - 1.0) * 100 if entry_nav > 0 else 0.0
    else:
        macro_drawdown, macro_roi = 0.0, 0.0

    # 3. 個股微觀回撤與停損計算
    stage2_stocks = []
    stage1_stocks = []
    missing_price_stocks = []

    for h in active_holdings:
        code = h["code"]
        name = h["name"]
        cost = h["cost"]
        shares = h["shares"]
        start_d = h["entry_date"]
        has_trimmed = h["has_trimmed"]

        if code not in latest_px_map:
            missing_price_stocks.append(f"{code} {name}")
            continue

        cur_price = latest_px_map[code]
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
    # 四大風控優先權判定 (忠實對齊 simulator.py)
    # ──────────────────────────────────────────────────────────────────────────

    # 優先權 1：宏觀帳戶高點移動停損 (總資產/單位淨值自高點跌 10% 全清倉)
    if macro_drawdown <= -10.0:
        clean_lines = [
            "台股風控通知",
            f"日期：{today_str}",
            "狀態：宏觀帳戶高點移動停損 (門檻 -10.0%)",
            "",
            f"‧ 本週期({entry_date_str})單位淨值自高點回撤已達 {macro_drawdown:.1f}%",
            " - **全面清倉**",
        ]
        for h in active_holdings:
            clean_lines.append(f"  - {h['code']} {h['name']}：全數賣出 {h['shares']} 股")

        send_risk_notify("\n".join(clean_lines))
        return "MACRO_STOPLOSS"

    # 優先權 2~4：若宏觀未清倉，微觀停損/減碼與宏觀停利合併通報，絕不互斥遺漏
    report_sections = []

    # 報價異常警告 (避免行情缺漏導致悄悄漏停損)
    if missing_price_stocks:
        sec = [
            "【報價異常警告】",
            "‧ 以下持倉於今日 market.parquet 查無收盤價，請手動確認：",
        ]
        for s in missing_price_stocks:
            sec.append(f"  - {s}")
        report_sections.append("\n".join(sec))

    # 微觀個股第二階段移動停損 (高點回撤 20% 全清倉)
    if stage2_stocks:
        sec = ["【微觀個股第二階段移動停損 (門檻 -20.0%)】"]
        for s in stage2_stocks:
            sec.append(
                f"‧ 標的：{s['code']} {s['name']}\n"
                f" - 高點回撤 {s['dd_pct']:.1f}%\n"
                f" - 數據：現價 {s['cur_price']:.2f} (成本 {s['cost']:.2f})\n"
                f" - **全砍** 賣出{s['shares']} 股"
            )
        report_sections.append("\n".join(sec))

    # 微觀個股第一階段移動減碼 (高點回撤 10% 砍半 50%)
    if stage1_stocks:
        sec = ["【微觀個股第一階段移動減碼 (門檻 -10.0%)】"]
        for s in stage1_stocks:
            trim_shares = int(s['shares'] * 0.5)
            sec.append(
                f"‧ 標的：{s['code']} {s['name']}\n"
                f" - 高點回撤 {s['dd_pct']:.1f}%\n"
                f" - 數據：現價 {s['cur_price']:.2f} (成本 {s['cost']:.2f})\n"
                f" - **砍半** 賣出 {trim_shares} 股"
            )
        report_sections.append("\n".join(sec))

    # 宏觀帳戶%數停利 (總資產獲利達 10% 砍 1/4 入袋)
    if macro_roi >= 10.0 and not has_taken_macro_profit:
        sec = [
            "【宏觀帳戶停利 (門檻 +10.0%)】",
            f"‧ 本週期({entry_date_str})單位淨值獲利已達 {macro_roi:+.1f}%\n"
            " - **各檔持股砍 1/4 獲利入袋降低波動率**",
        ]
        for h in active_holdings:
            quarter = int(h['shares'] * 0.25)
            remain = h['shares'] - quarter
            sec.append(f"  - {h['code']} {h['name']}：目前 {h['shares']} 股 -> 賣出 {quarter} 股 (剩餘 {remain} 股)")
        report_sections.append("\n".join(sec))

    if report_sections:
        final_message = f"台股風控通知\n日期：{today_str}\n\n" + "\n\n".join(report_sections)
        send_risk_notify(final_message)
        return "RISK_ACTIONS_TRIGGERED"
    else:
        print("[INFO] 風控巡檢完成：無觸發任何風控門檻，完全不發送訊息打擾。")
        return None


if __name__ == "__main__":
    run_risk_monitor()
