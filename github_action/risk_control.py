"""
github_action/risk_control.py
小狐狸量化 — 風控紀律巡檢中心

三大風控紀律 (100% 忠實對齊 simulator.py)：
  1. 宏觀帳戶高點移動停損 (總資產/單位淨值自高點跌 10% 全清倉)
  2. 微觀個股移動停損 (高點回撤 20% 全清倉防地雷)
  3. 宏觀帳戶%數停利 (總資產/單位淨值獲利達 10% 砍 1/4 入袋降波動率)

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


def chunk_message(content: str, max_limit: int = 1900) -> list:
    """將過長訊息依行安全切分為多則 Discord 訊息 (避免超過 2000 字元限制)"""
    if len(content) <= max_limit:
        return [content]

    chunks = []
    current_chunk = []
    current_len = 0

    for line in content.split("\n"):
        line_len = len(line) + 1
        if current_len + line_len > max_limit:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = line_len
        else:
            current_chunk.append(line)
            current_len += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def send_risk_notify(message: str) -> bool:
    """發送低熵風控警報至指定頻道 (平時安靜，觸發才響)"""
    webhook_url = os.environ.get("DC_RISK_WEBHOOK_URL")
    if not webhook_url:
        print("[WARN] 未設定 DC_RISK_WEBHOOK_URL")
        return False

    messages = chunk_message(message)
    success = True
    for msg in messages:
        try:
            res = requests.post(webhook_url, json={"content": msg}, timeout=10)
            if res.status_code not in (200, 204):
                print(f"[ERROR] Discord Webhook 發送失敗 ({res.status_code}): {res.text}")
                success = False
        except Exception as e:
            print(f"[ERROR] 發送風控警報異常: {e}")
            success = False

    return success


def get_0050_bear_market_info():
    """
    100% 線上獲取 0050 近 1 年（約 252 交易日）還原收盤價走勢，計算距離 1 年最高點回撤。
    純線上雲端抓取（支援 yfinance 與 Yahoo Finance API），完全不依賴本地 etf.csv。
    返回: (cur_price, peak_price, drawdown_pct)
    """
    # 直連 Yahoo Finance Chart API 獲取精確還原股價 (adjclose)
    try:
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/0050.TW?range=1y&interval=1d'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            d = r.json()['chart']['result'][0]
            closes = d['indicators']['adjclose'][0]['adjclose']
            df_s = pd.Series(closes).dropna()
            if len(df_s) >= 20:
                cur = float(df_s.iloc[-1])
                peak = float(df_s.max())
                dd = (cur / peak - 1.0) * 100
                return cur, peak, dd
    except Exception as e:
        print(f"[WARN] Yahoo Chart API 線上讀取 0050 異常: {e}")

    return None, None, None


def run_risk_monitor():
    print("[INFO] Starting Risk Control Evaluation...")

    # 1. 週末休市 (完全對齊 market.py 爬蟲語句)
    tw_now = pd.Timestamp.now(tz='Asia/Taipei')
    if tw_now.weekday() >= 5:
        day_str = ['六', '日'][tw_now.weekday() - 5]
        print(f"[INFO] 今日為週末 (週{day_str})，市場休市。")
        return None

    # 2. 國定假日/休市日 (對照證交所休市表 holidays.csv，完全對齊 market.py 爬蟲語句)
    holidays_file = config.MARKET_DATA_DIR / "holidays.csv"
    if holidays_file.exists():
        try:
            df_h = pd.read_csv(holidays_file)
            df_h = df_h[~df_h['name'].astype(str).str.contains('開始交易')]
            df_h['date'] = pd.to_datetime(df_h['date']).dt.date
            today_date = tw_now.date()
            matched = df_h[df_h['date'] == today_date]
            if not matched.empty:
                holiday_name = matched['name'].iloc[0]
                print(f"[INFO] 今日為國定假日/休市日 ({holiday_name})，市場休市。")
                return None
        except Exception as e:
            print(f"[WARN] 讀取休市清單失敗: {e}")

    journal_path = config.STOCK_JOURNAL_DIR / "journal.csv"
    navpu_path = config.FUND_STOCK_ADMINISTRATION_DIR / "navpu.csv"
    market_db_path = config.MARKET_DATA_DIR / "market.parquet"

    # 3. 休市防呆 (平日國定假日/颱風假未開市或爬蟲略過，100% 對齊證交所爬蟲語句)
    if not navpu_path.exists() or not market_db_path.exists():
        print("[INFO] 上市與上櫃皆無資料 (休市或爬蟲異常)，風控巡檢略過。")
        return None

    try:
        if not journal_path.exists():
            raise FileNotFoundError(f"找不到交易紀錄檔: {journal_path}")

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
        df_price = df_price.sort_values('date')
    except Exception as e:
        err_msg = f"台股風控系統異常警告\n狀態：**資料讀取失敗**\n‧ ERROR：{str(e)}"
        print(f"[ERROR] 讀取資料失敗: {e}")
        send_risk_notify(err_msg)
        return False

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
                "shares": int(round(cum_shares)),
                "cost": total_cost / cum_shares,
                "entry_date": cycle_start_date,
                "has_trimmed": has_trimmed,
            })

    # 防呆機制：若目前為 100% 現金空倉狀態，代表清倉/停損指令皆已執行，不需發出警報
    if not active_holdings:
        print("[INFO] 目前為 100% 現金空倉狀態，安全不打擾。")
        return None

    # 取得市場最新行情日期與當日收盤價
    latest_mkt_date = df_price['date'].max()
    today_str = latest_mkt_date.strftime("%Y-%m-%d") if pd.notnull(latest_mkt_date) else datetime.now().strftime("%Y-%m-%d")

    # 建立最新一天的價格映射 (過濾掉歷史舊價，確保 missing_price_stocks 能如實反映今日缺漏)
    latest_day_df = df_price[df_price['date'] == latest_mkt_date].dropna(subset=['close'])
    latest_px_map = dict(zip(latest_day_df['id'], latest_day_df['close']))

    # 2. 宏觀帳戶回撤與獲利計算（使用 nav_per_unit 排除出入金干擾，忠實反映真實策略績效）
    # 取得本週期真實進場日 (取所有持股的最小進場日)
    valid_entry_dates = [h["entry_date"] for h in active_holdings if pd.notnull(h["entry_date"])]
    entry_date = min(valid_entry_dates) if valid_entry_dates else df_record['date'].max()
    entry_date_str = entry_date.strftime('%m/%d') if pd.notnull(entry_date) else ""

    # 檢查是否已執行過宏觀停利
    curr_cycle = df_record['cycle'].max()
    curr_records = df_record[df_record['cycle'] == curr_cycle] if pd.notnull(curr_cycle) else df_record
    has_taken_macro_profit = (
        'reason' in curr_records.columns and
        curr_records['reason'].dropna().astype(str).str.strip().str.upper().eq('TP_QUARTER').any()
    )

    nav_col = 'nav_per_unit' if 'nav_per_unit' in df_history.columns else 'total_nav'
    period_df = df_history[df_history['date'] >= entry_date]

    if not period_df.empty:
        # 進場基準淨值：取進場日之前的最後一筆收盤淨值 (對齊 simulator.py 的 capital 起始基準)
        pre_entry_history = df_history[df_history['date'] < entry_date]
        if not pre_entry_history.empty:
            entry_nav = pre_entry_history[nav_col].iloc[-1]
        else:
            entry_nav = period_df[nav_col].iloc[0]

        cur_nav = df_history[nav_col].iloc[-1]
        peak_nav = max(entry_nav, period_df[nav_col].max())
        macro_drawdown = min(0.0, ((cur_nav / peak_nav) - 1.0) * 100) if peak_nav > 0 else 0.0
        macro_roi = ((cur_nav / entry_nav) - 1.0) * 100 if entry_nav > 0 else 0.0
    else:
        macro_drawdown, macro_roi = 0.0, 0.0

    # 3. 個股微觀回撤與停損計算
    stage2_stocks = []
    missing_price_stocks = []

    for h in active_holdings:
        code = h["code"]
        name = h["name"]
        cost = h["cost"]
        shares = h["shares"]
        start_d = h["entry_date"] if pd.notnull(h["entry_date"]) else df_record['date'].min()

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

        # 微觀個股移動停損 (高點回撤 20% 全清倉防地雷)
        if max_loss <= -20.0:
            stage2_stocks.append({
                "code": code, "name": name, "shares": shares, "cur_price": cur_price,
                "cost": cost, "dd_pct": dd_pct, "peak_price": peak_price
            })

    # ──────────────────────────────────────────────────────────────────────────
    # 四大風控優先權判定 (宏觀熊市熔斷 / 宏觀帳戶停損 / 微觀個股停損 / 宏觀停利)
    # ──────────────────────────────────────────────────────────────────────────
    report_sections = []
    action_type = None

    # 報價異常警告 (避免行情缺漏導致悄悄漏停損)
    if missing_price_stocks:
        sec_lines = [
            "報價異常警告",
            "今日查無收盤價：",
        ]
        for s in missing_price_stocks:
            sec_lines.append(f"  - {s}")
        report_sections.append("\n".join(sec_lines))
        action_type = "PRICE_MISSING_WARNING"

    # 優先權 0：宏觀大盤技術性熊市熔斷 (0050 近 1 年自高點回撤達 -20.0%)
    etf_cur, etf_peak, etf_dd = get_0050_bear_market_info()
    if etf_dd is not None:
        print(f"[INFO] 0050 大盤狀態: 現價 {etf_cur:.2f} | 1年最高 {etf_peak:.2f} | 回撤 {etf_dd:.2f}%")
        if etf_dd <= -20.0:
            report_sections.append("技術性熊市(0050回撤-20%)\n**全面清倉**")
            action_type = "BEAR_MARKET_STOPLOSS"

    # 優先權 1：宏觀帳戶高點移動停損 (總資產/單位淨值自高點跌 10% 全清倉)
    # 若已觸發熊市清倉，則無需重複判定
    if action_type != "BEAR_MARKET_STOPLOSS" and macro_drawdown <= -10.0:
        report_sections.append("宏觀帳戶移動停損 (門檻 -10.0%)\n**全面清倉**")
        action_type = "MACRO_STOPLOSS"

    # 優先權 2~3：若無全面清倉 (非熊市且非帳戶大停損)，微觀個股停損與宏觀停利合併通報
    if action_type not in ("BEAR_MARKET_STOPLOSS", "MACRO_STOPLOSS"):
        # 微觀個股移動停損 (高點回撤 20% 全清倉防地雷)
        if stage2_stocks:
            sec = ["微觀個股移動停損 (門檻 -20.0%)"]
            for s in stage2_stocks:
                sec.append(f"{s['code']} {s['name']}：**全數賣出**")
            report_sections.append("\n".join(sec))
            action_type = "RISK_ACTIONS_TRIGGERED"

        # 宏觀帳戶%數停利 (總資產獲利達 10% 砍 1/4 入袋)
        if macro_roi >= 10.0 and not has_taken_macro_profit:
            sec = [
                "宏觀帳戶停利 (門檻 +10.0%)",
                "**持股各砍 1/4 獲利入袋**",
            ]
            report_sections.append("\n".join(sec))
            action_type = "RISK_ACTIONS_TRIGGERED"

    # 統一收攏發送：全程式僅此一處組裝發送，避免代碼重複
    if report_sections:
        final_message = f"台股風控通知\n日期：{today_str}\n\n" + "\n\n".join(report_sections)
        success = send_risk_notify(final_message)
        return (action_type or "RISK_ACTIONS_TRIGGERED") if success else False
    else:
        print("[INFO] 風控巡檢完成：無觸發任何風控門檻，完全不發送訊息打擾。")
        return None


if __name__ == "__main__":
    result = run_risk_monitor()
    if result is False:
        sys.exit(1)
