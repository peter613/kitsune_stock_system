"""
部位管理與資金配重模組 (Position Sizing & Capital Allocation Module)

核心實戰防衛規則 (熊市 0/1 開關法則):
1. 常規市場 (0050 自歷史高點回撤 < 20%):
   => 進場 100% 滿倉，不預設立場預設減倉。
2. 技術性熊市 (0050 自歷史高點回撤 >= 20%):
   => 觸發熊市開關，0% 倉位 (暫停交易 / 持倉無條件砍倉清空保留 100% 現金防禦)。

_0050data 參數說明:
    pandas Series，內容為 0050 已調整收盤價（不含除權息斷層），
    來源：market_data/data/etf.csv 的 '0050' 欄位。
"""

import pandas as pd
from pathlib import Path
import config

# ── 0050 資料載入工具 ─────────────────────────────────────
try: 
    _ETF_CSV = config.MARKET_DATA_DIR/"etf.csv"
except: 
    _ETF_CSV = Path(__file__).resolve().parent.parent.parent.parent / "market_data" / "data" / "etf.csv"


def load_0050_series(etf_path=None):
    """
    讀取 etf.csv 並回傳 0050 已調整收盤價 Series（index 為日期）。
    若讀取失敗則回傳空 Series。
    """
    path = Path(etf_path) if etf_path else _ETF_CSV
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        s = df.set_index('date')['0050'].dropna()
        return s
    except Exception as e:
        print(f"[position_sizing] 警告：無法讀取 0050 資料 ({path}): {e}")
        return pd.Series(dtype=float)


def _return(_0050data):
    """
    判斷是否處於技術性熊市：0050 最新收盤價相對歷史高點回撤 ≥ 20%。
    回傳 float (負值表回撤)。
    """
    if _0050data is None or _0050data.empty:
        return 0.0
    drawdown = _0050data.iloc[-1] / _0050data.max() - 1.0   # 負值表回撤
    return drawdown


def get_position_scale(mk_val, _0050data=None):
    """
    根據 0050 技術性熊市指標回傳股票進場資金配置比例 (pos_scale)
    回傳值: float (1.00 或 0.00)
    """
    if _return(_0050data) <= -0.20:
        return 0.0
    else:
        return 1.0


def get_market_regime(mk_val, _0050data=None):
    """
    取得結構化大盤多空環境與進場配給資訊
    """
    mk_val = float(mk_val) if mk_val is not None else 1.01
    is_bull = mk_val > 0.98
    bear_market = _return(_0050data) <= -0.20
    _0050_return = f"{_return(_0050data) * 100:.2f}%"

    if bear_market:
        pos_scale = 0.00
        regime_label = "技術性熊市"
        action_text = "暫停交易 (0050高點回撤20%)"
    else:
        pos_scale = 1.00
        regime_label = "常規市場"
        action_text = "100% 滿倉入場"
        is_bull = True

    cash_ratio = 1.00 - pos_scale

    return {
        'mk_val': mk_val,
        'is_bull': is_bull,
        'bear_market': bear_market,
        'position_scale': pos_scale,
        'cash_ratio': cash_ratio,
        'regime_label': regime_label,
        'action_text': action_text,
        '_0050_return': _0050_return
    }

    return {
        'mk_val': mk_val,
        'is_bull': is_bull,
        'bear_market': bear_market,
        'position_scale': pos_scale,
        'cash_ratio': cash_ratio,
        'regime_label': regime_label,
        'action_text': action_text,
        '_0050_return': _0050_return
    }


def format_allocation_report(mk_val=None, _0050data=None):
    """
    產生格式化的部位配重與資金操作文字方塊 (供 prediction / simulator 輸出用)
    """
    info = get_market_regime(mk_val, _0050data)
    
    pos_pct = f"{info['position_scale'] * 100:.0f}%"
    cash_pct = f"{info['cash_ratio'] * 100:.0f}%"
    
    lines = [
        "==========================================================================",
        "   [POSITION SIZING] 資金配重與防禦指示 (Capital Allocation Report)",
        "==========================================================================",
        f" * 科技市場 Kama 均溫 : {info['mk_val']:.4f} (僅參考)",
        f" * 當前0050環境 : {info['regime_label']} (報酬/回撤 : {info['_0050_return']})",
        f" * 具體操作建議 : {info['action_text']}",
        #f" * 目標股票持倉比例 : {pos_pct}",
        #f" * 建議預留現金比例 : {cash_pct}"
        "==========================================================================\n",
    ]  
    return "\n".join(lines)