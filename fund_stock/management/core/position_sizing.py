"""
部位管理與資金配重模組 (Position Sizing & Capital Allocation Module)

核心實戰防衛規則 (三段式 KAMA 階梯配重法則):
1. 當大盤 kama 廣度均線 (market_kama_mean) > 1.00:
   => 市場處於多頭順風，進場 100% 滿倉。
2. 當大盤 kama 廣度均線 (market_kama_mean) 介於 0.90 ~ 1.00 之間:
   => 市場處於常規修正區，進場 75% 倉位，主動保留 1/4 (25%) 現金作防禦。
3. 當大盤 kama 廣度均線 (market_kama_mean) < 0.90:
   => 市場處於極端雪崩/冰點區，進場 50% 半倉防禦，主動保留 1/2 (50%) 現金水位。
"""

def get_position_scale(mk_val):
    """
    根據 market_kama_mean 回傳股票進場資金配置比例 (pos_scale)
    回傳值: float (1.00, 0.75 或 0.50)
    """
    if mk_val is None:
        return 1.00
    val = float(mk_val)
    if val > 1.00:
        return 1.00
    elif val >= 0.90:
        return 0.75
    else:
        return 0.50


def get_market_regime(mk_val):
    """
    取得結構化大盤多空環境與進場配給資訊
    """
    mk_val = float(mk_val) if mk_val is not None else 1.01
    is_bull = mk_val > 1.00

    if mk_val > 1.00:
        pos_scale = 1.00
        regime_label = "> 1.00"
        action_text = "100% 入場 (順風多頭)"
    elif mk_val >= 0.90:
        pos_scale = 0.75
        regime_label = "0.90 ~ 1.00"
        action_text = "75% 入場，保留 25% 現金水位 (常規修正)"
    else:
        pos_scale = 0.50
        regime_label = "< 0.90"
        action_text = "50% 入場，保留 50% 現金水位 (極端雪崩)"

    cash_ratio = 1.00 - pos_scale

    return {
        'mk_val': mk_val,
        'is_bull': is_bull,
        'position_scale': pos_scale,
        'cash_ratio': cash_ratio,
        'regime_label': regime_label,
        'action_text': action_text
    }


def format_allocation_report(mk_val, total_capital=None):
    """
    產生格式化的部位配重與資金操作文字方塊 (供 prediction / simulator 輸出用)
    """
    info = get_market_regime(mk_val)
    lines = [
        "==========================================================================",
        "  [POSITION SIZING] 資金配重與防禦指示 (Capital Allocation Report)",
        "==========================================================================",
        f" * 科技市場 Kama 均溫 : {info['mk_val']:.4f} ({info['regime_label']})",
        f" * 操作建議 : {info['action_text']}",
        "==========================================================================\n",
    ]
    return "\n".join(lines)
