# streamlit run c:/Users/Lynn610/Documents/kitsune_stock_system/dashboard/main.py
"""
dashboard.py
Kitsune Kernel — 主程式入口
職責：CSS 注入、資料初始化、警報觸發、Sidebar 路由
啟動：streamlit run dashboard.py
"""

import streamlit as st

from engine import (
    load_data,
    calculate_daily_history,
    calculate_settlement_data,
    check_rebalance_alerts,
    check_market_volatility_alerts,
    check_stoploss_alerts,
)
from pages import (
    page_performance_pct,
    page_performance_nav,
    page_allocation,
    page_history,
    page_monthly_performance,
)

# ──────────────────────────────────────────
# 全域設定
# ──────────────────────────────────────────
st.set_page_config(page_title="Kitsune Kernel", layout="wide")

_CSS = """
<style>
[data-testid="stSidebar"] {
    background-color: #fcf5ed !important;
}
[data-testid="stSidebar"] h1 {
    margin-top: -2rem !important;
    padding-top: 2px !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0rem !important;
}
.stApp h1 {
    font-size: 1.5rem !important;
    padding-bottom: 0.8rem !important;
    font-weight: 700 !important;
    color: #1a1a1a;
}
.stApp h3 {
    font-size: 1.1rem !important;
    margin-top: -5px !important;
    padding-bottom: 0.6rem !important;
    font-weight: 400 !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 1.0rem !important;
    font-weight: 500 !important;
    color: #333333 !important;
    opacity: 1 !important;
    margin-bottom: 0px !important;
}
[data-testid="stMetricValue"] div {
    font-size: 2.0rem !important;
    font-weight: 700 !important;
    color: #C0392B !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #1a1a1a !important;
    font-size: 1.05rem !important;
    margin-bottom: 8px !important;
}
[data-testid="stSidebar"] .st-emotion-cache-6qob1r {
    font-size: 1.1rem !important;
}
div[data-testid="stAlertContainer"] {
    background-color: #1A2B4C !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    height: 100% !important;
    width: 100% !important;
}
div[data-testid="stAlert"] p {
    color: #ffffff !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    letter-spacing: 3.0px !important;
}
</style>
"""

# ──────────────────────────────────────────
# 頁面路由對應表
# ──────────────────────────────────────────
_PAGES = {
    "累積報酬率":   lambda h, r, p: page_performance_pct(h),
    "累積權益":     lambda h, r, p: page_performance_nav(h, r, p),
    "目前配置":     lambda h, r, p: page_allocation(r, p),
    "損益結算中心": lambda h, r, p: page_history(r, p),
    "單月績效報告": lambda h, r, p: page_monthly_performance(h),
}

# ──────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────
def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    # 1. 資料讀取（含 loading 佔位符）
    loading = st.empty()
    with loading.container():
        st.write("🦊請稍候...")

    df_record, df_price, df_0050, df_00631L = load_data()
    df_history = calculate_daily_history(df_record, df_price, df_0050, df_00631L)
    loading.empty()

    # 2. 全域警報（大盤波動率 → 再平衡 → 個股停損）
    unrealized_data, _, _, _ = calculate_settlement_data(df_record, df_price)

    for msg in check_market_volatility_alerts(df_0050):
        st.warning(msg)
    for msg in check_rebalance_alerts(df_record, df_price):
        st.warning(msg)
    for msg in check_stoploss_alerts(unrealized_data):
        st.warning(msg)

    # 3. Sidebar 導覽
    with st.sidebar:
        st.title("KITSUNE KERNEL")
        st.caption("")
        selection = st.radio("選單", list(_PAGES.keys()))

    # 4. 渲染對應頁面
    _PAGES[selection](df_history, df_record, df_price)


if __name__ == "__main__":
    main()