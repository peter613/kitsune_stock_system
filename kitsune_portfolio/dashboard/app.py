"""
app_port.py
Kitsune Kernel — 投資者入口
啟動：streamlit run app_port.py
"""

import streamlit as st
from kitsune_portfolio.dashboard.engine import load_data, load_subscribers
from kitsune_portfolio.dashboard.pages import page_subscriber_nav, page_subscriber_performance

st.set_page_config(page_title="Kitsune Portfolio", layout="wide")

_CSS = """
<style>
/* ── 留白修正 ── */
[data-testid="stMainBlockContainer"] { padding-top: 2.0rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #fcf5ed !important;
    color: #1a1a1a !important;
    font-size: 1.05rem !important;
}
/* ── 左頁置頂空白 ── */
[data-testid="stSidebarUserContent"] {
    margin-top: -50px !important;
}
/* ── 選單page間隔 ── */
[data-testid="stSidebar"] .stRadio label {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}
/* ── 標題 ── */
.stApp h1 {
    font-size: 1.5rem !important;
    padding-top: 1.0rem !important;
    padding-bottom: 0.8rem !important;
    margin-top: 0rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
}
.stApp h3 {
    font-size: 1.2rem !important;
    margin-top: -5px !important;
    padding-bottom: 0.8rem !important;
    font-weight: 400 !important;
}

/* ── Metric label（小字） ── */
[data-testid="stMetricLabel"] {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #333333 !important;
}

/* ── Metric value（大數字，紅色）── */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #C0392B !important;
    padding-top: 0.3rem !important;
}

/* ── 一般 markdown 文字（排除 MetricValue 內的）── */
[data-testid="stMarkdownContainer"] p {
    color: #1a1a1a !important;
    font-size: 1.2rem !important;
    margin-bottom: 5px !important;
}
</style>
"""

_PAGES = {
    " 基金概況": page_subscriber_nav,
    " 月績效":   page_subscriber_performance,
}

def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    loading = st.empty()
    with loading.container():
        st.write("🦊 請稍候...")

    df_history, df_flows = load_data()
    loading.empty()

    subscribers = load_subscribers(df_flows)
    if not subscribers:
        st.error("transaction_quant.csv 中找不到任何認購者，請確認 subscriber 欄位。")
        return

    with st.sidebar:
        st.markdown("<h1 style='margin-bottom: -15px;'>小狐狸金庫</h1>", unsafe_allow_html=True)
        st.markdown(
            f'<span style="font-weight: normal; font-size: 16px;">'
            f'資料日期：{df_history["date"].max().strftime("%Y/%m/%d")}</span>',
            unsafe_allow_html=True,
        )
        st.divider()

        selected_name = st.radio("投資人", list(reversed(subscribers.keys())))
        st.divider()
        selected_page = st.radio("頁面", list(_PAGES.keys()))

    subscriber = subscribers[selected_name]
    df_equity  = subscriber.calculate_equity(df_history, df_flows)

    _PAGES[selected_page](df_equity, selected_name)

if __name__ == "__main__":
    main()