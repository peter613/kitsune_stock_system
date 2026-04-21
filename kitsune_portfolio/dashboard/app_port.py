"""
app_port.py
Kitsune Kernel — 投資者入口
啟動：streamlit run app_port.py
"""

import streamlit as st
from kitsune_portfolio.dashboard.engine_port import load_data, load_subscribers
from pages_port import page_subscriber_nav, page_subscriber_performance

st.set_page_config(page_title="Kitsune Portfolio", layout="wide")

_CSS = """
<style>
[data-testid="stAppViewContainer"] { padding-top: 0rem !important; }
[data-testid="stMainBlockContainer"] { padding-top: 3rem !important; }
[data-testid="stSidebar"] {
    background-color: #fcf5ed !important;
    color: #1a1a1a !important;
    font-size: 1.05rem !important;
}
[data-testid="stSidebarUserContent"] { margin-top: -50px !important; }
[data-testid="stSidebar"] .stRadio label {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}
.stApp h1 {
    font-size: 1.5rem !important;
    padding-top: 1.5rem !important;
    padding-bottom: 0.8rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
}
[data-testid="stMetricLabel"] {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #333333 !important;
}
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #C0392B !important;
    padding-top: 0.3rem !important;
}
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
        st.error("transaction.csv 中找不到任何認購者，請確認 subscriber 欄位。")
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