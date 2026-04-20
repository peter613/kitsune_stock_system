"""
pages.py
Kitsune Kernel — 頁面渲染層
職責：接收計算結果，負責所有 Streamlit / Plotly 渲染
規則：本檔案不得進行任何業務計算，只調用 engine 模組的函式
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from kitsune_fund.valuation.engine import calculate_settlement_data

# ══════════════════════════════════════════
# 初始化繪圖格式
# ══════════════════════════════════════════
#共用十字線
def customize_plotly(y_name):
    return dict(
        template="plotly_dark", yaxis_title=y_name, hovermode="x",
        yaxis=dict(tickformat=",.2f"),
        xaxis=dict(
            type='date', tickformat="%Y/%m/%d",
            showspikes=True,
            spikecolor="rgba(150,150,150,0.8)",
            spikethickness=1,
            spikedash="solid",
            spikemode="across",
            spikesnap="cursor",
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#4F247E",
            font=dict(size=13, color="black")
        )
    )

# 共用：updatemenus 按鈕設定
def build_period_buttons(df_history, vis_func):
    today = df_history['date'].max()
    cutoffs = {
        "ALL": df_history['date'].min(),
        "1M":  today - pd.DateOffset(months=1),
        "3M":  today - pd.DateOffset(months=3),
        "6M":  today - pd.DateOffset(months=6),
        "1Y":  today - pd.DateOffset(years=1),
    }
    periods = ["ALL", "1M", "3M", "6M", "1Y"]

    buttons = []
    for i, p in enumerate(periods):
        df_w    = df_history[df_history['date'] >= cutoffs[p]]
        x_start = df_w['date'].iloc[0].isoformat()
        x_end   = today.isoformat()
        buttons.append(dict(
            label=p,
            method="update",
            args=[
                {"visible": vis_func(i)},
                {"xaxis.range": [x_start, x_end]}
            ]
        ))
    return buttons, cutoffs, periods


UPDATEMENU_STYLE = dict(
    type="buttons",
    direction="right",
    x=0.0, y=1.15,
    xanchor="left",
    yanchor="top",
    active=0,
    showactive=True,
    bgcolor="#dedede",
    bordercolor="#555",
    borderwidth=1,
    font=dict(color="#000000", size=12),
    pad=dict(r=5, t=5, b=2, l=2),
)

# ══════════════════════════════════════════
# 頁面 1：概況
# ══════════════════════════════════════════
def page_fund_overview(df_history):
    st.title("基金概況")
    if df_history.empty:
        st.warning("無數據")
        return

    last_row    = df_history.iloc[-1]
    total_units = last_row['total_units']
    current_nav  = last_row['nav_per_unit']
    aum = total_units*current_nav

    col1, col2, col3 = st.columns(3)
    col1.metric("當前淨值(NAV)", f"${current_nav:,.2f}")
    col2.metric("流通股數(UNITS)", f"{total_units:,.2f}")   
    col3.metric("資金規模(AUM)",   f"${aum:,.0f}")           

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    fig1.add_trace(go.Scatter(
        x=df_history['date'],
        y=df_history['total_nav'],
        name="AUM",
        line=dict(color="#4F247E", width=2.5),
        hovertemplate='AUM: $%{y:,.0f}<extra></extra>'
    ), secondary_y=False)

    fig1.add_trace(go.Scatter(
        x=df_history['date'],
        y=df_history['total_units'],
        name="Units",
        line=dict(color="#68CCFD", width=2.0, dash='dot'),
        hovertemplate='Units: %{y:,.2f}<extra></extra>'
    ), secondary_y=True)

    fig1.update_layout(**customize_plotly(""))
    fig1.update_yaxes(title_text="規模 (TWD)", secondary_y=False)
    fig1.update_yaxes(title_text="單位數",     secondary_y=True, showgrid=False)
    st.plotly_chart(fig1, use_container_width=True)

# ══════════════════════════════════════════
# 頁面 2：淨值走勢
# ══════════════════════════════════════════
def page_fund_nav(df_history):
    st.title("淨值走勢")
    if df_history.empty:
        st.warning("無數據")
        return

    last_row    = df_history.iloc[-1]
    avg_nav     = df_history['nav_per_unit'].mean()
    current_nav = last_row['nav_per_unit']
    max_nav     = df_history['nav_per_unit'].max()          # 從整個 df 找最大值
    distance_high = (current_nav - max_nav) / max_nav * 100 # 改成百分比才能顯示 %

    col1, col2, col3 = st.columns(3)
    col1.metric("當前淨值", f"${current_nav:,.2f}")
    col2.metric("平均淨值",      f"${avg_nav:,.2f}")
    col3.metric("距離高點",        f"{distance_high:.2f}%")

    today = df_history['date'].max()
    cutoffs = {
        "ALL": df_history['date'].min(),
        "1M":  today - pd.DateOffset(months=1),
        "3M":  today - pd.DateOffset(months=3),
        "6M":  today - pd.DateOffset(months=6),
        "1Y":  today - pd.DateOffset(years=1),
    }
    periods = ["ALL", "1M", "3M", "6M", "1Y"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_history['date'],
        y=df_history['nav_per_unit'],
        name="淨值走勢",
        line=dict(color='#4F247E', width=2.5),
        hovertemplate='淨值: %{y:,.2f}<extra></extra>'
    ))

    for i, p in enumerate(periods):
        df_w    = df_history[df_history['date'] >= cutoffs[p]]
        avg     = df_w['nav_per_unit'].mean()
        x_start = df_w['date'].iloc[0]
        x_end   = df_w['date'].iloc[-1]

        fig.add_trace(go.Scatter(
            x=[x_start, x_end], y=[avg, avg],
            mode='lines+text',
            line=dict(color="#C0392B", width=2.0, dash='dot'),
            text=["", f"均值 {p}: {avg:,.2f}"],
            textposition="top left",
            textfont=dict(size=15, color="#C0392B"),
            showlegend=False,
            hovertemplate=f"平均淨值({p}): {avg:,.2f}<extra></extra>",
            visible=(i == 0)
        ))

        max_idx  = df_w['nav_per_unit'].idxmax()
        max_date = df_w.loc[max_idx, 'date']
        max_nav  = df_w.loc[max_idx, 'nav_per_unit']
        fig.add_trace(go.Scatter(
            x=[max_date], y=[max_nav],
            mode='markers+text',
            marker=dict(size=12, color="#FF0000", symbol='triangle-up'),
            text=[f"MAX: {max_nav:,.2f}"],
            textfont=dict(size=12, color="#FF0000"),
            textposition="top center",
            showlegend=False,
            hoverinfo='skip',
            visible=(i == 0)
        ))

    def _vis(active_i):
        vis = [True]
        for j in range(len(periods)):
            show = (j == active_i)
            vis += [show, show]
        return vis

    buttons, _, _ = build_period_buttons(df_history, _vis)

    fig.update_layout(
        **customize_plotly("NAV"),
        showlegend=False,
        updatemenus=[{**UPDATEMENU_STYLE, "buttons": buttons}]
    )
    fig.update_xaxes(rangeslider=dict(visible=False), type="date")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# 頁面 3：報酬
# ══════════════════════════════════════════
def page_fund_return(df_history):
    st.title("累積報酬")
    if df_history.empty:
        st.warning("無數據")
        return

    today = df_history['date'].max()
    cutoffs = {
        "ALL": df_history['date'].min(),
        "1M":  today - pd.DateOffset(months=1),
        "3M":  today - pd.DateOffset(months=3),
        "6M":  today - pd.DateOffset(months=6),
        "1Y":  today - pd.DateOffset(years=1),
    }
    periods = ["ALL", "1M", "3M", "6M", "1Y"]

    first = df_history.iloc[0]
    last  = df_history.iloc[-1]
    roi_window = ((1 + last['roi_total'])  / (1 + first['roi_total'])  - 1) * 100
    bm_0050    = ((1 + last['roi_0050'])   / (1 + first['roi_0050'])   - 1) * 100
    bm_00631L  = ((1 + last['roi_00631L']) / (1 + first['roi_00631L']) - 1) * 100
    alpha_0050 = roi_window - bm_0050
    alpha_631L = roi_window - bm_00631L

    daily_ret  = df_history['roi_total'].diff().dropna()
    win_rate   = (daily_ret > 0).sum() / len(daily_ret) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("累積報酬",        f"{roi_window:.2f}%")
    col2.metric("超額報酬 (0050)", f"{alpha_0050:.2f}%",
                help=f"正二超額報酬：{alpha_631L:.2f}%")
    col3.metric("日勝率",          f"{win_rate:.1f}%")

    configs = [
        {"col": "roi_total",  "name": "投資組合",      "color": "#4F247E", "width": 3.0},
        {"col": "roi_0050",   "name": "大盤 (0050)",   "color": "#68CCFD", "width": 1.25},
        {"col": "roi_00631L", "name": "正二 (00631L)", "color": "#AB86BC", "width": 1.25},
    ]
    n = len(configs)

    fig = go.Figure()

    for i, p in enumerate(periods):
        df_w = df_history[df_history['date'] >= cutoffs[p]].copy()
        for c in configs:
            base      = df_w[c['col']].iloc[0]
            y_rebased = ((1 + df_w[c['col']]) / (1 + base) - 1) * 100
            fig.add_trace(go.Scatter(
                x=df_w['date'],
                y=y_rebased,
                name=c['name'],
                mode='lines',
                line=dict(color=c['color'], width=c['width']),
                showlegend=False,   # 圖例改用 annotation 手動畫
                visible=(i == 0),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))

    def _vis(active_i):
        vis = []
        for j in range(len(periods)):
            vis += [j == active_i] * n
        return vis

    buttons, _, _ = build_period_buttons(df_history, _vis)

    # ── 手動圖例（annotation），永遠顯示在按鈕右側同高 ──
    legend_annotations = []
    for k, c in enumerate(configs):
        legend_annotations.append(dict(
            x=0.28 + k * 0.18,   # 按鈕在 x=0，圖例從 0.28 開始往右排
            y=1.08,
            xref="paper", yref="paper",
            xanchor="left",
            text=f'<b>─ {c["name"]}</b>',
            font=dict(color=c["color"], size=13),
            showarrow=False,
        ))

    fig.update_layout(
        **customize_plotly("報酬率 (%)"),
        showlegend=False,
        annotations=legend_annotations,
        updatemenus=[{**UPDATEMENU_STYLE, "buttons": buttons}]
    )
    fig.update_xaxes(rangeslider=dict(visible=False), type="date")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)

    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# 頁面 4：基金品質
# ══════════════════════════════════════════
def page_fund_quality(df_history):
    st.title("基金品質")
    if df_history.empty:
        st.warning("無數據")
        return

    first     = df_history.iloc[0]
    last      = df_history.iloc[-1]
    daily_ret = df_history['roi_total'].diff().dropna()

    # ── 全期指標 ──
    roi_window = ((1 + last['roi_total']) / (1 + first['roi_total']) - 1) * 100
    max_dd     = df_history['dd_pct'].min() * 100
    sharpe     = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0
    ann_ret    = (1 + roi_window / 100) ** (252 / max(len(df_history), 1)) - 1
    calmar     = ann_ret / abs(max_dd / 100) if max_dd < 0 else 0

    # ── 最新 rolling 20日波動率 ──
    port_vol = daily_ret.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe",   f"{sharpe:.2f}")
    col2.metric("Calmar",   f"{calmar:.2f}")
    col3.metric("20日波動率",   f"{port_vol:.1f}%")
    col4.metric("最大回撤", f"{max_dd:.2f}%")

    # ── 圖1：Rolling 20日 Sharpe ──
    rolling_sharpe = (
        daily_ret.rolling(20).mean() / daily_ret.rolling(20).std() * (252 ** 0.5)
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_history['date'].iloc[1:],
        y=rolling_sharpe,
        mode='lines',
        line=dict(color='#4F247E', width=2.0),
        hovertemplate='Sharpe: %{y:.2f}<extra></extra>'
    ))
    fig1.add_hline(
        y=0, line_dash='dot', line_color='gray', line_width=1
    )
    fig1.add_hline(
        y=1, line_dash='dot', line_color='#68CCFD', line_width=1,
        annotation_text="Sharpe = 1",
        annotation_font=dict(size=11, color="#68CCFD"),
        annotation_position="right"
    )
    fig1.update_layout(**customize_plotly("Rolling Sharpe (20日)"), showlegend=False)
    fig1.update_xaxes(rangeslider=dict(visible=False), type="date")
    st.plotly_chart(fig1, use_container_width=True)

    # ── 圖2：回撤面積圖 ──
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_history['date'],
        y=df_history['dd_pct'] * 100,
        fill='tozeroy',
        fillcolor='rgba(255,51,51,0.15)',
        line=dict(color="#C0392B", width=1.0),
        name="回撤",
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    fig2.update_layout(**customize_plotly("回撤 (%)"), showlegend=False)
    fig2.update_xaxes(rangeslider=dict(visible=False), type="date")
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════
# 頁面 5：狐狸基金資產配置
# ══════════════════════════════════════════
def page_allocation(df_history, df_record, df_price):
    st.title("目前配置比例")
    if df_record.empty:
        st.warning("無數據")
        return

    df_trade = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}

    allocation_data = []

    for aid in df_trade['id'].unique():
        sub = df_trade[df_trade['id'] == aid]
        shares = sub[sub['action'] == 'buy']['share'].sum() - sub[sub['action'] == 'sell']['share'].sum()
        if shares > 0:
            px_val = latest_px.get(aid, 0.0)
            if px_val == 0.0:
                last_buy = sub[sub['action'] == 'buy']
                px_val = last_buy['price'].iloc[-1] if not last_buy.empty else 0.0
            if px_val > 0:
                name = sub['name'].iloc[0] if 'name' in sub.columns else ""
                label_str = f"{aid.upper()} {name}".strip()
                mv = shares * px_val
                allocation_data.append({"標的": label_str, "市值": mv})

    if not df_history.empty:
        total_cash = df_history.iloc[-1]['cash']
        if total_cash > 1:
            allocation_data.append({"標的": "CASH", "市值": total_cash})

    if not allocation_data:
        st.info("無持倉配置")
        return

    df_alloc = pd.DataFrame(allocation_data)

    FUND_COLORS = [
        "#FFD1D1",  # 輕粉紅 (Soft Coral) 
        "#EAD1FF",  # 輕薰衣草紫 (Soft Orchid) 
        "#D1E4FF",  # 輕天藍 (Soft Sky) 
        "#D1FFED",  # 輕薄荷綠 (Soft Mint) 
        '#F5F5F0',
        "#FFF2CC",   # 輕奶油黃 (Soft Citrine) - 現金部位 (CASH) 

    ]

    fig = go.Figure(go.Pie(
        labels=df_alloc['標的'],
        values=df_alloc['市值'],
        hovertemplate="<b>%{label}</b><br>市值: $%{value:,.0f}<br>佔比: %{percent}<extra></extra>",
        textinfo="label+percent",
        marker=dict(
        colors=FUND_COLORS[:len(df_alloc)]),
        textfont=dict(size=16, family="Arial Black"),
        # --- 新增這個區塊來放大提示框 ---
        hoverlabel=dict(
            font_size=15,           # 放大提示框內的文字
            font_color='black',
            font_family="Arial",    # 提示框建議用一般 Arial，閱讀起來比較不壓迫
            bgcolor="white",        # 背景顏色
        ),
    ))
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(t=10, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# 頁面 6：資產結算中心
# ══════════════════════════════════════════
def page_history(df_record, df_price, df_history):
    st.title("資產結算中心")
    if df_record.empty:
        st.warning("無數據")
        return

    unrealized_data, realized_data, total_unrealized, total_realized = calculate_settlement_data(df_record, df_price)

    column_config_setting = {
        "代號": st.column_config.LinkColumn(
            "代號",
            display_text=r"quote/(\w+)"
        )
    }

    # ── 未實現 ──
    st.subheader(f"未實現損益 │ ${total_unrealized:,.0f}")
    if unrealized_data:
        df_un = pd.DataFrame(unrealized_data).sort_values("當前市值", ascending=False)
        st.dataframe(df_un, use_container_width=True, hide_index=True, column_config=column_config_setting)

    st.markdown("---")

    # ── 已實現 ──
    st.subheader(f"已實現損益 │ ${total_realized:,.0f}")
    if realized_data:
        df_re = pd.DataFrame(realized_data)
        st.dataframe(df_re, use_container_width=True, hide_index=True, column_config=column_config_setting)

    st.markdown("---")

# ══════════════════════════════════════════
# 頁面 6：再平衡細項(待開發)
# ══════════════════════════════════════════