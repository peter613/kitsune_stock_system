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

from fund_stock.management.dashboard.engine import (
    calculate_settlement_data,
    calc_beta,
    calc_dd_series,
    calculate_quality_comparison,
    calculate_subscriber_holding_ratio,
    calculate_macro_drawdown,
)

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
            rangebreaks=[dict(bounds=["sat", "mon"])],
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
    col1.metric("成立時間", "2026-03-10")
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
    st.plotly_chart(fig1, width='stretch')

    # ── 持股比例長條圖 ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("持股比例圖")
    render_holding_ratio_chart()


def render_holding_ratio_chart(df_flows=None):
    """
    渲染投資人持股比例長條圖：
    - x 軸為 %（如 90.5%）
    - y 軸為人名
    - 由比例高到低，上往下排序
    - 顏色統一用深藍色 (#1A2B4C)
    - 姓名左側常駐高對比顯示 + 長條外側雙重標記
    """
    df_holding = calculate_subscriber_holding_ratio(df_flows)
    if df_holding.empty:
        st.info("尚無認購持股數據。")
        return

    # 升序排列傳給 Plotly 水平長條圖：最大值排在最上方
    df_plot = df_holding.sort_values('ratio', ascending=True).reset_index(drop=True)

    names = df_plot['subscriber'].tolist()
    ratios = df_plot['ratio'].tolist()
    shares = df_plot['share'].tolist()

    max_ratio = max(ratios) if ratios else 0.0
    x_max = max(100.0, float(max_ratio) * 1.25)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ratios,
        y=names,
        orientation='h',
        marker=dict(
            color="#7B7A7A",
            line=dict(color='#1A2B4C', width=1)
        ),
        # 長條外側直接常駐顯示【姓名 + 百分比】，一目瞭然
        text=[f"  <b>{name}</b>  {r:.1f}%" for name, r in zip(names, ratios)],
        textposition='outside',
        textfont=dict(size=14, color='#1A2B4C'),
        cliponaxis=False,
        customdata=shares,
        hovertemplate='<b>%{y}</b><br>持股比例: %{x:.2f}%<br>持有單位: %{customdata:,.2f} UNITS<extra></extra>',
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(200, len(df_plot) * 55 + 70),
        margin=dict(t=20, l=20, r=90, b=20),
        xaxis=dict(
            title_text="持股比例 (%)",
            ticksuffix="%",
            range=[0, x_max],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.35)",
            tickfont=dict(size=13, color="#333333"),
        ),
        yaxis=dict(
            type="category",
            showticklabels=False,  # 隱藏左側姓名標籤（長條右側已有資訊）
            showline=False,
            showgrid=False,
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#1A2B4C",
            font_size=14,
            font_color="black"
        ),
        showlegend=False
    )
    st.plotly_chart(fig, width='stretch')

    # # 認購持股明細表（輔助速查）
    # df_table = df_holding[['subscriber', 'share', 'ratio_str']].copy()
    # df_table.columns = ['投資人', '持有單位數 (UNITS)', '持股比例']
    # df_table['持有單位數 (UNITS)'] = df_table['持有單位數 (UNITS)'].map('{:,.2f}'.format)
    # st.dataframe(df_table, width='stretch', hide_index=True)

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
    col3.metric("距離高點",        f"{np.abs(distance_high):.2f}%")

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
            textposition="top left",
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
    st.plotly_chart(fig, width='stretch')

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
    daily_ret  = df_history['roi_total'].diff().dropna()
    ret_0050   = df_history['roi_0050'].diff().dropna()
    ret_00631L = df_history['roi_00631L'].diff().dropna()

    beta_0050  = calc_beta(daily_ret, ret_0050)
    beta_631L  = calc_beta(daily_ret, ret_00631L)

    # 簡易 CAPM Alpha (不計無風險利率)
    alpha_0050 = roi_window - (beta_0050 * bm_0050)
    alpha_631L = roi_window - (beta_631L * bm_00631L)

    win_rate   = (daily_ret > 0).sum() / len(daily_ret) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("累積報酬",  f"{roi_window:.2f}%")
    col2.metric("0050 報酬", f"{bm_0050:.2f}%",   help=f"Alpha (CAPM): {alpha_0050:+.2f}%")
    col3.metric("正二 報酬", f"{bm_00631L:.2f}%", help=f"Alpha (CAPM): {alpha_631L:+.2f}%")

    configs = [
        {"col": "roi_total",  "name": "投資組合",      "color": "#4F247E", "width": 3.5, "visible": True,         "dash": "solid"},
        {"col": "roi_0050",   "name": "大盤 (0050)",   "color": "#7B7A7A", "width": 2, "visible": True,         "dash": "dash"},
        {"col": "roi_00631L", "name": "正二 (00631L)", "color": "#B39DDB", "width": 2, "visible": "legendonly", "dash": "dash"},
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
                line=dict(color=c['color'], width=c['width'], dash=c['dash']),
                showlegend=(i == 0),        # ← 只有第一組 period 顯示圖例
                legendgroup=c['name'],      # ← 同名的 trace 綁在一起，點一下全開/關
                visible=c['visible'] if i == 0 else False,
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
        showlegend=True,            # ← 開啟
        legend=dict(
            orientation="h",        # 水平排列，放在上方
            x=0.28,                 # 對齊你原本 annotation 的位置
            y=1.08,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=13),
            itemclick="toggle",          # 點擊切換單條
            itemdoubleclick="toggleothers",  # 雙擊只顯示這條
        ),
        annotations=[],             # ← 移除手動圖例 annotation
        updatemenus=[{**UPDATEMENU_STYLE, "buttons": buttons}]
    )
    fig.update_xaxes(rangeslider=dict(visible=False), type="date")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)

    st.plotly_chart(fig, width='stretch')

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
    downside_std = daily_ret[daily_ret < 0].std()
    sortino = (daily_ret.mean() / downside_std * (252 ** 0.5)) if downside_std > 0 else 0
    ann_ret    = (1 + roi_window / 100) ** (252 / max(len(df_history), 1)) - 1
    calmar     = ann_ret / abs(max_dd / 100) if max_dd < 0 else 0
    port_vol   = daily_ret.std() * (252 ** 0.5) * 100
    ret_0050   = df_history['roi_0050'].diff().dropna()
    beta_0050  = calc_beta(daily_ret, ret_0050)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Sharpe",     f"{sharpe:.2f}")
    col2.metric("Sortino",    f"{sortino:.2f}")
    col3.metric("Calmar",     f"{calmar:.2f}")
    col4.metric("波動率",     f"{port_vol:.1f}%")
    col5.metric("Beta", f"{beta_0050:.2f}")
    col6.metric("最大回撤",   f"{max_dd:.2f}%")

    dd_port   = df_history['dd_pct'] * 100
    dd_0050   = calc_dd_series(df_history['roi_0050'])
    dd_00631L = calc_dd_series(df_history['roi_00631L'])

    # ── 圖2：回撤面積圖 (加入對照線) ──
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_history['date'],
        y=dd_port,
        fill='tozeroy',
        fillcolor='rgba(255,51,51,0.08)',
        line=dict(color="#C0392B", width=1.5),
        name="投資組合",
        visible=True,
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    fig1.add_trace(go.Scatter(
        x=df_history['date'],
        y=dd_0050,
        line=dict(color="#9E9E9E", width=1.5, dash="dash"),
        name="大盤 (0050)",
        visible=True,
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    fig1.add_trace(go.Scatter(
        x=df_history['date'],
        y=dd_00631L,
        line=dict(color="#B39DDB", width=1.5, dash="dash"),
        name="正二 (00631L)",
        visible="legendonly",
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    fig1.update_layout(**customize_plotly("回撤 (%)"), showlegend=True)
    fig1.update_xaxes(rangeslider=dict(visible=False), type="date")
    st.plotly_chart(fig1, width='stretch')

    st.subheader("品質比較表")
    df_quality_table = calculate_quality_comparison(df_history)
    st.dataframe(df_quality_table, width='stretch', hide_index=True)

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
    st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════
# 頁面 6：持倉概況
# ══════════════════════════════════════════
def page_unrealized(df_record, df_price, df_history):
    macro_drawdown, entry_date_str = calculate_macro_drawdown(df_record, df_history)
    st.title("持倉概況")
    if df_record.empty:
        st.warning("無數據")
        return

    unrealized_data, realized_data, total_unrealized, total_realized = calculate_settlement_data(df_record, df_price)


    # ── 未實現 ──
    if unrealized_data:
        df_un = pd.DataFrame(unrealized_data)
        total_un_pnl = df_un["未實現損益"].sum()
        total_un_mv = df_un["當前市值"].sum()
        total_un_cost = total_un_mv - total_un_pnl
        total_un_roi = (total_un_pnl / total_un_cost * 100) if total_un_cost > 0 else 0
        # 將字串轉為浮點數進行排序
        df_un["sort_roi"] = df_un["累積報酬率"].str.replace("%", "").astype(float)
        df_un = df_un.sort_values("sort_roi", ascending=False)
        # 基金經理人不看絕對金額，只保留報酬率
        df_un["標的"] = df_un["代號"].astype(str) + " " + df_un["名稱"]
        df_un_display = df_un[["標的", "股數", "累積報酬率", "高點回撤"]]

        col1, col2 = st.columns(2)
        col1.metric("未實現損益",     f"{total_un_roi:.2f}%")
        col2.metric(f"高點回撤 ({entry_date_str} 起)",    f"{min(0.0, macro_drawdown):.2f}%")

        st.dataframe(df_un_display, width='stretch', hide_index=True)

        #這裡是k線，不要可以註解
        st.markdown("---")
        st.markdown("### 持倉個股走勢 (近20日)")
        
        @st.cache_data(show_spinner=False)
        def load_local_market_data():
            try:
                df = pd.read_parquet('c:/Users/Lynn610/Documents/_kitsune_Stock_System/market_data/data/market.parquet')
                df['date'] = pd.to_datetime(df['date'])
                df['id'] = df['id'].astype(str).str.replace('.0', '', regex=False)
                return df
            except Exception:
                return pd.DataFrame()
                
        stock_list = df_un["標的"].tolist()
        if stock_list:
            selected_stock = st.selectbox("選擇持倉標的查看線圖", stock_list, label_visibility="collapsed")
            selected_id = selected_stock.split(" ")[0]
            
            with st.spinner("載入本地報價資料 (首次載入較久)..."):
                market_df = load_local_market_data()
                
            if not market_df.empty:
                stock_data = market_df[market_df['id'] == selected_id].sort_values('date')
                if len(stock_data) > 0:
                    # 取最後 80 天計算 MA (讓前 20 天也能有 60 日線)
                    stock_data = stock_data.tail(80).copy()
                    stock_data['MA10'] = stock_data['close'].rolling(window=10).mean()
                    stock_data['MA20'] = stock_data['close'].rolling(window=20).mean()
                    stock_data['MA60'] = stock_data['close'].rolling(window=60).mean()
                    
                    # 顯示近 20 天
                    display_data = stock_data.tail(20)
                    
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.08, row_heights=[0.5, 0.25, 0.25],
                                        subplot_titles=("", "成交量", "法人買賣超"))
                    
                    # 1. K線
                    fig.add_trace(go.Candlestick(x=display_data['date'],
                                                 open=display_data['open'],
                                                 high=display_data['high'],
                                                 low=display_data['low'],
                                                 close=display_data['close'],
                                                 increasing_line_color='red', 
                                                 decreasing_line_color='green',
                                                 name="K線"), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=display_data['date'], y=display_data['MA10'], line=dict(color='blue', width=1.0), name="MA10"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=display_data['date'], y=display_data['MA20'], line=dict(color='orange', width=1.0), name="MA20"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=display_data['date'], y=display_data['MA60'], line=dict(color='purple', width=1.0), name="MA60"), row=1, col=1)
                    
                    # 2. 成交量 (降低飽和度)
                    vol_colors = ['rgba(255, 0, 0, 0.3)' if c >= o else 'rgba(0, 128, 0, 0.3)' for c, o in zip(display_data['close'], display_data['open'])]
                    fig.add_trace(go.Bar(x=display_data['date'], y=display_data['volume'], marker_color=vol_colors, showlegend=False), row=2, col=1)
                    
                    # 3. 籌碼 (堆疊，外資下、投信上，降低飽和度)
                    fig.add_trace(go.Bar(x=display_data['date'], y=display_data.get('Foreign_Net', 0), marker_color='rgba(255, 0, 0, 0.4)', name="外資", showlegend=False), row=3, col=1)
                    fig.add_trace(go.Bar(x=display_data['date'], y=display_data.get('Trust_Net', 0), marker_color='rgba(0, 0, 255, 0.4)', name="投信", showlegend=False), row=3, col=1)
                    
                    # 隱藏下方時間滑桿，過濾週末空白
                    dt_all = pd.date_range(start=display_data['date'].min(), end=display_data['date'].max())
                    dt_obs = [d.strftime("%Y-%m-%d") for d in display_data['date']]
                    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
                    
                    fig.update_layout(height=800, margin=dict(l=20, r=20, t=30, b=20),
                                      barmode='relative',
                                      xaxis_rangeslider_visible=False,
                                      xaxis_rangebreaks=[dict(values=dt_breaks)],
                                      showlegend=True,
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                                      
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("本地資料庫找不到此股票的歷史資料。")
    else:
        st.info("目前投組全數持有現金（空倉防禦狀態），無任何持倉股票")


# ══════════════════════════════════════════
# 頁面 7：週期獲利
# ══════════════════════════════════════════
def page_realized(df_record, df_price, df_history):
    st.title("週期獲利")
    if df_record.empty:
        st.warning("無數據")
        return
        
    _, realized_data, _, _ = calculate_settlement_data(df_record, df_price)

    # ── 已實現 ──
    if realized_data:
        df_re = pd.DataFrame(realized_data)
        total_re_pnl = df_re["已實現損益"].sum()
        total_re_cost = df_re["總成本"].sum()
        total_re_roi = (total_re_pnl / total_re_cost * 100) if total_re_cost > 0 else 0
        
        # ── 週期平均期望值 ──
        cycle_avg_expected = 0.0
        if "進場週期" in df_re.columns:
            cycle_stats = df_re.groupby("進場週期").agg(
                總已實現損益=("已實現損益", "sum"),
                總成本=("總成本", "sum")
            ).reset_index()
            cycle_stats["週期報酬率"] = (cycle_stats["總已實現損益"] / cycle_stats["總成本"]) * 100
            cycle_avg_expected = cycle_stats["週期報酬率"].mean()

        # ── 年化報酬 (基於天數) ──
        annual_return = 0.0
        years = 0.0
        if not df_history.empty:
            first_day = df_history["date"].iloc[0]
            last_day = df_history["date"].iloc[-1]
            days = (last_day - first_day).days
            if days > 0:
                years = days / 365.0
                roi_window = (df_history["nav_per_unit"].iloc[-1] / df_history["nav_per_unit"].iloc[0])
                annual_return = (roi_window ** (1 / years) - 1) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("平均單筆報酬", f"{total_re_roi:.2f}%")
        col2.metric("週期期望值", f"{cycle_avg_expected:.2f}%")
        col3.metric(f"年化報酬 ({years:.2f}年)", f"{annual_return:.2f}%")
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # 建立週期報酬柱狀圖
        if "進場週期" in df_re.columns:
            cycle_stats = df_re.groupby("進場週期").agg(
                總已實現損益=("已實現損益", "sum"),
                總成本=("總成本", "sum")
            ).reset_index()
            
            # 排序週期 (確保時間軸順序)
            cycle_stats = cycle_stats.sort_values("進場週期")
            
            # 計算週期總報酬率
            cycle_stats["週期報酬率"] = (cycle_stats["總已實現損益"] / cycle_stats["總成本"]) * 100
            
            # 建立 hovertext
            hover_texts = []
            for cycle in cycle_stats["進場週期"]:
                sub_df = df_re[df_re["進場週期"] == cycle]
                cycle_ret = cycle_stats.loc[cycle_stats["進場週期"]==cycle, "週期報酬率"].values[0]
                hover_str = f"<b>總報酬: {cycle_ret:.2f}%</b><br><br>"
                for _, row in sub_df.iterrows():
                    hover_str += f"{row['名稱']} ({row['代號']}): {row['累積報酬率']}<br>"
                hover_texts.append(hover_str)
                
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=cycle_stats["進場週期"],
                    y=cycle_stats["週期報酬率"],
                    marker_color="#4F247E",
                    hovertext=hover_texts,
                    hoverinfo="text"
                )
            ])
            fig_bar.update_layout(
                template="plotly_dark",
                yaxis_title="總報酬率 (%)",
                xaxis_title="進場週期",
                margin=dict(t=40, l=10, r=10, b=10),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=15,
                    font_family="Arial",
                    font_color="black"
                )
            )
            df_re["sort_roi"] = df_re["累積報酬率"].str.replace("%", "").astype(float)
            
            view_mode = st.selectbox("圖表模式", ["獲利紀錄", "投組直方圖", "單筆交易直方圖"], label_visibility="collapsed")
            
            if view_mode == "獲利紀錄":
                st.plotly_chart(fig_bar, width='stretch')
            elif view_mode == "投組直方圖":
                port_data = cycle_stats["週期報酬率"].dropna()
                min_roi = np.floor(port_data.min() / 5) * 5
                max_roi = np.ceil(port_data.max() / 5) * 5
                if min_roi == max_roi: max_roi += 5
                
                fig_dist_port = go.Figure()
                fig_dist_port.add_trace(go.Histogram(
                    x=port_data,
                    xbins=dict(start=min_roi, end=max_roi, size=5),
                    marker=dict(color="#4F247E", line=dict(color="white", width=3.5)),
                    texttemplate="%{y}",
                    name="出現月數",
                ))

                # KDE 曲線 (投組直方圖)
                if len(port_data) > 1 and port_data.std() > 0:
                    from scipy.stats import gaussian_kde
                    kde_port = gaussian_kde(port_data)
                    x_norm_port = np.linspace(min_roi, max_roi, 200)
                    bin_width = 5.0
                    y_norm_port = len(port_data) * bin_width * kde_port(x_norm_port)
                    fig_dist_port.add_trace(go.Scatter(
                        x=x_norm_port,
                        y=y_norm_port,
                        mode='lines',
                        name='KDE 分布',
                        line=dict(color="red", width=3, shape='spline', dash="dot"),
                        hoverinfo='skip'
                    ))

                # # 加上 0% 紅色虛線
                # fig_dist_port.add_vline(
                #     x=0,
                #     line_color="red",
                #     line_width=2.0
                # )
                fig_dist_port.update_layout(
                    template="plotly_dark",
                    yaxis_title="出現月數",
                    xaxis_title="投組獲利區間",
                    xaxis=dict(tick0=min_roi, dtick=5, ticksuffix="%"),
                    margin=dict(t=40, l=10, r=10, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_dist_port, width='stretch')
            else:
                roi_data = df_re["sort_roi"].dropna()
                min_roi = np.floor(roi_data.min() / 5) * 5
                max_roi = np.ceil(roi_data.max() / 5) * 5
                if min_roi == max_roi: max_roi += 5
                
                fig_dist = go.Figure()

                # 直方圖（使用 #C0392B，白色間距加寬）
                fig_dist.add_trace(go.Histogram(
                    x=roi_data,
                    xbins=dict(start=min_roi, end=max_roi, size=5),
                    marker=dict(color="#4F247E", line=dict(color="white", width=3.5)),
                    texttemplate="%{y}",
                    name="交易次數",
                ))

                # KDE 曲線 (單筆交易直方圖)
                if len(roi_data) > 1 and roi_data.std() > 0:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(roi_data)
                    x_norm = np.linspace(min_roi, max_roi, 200)
                    bin_width = 5.0
                    y_norm = len(roi_data) * bin_width * kde(x_norm)
                    fig_dist.add_trace(go.Scatter(
                        x=x_norm,
                        y=y_norm,
                        mode='lines',
                        name='KDE 分布',
                        line=dict(color="red", width=3, shape='spline', dash="dot"),
                        hoverinfo='skip'
                    ))

                # # 加上 0% 紅色虛線
                # fig_dist.add_vline(
                #     x=0,
                #     line_color="red",
                #     line_width=2.0
                # )
                fig_dist.update_layout(
                    template="plotly_dark",
                    yaxis_title="交易次數",
                    xaxis_title="獲利區間",
                    xaxis=dict(tick0=min_roi, dtick=5, ticksuffix="%"),
                    margin=dict(t=40, l=10, r=10, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_dist, width='stretch')
            
            # 準備 Top 5 排行榜
            df_re_top5 = df_re.sort_values("sort_roi", ascending=False).head(5)
            
            # 建立標的名稱
            df_re_top5["標的"] = df_re_top5["代號"].astype(str) + " " + df_re_top5["名稱"]
            # 只顯示關鍵欄位
            df_re_display = df_re_top5[["進場週期", "標的", "累積報酬率"]]
        else:
            df_re["標的"] = df_re["代號"].astype(str) + " " + df_re["名稱"]
            df_re_display = df_re[["日期", "標的", "累積報酬率"]]

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("最佳個股報酬 (Top 5)")
        st.dataframe(df_re_display, width='stretch', hide_index=True)
    else:
        st.subheader("已實現損益 │ 0.00%")

    st.markdown("---")
