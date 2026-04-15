"""
pages.py
Kitsune Kernel — 頁面渲染層
職責：接收計算結果，負責所有 Streamlit / Plotly 渲染
規則：本檔案不得進行任何業務計算，只調用 engine 模組的函式
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import calculate_settlement_data

# ══════════════════════════════════════════
# 頁面 1：累積報酬率
# ══════════════════════════════════════════
def page_performance_pct(df_history):
    st.title("累積報酬率(%)")
    if df_history.empty:
        st.warning("無數據")
        return

    last_row   = df_history.iloc[-1]
    roi_pct    = last_row['roi_total']  * 100
    roi_0050   = last_row['roi_0050']   * 100
    roi_00631  = last_row['roi_00631L'] * 100
    alpha_0050 = (((1 + last_row['roi_total']) / (1 + last_row['roi_0050']))   - 1) * 100
    alpha_00631l = (((1 + last_row['roi_total']) / (1 + last_row['roi_00631L'])) - 1) * 100
    max_dd     = df_history["dd_pct"].min() * 100

    col1, col2, col3, = st.columns(3)
    col1.metric("累積報酬", f"{roi_pct:.2f}%")
    col2.metric("超額報酬",  f"{alpha_0050:.2f}%",
                help=f"正二超額報酬 : {alpha_00631l:.2f}%")
    col3.metric("最大回撤",  f"{max_dd:,.2f}%")

    # ── 主要折線圖 ────────────────────────
    fig = go.Figure()
    configs = [
        {"y": "roi_total",  "name": "投資組合", "color": "#4F247E", "width": 2.5,   "visible": True},
        {"y": "roi_0050",   "name": "大盤 (0050)",         "color": "#68CCFD", "width": 2,   "visible": True},
        {"y": "roi_00631L", "name": "正二 (00631L)",       "color": "#A172B6", "width": 2,   "visible": "legendonly"},
        {"y": "roi_ai",     "name": "AI 投資",             "color": "#5F6DEE", "width": 1.5, "visible": "legendonly"},
        {"y": "roi_etf",    "name": "ETF 投資",            "color": "#47FB1F", "width": 1.5, "visible": "legendonly"},
        {"y": "roi_oth",    "name": "Other",               "color": "#7F8C8D", "width": 1.5, "visible": "legendonly"},
    ]   
    for c in configs:
        fig.add_trace(go.Scatter(
            x=df_history['date'].dt.strftime('%Y-%m-%d'),
            y=df_history[c['y']] * 100,
            name=c['name'], mode='lines',
            line=dict(color=c['color'], width=c['width']),
            visible=c['visible'],
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
    fig.update_layout(
        template="plotly_dark", hovermode="x unified",
        xaxis=dict(type="category", tickformat="%Y-%m-%d"),
        yaxis=dict(title="累積報酬率", ticksuffix="%")
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 回撤折線圖 ────────────────────────
    st.subheader("風險監控：投組最大回撤幅度")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'),
        y=df_history['dd_pct'] * 100,
        fill='tozeroy', fillcolor='rgba(255, 51, 51, 0.2)',
        line=dict(color="#505050", width=1.5), name="回撤(%)",
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    fig_dd.update_layout(
        template="plotly_dark", height=250, margin=dict(t=10, b=10),
        xaxis=dict(type="category", tickformat="%Y-%m-%d"),
        yaxis=dict(title="回撤 %", ticksuffix="%")
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ══════════════════════════════════════════
# 頁面 2：累積權益淨值
# ══════════════════════════════════════════
def page_performance_nav(df_history, df_record, df_price):
    st.title("累積權益(淨值)")
    if df_history.empty:
        st.warning("無數據")
        return

    # ── 核心變數計算 ──
    INIT_CAP = 335800
    last_row = df_history.iloc[-1]

    # 歷史總投入 (包含 INIT_CAP 與所有後續入金)
    ti = last_row['principal'] 
    
    # 計算真實的累積提款 (從現金紀錄中找 action='out' 的絕對值總和)
    cash_rec = df_record[df_record['id'] == 'cash']
    total_withdraw = abs(cash_rec[cash_rec['action'] == 'out']['cash_flow'].sum())
    
    # 淨投入本金 = 歷史總投入 - 累積提領
    net_principal = ti - total_withdraw 
    
    # 從 settlement 取得損益明細
    _, _, total_unrealized, total_realized, _ = calculate_settlement_data(df_record, df_price)
    
    # 總權益
    tw = last_row['total_nav']
    # 累積股利 (日本生活費專戶)
    div = last_row.get('div_cash', 0.0)
    
    # 總資本利得 = 總權益 - 淨投入本金 - 股利
    capital_gain = tw - net_principal - div
    # 股票總損益 = 資本利得 + 股利
    total_profit = tw - net_principal

    #===============
    col1, col2, col3 = st.columns(3)

    # 指標 1：總權益
    col1.metric("總權益" if tw<15000000 else "總權益 (你，終抵繁星)", f"${tw:,.0f}")
    
    # 指標 2：股票總損益 (隱藏利得與股利細節)
    help_profit = f"""
    ・資本利得：{capital_gain:,.0f}\n\n
    　(未實現 {total_unrealized:,.0f}  |  已實現 {total_realized:,.0f})\n\n
    ・累積股利：{div:,.0f}
    """
    col2.metric("股票總損益", f"${total_profit:,.0f}", help=help_profit)
    
    # 指標 3：淨投入本金 (隱藏出入金細節)
    help_principal = f"""
    ・初始本金：{INIT_CAP:,.0f}\n\n
    ・投入：{ti - INIT_CAP:,.0f}\n\n
    ・提領：{total_withdraw:,.0f}\n\n
    """

    label_name = "淨投入" if net_principal >= 0 else "淨提取"
    display_value = f"${net_principal:,.0f}" if net_principal >= 0 else f"${abs(net_principal):,.0f}"
    col3.metric(label_name, display_value, help=help_principal)

    # ── 主要折線圖 ────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'), y=df_history['total_nav'],
        name="總淨值(NAV)", line=dict(color='#4F247E'),
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'), y=df_history['principal'],
        name=label_name, line=dict(color='#68CCFD', dash='dot'),
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        template="plotly_dark", yaxis_title="TWD", hovermode="x unified",
        yaxis=dict(tickformat=",.0f"), xaxis=dict(type='category')
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 回撤金額折線圖 ────────────────────
    st.subheader("風險監控：資產回撤金額")
    fig_dd_twd = go.Figure()
    fig_dd_twd.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'), y=df_history['dd_twd'],
        fill='tozeroy', fillcolor='rgba(255, 51, 51, 0.2)',
        line=dict(color='#505050', width=1.5), name="回撤金額",
        hovertemplate='$%{y:,.0f}<extra></extra>'
    ))
    fig_dd_twd.update_layout(
        template="plotly_dark", height=250, margin=dict(t=10, b=10),
        xaxis=dict(type="category", tickformat="%Y-%m-%d"),
        yaxis=dict(title="TWD")
    )
    st.plotly_chart(fig_dd_twd, use_container_width=True)


# ══════════════════════════════════════════
# 頁面 3：配置比例
# ══════════════════════════════════════════
def page_allocation(df_history, df_record, df_price):
    st.title("目前配置比例")
    if df_record.empty:
        st.warning("無數據")
        return

    # 1. 取得最新報價與持倉
    df_trade = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}

    allocation_data = []

    # ── 處理股票部位 ────────────────────────
    for aid in df_trade['id'].unique():
        sub = df_trade[df_trade['id'] == aid]
        shares = sub[sub['action'] == 'buy']['share'].sum() - sub[sub['action'] == 'sell']['share'].sum()
        
        if shares > 0:
            px_val = latest_px.get(aid, 0.0)
            if px_val == 0.0:
                last_buy = sub[sub['action'] == 'buy']
                px_val = last_buy['price'].iloc[-1] if not last_buy.empty else 0.0
            
            if px_val > 0:
                # 取得該 ID 的標籤並美化
                raw_label = str(sub['label'].iloc[0]).upper() if 'label' in sub.columns else "OTHER"
                name = sub['name'].iloc[0] if 'name' in sub.columns else ""
                label_str = f"{aid.upper()} {name}".strip()
                mv = shares * px_val
                
                allocation_data.append({
                    "層級一": raw_label,
                    "層級二": label_str,
                    "市值": mv
                })

    # ── 處理現金池 (含 DIV. 特別股拆分) ──────────
    if not df_history.empty:
        last_h = df_history.iloc[-1]
        total_cash = last_h['current_cash']
        div_cash   = last_h.get('div_cash', 0.0)
        base_cash  = total_cash - div_cash

        if div_cash > 1:
            allocation_data.append({"層級一": "CASH", "層級二": "DIV.", "市值": div_cash})
        if base_cash > 1:
            allocation_data.append({"層級一": "CASH", "層級二": "CAPITAL ", "市值": base_cash})

    if not allocation_data:
        st.info("無持倉配置")
        return

    df_alloc = pd.DataFrame(allocation_data)

    # 2. 轉換為 go.Sunburst 要求的結構
    ids, labels, parents, values, colors = [], [], [], [], []
    color_map = {
        'AI': "#3447F9", 
        'ETF': "#47FB1F", 
        'OTHER': '#7F8C8D', 
        'CASH': '#F8B54A'
    }

    # 處理大類別 (父層)
    parent_totals = df_alloc.groupby('層級一')['市值'].sum().to_dict()
    for p, val in parent_totals.items():
        if val > 0:
            ids.append(p)
            labels.append(p)
            parents.append("") 
            values.append(val)
            colors.append(color_map.get(p, "#7F8C8D"))

    # 處理細項 (子層)
    for _, row in df_alloc.iterrows():
        p = row['層級一']
        c = row['層級二']
        v = row['市值']
        if v > 0:
            ids.append(f"{c} ({p})") # 確保 ID 唯一性，避免重複錯誤
            labels.append(c)
            parents.append(p)
            values.append(v)
            colors.append(color_map.get(p, "#7F8C8D")) # 子層繼承大類的顏色

    # 3. 繪製旭日圖 (捨棄 maxdepth=1，恢復點擊展開功能)
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo="label+percent parent", # 維持你的文字比例設定
        hovertemplate="<b>%{label}</b><br>市值: $%{value:,.0f}<br>佔比: %{percentParent:.1%}<extra></extra>",
        marker=dict(colors=colors)
    ))

    # 4. 優化外觀與互動 (完全保留你的樣式與顏色)
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(t=10, l=10, r=10, b=10),
        hoverlabel=dict(
            bgcolor="#fff",      
            bordercolor="#333",  
            font_size=20,        
            font_color="black"   
        ),
        font=dict(size=15)
    )

    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# 頁面 4：資產結算中心
# ══════════════════════════════════════════
def page_history(df_record, df_price):
    st.title("資產結算中心")
    if df_record.empty:
        st.warning("無數據")
        return

    unrealized_data, realized_data, total_unrealized, total_realized,_ = calculate_settlement_data(df_record, df_price)

    column_config_setting = {
        "代號": st.column_config.LinkColumn(
            "代號",
            display_text=r"quote/(\d+)" # 正規表達式：只顯示 quote/ 後面的數字
        )
    }

    st.subheader(f"未實現損益 │ ${total_unrealized:,.0f}")
    if unrealized_data:
        df_un = pd.DataFrame(unrealized_data).sort_values("當前市值", ascending=False)
        st.dataframe(df_un, use_container_width=True, hide_index=True, column_config=column_config_setting)

    st.markdown("---")

    st.subheader(f"已實現損益 │ ${total_realized:,.0f}")
    if realized_data:
        df_re = pd.DataFrame(realized_data)
        st.dataframe(df_re, use_container_width=True, hide_index=True, column_config=column_config_setting)

# ══════════════════════════════════════════
# 頁面 5：月度績效結算
# ══════════════════════════════════════════
def page_monthly_performance(df_history):
    st.title("月度績效結算")
    if df_history.empty:
        st.warning("無數據")
        return

    df = df_history.copy()

    # ── 使用 engine 計算好的策略月份（每月 11 日或順延交易日為起點）──
    if 'strategy_month' not in df.columns:
        # 保險：若舊版 engine 未產生此欄，退回固定偏移（但精度較低）
        df['strategy_month'] = (df['date'] - pd.Timedelta(days=10)).dt.strftime('%Y-%m')

    df['month'] = df['strategy_month']
    df['daily_ret'] = df['total_nav'].pct_change().fillna(0)
    m_groups    = df.groupby('month')
    monthly_end = m_groups.last().reset_index()

    # ── 月報酬、大盤月報酬、Alpha ─────────
    monthly_end['prev_nav']        = monthly_end['roi_total'].shift(1).fillna(0) + 1.0
    monthly_end['monthly_ret_pct'] = ((monthly_end['roi_total'] + 1.0) / monthly_end['prev_nav'] - 1) * 100
    monthly_end.loc[0, 'monthly_ret_pct'] = monthly_end.loc[0, 'roi_total'] * 100

    monthly_end['prev_0050'] = monthly_end['roi_0050'].shift(1).fillna(0) + 1.0
    monthly_end['m_0050']    = ((monthly_end['roi_0050'] + 1.0) / monthly_end['prev_0050'] - 1) * 100
    monthly_end.loc[0, 'm_0050'] = monthly_end.loc[0, 'roi_0050'] * 100

    monthly_end['prev_00631L'] = monthly_end['roi_00631L'].shift(1).fillna(0) + 1.0
    monthly_end['m_00631L']    = ((monthly_end['roi_00631L'] + 1.0) / monthly_end['prev_00631L'] - 1) * 100
    monthly_end.loc[0, 'm_00631L'] = monthly_end.loc[0, 'roi_00631L'] * 100

    monthly_end['monthly_alpha_pct'] = (
        ((1 + monthly_end['monthly_ret_pct'] / 100) / (1 + monthly_end['m_0050'] / 100)) - 1
    ) * 100
    monthly_end['monthly_alpha_pct_00631L'] = (
        ((1 + monthly_end['monthly_ret_pct'] / 100) / (1 + monthly_end['m_00631L'] / 100)) - 1
    ) * 100
    
    # ── 月損益金額 ────────────────────────
    monthly_end['prev_total_nav']   = monthly_end['total_nav'].shift(1).fillna(0)
    monthly_end['principal_added']  = monthly_end['principal'] - monthly_end['principal'].shift(1).fillna(0)
    monthly_end['monthly_profit']   = (
        monthly_end['total_nav'] - monthly_end['prev_total_nav'] - monthly_end['principal_added']
    )

    # ── 月度 Sharpe / Sortino (必須移到提取 dataframe 之前) ─
    risk_stats = []
    for m, group in m_groups:
        d_ret   = group['daily_ret']
        avg, std= d_ret.mean(), d_ret.std()
        sharpe  = (avg / std) * np.sqrt(252) if std > 0 else 0
        d_std   = d_ret[d_ret < 0].std()
        sortino = (avg / d_std) * np.sqrt(252) if d_std > 0 else (sharpe if avg > 0 else 0)
        risk_stats.append({'month': m, 'Sharpe': sharpe, 'Sortino': sortino})
    monthly_end = monthly_end.merge(pd.DataFrame(risk_stats), on='month')

    # ── 量化表格 ──────────────────────────
    st.markdown("---")
    report_df = monthly_end[['month', 'monthly_ret_pct', 'monthly_profit', 'monthly_alpha_pct', 'Sharpe', 'Sortino']].copy()
    report_df.columns = ['月份', '報酬率', '損益', 'Alpha', 'Sharpe', 'Sortino']
    #將資料依「月份」由新到舊（降冪）排序
    report_df = report_df.sort_values(by='月份', ascending=False)
    st.dataframe(
        report_df.style.format({'報酬率': '{:.2f}%', '損益': '{:,.0f}', 'Alpha': '{:.2f}%', 'Sharpe': '{:.2f}', 'Sortino': '{:.2f}'}),
        use_container_width=True, hide_index=True
    )

    st.divider()

    avg_ret, med_ret   = monthly_end['monthly_ret_pct'].mean(),   monthly_end['monthly_ret_pct'].median()
    avg_prof, med_prof = monthly_end['monthly_profit'].mean(),    monthly_end['monthly_profit'].median()

    # ── 月報酬率 ───
    st.subheader("月報酬率")
    st.caption(f"Avg: {avg_ret:.1f}%  |  Med: {med_ret:.1f}%")
    colors_ret = ["#FD9292" if val >= 0 else "#A4FFA4" for val in monthly_end['monthly_ret_pct']]
    fig_ret = go.Figure(go.Bar(
        x=monthly_end['month'], y=monthly_end['monthly_ret_pct'],
        marker_color=colors_ret,
        text=monthly_end['monthly_ret_pct'].round(2).astype(str) + '%',
        textposition='outside'
    ))
    fig_ret.update_layout(
        template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20), height=500,
        xaxis=dict(type="category")
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # ── 月損益 ──
    st.subheader("月損益")
    st.caption(f"Avg: {avg_prof:,.0f}  |  Med: {med_prof:,.0f}")
    colors_prof = ['#FD9292' if val >= 0 else '#A4FFA4' for val in monthly_end['monthly_profit']]
    fig_prof = go.Figure(go.Bar(
        x=monthly_end['month'], y=monthly_end['monthly_profit'],
        marker_color=colors_prof,
        text=monthly_end['monthly_profit'].map('{:,.0f}'.format),
        textposition='outside'
    ))
    fig_prof.update_layout(
        template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20), height=500,
        xaxis=dict(type="category")
    )
    st.plotly_chart(fig_prof, use_container_width=True)

    st.caption("")

    # # ── Alpha ───
    # st.subheader("Alpha 趨勢")
    # fig_alpha = go.Figure() # 修正：先建立空白畫布
    
    # fig_alpha.add_trace(go.Scatter(
    #     x=monthly_end['month'], y=monthly_end['monthly_alpha_pct'],
    #     mode='lines+markers',
    #     name="0050",
    #     line=dict(color='#68CCFD', width=3),
    #     fill='tozeroy', fillcolor='rgba(22, 160, 133, 0.1)'
    # ))
    
    # fig_alpha.add_trace(go.Scatter(
    #     x=monthly_end['month'], y=monthly_end['monthly_alpha_pct_00631L'],
    #     mode='lines+markers',
    #     name="正二",
    #     line=dict(color='#A172B6', width=3),
    #     fill='tozeroy', fillcolor='rgba(22, 160, 133, 0.1)'
    # )) 
    
    # fig_alpha.update_layout(
    #     template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    #     margin=dict(l=20, r=20, t=30, b=20), height=300,
    #     yaxis_ticksuffix="%", xaxis=dict(type="category")
    # )
    # st.plotly_chart(fig_alpha, use_container_width=True)

    # # ── 夏普/索提諾 ───
    # st.subheader("夏普/索提諾")
    # fig_risk = go.Figure()
    # fig_risk.add_trace(go.Scatter(
    #     x=monthly_end['month'], y=monthly_end['Sharpe'],
    #     name="Sharpe", line=dict(color='#4F247E', width=2)
    # ))
    # fig_risk.add_trace(go.Scatter(
    #     x=monthly_end['month'], y=monthly_end['Sortino'],
    #     name="Sortino", line=dict(color='#db3b00', width=2, dash='dot')
    # ))
    # fig_risk.update_layout(
    #     template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    #     margin=dict(l=20, r=20, t=30, b=20), height=300,
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    #     xaxis=dict(type="category")
    # )
    # st.plotly_chart(fig_risk, use_container_width=True)

# ══════════════════════════════════════════
# 頁面 6：再平衡細項(待開發)
# ══════════════════════════════════════════