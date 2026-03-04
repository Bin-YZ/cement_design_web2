# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pdf_generator import create_pdf_report

@st.fragment
def calculate_topsis(df, objective_config):
    """
    df: Pareto DataFrame
    objective_config: List of dicts e.g. [{'col': 'E', 'impact': '+'}, {'col': 'Cost', 'impact': '-'}]
    """
    criteria_cols = [obj['col'] for obj in objective_config if obj['col'] in df.columns]
    impacts = [obj['impact'] for obj in objective_config if obj['col'] in df.columns]
    
    if not criteria_cols:
        return np.zeros(len(df)) # Fallback

    d = df[criteria_cols].copy()
    
    # 1. Normalize
    norm_d = d / np.sqrt((d**2).sum())
    
    # 2. Weights (Assume equal importance for selected goals)
    n = len(criteria_cols)
    weights = [1.0 / n] * n
    weighted_d = norm_d * weights
    
    # 3. Ideal Solutions
    ideal_best = []
    ideal_worst = []
    
    for i, col in enumerate(criteria_cols):
        if impacts[i] == '+': # Maximize
            ideal_best.append(weighted_d[col].max())
            ideal_worst.append(weighted_d[col].min())
        else: # Minimize
            ideal_best.append(weighted_d[col].min())
            ideal_worst.append(weighted_d[col].max())
            
    # 4. Euclidean Distance
    s_plus = np.sqrt(((weighted_d - ideal_best)**2).sum(axis=1))
    s_minus = np.sqrt(((weighted_d - ideal_worst)**2).sum(axis=1))
    
    # 5. Score (Handling division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = s_minus / (s_plus + s_minus)
        scores = np.nan_to_num(scores) # Handle NaN
        
    return scores

@st.fragment
def show_results(COLORS, MATERIALS_CONFIG):
    """
    接收主程序的 COLORS 和 MATERIALS_CONFIG 配置，渲染结果仪表盘
    """
    # 如果还没有优化结果，直接返回，不渲染任何东西
    if st.session_state.get("df_pareto") is None:
        return

    # ----- 🌟 进阶优化：检查是否已缓存处理过的数据，避免每次切下拉框都重算 TOPSIS -----
    if "processed_df_display" not in st.session_state:
        # 1. 准备初始数据
        df_display = st.session_state.df_pareto.copy()
        df_all_display = st.session_state.df_all.copy() if st.session_state.get("df_all") is not None else None
        
        # 2. 通过 st.session_state 读取主程序里 Checkbox 的勾选状态
        objective_config = []
        if st.session_state.get("obj_e_max"):
            objective_config.append({'col': 'E', 'impact': '+', 'name': 'Max Strength'})
        if st.session_state.get("obj_cost_min"):
            objective_config.append({'col': 'Cost', 'impact': '-', 'name': 'Min Cost'})
        if st.session_state.get("obj_co2_min"):
            objective_config.append({'col': 'CO2_emission', 'impact': '-', 'name': 'Min CO2'})
        if st.session_state.get("obj_net_min"):
            objective_config.append({'col': 'Net_emission', 'impact': '-', 'name': 'Min Net CO2'})
        if st.session_state.get("obj_co2abs_max"):
            objective_config.append({'col': 'CO2_abs', 'impact': '+', 'name': 'Max Uptake'})
        
        if not objective_config:
            objective_config = [{'col': 'E', 'impact': '+', 'name': 'Max Strength'}]
        
        # 3. 计算 TOPSIS 分数
        df_display["Decision_Score"] = calculate_topsis(df_display, objective_config)
        df_display = df_display.sort_values("Decision_Score", ascending=False).reset_index(drop=True)

        co2_cols = ["CO2_emission", "Net_emission", "CO2_abs"]
        for col in co2_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col]
            if df_all_display is not None and col in df_all_display.columns:
                df_all_display[col] = df_all_display[col] 
                
        # 4. 保存计算好的结果到缓存
        st.session_state["processed_df_display"] = df_display
        st.session_state["processed_df_all"] = df_all_display
        st.session_state["processed_objective_config"] = objective_config

    # 直接从缓存读取，跳过重复计算
    df_display = st.session_state["processed_df_display"]
    df_all_display = st.session_state["processed_df_all"]
    objective_config = st.session_state["processed_objective_config"]

    # ----- 渲染 UI 面板 -----
    with st.expander("📊 Analytics Dashboard & Results", expanded=True):
        
        # 🌟 使用原生的 st.tabs 代替 st.radio，实现前端秒切换，不卡顿！
        tab_names = [
            "📉 Pareto Analysis", 
            "🎞️ GA Animation", 
            "🕸️ Parallel Coordinates", 
            "📋 Data Table", 
            "⚖️ Benchmark Comparison"
        ]
        tab_pareto, tab_anim, tab_parallel, tab_table, tab_bench = st.tabs(tab_names)
    
        axis_options = {
            "E": "E (GPa) - MAX",
            "CO2_abs": "CO₂ Uptake (kg/kg) - MAX",
            "CO2_emission": "CO₂ Emission (kg/kg) - MIN", 
            "Cost": "Cost (€/kg) - MIN",
            "Net_emission": "Net Emission (kg/kg) - MIN",
        }


# ==========================================
        # Tab 1: Pareto Analysis (🖱️ 点石成金：修复点击变灰问题)
        # ==========================================
        with tab_pareto:
            with st.container():
                # --- 顶部标题 ---
                t1, t2 = st.columns([12, 1], gap="small")
                with t1:
                    st.markdown("### 📈 Interactive Pareto Visualization")
                with t2:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    with st.popover("ℹ️"):
                        st.markdown("### How to play with this plot")
                        st.write("• **🖱️ Click-to-Reveal**: Click ANY point (colored or grey) to reveal its exact recipe.")
                        st.write("• **Grey points (Explored)**: Candidates evaluated during the search. Click to see which Generation they belong to.")
                        st.write("• **Colored points (Pareto)**: The optimal frontier.")

                chart_options = [k for k in axis_options.keys() if k in df_display.columns]
                
                c_ctrl = st.columns(5)
                default_x = chart_options[0] if len(chart_options) >= 1 else chart_options[0]
                default_y = chart_options[1] if len(chart_options) >= 2 else chart_options[0]
                default_z = chart_options[2] if len(chart_options) >= 3 else chart_options[0]
                
                x_axis = c_ctrl[0].selectbox("X Axis", chart_options, index=chart_options.index(default_x) if default_x in chart_options else 0, format_func=lambda x: axis_options[x], key="pareto_x")
                y_axis = c_ctrl[1].selectbox("Y Axis", chart_options, index=chart_options.index(default_y) if default_y in chart_options else 0, format_func=lambda x: axis_options[x], key="pareto_y")
                material_cols = [m[1] for m in MATERIALS_CONFIG]
                
                color_candidates = ["Decision_Score"] + chart_options + material_cols
                color_options = [c for c in color_candidates if c in df_display.columns]
                default_color_idx = color_options.index("Decision_Score") if "Decision_Score" in color_options else 0
                
                color_col = c_ctrl[2].selectbox("Color Points By", color_options, index=default_color_idx, key="pareto_color")
                show_all = c_ctrl[3].toggle("Show History", value=True, key="pareto_hist")
                
                use_3d_allowed = len(chart_options) >= 3
                use_3d = c_ctrl[4].toggle("3D View", value=False, key="pareto_3d") if use_3d_allowed else False
                
                if use_3d and use_3d_allowed:
                    z_axis = st.selectbox("Z Axis", chart_options, index=chart_options.index(default_z), format_func=lambda x: axis_options[x], key="pareto_z")
                else:
                    z_axis = None
    
                # 🌟 左边放散点图，右边放详情
                col_plot, col_details = st.columns([2.8, 1.2], gap="large")
                
                # ------ 1. 左侧渲染主图 ------

                with col_plot:
                    layout_settings = dict(
                        height=550, 
                        margin=dict(l=20, r=20, t=30, b=20), 
                        paper_bgcolor="rgba(0,0,0,0)", 
                        plot_bgcolor="rgba(255,255,255,0.5)", 
                        font=dict(family="Inter", color=COLORS["text_body"])
                    )
                    fig = go.Figure()
        
                    # 1. 渲染灰色点 (Explored)
                    if show_all and df_all_display is not None:
                        # ✨ 纯字符串标识，彻底解决 KeyError: 0
                        customdata_exp = [f"exp_{i}" for i in df_all_display.index]
                        hover_text_exp = [f"Explored Mix #{i+1}<br>Gen: {row.get('Generation', '-')}<br>🖱️ Click for details!" for i, row in df_all_display.iterrows()]
                        
                        if use_3d and z_axis:
                            fig.add_trace(go.Scatter3d(
                                x=df_all_display[x_axis], y=df_all_display[y_axis], z=df_all_display[z_axis], 
                                mode="markers", name="Explored", 
                                marker=dict(size=3, color="lightgrey", opacity=0.35), 
                                customdata=customdata_exp, text=hover_text_exp, hoverinfo="text"
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=df_all_display[x_axis], y=df_all_display[y_axis], 
                                mode="markers", name="Explored", 
                                marker=dict(size=6, color="lightgrey", opacity=0.4),
                                customdata=customdata_exp, text=hover_text_exp, hoverinfo="text",
                                # ✅ 修复：删除了不允许的 line 属性，用改变颜色和放大来高亮
                                selected=dict(marker=dict(size=12, color=COLORS['accent'], opacity=1.0)),
                                unselected=dict(marker=dict(color="lightgrey", opacity=0.4))
                            ))
                    
                    # 2. 渲染彩色点 (Pareto)
                    colorbar_title = "Decision Score<br>(TOPSIS)" if color_col == "Decision_Score" else color_col
                    customdata_par = [f"par_{i}" for i in df_display.index]
                    hover_text_par = [
                        f"🏆 Rank #{i+1}<br>Score={row.get('Decision_Score', 0):.3f}<br>🖱️ Click for details!" for i, row in df_display.iterrows()
                    ]
        
                    if use_3d and z_axis:
                        fig.add_trace(go.Scatter3d(
                            x=df_display[x_axis], y=df_display[y_axis], z=df_display[z_axis],
                            mode="markers", name="Pareto",
                            customdata=customdata_par,
                            marker=dict(
                                size=6, color=df_display[color_col], colorscale="Viridis", opacity=0.95, showscale=True,
                                colorbar=dict(title=dict(text=colorbar_title, side="right"))
                            ),
                            text=hover_text_par, hoverinfo="text", showlegend=True
                        ))
                        fig.update_layout(scene=dict(xaxis_title=axis_options[x_axis], yaxis_title=axis_options[y_axis], zaxis_title=axis_options[z_axis]))
                    else:
                        fig.add_trace(go.Scatter(
                            x=df_display[x_axis], y=df_display[y_axis],
                            mode="markers", name="Pareto",
                            customdata=customdata_par,
                            marker=dict(
                                size=12, color=df_display[color_col], colorscale="Viridis", opacity=0.95, showscale=True,
                                colorbar=dict(title=dict(text=colorbar_title, side="right"), thickness=15, xpad=10)
                            ),
                            text=hover_text_par, hoverinfo="text", showlegend=True,
                            # ✅ 修复：删除了不允许的 line 属性，选中时把点变成红色并放大到 18
                            selected=dict(marker=dict(size=18, color="red", opacity=1.0)),
                            unselected=dict(marker=dict(opacity=0.95))
                        ))
                    
                    # 3. 渲染 Top-1 五角星
                    if "Decision_Score" in df_display.columns and len(df_display) > 0:
                        best_row = df_display.iloc[0]
                        if use_3d and z_axis:
                             fig.add_trace(go.Scatter3d(
                                x=[best_row[x_axis]], y=[best_row[y_axis]], z=[best_row[z_axis]],
                                mode="markers+text", name="Top-1", customdata=["par_0"],
                                marker=dict(size=10, symbol="diamond", color="red", line=dict(color="black", width=2)),
                                text=["Top-1"], textposition="top center", hoverinfo="none"
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=[best_row[x_axis]], y=[best_row[y_axis]],
                                mode="markers+text", name="Top-1", customdata=["par_0"],
                                marker=dict(size=20, symbol="star", color="red", line=dict(color="black", width=1)),
                                text=["Top-1"], textposition="top center", showlegend=True, hoverinfo="none"
                            ))
                    
                    if not (use_3d and z_axis):
                        fig.update_xaxes(title=axis_options[x_axis], showgrid=True, gridcolor=COLORS["border"])
                        fig.update_yaxes(title=axis_options[y_axis], showgrid=True, gridcolor=COLORS["border"])
        
                    fig.update_layout(
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)", bordercolor=COLORS['border'], borderwidth=1),
                        **layout_settings
                    )
                    
                    plot_event = st.plotly_chart(
                        fig, 
                        use_container_width=True, 
                        on_select="rerun", 
                        selection_mode="points",
                        key="pareto_click_event"
                    )

                # ------ 2. 右侧渲染点击后的配方详情 ------
                with col_details:
                    st.markdown("#### 🔍 Recipe Spotlight")
                    
                    if "selected_mix_info" not in st.session_state:
                        st.session_state.selected_mix_info = None
                    
                    # ✨ 安全解析点击事件
                    if plot_event and "selection" in plot_event:
                        pts = plot_event["selection"]["points"]
                        if len(pts) > 0 and "customdata" in pts[0]:
                            cd = pts[0]["customdata"]
                            # 无论 Streamlit 回传的是数组还是单个字符串，统一提取出来
                            cd_val = cd[0] if isinstance(cd, list) else cd
                            
                            if isinstance(cd_val, str):
                                if cd_val.startswith("exp_"):
                                    st.session_state.selected_mix_info = {"idx": int(cd_val.split("_")[1]), "type": 0}
                                elif cd_val.startswith("par_"):
                                    st.session_state.selected_mix_info = {"idx": int(cd_val.split("_")[1]), "type": 1}
                    
                    # 读取状态并展示面板
                    sm = st.session_state.selected_mix_info
                    if sm is not None:
                        mix_idx = sm["idx"]
                        mix_type = sm["type"]
                        
                        sel_row = None
                        
                        # 根据类型提取数据
                        if mix_type == 1 and mix_idx < len(df_display):
                            sel_row = df_display.iloc[mix_idx]
                            score_val = sel_row.get("Decision_Score", 0)
                            st.markdown(f"""
                            <div style="background:{COLORS['primary']}15; padding:15px; border-radius:10px; border:1px solid {COLORS['primary']}50; text-align:center; margin-bottom:15px;">
                                <h3 style="margin:0; color:{COLORS['primary']};">🏆 Rank #{mix_idx + 1}</h3>
                                <span style="font-size:0.9rem; color:{COLORS['text_sub']};">TOPSIS Score: <b>{score_val:.3f}</b></span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        elif mix_type == 0 and df_all_display is not None and mix_idx < len(df_all_display):
                            sel_row = df_all_display.iloc[mix_idx]
                            gen = sel_row.get("Generation", "-")
                            st.markdown(f"""
                            <div style="background:#F1F5F9; padding:15px; border-radius:10px; border:1px solid #CBD5E1; text-align:center; margin-bottom:15px;">
                                <h3 style="margin:0; color:#475569;">🔍 Explored #{mix_idx + 1}</h3>
                                <span style="font-size:0.9rem; color:#64748B;">Generation: <b>{gen}</b> (Not in Pareto)</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        if sel_row is not None:
                            # 核心指标概览
                            m_c1, m_c2 = st.columns(2)
                            if "E" in sel_row: m_c1.metric("E-Modulus", f"{sel_row['E']:.1f} GPa")
                            if "Cost" in sel_row: m_c2.metric("Cost", f"€{sel_row['Cost']:.3f}")
                            m_c3, m_c4 = st.columns(2)
                            if "CO2_emission" in sel_row: m_c3.metric("CO₂ Emis.", f"{sel_row['CO2_emission']:.2f}")
                            if "Net_emission" in sel_row: m_c4.metric("Net CO₂", f"{sel_row['Net_emission']:.2f}")
                            
                            st.markdown("---")
                            
                            # 生成环形图展示配方成分
                            mats = {k: float(sel_row[k]) for k in [m[1] for m in MATERIALS_CONFIG] if k in sel_row.index and sel_row[k] > 0}
                            if mats:
                                df_m = pd.DataFrame(list(mats.items()), columns=["Material", "Value"])
                                fig_p = px.pie(df_m, values="Value", names="Material", hole=0.6, color_discrete_sequence=px.colors.qualitative.Prism)
                                fig_p.update_traces(textposition="inside", textinfo="none", hovertemplate="%{label}: %{value:.1f}g<extra></extra>")
                                fig_p.update_layout(
                                    showlegend=True, 
                                    legend=dict(orientation="h", y=-0.2),
                                    margin=dict(t=0, b=0, l=0, r=0), 
                                    height=220, 
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    annotations=[dict(text="Binder<br>Mix", x=0.5, y=0.5, font_size=14, showarrow=False, font=dict(color=COLORS['text_sub'], family="Inter"))]
                                )
                                st.plotly_chart(fig_p, use_container_width=True)
                    else:
                        # 默认状态提示
                        st.markdown(f"""
                        <div style="border: 2px dashed {COLORS['border']}; border-radius: 12px; padding: 60px 20px; text-align: center; color: {COLORS['text_sub']}; margin-top: 20px;">
                            <h1 style="font-size: 3rem; margin-bottom: 10px; color:{COLORS['primary']};">🖱️</h1>
                            <p style="font-size: 1rem; margin: 0;"><b>Click any point</b> on the plot (colored or grey) to reveal its exact recipe.</p>
                        </div>
                        """, unsafe_allow_html=True)

        # ==========================================

# ==========================================
        # Tab 2: GA Animation (🌌 历史残影版)
        # ==========================================
        with tab_anim:
            a1, a2 = st.columns([12, 1], gap="small")
            with a1:
                st.markdown("### 🎞️ Evolution of Population")
            with a2:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to read this animation")
                    st.write("• **🌌 Ghost Trails**: This frame shows current dots in color, while past dots remain as faint grey traces to show how the algorithm has converged over time.")

            if df_all_display is None or "Generation" not in df_all_display.columns:
                st.warning("⚠️ No history found. Please re-run optimization.")
            else:
                anim_axis_candidates = [k for k in axis_options.keys() if k in df_all_display.columns]
                
                c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1], gap="medium")
                default_x = "Net_emission" if "Net_emission" in anim_axis_candidates else anim_axis_candidates[0]
                default_y = "E" if "E" in anim_axis_candidates else anim_axis_candidates[1]

                x_anim = c1.selectbox("X Axis", anim_axis_candidates, index=anim_axis_candidates.index(default_x), format_func=lambda x: axis_options.get(x, x), key="anim_x")
                y_anim = c2.selectbox("Y Axis", anim_axis_candidates, index=anim_axis_candidates.index(default_y), format_func=lambda x: axis_options.get(x, x), key="anim_y")
                
                # 特效开关
                show_ghosts = c3.toggle("🌌 Show Ghost Trails", value=True)
                speed = c4.select_slider("Animation Speed", options=[50, 150, 300, 600], value=150)

                df_anim = df_all_display.copy()
                x_min, x_max = df_anim[x_anim].min()*0.95, df_anim[x_anim].max()*1.05
                y_min, y_max = df_anim[y_anim].min()*0.95, df_anim[y_anim].max()*1.05

                # --- 🎬 核心魔术：构建带残影的每一帧 ---
                frames = []
                gens = sorted(df_anim["Generation"].unique())
                
                for g in gens:
                    # 获取当前代的点
                    df_curr = df_anim[df_anim["Generation"] == g]
                    
                    data_in_frame = []
                    
                    # 1. 如果开启残影，先画出之前所有代的点（Trace 0）
                    if show_ghosts and g > gens[0]:
                        df_past = df_anim[df_anim["Generation"] < g]
                        data_in_frame.append(go.Scatter(
                            x=df_past[x_anim], y=df_past[y_anim],
                            mode="markers",
                            marker=dict(size=4, color="lightgrey", opacity=0.15),
                            name="Past Generations",
                            hoverinfo="none"
                        ))
                    else:
                        # 没残影时传个空点，保持 Trace 索引一致
                        data_in_frame.append(go.Scatter(x=[], y=[]))
                    
                    # 2. 画出当前代的点（Trace 1）
                    data_in_frame.append(go.Scatter(
                        x=df_curr[x_anim], y=df_curr[y_anim],
                        mode="markers",
                        marker=dict(size=9, color=COLORS['primary'], opacity=0.7, line=dict(width=1, color="white")),
                        name=f"Generation {g}"
                    ))
                    
                    frames.append(go.Frame(
                        name=str(g),
                        data=data_in_frame,
                        traces=[0, 1]  # 明确告诉 Plotly 这一帧要更新前两个 Trace
                    ))

                # --- 初始画布设置 ---
                fig_anim = go.Figure(
                    data=[
                        # Trace 0: 预留给残影
                        go.Scatter(x=[], y=[], mode="markers", showlegend=False), 
                        # Trace 1: 预留给当前代
                        go.Scatter(
                            x=df_anim[df_anim["Generation"] == gens[0]][x_anim],
                            y=df_anim[df_anim["Generation"] == gens[0]][y_anim],
                            mode="markers",
                            marker=dict(size=9, color=COLORS['primary'], opacity=0.7, line=dict(width=1, color="white")),
                            name="Initial Population"
                        )
                    ],
                    layout=go.Layout(
                        xaxis=dict(range=[x_min, x_max], title=axis_options.get(x_anim), gridcolor=COLORS['border'], zeroline=False),
                        yaxis=dict(range=[y_min, y_max], title=axis_options.get(y_anim), gridcolor=COLORS['border'], zeroline=False),
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=20, b=40),
                        hovermode="closest",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.7)",
                    ),
                    frames=frames
                )

                # 叠加最终 Pareto 作为参考（Trace 2 - 永远静止）
                fig_anim.add_trace(go.Scatter(
                    x=df_display[x_anim], y=df_display[y_anim],
                    mode="markers",
                    name="Target Frontier",
                    marker=dict(size=10, symbol="diamond-open", color=COLORS['accent'], line=dict(width=2))
                ))

                # 播放按钮与滑块
                fig_anim.update_layout(
                    updatemenus=[dict(
                        type="buttons", showactive=False, x=0, y=1.15,
                        buttons=[
                            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=speed, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                            dict(label="|| Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                        ]
                    )],
                    sliders=[dict(
                        active=0, currentvalue={"prefix": "Evolution Progress: Gen "},
                        pad={"t": 50}, x=0, y=0, len=1.0,
                        steps=[dict(label=str(g), method="animate", args=[[str(g)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for g in gens]
                    )]
                )

                st.plotly_chart(fig_anim, use_container_width=True)

        # ==========================================

# ==========================================
        # Tab 3: Parallel Coordinates (🕸️ 进阶好玩版)
        # ==========================================
        with tab_parallel:
            p1, p2 = st.columns([12, 1], gap="small")
            with p1:
                st.markdown("##### 🕸️ Multi-Dimensional Mix Analysis")
            with p2:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to play with this plot")
                    st.write("• **Brush & Filter**: Drag vertically on any axis to filter data ranges.")
                    st.write("• **🔦 Spotlight**: Pick a specific mix to highlight its exact path in red!")
                    st.write("• **🎨 Color Theme**: Change the visual vibe of your data.")

            all_input_cols = [m[1] for m in MATERIALS_CONFIG] 
            all_output_cols = ["E", "CO2_emission", "Cost", "Net_emission", "Decision_Score"]
            all_available_cols = [c for c in all_input_cols + all_output_cols if c in df_display.columns]
            
            default_selection = ["C3S", "C2S", "GGBFS", "E", "Cost", "CO2_emission"]
            default_selection = [c for c in default_selection if c in all_available_cols]

            # --- ✨ 新增的好玩控制面板 ---
            c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([3, 1.5, 1.5, 1.5])
            
            selected_cols = c_ctrl1.multiselect("📌 Active Dimensions", options=all_available_cols, default=default_selection)
            
            if len(selected_cols) > 1:
                color_col = c_ctrl2.selectbox("🎯 Color Lines By", options=[c for c in all_available_cols if c in df_display.columns], index=all_available_cols.index("E") if "E" in all_available_cols else 0)
                
                theme = c_ctrl3.selectbox("🎨 Color Theme", options=["Turbo", "Viridis", "Plasma", "Inferno", "RdBu"])
                
                # 聚光灯下拉框
                spotlight_mix = c_ctrl4.selectbox(
                    "🔦 Spotlight Mix", 
                    options=["None"] + [f"Rank #{i+1}" for i in range(len(df_display))],
                    index=0,
                    help="Highlight a specific mix to trace its path across all dimensions."
                )

                df_pc = df_display.copy()
                
                # --- 🔮 核心魔术：聚光灯逻辑 ---
                if spotlight_mix != "None":
                    mix_idx = int(spotlight_mix.split("#")[1]) - 1
                    
                    # 创建一个临时的颜色列，选中的为 1，没选中的为 0
                    df_pc["Spotlight"] = 0
                    df_pc.loc[mix_idx, "Spotlight"] = 1
                    
                    # ⚠️ 关键动作：把选中的那一行移到 DataFrame 的最后面！
                    # 这样 Plotly 在画图时，亮红色线才会画在最顶层，不会被灰线遮挡
                    target_row = df_pc.iloc[[mix_idx]]
                    df_pc = pd.concat([df_pc.drop(mix_idx), target_row], ignore_index=True)
                    
                    color_target = "Spotlight"
                    # 自定义极度两极分化的色带：0(灰) -> 1(亮红)
                    color_scale = [[0, 'rgba(200, 200, 200, 0.15)'], [1, 'rgba(255, 0, 50, 1)']]
                    hide_colorbar = True # 聚光灯模式下隐藏无用的色条
                else:
                    color_target = color_col
                    color_scale = theme.lower() # 使用用户选择的主题
                    hide_colorbar = False

                # 渲染图表
                fig_par = px.parallel_coordinates(
                    df_pc, 
                    dimensions=selected_cols, 
                    color=color_target, 
                    labels={col: col.replace('_', ' ') for col in selected_cols}, 
                    color_continuous_scale=color_scale 
                )
                
                # 隐藏/显示色条
                if hide_colorbar:
                    fig_par.update_coloraxes(showscale=False)
                else:
                    fig_par.update_coloraxes(colorbar_title=color_col)

                # UI 细节打磨
                fig_par.update_traces(
                    labelfont=dict(size=14, color=COLORS['text_head'], family="Inter", weight="bold"),
                    tickfont=dict(size=11, color=COLORS['text_sub'], family="Inter"),
                    rangefont=dict(size=10, color=COLORS['text_sub'], family="Inter")
                )
                
                fig_par.update_layout(
                    margin=dict(l=50, r=50, t=60, b=40), 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig_par, use_container_width=True)
            else:
                st.warning("⚠️ Select at least two dimensions to visualize.")

        # ==========================================
        # Tab 4: Data Table
        # ==========================================
        with tab_table:
            tab_pareto_sub, tab_search_sub = st.tabs(["🏆 Pareto Optimal Mixes", "🎯 Target Search (All Mixes)"])
            
            with tab_pareto_sub:
                d1, d2 = st.columns([12, 1], gap="small")
                with d1:
                    st.markdown("### 📋 Pareto Optimal Mixes")
                with d2:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    with st.popover("ℹ️"):
                        st.write("• Each row is one **Pareto-optimal** mix (non-dominated trade-off).")

                m1, m2, m3 = st.columns(3)
                if "E" in df_display.columns: m1.metric("Max E Found", f"{df_display['E'].max():.2f} GPa")
                if "Cost" in df_display.columns: m2.metric("Min Cost", f"€{df_display['Cost'].min():.2f} /kg") 
                if "Net_emission" in df_display.columns: m3.metric("Min Net CO₂", f"{df_display['Net_emission'].min():.3f} kg/kg")

                column_configuration = {}
                if "E" in df_display.columns:
                    column_configuration["E"] = st.column_config.ProgressColumn("E (GPa)", format="%.2f", min_value=float(df_display["E"].min()), max_value=float(df_display["E"].max()))
                if "Cost" in df_display.columns:
                    column_configuration["Cost"] = st.column_config.ProgressColumn("Cost (€/kg)", format="€%.4f", min_value=float(df_display["Cost"].min()), max_value=float(df_display["Cost"].max()))
                if "CO2_emission" in df_display.columns:
                    column_configuration["CO2_emission"] = st.column_config.ProgressColumn("CO₂ (kg/kg)", format="%.3f", min_value=float(df_display["CO2_emission"].min()), max_value=float(df_display["CO2_emission"].max()))
                if "Net_emission" in df_display.columns:
                    column_configuration["Net_emission"] = st.column_config.ProgressColumn("Net CO₂ (kg/kg)", format="%.3f", min_value=float(df_display["Net_emission"].min()), max_value=float(df_display["Net_emission"].max()))
                if "CO2_abs" in df_display.columns:
                    column_configuration["CO2_abs"] = st.column_config.NumberColumn("Uptake (kg/kg)", format="%.3f")
                if "Decision_Score" in df_display.columns:
                    column_configuration["Decision_Score"] = st.column_config.ProgressColumn("Decision Score (TOPSIS)", format="%.3f", min_value=0.0, max_value=1.0)

                material_cols = [m[1] for m in MATERIALS_CONFIG]
                for col in material_cols:
                    if col in df_display.columns:
                        column_configuration[col] = st.column_config.NumberColumn(f"{col} (gram)", format="%.1f")

                st.dataframe(df_display, column_config=column_configuration, use_container_width=True, height=520)
                st.download_button("📥 Download Pareto CSV", df_display.to_csv(index=False).encode("utf-8"), "pareto_results.csv", "text/csv", type="primary")

            with tab_search_sub:
                st.markdown("### 🎯 Target Search (Goal-Seeker Box)")
                st.markdown("<span style='color:#64748B; font-size:0.9rem;'>Use the dual-ended sliders to draw a <b>bounding box</b> and target a specific zone of mixes.</span>", unsafe_allow_html=True)
                st.markdown("---")
                
                if df_all_display is None:
                    st.warning("⚠️ Optimization history is required for Target Search. Please run the optimization first.")
                else:
                    c_filters, c_plot = st.columns([1, 2.5], gap="large")
                    with c_filters:
                        st.markdown("#### 1. Set Target Box")
                        # 获取所有指标的全局最小/最大值
                        min_e, max_e = float(df_all_display["E"].min()), float(df_all_display["E"].max())
                        min_cost, max_cost = float(df_all_display["Cost"].min()), float(df_all_display["Cost"].max())
                        min_co2, max_co2 = float(df_all_display["CO2_emission"].min()), float(df_all_display["CO2_emission"].max())
                        
                        # 设置合理的默认框选范围
                        def_e_range = (max(min_e, min(15.0, max_e)), max_e)
                        def_cost_range = (min_cost, max(min_cost, min(0.12, max_cost)))
                        def_co2_range = (min_co2, max(min_co2, min(0.60, max_co2)))
                        
                        # ✨ 魔法：传入 tuple，自动变成可以两头拖拽的 Range Slider
                        target_e = st.slider("⚡ E-Modulus Range (GPa)", min_value=min_e, max_value=max_e, value=def_e_range, step=0.5)
                        target_co2 = st.slider("🏭 CO₂ Emission Range (kg/kg)", min_value=min_co2, max_value=max_co2, value=def_co2_range, step=0.01, format="%.3f")
                        target_cost = st.slider("💰 Cost Range (€/kg)", min_value=min_cost, max_value=max_cost, value=def_cost_range, step=0.005, format="%.3f")
                        
                        # 过滤逻辑：要求数据必须同时落在我们设定的最小值和最大值之间
                        mask = (
                            (df_all_display["E"] >= target_e[0]) & (df_all_display["E"] <= target_e[1]) &
                            (df_all_display["CO2_emission"] >= target_co2[0]) & (df_all_display["CO2_emission"] <= target_co2[1]) &
                            (df_all_display["Cost"] >= target_cost[0]) & (df_all_display["Cost"] <= target_cost[1])
                        )
                        df_filtered = df_all_display[mask].copy()
                        
                        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                        if len(df_filtered) > 0:
                            st.success(f"✅ Found **{len(df_filtered)}** viable mixes in the box!")
                        else:
                            st.error("❌ No mixes found in this zone. Try expanding the box.")
                    
                    with c_plot:
                        st.markdown("#### 2. Target Zone")
                        fig_target = go.Figure()
                        
                        # 画出背景所有的探索点（灰色）
                        fig_target.add_trace(go.Scatter(
                            x=df_all_display["CO2_emission"], y=df_all_display["E"], 
                            mode="markers", name="All Explored", 
                            marker=dict(size=5, color="lightgrey", opacity=0.3), hoverinfo="none"
                        ))
                        
                        # ✨ 魔法：画出红色的“上下左右四条线”组成的锁定框 (Bounding Box)
                        fig_target.add_shape(
                            type="rect",
                            x0=target_co2[0], y0=target_e[0], # 左下角坐标
                            x1=target_co2[1], y1=target_e[1], # 右上角坐标
                            line=dict(color="red", width=2, dash="dash"), # 红色虚线边框
                            fillcolor="rgba(255, 0, 0, 0.05)", # 内部填一点淡淡的红色
                            layer="below"
                        )
                        
                        # 画出真正落在框里的有效点
                        if len(df_filtered) > 0:
                            fig_target.add_trace(go.Scatter(
                                x=df_filtered["CO2_emission"], y=df_filtered["E"], 
                                mode="markers", name="Viable Mixes",
                                marker=dict(size=8, color=df_filtered["Cost"], colorscale="Viridis", showscale=True, colorbar=dict(title="Cost (€/kg)", thickness=15), line=dict(width=1, color="white")),
                                text=[f"Mix ID: {i}<br>E: {row['E']:.2f}<br>CO2: {row['CO2_emission']:.3f}<br>Cost: {row['Cost']:.3f}" for i, row in df_filtered.iterrows()], hoverinfo="text"
                            ))
                        
                        fig_target.update_layout(
                            xaxis_title="CO₂ Emission (kg/kg) - Lower is better", yaxis_title="E-Modulus (GPa) - Higher is better",
                            height=400, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
                            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)")
                        )
                        st.plotly_chart(fig_target, use_container_width=True)
                    
                    if len(df_filtered) > 0:
                        st.markdown("#### 3. Viable Mixes Data")
                        df_filtered_sorted = df_filtered.sort_values(by="CO2_emission", ascending=True).reset_index(drop=True)
                        display_cols = ["E", "CO2_emission", "Cost", "Net_emission"] + [m[1] for m in MATERIALS_CONFIG if m[1] in df_filtered_sorted.columns]
                        st.dataframe(
                            df_filtered_sorted[display_cols], use_container_width=True, height=250,
                            column_config={
                                "E": st.column_config.NumberColumn("E (GPa)", format="%.2f"),
                                "CO2_emission": st.column_config.NumberColumn("CO₂ (kg/kg)", format="%.3f"),
                                "Net_emission": st.column_config.NumberColumn("Net CO₂ (kg/kg)", format="%.3f"),
                                "Cost": st.column_config.NumberColumn("Cost (€/kg)", format="%.3f"),
                            }
                        )
                        st.download_button("📥 Download Viable Mixes", data=df_filtered_sorted[display_cols].to_csv(index=False).encode("utf-8"), file_name="target_search.csv", mime="text/csv", type="primary")
        # ==========================================
        # Tab 5: Benchmark Comparison & PDF Report
        with tab_bench:
            with st.expander("⚙️ Configure Baseline (OPC)", expanded=False):
                b_c1, b_c2, b_c3, b_c4 = st.columns(4)
                opc_e = b_c1.number_input("Ref E (GPa)", value=17.0, step=0.5, key="bench_e")
                opc_co2 = b_c2.number_input("Ref CO₂ (kg/kg)", value=0.76, step=0.01, format="%.3f", key="bench_co2")
                opc_cost = b_c3.number_input("Ref Cost (€/kg)", value=0.22, step=0.01, format="%.3f", key="bench_cost")
                opc_net = b_c4.number_input("Ref Net CO₂ (kg/kg)", value=0.40, step=0.01, key="bench_net")
             
            c_left2, c_right2 = st.columns([1, 2.5], gap="large")
             
            with c_left2:
                st.info("Select a mix to compare details.")
                pareto_opts = {i: f"Mix #{i+1} (E={r['E']:.1f})" for i, r in df_display.iterrows()}
                selected_idx = st.selectbox("Select Solution", options=list(pareto_opts.keys()), format_func=lambda x: pareto_opts[x])
                
                if selected_idx is not None:
                    row = df_display.iloc[selected_idx]
                    baseline_data = {
                        "E": float(opc_e),
                        "CO2_emission": float(opc_co2),
                        "Cost": float(opc_cost),
                        "Net_emission": float(opc_net),
                    }
                    st.markdown("---")
                    
# ---------------- 这里是修复后的函数 ----------------
                    def delta_metric_card(label, val, ref, unit, inverse=False):
                        diff = val - ref
                        pct = (diff / ref * 100) if ref != 0 else 0
                        
                        # inverse 模式：正数为红，负数为绿 (适用于碳排、成本)
                        # normal 模式：正数为绿，负数为红 (适用于强度)
                        color_mode = "inverse" if inverse else "normal"
                        
                        fmt = "%.3f" if "kg" in unit else "%.2f"
                        
                        # ✅ 修复点：完整传入所有的位置参数 (label, value) 和关键字参数 (delta, delta_color)
                        st.metric(
                            label=label, 
                            value=f"{val:{fmt[1:]}} {unit}", 
                            delta=f"{diff:{fmt[1:]}} ({pct:+.1f}%)", 
                            delta_color=color_mode
                        )
                    # ----------------------------------------------------
                    
                    delta_metric_card("E-Modulus", row["E"], opc_e, "GPa")
                    delta_metric_card("CO₂ Emission", row["CO2_emission"], opc_co2, "kg/kg", inverse=True)
                    delta_metric_card("Cost", float(row["Cost"]), float(opc_cost), "€/kg", inverse=True)

            with c_right2:
                if selected_idx is not None:
                    row = df_display.iloc[selected_idx]
            
                    row1_col1, row1_col2 = st.columns(2, gap="medium")
            
                    with row1_col1:
                        st.markdown("#### 🍩 Composition (gram)")
                        mat_cols = [m[1] for m in MATERIALS_CONFIG]
                        materials = {k: float(row[k]) for k in mat_cols if k in df_display.columns}
                        df_mat = pd.DataFrame(list(materials.items()), columns=["Material", "Value"])
                        df_mat = df_mat[df_mat["Value"] > 0]
            
                        fig_pie = px.pie(df_mat, values="Value", names="Material", hole=0.55, color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_pie.update_traces(textposition="inside", textinfo="none", texttemplate="%{value:.1f}%")
                        fig_pie.update_layout(showlegend=False, margin=dict(t=30, b=10, l=10, r=10), height=280, annotations=[dict(text="Mix", x=0.5, y=0.5, font_size=14, showarrow=False)])
                        st.plotly_chart(fig_pie, use_container_width=True)
            
                    with row1_col2:
                        st.markdown("#### 📊 Metric Comparison")
                        metrics_data = [
                            {"Metric": "E (GPa)",     "Type": "Baseline", "Value": float(opc_e)},
                            {"Metric": "E (GPa)",     "Type": "Selected", "Value": float(row["E"])},
                            {"Metric": "CO₂ (kg/kg)", "Type": "Baseline", "Value": float(opc_co2)},
                            {"Metric": "CO₂ (kg/kg)", "Type": "Selected", "Value": float(row["CO2_emission"])},
                            {"Metric": "Cost (€/kg)", "Type": "Baseline", "Value": float(opc_cost)},
                            {"Metric": "Cost (€/kg)", "Type": "Selected", "Value": float(row["Cost"])},
                        ]
            
                        fig_bar = px.bar(pd.DataFrame(metrics_data), x="Metric", y="Value", color="Type", barmode="group", text_auto=".3f")
                        fig_bar.update_layout(height=280, margin=dict(t=20, b=10, l=0, r=0), legend=dict(orientation="h", y=1.02, x=0.6), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)")
                        st.plotly_chart(fig_bar, use_container_width=True)
            
                    # ✨ 魔法 2：环保成就翻译机 (Eco-Impact Translator)
                    st.markdown("#### 🌳 Real-World Impact (Per 1 Ton of Binder)")
                    co2_saved_per_ton = (opc_co2 - row['CO2_emission']) * 1000 # 换算成公斤
                    
                    if co2_saved_per_ton > 0:
                        # 一棵树每年大约吸收 22 kg CO2；一辆普通燃油车每公里排放约 0.12 kg CO2
                        trees_eq = int(co2_saved_per_ton / 22)
                        car_km_eq = int(co2_saved_per_ton / 0.12)
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #10B98115, #0F766E20); border-left: 4px solid #10B981; padding: 15px 20px; border-radius: 8px; display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px;">
                            <div style="text-align: center;">
                                <div style="font-size: 2rem;">☁️</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: #0F766E;">{co2_saved_per_ton:.0f} kg</div>
                                <div style="font-size: 0.8rem; color: #64748B;">CO₂ Avoided</div>
                            </div>
                            <div style="font-size: 2.5rem; color: #1E293B;">=</div>
                            <div style="text-align: center;">
                                <div style="font-size: 2rem;">🌲</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: #10B981;">{trees_eq} Trees</div>
                                <div style="font-size: 0.8rem; color: #64748B;">Planted (1 Year)</div>
                            </div>
                            <div style="font-size: 2.5rem; color: #1E293B;">or</div>
                            <div style="text-align: center;">
                                <div style="font-size: 2rem;">🚗</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: #F59E0B;">{car_km_eq:,} km</div>
                                <div style="font-size: 0.8rem; color: #64748B;">Not Driven</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: #FEE2E2; border-left: 4px solid #EF4444; padding: 15px; border-radius: 8px;">
                            <span style="color: #B91C1C; font-weight: 600;">⚠️ Higher Emissions:</span> This mix produces {abs(co2_saved_per_ton):.0f} kg MORE CO₂ per ton than the baseline.
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    h1, h2 = st.columns([12, 1], gap="small")
                    with h1:
                        st.markdown("#### 📌 Position on Frontier")
                    with h2:
                        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                        with st.popover("ℹ️"):
                            st.write("• Grey: explored candidates • Colored: Pareto")
                            st.write("• ⭐ Selected mix • ✖ Baseline (OPC)")
                    
                    chart_options = [k for k in axis_options.keys() if k in df_display.columns]
                    material_cols = [m[1] for m in MATERIALS_CONFIG]
                    color_candidates = ["Decision_Score"] + chart_options + material_cols
                    color_options = [c for c in color_candidates if c in df_display.columns]
                    
                    default_x = "CO2_emission" if "CO2_emission" in chart_options else (chart_options[0] if chart_options else None)
                    default_y = "E" if "E" in chart_options else (chart_options[1] if len(chart_options) > 1 else (chart_options[0] if chart_options else None))
                    default_z = "Cost" if "Cost" in chart_options else (chart_options[2] if len(chart_options) > 2 else None)
                    
                    c_ctrl = st.columns(5)
                    x_axis = c_ctrl[0].selectbox("X Axis", chart_options, index=chart_options.index(default_x) if default_x in chart_options else 0, format_func=lambda x: axis_options.get(x, x), key="bench_pos_x")
                    y_axis = c_ctrl[1].selectbox("Y Axis", chart_options, index=chart_options.index(default_y) if default_y in chart_options else 0, format_func=lambda x: axis_options.get(x, x), key="bench_pos_y")
                    
                    default_color_idx = color_options.index("Decision_Score") if "Decision_Score" in color_options else 0
                    color_col = c_ctrl[2].selectbox("Color Points By", color_options, index=default_color_idx, key="bench_pos_color")
                    show_all = c_ctrl[3].toggle("Show History", value=True, key="bench_pos_showhist")
                    
                    use_3d_allowed = len(chart_options) >= 3
                    use_3d = c_ctrl[4].toggle("3D View", value=False, key="bench_pos_3d") if use_3d_allowed else False
                    z_axis = None
                    if use_3d and use_3d_allowed:
                        z_axis = st.selectbox("Z Axis", chart_options, index=chart_options.index(default_z) if (default_z in chart_options) else 0, format_func=lambda x: axis_options.get(x, x), key="bench_pos_z")
                    
                    layout_settings = dict(height=520, margin=dict(l=20, r=80, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)", font=dict(family="Inter", color=COLORS["text_body"]))
                    fig_pos = go.Figure()
                    
                    if show_all and df_all_display is not None:
                        if use_3d and z_axis:
                            fig_pos.add_trace(go.Scatter3d(x=df_all_display[x_axis], y=df_all_display[y_axis], z=df_all_display[z_axis], mode="markers", name="Explored", marker=dict(size=2, color="lightgrey", opacity=0.2)))
                        else:
                            fig_pos.add_trace(go.Scatter(x=df_all_display[x_axis], y=df_all_display[y_axis], mode="markers", name="Explored", marker=dict(size=5, color="lightgrey", opacity=0.25)))
                    
                    colorbar_title = "Decision Score<br>(TOPSIS)" if color_col == "Decision_Score" else color_col
                    hover_text = [f"Rank #{i+1}<br>Score={row_i['Decision_Score']:.3f}" if "Decision_Score" in df_display.columns else f"Mix #{i+1}" for i, row_i in df_display.iterrows()]
                    
                    if use_3d and z_axis:
                        fig_pos.add_trace(go.Scatter3d(
                            x=df_display[x_axis], y=df_display[y_axis], z=df_display[z_axis], mode="markers", name="Pareto",
                            marker=dict(size=6, color=df_display[color_col], colorscale="Viridis", opacity=0.95, showscale=True, colorbar=dict(title=dict(text=colorbar_title, side="right"))),
                            text=hover_text, showlegend=True
                        ))
                        fig_pos.update_layout(scene=dict(xaxis_title=axis_options.get(x_axis, x_axis), yaxis_title=axis_options.get(y_axis, y_axis), zaxis_title=axis_options.get(z_axis, z_axis)))
                    else:
                        fig_pos.add_trace(go.Scatter(
                            x=df_display[x_axis], y=df_display[y_axis], mode="markers", name="Pareto",
                            marker=dict(size=10, color=df_display[color_col], colorscale="Viridis", opacity=0.95, showscale=True, colorbar=dict(title=dict(text=colorbar_title, side="right"), thickness=15, xpad=10)),
                            text=hover_text, showlegend=True
                        ))
                        fig_pos.update_xaxes(title=axis_options.get(x_axis, x_axis), showgrid=True, gridcolor=COLORS["border"])
                        fig_pos.update_yaxes(title=axis_options.get(y_axis, y_axis), showgrid=True, gridcolor=COLORS["border"])
                    
                    def _baseline_has(axis_name: str) -> bool:
                        return (axis_name in baseline_data)
                    
                    if _baseline_has(x_axis) and _baseline_has(y_axis) and (not use_3d or _baseline_has(z_axis)):
                        if use_3d and z_axis:
                            fig_pos.add_trace(go.Scatter3d(x=[baseline_data[x_axis]], y=[baseline_data[y_axis]], z=[baseline_data[z_axis]], mode="markers+text", name="Baseline (OPC)", marker=dict(size=9, symbol="x", color="gray", line=dict(width=2)), text=["Baseline"], textposition="top center"))
                        else:
                            fig_pos.add_trace(go.Scatter(x=[baseline_data[x_axis]], y=[baseline_data[y_axis]], mode="markers+text", name="Baseline (OPC)", marker=dict(size=14, symbol="x", color="gray", line=dict(width=2)), text=["Baseline"], textposition="top center"))
                    
                    if use_3d and z_axis:
                        fig_pos.add_trace(go.Scatter3d(x=[float(row[x_axis])], y=[float(row[y_axis])], z=[float(row[z_axis])], mode="markers+text", name="Selected Mix", marker=dict(size=9, symbol="star", color="red", line=dict(color="black", width=1)), text=["Selected"], textposition="top center"))
                    else:
                        fig_pos.add_trace(go.Scatter(x=[float(row[x_axis])], y=[float(row[y_axis])], mode="markers+text", name="Selected Mix", marker=dict(size=18, symbol="star", color="red", line=dict(color="black", width=1)), text=["Selected"], textposition="top center"))
                    
                    fig_pos.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)", bordercolor=COLORS['border'], borderwidth=1), **layout_settings)
                    st.plotly_chart(fig_pos, use_container_width=True)

            # ==========================================
            # FINAL SECTION: REPORT GENERATION
            # ==========================================
            st.markdown("---")
            te1, te2 = st.columns([12, 1], gap="small")
            with te1:
                st.markdown("### 📄 Technical Export")
            with te2:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### About the PDF Technical Audit")
                    st.write("• Results are based on model-predicted properties and must be experimentally validated.")

            try:
                report_params = {
                    "Curing Time": f"{st.session_state.get('time_input', 28)} Days",
                    "w/c Ratio": 0.5,
                    "Temperature": "25.0 °C",
                    "Target Binder": "96.0 g"
                }
        
                search_bounds = {
                    "clinker": {
                        "C3S (%)": st.session_state.get("c3s_rng_slider", (45.0, 80.0)),
                        "C2S (%)": st.session_state.get("c2s_rng_slider", (10.0, 32.0)),
                        "C3A (%)": st.session_state.get("c3a_rng_slider", (0.0, 14.0)),
                        "C4AF (%)": st.session_state.get("c4af_rng_slider", (0.0, 15.0))
                    },
                    "scms": {
                        "Silica Fume (g)": st.session_state.get("sf_rng_slider", (0.0, 10.0)),
                        "GGBFS (g)": st.session_state.get("gg_rng_slider", (0.0, 80.0)),
                        "Fly Ash (g)": st.session_state.get("fa_rng_slider", (0.0, 35.0)),
                        "Calcined Clay (g)": st.session_state.get("cc_rng_slider", (0.0, 35.0)),
                        "Limestone (g)": st.session_state.get("ls_rng_slider", (0.0, 35.0))
                    },
                    "total_clinker": st.session_state.get("cl_sum_rng_slider", (20.0, 96.0))
                }
        
                ga_conf = {
                    "pop": st.session_state.get("ga_pop", 100),
                    "gen": st.session_state.get("ga_gen", 20),
                    "seed": st.session_state.get("ga_seed", 1)
                }
        
                if st.button("🚀 Generate PDF Technical Audit", type="primary", use_container_width=True):
                    with st.spinner("Compiling technical report..."):
                        pdf_bytes = create_pdf_report(
                            st.session_state.df_pareto, 
                            report_params, 
                            baseline_data, 
                            objective_config, 
                            search_bounds, 
                            ga_conf
                        )
                        st.session_state['pdf_data'] = pdf_bytes
                        st.success("Report Compiled Successfully!")
        
                if 'pdf_data' in st.session_state:
                    st.download_button(
                        label="📥 Download Technical Report",
                        data=st.session_state['pdf_data'],
                        file_name="Mix_Optimization_Audit.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f"Waiting for optimization settings to initialize... ({e})")