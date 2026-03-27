# dashboard.py (REFINED RESULTS MODULE)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pdf_generator import create_pdf_report


# ==========================================================
# 1) Fast TOPSIS (cached)
# ==========================================================
@st.cache_data(show_spinner=False)
def calculate_topsis_cached(df: pd.DataFrame, criteria_cols: tuple, impacts: tuple) -> np.ndarray:
    """
    Vectorized TOPSIS (fast + cached)

    criteria_cols: tuple[str]
    impacts: tuple[str] each in ('+','-')
    return: np.ndarray scores in [0,1]
    """
    if df is None or len(df) == 0 or len(criteria_cols) == 0:
        return np.zeros(0 if df is None else len(df), dtype=float)

    # Extract numeric matrix
    X = df.loc[:, list(criteria_cols)].to_numpy(dtype=float)

    # 1) Normalize by column L2 norm (avoid div0)
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    Xn = X / denom

    # 2) Equal weights
    w = 1.0 / len(criteria_cols)
    Xw = Xn * w

    # 3) Ideal best / worst (vectorized)
    # For '+' maximize: best=max, worst=min
    # For '-' minimize: best=min, worst=max
    Xw_max = Xw.max(axis=0)
    Xw_min = Xw.min(axis=0)

    impacts_arr = np.array(impacts, dtype=str)
    is_plus = impacts_arr == "+"

    ideal_best = np.where(is_plus, Xw_max, Xw_min)
    ideal_worst = np.where(is_plus, Xw_min, Xw_max)

    # 4) Distances
    s_plus = np.sqrt(((Xw - ideal_best) ** 2).sum(axis=1))
    s_minus = np.sqrt(((Xw - ideal_worst) ** 2).sum(axis=1))

    # 5) Score
    denom2 = s_plus + s_minus
    denom2[denom2 == 0] = 1.0
    scores = s_minus / denom2
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _objective_config_from_session() -> list:
    """Read objective checkboxes from session_state (same logic you had)."""
    cfg = []
    if st.session_state.get("obj_e_max"):
        cfg.append({"col": "E", "impact": "+", "name": "Max Strength"})
    if st.session_state.get("obj_cost_min"):
        cfg.append({"col": "Cost", "impact": "-", "name": "Min Cost"})
    if st.session_state.get("obj_co2_min"):
        cfg.append({"col": "CO2_emission", "impact": "-", "name": "Min CO2"})
    if st.session_state.get("obj_net_min"):
        cfg.append({"col": "Net_emission", "impact": "-", "name": "Min Net CO2"})
    if st.session_state.get("obj_co2abs_max"):
        cfg.append({"col": "CO2_abs", "impact": "+", "name": "Max Uptake"})
    if not cfg:
        cfg = [{"col": "E", "impact": "+", "name": "Max Strength"}]
    return cfg


def _objective_signature(cfg: list) -> tuple:
    """Make a stable signature for cache invalidation."""
    return tuple((d["col"], d["impact"]) for d in cfg)


# ==========================================================
# 2) Display pipeline (cached)
#    - compute decision score once
#    - sort once
#    - optional sampling for history
# ==========================================================
@st.cache_data(show_spinner=False)
def prepare_display_pipeline(
    df_pareto: pd.DataFrame,
    df_all: pd.DataFrame | None,
    obj_sig: tuple,
    history_sample_cap: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame | None, tuple]:
    """
    Returns:
      df_display (pareto with Decision_Score, sorted)
      df_all_display (history possibly sampled)
      (criteria_cols, impacts) as tuples
    """
    df_display = df_pareto.copy()

    criteria_cols = tuple([c for (c, imp) in obj_sig if c in df_display.columns])
    impacts = tuple([imp for (c, imp) in obj_sig if c in df_display.columns])

    if len(df_display) > 0:
        scores = calculate_topsis_cached(df_display, criteria_cols, impacts)
        df_display["Decision_Score"] = scores
        df_display = df_display.sort_values("Decision_Score", ascending=False).reset_index(drop=True)
    else:
        df_display["Decision_Score"] = np.zeros(len(df_display), dtype=float)

    df_all_display = None
    if df_all is not None and len(df_all) > 0:
        df_all_display = df_all.copy()

        # --- speed: sample history if huge (Plotly Cloud killer) ---
        if len(df_all_display) > history_sample_cap:
            # keep some structure if Generation exists (stratified-ish)
            if "Generation" in df_all_display.columns:
                # sample per generation proportionally
                gens = df_all_display["Generation"].to_numpy()
                uniq = np.unique(gens)
                per_gen = max(1, history_sample_cap // max(1, len(uniq)))
                chunks = []
                for g in uniq:
                    sub = df_all_display[df_all_display["Generation"] == g]
                    if len(sub) > per_gen:
                        chunks.append(sub.sample(per_gen, random_state=1))
                    else:
                        chunks.append(sub)
                df_all_display = pd.concat(chunks, ignore_index=True)
            else:
                df_all_display = df_all_display.sample(history_sample_cap, random_state=1).reset_index(drop=True)

    return df_display, df_all_display, (criteria_cols, impacts)


# ==========================================================
# 3) Plot builders (cached)
# ==========================================================
@st.cache_data(show_spinner=False)
def build_pareto_figure(
    df_display: pd.DataFrame,
    df_all_display: pd.DataFrame | None,
    axis_options: dict,
    x_axis: str,
    y_axis: str,
    z_axis: str | None,
    color_col: str,
    show_all: bool,
    use_3d: bool,
    colors: dict,
) -> go.Figure:
    layout_settings = dict(
        height=550,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.5)",
        font=dict(family="Inter", color=colors["text_body"]),
        uirevision="pareto_fixed",  # keep zoom/pan stable across reruns
    )

    fig = go.Figure()

    # --- history (explored): use Scattergl for speed in 2D ---
    if show_all and df_all_display is not None and len(df_all_display) > 0:
        customdata_exp = np.char.add("exp_", np.arange(len(df_all_display)).astype(str))

        if use_3d and z_axis:
            fig.add_trace(
                go.Scatter3d(
                    x=df_all_display[x_axis],
                    y=df_all_display[y_axis],
                    z=df_all_display[z_axis],
                    mode="markers",
                    name="Explored",
                    marker=dict(size=3, color="lightgrey", opacity=0.25),
                    customdata=customdata_exp,
                    hoverinfo="skip",  # big speed-up
                )
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=df_all_display[x_axis],
                    y=df_all_display[y_axis],
                    mode="markers",
                    name="Explored",
                    marker=dict(size=6, color="lightgrey", opacity=0.35),
                    customdata=customdata_exp,
                    hoverinfo="skip",  # big speed-up
                    selected=dict(marker=dict(size=12, color=colors["accent"], opacity=1.0)),
                    unselected=dict(marker=dict(color="lightgrey", opacity=0.35)),
                )
            )

    # --- pareto: keep hover light ---
    customdata_par = np.char.add("par_", np.arange(len(df_display)).astype(str))
    colorbar_title = "Decision Score<br>(TOPSIS)" if color_col == "Decision_Score" else color_col

    if use_3d and z_axis:
        fig.add_trace(
            go.Scatter3d(
                x=df_display[x_axis],
                y=df_display[y_axis],
                z=df_display[z_axis],
                mode="markers",
                name="Pareto",
                customdata=customdata_par,
                marker=dict(
                    size=6,
                    color=df_display[color_col],
                    colorscale="Viridis",
                    opacity=0.95,
                    showscale=True,
                    colorbar=dict(title=dict(text=colorbar_title, side="right")),
                ),
                hovertemplate=(
                    "Rank=%{customdata}<br>"
                    + f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}"
                    + (f"<br>{z_axis}=%{{z}}" if z_axis else "")
                    + "<extra></extra>"
                ),
                showlegend=True,
            )
        )
        fig.update_layout(scene=dict(
            xaxis_title=axis_options[x_axis],
            yaxis_title=axis_options[y_axis],
            zaxis_title=axis_options[z_axis],
        ))
    else:
        fig.add_trace(
            go.Scattergl(
                x=df_display[x_axis],
                y=df_display[y_axis],
                mode="markers",
                name="Pareto",
                customdata=customdata_par,
                marker=dict(
                    size=12,
                    color=df_display[color_col],
                    colorscale="Viridis",
                    opacity=0.95,
                    showscale=True,
                    colorbar=dict(title=dict(text=colorbar_title, side="right"), thickness=15, xpad=10),
                ),
                hovertemplate=(
                    "%{customdata}<br>"
                    + f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra></extra>"
                ),
                showlegend=True,
                selected=dict(marker=dict(size=18, color="red", opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.95)),
            )
        )
        fig.update_xaxes(title=axis_options[x_axis], showgrid=True, gridcolor=colors["border"])
        fig.update_yaxes(title=axis_options[y_axis], showgrid=True, gridcolor=colors["border"])

    # --- Top-1 marker ---
    if len(df_display) > 0:
        best_row = df_display.iloc[0]
        if use_3d and z_axis:
            fig.add_trace(
                go.Scatter3d(
                    x=[best_row[x_axis]], y=[best_row[y_axis]], z=[best_row[z_axis]],
                    mode="markers+text",
                    name="Top-1",
                    customdata=["par_0"],
                    marker=dict(size=10, symbol="diamond", color="red", line=dict(color="black", width=2)),
                    text=["Top-1"],
                    textposition="top center",
                    hoverinfo="skip",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[best_row[x_axis]], y=[best_row[y_axis]],
                    mode="markers+text",
                    name="Top-1",
                    customdata=["par_0"],
                    marker=dict(size=20, symbol="star", color="red", line=dict(color="black", width=1)),
                    text=["Top-1"],
                    textposition="top center",
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=colors["border"],
            borderwidth=1,
        ),
        **layout_settings,
    )
    return fig


@st.cache_data(show_spinner=False)
def build_position_figure(
    df_display: pd.DataFrame,
    df_all_display: pd.DataFrame | None,
    axis_options: dict,
    baseline_data: dict,
    selected_row: pd.Series,
    x_axis: str, y_axis: str, z_axis: str | None,
    color_col: str,
    show_all: bool,
    use_3d: bool,
    colors: dict,
) -> go.Figure:
    layout_settings = dict(
        height=520,
        margin=dict(l=20, r=80, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.5)",
        font=dict(family="Inter", color=colors["text_body"]),
        uirevision="bench_fixed",
    )
    fig = go.Figure()

    if show_all and df_all_display is not None and len(df_all_display) > 0:
        if use_3d and z_axis:
            fig.add_trace(go.Scatter3d(
                x=df_all_display[x_axis], y=df_all_display[y_axis], z=df_all_display[z_axis],
                mode="markers", name="Explored",
                marker=dict(size=2, color="lightgrey", opacity=0.2),
                hoverinfo="skip",
            ))
        else:
            fig.add_trace(go.Scattergl(
                x=df_all_display[x_axis], y=df_all_display[y_axis],
                mode="markers", name="Explored",
                marker=dict(size=5, color="lightgrey", opacity=0.25),
                hoverinfo="skip",
            ))

    colorbar_title = "Decision Score<br>(TOPSIS)" if color_col == "Decision_Score" else color_col

    if use_3d and z_axis:
        fig.add_trace(go.Scatter3d(
            x=df_display[x_axis], y=df_display[y_axis], z=df_display[z_axis],
            mode="markers", name="Pareto",
            marker=dict(
                size=6, color=df_display[color_col], colorscale="Viridis",
                opacity=0.95, showscale=True,
                colorbar=dict(title=dict(text=colorbar_title, side="right")),
            ),
            hoverinfo="skip",
        ))
        fig.update_layout(scene=dict(
            xaxis_title=axis_options.get(x_axis, x_axis),
            yaxis_title=axis_options.get(y_axis, y_axis),
            zaxis_title=axis_options.get(z_axis, z_axis),
        ))
    else:
        fig.add_trace(go.Scattergl(
            x=df_display[x_axis], y=df_display[y_axis],
            mode="markers", name="Pareto",
            marker=dict(
                size=10, color=df_display[color_col], colorscale="Viridis",
                opacity=0.95, showscale=True,
                colorbar=dict(title=dict(text=colorbar_title, side="right"), thickness=15, xpad=10),
            ),
            hoverinfo="skip",
        ))
        fig.update_xaxes(title=axis_options.get(x_axis, x_axis), showgrid=True, gridcolor=colors["border"])
        fig.update_yaxes(title=axis_options.get(y_axis, y_axis), showgrid=True, gridcolor=colors["border"])

    # Baseline marker (if available)
    def _baseline_has(k: str | None) -> bool:
        return (k is not None) and (k in baseline_data)

    if _baseline_has(x_axis) and _baseline_has(y_axis) and (not use_3d or _baseline_has(z_axis)):
        if use_3d and z_axis:
            fig.add_trace(go.Scatter3d(
                x=[baseline_data[x_axis]], y=[baseline_data[y_axis]], z=[baseline_data[z_axis]],
                mode="markers+text", name="Baseline (OPC)",
                marker=dict(size=9, symbol="x", color="gray", line=dict(width=2)),
                text=["Baseline"], textposition="top center",
                hoverinfo="skip",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[baseline_data[x_axis]], y=[baseline_data[y_axis]],
                mode="markers+text", name="Baseline (OPC)",
                marker=dict(size=14, symbol="x", color="gray", line=dict(width=2)),
                text=["Baseline"], textposition="top center",
                hoverinfo="skip",
            ))

    # Selected marker
    if use_3d and z_axis:
        fig.add_trace(go.Scatter3d(
            x=[float(selected_row[x_axis])], y=[float(selected_row[y_axis])], z=[float(selected_row[z_axis])],
            mode="markers+text", name="Selected Mix",
            marker=dict(size=9, symbol="star", color="red", line=dict(color="black", width=1)),
            text=["Selected"], textposition="top center",
            hoverinfo="skip",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[float(selected_row[x_axis])], y=[float(selected_row[y_axis])],
            mode="markers+text", name="Selected Mix",
            marker=dict(size=18, symbol="star", color="red", line=dict(color="black", width=1)),
            text=["Selected"], textposition="top center",
            hoverinfo="skip",
        ))

    fig.update_layout(
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=colors["border"],
            borderwidth=1,
        ),
        **layout_settings,
    )
    return fig


# ==========================================================
# 4) Main Results UI
# ==========================================================
@st.fragment
def show_results(COLORS, MATERIALS_CONFIG):
    """
    Render the Analytics Dashboard & Results.
    - Keeps all your features
    - Faster in Streamlit Cloud
    """
    if st.session_state.get("df_pareto") is None:
        return

    df_pareto = st.session_state.df_pareto
    df_all = st.session_state.get("df_all")

    # ---- prepare once (cached) ----
    obj_cfg = _objective_config_from_session()
    obj_sig = _objective_signature(obj_cfg)

    df_display, df_all_display, (criteria_cols, impacts) = prepare_display_pipeline(
        df_pareto=df_pareto,
        df_all=df_all,
        obj_sig=obj_sig,
        history_sample_cap=25000,  # tune this if you want
    )

    # Keep same naming for downstream code
    objective_config = obj_cfg

    axis_options = {
        "E": "E (GPa) - MAX",
        "CO2_abs": "CO₂ Uptake (kg/kg) - MAX",
        "CO2_emission": "CO₂ Emission (kg/kg) - MIN",
        "Cost": "Cost (€/kg) - MIN",
        "Net_emission": "Net Emission (kg/kg) - MIN",
    }

    with st.expander("📊 Analytics Dashboard & Results", expanded=True):

        tab_names = [
            "📉 Pareto Analysis",
            "🎞️ GA Animation",
            "🕸️ Parallel Coordinates",
            "📋 Data Table",
            "⚖️ Benchmark Comparison",
        ]
        tab_pareto, tab_anim, tab_parallel, tab_table, tab_bench = st.tabs(tab_names)

        # ==========================================================
        # Tab 1: Pareto Analysis
        # ==========================================================
        with tab_pareto:
            t1, t2 = st.columns([12, 1], gap="small")
            with t1:
                st.markdown("### 📈 Interactive Pareto Visualization")
            with t2:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to play with this plot")
                    st.write("• Click any point to reveal recipe on the right.")
                    st.write("• Grey = explored (history), Colored = Pareto frontier.")

            chart_options = [k for k in axis_options.keys() if k in df_display.columns]
            if len(chart_options) == 0:
                st.warning("No valid axis columns found in df_pareto.")
                return

            c_ctrl = st.columns(5)
            default_x = chart_options[0]
            default_y = chart_options[1] if len(chart_options) >= 2 else chart_options[0]
            default_z = chart_options[2] if len(chart_options) >= 3 else chart_options[0]

            x_axis = c_ctrl[0].selectbox(
                "X Axis",
                chart_options,
                index=chart_options.index(default_x),
                format_func=lambda x: axis_options[x],
                key="pareto_x",
            )
            y_axis = c_ctrl[1].selectbox(
                "Y Axis",
                chart_options,
                index=chart_options.index(default_y),
                format_func=lambda x: axis_options[x],
                key="pareto_y",
            )

            material_cols = [m[1] for m in MATERIALS_CONFIG]
            color_candidates = ["Decision_Score"] + chart_options + material_cols
            color_options = [c for c in color_candidates if c in df_display.columns]
            default_color_idx = color_options.index("Decision_Score") if "Decision_Score" in color_options else 0

            color_col = c_ctrl[2].selectbox("Color Points By", color_options, index=default_color_idx, key="pareto_color")
            show_all = c_ctrl[3].toggle("Show History", value=True, key="pareto_hist")

            use_3d_allowed = len(chart_options) >= 3
            use_3d = c_ctrl[4].toggle("3D View", value=False, key="pareto_3d") if use_3d_allowed else False
            z_axis = st.selectbox(
                "Z Axis",
                chart_options,
                index=chart_options.index(default_z),
                format_func=lambda x: axis_options[x],
                key="pareto_z",
            ) if (use_3d and use_3d_allowed) else None

            col_plot, col_details = st.columns([2.8, 1.2], gap="large")

            with col_plot:
                fig = build_pareto_figure(
                    df_display=df_display,
                    df_all_display=df_all_display,
                    axis_options=axis_options,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    z_axis=z_axis,
                    color_col=color_col,
                    show_all=show_all,
                    use_3d=bool(use_3d and z_axis),
                    colors=COLORS,
                )

                plot_event = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key="pareto_click_event",
                )

            with col_details:
                st.markdown("#### 🔍 Recipe Spotlight")

                if "selected_mix_info" not in st.session_state:
                    st.session_state.selected_mix_info = None

                # Safe parse selection
                if plot_event and isinstance(plot_event, dict) and "selection" in plot_event:
                    pts = plot_event["selection"].get("points", [])
                    if pts and "customdata" in pts[0]:
                        cd = pts[0]["customdata"]
                        cd_val = cd[0] if isinstance(cd, list) else cd
                        if isinstance(cd_val, str):
                            if cd_val.startswith("exp_"):
                                st.session_state.selected_mix_info = {"idx": int(cd_val.split("_")[1]), "type": 0}
                            elif cd_val.startswith("par_"):
                                st.session_state.selected_mix_info = {"idx": int(cd_val.split("_")[1]), "type": 1}

                sm = st.session_state.selected_mix_info
                if sm is not None:
                    mix_idx = sm["idx"]
                    mix_type = sm["type"]

                    sel_row = None
                    if mix_type == 1 and mix_idx < len(df_display):
                        sel_row = df_display.iloc[mix_idx]
                        score_val = float(sel_row.get("Decision_Score", 0.0))
                        st.markdown(
                            f"""
                            <div style="background:{COLORS['primary']}15; padding:15px; border-radius:10px; border:1px solid {COLORS['primary']}50; text-align:center; margin-bottom:15px;">
                                <h3 style="margin:0; color:{COLORS['primary']};">🏆 Rank #{mix_idx + 1}</h3>
                                <span style="font-size:0.9rem; color:{COLORS['text_sub']};">TOPSIS Score: <b>{score_val:.3f}</b></span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif mix_type == 0 and df_all_display is not None and mix_idx < len(df_all_display):
                        sel_row = df_all_display.iloc[mix_idx]
                        gen = sel_row.get("Generation", "-")
                        st.markdown(
                            f"""
                            <div style="background:#F1F5F9; padding:15px; border-radius:10px; border:1px solid #CBD5E1; text-align:center; margin-bottom:15px;">
                                <h3 style="margin:0; color:#475569;">🔍 Explored #{mix_idx + 1}</h3>
                                <span style="font-size:0.9rem; color:#64748B;">Generation: <b>{gen}</b> (Not in Pareto)</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if sel_row is not None:
                        m_c1, m_c2 = st.columns(2)
                        if "E" in sel_row: m_c1.metric("E-Modulus", f"{float(sel_row['E']):.1f} GPa")
                        if "Cost" in sel_row: m_c2.metric("Cost", f"€{float(sel_row['Cost']):.3f}")
                        m_c3, m_c4 = st.columns(2)
                        if "CO2_emission" in sel_row: m_c3.metric("CO₂ Emis.", f"{float(sel_row['CO2_emission']):.2f} kg/kg")
                        if "Net_emission" in sel_row: m_c4.metric("Net CO₂", f"{float(sel_row['Net_emission']):.2f} kg/kg")

                        st.markdown("---")

                        mats = {
                            k: float(sel_row[k])
                            for k in material_cols
                            if (k in sel_row.index) and (float(sel_row[k]) > 0)
                        }
                        if mats:
                            df_m = pd.DataFrame(list(mats.items()), columns=["Material", "Value"])
                            fig_p = px.pie(df_m, values="Value", names="Material", hole=0.6)
                            fig_p.update_traces(
                                textposition="inside",
                                textinfo="none",
                                hovertemplate="%{label}: %{value:.1f}g<extra></extra>",
                            )
                            fig_p.update_layout(
                                showlegend=True,
                                legend=dict(orientation="h", y=-0.2),
                                margin=dict(t=0, b=0, l=0, r=0),
                                height=220,
                                paper_bgcolor="rgba(0,0,0,0)",
                                annotations=[dict(
                                    text="Binder<br>Mix", x=0.5, y=0.5, font_size=14,
                                    showarrow=False, font=dict(color=COLORS["text_sub"], family="Inter")
                                )],
                            )
                            st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.markdown(
                        f"""
                        <div style="border: 2px dashed {COLORS['border']}; border-radius: 12px; padding: 60px 20px; text-align: center; color: {COLORS['text_sub']}; margin-top: 20px;">
                            <h1 style="font-size: 3rem; margin-bottom: 10px; color:{COLORS['primary']};">🖱️</h1>
                            <p style="font-size: 1rem; margin: 0;"><b>Click colored point</b> on the plot to reveal its exact recipe.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ==========================================================
        # Tab 2: GA Animation (keep your logic; slightly lighter)

        with tab_anim:
            a1, a2 = st.columns([12, 1], gap="small")
            with a1:
                st.markdown("### 🎞️ Evolution of Population")
            with a2:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to read this animation")
                    st.write("• Current gen in color; past dots can be shown as ghost trails.")
        
            if df_all_display is None or "Generation" not in df_all_display.columns:
                st.warning("⚠️ No history found. Please re-run optimization.")
            else:
                anim_axis_candidates = [k for k in axis_options.keys() if k in df_all_display.columns]
                if len(anim_axis_candidates) < 2:
                    st.warning("Not enough columns for animation axes.")
                else:
                    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1], gap="medium")
                    default_x = "Net_emission" if "Net_emission" in anim_axis_candidates else anim_axis_candidates[0]
                    default_y = "E" if "E" in anim_axis_candidates else anim_axis_candidates[1]
        
                    x_anim = c1.selectbox(
                        "X Axis",
                        anim_axis_candidates,
                        index=anim_axis_candidates.index(default_x),
                        format_func=lambda x: axis_options.get(x, x),
                        key="anim_x",
                    )
                    y_anim = c2.selectbox(
                        "Y Axis",
                        anim_axis_candidates,
                        index=anim_axis_candidates.index(default_y),
                        format_func=lambda x: axis_options.get(x, x),
                        key="anim_y",
                    )
        
                    show_ghosts = c3.toggle("🌌 Show Ghost Trails", value=True)
                    speed = c4.select_slider("Animation Speed", options=[50, 150, 300, 600], value=150)
        
                    # --- clean + numeric (avoid NaN ranges => blank plot) ---
                    df_anim = df_all_display[[x_anim, y_anim, "Generation"]].copy()
                    df_anim = df_anim.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_anim, y_anim, "Generation"])
                    if len(df_anim) == 0:
                        st.warning("⚠️ No valid animation data after removing NaNs/Infs.")
                    else:
                        x_min = float(df_anim[x_anim].min())
                        x_max = float(df_anim[x_anim].max())
                        y_min = float(df_anim[y_anim].min())
                        y_max = float(df_anim[y_anim].max())
        
                        # pad
                        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
                        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
                        x_min, x_max = x_min - x_pad, x_max + x_pad
                        y_min, y_max = y_min - y_pad, y_max + y_pad
        
                        # --- generations (keep order stable) ---
                        gens = sorted(df_anim["Generation"].unique(), key=lambda v: (str(type(v)), str(v)))
        
                        # --- sampling to keep animation light (critical on Cloud) ---
                        # total cap across all frames:
                        CAP_TOTAL = 20000
                        per_gen_cap = max(200, min(2000, CAP_TOTAL // max(1, len(gens))))
        
                        frames = []
                        for g in gens:
                            df_curr = df_anim[df_anim["Generation"] == g]
                            if len(df_curr) > per_gen_cap:
                                df_curr = df_curr.sample(per_gen_cap, random_state=1)
        
                            data_in_frame = []
        
                            # Trace 0: ghosts
                            if show_ghosts and g != gens[0]:
                                df_past = df_anim[df_anim["Generation"] < g]
                                # sample past too (otherwise grows huge)
                                if len(df_past) > (per_gen_cap * 6):
                                    df_past = df_past.sample(per_gen_cap * 6, random_state=1)
        
                                data_in_frame.append(go.Scatter(
                                    x=df_past[x_anim],
                                    y=df_past[y_anim],
                                    mode="markers",
                                    marker=dict(size=4, color="lightgrey", opacity=0.12),
                                    hoverinfo="skip",
                                    showlegend=False,
                                ))
                            else:
                                data_in_frame.append(go.Scatter(x=[], y=[], mode="markers", showlegend=False))
        
                            # Trace 1: current gen
                            data_in_frame.append(go.Scatter(
                                x=df_curr[x_anim],
                                y=df_curr[y_anim],
                                mode="markers",
                                marker=dict(size=9, color=COLORS["primary"], opacity=0.75, line=dict(width=1, color="white")),
                                hoverinfo="skip",
                                showlegend=False,
                            ))
        
                            frames.append(go.Frame(
                                name=str(g),
                                data=data_in_frame,
                                traces=[0, 1],
                            ))
        
                        # --- initial data ---
                        df0 = df_anim[df_anim["Generation"] == gens[0]]
                        if len(df0) > per_gen_cap:
                            df0 = df0.sample(per_gen_cap, random_state=1)
        
                        fig_anim = go.Figure(
                            data=[
                                # trace 0: ghosts placeholder
                                go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip"),
                                # trace 1: current gen initial
                                go.Scatter(
                                    x=df0[x_anim],
                                    y=df0[y_anim],
                                    mode="markers",
                                    marker=dict(size=9, color=COLORS["primary"], opacity=0.75, line=dict(width=1, color="white")),
                                    showlegend=False,
                                    hoverinfo="skip",
                                ),
                            ],
                            layout=go.Layout(
                                xaxis=dict(range=[x_min, x_max], title=axis_options.get(x_anim), gridcolor=COLORS["border"], zeroline=False),
                                yaxis=dict(range=[y_min, y_max], title=axis_options.get(y_anim), gridcolor=COLORS["border"], zeroline=False),
                                template="plotly_white",
                                margin=dict(l=40, r=40, t=20, b=40),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(255,255,255,0.7)",
                                hovermode="closest",
                                uirevision="anim_fixed",
                            ),
                            frames=frames,
                        )
        
                        # --- static overlay: pareto frontier (optional) ---
                        if (x_anim in df_display.columns) and (y_anim in df_display.columns) and (len(df_display) > 0):
                            fig_anim.add_trace(go.Scatter(
                                x=df_display[x_anim],
                                y=df_display[y_anim],
                                mode="markers",
                                name="Target Frontier",
                                marker=dict(size=10, symbol="diamond-open", color=COLORS["accent"], line=dict(width=2)),
                                hoverinfo="skip",
                            ))
        
                        # --- controls ---
                        fig_anim.update_layout(
                            updatemenus=[dict(
                                type="buttons",
                                showactive=False,
                                x=0,
                                y=1.15,
                                buttons=[
                                    dict(
                                        label="▶ Play",
                                        method="animate",
                                        args=[None, dict(
                                            frame=dict(duration=int(speed), redraw=True),
                                            fromcurrent=True,
                                            transition=dict(duration=0),
                                            mode="immediate"
                                        )]
                                    ),
                                    dict(
                                        label="|| Pause",
                                        method="animate",
                                        args=[[None], dict(
                                            frame=dict(duration=0, redraw=False),
                                            mode="immediate",
                                            transition=dict(duration=0)
                                        )]
                                    ),
                                ],
                            )],
                            sliders=[dict(
                                active=0,
                                currentvalue={"prefix": "Evolution Progress: Gen "},
                                pad={"t": 50},
                                x=0,
                                y=0,
                                len=1.0,
                                steps=[
                                    dict(
                                        label=str(g),
                                        method="animate",
                                        args=[[str(g)], dict(
                                            mode="immediate",
                                            frame=dict(duration=0, redraw=True),
                                            transition=dict(duration=0)
                                        )]
                                    )
                                    for g in gens
                                ],
                            )],
                        )
        
                        st.plotly_chart(fig_anim, use_container_width=True)

        # ==========================================================
        # Tab 3: Parallel Coordinates (kept)
        # ==========================================================
        with tab_parallel:
            p1, p2 = st.columns([12, 1], gap="small")
            with p1:
                st.markdown("##### 🕸️ Multi-Dimensional Mix Analysis")
            with p2:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to play with this plot")
                    st.write("• Drag vertically on any axis to filter ranges.")
                    st.write("• Spotlight highlights a single mix path.")

            all_input_cols = [m[1] for m in MATERIALS_CONFIG]
            all_output_cols = ["E", "CO2_emission", "Cost", "Net_emission", "Decision_Score"]
            all_available_cols = [c for c in (all_input_cols + all_output_cols) if c in df_display.columns]

            default_selection = ["C3S", "C2S", "GGBFS", "E", "Cost", "CO2_emission"]
            default_selection = [c for c in default_selection if c in all_available_cols]

            c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([3, 1.5, 1.5, 1.5])
            selected_cols = c_ctrl1.multiselect("📌 Active Dimensions", options=all_available_cols, default=default_selection)

            if len(selected_cols) > 1:
                color_pick_options = [c for c in all_available_cols if c in df_display.columns]
                default_color = "E" if "E" in color_pick_options else color_pick_options[0]
                color_col_pc = c_ctrl2.selectbox("🎯 Color Lines By", options=color_pick_options, index=color_pick_options.index(default_color))
                theme = c_ctrl3.selectbox("🎨 Color Theme", options=["Turbo", "Viridis", "Plasma", "Inferno", "RdBu"])
                spotlight_mix = c_ctrl4.selectbox(
                    "🔦 Spotlight Mix",
                    options=["None"] + [f"Rank #{i+1}" for i in range(len(df_display))],
                    index=0,
                )

                df_pc = df_display.copy()

                if spotlight_mix != "None":
                    mix_idx = int(spotlight_mix.split("#")[1]) - 1
                    df_pc["Spotlight"] = 0
                    if 0 <= mix_idx < len(df_pc):
                        df_pc.loc[mix_idx, "Spotlight"] = 1
                        target_row = df_pc.iloc[[mix_idx]]
                        df_pc = pd.concat([df_pc.drop(mix_idx), target_row], ignore_index=True)

                    color_target = "Spotlight"
                    color_scale = [[0, "rgba(200, 200, 200, 0.12)"], [1, "rgba(255, 0, 50, 1)"]]
                    hide_colorbar = True
                else:
                    color_target = color_col_pc
                    color_scale = theme.lower()
                    hide_colorbar = False

                fig_par = px.parallel_coordinates(
                    df_pc,
                    dimensions=selected_cols,
                    color=color_target,
                    labels={col: col.replace("_", " ") for col in selected_cols},
                    color_continuous_scale=color_scale,
                )

                if hide_colorbar:
                    fig_par.update_coloraxes(showscale=False)
                else:
                    fig_par.update_coloraxes(colorbar_title=color_col_pc)

                fig_par.update_traces(
                    labelfont=dict(size=14, color=COLORS["text_head"], family="Inter"),
                    tickfont=dict(size=11, color=COLORS["text_sub"], family="Inter"),
                    rangefont=dict(size=10, color=COLORS["text_sub"], family="Inter"),
                )
                fig_par.update_layout(
                    margin=dict(l=50, r=50, t=60, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    uirevision="pc_fixed",
                )

                st.plotly_chart(fig_par, use_container_width=True)
            else:
                st.warning("⚠️ Select at least two dimensions to visualize.")

        # ==========================================================
        # Tab 4: Data Table (kept; only minor cleanups)
        # ==========================================================
        with tab_table:
            tab_pareto_sub, tab_search_sub = st.tabs(["🏆 Pareto Optimal Mixes", "🎯 Target Search (All Mixes)"])

            with tab_pareto_sub:
                d1, d2 = st.columns([12, 1], gap="small")
                with d1:
                    st.markdown("### 📋 Pareto Optimal Mixes")
                with d2:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    with st.popover("ℹ️"):
                        st.write("• Each row is one Pareto-optimal mix.")

                m1, m2, m3 = st.columns(3)
                if "E" in df_display.columns: m1.metric("Max E Found", f"{float(df_display['E'].max()):.2f} GPa")
                if "Cost" in df_display.columns: m2.metric("Min Cost", f"€{float(df_display['Cost'].min()):.2f} /kg")
                if "Net_emission" in df_display.columns: m3.metric("Min Net CO₂", f"{float(df_display['Net_emission'].min()):.3f} kg/kg")

                column_configuration = {}
                if "E" in df_display.columns:
                    column_configuration["E"] = st.column_config.ProgressColumn(
                        "E (GPa)", format="%.2f",
                        min_value=float(df_display["E"].min()),
                        max_value=float(df_display["E"].max()),
                    )
                if "Cost" in df_display.columns:
                    column_configuration["Cost"] = st.column_config.ProgressColumn(
                        "Cost (€/kg)", format="€%.4f",
                        min_value=float(df_display["Cost"].min()),
                        max_value=float(df_display["Cost"].max()),
                    )
                if "CO2_emission" in df_display.columns:
                    column_configuration["CO2_emission"] = st.column_config.ProgressColumn(
                        "CO₂ (kg/kg)", format="%.3f",
                        min_value=float(df_display["CO2_emission"].min()),
                        max_value=float(df_display["CO2_emission"].max()),
                    )
                if "Net_emission" in df_display.columns:
                    column_configuration["Net_emission"] = st.column_config.ProgressColumn(
                        "Net CO₂ (kg/kg)", format="%.3f",
                        min_value=float(df_display["Net_emission"].min()),
                        max_value=float(df_display["Net_emission"].max()),
                    )
                if "CO2_abs" in df_display.columns:
                    column_configuration["CO2_abs"] = st.column_config.NumberColumn("Uptake (kg/kg)", format="%.3f")
                if "Decision_Score" in df_display.columns:
                    column_configuration["Decision_Score"] = st.column_config.ProgressColumn(
                        "Decision Score (TOPSIS)", format="%.3f", min_value=0.0, max_value=1.0
                    )

                for col in material_cols:
                    if col in df_display.columns:
                        column_configuration[col] = st.column_config.NumberColumn(f"{col} (gram)", format="%.1f")

                st.dataframe(df_display, column_config=column_configuration, use_container_width=True, height=520)
                st.download_button(
                    "📥 Download Pareto CSV",
                    df_display.to_csv(index=False).encode("utf-8"),
                    "pareto_results.csv",
                    "text/csv",
                    type="primary",
                )

            with tab_search_sub:
                st.markdown("### 🎯 Target Search (Goal-Seeker Box)")
                st.markdown(
                    "<span style='color:#64748B; font-size:0.9rem;'>Use the dual-ended sliders to draw a <b>bounding box</b> and target a specific zone of mixes.</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

                if df_all_display is None:
                    st.warning("⚠️ Optimization history is required for Target Search. Please run the optimization first.")
                else:
                    c_filters, c_plot = st.columns([1, 2.5], gap="large")
                    with c_filters:
                        st.markdown("#### 1. Set Target Box")

                        min_e, max_e = float(df_all_display["E"].min()), float(df_all_display["E"].max())
                        min_cost, max_cost = float(df_all_display["Cost"].min()), float(df_all_display["Cost"].max())
                        min_co2, max_co2 = float(df_all_display["CO2_emission"].min()), float(df_all_display["CO2_emission"].max())

                        def_e_range = (max(min_e, min(15.0, max_e)), max_e)
                        def_cost_range = (min_cost, max(min_cost, min(0.12, max_cost)))
                        def_co2_range = (min_co2, max(min_co2, min(0.60, max_co2)))

                        target_e = st.slider("⚡ E-Modulus Range (GPa)", min_value=min_e, max_value=max_e, value=def_e_range, step=0.5)
                        target_co2 = st.slider("🏭 CO₂ Emission Range (kg/kg)", min_value=min_co2, max_value=max_co2, value=def_co2_range, step=0.01, format="%.3f")
                        target_cost = st.slider("💰 Cost Range (€/kg)", min_value=min_cost, max_value=max_cost, value=def_cost_range, step=0.005, format="%.3f")

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

                        fig_target.add_trace(go.Scattergl(
                            x=df_all_display["CO2_emission"], y=df_all_display["E"],
                            mode="markers", name="All Explored",
                            marker=dict(size=5, color="lightgrey", opacity=0.25),
                            hoverinfo="skip",
                        ))

                        fig_target.add_shape(
                            type="rect",
                            x0=target_co2[0], y0=target_e[0],
                            x1=target_co2[1], y1=target_e[1],
                            line=dict(color="red", width=2, dash="dash"),
                            fillcolor="rgba(255, 0, 0, 0.05)",
                            layer="below",
                        )

                        if len(df_filtered) > 0:
                            fig_target.add_trace(go.Scattergl(
                                x=df_filtered["CO2_emission"], y=df_filtered["E"],
                                mode="markers", name="Viable Mixes",
                                marker=dict(
                                    size=8,
                                    color=df_filtered["Cost"],
                                    colorscale="Viridis",
                                    showscale=True,
                                    colorbar=dict(title="Cost (€/kg)", thickness=15),
                                    line=dict(width=1, color="white"),
                                ),
                                hovertemplate="E=%{y:.2f}<br>CO2=%{x:.3f}<br>Cost=%{marker.color:.3f}<extra></extra>",
                            ))

                        fig_target.update_layout(
                            xaxis_title="CO₂ Emission (kg/kg) - Lower is better",
                            yaxis_title="E-Modulus (GPa) - Higher is better",
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.5)",
                            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)"),
                            uirevision="target_fixed",
                        )
                        st.plotly_chart(fig_target, use_container_width=True)

                    if len(df_filtered) > 0:
                        st.markdown("#### 3. Viable Mixes Data")
                        df_filtered_sorted = df_filtered.sort_values(by="CO2_emission", ascending=True).reset_index(drop=True)

                        display_cols = ["E", "CO2_emission", "Cost", "Net_emission"] + [m[1] for m in MATERIALS_CONFIG if m[1] in df_filtered_sorted.columns]
                        st.dataframe(
                            df_filtered_sorted[display_cols],
                            use_container_width=True,
                            height=250,
                            column_config={
                                "E": st.column_config.NumberColumn("E (GPa)", format="%.2f"),
                                "CO2_emission": st.column_config.NumberColumn("CO₂ (kg/kg)", format="%.3f"),
                                "Net_emission": st.column_config.NumberColumn("Net CO₂ (kg/kg)", format="%.3f"),
                                "Cost": st.column_config.NumberColumn("Cost (€/kg)", format="%.3f"),
                            },
                        )
                        st.download_button(
                            "📥 Download Viable Mixes",
                            data=df_filtered_sorted[display_cols].to_csv(index=False).encode("utf-8"),
                            file_name="target_search.csv",
                            mime="text/csv",
                            type="primary",
                        )

        # ==========================================================
        # Tab 5: Benchmark Comparison & PDF Report (kept, faster plots)
        # ==========================================================
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
                pareto_opts = {i: f"Mix #{i+1} (E={float(r['E']):.1f})" for i, r in df_display.iterrows()}
                selected_idx = st.selectbox("Select Solution", options=list(pareto_opts.keys()), format_func=lambda x: pareto_opts[x])

                if selected_idx is not None:
                    row = df_display.iloc[int(selected_idx)]
                    baseline_data = {
                        "E": float(opc_e),
                        "CO2_emission": float(opc_co2),
                        "Cost": float(opc_cost),
                        "Net_emission": float(opc_net),
                    }
                    st.markdown("---")

                    def delta_metric_card(label, val, ref, unit, inverse=False):
                        diff = float(val) - float(ref)
                        pct = (diff / float(ref) * 100.0) if float(ref) != 0 else 0.0
                        color_mode = "inverse" if inverse else "normal"
                        fmt = "%.3f" if "kg" in unit else "%.2f"
                        st.metric(
                            label=label,
                            value=f"{float(val):{fmt[1:]}} {unit}",
                            delta=f"{diff:{fmt[1:]}} ({pct:+.1f}%)",
                            delta_color=color_mode,
                        )

                    delta_metric_card("E-Modulus", row["E"], opc_e, "GPa")
                    delta_metric_card("CO₂ Emission", row["CO2_emission"], opc_co2, "kg/kg", inverse=True)
                    delta_metric_card("Cost", row["Cost"], opc_cost, "€/kg", inverse=True)

            with c_right2:
                if selected_idx is not None:
                    row = df_display.iloc[int(selected_idx)]
                    baseline_data = {
                        "E": float(opc_e),
                        "CO2_emission": float(opc_co2),
                        "Cost": float(opc_cost),
                        "Net_emission": float(opc_net),
                    }

                    row1_col1, row1_col2 = st.columns(2, gap="medium")

                    with row1_col1:
                        st.markdown("#### 🍩 Composition (gram)")
                        mat_cols = [m[1] for m in MATERIALS_CONFIG]
                        materials = {k: float(row[k]) for k in mat_cols if k in df_display.columns}
                        df_mat = pd.DataFrame(list(materials.items()), columns=["Material", "Value"])
                        df_mat = df_mat[df_mat["Value"] > 0]

                        fig_pie = px.pie(df_mat, values="Value", names="Material", hole=0.55)
                        fig_pie.update_traces(textposition="inside", textinfo="none")
                        fig_pie.update_layout(showlegend=False, margin=dict(t=30, b=10, l=10, r=10), height=280,
                                              annotations=[dict(text="Mix", x=0.5, y=0.5, font_size=14, showarrow=False)])
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
                        fig_bar.update_layout(height=280, margin=dict(t=20, b=10, l=0, r=0),
                                              legend=dict(orientation="h", y=1.02, x=0.6),
                                              paper_bgcolor="rgba(0,0,0,0)",
                                              plot_bgcolor="rgba(255,255,255,0.5)")
                        st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown("#### 🌳 Real-World Impact (Per 1 Ton of Binder)")
                    co2_saved_per_ton = (float(opc_co2) - float(row["CO2_emission"])) * 1000.0

                    if co2_saved_per_ton > 0:
                        trees_eq = int(co2_saved_per_ton / 22.0)
                        car_km_eq = int(co2_saved_per_ton / 0.12)
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #10B98115, #0F766E20); border-left: 4px solid #10B981; padding: 15px 20px; border-radius: 8px; display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px;">
                                <div style="text-align: center;">
                                    <div style="font-size: 2rem;">☁️</div>
                                    <div style="font-size: 1.2rem; font-weight: 700; color: #0F766E;">{co2_saved_per_ton:.0f} kg</div>
                                    <div style="font-size: 0.8rem; color: #64748B;">CO₂ Avoided</div>
                                </div>
                                <div style="font-size: 1.5rem; color: #475569;">=</div>
                                <div style="text-align: center;">
                                    <div style="font-size: 2rem;">🌲</div>
                                    <div style="font-size: 1.2rem; font-weight: 700; color: #10B981;">{trees_eq} Trees</div>
                                    <div style="font-size: 0.8rem; color: #64748B;">Planted (1 Year)</div>
                                </div>
                                <div style="font-size: 1.5rem; color: #475569;">or</div>
                                <div style="text-align: center;">
                                    <div style="font-size: 2rem;">🚗</div>
                                    <div style="font-size: 1.2rem; font-weight: 700; color: #F59E0B;">{car_km_eq:,} km</div>
                                    <div style="font-size: 0.8rem; color: #64748B;">Not Driven</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="background: #FEE2E2; border-left: 4px solid #EF4444; padding: 15px; border-radius: 8px;">
                                <span style="color: #B91C1C; font-weight: 600;">⚠️ Higher Emissions:</span> This mix produces {abs(co2_saved_per_ton):.0f} kg MORE CO₂ per ton than the baseline.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

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
                    color_candidates = ["Decision_Score"] + chart_options + material_cols
                    color_options = [c for c in color_candidates if c in df_display.columns]

                    default_x = "CO2_emission" if "CO2_emission" in chart_options else chart_options[0]
                    default_y = "E" if "E" in chart_options else (chart_options[1] if len(chart_options) > 1 else chart_options[0])
                    default_z = "Cost" if "Cost" in chart_options else (chart_options[2] if len(chart_options) > 2 else chart_options[0])

                    c_ctrl = st.columns(5)
                    x_axis = c_ctrl[0].selectbox("X Axis", chart_options, index=chart_options.index(default_x), format_func=lambda x: axis_options.get(x, x), key="bench_pos_x")
                    y_axis = c_ctrl[1].selectbox("Y Axis", chart_options, index=chart_options.index(default_y), format_func=lambda x: axis_options.get(x, x), key="bench_pos_y")
                    default_color_idx = color_options.index("Decision_Score") if "Decision_Score" in color_options else 0
                    color_col = c_ctrl[2].selectbox("Color Points By", color_options, index=default_color_idx, key="bench_pos_color")
                    show_all = c_ctrl[3].toggle("Show History", value=True, key="bench_pos_showhist")
                    use_3d_allowed = len(chart_options) >= 3
                    use_3d = c_ctrl[4].toggle("3D View", value=False, key="bench_pos_3d") if use_3d_allowed else False
                    z_axis = st.selectbox("Z Axis", chart_options, index=chart_options.index(default_z), format_func=lambda x: axis_options.get(x, x), key="bench_pos_z") if (use_3d and use_3d_allowed) else None

                    fig_pos = build_position_figure(
                        df_display=df_display,
                        df_all_display=df_all_display,
                        axis_options=axis_options,
                        baseline_data=baseline_data,
                        selected_row=row,
                        x_axis=x_axis,
                        y_axis=y_axis,
                        z_axis=z_axis,
                        color_col=color_col,
                        show_all=show_all,
                        use_3d=bool(use_3d and z_axis),
                        colors=COLORS,
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)

            # ==========================================================
            # FINAL SECTION: REPORT GENERATION (kept)
            # ==========================================================
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
                    "Target Binder": "96.0 g",
                }

                search_bounds = {
                    "clinker": {
                        "C3S (%)": st.session_state.get("c3s_rng_slider", (45.0, 80.0)),
                        "C2S (%)": st.session_state.get("c2s_rng_slider", (10.0, 32.0)),
                        "C3A (%)": st.session_state.get("c3a_rng_slider", (0.0, 14.0)),
                        "C4AF (%)": st.session_state.get("c4af_rng_slider", (0.0, 15.0)),
                    },
                    "scms": {
                        "Silica Fume (g)": st.session_state.get("sf_rng_slider", (0.0, 10.0)),
                        "GGBFS (g)": st.session_state.get("gg_rng_slider", (0.0, 80.0)),
                        "Fly Ash (g)": st.session_state.get("fa_rng_slider", (0.0, 35.0)),
                        "Calcined Clay (g)": st.session_state.get("cc_rng_slider", (0.0, 35.0)),
                        "Limestone (g)": st.session_state.get("ls_rng_slider", (0.0, 35.0)),
                    },
                    "total_clinker": st.session_state.get("cl_sum_rng_slider", (20.0, 96.0)),
                }

                ga_conf = {
                    "pop": st.session_state.get("ga_pop", 100),
                    "gen": st.session_state.get("ga_gen", 20),
                    "seed": st.session_state.get("ga_seed", 1),
                }

                # baseline_data may not exist if user never opened benchmark tab
                baseline_data_safe = {
                    "E": float(st.session_state.get("bench_e", 17.0)),
                    "CO2_emission": float(st.session_state.get("bench_co2", 0.76)),
                    "Cost": float(st.session_state.get("bench_cost", 0.22)),
                    "Net_emission": float(st.session_state.get("bench_net", 0.40)),
                }

                if st.button("🚀 Generate PDF Technical Audit", type="primary", use_container_width=True):
                    with st.spinner("Compiling technical report..."):
                        pdf_bytes = create_pdf_report(
                            df_display,
                            report_params,
                            baseline_data_safe,
                            objective_config,
                            search_bounds,
                            ga_conf,
                        )
                        st.session_state["pdf_data"] = pdf_bytes
                        st.success("Report Compiled Successfully!")

                if "pdf_data" in st.session_state and st.session_state["pdf_data"] is not None:
                    st.download_button(
                        label="📥 Download Technical Report",
                        data=st.session_state["pdf_data"],
                        file_name="Mix_Optimization_Audit.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Waiting for optimization settings to initialize... ({e})")
