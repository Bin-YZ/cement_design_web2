import base64

import streamlit as st
import streamlit.components.v1 as components

from app.paths import get_logo_path


DEFAULT_FONT_PRESET = "Standard"

FONT_SCALE_PRESETS = {
    "Standard": {
        "body": "18px",
        "small": "16px",
        "slider_label": "17px",
        "tab": "18px",
        "metric_label": "16px",
        "metric_value": "28px",
        "button": "18px",
        "h1": "2.45rem",
        "h2": "2rem",
        "h3": "1.55rem",
        "h4": "1.3rem",
        "header_title": "2rem",
        "header_version": "1.05rem",
        "header_subtitle": "1.02rem",
        "header_meta": "0.95rem",
        "header_note": "0.9rem",
        "header_ref": "0.82rem",
        "panel_title": "1.15rem",
        "fixed_card": "1rem",
        "header_height": 455,
    },
    "Large": {
        "body": "20px",
        "small": "18px",
        "slider_label": "19px",
        "tab": "20px",
        "metric_label": "18px",
        "metric_value": "32px",
        "button": "20px",
        "h1": "2.65rem",
        "h2": "2.15rem",
        "h3": "1.7rem",
        "h4": "1.45rem",
        "header_title": "2.2rem",
        "header_version": "1.12rem",
        "header_subtitle": "1.08rem",
        "header_meta": "1.05rem",
        "header_note": "1rem",
        "header_ref": "0.9rem",
        "panel_title": "1.25rem",
        "fixed_card": "1.08rem",
        "header_height": 490,
    },
    "Extra Large": {
        "body": "22px",
        "small": "20px",
        "slider_label": "21px",
        "tab": "22px",
        "metric_label": "20px",
        "metric_value": "36px",
        "button": "22px",
        "h1": "2.9rem",
        "h2": "2.35rem",
        "h3": "1.85rem",
        "h4": "1.58rem",
        "header_title": "2.4rem",
        "header_version": "1.2rem",
        "header_subtitle": "1.16rem",
        "header_meta": "1.12rem",
        "header_note": "1.08rem",
        "header_ref": "0.98rem",
        "panel_title": "1.35rem",
        "fixed_card": "1.16rem",
        "header_height": 525,
    },
}


def get_font_scale_config(preset):
    selected_preset = preset if preset in FONT_SCALE_PRESETS else DEFAULT_FONT_PRESET
    font_config = FONT_SCALE_PRESETS[selected_preset].copy()
    font_config["preset"] = selected_preset
    return font_config


@st.cache_data(show_spinner=False)
def get_base64_img(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode()
    except Exception:
        return None


def apply_page_style(colors, font_config):
    note_html = f"""
<ul style="margin:10px 0 0 18px; padding:0; font-size:{font_config['header_note']}; color:#334155; line-height:1.45;">
<li>
  <b>ANN Model outputs:</b> E-modulus (GPa) + max CO2 uptake (kg/kg). The E-modulus is derived by applying
  <b>micromechanical homogenization</b> to the <b>GEMS-simulated hydrate assemblages</b>.
  <details style="display:inline; cursor:pointer; color:#0F766E;">
    <summary style="list-style:none; display:inline; font-size:{font_config['header_ref']}; text-decoration:underline;">[Ref]</summary>
    <div style="font-size:{font_config['header_ref']}; color:#64748B; background:#F8FAFC; padding:10px; border-left:2px solid #0F766E; margin-top:5px; line-height:1.3;">
      <b>Citations:</b><br>
      1. Kulik D.A., et al. (2013): GEM-Selektor geochemical modeling package. Comput. Geosci. 17, 1-24.<br>
      2. Wagner T., et al. (2012): GEM-Selektor package: TSolMod library. Can. Mineral. 50, 1173-1195.<br>
      3. Miron G.D., et al. (2015): GEMSFITS: code package for optimization. Appl. Geochem. 55, 28-45.<br>
      4. C.-J. Haecker, E.J. Garboczi, J.W. Bullard, R.B. Bohn, Z. Sun, S.P. Shah, T. Voigt (2005): Modeling the linear elastic properties of Portland cement paste. Cem. Concr. Res. 35, 1948-1960.<br>
      5. Z. Sun, E.J. Garboczi, S.P. Shah (2007): Modeling the elastic properties of concrete composites: Experiment, differential effective medium theory, and numerical simulation. Cem. Concr. Compos. 29, 22-38.
    </div>
  </details>
  The ANN is then trained on these datasets.
</li>
  <li><b>Recipe mass balance & limits:</b> 100 g binder = <b>96 g (clinker+SCMs)</b> + <b>4 g gypsum</b>; clinker <b>20-96 g</b> => SCM <b>0-76 g</b>.</li>
  <li><b>Panel inputs & factors:</b> set ranges + curing time; CO2 factors default <b>ecoinvent</b> (editable); <b>cost is user-defined</b> (EUR/kg).</li>
  <li><b>Quick workflow:</b> set ranges & curing time -> Material Factors -> choose goals -> click <b>START OPTIMIZATION</b>.</li>
  <li><b>Results modules:</b>
  <ul style="margin:6px 0 0 18px; padding:0; line-height:1.45;">
    <li><b>Pareto Analysis:</b> interactive frontier plot showing explored points vs Pareto-optimal trade-offs; highlights the top-ranked mix (TOPSIS).</li>
    <li><b>GA Animation:</b> generation-by-generation visualization of NSGA-II population evolution toward the final frontier.</li>
    <li><b>Parallel Coordinates:</b> multi-dimensional view of compositions and metrics to inspect trade-offs and filter ranges.</li>
    <li><b>Data Table:</b> sortable Pareto solutions with export to CSV, including Decision Score and all metrics/materials.</li>
    <li><b>Benchmark Comparison:</b> compare any optimized mix against a user-defined OPC baseline (deltas + plots).</li>
    <li><b>Technical Export:</b> generate a PDF audit summarizing settings, objectives, Pareto results, and the selected recommendation.</li>
  </ul>
</li>
  <li><b>Need help?</b> click the <b>i</b> icons on panels for "how to read" and metric definitions.</li>
  <li><b>Disclaimer:</b> The recommended mixes are model-based screening results and must be validated experimentally before any practical or safety-critical use.</li>
</ul>
""".strip()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        :root {{
            --font-body: {font_config['body']};
            --font-small: {font_config['small']};
            --font-slider-label: {font_config['slider_label']};
            --font-tab: {font_config['tab']};
            --font-metric-label: {font_config['metric_label']};
            --font-metric-value: {font_config['metric_value']};
            --font-button: {font_config['button']};
            --font-h1: {font_config['h1']};
            --font-h2: {font_config['h2']};
            --font-h3: {font_config['h3']};
            --font-h4: {font_config['h4']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            background-color: {colors['bg_app']} !important;
            color: {colors['text_body']} !important;
        }}

        p, .stMarkdown, .stText, label, .stSelectbox, .stNumberInput, div[data-baseweb="select"] {{
            font-size: var(--font-body) !important;
            line-height: 1.5 !important;
        }}

        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaptionContainer"] p,
        div[data-testid="stRadio"] p,
        div[data-testid="stCheckbox"] label p {{
            font-size: var(--font-body) !important;
        }}

        small, .stCaption {{
            font-size: var(--font-small) !important;
        }}

        h1 {{ font-size: var(--font-h1) !important; font-weight: 800 !important; }}
        h2 {{ font-size: var(--font-h2) !important; font-weight: 700 !important; }}
        h3 {{ font-size: var(--font-h3) !important; font-weight: 700 !important; }}
        h4, h5 {{ font-size: var(--font-h4) !important; font-weight: 600 !important; }}

        .css-card {{
            background-color: {colors['bg_card']};
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid {colors['border']};
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}

        .stButton > button {{
            font-size: var(--font-button) !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            background: {colors['primary']} !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            transition: all 0.2s;
        }}
        .stButton > button:hover {{
            opacity: 0.9;
            transform: translateY(-1px);
        }}

        div.stSlider {{
            padding-top: 10px;
        }}
        div.stSlider div[data-testid="stMarkdownContainer"] p {{
            font-size: var(--font-slider-label) !important;
            color: {colors['text_sub']} !important;
            font-weight: 600 !important;
        }}
        div.stSlider div[data-baseweb="slider"] div[role="slider"] {{
            background-color: {colors['accent']} !important;
            border-color: {colors['accent']} !important;
        }}
        div.stSlider div[data-baseweb="slider"] div[data-testid="stTickBar"] + div {{
            background-color: {colors['primary']} !important;
        }}
        div.stSlider div[data-baseweb="slider"] > div > div > div {{
            background-color: {colors['primary']} !important;
        }}
        div.stSlider div[style*="background-color: rgb(255, 75, 75)"] {{
            background-color: {colors['primary']} !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            font-size: var(--font-tab) !important;
            padding: 10px 24px !important;
            color: {colors['text_sub']};
        }}
        .stTabs [aria-selected="true"] {{
            color: {colors['primary']} !important;
            border-bottom-color: {colors['primary']} !important;
        }}

        div[data-testid="stMetricLabel"] {{
            font-size: var(--font-metric-label) !important;
            color: {colors['text_sub']};
        }}
        div[data-testid="stMetricValue"] {{
            font-size: var(--font-metric-value) !important;
            font-weight: 700 !important;
            color: {colors['text_head']};
        }}

        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"] {{
            font-size: var(--font-body) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    img_base64 = get_base64_img(get_logo_path())
    logo_html = '<img src="data:image/png;base64,{}" style="height: 85px;">'.format(img_base64) if img_base64 else "Concrete"

    meta_html = (
        '<div style="margin-top:12px;padding-top:12px;border-top:1px solid #E2E8F0;'
        f'color:#334155;font-size:{font_config["header_meta"]};line-height:1.35;">'
        '<div style="margin-bottom:8px;"><b>Developed by:</b> '
        '<a href="https://www.psi.ch/en/les" target="_blank" '
        'style="color:#0F172A;font-weight:700;text-decoration:none;">PSI - LES Team</a> '
        '&nbsp;&nbsp;|&nbsp;&nbsp;<b>PI:</b> Nikolaos I. Prasianakis'
        "</div>"
        '<div style="margin-bottom:8px;"><b>Contact:</b> Bin Xi '
        '(<a href="mailto:bin.xi@psi.ch" style="color:#0F766E;text-decoration:none;">bin.xi@psi.ch</a>)'
        "</div>"
        '<div style="margin-bottom:6px;"><b>Reference:</b></div>'
        '<div style="margin-left:18px;margin-bottom:6px;">'
        "1. Xi B., Boiger R., Miron G.-D., Provis J.L., Churakov S.V., Prasianakis N.I. (under review): "
        "<i>A Physicochemical Simulation-Driven Machine Learning Framework for Optimization of Green Cement Recipes</i>."
        "</div>"
        '<div style="margin-left:18px;margin-bottom:8px;">'
        "2. Boiger R., Xi B., Miron G.D., et al. (2025): "
        "<i>Machine learning-accelerated discovery of green cement recipes</i>. "
        '<span style="font-weight:600;">Materials and Structures</span>, 58(5), 173.'
        "</div>"
        '<div><b>Acknowledgement:</b> Received funding from the ETH Board in the framework of the Joint Initiative '
        "<b>SCENE</b> (Swiss Center of Excellence on Net Zero emissions).</div>"
        "</div>"
    )

    header_html = (
        f'<div style="background-color:#fff;padding:25px;border-radius:16px;border:1px solid #E2E8F0;'
        f'box-shadow:0 20px 25px -5px rgba(0,0,0,0.05);margin-bottom:25px;">'
        f'<div style="display:flex;align-items:flex-start;gap:22px;">'
        f'<div style="flex:0 0 240px;display:flex;flex-direction:column;">'
        f'<div style="display:flex;align-items:center;">{logo_html}</div>'
        f"{meta_html}"
        f"</div>"
        f'<div style="flex:1;border-left:2px solid #E2E8F0;padding-left:20px;">'
        f'<h1 style="margin:0;font-size:{font_config["header_title"]};color:#0F172A;">'
        f'Cement Mix Optimizer <span style="font-size:{font_config["header_version"]}; font-weight:700; color:#64748B;">v0.1.2026</span>'
        f"</h1>"
        f'<p style="margin:6px 0 0 0;font-size:{font_config["header_subtitle"]};color:#64748B;font-weight:400;">'
        f'This platform integrates a <b>pretrained, physically consistent</b> machine learning model with '
        f'<b>multi-objective optimization</b> (NSGA-II) to accelerate cement mix design discovery. '
        f'It rapidly predicts key performance indicators and identifies <b>Pareto-optimal</b> trade-offs among '
        f"stiffness, cost, and embodied carbon."
        f"</p>"
        f"{note_html}"
        f"</div>"
        f"</div>"
        f"</div>"
    )

    components.html(header_html, height=font_config["header_height"], scrolling=True)


def init_session_state():
    if "df_pareto" not in st.session_state:
        st.session_state.df_pareto = None
    if "df_all" not in st.session_state:
        st.session_state.df_all = None
    if "time_input" not in st.session_state:
        st.session_state.time_input = 28

    st.session_state.setdefault("ga_pop", 100)
    st.session_state.setdefault("ga_gen", 20)
    st.session_state.setdefault("ga_seed", 1)
    st.session_state.setdefault("obj_e_max", True)
    st.session_state.setdefault("obj_co2_min", True)
    st.session_state.setdefault("obj_net_min", False)
    st.session_state.setdefault("obj_cost_min", False)
    st.session_state.setdefault("obj_co2abs_max", False)
    st.session_state.setdefault("font_size_preset", DEFAULT_FONT_PRESET)


def set_time(val):
    st.session_state.time_input = val
    st.session_state["time_days_pending"] = float(val)


def reset_run_results():
    keys_to_clear = [
        "df_pareto",
        "df_all",
        "pdf_data",
        "run_seed",
        "results_tab",
        "processed_df_display",
        "processed_df_all",
        "processed_objective_config",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None


@st.fragment
def slider_with_input(label, min_val, max_val, default, step, key_base):
    ss = st.session_state
    master_key = key_base
    slider_key = f"{key_base}_slider"
    box_key = f"{key_base}_box"
    pending_key = f"{key_base}_pending"

    if master_key not in ss:
        ss[master_key] = float(default)

    if pending_key in ss:
        ss[master_key] = float(ss[pending_key])
        ss[slider_key] = ss[master_key]
        ss[box_key] = ss[master_key]
        del ss[pending_key]

    if slider_key not in ss:
        ss[slider_key] = ss[master_key]
    if box_key not in ss:
        ss[box_key] = ss[master_key]

    def _on_slider_change():
        ss[pending_key] = float(ss[slider_key])

    def _on_box_change():
        ss[pending_key] = float(ss[box_key])

    col_slider, col_box = st.columns([3, 1], gap="small")
    with col_slider:
        st.slider(
            label=label,
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=slider_key,
            on_change=_on_slider_change,
        )

    with col_box:
        st.number_input(
            "Value",
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=box_key,
            on_change=_on_box_change,
            label_visibility="hidden",
            format="%.4f",
        )
    return float(ss[master_key])


@st.fragment
def range_slider_with_inputs(label, min_val, max_val, default_range, step, key_base):
    ss = st.session_state
    master_key = key_base
    slider_key = f"{key_base}_slider"
    box_lo_key = f"{key_base}_box_lo"
    box_hi_key = f"{key_base}_box_hi"
    pending_key = f"{key_base}_pending"

    if master_key not in ss:
        lo, hi = default_range
        ss[master_key] = (float(lo), float(hi))

    if pending_key in ss:
        lo, hi = ss[pending_key]
        ss[master_key] = (float(lo), float(hi))
        ss[slider_key] = ss[master_key]
        ss[box_lo_key] = float(lo)
        ss[box_hi_key] = float(hi)
        del ss[pending_key]
    else:
        lo, hi = ss[master_key]

    if slider_key not in ss:
        ss[slider_key] = (lo, hi)
    if box_lo_key not in ss:
        ss[box_lo_key] = lo
    if box_hi_key not in ss:
        ss[box_hi_key] = hi

    def _on_slider_change():
        lo, hi = ss[slider_key]
        ss[pending_key] = (float(lo), float(hi))

    def _on_box_lo_change():
        lo = float(ss[box_lo_key])
        hi = float(ss[box_hi_key])
        if lo > hi:
            lo = hi
            ss[box_lo_key] = lo
        ss[pending_key] = (lo, hi)

    def _on_box_hi_change():
        lo = float(ss[box_lo_key])
        hi = float(ss[box_hi_key])
        if hi < lo:
            hi = lo
            ss[box_hi_key] = hi
        ss[pending_key] = (lo, hi)

    col_slider, col_min, col_max = st.columns([4, 1.2, 1.2], gap="small")
    with col_slider:
        st.slider(
            label=label,
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=slider_key,
            on_change=_on_slider_change,
        )

    with col_min:
        st.number_input(
            "Min",
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=box_lo_key,
            on_change=_on_box_lo_change,
            label_visibility="collapsed",
            format="%.1f",
        )
    with col_max:
        st.number_input(
            "Max",
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=box_hi_key,
            on_change=_on_box_hi_change,
            label_visibility="collapsed",
            format="%.1f",
        )

    lo, hi = ss[master_key]
    return float(lo), float(hi)
