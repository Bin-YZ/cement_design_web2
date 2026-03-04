# main.py
import streamlit as st
import os
import warnings
import logging
import numpy as np
import pandas as pd
import sys
import traceback
import tempfile
from pathlib import Path
import base64

# ✅ 1. 导入我们刚刚拆分出去的仪表盘模块
from dashboard import show_results

# 1. 统一获取基础目录（脚本所在的文件夹）
BASE_DIR = Path(__file__).resolve().parent

# 2. 定义图片路径，逻辑与 model_path 完全一致
logo_path = BASE_DIR / "psi_logo.png"

def get_base64_img(path):
    # 检查图片是否存在，逻辑与 if not os.path.exists(model_path) 一致
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

# 3. 提取图片 Base64
img_base64 = get_base64_img(logo_path)

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)

# Pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Custom modules check
try:
    from model_wrapper import ModelWrapper
    from metrics import MetricsCalculator
    from nsga_problem import ConcreteMixProblem
except ImportError as e:
    st.error(f"Missing required modules: {e}. Please ensure model_wrapper.py, metrics.py, etc. are in the directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Cement Mix Optimizer v0.1.2026",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Color Palette ---
COLORS = {
    'bg_app':      '#F1F5F9',
    'bg_card':     '#FFFFFF',
    'text_head':   '#0F172A',
    'text_body':   '#334155',
    'text_sub':    '#64748B',
    'primary':     '#0F766E',
    'accent':      '#F97316',
    'border':      '#E2E8F0',
    'success':     '#10B981',
    'warning':     '#F59E0B',
    'chart_main':  '#0F766E',
    'chart_sec':   '#F97316',
}

# (Label, Key, CO2_default (kg/kg), Cost_default (€/kg))
MATERIALS_CONFIG = [
    ("C3S", "C3S", 0.82, 0.05),
    ("C2S", "C2S", 0.69, 0.04),
    ("C3A", "C3A", 0.73, 0.06),
    ("C4AF", "C4AF", 0.55, 0.035),
    ("Silica fume", "silica_fume", 0.0035, 0.8),
    ("GGBFS", "GGBFS", 0.13, 0.1),
    ("Fly ash", "fly_ash", 0.004, 0.02),
    ("Calcined clay", "calcined_clay", 0.27, 0.1),
    ("Limestone", "limestone", 0.0023, 0.03),
    ("Gypsum", "Gypsum", 0.0082, 0.05),
]

FIXED_WC = 0.5
FIXED_GYPSUM = 4.0
FIXED_TEMP = 25.0
TOTAL_BINDER_TARGET = 100.0 - FIXED_GYPSUM

note_html = """
<ul style="margin:10px 0 0 18px; padding:0; font-size:0.78rem; color:#334155; line-height:1.45;">
<li>
  <b>ANN Model outputs:</b> E-modulus (GPa) + max CO₂ uptake (kg/kg). The E-modulus is derived by applying 
  <b>micromechanical homogenization</b> to the <b>GEMS-simulated hydrate assemblages</b>.
  <details style="display:inline; cursor:pointer; color:#0F766E;">
    <summary style="list-style:none; display:inline; font-size:0.7rem; text-decoration:underline;">[Ref]</summary>
    <div style="font-size:0.7rem; color:#64748B; background:#F8FAFC; padding:10px; border-left:2px solid #0F766E; margin-top:5px; line-height:1.3;">
      <b>Citations:</b><br>
      1. Kulik D.A., et al. (2013): GEM-Selektor geochemical modeling package. Comput. Geosci. 17, 1-24.<br>
      2. Wagner T., et al. (2012): GEM-Selektor package: TSolMod library. Can. Mineral. 50, 1173-1195.<br>
      3. Miron G.D., et al. (2015): GEMSFITS: code package for optimization. Appl. Geochem. 55, 28-45.<br>
      4. C.-J. Haecker, E.J. Garboczi, J.W. Bullard, R.B. Bohn, Z. Sun, S.P. Shah, T. Voigt (2005): Modeling the linear elastic properties of Portland cement paste. Cem. Concr. Res. 35, 1948–1960.<br>
      5. Z. Sun, E.J. Garboczi, S.P. Shah (2007): Modeling the elastic properties of concrete composites: Experiment, differential effective medium theory, and numerical simulation. Cem. Concr. Compos. 29, 22–38.
    </div>
  </details>
  The ANN is then trained on these datasets.
</li>
  <li><b>Recipe mass balance & limits:</b> 100 g binder = <b>96 g (clinker+SCMs)</b> + <b>4 g gypsum</b>; clinker <b>20–96 g</b> ⇒ SCM <b>0–76 g</b>.</li>
  <li><b>Panel inputs & factors:</b> set ranges + curing time; CO₂ factors default <b>ecoinvent</b> (editable); <b>cost is user-defined</b> (€/kg).</li>
  <li><b>Quick workflow:</b> set ranges & curing time → Material Factors → choose goals → click <b>START OPTIMIZATION</b>.</li>
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
  <li><b>Need help?</b> click the <b>ℹ️</b> icons on panels for “how to read” and metric definitions.</li>
  <li><b>Disclaimer:</b> The recommended mixes are model-based screening results and must be validated experimentally before any practical or safety-critical use.</li>
</ul>
""".strip()

# --- CSS Styling ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {COLORS['bg_app']} !important;
        color: {COLORS['text_body']} !important;
    }}

    p, .stMarkdown, .stText, label, .stSelectbox, .stNumberInput, div[data-baseweb="select"] {{
        font-size: 16px !important;
        line-height: 1.5 !important;
    }}
    
    h1 {{ font-size: 2.2rem !important; font-weight: 800 !important; }}
    h2 {{ font-size: 1.8rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 1.4rem !important; font-weight: 700 !important; }}
    h4, h5 {{ font-size: 1.2rem !important; font-weight: 600 !important; }}

    .css-card {{
        background-color: {COLORS['bg_card']};
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }}

    .stButton > button {{
        font-size: 16px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        background: {COLORS['primary']} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}

    /* Slider Adjustments */
    div.stSlider {{ padding-top: 10px; }}
    div.stSlider div[data-testid="stMarkdownContainer"] p {{
        font-size: 15px !important;
        color: {COLORS['text_sub']} !important;
        font-weight: 600 !important;
    }}
    div.stSlider div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {COLORS['accent']} !important;
        border-color: {COLORS['accent']} !important;
    }}
    div.stSlider div[data-baseweb="slider"] div[data-testid="stTickBar"] + div {{
        background-color: {COLORS['primary']} !important;
    }}
    div.stSlider div[data-baseweb="slider"] > div > div > div {{
        background-color: {COLORS['primary']} !important; 
    }}
    div.stSlider div[style*="background-color: rgb(255, 75, 75)"] {{
        background-color: {COLORS['primary']} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        font-size: 16px !important;
        padding: 10px 24px !important;
        color: {COLORS['text_sub']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['primary']} !important;
        border-bottom-color: {COLORS['primary']} !important;
    }}

    div[data-testid="stMetricLabel"] {{ font-size: 14px !important; color: {COLORS['text_sub']}; }}
    div[data-testid="stMetricValue"] {{ font-size: 24px !important; font-weight: 700 !important; color: {COLORS['text_head']}; }}

    .main-header {{
        background: white;
        padding: 20px 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        display: flex; justify-content: space-between; align-items: center;
    }}
    </style>
""", unsafe_allow_html=True)


# --- Resource Caching ---
@st.cache_resource
def load_model_and_metrics():
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "my_model16082025.h5"
    if not os.path.exists(model_path):
        return None, None
    model = ModelWrapper(model_path)
    metrics = MetricsCalculator(FIXED_WC, FIXED_GYPSUM, FIXED_TEMP)
    return model, metrics

model, metrics = load_model_and_metrics()

if model is None:
    st.error("❌ Model file 'my_model16082025.h5' not found.")
    st.stop()

# --- Helper Functions ---
def set_time(val):
    st.session_state.time_input = val
    st.session_state["time_days_pending"] = float(val)

if 'df_pareto' not in st.session_state:
    st.session_state.df_pareto = None
if 'df_all' not in st.session_state:
    st.session_state.df_all = None
if 'time_input' not in st.session_state:
    st.session_state.time_input = 28

st.session_state.setdefault("ga_pop", 100)
st.session_state.setdefault("ga_gen", 20)
st.session_state.setdefault("ga_seed", 1)

st.session_state.setdefault("obj_e_max", True)        
st.session_state.setdefault("obj_co2_min", True)      
st.session_state.setdefault("obj_net_min", False)
st.session_state.setdefault("obj_cost_min", False)
st.session_state.setdefault("obj_co2abs_max", False)

def reset_run_results():
    keys_to_clear = [
        "df_pareto", "df_all", "pdf_data", "run_seed", "results_tab",
        "processed_df_display", "processed_df_all", "processed_objective_config" # 👈 新增缓存清理
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            st.session_state[k] = None

@st.fragment
def slider_with_input(label, min_val, max_val, default, step, key_base):
    ss = st.session_state
    master_key = key_base
    slider_key = f"{key_base}_slider"
    box_key    = f"{key_base}_box"
    pending_key = f"{key_base}_pending"

    if master_key not in ss:
        ss[master_key] = float(default)

    if pending_key in ss:
        ss[master_key] = float(ss[pending_key])
        ss[slider_key] = ss[master_key]
        ss[box_key]    = ss[master_key]
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
            format="%.4f" 
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
        if lo > hi: lo = hi; ss[box_lo_key] = lo
        ss[pending_key] = (lo, hi)
    def _on_box_hi_change():
        lo = float(ss[box_lo_key])
        hi = float(ss[box_hi_key])
        if hi < lo: hi = lo; ss[box_hi_key] = hi
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
        st.number_input("Min", min_value=float(min_val), max_value=float(max_val), step=float(step), key=box_lo_key, on_change=_on_box_lo_change, label_visibility="collapsed", format="%.1f")
    with col_max:
        st.number_input("Max", min_value=float(min_val), max_value=float(max_val), step=float(step), key=box_hi_key, on_change=_on_box_hi_change, label_visibility="collapsed", format="%.1f")
    
    lo, hi = ss[master_key]
    return float(lo), float(hi)

# ==========================================
#                PAGE LAYOUT
# ==========================================

import streamlit.components.v1 as components

logo_html = '<img src="data:image/png;base64,{}" style="height: 85px;">'.format(img_base64) if img_base64 else "🏗️"

meta_html = (
'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #E2E8F0;'
'color:#334155;font-size:0.82rem;line-height:1.35;">'
'<div style="margin-bottom:8px;">🏢 <b>Developed by:</b> '
'<a href="https://www.psi.ch/en/les" target="_blank" '
'style="color:#0F172A;font-weight:700;text-decoration:none;">PSI - LES Team</a> '
'&nbsp;&nbsp;|&nbsp;&nbsp;🧑‍🔬 <b>PI:</b> Nikolaos I. Prasianakis'
'</div>'
'<div style="margin-bottom:8px;">👤 <b>Contact:</b> Bin Xi '
'(<a href="mailto:bin.xi@psi.ch" style="color:#0F766E;text-decoration:none;">bin.xi@psi.ch</a>)'
'</div>'
'<div style="margin-bottom:6px;">📚 <b>Reference:</b></div>'
'<div style="margin-left:18px;margin-bottom:6px;">'
'1. Xi B., Boiger R., Miron G.-D., Provis J.L., Churakov S.V., Prasianakis N.I. (under review): '
'<i>A Physicochemical Simulation-Driven Machine Learning Framework for Optimization of Green Cement Recipes</i>.'
'</div>'
'<div style="margin-left:18px;margin-bottom:8px;">'
'2. Boiger R., Xi B., Miron G.D., et al. (2025): '
'<i>Machine learning-accelerated discovery of green cement recipes</i>. '
'<span style="font-weight:600;">Materials and Structures</span>, 58(5), 173.'
'</div>'
'<div><b>Acknowledgement:</b> Received funding from the ETH Board in the framework of the Joint Initiative '
'<b>SCENE</b> (Swiss Center of Excellence on Net Zero emissions).</div>'
'</div>'
)

header_html = (
f'<div style="background-color:#fff;padding:25px;border-radius:16px;border:1px solid #E2E8F0;'
f'box-shadow:0 20px 25px -5px rgba(0,0,0,0.05);margin-bottom:25px;">'
f'<div style="display:flex;align-items:flex-start;gap:22px;">'
f'<div style="flex:0 0 240px;display:flex;flex-direction:column;">'
f'<div style="display:flex;align-items:center;">{logo_html}</div>'
f'{meta_html}'
f'</div>'
f'<div style="flex:1;border-left:2px solid #E2E8F0;padding-left:20px;">'
f'<h1 style="margin:0;font-size:1.8rem;color:#0F172A;">'
f'Cement Mix Optimizer <span style="font-size:0.95rem; font-weight:700; color:#64748B;">v0.1.2026</span>'
f'</h1>'
f'<p style="margin:6px 0 0 0;font-size:0.9rem;color:#64748B;font-weight:400;">'
f'This platform integrates a <b>pretrained, physically consistent</b> machine learning model with '
f'<b>multi-objective optimization</b> (NSGA-II) to accelerate cement mix design discovery. '
f'It rapidly predicts key performance indicators and identifies <b>Pareto-optimal</b> trade-offs among '
f'stiffness, cost, and embodied carbon.'
f'</p>'
f'{note_html}'
f'</div>'
f'</div>'
f'</div>'
)

components.html(header_html, height=420, scrolling=True)

left_col, right_col = st.columns([0.38, 0.62], gap="large")

with left_col:
    with st.container():
        st.markdown(f"### 🎛️ Control Panel")
        
        with st.expander("📦 Clinker System Configuration", expanded=True):
            h1, h2 = st.columns([12, 1], gap="small")
            with h1:
                st.markdown(
                    f"<span style='color:{COLORS['text_head']}; font-weight:600; font-size:1.05rem;'>"
                    "Total Mass Constraint (g/100g)</span>",
                    unsafe_allow_html=True
                )
            with h2:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to read this control")
                    st.write("This sets the allowable range for **total clinker mass** in the binder.")
                    st.code("clinker_total (20–96 g) + ΣSCMs = 96 g   (binder=100 g incl. 4 g gypsum)")
                    st.write("So ΣSCMs = 96 − clinker_total (max ΣSCMs = 76 g when clinker = 20 g).")
            
            st.markdown(
                "<span style='font-size:0.8rem; color:#64748B;'>"
                "Allowed clinker total range: <b>20–96 g</b>. "
                "Binder is fixed at <b>100 g</b> including <b>4 g gypsum</b>."
                "</span>",
                unsafe_allow_html=True
            )
            
            cl_sum_rng = range_slider_with_inputs("", 20.0, 96.0, (20.0, 96.0), 0.5, "cl_sum_rng")
                        
            st.markdown("---")
            p1, p2 = st.columns([12, 1], gap="small")
            with p1:
                st.markdown(
                    f"<span style='color:{COLORS['text_head']}; font-weight:600; font-size:1.05rem;'>"
                    "Phase Composition (%)</span>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<span style='font-size:0.8rem; color:#64748B;'>"
                    "Clinker consists of four mineral phases. Each slider sets an allowable range for that phase. "
                    "Final phase fractions are normalized to sum to 100% within the chosen clinker total."
                    "</span>",
                    unsafe_allow_html=True
                )
            with p2:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown(
                        "**How to read these sliders**\n\n"
                        "- The ranges are **phase fraction bounds (%)** inside clinker (not grams).\n"
                        "- The optimizer first samples C3S/C2S/C3A/C4AF within these bounds.\n"
                        "- Then it **rescales/normalizes** the four phases so they sum to **100% of clinker**.\n"
                        "- Finally, these percentages are converted to grams using **clinker_total (g)**."
                    )
            
            c3s_rng = range_slider_with_inputs("C3S", 45.0, 80.0, (45.0, 80.0), 0.5, "c3s_rng")
            c2s_rng = range_slider_with_inputs("C2S", 10.0, 40.0, (10.0, 32.0), 0.5, "c2s_rng")
            c3a_rng = range_slider_with_inputs("C3A", 0.0, 15.0, (0.0, 14.0), 0.5, "c3a_rng")
            c4af_rng = range_slider_with_inputs("C4AF", 0.0, 20.0, (0.0, 15.0), 0.5, "c4af_rng")

        with st.expander("🌱 Supplementary Cementitious Materials (g)", expanded=True):
            top_l, top_r = st.columns([12, 1])
            with top_l:
                st.markdown(
                    "<span style='font-size:0.8rem; color:#64748B;'>"
                    "SCMs are defined by their absolute mass (g) and contribute to the total binder."
                    "</span>",
                    unsafe_allow_html=True
                )
            with top_r:
                with st.popover("ℹ️"):
                    st.markdown("### How to read this control")
                    st.write("Each slider defines the **allowed range (g)** for that SCM during sampling.")
                    st.write("Because binder mass is fixed, the optimizer enforces **mass balance**:")
                    st.code("ΣSCMs = 96 − clinker_total   (binder=100 g incl. 4 g gypsum)")
                    st.write("Implementation detail in this app:")
                    st.write("• The algorithm first samples SCM values within your min–max ranges.")
                    st.write("• Then it **rescales all SCMs proportionally** so their sum matches the required ΣSCMs.")
                    st.write("• A solution is kept only if every SCM still remains within its bounds after rescaling.")
                    st.write("")
                    st.write("Example:")
                    st.write("If clinker_total = 56 g → ΣSCMs must equal 40 g. The final SCM split is adjusted to sum to 40 g.")

            sf_rng = range_slider_with_inputs("Silica Fume (g)", 0.0, 10.0, (0.0, 10.0), 0.5, "sf_rng")
            gg_rng = range_slider_with_inputs("GGBFS (g)", 0.0, 80.0, (0.0, 80.0), 0.5, "gg_rng")
            fa_rng = range_slider_with_inputs("Fly Ash (g)", 0.0, 35.0, (0.0, 35.0), 0.5, "fa_rng")
            cc_rng = range_slider_with_inputs("Calcined Clay (g)", 0.0, 35.0, (0.0, 35.0), 0.5, "cc_rng")
            ls_rng = range_slider_with_inputs("Limestone (g)", 0.0, 35.0, (0.0, 35.0), 0.5, "ls_rng")

        with st.expander("⏱️ Curing & Constraints", expanded=True):
            t1, t2 = st.columns([12, 1], gap="small")
            with t1:
                st.markdown("<div style='font-weight:600; margin-bottom:8px;'>Curing Time Selection</div>", unsafe_allow_html=True)
            with t2:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to read this control")
                    st.write("Curing time (days) is a **direct input to the ML model**.")
                    st.write("Different ages lead to different hydration products → different predicted E and CO₂ uptake.")
                    st.write("• Quick buttons set common ages (7d/28d/90d/1y/10y).")
                    st.write("• The slider below sets the exact number of days (1–36500).")
            
            cols_time = st.columns(5)
            for idx, (label, val) in enumerate([("7d", 7), ("28d", 28), ("90d", 90), ("1y", 365), ("10y", 3650)]):
                if cols_time[idx].button(label, use_container_width=True):
                    set_time(val)
            
            time_val = slider_with_input("Custom Time (days)", 1, 36500, st.session_state.time_input, 1, "time_days")
            st.session_state.time_input = time_val

            st.markdown("---")
            f1, f2 = st.columns([12, 1], gap="small")
            with f1:
                st.markdown("<div style='font-weight:600; margin-bottom:6px;'>Fixed Parameters</div>", unsafe_allow_html=True)
            with f2:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to read this control")
                    st.write("These values are fixed to match the model’s training/assumptions and to keep optimization consistent.")
                    st.write("• The water-to-cement ratio (w/c) is fixed at 0.50 in accordance with EN 196-1.")
                    st.write("• Gypsum is fixed at 4 g (part of the 100 g binder definition).")
                    st.write("• Target binder = 96 g because 100 g includes 4 g gypsum (fixed).")

            st.markdown(f"""
            <div style='background:{COLORS['bg_app']}; padding:12px; border-radius:8px; border:1px solid {COLORS['border']}; font-size:0.9rem;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'><span>💧 w/c Ratio:</span><b>{FIXED_WC}</b></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'><span>⚪ Gypsum:</span><b>{FIXED_GYPSUM}g</b></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'><span>🌡️ Temp:</span><b>{FIXED_TEMP}°C</b></div>
                <div style='display:flex; justify-content:space-between;'><span>🎯 Target Binder:</span><b>{TOTAL_BINDER_TARGET}g</b></div>
                <div style='margin-top:8px; font-size:0.8rem; color:#64748B;'>
        These parameters are fixed throughout the optimization and are not subject to variation.</div>
            </div>
            """, unsafe_allow_html=True)
    

with right_col:
    with st.container():
        col_factors, col_objs = st.columns([1.2, 1], gap="medium")
        
        with col_factors:
             with st.expander("📊 Material Factors", expanded=False):
                tab_co2, tab_cost = st.tabs(["🏭 CO₂ Impact", "💰 Cost Factor"])
                
                with tab_co2:
                    c1, c2 = st.columns([12, 1], gap="small")
                    with c1:
                        st.caption("Adjust CO₂ emission factors (kg CO₂/kg material). Default values come from ecoinvent (editable).")
                    with c2:
                        with st.popover("ℹ️"):
                            st.markdown("### How to read CO₂ Impact")
                            st.write("These sliders define the **CO₂ emission factor** for each material (kg CO₂ per kg material).")
                            st.write("• **Defaults are from ecoinvent** (typical Swiss background LCA values).")
                            st.write("• You can **override** them to match your scenario (region, energy mix, supplier, transport, etc.).")
                            st.write("• The optimizer uses these factors to compute **CO₂ emission** and **net emission** objectives.")
                
                    co2_factors = {}
                    for (lbl, key, emis, cost) in MATERIALS_CONFIG:
                        co2_factors[key] = slider_with_input(lbl, 0.0, 1.5, float(emis), 0.0005, f"co2_{key}")
                
                with tab_cost:
                    c1, c2 = st.columns([12, 1], gap="small")
                    with c1:
                        st.caption("Set material cost factors (€/kg). Values should be defined by the user for the target scenario.")
                    with c2:
                        with st.popover("ℹ️"):
                            st.markdown("### How to read Cost Factor")
                            st.write("These sliders define the **unit cost** for each material (€/kg).")
                            st.write("• **No universal default**: costs vary by market, supplier, contracts, and time.")
                            st.write("• Please **enter your scenario-specific costs** to make the cost objective meaningful.")
                            st.write("• The optimizer uses these values to compute the **Cost** objective (€/kg binder).")
                
                    cost_factors = {}
                    for (lbl, key, emis, cost) in MATERIALS_CONFIG:
                        raw_val = slider_with_input(lbl, 0.0, 10.0, float(cost), 0.0005, f"cost_{key}")
                        cost_factors[key] = raw_val

        with col_objs:
            with st.expander("🎯 Optimization Goals", expanded=True):
                st.markdown(f"""
                <div style="background:white; border:1px solid {COLORS['border']};
                            border-radius:10px; padding:15px; height: 100%;">
                """, unsafe_allow_html=True)
        
                g1, g2 = st.columns([12, 1], gap="small")
                with g1:
                    st.caption("Select objectives (at least one). The optimizer searches for Pareto-optimal trade-offs.")
                with g2:
                    with st.popover("ℹ️"):
                        st.markdown("### How each objective is calculated")
        
                        st.markdown("**1) 📈 Max E-Modulus (E)**")
                        st.write("• **What it is:** stiffness of the hardened cement paste (Young’s modulus, **GPa**). Higher = stiffer/stronger.")
                        st.write("• **Where it comes from (training):** for each mix and curing age, **GEMS** predicts the amounts of hydration products (hydrate assemblage).")
                        st.write("• **How E label is obtained:** the hydrate assemblage is converted into an effective stiffness using a micromechanics homogenization step.")
                        st.write("• **What the ANN does:** it learns this GEMS→E mapping, so the web app can predict E instantly for new mixes.")
                        
                        st.markdown("**2) 🌱 Max CO₂ Uptake**")
                        st.write("• **What it is:** the **maximum theoretical** CO₂ that the hydrated paste could chemically bind (upper bound), expressed per **100 g cement** (clinker+SCMs).")
                        st.write("• **Where it comes from (training):** GEMS provides the **moles of each carbonation-reactive hydrate** at a given age.")
                        st.write("• **How uptake is calculated:** each hydrate has a stoichiometric CO₂ capacity (from its chemical formula). Total uptake is the sum:")
                        st.code("Total CO₂ (mol) = Σ_j [ n_j (mol hydrate) × ν_j (mol CO₂ per mol hydrate) ]")
                        st.write("• **Easy example (portlandite):**")
                        st.code("Ca(OH)₂ + CO₂ → CaCO₃ + H₂O   ⇒ 1 mol Ca(OH)₂ binds 1 mol CO₂")
                        st.write("  If GEMS predicts **0.10 mol Ca(OH)₂**, then uptake from portlandite is **0.10 mol CO₂ ≈ 4.4 g CO₂**.")
                        st.write("• Finally, the tool reports **g CO₂ per 100 g cement** (maximum end-of-life carbonation potential).")

        
                        st.markdown("**3) ♻️ Min CO₂ Emission (CO₂_emission)**")
                        st.write("• Computed from the mix using emission factors (default ecoinvent, editable):")
                        st.code("CO₂_emission = Σ_i ( mass_i × EF_i ) / total_binder_mass")
        
                        st.markdown("**4) 🌍 Min Net CO₂ (Net_emission)**")
                        st.write("• Net balance:")
                        st.code("Net_emission = CO₂_emission − CO₂_abs")
        
                        st.markdown("**5) 💵 Min Cost (Cost)**")
                        st.write("• Computed from the mix using user-defined prices (€/kg):")
                        st.code("Cost = Σ_i ( mass_i × Price_i ) / total_binder_mass")
        
                c_obj1, c_obj2 = st.columns(2)
                with c_obj1:
                    obj_e_max = st.checkbox("📈 Max E-Modulus", key="obj_e_max")
                    obj_net_min = st.checkbox("🌍 Min Net CO₂", key="obj_net_min")
                    obj_cost_min = st.checkbox("💵 Min Cost", key="obj_cost_min")
        
                with c_obj2:
                    obj_co2abs_max = st.checkbox("🌱 Max CO2 Uptake", key="obj_co2abs_max")
                    obj_co2_min    = st.checkbox("♻️ Min CO2 Emission", key="obj_co2_min")
        
                st.markdown("</div>", unsafe_allow_html=True)

            objectives_list = [k for k, v in [("E_max", obj_e_max), ("CO2abs_max", obj_co2abs_max), ("CO2_min", obj_co2_min), ("Cost_min", obj_cost_min), ("Net_min", obj_net_min)] if v]

    st.markdown(" ")
    
    with st.container():
        c_algo, c_btn = st.columns([2, 1], gap="medium")
    
        # =========================
        #  LEFT: Advanced Settings
        # =========================
        with c_algo:
            with st.expander("⚙️ Advanced Algorithm Settings", expanded=False):
    
                with st.form("ga_settings_form", clear_on_submit=False):
    
                    a1, a2 = st.columns([12, 1], gap="small")
                    with a1:
                        st.caption("Optional: tune NSGA-II search effort and reproducibility.")
                    with a2:
                        with st.popover("ℹ️"):
                            st.markdown("### How to read these settings")
                            st.write("These parameters control the NSGA-II genetic algorithm search.")
                            st.write("• **Population** = number of candidate mixes evaluated per generation.")
                            st.write("  ↑ larger → better exploration, but slower.")
                            st.write("• **Generations** = how many evolution steps are performed.")
                            st.write("  ↑ larger → better convergence, but slower.")
                            st.write("• **Random Seed** = controls randomness for reproducible runs.")
                            st.write("  Same seed + same settings → very similar results.")
                            st.write("")
                            st.write("Rule of thumb:")
                            st.write("• Quick test: 50–100 pop, 10–20 gen")
                            st.write("• Better quality: 150–300 pop, 50+ gen")
    
                    ac1, ac2, ac3 = st.columns(3)
                    ga_pop_tmp = ac1.number_input("Population", 10, 500, int(st.session_state.get("ga_pop", 100)), step=10, key="ga_pop_tmp")
                    ga_gen_tmp = ac2.number_input("Generations", 10, 500, int(st.session_state.get("ga_gen", 20)), step=10, key="ga_gen_tmp")
                    ga_seed_tmp = ac3.number_input("Random Seed", 0, 1000, int(st.session_state.get("ga_seed", 1)), key="ga_seed_tmp")
    
                    b1, b2 = st.columns([1, 1], gap="small")
                    apply_ga = b1.form_submit_button("✅ Apply Settings", use_container_width=True)
                    reset_ga = b2.form_submit_button("↩ Reset Defaults", use_container_width=True)
    
                if apply_ga:
                    st.session_state["ga_pop"] = int(ga_pop_tmp)
                    st.session_state["ga_gen"] = int(ga_gen_tmp)
                    st.session_state["ga_seed"] = int(ga_seed_tmp)
                    st.toast("GA settings applied.", icon="✅")
    
                if reset_ga:
                    st.session_state["ga_pop"] = 100
                    st.session_state["ga_gen"] = 20
                    st.session_state["ga_seed"] = 1
                    st.session_state["ga_pop_tmp"] = 100
                    st.session_state["ga_gen_tmp"] = 20
                    st.session_state["ga_seed_tmp"] = 1
                    st.toast("GA settings reset.", icon="↩")
    
        ga_pop  = int(st.session_state.get("ga_pop", 100))
        ga_gen  = int(st.session_state.get("ga_gen", 20))
        ga_seed = int(st.session_state.get("ga_seed", 1))
    
        # =========================
        #  RIGHT: Button
        # =========================
        with c_btn:
            st.markdown("<div style='height: 5px'></div>", unsafe_allow_html=True)
            run_btn = st.button("🚀 START OPTIMIZATION", type="primary", use_container_width=True)
            clear_placeholder = st.empty()

    # ==========================================
    # --- OPTIMIZATION LOGIC ---
    # ==========================================
    if run_btn:
        reset_run_results() 
    
        if not objectives_list:
            st.error("⚠️ Select at least one objective.")
        else:
            with st.status("🚀 initializing Optimization Engine...", expanded=True) as status:
                try:
                    status.write("• Configuring physical boundary conditions...")
                    scm_bounds_list = [sf_rng, gg_rng, fa_rng, cc_rng, ls_rng]
                    all_scm_zero = all((hi <= 0.0 and lo <= 0.0) for (lo, hi) in scm_bounds_list)
                    
                    if all_scm_zero:
                        cl_sum_rng = (TOTAL_BINDER_TARGET, TOTAL_BINDER_TARGET)  # 96–96
                        st.info("SCMs are all fixed to 0 g → enforcing clinker_total = 96 g to satisfy mass balance.")
                    clinker_bounds = {"C3S": c3s_rng, "C2S": c2s_rng, "C3A": c3a_rng, "C4AF": c4af_rng}
                    scms_bounds = {
                        "silica_fume": sf_rng, "GGBFS": gg_rng, "fly_ash": fa_rng,
                        "calcined_clay": cc_rng, "limestone": ls_rng,
                    }
                    
                    problem = ConcreteMixProblem(
                        model, metrics, clinker_bounds, scms_bounds, cl_sum_rng, TOTAL_BINDER_TARGET,
                        int(time_val), co2_factors, cost_factors, objectives_list,
                    )
                    
                    status.write("• Running NSGA-II Genetic Algorithm...")
                    algorithm = NSGA2(
                        pop_size=ga_pop, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15.0),
                        mutation=PM(prob=0.1, eta=20.0), eliminate_duplicates=True,
                    )
                    termination = get_termination("n_gen", ga_gen)
                    
                    res = minimize(problem, algorithm, termination, seed=ga_seed, verbose=False, save_history=True)
                    
                    status.write("• Decoding Pareto front & calculating metrics...")
                    X = res.X
                    if X is None or len(np.atleast_1d(X)) == 0:
                        status.update(label="❌ Optimization Failed: no feasible solutions", state="error")
                        st.error("No feasible solutions found under current bounds. "
                                 "If all SCMs are fixed to 0 g, mass balance requires clinker_total = 96 g.")
                        st.stop()
                    
                    mixes = problem.decode(X)
                    preds = model.predict(mixes)
                    df_pareto = metrics.add_metrics(mixes, preds[:, 0], preds[:, 1], co2_factors, cost_factors)
                    
                    # --- History (per generation) ---
                    X_hist_list = []
                    gen_hist_list = []
                    
                    for g, algo_g in enumerate(res.history):
                        Xg = algo_g.pop.get("X")
                        if Xg is None or len(Xg) == 0:
                            continue
                        X_hist_list.append(Xg)
                        gen_hist_list.append(np.full((Xg.shape[0],), g, dtype=int))
                    
                    X_hist = np.concatenate(X_hist_list, axis=0)
                    gen_hist = np.concatenate(gen_hist_list, axis=0)
                    
                    mixes_hist = problem.decode(X_hist)
                    preds_hist = model.predict(mixes_hist)
                    
                    df_all = metrics.add_metrics(
                        mixes_hist,
                        preds_hist[:, 0],
                        preds_hist[:, 1],
                        co2_factors,
                        cost_factors
                    )
                    df_all["Generation"] = gen_hist  
                    
                    st.session_state.df_pareto = df_pareto.reset_index(drop=True)
                    st.session_state.df_all = df_all.reset_index(drop=True)
                    st.session_state.run_seed = ga_seed
                    
                    status.update(label="✅ Optimization Successfully Completed!", state="complete", expanded=False)
                    st.balloons()
                    st.toast(f"🎉 Success! Found {len(df_pareto)} optimal mixes!", icon="🏗️")
                except Exception as e:
                    status.update(label="❌ Optimization Failed", state="error")
                    st.error(f"Error: {str(e)}")
                    st.code(traceback.format_exc())

    # ==========================================
    # --- FILL THE CLEAR BUTTON PLACEHOLDER ---
    # ==========================================
    has_results = (
        st.session_state.get("df_pareto") is not None
        or st.session_state.get("df_all") is not None
        or st.session_state.get("pdf_data") is not None
    )

    if has_results:
        with clear_placeholder.container():
            ccl1, ccl2 = st.columns([1, 6], gap="small")
            with ccl1:
                clear_btn = st.button("🧹", help="Clear results", use_container_width=True)
            with ccl2:
                st.caption("Clear results")

            if clear_btn:
                reset_run_results()
                st.toast("Cleared.", icon="🧹")
                st.rerun()
    else:
        st.markdown(f"""
        <div style='text-align: center; padding: 60px; color: {COLORS['text_sub']}; border: 2px dashed {COLORS['border']}; border-radius: 12px; margin-top: 40px;'>
            <h3>👋 Ready to Optimize</h3>
            <p>Configure your material constraints (in grams) on the left panel, then click <b>START OPTIMIZATION</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    # ✅ 3. 调用拆分出去的绘图模块
    show_results(COLORS=COLORS, MATERIALS_CONFIG=MATERIALS_CONFIG)