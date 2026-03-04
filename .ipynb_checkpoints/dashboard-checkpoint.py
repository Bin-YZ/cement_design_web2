
import streamlit as st
import os
import warnings
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import traceback
import tempfile
from pathlib import Path
import base64
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
    from pdf_generator import create_pdf_report
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
# 建议修改后的 note_html
# 将这段代码替换你原来的 note_html 变量的开头部分：
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

# --- Objective flags (GLOBAL, SAFE) ---
# --- Objective flags (DEFAULT SELECTION) ---
st.session_state.setdefault("obj_e_max", True)        # ✅ 默认选中
st.session_state.setdefault("obj_co2_min", True)     # ✅ 默认选中

st.session_state.setdefault("obj_net_min", False)
st.session_state.setdefault("obj_cost_min", False)
st.session_state.setdefault("obj_co2abs_max", False)
def reset_run_results():
    # 清掉上一次运行产生的大对象，避免每次 rerun 都背着它们
    keys_to_clear = [
        "df_pareto", "df_all",
        "pdf_data",
        "run_seed",
        # 如果你后面还有缓存的 figure/frame/selected index，也一起清
        "results_tab",
        # 如果你存过更细的中间变量，也可加在这里
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            st.session_state[k] = None

    # 可选：强制释放一点点内存引用（通常不需要）
    # import gc; gc.collect()
@st.fragment
# Custom Slider Components
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
        # 这里不需要再传 value=...，因为 key 已经在上面被初始化进 session_state 了
        st.number_input(
            "Value", 
            min_value=float(min_val), 
            max_value=float(max_val), 
            step=float(step), 
            key=box_key, 
            on_change=_on_box_change, 
            label_visibility="hidden",
            # 建议加上 format，防止浮点数显示过长 (例如 0.141000002)
            format="%.4f" 
        )
    return float(ss[master_key])
@st.fragment
def calculate_topsis(df, objective_config):
    """
    df: Pareto DataFrame
    objective_config: List of dicts e.g. [{'col': 'E', 'impact': '+'}, {'col': 'Cost', 'impact': '-'}]
    """
    # Extract criteria columns and impacts from config
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
def range_slider_with_inputs(label, min_val, max_val, default_range, step, key_base):
    ss = st.session_state
    master_key = key_base
    slider_key = f"{key_base}_slider"
    box_lo_key = f"{key_base}_box_lo"
    box_hi_key = f"{key_base}_box_hi"
    pending_key = f"{key_base}_pending"

    # 1. 初始化 Master
    if master_key not in ss:
        lo, hi = default_range
        ss[master_key] = (float(lo), float(hi))

    # 2. 同步逻辑
    if pending_key in ss:
        lo, hi = ss[pending_key]
        ss[master_key] = (float(lo), float(hi))
        ss[slider_key] = ss[master_key]
        ss[box_lo_key] = float(lo)
        ss[box_hi_key] = float(hi)
        del ss[pending_key]
    else:
        # 如果没有挂起的更改，读取当前 master 值
        lo, hi = ss[master_key]

    # --- 🛠️ 修复点: 强制初始化所有 Key ---
    if slider_key not in ss:
        ss[slider_key] = (lo, hi)
    if box_lo_key not in ss:
        ss[box_lo_key] = lo
    if box_hi_key not in ss:
        ss[box_hi_key] = hi
    # -----------------------------------

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

# 1. 准备 Logo
logo_html = '<img src="data:image/png;base64,{}" style="height: 85px;">'.format(img_base64) if img_base64 else "🏗️"

# 2. 插入 Header 与 新人引导说明 (移除所有特殊空白符)

import streamlit.components.v1 as components

meta_html = (
'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #E2E8F0;'
'color:#334155;font-size:0.82rem;line-height:1.35;">'

# Developed by (single line) + PI right after
'<div style="margin-bottom:8px;">🏢 <b>Developed by:</b> '
'<a href="https://www.psi.ch/en/les" target="_blank" '
'style="color:#0F172A;font-weight:700;text-decoration:none;">PSI - LES Team</a> '
'&nbsp;&nbsp;|&nbsp;&nbsp;🧑‍🔬 <b>PI:</b> Nikolaos I. Prasianakis'
'</div>'

# Contact (single line)
'<div style="margin-bottom:8px;">👤 <b>Contact:</b> Bin Xi '
'(<a href="mailto:bin.xi@psi.ch" style="color:#0F766E;text-decoration:none;">bin.xi@psi.ch</a>)'
'</div>'

# Reference (numbered format, like your GEM citations)
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

# Acknowledgement (single line label + text)
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

# ✅ 注意：不要再用 st.markdown(header_html, unsafe_allow_html=True)
components.html(header_html, height=420, scrolling=True)



# 3. 后续布局
left_col, right_col = st.columns([0.38, 0.62], gap="large")

# --- LEFT COLUMN (Control Panel) ---
with left_col:
    with st.container():
        st.markdown(f"### 🎛️ Control Panel")
        # st.markdown("---")
        
        # --- Clinker System Card ---
        with st.expander("📦 Clinker System Configuration", expanded=True):
             # --- Total Mass Constraint (header + info icon) ---
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
            
            # ✅ 注释：独立放一行，确保一定显示
            st.markdown(
                "<span style='font-size:0.8rem; color:#64748B;'>"
                "Allowed clinker total range: <b>20–96 g</b>. "
                "Binder is fixed at <b>100 g</b> including <b>4 g gypsum</b>."
                "</span>",
                unsafe_allow_html=True
            )
            
            # slider
            cl_sum_rng = range_slider_with_inputs("", 20.0, 96.0, (20.0, 96.0), 0.5, "cl_sum_rng")

                        
            st.markdown("---")
            # --- Phase Composition (%) header + help ---
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
            
            # --- Your existing phase sliders ---
            c3s_rng = range_slider_with_inputs("C3S", 45.0, 80.0, (45.0, 80.0), 0.5, "c3s_rng")
            c2s_rng = range_slider_with_inputs("C2S", 10.0, 40.0, (10.0, 32.0), 0.5, "c2s_rng")
            c3a_rng = range_slider_with_inputs("C3A", 0.0, 15.0, (0.0, 14.0), 0.5, "c3a_rng")
            c4af_rng = range_slider_with_inputs("C4AF", 0.0, 20.0, (0.0, 15.0), 0.5, "c4af_rng")


        # --- SCMs Card ---
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



        # --- Time & Fixed ---
        with st.expander("⏱️ Curing & Constraints", expanded=True):
            # --- Curing Time header + blue warning popover ---
            t1, t2 = st.columns([12, 1], gap="small")
            
            with t1:
                st.markdown(
                    "<div style='font-weight:600; margin-bottom:8px;'>Curing Time Selection</div>",
                    unsafe_allow_html=True
                )
            
            with t2:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to read this control")
                    st.write("Curing time (days) is a **direct input to the ML model**.")
                    st.write("Different ages lead to different hydration products → different predicted E and CO₂ uptake.")
                    st.write("• Quick buttons set common ages (7d/28d/90d/1y/10y).")
                    st.write("• The slider below sets the exact number of days (1–36500).")
            
            # --- Quick buttons ---
            cols_time = st.columns(5)
            for idx, (label, val) in enumerate([("7d", 7), ("28d", 28), ("90d", 90), ("1y", 365), ("10y", 3650)]):
                if cols_time[idx].button(label, use_container_width=True):
                    set_time(val)
            
            # --- Custom slider ---
            time_val = slider_with_input("Custom Time (days)", 1, 36500, st.session_state.time_input, 1, "time_days")
            st.session_state.time_input = time_val

            st.markdown("---")
            # --- Fixed params header + help ---
            f1, f2 = st.columns([12, 1], gap="small")
            
            with f1:
                st.markdown(
                    "<div style='font-weight:600; margin-bottom:6px;'>Fixed Parameters</div>",
                    unsafe_allow_html=True
                )
            
            with f2:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                with st.popover("ℹ️"):
                    st.markdown("### How to read this control")
                    st.write("These values are fixed to match the model’s training/assumptions and to keep optimization consistent.")
                    st.write("• The water-to-cement ratio (w/c) is fixed at 0.50 in accordance with EN 196-1 (standard mortar for strength testing: cement:water = 1:0.5 by mass, i.e., 450 g cement and 225 g water).")
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
    


# --- RIGHT COLUMN (Objectives & Results) ---
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
        
                # ✅ 就放在这里：在 checkbox 之前
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
        
                # ✅ 原来的 checkbox 区域保持不变
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
        #  LEFT: Advanced Settings (NO auto apply)
        # =========================
        with c_algo:
            with st.expander("⚙️ Advanced Algorithm Settings", expanded=False):
    
                # ✅ 用 form：只有点 Apply 才更新 session_state（不会“改一下就影响全局”）
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
    
                    # 这里用 tmp key，避免你改的时候立刻影响全局
                    ac1, ac2, ac3 = st.columns(3)
                    ga_pop_tmp = ac1.number_input(
                        "Population", 10, 500,
                        int(st.session_state.get("ga_pop", 100)),
                        step=10, key="ga_pop_tmp"
                    )
                    ga_gen_tmp = ac2.number_input(
                        "Generations", 10, 500,
                        int(st.session_state.get("ga_gen", 20)),
                        step=10, key="ga_gen_tmp"
                    )
                    ga_seed_tmp = ac3.number_input(
                        "Random Seed", 0, 1000,
                        int(st.session_state.get("ga_seed", 1)),
                        key="ga_seed_tmp"
                    )
    
                    b1, b2 = st.columns([1, 1], gap="small")
                    apply_ga = b1.form_submit_button("✅ Apply Settings", use_container_width=True)
                    reset_ga = b2.form_submit_button("↩ Reset Defaults", use_container_width=True)
    
                # form 外：点按钮才真正写入全局 ga_*
                if apply_ga:
                    st.session_state["ga_pop"] = int(ga_pop_tmp)
                    st.session_state["ga_gen"] = int(ga_gen_tmp)
                    st.session_state["ga_seed"] = int(ga_seed_tmp)
                    st.toast("GA settings applied.", icon="✅")
    
                if reset_ga:
                    st.session_state["ga_pop"] = 100
                    st.session_state["ga_gen"] = 20
                    st.session_state["ga_seed"] = 1
                    # 同步 tmp（让输入框也回到默认）
                    st.session_state["ga_pop_tmp"] = 100
                    st.session_state["ga_gen_tmp"] = 20
                    st.session_state["ga_seed_tmp"] = 1
                    st.toast("GA settings reset.", icon="↩")
    
        # ✅ ✅ ✅ 关键：在 Start 按钮之前读取“已应用”的值
        ga_pop  = int(st.session_state.get("ga_pop", 100))
        ga_gen  = int(st.session_state.get("ga_gen", 20))
        ga_seed = int(st.session_state.get("ga_seed", 1))
    
        # =========================
        #  RIGHT: Butto
        # =========================
        with c_btn:
            st.markdown("<div style='height: 5px'></div>", unsafe_allow_html=True)
    
            # 主按钮
            run_btn = st.button("🚀 START OPTIMIZATION", type="primary", use_container_width=True)
            
            # ✅ 在这里创建一个占位符（Placeholder），专门留给扫把图标
            clear_placeholder = st.empty()

    # ==========================================
    # --- OPTIMIZATION LOGIC ---
    # ==========================================
    if run_btn:
        reset_run_results()   # ✅ 新增：先清空上次结果
    
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
                    df_all["Generation"] = gen_hist  # ✅ 用于动画帧
                    
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
    # ✅ 等所有优化逻辑走完，统一判断是否有结果
    has_results = (
        st.session_state.get("df_pareto") is not None
        or st.session_state.get("df_all") is not None
        or st.session_state.get("pdf_data") is not None
    )

    if has_results:
        # ✅ 使用上面留在 c_btn 里的空位来渲染扫把图标
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


    # --- RESULTS DASHBOARD ---
# --- RESULTS DASHBOARD ---
# --- RESULTS DASHBOARD ---
    @st.fragment
    def show_results():
        if st.session_state.df_pareto is not None:
            # 1. 准备显示数据 (转换为工程标准单位)
            df_display = st.session_state.df_pareto.copy()
            df_all_display = st.session_state.df_all.copy() if st.session_state.df_all is not None else None
            # ---------- Decision Score (TOPSIS) ----------
            # 用当前勾选目标生成 objective_config（确保与报告一致）
            objective_config = []
            if obj_e_max:
                objective_config.append({'col': 'E', 'impact': '+', 'name': 'Max Strength'})
            if obj_cost_min:
                objective_config.append({'col': 'Cost', 'impact': '-', 'name': 'Min Cost'})
            if obj_co2_min:
                objective_config.append({'col': 'CO2_emission', 'impact': '-', 'name': 'Min CO2'})
            if obj_net_min:
                objective_config.append({'col': 'Net_emission', 'impact': '-', 'name': 'Min Net CO2'})
            if obj_co2abs_max:
                objective_config.append({'col': 'CO2_abs', 'impact': '+', 'name': 'Max Uptake'})
            
            # 兜底：如果用户什么都没选（理论上前面已拦截），至少用 E
            if not objective_config:
                objective_config = [{'col': 'E', 'impact': '+', 'name': 'Max Strength'}]
            
            # 计算 TOPSIS 分数（0~1，越大越好）
            df_display["Decision_Score"] = calculate_topsis(df_display, objective_config)
            
            # 建议：Data Table / 后续分析默认按决策分数排序
            df_display = df_display.sort_values("Decision_Score", ascending=False).reset_index(drop=True)
    # --------------------------------------------
    
            co2_cols = ["CO2_emission", "Net_emission", "CO2_abs"]
            for col in co2_cols:
                if col in df_display.columns:
                    df_display[col] = df_display[col]
                if df_all_display is not None and col in df_all_display.columns:
                    df_all_display[col] = df_all_display[col] 
    
    
            
            with st.expander("📊 Analytics Dashboard & Results", expanded=True):
                
                tab = st.radio(
                    "View Mode",
                    ["📉 Pareto Analysis", "🎞️ GA Animation", "🕸️ Parallel Coordinates", "📋 Data Table", "⚖️ Benchmark Comparison"],
                    key="results_tab",
                    horizontal=True,
                    label_visibility="collapsed",
                )
            
                st.markdown("---")
    
                # 轴标签更新为 kg/kg
                axis_options = {
                    "E": "E (GPa) - MAX",
                    "CO2_abs": "CO₂ Uptake (kg/kg) - MAX",
                    "CO2_emission": "CO₂ Emission (kg/kg) - MIN", 
                    "Cost": "Cost (€/kg) - MIN",
                    "Net_emission": "Net Emission (kg/kg) - MIN",
                }
    
                if tab == "📉 Pareto Analysis":
                            with st.container():
                                t1, t2 = st.columns([12, 1], gap="small")
                                with t1:
                                    st.markdown("### 📈 Interactive Pareto Visualization")
                                with t2:
                                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                                    with st.popover("ℹ️"):
                                        st.markdown("### How to read this plot")
                                        st.write("**What you see**")
                                        st.write("• **Grey points (Explored):** all candidate mixes evaluated during NSGA-II search.")
                                        st.write("• **Colored points (Pareto):** final **Pareto-optimal** solutions (best trade-offs).")
                                        st.write("• **⭐ Top-1:** the recommended solution based on **Decision Score (TOPSIS)**.")
                            
                                        st.write("**How it is generated**")
                                        st.write("1) NSGA-II samples many mixes within your bounds.")
                                        st.write("2) The ML model predicts E and CO₂ uptake; metrics compute CO₂ emission/cost/net.")
                                        st.write("3) Non-dominated solutions form the Pareto front; TOPSIS ranks them if enabled.")
                            
                                        st.write("**How to interpret axes**")
                                        st.write("• Use **X Axis / Y Axis** dropdowns to choose what to compare.")
                                        st.write("• Labels show **MAX** or **MIN** to indicate whether higher or lower is better.")
                            
                                        st.write("**What color means**")
                                        st.write("• **Color Points By** selects which variable controls the color scale.")
                                        st.write("• If set to **Decision_Score**, warmer colors = higher TOPSIS score (more recommended).")
                                        st.write("• If set to a metric/material, color reflects that value across Pareto solutions.")
                            
                                        st.write("**Decision Score (TOPSIS) — how it works**")
                                        st.write("• TOPSIS ranks the Pareto solutions by closeness to an **ideal best** trade-off.")
                                        st.write("• MAX objectives → highest value, MIN objectives → lowest value (equal weights).")
                                        st.write("• Uses distances to ideal best (S+) and worst (S−):")
                                        st.code("Decision_Score = S− / (S+ + S−)    (0–1; higher = better)")
                                        st.write("• **Top-1** is the highest score among the Pareto set.")
                            
                                        st.write("**Toggles**")
                                        st.write("• **Show History:** show/hide explored points (grey).")
                                        st.write("• **3D View:** add a third axis (Z) to visualize 3 metrics at once.")
    
    
    
                            chart_options = [k for k in axis_options.keys() if k in df_display.columns]
                            
                            c_ctrl = st.columns(5)
                            default_x = chart_options[0] if len(chart_options) >= 1 else chart_options[0]
                            default_y = chart_options[1] if len(chart_options) >= 2 else chart_options[0]
                            default_z = chart_options[2] if len(chart_options) >= 3 else chart_options[0]
                            
                            x_axis = c_ctrl[0].selectbox("X Axis", chart_options, index=chart_options.index(default_x) if default_x in chart_options else 0, format_func=lambda x: axis_options[x])
                            y_axis = c_ctrl[1].selectbox("Y Axis", chart_options, index=chart_options.index(default_y) if default_y in chart_options else 0, format_func=lambda x: axis_options[x])
                            material_cols = [m[1] for m in MATERIALS_CONFIG]
                            
                            color_candidates = ["Decision_Score"] + chart_options + material_cols
                            color_options = [c for c in color_candidates if c in df_display.columns]
                            default_color_idx = color_options.index("Decision_Score") if "Decision_Score" in color_options else 0
                            
                            color_col = c_ctrl[2].selectbox("Color Points By", color_options, index=default_color_idx)
                            show_all = c_ctrl[3].toggle("Show History", value=True)
                            
                            use_3d_allowed = len(chart_options) >= 3
                            use_3d = c_ctrl[4].toggle("3D View", value=False) if use_3d_allowed else False
                            
                            if use_3d and use_3d_allowed:
                                z_axis = st.selectbox("Z Axis", chart_options, index=chart_options.index(default_z), format_func=lambda x: axis_options[x])
                            else:
                                z_axis = None
                
                            # 🛠️ 调整 margin，右侧(r)留大一点给 Colorbar Title
                            layout_settings = dict(
                                height=600, 
                                width=700, 
                                margin=dict(l=20, r=80, t=30, b=20), 
                                paper_bgcolor="rgba(0,0,0,0)", 
                                plot_bgcolor="rgba(255,255,255,0.5)", 
                                font=dict(family="Inter", color=COLORS["text_body"])
                            )
                            fig = go.Figure()
                
                            # 1. Explored points
                            if show_all and df_all_display is not None:
                                if use_3d and z_axis:
                                    fig.add_trace(go.Scatter3d(
                                        x=df_all_display[x_axis], y=df_all_display[y_axis], z=df_all_display[z_axis], 
                                        mode="markers", name="Explored", 
                                        marker=dict(size=2, color="lightgrey", opacity=0.2)
                                    ))
                                else:
                                    fig.add_trace(go.Scatter(
                                        x=df_all_display[x_axis], y=df_all_display[y_axis], 
                                        mode="markers", name="Explored", 
                                        marker=dict(size=5, color="lightgrey", opacity=0.25)
                                    ))
                            
                            # 2. Pareto points
                            colorbar_title = "Decision Score<br>(TOPSIS)" if color_col == "Decision_Score" else color_col
                            
                            hover_text = [
                                f"Rank #{i+1}<br>Score={row['Decision_Score']:.3f}" if "Decision_Score" in df_display.columns else f"Mix #{i+1}"
                                for i, row in df_display.iterrows()
                            ]
                
                            if use_3d and z_axis:
                                fig.add_trace(go.Scatter3d(
                                    x=df_display[x_axis], y=df_display[y_axis], z=df_display[z_axis],
                                    mode="markers", name="Pareto",
                                    marker=dict(
                                        size=6, color=df_display[color_col], colorscale="Viridis", opacity=0.95, showscale=True,
                                        colorbar=dict(title=dict(text=colorbar_title, side="right"))
                                    ),
                                    text=hover_text, showlegend=True
                                ))
                                fig.update_layout(scene=dict(xaxis_title=axis_options[x_axis], yaxis_title=axis_options[y_axis], zaxis_title=axis_options[z_axis]))
                            
                            else:
                                # 2D Plot
                                fig.add_trace(go.Scatter(
                                    x=df_display[x_axis], y=df_display[y_axis],
                                    mode="markers", name="Pareto",
                                    marker=dict(
                                        size=12, color=df_display[color_col], colorscale="Viridis", opacity=0.95, showscale=True,
                                        # 🛠️ Colorbar 调整
                                        colorbar=dict(
                                            title=dict(text=colorbar_title, side="right"),
                                            thickness=15,
                                            xpad=10 # 增加一点与图表的间距
                                        )
                                    ),
                                    text=hover_text, showlegend=True
                                ))
                            
                            # 3. Top-1 Highlight
                            if "Decision_Score" in df_display.columns and len(df_display) > 0:
                                best_row = df_display.iloc[0]
                                if use_3d and z_axis:
                                     fig.add_trace(go.Scatter3d(
                                        x=[best_row[x_axis]], y=[best_row[y_axis]], z=[best_row[z_axis]],
                                        mode="markers+text", name="Top-1",
                                        marker=dict(size=10, symbol="diamond", color="red", line=dict(color="black", width=2)),
                                        text=["Top-1"], textposition="top center"
                                    ))
                                else:
                                    fig.add_trace(go.Scatter(
                                        x=[best_row[x_axis]], y=[best_row[y_axis]],
                                        mode="markers+text", name="Top-1",
                                        marker=dict(size=20, symbol="star", color="red", line=dict(color="black", width=1)),
                                        text=["Top-1"], textposition="top center", showlegend=True
                                    ))
                            
                            # Layout Updates (Axes & Legend Fix)
                            if not (use_3d and z_axis):
                                fig.update_xaxes(title=axis_options[x_axis], showgrid=True, gridcolor=COLORS["border"])
                                fig.update_yaxes(title=axis_options[y_axis], showgrid=True, gridcolor=COLORS["border"])
                                
                                # Annotation
                                best_idx = df_display[x_axis].idxmax() if "MAX" in axis_options[x_axis] else df_display[x_axis].idxmin()
                                row_best = df_display.loc[best_idx]
                                fig.add_annotation(x=row_best[x_axis], y=row_best[y_axis], text=f"Best {x_axis}", showarrow=True, arrowhead=2, ax=0, ay=-40)
                
                            # 🛠️ 关键修复：将 Legend 移到图表内部左上角，彻底解决与右侧 Colorbar 的冲突
                            fig.update_layout(
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01,
                                    bgcolor="rgba(255,255,255,0.8)", # 半透明背景防止遮挡数据
                                    bordercolor=COLORS['border'],
                                    borderwidth=1
                                ),
                                **layout_settings
                            )
                            
                            st.plotly_chart(fig, use_container_width=False)
                
                elif tab == "🎞️ GA Animation":
                
                    a1, a2 = st.columns([12, 1], gap="small")
                    with a1:
                        st.markdown("### 🎞️ Evolution of Population")
                    with a2:
                        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                        with st.popover("ℹ️"):
                            st.markdown("### How to read this animation")
                
                            st.write("**What this shows**")
                            st.write("• Each frame is one **generation** of the NSGA-II population.")
                            st.write("• Points = candidate mixes evaluated in that generation.")
                
                            st.write("**Axes**")
                            st.write("• Use **X Axis / Y Axis** to choose which two metrics you want to track over time.")
                            st.write("• Labels include **MAX/MIN** to indicate direction of improvement.")
                
                            st.write("**Colors / markers**")
                            st.write("• The moving points are the population for the current generation.")
                            st.write("• If **Highlight Final Pareto** is ON, the diamond markers show the **final Pareto front** as a reference target.")
                
                            st.write("**Controls**")
                            st.write("• **Play/Pause** runs through generations.")
                            st.write("• The slider below the plot lets you jump to a specific generation.")
                            st.write("• **Animation Speed** controls how fast frames advance (lower = faster).")
                
                            st.write("**How to interpret progress**")
                            st.write("• Over generations, points should move toward better trade-offs and concentrate near the final Pareto frontier.")
                
                    # 下面你原来的逻辑保持不变
                    st.markdown(" ")  # 如果你原来有空行可以保留/删除
    
    
                    if df_all_display is None or "Generation" not in df_all_display.columns:
                        st.warning("⚠️ No history found. Please re-run optimization.")
                    else:
                        anim_axis_candidates = [k for k in axis_options.keys() if k in df_all_display.columns]
                        
                        # --- 控制栏美化 ---
                        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1], gap="medium")
                        default_x = "Net_emission" if "Net_emission" in anim_axis_candidates else anim_axis_candidates[0]
                        default_y = "E" if "E" in anim_axis_candidates else anim_axis_candidates[1]
        
                        x_anim = c1.selectbox("X Axis", anim_axis_candidates, index=anim_axis_candidates.index(default_x), format_func=lambda x: axis_options.get(x, x), key="anim_x")
                        y_anim = c2.selectbox("Y Axis", anim_axis_candidates, index=anim_axis_candidates.index(default_y), format_func=lambda x: axis_options.get(x, x), key="anim_y")
                        show_pareto_overlay = c3.toggle("🌟 Highlight Final Pareto", value=True)
                        speed = c4.select_slider("Animation Speed", options=[50, 150, 300, 600], value=150)
        
                        df_anim = df_all_display.copy()
                        x_min, x_max = df_anim[x_anim].min()*0.95, df_anim[x_anim].max()*1.05
                        y_min, y_max = df_anim[y_anim].min()*0.95, df_anim[y_anim].max()*1.05
        
                        # --- 构建每一帧 ---
                        frames = []
                        gens = sorted(df_anim["Generation"].unique())
                        for g in gens:
                            dfg = df_anim[df_anim["Generation"] == g]
                            frames.append(go.Frame(
                                name=str(g),
                                data=[go.Scatter(
                                    x=dfg[x_anim], y=dfg[y_anim],
                                    mode="markers",
                                    marker=dict(
                                        size=9, 
                                        color=COLORS['primary'], 
                                        opacity=0.6,
                                        line=dict(width=1, color="white") # 给点加个白边，更精致
                                    ),
                                    name=f"Generation {g}"
                                )],
                                traces=[0]
                            ))
        
                        # --- 初始画布 ---
                        fig_anim = go.Figure(
                            data=[go.Scatter(
                                x=df_anim[df_anim["Generation"] == gens[0]][x_anim],
                                y=df_anim[df_anim["Generation"] == gens[0]][y_anim],
                                mode="markers",
                                marker=dict(size=9, color=COLORS['primary'], opacity=0.6, line=dict(width=1, color="white")),
                                name="Initial Population"
                            )],
                            layout=go.Layout(
                                xaxis=dict(range=[x_min, x_max], title=axis_options.get(x_anim), gridcolor=COLORS['border'], zeroline=False),
                                yaxis=dict(range=[y_min, y_max], title=axis_options.get(y_anim), gridcolor=COLORS['border'], zeroline=False),
                                template="plotly_white", # 使用纯白模板
                                margin=dict(l=40, r=40, t=20, b=40),
                                hovermode="closest",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(255,255,255,0.7)",
                            ),
                            frames=frames
                        )
        
                        # --- 叠加最终 Pareto (作为静止的参考背景) ---
                        if show_pareto_overlay:
                            fig_anim.add_trace(go.Scatter(
                                x=df_display[x_anim], y=df_display[y_anim],
                                mode="markers",
                                name="Target Frontier",
                                marker=dict(size=10, symbol="diamond-open", color=COLORS['accent'], line=dict(width=2))
                            ))
        
                        # --- 播放逻辑与美化控件 ---
                        fig_anim.update_layout(
                            updatemenus=[dict(
                                type="buttons", showactive=False, x=0, y=1.15,
                                buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=speed, redraw=True), fromcurrent=True)]),
                                         dict(label="|| Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])]
                            )],
                            sliders=[dict(
                                active=0, currentvalue={"prefix": "Evolution Progress: Gen "},
                                pad={"t": 50}, x=0, y=0, len=1.0,
                                steps=[dict(label=str(g), method="animate", args=[[str(g)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for g in gens]
                            )]
                        )
        
                        st.plotly_chart(fig_anim, use_container_width=True)
                elif tab == "🕸️ Parallel Coordinates":
                
                    p1, p2 = st.columns([12, 1], gap="small")
                    with p1:
                        st.markdown("##### Multi-Dimensional Mix Analysis")
                    with p2:
                        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                        with st.popover("ℹ️"):
                            st.markdown("### How to read this plot")
                            st.write("• Each polyline represents **one Pareto solution** (one mix).")
                            st.write("• Each vertical axis is one selected **dimension** (materials or metrics).")
                            st.write("• The position on an axis shows that mix’s value for that variable.")
                            st.write("")
                            st.write("**How to use it**")
                            st.write("• Use **Active Dimensions** to add/remove axes.")
                            st.write("• Drag on an axis to **filter a range** (brushing).")
                            st.write("• Crossing lines indicate trade-offs between variables.")
                            st.write("")
                            st.write("**Color**")
                            st.write("• Line color is mapped to **E** (or the selected color metric in the code).")
                            st.write("• Darker/higher color ≈ higher E (depending on the color scale).")
                
                    all_input_cols = [m[1] for m in MATERIALS_CONFIG] 
                    all_output_cols = ["E", "CO2_emission", "Cost", "Net_emission"]
                    all_available_cols = [c for c in all_input_cols + all_output_cols if c in df_display.columns]
                    default_selection = ["C3S", "C2S", "GGBFS", "E", "Cost", "Net_emission"]
                    default_selection = [c for c in default_selection if c in all_available_cols]
        
                    selected_cols = st.multiselect("Active Dimensions:", options=all_available_cols, default=default_selection)
                    
                    if len(selected_cols) > 1:
                        color_col = "E" if "E" in df_display.columns else selected_cols[0]
                        
                        # 换成和前面 Pareto 图统一的 Viridis 色带，或者用更现代的 px.colors.sequential.Teal
                        fig_par = px.parallel_coordinates(
                            df_display, 
                            dimensions=selected_cols, 
                            color=color_col, 
                            labels={col: col.replace('_', ' ') for col in selected_cols}, 
                            color_continuous_scale="Viridis" 
                        )
                        
                        # 拯救颜值核心：干掉纯蓝纯黑，使用你预定义的全局变量 COLORS，并统一 Inter 字体
                        fig_par.update_traces(
                            labelfont=dict(size=14, color=COLORS['text_head'], family="Inter"),
                            tickfont=dict(size=11, color=COLORS['text_sub'], family="Inter"),
                            rangefont=dict(size=10, color=COLORS['text_sub'], family="Inter")
                        )
                        
                        # 调整边距并让背景透明，融入整体 UI
                        fig_par.update_layout(
                            margin=dict(l=50, r=50, t=60, b=40),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter")
                        )
                        
                        st.plotly_chart(fig_par, use_container_width=True)
                    else:
                        st.warning("⚠️ Select at least two dimensions.")
        

                elif tab == "📋 Data Table":
                    
                    # 🌟 创建两个子 Tab
                    tab_pareto, tab_search = st.tabs(["🏆 Pareto Optimal Mixes", "🎯 Target Search (All Mixes)"])
                    
                    # ----------------- 子 Tab 1: 帕累托最优表 -----------------
                    with tab_pareto:
                        d1, d2 = st.columns([12, 1], gap="small")
                        with d1:
                            st.markdown("### 📋 Pareto Optimal Mixes")
                        with d2:
                            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                            with st.popover("ℹ️"):
                                st.markdown("### How to read this table")
                                st.write("• Each row is one **Pareto-optimal** mix (non-dominated trade-off).")
                                st.write("• Material columns are shown in **grams** for the 100 g binder definition.")
                                st.write("  (Binder = 100 g incl. 4 g gypsum → clinker + SCMs sum to 96 g).")
                                st.write("• Metrics columns (E, CO₂, Cost, Net CO₂) are computed using:")
                                st.write("  - ANN predictions for **E** and **CO₂_abs**")
                                st.write("  - user factors for **CO₂_emission** (ecoinvent defaults editable) and **Cost** (scenario-defined)")
                                st.write("• **Decision_Score (TOPSIS)** ranks the Pareto set (0–1, higher = more recommended).")
                                st.write("• Use sorting/filtering to find mixes that match your priorities, then export CSV.")
        
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

                    # ----------------- 子 Tab 2: 目标逆向搜索 -----------------
                    with tab_search:
                        st.markdown("### 🎯 Target Search (Goal-Seeker)")
                        st.markdown("<span style='color:#64748B; font-size:0.9rem;'>Set hard constraints to filter through <b>ALL</b> explored mixes (not just the Pareto front) to find recipes that meet specific project requirements.</span>", unsafe_allow_html=True)
                        st.markdown("---")
                        
                        if df_all_display is None:
                            st.warning("⚠️ Optimization history is required for Target Search. Please run the optimization first.")
                        else:
                            c_filters, c_plot = st.columns([1, 2.5], gap="large")
                            
                            with c_filters:
                                st.markdown("#### 1. Set Thresholds")
                                min_e, max_e = float(df_all_display["E"].min()), float(df_all_display["E"].max())
                                min_cost, max_cost = float(df_all_display["Cost"].min()), float(df_all_display["Cost"].max())
                                min_co2, max_co2 = float(df_all_display["CO2_emission"].min()), float(df_all_display["CO2_emission"].max())
                                
                                def_e = max(min_e, min(20.0, 10))  
                                def_cost = max(min_cost, min(0.12, max_cost)) 
                                def_co2 = max(min_co2, min(0.60, max_co2)) 
                                
                                target_e = st.slider("⚡ Min Elastic modulus (E, GPa)", min_value=min_e, max_value=max_e, value=def_e, step=0.5)
                                target_cost = st.slider("💰 Max Cost (€/kg)", min_value=min_cost, max_value=max_cost, value=def_cost, step=0.005, format="%.3f")
                                target_co2 = st.slider("🏭 Max CO₂ Emission (kg/kg)", min_value=min_co2, max_value=max_co2, value=def_co2, step=0.01, format="%.3f")
                                
                                mask = (df_all_display["E"] >= target_e) & (df_all_display["Cost"] <= target_cost) & (df_all_display["CO2_emission"] <= target_co2)
                                df_filtered = df_all_display[mask].copy()
                                
                                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                                if len(df_filtered) > 0:
                                    st.success(f"✅ Found **{len(df_filtered)}** viable mixes!")
                                else:
                                    st.error("❌ No mixes found. Try relaxing your constraints.")
                            
                            with c_plot:
                                st.markdown("#### 2. Feasible Region")
                                fig_target = go.Figure()
                                fig_target.add_trace(go.Scatter(x=df_all_display["CO2_emission"], y=df_all_display["E"], mode="markers", name="All Explored", marker=dict(size=5, color="lightgrey", opacity=0.3), hoverinfo="none"))
                                if len(df_filtered) > 0:
                                    fig_target.add_trace(go.Scatter(
                                        x=df_filtered["CO2_emission"], y=df_filtered["E"], mode="markers", name="Viable Mixes",
                                        marker=dict(size=8, color=df_filtered["Cost"], colorscale="Viridis", showscale=True, colorbar=dict(title="Cost (€/kg)", thickness=15), line=dict(width=1, color="white")),
                                        text=[f"Mix ID: {i}<br>E: {row['E']:.2f}<br>CO2: {row['CO2_emission']:.3f}<br>Cost: {row['Cost']:.3f}" for i, row in df_filtered.iterrows()], hoverinfo="text"
                                    ))
                                fig_target.add_hline(y=target_e, line_dash="dash", line_color="red", annotation_text="Min E")
                                fig_target.add_vline(x=target_co2, line_dash="dash", line_color="red", annotation_text="Max CO₂")
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
                                st.download_button("📥 Download Viable Mixes (CSV)", data=df_filtered_sorted[display_cols].to_csv(index=False).encode("utf-8"), file_name="target_search_viable_mixes.csv", mime="text/csv", type="primary")
        
                elif tab == "⚖️ Benchmark Comparison":
                            with st.expander("⚙️ Configure Baseline (OPC)", expanded=False):
                                b_c1, b_c2, b_c3, b_c4 = st.columns(4)
                                # 默认值调整为 kg/kg 量级
                                opc_e = b_c1.number_input("Ref E (GPa)", value=17.0, step=0.5, key="bench_e")
                                opc_co2 = b_c2.number_input("Ref CO₂ (kg/kg)", value=0.76, step=0.01, format="%.3f", key="bench_co2")
                                opc_cost = b_c3.number_input("Ref Cost (€/kg)", value=0.22, step=0.01, format="%.3f", key="bench_cost")
                                # opc_net = b_c4.number_input("Ref Net CO₂ (kg/kg)", value=0.40, step=0.01, key="bench_net")
                             
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
                                    # "Net_emission": float(opc_net),
                                }
        
                                    st.markdown("---")
                                    
                                    def delta_metric_card(label, val, ref, unit, inverse=False):
                                        diff = val - ref
                                        pct = (diff / ref * 100) if ref != 0 else 0
                                        # Inverse: 意味着值越小越好（如Cost, CO2），如果 diff < 0 (下降) 则是好事(绿色/normal)
                                        # Normal: 意味着值越大越好 (如E)，如果 diff > 0 (上升) 则是好事
                                        if inverse:
                                            color = "normal" if diff < 0 else "inverse"
                                        else:
                                            color = "normal" if diff > 0 else "inverse"
                                            
                                        # 使用3位小数显示 kg/kg
                                        fmt = "%.3f" if "kg" in unit else "%.2f"
                                        st.metric(label=label, value=f"{val:{fmt[1:]}} {unit}", delta=f"{diff:{fmt[1:]}} ({pct:+.1f}%)", delta_color=color)
                                    
                                    delta_metric_card("E-Modulus", row["E"], opc_e, "GPa")
                                    delta_metric_card("CO₂ Emission", row["CO2_emission"], opc_co2, "kg/kg", inverse=True)
                                    delta_metric_card("Cost", float(row["Cost"]), float(opc_cost), "€/kg", inverse=True)
        
                                    # 修正上一行逻辑
                                    # st.empty() # 清除上面的错误 metric
                                    # st.metric(label="Cost", value=f"{row['Cost']:.4f} €/kg", delta=f"{row['Cost']-opc_cost:.4f} ({(row['Cost']-opc_cost)/opc_cost*100:+.1f}%)", delta_color="normal" if row['Cost'] < opc_cost else "inverse")
                

                            with c_right2:
                                if selected_idx is not None:
                                    row = df_display.iloc[selected_idx]
                            
                                    # =========================
                                    # 1) Composition (gram)
                                    # =========================
                                    row1_col1, row1_col2 = st.columns(2, gap="medium")
                            
                                    with row1_col1:
                                        st.markdown("#### 🍩 Composition (gram)")
                                        mat_cols = [m[1] for m in MATERIALS_CONFIG]
                                        materials = {k: float(row[k]) for k in mat_cols if k in df_display.columns}
                                        df_mat = pd.DataFrame(list(materials.items()), columns=["Material", "Value"])
                                        df_mat = df_mat[df_mat["Value"] > 0]
                            
                                        fig_pie = px.pie(
                                            df_mat,
                                            values="Value",
                                            names="Material",
                                            hole=0.55,
                                            color_discrete_sequence=px.colors.qualitative.Pastel
                                        )
                                        fig_pie.update_traces(textposition="inside", textinfo="none", texttemplate="%{value:.1f}%")
                                        fig_pie.update_layout(
                                            showlegend=False,
                                            margin=dict(t=30, b=10, l=10, r=10),
                                            height=280,
                                            annotations=[dict(text="Mix", x=0.5, y=0.5, font_size=14, showarrow=False)]
                                        )
                                        st.plotly_chart(fig_pie, use_container_width=True)
                            
                                    # =========================
                                    # 2) Metric Comparison 先放右上
                                    # =========================
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
                            
                                        fig_bar = px.bar(
                                            pd.DataFrame(metrics_data),
                                            x="Metric",
                                            y="Value",
                                            color="Type",
                                            barmode="group",
                                            text_auto=".3f"
                                        )
                                        fig_bar.update_layout(
                                            height=280,  # 和 pie 对齐高度
                                            margin=dict(t=20, b=10, l=0, r=0),
                                            legend=dict(orientation="h", y=1.02, x=0.6),
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            plot_bgcolor="rgba(255,255,255,0.5)",
                                        )
                                        st.plotly_chart(fig_bar, use_container_width=True)
                            
                                    st.markdown("---")
                            

                                    # 3) Position on Frontier (Interactive like Pareto Analysis)
                                    # =========================
                                    h1, h2 = st.columns([12, 1], gap="small")
                                    with h1:
                                        st.markdown("#### 📌 Position on Frontier")
                                    with h2:
                                        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                                        with st.popover("ℹ️"):
                                            st.markdown("### How to read this plot")
                                            st.write("• Grey: explored candidates • Colored: Pareto")
                                            st.write("• ⭐ Selected mix • ✖ Baseline (OPC)")
                                            st.write("• Choose X/Y (and Z) axes to explore different trade-offs.")
                                            st.write("• Color can map Decision_Score or any metric/material.")
                                    
                                    # 可用维度（和 Pareto Analysis 一致：指标 + 材料 + Decision_Score）
                                    chart_options = [k for k in axis_options.keys() if k in df_display.columns]
                                    material_cols = [m[1] for m in MATERIALS_CONFIG]
                                    color_candidates = ["Decision_Score"] + chart_options + material_cols
                                    color_options = [c for c in color_candidates if c in df_display.columns]
                                    
                                    # 默认轴（更合理：X=CO2_emission, Y=E）
                                    default_x = "CO2_emission" if "CO2_emission" in chart_options else (chart_options[0] if chart_options else None)
                                    default_y = "E" if "E" in chart_options else (chart_options[1] if len(chart_options) > 1 else (chart_options[0] if chart_options else None))
                                    default_z = "Cost" if "Cost" in chart_options else (chart_options[2] if len(chart_options) > 2 else None)
                                    
                                    # --- 控件区 ---
                                    c_ctrl = st.columns(5)
                                    x_axis = c_ctrl[0].selectbox(
                                        "X Axis", chart_options,
                                        index=chart_options.index(default_x) if default_x in chart_options else 0,
                                        format_func=lambda x: axis_options.get(x, x),
                                        key="bench_pos_x"
                                    )
                                    y_axis = c_ctrl[1].selectbox(
                                        "Y Axis", chart_options,
                                        index=chart_options.index(default_y) if default_y in chart_options else (0 if chart_options else 0),
                                        format_func=lambda x: axis_options.get(x, x),
                                        key="bench_pos_y"
                                    )
                                    
                                    default_color_idx = color_options.index("Decision_Score") if "Decision_Score" in color_options else 0
                                    color_col = c_ctrl[2].selectbox(
                                        "Color Points By", color_options,
                                        index=default_color_idx,
                                        key="bench_pos_color"
                                    )
                                    
                                    show_all = c_ctrl[3].toggle("Show History", value=True, key="bench_pos_showhist")
                                    
                                    use_3d_allowed = len(chart_options) >= 3
                                    use_3d = c_ctrl[4].toggle("3D View", value=False, key="bench_pos_3d") if use_3d_allowed else False
                                    z_axis = None
                                    if use_3d and use_3d_allowed:
                                        z_axis = st.selectbox(
                                            "Z Axis", chart_options,
                                            index=chart_options.index(default_z) if (default_z in chart_options) else 0,
                                            format_func=lambda x: axis_options.get(x, x),
                                            key="bench_pos_z"
                                        )
                                    
                                    # --- 绘图 ---
                                    layout_settings = dict(
                                        height=520,
                                        margin=dict(l=20, r=80, t=20, b=20),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(255,255,255,0.5)",
                                        font=dict(family="Inter", color=COLORS["text_body"])
                                    )
                                    
                                    fig_pos = go.Figure()
                                    
                                    # 1) explored points (grey)
                                    if show_all and df_all_display is not None:
                                        if use_3d and z_axis:
                                            fig_pos.add_trace(go.Scatter3d(
                                                x=df_all_display[x_axis], y=df_all_display[y_axis], z=df_all_display[z_axis],
                                                mode="markers", name="Explored",
                                                marker=dict(size=2, color="lightgrey", opacity=0.2)
                                            ))
                                        else:
                                            fig_pos.add_trace(go.Scatter(
                                                x=df_all_display[x_axis], y=df_all_display[y_axis],
                                                mode="markers", name="Explored",
                                                marker=dict(size=5, color="lightgrey", opacity=0.25)
                                            ))
                                    
                                    # 2) pareto points (colored)
                                    colorbar_title = "Decision Score<br>(TOPSIS)" if color_col == "Decision_Score" else color_col
                                    hover_text = [
                                        f"Rank #{i+1}<br>Score={row_i['Decision_Score']:.3f}" if "Decision_Score" in df_display.columns else f"Mix #{i+1}"
                                        for i, row_i in df_display.iterrows()
                                    ]
                                    
                                    if use_3d and z_axis:
                                        fig_pos.add_trace(go.Scatter3d(
                                            x=df_display[x_axis], y=df_display[y_axis], z=df_display[z_axis],
                                            mode="markers", name="Pareto",
                                            marker=dict(
                                                size=6,
                                                color=df_display[color_col],
                                                colorscale="Viridis",
                                                opacity=0.95,
                                                showscale=True,
                                                colorbar=dict(title=dict(text=colorbar_title, side="right"))
                                            ),
                                            text=hover_text,
                                            showlegend=True
                                        ))
                                        fig_pos.update_layout(scene=dict(
                                            xaxis_title=axis_options.get(x_axis, x_axis),
                                            yaxis_title=axis_options.get(y_axis, y_axis),
                                            zaxis_title=axis_options.get(z_axis, z_axis),
                                        ))
                                    else:
                                        fig_pos.add_trace(go.Scatter(
                                            x=df_display[x_axis], y=df_display[y_axis],
                                            mode="markers", name="Pareto",
                                            marker=dict(
                                                size=10,
                                                color=df_display[color_col],
                                                colorscale="Viridis",
                                                opacity=0.95,
                                                showscale=True,
                                                colorbar=dict(
                                                    title=dict(text=colorbar_title, side="right"),
                                                    thickness=15,
                                                    xpad=10
                                                )
                                            ),
                                            text=hover_text,
                                            showlegend=True
                                        ))
                                        fig_pos.update_xaxes(title=axis_options.get(x_axis, x_axis), showgrid=True, gridcolor=COLORS["border"])
                                        fig_pos.update_yaxes(title=axis_options.get(y_axis, y_axis), showgrid=True, gridcolor=COLORS["border"])
                                    
                                    # 3) baseline point (X) — only if baseline has these metrics
                                    def _baseline_has(axis_name: str) -> bool:
                                        return (axis_name in baseline_data)
                                    
                                    if _baseline_has(x_axis) and _baseline_has(y_axis) and (not use_3d or _baseline_has(z_axis)):
                                        if use_3d and z_axis:
                                            fig_pos.add_trace(go.Scatter3d(
                                                x=[baseline_data[x_axis]], y=[baseline_data[y_axis]], z=[baseline_data[z_axis]],
                                                mode="markers+text", name="Baseline (OPC)",
                                                marker=dict(size=9, symbol="x", color="gray", line=dict(width=2)),
                                                text=["Baseline"], textposition="top center"
                                            ))
                                        else:
                                            fig_pos.add_trace(go.Scatter(
                                                x=[baseline_data[x_axis]], y=[baseline_data[y_axis]],
                                                mode="markers+text", name="Baseline (OPC)",
                                                marker=dict(size=14, symbol="x", color="gray", line=dict(width=2)),
                                                text=["Baseline"], textposition="top center"
                                            ))
                                    
                                    # 4) selected point (⭐) — only if selected row has these columns (always true for chart_options)
                                    if use_3d and z_axis:
                                        fig_pos.add_trace(go.Scatter3d(
                                            x=[float(row[x_axis])], y=[float(row[y_axis])], z=[float(row[z_axis])],
                                            mode="markers+text", name="Selected Mix",
                                            marker=dict(size=9, symbol="star", color="red", line=dict(color="black", width=1)),
                                            text=["Selected"], textposition="top center"
                                        ))
                                    else:
                                        fig_pos.add_trace(go.Scatter(
                                            x=[float(row[x_axis])], y=[float(row[y_axis])],
                                            mode="markers+text", name="Selected Mix",
                                            marker=dict(size=18, symbol="star", color="red", line=dict(color="black", width=1)),
                                            text=["Selected"], textposition="top center"
                                        ))
                                    
                                    # legend fix (avoid colorbar clash)
                                    fig_pos.update_layout(
                                        legend=dict(
                                            yanchor="top", y=0.99,
                                            xanchor="left", x=0.01,
                                            bgcolor="rgba(255,255,255,0.8)",
                                            bordercolor=COLORS['border'],
                                            borderwidth=1
                                        ),
                                        **layout_settings
                                    )
                                    
                                    st.plotly_chart(fig_pos, use_container_width=True)
                            
                                    # 下面如果你还有 “Technical Export” 等内容，继续接在这里即可
                                    # ==========================================
                                    #      FINAL SECTION: REPORT GENERATION
                                    # ==========================================
                                    st.markdown("---")
                                    
                                    te1, te2 = st.columns([12, 1], gap="small")
                                    with te1:
                                        st.markdown("### 📄 Technical Export")
                                    with te2:
                                        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                                        with st.popover("ℹ️"):
                                            st.markdown("### About the PDF Technical Audit")
                                    
                                            st.write("**What the report contains**")
                                            st.write("• Your selected objectives, bounds, and algorithm settings.")
                                            st.write("• The Pareto-optimal solutions and TOPSIS ranking (if enabled).")
                                            st.write("• Baseline (OPC) vs selected mix comparison and key plots.")
                                    
                                            st.write("**How the recommended mix is chosen**")
                                            st.write("• The report highlights a ‘recommended’ mix based on **Decision Score (TOPSIS)**.")
                                            st.write("• This recommendation is **computed from model predictions + stoichiometric/impact calculations** under your chosen factors.")
                                    
                                            st.write("**Important limitation (validation)**")
                                            st.write("• Results are based on **simulation-derived / model-predicted** properties and **scenario-dependent factors**.")
                                            st.write("• The recommended formulation should be treated as a **screening suggestion** and **must be experimentally validated** (e.g., strength, durability, carbonation) before practical use.")
    
                                    
                                    # 只有当优化运行过（有结果）时才显示报告生成区
                                    if st.session_state.df_pareto is not None:
                                        # 1. 确保变量已经定义（防止顺序问题报错）
                                        # 这里通过从 session_state 或直接读取变量来保证安全
                                        try:
                                            report_params = {
                                                "Curing Time": f"{st.session_state.get('time_input', 28)} Days",
                                                "w/c Ratio": 0.5,
                                                "Temperature": "25.0 °C",
                                                "Target Binder": "96.0 g"
                                            }
                                    
                                            # 2. 捕获搜索边界 (确保这些变量名与你 slider 处的 key 一致)
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
                                    
                                            # 3. 捕获遗传算法设置 (直接读取之前定义的输入框变量)
                                            # 注意：这里要确保 ga_pop 等变量在上面已经通过 number_input 定义过
                                            ga_conf = {
                                                "pop": ga_pop,
                                                "gen": ga_gen,
                                                "seed": ga_seed
                                            }
                                    
                                            # 4. 优化目标
                                            opt_config = []
                                            if st.session_state.get("obj_e_max"):
                                                opt_config.append({'col': 'E', 'impact': '+', 'name': 'Maximize E-Modulus'})
                                            
                                            if st.session_state.get("obj_cost_min"):
                                                opt_config.append({'col': 'Cost', 'impact': '-', 'name': 'Minimize Cost'})
                                            
                                            if st.session_state.get("obj_co2_min"):
                                                opt_config.append({'col': 'CO2_emission', 'impact': '-', 'name': 'Minimize CO2 Emission'})
                                            
                                            if st.session_state.get("obj_net_min"):
                                                opt_config.append({'col': 'Net_emission', 'impact': '-', 'name': 'Minimize Net CO2'})
                                            
                                            if st.session_state.get("obj_co2abs_max"):
                                                opt_config.append({'col': 'CO2_abs', 'impact': '+', 'name': 'Maximize CO2 Uptake'})
        
                                    
                                            # 基准数据
                                            baseline_data = {
                                                "E": float(st.session_state.get("bench_e", 30.0)),
                                                "CO2_emission": float(st.session_state.get("bench_co2", 0.76)),
                                                "Cost": float(st.session_state.get("bench_cost", 0.12)),
                                                "Net_emission": float(st.session_state.get("bench_net", 0.40)),
                                            }
                                    
                                            # 5. 生成按钮
                                            if st.button("🚀 Generate PDF Technical Audit", type="primary", use_container_width=True):
                                                with st.spinner("Compiling technical report..."):
                                                    pdf_bytes = create_pdf_report(
                                                        st.session_state.df_pareto, 
                                                        report_params, 
                                                        baseline_data, 
                                                        opt_config,
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
                                                
                                        except NameError as e:
                                            st.warning(f"Waiting for optimization settings to initialize... ({e})")
    
    
                         
        else:
            st.markdown(f"""
            <div style='text-align: center; padding: 60px; color: {COLORS['text_sub']}; border: 2px dashed {COLORS['border']}; border-radius: 12px; margin-top: 40px;'>
                <h3>👋 Ready to Optimize</h3>
                <p>Configure your material constraints (in grams) on the left panel, then click <b>START OPTIMIZATION</b>.</p>
            </div>
            """, unsafe_allow_html=True)
    show_results()