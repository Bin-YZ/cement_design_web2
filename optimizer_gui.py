import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import ipywidgets as w
from IPython.display import display, clear_output
import plotly.express as px
import sys
import traceback

from model_wrapper import ModelWrapper
from metrics import MetricsCalculator
from pareto_optimizer import ParetoOptimizer

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from nsga_problem import ConcreteMixProblem
from IPython.display import HTML, display
import plotly.graph_objects as go
class OptimizerGUI:
    """Enhanced concrete mix optimizer GUI with beautiful styling"""

    FIXED_WC = 0.5
    FIXED_GYPSUM = 4.0
    FIXED_TEMP = 25.0
    TOTAL_BINDER_TARGET = 100.0 - FIXED_GYPSUM

    MATERIALS = [
        ("C3S", "C3S", 0.82, 0.141),
        ("C2S", "C2S", 0.69, 0.141),
        ("C3A", "C3A", 0.73, 0.141),
        ("C4AF", "C4AF", 0.55, 0.141),
        ("Silica fume", "silica_fume", 0.0035, 4.92),
        ("GGBFS", "GGBFS", 0.13, 0.056),
        ("Fly ash", "fly_ash", 0.004, 0.02),
        ("Calcined clay", "calcined_clay", 0.27, 0.11),
        ("Limestone", "limestone", 0.0023, 0.0227),
        ("Gypsum", "Gypsum", 0.0082, 0.051),
    ]

    # 🎨 Color scheme
    COLORS = {
        'primary': '#7D8CA3',      # Steel gray blue - 主色（冷静、专业）
        'success': '#9BAE91',      # Sage green - 柔和成功色
        'warning': '#C7A36E',      # Muted sand - 温暖提示色
        'danger':  '#B66A6A',      # Dusty rose red - 低饱和红
        'info':    '#7FA6A1',      # Mist teal - 信息提示色
        'dark':    '#4B4B4B',      # Charcoal gray - 深灰文本
        'light':   '#F2F2F2',      # Off-white - 背景浅灰
        'border':  '#C9C9C9',      # Light border gray
        'clinker': '#A98B73',      # Taupe brown - 熟料相
        'scm':     '#9CA68C',      # Olive gray - SCM 类材料
        'objectives': '#8C7B9E',   # Dusty lavender - 目标模块
        'params':  '#7B93A3',      # Slate blue gray - 参数模块
    }

    def __init__(self, model_path: str):
        self.model = ModelWrapper(model_path)
        self.metrics = MetricsCalculator(self.FIXED_WC, self.FIXED_GYPSUM, self.FIXED_TEMP)
        self.run_status = w.HTML(value="")  # ⏳ 运行状态提示
        # 熟料相范围 - 添加颜色样式
        slider_style = {'description_width': 'auto', 'handle_color': self.COLORS['clinker']}
        self.c3s_rng = w.FloatRangeSlider(value=[45.0, 80.0], min=45, max=80, step=0.5,
                                          description="🔶 C3S %", layout=w.Layout(width="280px"),
                                          readout_format=".1f", style=slider_style)
        self.c2s_rng = w.FloatRangeSlider(value=[10.0, 32.0], min=10, max=40, step=0.5,
                                          description="🔶 C2S %", layout=w.Layout(width="280px"),
                                          readout_format=".1f", style=slider_style)
        self.c3a_rng = w.FloatRangeSlider(value=[0.0, 14.0], min=0, max=15, step=0.5,
                                          description="🔶 C3A %", layout=w.Layout(width="280px"),
                                          readout_format=".1f", style=slider_style)
        self.c4af_rng = w.FloatRangeSlider(value=[0.0, 15.0], min=0, max=20, step=0.5,
                                           description="🔶 C4AF %", layout=w.Layout(width="280px"),
                                           readout_format=".1f", style=slider_style)
        self.cl_sum_rng = w.FloatRangeSlider(value=[20.0, 96.0], min=20, max=96, step=0.5,
                                             description="🔶 Clinker total %", layout=w.Layout(width="280px"),
                                             readout_format=".1f", style=slider_style)
        
        # 时间滑块
        time_style = {'description_width': 'auto', 'handle_color': self.COLORS['info']}
        self.time_in = w.IntSlider(value=28, min=1, max=36500, step=1,
                                   description="⏱️ Time (days)", layout=w.Layout(width="280px"),
                                   style=time_style)
        
        # SCM 范围 - 绿色主题
        scm_style = {'description_width': 'auto', 'handle_color': self.COLORS['scm']}
        self.sf_rng = w.FloatRangeSlider(value=[0.0, 10.0], min=0, max=10, step=0.5,
                                         description="🟢 Silica fume %", layout=w.Layout(width="280px"),
                                         readout_format=".1f", style=scm_style)
        self.gg_rng = w.FloatRangeSlider(value=[0.0, 80.0], min=0, max=80, step=0.5,
                                         description="🟢 GGBFS %", layout=w.Layout(width="280px"),
                                         readout_format=".1f", style=scm_style)
        self.fa_rng = w.FloatRangeSlider(value=[0.0, 35.0], min=0, max=35, step=0.5,
                                         description="🟢 Fly ash %", layout=w.Layout(width="280px"),
                                         readout_format=".1f", style=scm_style)
        self.cc_rng = w.FloatRangeSlider(value=[0.0, 35.0], min=0, max=35, step=0.5,
                                         description="🟢 Calcined clay %", layout=w.Layout(width="280px"),
                                         readout_format=".1f", style=scm_style)
        self.ls_rng = w.FloatRangeSlider(value=[0.0, 35.0], min=0, max=35, step=0.5,
                                         description="🟢 Limestone %", layout=w.Layout(width="280px"),
                                         readout_format=".1f", style=scm_style)

        # 固定参数
        fixed_style = {'description_width': 'auto'}
        self.wc_fixed = w.FloatText(self.FIXED_WC, description="💧 w/c", disabled=True,
                                    layout=w.Layout(width="240px"), style=fixed_style)
        self.gy_fixed = w.FloatText(self.FIXED_GYPSUM, description="⚪ Gypsum %", disabled=True,
                                    layout=w.Layout(width="240px"), style=fixed_style)
        self.temp_fixed = w.FloatText(self.FIXED_TEMP, description="🌡️ Temp °C", disabled=True,
                                      layout=w.Layout(width="240px"), style=fixed_style)
        self.total_bind = w.FloatText(self.TOTAL_BINDER_TARGET, description="📊 Clinker+SCMs %",
                                      disabled=True, layout=w.Layout(width="240px"),
                                      style=fixed_style)

        # 物料因子滑块
        self.co2_sliders, self.cost_sliders = {}, {}
        emission_style = {'description_width': 'auto', 'handle_color': self.COLORS['danger']}
        cost_style = {'description_width': 'auto', 'handle_color': self.COLORS['warning']}
        
        for label, key, emis, cost in self.MATERIALS:
            self.co2_sliders[key] = w.FloatSlider(value=float(emis), min=0.0, max=1.5, step=0.0005,
                                                  description=f"🏭 {label}", layout=w.Layout(width="280px"),
                                                  readout_format=".4f", style=emission_style)
            self.cost_sliders[key] = w.FloatSlider(value=float(cost), min=0.0, max=10.0, step=0.0005,
                                                   description=f"💰 {label}", layout=w.Layout(width="280px"),
                                                   readout_format=".4f", style=cost_style)

        # 目标复选框 - 添加图标和颜色
        self.objective_boxes = {
            "E_max": w.Checkbox(value=True, description="📈 Maximize E", 
                               style={'description_width': 'initial'}),
            "CO2abs_max": w.Checkbox(value=False, description="🌱 Maximize CO₂ uptake",
                                    style={'description_width': 'initial'}),
            "CO2_min": w.Checkbox(value=False, description="♻️ Minimize CO₂ emission",
                                 style={'description_width': 'initial'}),
            "Cost_min": w.Checkbox(value=False, description="💵 Minimize Cost",
                                  style={'description_width': 'initial'}),
            "Net_min": w.Checkbox(value=True, description="🌍 Minimize Net emission",
                                 style={'description_width': 'initial'}),
        }
        
        self.obj_box_group = w.VBox(
            [w.HTML(f"<b style='color:{self.COLORS['objectives']};font-size:15px;'>🎯 Optimization Objectives</b>")] + 
            list(self.objective_boxes.values()),
            layout=w.Layout(gap="4px", padding="8px", border=f"2px solid {self.COLORS['objectives']}", 
                          border_radius="8px", background_color="#f8f9fa")
        )

        # 可视化控制
        viz_style = {'description_width': 'auto'}
        self.show_all = w.Checkbox(True, description="👁️ Show all points",
                                   layout=w.Layout(width="240px"), style=viz_style)
        self.alpha = w.FloatSlider(value=0.3, min=0.05, max=1.0, step=0.05,
                                   description="🔍 Alpha", layout=w.Layout(width="260px"),
                                   readout_format=".2f", style={'description_width': 'auto', 
                                                               'handle_color': self.COLORS['primary']})

        # NSGA-II 参数
        param_style = {'description_width': 'auto', 'handle_color': self.COLORS['params']}
        self.ga_pop = w.IntSlider(value=10, min=40, max=500, step=10,
                                  description="👥 Population", layout=w.Layout(width="220px"),
                                  style=param_style)
        self.ga_gen = w.IntSlider(value=5, min=20, max=500, step=10,
                                  description="🔄 Generations", layout=w.Layout(width="320px"),
                                  style=param_style)
        self.ga_pc = w.FloatSlider(value=0.9, min=0.0, max=1.0, step=0.01,
                                   description="🧬 Crossover prob", layout=w.Layout(width="220px"),
                                   readout_format=".2f", style=param_style)
        self.ga_pm = w.FloatSlider(value=0.1, min=0.0, max=1.0, step=0.01,
                                   description="🔀 Mutation prob", layout=w.Layout(width="220px"),
                                   readout_format=".2f", style=param_style)
        self.ga_seed = w.IntSlider(value=12345, min=0, max=1_000_000, step=1,
                                   description="🎲 Seed", layout=w.Layout(width="220px"),
                                   style=param_style)
        self.btn_nsga = w.Button(description="▶️ Run NSGA-II", button_style="success",
                                 layout=w.Layout(width="240px", height="45px", font_weight="bold"),
                                 style={'button_color': self.COLORS['success']})

        # 输出区
                # 输出区
        # 输出区
        self.plot_out = w.Output()     # 图
        self.table_out = w.Output()    # 配合比表格
        self.log_out = w.Output()

        # Result 里的 Tab
        self.results_tabs = w.Tab(children=[self.plot_out, self.table_out])
        self.results_tabs.set_title(0, "📊 Plot")
        self.results_tabs.set_title(1, "📋 Table")


        # 下载链接（结果生成后填充）
        self.download_link = w.HTML(value="")  # will be filled after CSV is saved

        # 是否启用自定义 NSGA-II 参数（默认关闭=使用默认值并隐藏参数框）
        self.use_custom_nsga = w.Checkbox(
            value=False, 
            description="Customize NSGA-II parameters",
            style={'description_width': 'initial'}
        )
                # ========= Pareto vs OPC 比较控件 =========
        # 保存 NSGA-II 的 Pareto 结果，供后面比较用
        self.df_pareto = None

        # 选择 Pareto 配方
        self.compare_pareto_dropdown = w.Dropdown(
            options=[],
            description="Pareto mix",
            layout=w.Layout(width="340px"),
            style={'description_width': 'initial'}
        )
        # 比较时选择具体指标（E / CO2 / Cost / Net）
        self.compare_metric_dropdown = w.Dropdown(
            options=[
                ("E (GPa)", "E"),
                ("CO₂ emission (kg/kg)", "CO2_emission"),
                ("Cost (€/kg)", "Cost"),
                ("Net emission (kg/kg)", "Net_emission"),
            ],
            value="E",
            description="Select metric",
            layout=w.Layout(width="340px"),
            style={'description_width': 'initial'}
        )

        # OPC 参考水泥输入
        opc_base_style = {'description_width': 'initial'}

        self.opc_name = w.Text(
            value="OPC reference",
            description="Name",
            layout=w.Layout(width="260px"),
            style=opc_base_style
        )

        # 📏 E：用主色
        self.opc_E = w.FloatSlider(
            value=30.0,
            min=15.0,
            max=60.0,
            step=0.5,
            description="📏 E (GPa)",
            layout=w.Layout(width="260px"),
            readout_format=".1f",
            style={**opc_base_style, 'handle_color': self.COLORS['primary']}
        )

        # 🌫️ CO2：用红色系（排放）
        self.opc_CO2 = w.FloatSlider(
            value=0.78,
            min=0.50,
            max=1.10,
            step=0.01,
            description="🌫️ CO₂ emission (kg/kg)",
            layout=w.Layout(width="260px"),
            readout_format=".2f",
            style={**opc_base_style, 'handle_color': self.COLORS['danger']}
        )

        # 💰 Cost：用黄色系（经济）
        self.opc_cost = w.FloatSlider(
            value=0.12,
            min=0.05,
            max=0.30,
            step=0.005,
            description="💰 Cost (€/kg)",
            layout=w.Layout(width="260px"),
            readout_format=".3f",
            style={**opc_base_style, 'handle_color': self.COLORS['warning']}
        )

        # 🌍 Net emission：用绿色系（可持续）
        self.opc_net = w.FloatSlider(
            value=0.78,
            min=0.50,
            max=1.10,
            step=0.01,
            description="🌍 Net emission (kg/kg)",
            layout=w.Layout(width="260px"),
            readout_format=".2f",
            style={**opc_base_style, 'handle_color': self.COLORS['success']}
        )



        # 比较按钮 + 输出区
        self.btn_compare = w.Button(
            description="🔍 Compare with OPC",
            button_style="info",
            layout=w.Layout(width="220px", height="40px"),
            style={'button_color': self.COLORS['info'], 'font_weight': 'bold'}
        )
        self.compare_out = w.Output()

        # 绑定事件
        self.btn_compare.on_click(self._on_compare_click)


        # 时间快速选择按钮
        self._time_presets = {
            "7d": 7,
            "28d": 28,
            "90d": 90,
            "1yr": 365,
            "10yr": 3650,
        }
        self._time_btns = {}
        for k in self._time_presets.keys():
            self._time_btns[k] = w.Button(
                description=k, 
                button_style="info",
                layout=w.Layout(width="64px", height="32px"),
                style={'button_color': self.COLORS['info']}
            )
            days = self._time_presets[k]
            self._time_btns[k].on_click(lambda _b, d=days: setattr(self.time_in, "value", d))
        
        self.time_quick = w.VBox([
            w.HTML(f"<b style='color:{self.COLORS['info']};'>⚡ Quick Select</b>"),
            w.HBox(list(self._time_btns.values()), layout=w.Layout(gap="6px"))
        ], layout=w.Layout(gap="4px"))

        # 组装 UI & 事件
        self._build_ui()
        self.btn_nsga.on_click(self._on_nsga_click)

    def _collapsible_section(self, title: str, content, collapsed=False, color=None):
        if color is None:
            color = self.COLORS['dark']
        
        toggle_btn = w.Button(
            description=f"{'▶️' if collapsed else '🔽'} {title}",
            button_style='',
            layout=w.Layout(width='100%', font_weight='bold'),
            style={'button_color': color, 'font_color': 'white'}
        )
        
        content_box = w.VBox([content], layout=w.Layout(
            border=f'2px solid {color}',
            padding='10px',
            margin='0 0 10px 0',
            border_radius='8px',
            display='none' if collapsed else 'block',
            background_color='#fafafa'
        ))

        def toggle_visibility(btn):
            if content_box.layout.display == 'none':
                content_box.layout.display = 'block'
                btn.description = f"🔽 {title}"
            else:
                content_box.layout.display = 'none'
                btn.description = f"▶️ {title}"

        toggle_btn.on_click(toggle_visibility)
        return w.VBox([toggle_btn, content_box], layout=w.Layout(margin='0 0 5px 0'))

    def _read_factors(self):
        emis, cost = {}, {}
        for _, key, _, _ in self.MATERIALS:
            emis[key] = float(self.co2_sliders[key].value)
            cost[key] = float(self.cost_sliders[key].value)
        return emis, cost
    def _build_ui(self):
        # 标题
        title = w.HTML(f"""
            <div style='text-align:center; padding:15px; background:linear-gradient(135deg, {self.COLORS['primary']}, {self.COLORS['info']}); 
                        border-radius:12px; margin:8px 0 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h1 style='color:white; margin:0; font-size:28px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                    🏗️ Cement Mix Optimizer
                </h1>
                <p style='color:#ecf0f1; margin:5px 0 0 0; font-size:14px;'>
                    NSGA-II Multi-Objective Optimization
                </p>
            </div>
        """)

        # 左侧栏 - Materials
        materials_grid = w.VBox([
            w.HTML(f"<b style='display:block; margin:2px 0 6px; color:{self.COLORS['clinker']}; font-size:15px;'>🔶 Clinker Phases</b>"),
            self.c3s_rng, self.c2s_rng, self.c3a_rng, self.c4af_rng, self.cl_sum_rng,
            w.HTML(f"<b style='display:block; margin:12px 0 6px; color:{self.COLORS['scm']}; font-size:15px;'>🟢 SCMs Ranges</b>"),
            self.sf_rng, self.gg_rng, self.fa_rng, self.cc_rng, self.ls_rng,
            self.time_quick,
            self.time_in
        ], layout=w.Layout(gap="6px"))
        
        materials_sec = self._collapsible_section(
            "📦 Materials",
            materials_grid, 
            collapsed=False,
            color=self.COLORS['primary']
        )

        # 简要说明
        info_html = w.HTML(f"""
            <div style='font-size:13px; line-height:1.35;'>
                <b>Defaults</b>: cement 100 g (gypsum 4 g + clinkers + SCMs); w/c 0.50; temperature 25&nbsp;°C.<br/>
                <b>Elastic modulus (cement paste)</b>: estimated via the <i>Mari Tanaka</i> model,
                considering the volume fractions of hydrated and unhydrated products.<br/>
                <b>Max CO₂ uptake</b>: computed from the mass of carbonateable hydrates and CO₂ stoichiometry
                (theoretical full carbonation). Example:
                <code>Ca(OH)₂ + CO₂ → CaCO₃ + H₂O</code>.
            </div>
        """)
        info_sec = self._collapsible_section(
            "ℹ️ Notes",
            info_html,
            collapsed=False,
            color=self.COLORS['info']
        )

        # 左侧侧边栏
        sidebar = w.VBox(
            [materials_sec, info_sec],
            layout=w.Layout(
                width="360px",
                min_width="360px",
                max_width="360px",
                border=f"2px solid {self.COLORS['border']}",
                padding="8px",
                border_radius="12px",
                overflow_y="auto",
                max_height="80vh",
                gap="6px",
                background_color="white",
                box_shadow="0 2px 8px rgba(0,0,0,0.1)"
            )
        )

        # --------- Material Factors 内容 ----------
        co2_col = w.VBox(
            [w.HTML(f"<b style='color:{self.COLORS['danger']};'>🏭 CO₂ Emission (kg/kg)</b>")] +
            [self.co2_sliders[key] for _, key, _, _ in self.MATERIALS],
            layout=w.Layout(gap="2px")
        )
        cost_col = w.VBox(
            [w.HTML(f"<b style='color:{self.COLORS['warning']};'>💰 Cost (€/kg)</b>")] +
            [self.cost_sliders[key] for _, key, _, _ in self.MATERIALS],
            layout=w.Layout(gap="2px")
        )
        material_factors_tab = w.VBox(
            [
                w.HTML(
                    f"<div style='margin-bottom:6px; padding:10px; background:{self.COLORS['light']}; "
                    f"border-radius:6px; font-weight:bold;'>📦 Material CO₂ & Cost Factors</div>"
                ),
                w.HBox([co2_col, cost_col], layout=w.Layout(gap="12px")),
            ],
            layout=w.Layout(padding="8px")
        )

        # --------- Optimization Settings 内容 ----------
        # 自定义参数勾选
        toggle_row = w.HBox(
            [self.use_custom_nsga],
            layout=w.Layout(align_items="center", padding="0 0 6px 0")
        )

        # 具体参数框（初始隐藏）
        nsga_params_box = w.VBox(
            [self.ga_pop, self.ga_gen, self.ga_pc, self.ga_pm, self.ga_seed],
            layout=w.Layout(gap="6px", display="none")
        )

        # 标题 + 运行按钮
        nsga_params = w.VBox(
            [
                w.HTML(f"<b style='color:{self.COLORS['params']};font-size:15px;'>⚙️ NSGA-II Parameters</b>"),
                toggle_row,
                nsga_params_box,
                self.btn_nsga,
                self.run_status,
            ],
            layout=w.Layout(
                gap="6px",
                padding="8px",
                border=f"2px solid {self.COLORS['params']}",
                border_radius="8px",
                background_color="#f8f9fa"
            )
        )

        # 勾选联动显示/隐藏
        def _toggle_nsga(change):
            nsga_params_box.layout.display = "block" if change["new"] else "none"

        self.use_custom_nsga.observe(_toggle_nsga, names="value")

        opt_settings_tab = w.VBox(
            [
                w.HTML(
                    f"<div style='font-weight:bold; margin-bottom:6px; padding:10px; "
                    f"background:{self.COLORS['light']}; border-radius:6px;'>Configure optimization settings</div>"
                ),
            w.HBox(
                [
                    w.VBox([self.obj_box_group], layout=w.Layout(width="40%")),
                    w.VBox([nsga_params], layout=w.Layout(width="50%"))
                ],
                layout=w.Layout(gap="20px")
            )

            ],
            layout=w.Layout(padding="8px", gap="12px")
        )

        # --------- Results 内容 ----------
        results_header = w.HTML("<b>Results</b>")
        divider = w.HTML("<hr style='margin:8px 0;'>")
        results_tab = w.VBox(
            [
                results_header,
                self.download_link,   # 下载链接
                divider,
                self.results_tabs     # ← 用 Tab 替代原来的 result_out
            ],
            layout=w.Layout(padding="8px", gap="6px")
        )



        # --------- 给三个区域加“标题栏” ----------
        material_factors_panel = self._collapsible_section(
            "📊 Material Factors",
            material_factors_tab,
            collapsed=False,
            color=self.COLORS['primary']
        )
        
        optimization_panel = self._collapsible_section(
            "⚙️ Optimization",
            opt_settings_tab,
            collapsed=False,
            color=self.COLORS['params']
        )


        results_panel = self._collapsible_section(
            "📊 Results",
            results_tab,
            collapsed=False,               # 你想默认折叠的话改成 True
            color=self.COLORS['info']
        )


        # 宽度设置，方便排版
        material_factors_panel.layout.width = "50%"
        optimization_panel.layout.width = "50%"
        results_panel.layout.width = "50%"

        # 上面：左 Material Factors，右 Optimization
        top_row = w.HBox(
            [material_factors_panel, results_panel],
            layout=w.Layout(
                width="100%",
                gap="12px",
                align_items="flex-start"
            )
        )

        # 下面：左 Results，右预留空白
        # --------- Pareto vs OPC Compare 内容 ----------
        compare_form = w.VBox(
            [
                w.HTML("<b>Choose a Pareto mix and an OPC cement to compare:</b>"),
                self.compare_pareto_dropdown,
                self.compare_metric_dropdown,   # ← 加这一行
                self.opc_name,

                w.HBox([self.opc_E, self.opc_CO2], layout=w.Layout(gap="8px")),
                w.HBox([self.opc_cost, self.opc_net], layout=w.Layout(gap="8px")),
                self.btn_compare,
                self.compare_out
            ],
            layout=w.Layout(gap="6px", padding="4px")
        )

        compare_panel = self._collapsible_section(
            "🧪 Compare Pareto mix vs OPC",
            compare_form,
            collapsed=True,   # 默认折叠起来
            color=self.COLORS['info']
        )
        compare_panel.layout.width = "50%"

        # 下面：左 Results，右 Compare
        bottom_row = w.HBox(
            [
               optimization_panel,
                compare_panel
            ],
            layout=w.Layout(
                width="100%",
                gap="12px",
                align_items="flex-start"
            )
        )


        center_card = w.VBox(
            [top_row, bottom_row],
            layout=w.Layout(
                border=f"2px solid {self.COLORS['border']}",
                padding="12px",
                border_radius="12px",
                gap="12px",
                width="100%",
                min_height="70vh",
                background_color="white",
                box_shadow="0 2px 8px rgba(0,0,0,0.1)"
            )
        )

        layout = w.HBox(
            [sidebar, center_card],
            layout=w.Layout(width="100%", gap="14px", align_items="flex-start")
        )
        page = w.VBox(
            [title, layout],
            layout=w.Layout(width="100%", padding="6px 8px")
        )
        display(page)



    def _plot_pareto_front(self, df, df_pareto, objectives_dict, title="Pareto Front",
                           show_all_points=True, all_points_alpha=0.3):

    
        keys = list(objectives_dict.keys())
        axis_labels = {
            "E": "E (GPa)",
            "CO2_abs": "CO₂ uptake (kg/kg)",
            "CO2_emission": "CO₂ emission (kg/kg)",
            "Cost": "Cost (€/kg)",
            "Net_emission": "Net emission (kg/kg)"
        }
    
        pareto_color = self.COLORS['danger']
        all_points_color = self.COLORS['primary']
        FIG_W, FIG_H, MARGIN = 550, 320, 20
    
        # --- 1) Build empty figure
        fig = go.Figure()
    
        # --- 2) Background points (always先画它)
        if show_all_points:
            if len(keys) == 1:
                yk = keys[0]
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[yk],
                    mode="markers",
                    name="All points",
                    marker=dict(
                        size=7, color=all_points_color,
                        opacity=float(all_points_alpha),
                        line=dict(width=0.8, color="rgba(0,0,0,0.35)")
                    ),
                    hoverinfo="x+y"
                ))
            elif len(keys) == 2:
                xk, yk = keys
                fig.add_trace(go.Scatter(
                    x=df[xk], y=df[yk],
                    mode="markers",
                    name="All points",
                    marker=dict(
                        size=7, color=all_points_color,
                        opacity=float(all_points_alpha),
                        line=dict(width=0.8, color="rgba(0,0,0,0.35)")
                    ),
                    hoverinfo="x+y"
                ))
            else:
                xk, yk, zk = keys[:3]
                fig.add_trace(go.Scatter3d(
                    x=df[xk], y=df[yk], z=df[zk],
                    mode="markers",
                    name="All points",
                    marker=dict(
                        size=4, color=all_points_color,
                        opacity=float(all_points_alpha)
                    )
                ))
    
        # --- 3) Pareto front (最后加；更醒目)
        if len(keys) == 1:
            yk = keys[0]
            fig.add_trace(go.Scatter(
                x=df_pareto.index, y=df_pareto[yk],
                mode="markers",
                name="Pareto Front",
                marker=dict(
                    size=10, color=pareto_color, symbol="diamond",
                    line=dict(width=1.8, color="black")
                ),
                hoverinfo="x+y"
            ))
            fig.update_yaxes(title=axis_labels.get(yk, yk))
    
        elif len(keys) == 2:
            xk, yk = keys
            fig.add_trace(go.Scatter(
                x=df_pareto[xk], y=df_pareto[yk],
                mode="markers",
                name="Pareto Front",
                marker=dict(
                    size=10, color=pareto_color, symbol="diamond",
                    line=dict(width=1.8, color="black")
                ),
                hoverinfo="x+y"
            ))
            fig.update_xaxes(title=axis_labels.get(xk, xk))
            fig.update_yaxes(title=axis_labels.get(yk, yk))
    
        else:
            xk, yk, zk = keys[:3]
            fig.add_trace(go.Scatter3d(
                x=df_pareto[xk], y=df_pareto[yk], z=df_pareto[zk],
                mode="markers",
                name="Pareto Front",
                marker=dict(
                    size=6, color=pareto_color, symbol="diamond",
                    line=dict(width=1.2, color="black")
                )
            ))
            fig.update_scenes(
                xaxis_title=axis_labels.get(xk, xk),
                yaxis_title=axis_labels.get(yk, yk),
                zaxis_title=axis_labels.get(zk, zk),
            )
    
        # --- 4) Layout
        fig.update_layout(
            title=title,
            template="plotly_white",
            width=FIG_W, height=FIG_H,
            margin=dict(l=MARGIN, r=MARGIN, t=60, b=MARGIN),
            showlegend=True,
            legend=dict(itemsizing="constant")
        )
    
        # 边框
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)",
            layer="below"  # ✅ 形状放到最下层，避免遮挡
        )
    
        # --- 5) 关键：把 Pareto trace 放到最顶层
        fig.data = tuple(
            [t for t in fig.data if t.name != "Pareto Front"] +
            [t for t in fig.data if t.name == "Pareto Front"]
        )
    
        fig.show()


    def _on_nsga_click(self, btn):
        # 清空输出
        # 清空输出
        with self.log_out:
            clear_output(wait=True)
        with self.plot_out:
            clear_output(wait=True)
        with self.table_out:
            clear_output(wait=True)

    
        # ⏳ 开始：禁用按钮 + 英文提示
        self.btn_nsga.disabled = True
        _old_desc = self.btn_nsga.description
        self.btn_nsga.description = "⏳ Simulating... please wait"
        self.run_status.value = (
            "<div style='padding:6px 10px; border:1px solid #C9C9C9; "
            "border-radius:8px; background:#FFFBEA; font-size:13px;'>"
            "⏳ <b>Simulation in progress — please wait...</b></div>"
        )


        try:
            with self.log_out:
                objectives = [key for key, cb in self.objective_boxes.items() if cb.value]
                if not objectives:
                    print("❌ ERROR: Please select at least one objective.")
                    return

                clinker_bounds = {
                    "C3S": tuple(self.c3s_rng.value),
                    "C2S": tuple(self.c2s_rng.value),
                    "C3A": tuple(self.c3a_rng.value),
                    "C4AF": tuple(self.c4af_rng.value),
                }
                scms_bounds = {
                    "silica_fume": tuple(self.sf_rng.value),
                    "GGBFS": tuple(self.gg_rng.value),
                    "fly_ash": tuple(self.fa_rng.value),
                    "calcined_clay": tuple(self.cc_rng.value),
                    "limestone": tuple(self.ls_rng.value),
                }
                clinker_sum_rng = tuple(self.cl_sum_rng.value)
                time_val = int(self.time_in.value)
                emis, cost = self._read_factors()

                # === GA 超参数：若未勾选自定义，则使用默认值；勾选后用滑块值 ===
                if bool(self.use_custom_nsga.value):
                    pop = int(self.ga_pop.value)
                    gen = int(self.ga_gen.value)
                    pc  = float(self.ga_pc.value)
                    pm  = float(self.ga_pm.value)
                    seed = int(self.ga_seed.value)
                else:
                    pop = 160     # 默认值
                    gen = 120
                    pc  = 0.90
                    pm  = 0.10
                    seed = 12345


                eta_c = 15.0
                eta_m = 20.0

                problem = ConcreteMixProblem(
                    model=self.model,
                    metrics_calc=self.metrics,
                    clinker_bounds=clinker_bounds,
                    scms_bounds=scms_bounds,
                    clinker_sum_rng=clinker_sum_rng,
                    total_binder_target=self.TOTAL_BINDER_TARGET,
                    time_val=time_val,
                    emission_factors=emis,
                    cost_factors=cost,
                    objectives=objectives
                )

                algorithm = NSGA2(
                    pop_size=pop,
                    sampling=FloatRandomSampling(),
                    crossover=SBX(prob=pc, eta=eta_c),
                    mutation=PM(prob=pm, eta=eta_m),
                    eliminate_duplicates=True,
                )
                termination = get_termination("n_gen", gen)

                print(f"🚀 Running NSGA-II: pop={pop}, gen={gen}, seed={seed}")
                show_all = bool(self.show_all.value)
                res = minimize(problem, algorithm, termination=termination, seed=seed,
                               save_history=show_all, verbose=True)

                X = res.X
                mixes = problem.decode(X)
                preds = self.model.predict(mixes)
                df_pareto = self.metrics.add_metrics(mixes, preds[:, 0], preds[:, 1], emis, cost)

                if show_all and hasattr(res, 'history') and res.history:
                    all_mixes = []
                    for h in res.history:
                        X_hist = h.pop.get("X")
                        if X_hist is not None and len(X_hist) > 0:
                            mixes_hist = problem.decode(X_hist)
                            all_mixes.append(mixes_hist)
                    if all_mixes:
                        df_all = pd.concat(all_mixes, ignore_index=True)
                        preds_all = self.model.predict(df_all)
                        df_all = self.metrics.add_metrics(df_all, preds_all[:, 0], preds_all[:, 1], emis, cost)
                    else:
                        df_all = df_pareto
                else:
                    df_all = df_pareto

                print(f"✅ [NSGA-II] Found {len(df_pareto)} Pareto-optimal solutions.")
                                # 保存 Pareto 结果供后续比较使用
                self.df_pareto = df_pareto.reset_index(drop=True)

                # 为比较部分构建下拉选项
                options = []
                for idx, row in self.df_pareto.iterrows():
                    label = (
                        f"#{idx+1}  "
                        f"E={row['E']:.2f} GPa, "
                        f"Net={row['Net_emission']:.2f}, "
                        f"CO₂={row['CO2_emission']:.2f}"
                    )
                    options.append((label, idx))
                self.compare_pareto_dropdown.options = options
                if options:
                    self.compare_pareto_dropdown.value = options[0][1]


            # 图放到 plot_out 这个 tab 里
            with self.plot_out:
                objectives_dict = {}
                for obj in objectives:
                    if obj == "E_max":
                        objectives_dict["E"] = df_all["E"]
                    elif obj == "CO2abs_max":
                        objectives_dict["CO2_abs"] = df_all["CO2_abs"]
                    elif obj == "CO2_min":
                        objectives_dict["CO2_emission"] = df_all["CO2_emission"]
                    elif obj == "Cost_min":
                        objectives_dict["Cost"] = df_all["Cost"]
                    elif obj == "Net_min":
                        objectives_dict["Net_emission"] = df_all["Net_emission"]

                alpha = float(self.alpha.value)
                self._plot_pareto_front(
                    df_all, df_pareto, objectives_dict,
                    title="📊 Pareto Front (NSGA-II)",
                    show_all_points=show_all, all_points_alpha=alpha
                )

            # 表格+下载链接放到 table_out 这个 tab 里
            with self.table_out:
                self._display_nsga_results(df_pareto, seed)


            with self.log_out:
                print(f"💾 [NSGA-II] Completed successfully. Results saved to nsga2_pareto_recipes.csv (seed={seed})")
                 # ✅ 成功后给个英文完成提示
            self.run_status.value = (
                "<div style='padding:6px 10px; border:1px solid #C9C9C9; "
                "border-radius:8px; background:#F2FFF2; font-size:13px;'>"
                "✅ <b>Done.</b> You can review the results in Results section.</div>"
            )

        except Exception as e:
            with self.log_out:
                print("❌ ERROR (NSGA-II):", e)
                traceback.print_exc(file=sys.stdout)
        finally:
            # 🔁 无论成功失败都恢复按钮
            self.btn_nsga.disabled = False
            self.btn_nsga.description = _old_desc
    def _on_compare_click(self, btn):
        from IPython.display import display
        import plotly.express as px
        # OPC 的值映射
        self.opc_metrics_map = {
            "E": self.opc_E.value,
            "CO2_emission": self.opc_CO2.value,
            "Cost": self.opc_cost.value,
            "Net_emission": self.opc_net.value,
        }


        with self.compare_out:
            clear_output(wait=True)

            if self.df_pareto is None or not len(self.compare_pareto_dropdown.options):
                print("Please run NSGA-II first to generate Pareto mixes.")
                return

            idx = self.compare_pareto_dropdown.value
            row = self.df_pareto.loc[idx]

            # 构造对比数据：OPC vs Pareto
            metrics = [
                ("E (GPa)",          "E",             self.opc_E.value),
                ("CO₂ emission",     "CO2_emission",  self.opc_CO2.value),
                ("Cost (€/kg)",      "Cost",          self.opc_cost.value),
                ("Net emission",     "Net_emission",  self.opc_net.value),
            ]

            data = []
            deltas = []
            for label, col, opc_val in metrics:
                pareto_val = float(row[col])
                opc_val = float(opc_val)
                data.append({"Metric": label, "Type": self.opc_name.value, "Value": opc_val})
                data.append({"Metric": label, "Type": "Pareto mix", "Value": pareto_val})

                # 百分比变化（Pareto 相对 OPC）
                if opc_val != 0:
                    delta_pct = (pareto_val - opc_val) / opc_val * 100.0
                else:
                    delta_pct = float('nan')
                deltas.append((label, pareto_val - opc_val, delta_pct))

            df_plot = pd.DataFrame(data)

            # 分组柱状图
            # 取得选择的指标
            metric_col = self.compare_metric_dropdown.value
            metric_label = {
                "E": "E (GPa)",
                "CO2_emission": "CO₂ emission (kg/kg)",
                "Cost": "Cost (€/kg)",
                "Net_emission": "Net emission (kg/kg)"
            }[metric_col]
            
            # 数据表
            df_plot = pd.DataFrame({
                "Type": [self.opc_name.value, "Pareto mix"],
                "Value": [float(self.opc_metrics_map[metric_col]), float(row[metric_col])],
            })
            
            # 不同类型使用不同颜色
            color_map = {
                self.opc_name.value: self.COLORS["primary"],   # 比如蓝灰
                "Pareto mix":        self.COLORS["danger"],    # 比如暗红
            }
            
            fig = px.bar(
                df_plot,
                x="Type",
                y="Value",
                color="Type",
                color_discrete_map=color_map,
                title=f"{metric_label}: Pareto mix vs {self.opc_name.value}"
            )
            
            fig.update_layout(width=550, height=320)
            fig.update_yaxes(title=metric_label)
            
            # 柱子边框稍微描一下，更有层次感
            fig.update_traces(marker_line_width=1.2, marker_line_color="rgba(0,0,0,0.35)")
            
            fig.show()



            # 打印文字总结
            # ===== 美化文字报告 =====

            display(HTML(
    f"<div style='font-weight:bold; font-size:15px; margin-bottom:6px;'>"
    f"📘 Comparison Report — Pareto mix #{idx+1} vs {self.opc_name.value}"
    f"</div>"
))
            print("────────────────────────────────────────────────────────\n")

            for label, diff_abs, diff_pct in deltas:

                # 趋势箭头
                if diff_abs > 0:
                    arrow = "↑"
                    trend = "higher"
                elif diff_abs < 0:
                    arrow = "↓"
                    trend = "lower"
                else:
                    arrow = "→"
                    trend = "no change"

                # 趋势增强词
                if abs(diff_pct) > 80:
                    intensity = " (significantly {})".format(trend)
                elif abs(diff_pct) > 30:
                    intensity = " ({} noticeably)".format(trend)
                else:
                    intensity = ""

                print(f"• {label}")
                print(f"    Trend:       {arrow}  ({trend}{intensity})")
                print(f"    Δ value:     {diff_abs:.3f}")
                if not pd.isna(diff_pct):
                    print(f"    Δ percent:   {diff_pct:+.1f} %")
                print()

            print("────────────────────────────────────────────────────────")


    def _display_nsga_results(self, dfm, seed):
        cols = [
            "C3S", "C2S", "C3A", "C4AF",
            "silica_fume", "GGBFS", "fly_ash", "calcined_clay", "limestone",
            "E", "CO2_abs", "CO2_emission", "Cost", "Net_emission",
            "w/c_fixed", "gypsum_fixed_%", "temp_fixed_C"
        ]
        df_out = dfm[cols].copy()
        df_out.insert(0, "seed", seed)

        # 计算 SCM 总量
        scm_cols = ["silica_fume", "GGBFS", "fly_ash", "calcined_clay", "limestone"]
        df_out["SCM (%)"] = df_out[scm_cols].sum(axis=1)

        # --------- 四舍五入设置 ---------
        # 配合比（熟料+SCM+SCM总量）保留 1 位小数
        mix_cols = ["C3S", "C2S", "C3A", "C4AF",
                    "silica_fume", "GGBFS", "fly_ash",
                    "calcined_clay", "limestone", "SCM (%)"]

        # 性能指标保留 2 位小数
        metric_cols = ["E", "CO2_abs", "CO2_emission", "Cost", "Net_emission"]

        # 固定参数也统一 2 位（可选）
        fixed_cols = ["w/c_fixed", "gypsum_fixed_%", "temp_fixed_C"]

        df_rounded = df_out.copy()
        df_rounded[mix_cols] = df_rounded[mix_cols].round(1)
        df_rounded[metric_cols] = df_rounded[metric_cols].round(2)
        df_rounded[fixed_cols] = df_rounded[fixed_cols].round(2)

        # 列名美化
        pretty = {
            "C3S": "C3S (%)", "C2S": "C2S (%)", "C3A": "C3A (%)", "C4AF": "C4AF (%)",
            "silica_fume": "Silica fume (%)", "GGBFS": "GGBFS (%)", "fly_ash": "Fly ash (%)",
            "calcined_clay": "Calcined clay (%)", "limestone": "Limestone (%)",
            "SCM (%)": "SCM (%)",
            "E": "E (GPa)", "CO2_abs": "CO₂ uptake (kg/kg)", "CO2_emission": "CO₂ emission (kg/kg)",
            "Cost": "Cost (€/kg)", "Net_emission": "Net emission (kg/kg)",
            "w/c_fixed": "w/c", "gypsum_fixed_%": "Gypsum (%)", "temp_fixed_C": "Temperature (°C)",
            "seed": "Seed"
        }
        view = df_rounded.rename(columns=pretty)

        # 显示格式：配合比 1 位小数，指标 2 位小数
        fmt = {
            "C3S (%)": "{:.1f}",
            "C2S (%)": "{:.1f}",
            "C3A (%)": "{:.1f}",
            "C4AF (%)": "{:.1f}",
            "Silica fume (%)": "{:.1f}",
            "GGBFS (%)": "{:.1f}",
            "Fly ash (%)": "{:.1f}",
            "Calcined clay (%)": "{:.1f}",
            "Limestone (%)": "{:.1f}",
            "SCM (%)": "{:.1f}",
            "E (GPa)": "{:.2f}",
            "CO₂ uptake (kg/kg)": "{:.2f}",
            "CO₂ emission (kg/kg)": "{:.2f}",
            "Cost (€/kg)": "{:.2f}",
            "Net emission (kg/kg)": "{:.2f}",
            "w/c": "{:.2f}",
            "Gypsum (%)": "{:.2f}",
            "Temperature (°C)": "{:.2f}",
        }

        from IPython.display import display, FileLink

        # 先更新下载链接（显示在 Results 面板顶部）
        df_rounded.to_csv("nsga2_pareto_recipes.csv", index=False)
        
        link = FileLink(
            "nsga2_pareto_recipes.csv",
            result_html_prefix="<span style='font-weight:bold;'>📥 Download results (CSV)</span>: "
        )
        
        self.download_link.value = link._repr_html_()


        # 再显示前 5 条结果表格
        try:
            display(view.head(5).style.format(fmt).hide_index())
        except Exception:
            display(view.head(5))


