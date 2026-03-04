from fpdf import FPDF
import pandas as pd
import numpy as np
import os

def calculate_topsis(df: pd.DataFrame, objective_config: list) -> np.ndarray:
    """计算 TOPSIS 决策分数"""
    criteria_cols = [obj['col'] for obj in objective_config if obj.get('col') in df.columns]
    impacts = [obj['impact'] for obj in objective_config if obj.get('col') in df.columns]

    if not criteria_cols:
        return np.zeros(len(df))

    d = df[criteria_cols].astype(float).copy()
    denom = np.sqrt((d ** 2).sum(axis=0)).replace(0, np.nan)
    norm_d = (d / denom).fillna(0.0)

    n = len(criteria_cols)
    weighted_d = norm_d * (1.0 / n)

    ideal_best = []
    ideal_worst = []
    for i, col in enumerate(criteria_cols):
        if impacts[i] == '+':
            ideal_best.append(weighted_d[col].max())
            ideal_worst.append(weighted_d[col].min())
        else:
            ideal_best.append(weighted_d[col].min())
            ideal_worst.append(weighted_d[col].max())

    wd = weighted_d.values
    s_plus = np.sqrt(((wd - np.array(ideal_best)) ** 2).sum(axis=1))
    s_minus = np.sqrt(((wd - np.array(ideal_worst)) ** 2).sum(axis=1))

    return np.divide(s_minus, (s_plus + s_minus), out=np.zeros_like(s_minus), where=(s_plus + s_minus) != 0)

class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(15, 118, 110) # 墨绿色主题
        self.rect(0, 0, 210, 20, 'F')
        self.set_font('Arial', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'Technical Audit: Optimized Cement Mix Designs', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(15, 118, 110)
        self.cell(0, 8, f" {label}", 0, 1, 'L', 1)
        self.ln(2)

    def info_row(self, label, value):
        self.set_font('Arial', 'B', 8)
        self.set_text_color(50, 50, 50)
        self.cell(45, 6, f"{label}:", 0, 0)
        self.set_font('Arial', '', 8)
        self.cell(0, 6, f"{value}", 0, 1)

def create_pdf_report(df_pareto, params, baseline_data, objective_config, search_bounds, ga_settings):
    # 1. 计算评分并排序
    df = df_pareto.copy()
    df['Score'] = calculate_topsis(df, objective_config)
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    pdf = PDFReport()
    pdf.add_page()
    
    # --- Section 1: 基本参数 ---
    pdf.chapter_title("1. Engineering Context & Constraints")
    for k, v in params.items():
        pdf.info_row(k, v)
    pdf.ln(2)

    # --- Section 2: 搜索边界 ---
    pdf.chapter_title("2. Material Design Space (Search Boundaries)")
    pdf.set_font('Arial', 'B', 8)
    pdf.set_fill_color(235, 235, 235)
    pdf.cell(70, 6, "Component", 1, 0, 'C', 1)
    pdf.cell(60, 6, "Min Bound", 1, 0, 'C', 1)
    pdf.cell(60, 6, "Max Bound", 1, 1, 'C', 1)
    
    pdf.set_font('Arial', '', 8)
    for mat, rng in search_bounds['clinker'].items():
        pdf.cell(70, 6, f"Clinker phase: {mat}", 1, 0)
        pdf.cell(60, 6, f"{rng[0]:.1f}%", 1, 0, 'C')
        pdf.cell(60, 6, f"{rng[1]:.1f}%", 1, 1, 'C')
    for mat, rng in search_bounds['scms'].items():
        pdf.cell(70, 6, f"SCM: {mat}", 1, 0)
        pdf.cell(60, 6, f"{rng[0]:.1f} g", 1, 0, 'C')
        pdf.cell(60, 6, f"{rng[1]:.1f} g", 1, 1, 'C')
    pdf.ln(4)

    # --- Section 3: 优化目标 ---
    pdf.chapter_title("3. Optimization Objectives (NSGA-II)")
    for obj in objective_config:
        impact = "Maximize" if obj['impact'] == '+' else "Minimize"
        pdf.info_row("Objective", f"{obj['name']} ({impact})")
    pdf.ln(4)

    # --- Section 4: 帕累托最优配合比表格 (核心修改内容) ---
    pdf.chapter_title("4. Pareto Optimal Mix Proportions (Top Ranked)")
    
    # 定义表格要展示的列（熟料相+主要SCMs+核心指标）
    # 这里的列名需与 app.py 中的 MATERIALS_CONFIG 对应
    mat_cols = ["C3S", "C2S", "silica_fume", "GGBFS", "fly_ash", "calcined_clay", "limestone"]
    metric_cols = ["E", "CO2_emission", "Cost"]
    
    # 表格宽度分配
    col_width = 190 / (len(mat_cols) + len(metric_cols) + 1)
    
    # 绘制表头
    pdf.set_fill_color(15, 118, 110)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 7)
    pdf.cell(col_width, 8, "Rank", 1, 0, 'C', 1)
    for c in mat_cols + metric_cols:
        pdf.cell(col_width, 8, c[:6], 1, 0, 'C', 1) # 截断长名称
    pdf.ln(8)
    
    # 填充数据（仅展示前 15 名以防溢出页面）
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 7)
    for i in range(min(15, len(df_sorted))):
        row = df_sorted.iloc[i]
        # 背景颜色交替
        fill = i % 2 == 1
        if fill: pdf.set_fill_color(245, 245, 245)
        
        pdf.cell(col_width, 7, f"#{i+1}", 1, 0, 'C', fill)
        for c in mat_cols:
            pdf.cell(col_width, 7, f"{row[c]:.1f}", 1, 0, 'C', fill)
        for c in metric_cols:
            fmt = "%.2f" if c == "E" else "%.3f"
            pdf.cell(col_width, 7, fmt % row[c], 1, 0, 'C', fill)
        pdf.ln(7)

    pdf.ln(5)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "Note: Clinker phases are in % of total clinker, SCMs are in grams per 100g binder. Performance metrics: E (GPa), CO2 (kg/kg), Cost (EUR/kg).")

    return bytes(pdf.output())