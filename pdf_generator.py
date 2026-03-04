from fpdf import FPDF
import pandas as pd
import numpy as np
import os

# ✅ 把这些符号换成 FPDF(Arial) 更稳的版本： ->  =>  EUR
REPORT_NOTES = [
    ("ANN Model inputs",
     "clinker phase (C3S / C2S / C3A / C4AF, g), SCMs (Silica fume / GGBFS / Fly ash / Calcined clay / Limestone, g), curing time (days)."),
    ("ANN Model outputs",
     "E-modulus (GPa) + max CO2 uptake (kg/kg). The E-modulus is derived by applying micromechanical homogenization to the GEMS-simulated hydrate assemblages. The ANN is then trained on these datasets."),
    ("Recipe mass balance & limits",
     "100 g binder = 96 g (clinker + SCMs) + 4 g gypsum; clinker 20-96 g => SCM 0-76 g."),
    ("Panel inputs & factors",
     "set ranges + curing time; CO2 factors default ecoinvent (editable); cost is user-defined (EUR/kg)."),
    ("Quick workflow",
     "set ranges & curing time -> Material Factors -> choose goals -> click START OPTIMIZATION."),
    ("Results modules", [
        "Pareto Analysis: interactive frontier plot showing explored points vs Pareto-optimal trade-offs; highlights the top-ranked mix (TOPSIS).",
        "GA Animation: generation-by-generation visualization of NSGA-II population evolution toward the final frontier.",
        "Parallel Coordinates: multi-dimensional view of compositions and metrics to inspect trade-offs and filter ranges.",
        "Data Table: sortable Pareto solutions with export to CSV, including Decision Score and all metrics/materials.",
        "Benchmark Comparison: compare any optimized mix against a user-defined OPC baseline (deltas + plots).",
        "Technical Export: generate a PDF audit summarizing settings, objectives, Pareto results, and the selected recommendation."
    ]),
    ("Disclaimer",
     "The recommended mixes are model-based screening results and must be validated experimentally before any practical or safety-critical use.")
]

# ✅ 超简单：把容易导致 FPDF 崩的字符替换掉 + 让 / 可断行
def safe(s):
    if s is None:
        return ""
    s = str(s)

    # 1) 替换 Unicode 符号（Arial core font 很容易炸）
    s = (s.replace("→", "->")
           .replace("⇒", "=>")
           .replace("€", "EUR")
           .replace("–", "-")
           .replace("—", "-")
           .replace("“", '"').replace("”", '"')
           .replace("’", "'"))

    # 2) 让斜杠可断行（比如 C3S/C2S/...）
    s = s.replace("/", " / ")

    # 3) 避免极端长串无空格导致无法断行（很少见，但保险）
    if len(s) > 3000:
        s = s[:3000] + " ..."

    return s


def calculate_topsis(df: pd.DataFrame, objective_config: list) -> np.ndarray:
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
        self.set_fill_color(15, 118, 110)
        self.rect(0, 0, 210, 20, 'F')
        self.set_font('Arial', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'Technical Audit: Optimized Cement Mix Designs', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(15, 118, 110)
        self.cell(0, 8, safe(f" {label}"), 0, 1, 'L', 1)
        self.ln(2)

    def info_row(self, label, value):
        self.set_font('Arial', 'B', 8)
        self.set_text_color(50, 50, 50)
        self.cell(45, 6, safe(f"{label}:"), 0, 0)
        self.set_font('Arial', '', 8)
        self.cell(0, 6, safe(f"{value}"), 0, 1)


    def write_kv_block(self, title, content, bullet=False):
        # ✅ 强制回到左边距，避免 x 停在行尾导致 multi_cell 没空间
        self.set_x(self.l_margin)
    
        self.set_text_color(30, 30, 30)
        self.set_font('Arial', 'B', 9)
        self.multi_cell(0, 5, safe(f"{title}:"))
    
        # multi_cell 后通常会回到左边距，但我们再保险一次
        self.set_x(self.l_margin)
        self.set_font('Arial', '', 8)
    
        indent = 4  # ✅ 固定缩进 4mm
    
        if isinstance(content, list):
            for line in content:
                self.set_x(self.l_margin + indent)
                self.multi_cell(0, 5, safe(f"- {line}"))
                self.set_x(self.l_margin)  # 每条后回到左边距
        else:
            if bullet:
                self.set_x(self.l_margin + indent)
                self.multi_cell(0, 5, safe(f"- {content}"))
            else:
                self.set_x(self.l_margin)
                self.multi_cell(0, 5, safe(content))
    
        self.ln(1)


def create_pdf_report(df_pareto, params, baseline_data, objective_config, search_bounds, ga_settings):
    df = df_pareto.copy()
    df['Score'] = calculate_topsis(df, objective_config)
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    pdf = PDFReport()
    pdf.add_page()

    # Section 0
    pdf.chapter_title("0. Model Summary & Workflow Notes")
    for title, body in REPORT_NOTES:
        pdf.write_kv_block(title, body)
    pdf.ln(2)

    # Section 1
    pdf.chapter_title("1. Engineering Context & Constraints")
    for k, v in params.items():
        pdf.info_row(k, v)
    pdf.ln(2)

    # Section 2
    pdf.chapter_title("2. Material Design Space (Search Boundaries)")
    pdf.set_font('Arial', 'B', 8)
    pdf.set_fill_color(235, 235, 235)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(70, 6, "Component", 1, 0, 'C', 1)
    pdf.cell(60, 6, "Min Bound", 1, 0, 'C', 1)
    pdf.cell(60, 6, "Max Bound", 1, 1, 'C', 1)

    pdf.set_font('Arial', '', 8)
    for mat, rng in search_bounds['clinker'].items():
        pdf.cell(70, 6, safe(f"Clinker phase: {mat}"), 1, 0)
        pdf.cell(60, 6, safe(f"{rng[0]:.1f}%"), 1, 0, 'C')
        pdf.cell(60, 6, safe(f"{rng[1]:.1f}%"), 1, 1, 'C')
    for mat, rng in search_bounds['scms'].items():
        pdf.cell(70, 6, safe(f"SCM: {mat}"), 1, 0)
        pdf.cell(60, 6, safe(f"{rng[0]:.1f} g"), 1, 0, 'C')
        pdf.cell(60, 6, safe(f"{rng[1]:.1f} g"), 1, 1, 'C')
    pdf.ln(4)

    # Section 3
    pdf.chapter_title("3. Optimization Objectives (NSGA-II)")
    for obj in objective_config:
        impact = "Maximize" if obj['impact'] == '+' else "Minimize"
        pdf.info_row("Objective", f"{obj['name']} ({impact})")
    pdf.ln(4)

    # Section 4
    pdf.chapter_title("4. Pareto Optimal Mix Proportions (Top Ranked)")

    mat_cols = ["C3S", "C2S", "silica_fume", "GGBFS", "fly_ash", "calcined_clay", "limestone"]
    metric_cols = ["E", "CO2_emission", "Cost"]

    col_width = 190 / (len(mat_cols) + len(metric_cols) + 1)

    pdf.set_fill_color(15, 118, 110)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 7)
    pdf.cell(col_width, 8, "Rank", 1, 0, 'C', 1)
    for c in mat_cols + metric_cols:
        pdf.cell(col_width, 8, safe(c[:6]), 1, 0, 'C', 1)
    pdf.ln(8)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 7)
    for i in range(min(15, len(df_sorted))):
        row = df_sorted.iloc[i]
        fill = i % 2 == 1
        if fill:
            pdf.set_fill_color(245, 245, 245)

        pdf.cell(col_width, 7, safe(f"#{i+1}"), 1, 0, 'C', fill)
        for c in mat_cols:
            pdf.cell(col_width, 7, safe(f"{float(row.get(c, 0.0)):.1f}"), 1, 0, 'C', fill)
        for c in metric_cols:
            val = float(row.get(c, 0.0))
            fmt = "%.2f" if c == "E" else "%.3f"
            pdf.cell(col_width, 7, safe(fmt % val), 1, 0, 'C', fill)
        pdf.ln(7)

    pdf.ln(5)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(
        0, 5,
        safe("Note: Clinker phases are in % of total clinker, SCMs are in grams per 100g binder. "
             "Performance metrics: E (GPa), CO2 (kg/kg), Cost (EUR/kg).")
    )

    # ✅ fpdf2 推荐用 output(dest="S") 拿 bytes（更稳）
    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytearray, bytes)) else out.encode("latin-1", errors="ignore")