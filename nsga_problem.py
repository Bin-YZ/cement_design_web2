
import numpy as np
import pandas as pd

try:
    from pymoo.core.problem import Problem
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    Problem = object  # Dummy class


class ConcreteMixProblem(Problem):
    """Pymoo problem for multi-objective concrete mix optimization
    —— 归一化 + 显式约束（越界即不可行）
    """

    def __init__(self, model, metrics_calc, clinker_bounds, scms_bounds,
                 clinker_sum_rng, total_binder_target, time_val,
                 emission_factors, cost_factors, objectives,
                 sum_tol=1e-2, big_penalty=1e6):
        self.model = model
        self.metrics_calc = metrics_calc
        self.clinker_bounds = clinker_bounds
        self.scms_bounds = scms_bounds
        self.clinker_sum_rng = clinker_sum_rng
        self.total_binder_target = total_binder_target
        self.time_val = time_val
        self.emission_factors = emission_factors
        self.cost_factors = cost_factors
        self.objectives = objectives
        self.sum_tol = float(sum_tol)
        self.big_penalty = float(big_penalty)

        # 变量顺序： [clinker_sum] + clinker phases... + scms...
        self.clinker_names = list(clinker_bounds.keys())
        self.scms_names = list(scms_bounds.keys())
        n_vars = 1 + len(self.clinker_names) + len(self.scms_names)

        xl = [clinker_sum_rng[0]]
        xu = [clinker_sum_rng[1]]
        for name in self.clinker_names:
            lo, hi = clinker_bounds[name]
            xl.append(lo)
            xu.append(hi)
        for name in self.scms_names:
            lo, hi = scms_bounds[name]
            xl.append(lo)
            xu.append(hi)

        # 约束数量：
        #   - 每个 SCM 两个（下/上限）
        #   - 每个熟料相一个（绝对下限）
        #   - 总和约束一个（等于 total_binder_target）
        n_constr = 2 * len(self.scms_names) + len(self.clinker_names) + 1

        super().__init__(n_var=n_vars, n_obj=len(objectives), n_constr=n_constr,
                         xl=np.array(xl, dtype=float), xu=np.array(xu, dtype=float))

    # ---------- 解码：仅归一化，不做投影修复 ----------
    def decode(self, X):
        """返回 DataFrame（与现有 ModelWrapper 接口兼容）"""
        X = np.atleast_2d(X)
        rows = []

        for x in X:
            clinker_sum = float(x[0])
            scms_sum = float(self.total_binder_target - clinker_sum)

            idx = 1
            # 熟料相：先归一化到 1，再乘以 clinker_sum 得到绝对量
            clinker_raw = np.array(x[idx:idx + len(self.clinker_names)], dtype=float)
            idx += len(self.clinker_names)
            if clinker_raw.sum() <= 0:
                clinker_raw = np.ones_like(clinker_raw)
            clinker_vals_abs = (clinker_raw / clinker_raw.sum()) * clinker_sum

            # SCM：归一化到 1，再乘以 scms_sum 得到绝对量
            scms_raw = np.array(x[idx:idx + len(self.scms_names)], dtype=float)
            if scms_raw.sum() <= 0:
                scms_raw = np.ones_like(scms_raw)
            scms_vals_abs = (scms_raw / scms_raw.sum()) * scms_sum

            row = {}
            for i, n in enumerate(self.clinker_names):
                row[n] = clinker_vals_abs[i]
            for i, n in enumerate(self.scms_names):
                row[n] = scms_vals_abs[i]
            row["time"] = self.time_val
            rows.append(row)

        return pd.DataFrame(rows)

    # ---------- 约束：G <= 0 可行 ----------
    def _constraints(self, mixes: pd.DataFrame):
        """
        顺序：
          [SCM_lo..., SCM_hi..., ClinkerPhase_abs_lo..., TotalSum_eq]
        """
        n = len(mixes)
        G_list = []

        # 1) 各 SCM 的绝对下/上限
        for name in self.scms_names:
            lo, hi = self.scms_bounds[name]
            vals = mixes[name].values
            G_list.append(np.maximum(0.0, lo - vals))   # 低于下限
        for name in self.scms_names:
            lo, hi = self.scms_bounds[name]
            vals = mixes[name].values
            G_list.append(np.maximum(0.0, vals - hi))   # 高于上限

        # 2) 各熟料相的绝对下限（用熟料总量下限推导的绝对需求）
        clinker_min = float(self.clinker_sum_rng[0])
        for name in self.clinker_names:
            lo_pct = self.clinker_bounds[name][0]  # %
            required_abs = lo_pct / 100.0 * clinker_min
            vals = mixes[name].values
            G_list.append(np.maximum(0.0, required_abs - vals))

        # 3) 总和约束：∑(熟料+SCM) == total_binder_target（容差）
        sum_vals = mixes[self.clinker_names + self.scms_names].sum(axis=1).values
        total_violation = np.maximum(0.0, np.abs(sum_vals - self.total_binder_target) - self.sum_tol)
        G_list.append(total_violation)

        return np.vstack(G_list).T  # (n, n_constr)

    # ---------- 目标评估 ----------
    def _evaluate(self, X, out, *args, **kwargs):
        mixes = self.decode(X)          # DataFrame —— 与 ModelWrapper 兼容
        G = self._constraints(mixes)    # 约束矩阵
        out["G"] = G

        preds = self.model.predict(mixes)  # 例如返回 [E, CO2_abs]
        df = self.metrics_calc.add_metrics(
            mixes, preds[:, 0], preds[:, 1],
            self.emission_factors, self.cost_factors
        )

        F_cols = []
        for obj in self.objectives:
            if obj == "E_max":
                F_cols.append(-df["E"].values)
            elif obj == "CO2abs_max":
                F_cols.append(-df["CO2_abs"].values)
            elif obj == "CO2_min":
                F_cols.append(df["CO2_emission"].values)
            elif obj == "Cost_min":
                F_cols.append(df["Cost"].values)
            elif obj == "Net_min":
                F_cols.append(df["Net_emission"].values)
            else:
                raise ValueError(f"Unknown objective tag: {obj}")

        F = np.column_stack(F_cols)

        # 双保险：若算法设置未启用约束处理，对不可行解添加大惩罚
        infeasible = (G > 0).any(axis=1)
        if infeasible.any():
            F[infeasible, :] = F[infeasible, :] + self.big_penalty

        out["F"] = F
