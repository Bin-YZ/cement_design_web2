"""
Core optimization and model utilities for the refactored Streamlit app.
"""

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    from pymoo.core.problem import Problem

    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    Problem = object


class ModelWrapper:
    """Wrapper for loading and running the trained Keras model."""

    FEATURES = [
        "C3S",
        "C2S",
        "C3A",
        "time",
        "C4AF",
        "silica_fume",
        "GGBFS",
        "fly_ash",
        "calcined_clay",
        "limestone",
    ]

    @staticmethod
    def _weighted_mse(y_true, y_pred):
        e_err = tf.square(y_true[:, 0] - y_pred[:, 0])
        co2_err = tf.square(y_true[:, 1] - y_pred[:, 1])
        return tf.reduce_mean(e_err + co2_err)

    def __init__(self, model_path: str, suppress_warnings: bool = True):
        self.model_path = model_path

        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model = load_model(
                    model_path,
                    custom_objects={"weighted_mse": ModelWrapper._weighted_mse},
                )
                dummy_input = np.zeros((1, len(self.FEATURES)), dtype="float32")
                _ = self.model.predict(dummy_input, verbose=0)
        else:
            self.model = load_model(
                model_path,
                custom_objects={"weighted_mse": ModelWrapper._weighted_mse},
            )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[ModelWrapper.FEATURES].values.astype("float32")
        return self.model.predict(x, verbose=0)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.model_path:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model = load_model(
                    self.model_path,
                    custom_objects={"weighted_mse": ModelWrapper._weighted_mse},
                )
                dummy_input = np.zeros((1, len(self.FEATURES)), dtype="float32")
                _ = self.model.predict(dummy_input, verbose=0)


class MetricsCalculator:
    """Compute emissions, cost, and derived metrics with consistent units."""

    def __init__(self, wc_fixed: float, gypsum_fixed: float, temp_fixed: float):
        self.wc = float(wc_fixed)
        self.gypsum = float(gypsum_fixed)
        self.temp = float(temp_fixed)

    def add_metrics(self, df, e_pred, co2_abs_pred, emission_factors, cost_factors):
        df = df.copy()
        df["E"] = np.asarray(e_pred, dtype=float)
        df["CO2_abs"] = np.asarray(co2_abs_pred, dtype=float) / 100.0

        materials = [
            "C3S",
            "C2S",
            "C3A",
            "C4AF",
            "silica_fume",
            "GGBFS",
            "fly_ash",
            "calcined_clay",
            "limestone",
        ]

        for material in materials:
            if material not in df.columns:
                df[material] = 0.0

        emis_vec = np.array(
            [float(emission_factors.get(material, 0.0)) for material in materials],
            dtype=float,
        )
        cost_vec = np.array(
            [float(cost_factors.get(material, 0.0)) for material in materials],
            dtype=float,
        )

        kg_per_100g = df[materials].to_numpy(dtype=float) / 1000.0
        co2_per_100g = kg_per_100g @ emis_vec
        cost_per_100g = kg_per_100g @ cost_vec

        gypsum_kg_per_100g = self.gypsum / 1000.0
        co2_per_100g += gypsum_kg_per_100g * float(emission_factors.get("Gypsum", 0.0))
        cost_per_100g += gypsum_kg_per_100g * float(cost_factors.get("Gypsum", 0.0))

        df["CO2_emission"] = co2_per_100g * 10.0
        df["Cost"] = cost_per_100g * 10.0
        df["Net_emission"] = df["CO2_emission"] - df["CO2_abs"]
        df["w/c_fixed"] = self.wc
        df["gypsum_fixed_%"] = self.gypsum
        df["temp_fixed_C"] = self.temp
        return df


def project_to_bounds_with_sum(weights, lo, hi, total):
    """Project vector to satisfy lo <= x <= hi and sum(x) = total."""

    weights = np.asarray(weights, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    weights = np.maximum(weights, 1e-12)
    cap = np.maximum(hi - lo, 0.0)
    must = float(lo.sum())

    if must > total + 1e-9:
        x = lo.copy()
        if must > 0:
            x *= total / must
        return np.clip(x, lo, hi)

    rem = total - must
    x = lo.copy()
    free = cap.copy()

    for _ in range(50):
        if rem <= 1e-12 or np.all(free <= 1e-12):
            break
        mask = free > 1e-12
        ww = weights[mask] / weights[mask].sum()
        give = np.minimum(free[mask], rem * ww)
        x[mask] += give
        free[mask] -= give
        rem -= give.sum()

    return np.clip(x, lo, hi)


class Sampler:
    """Handles range parsing and feasible random sampling for clinker and SCMs."""

    @staticmethod
    def parse_range(text: str, default=(0.0, 1.0)):
        try:
            a, b = [float(x.strip()) for x in str(text).split(",")]
            if a > b:
                a, b = b, a
            return (a, b)
        except Exception:
            return default

    @staticmethod
    def sample_group(bounds, total=100.0, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        names = list(bounds.keys())
        lo = np.array([bounds[name][0] for name in names])
        hi = np.array([bounds[name][1] for name in names])

        if lo.sum() > total or hi.sum() < total:
            return None

        cap = hi - lo
        rem = total - lo.sum()
        weights = rng.dirichlet(np.ones_like(cap))
        x = lo + rem * weights
        return dict(zip(names, np.clip(x, lo, hi)))

    @staticmethod
    def sample_mixes(n, clinker_sum_rng, clinker_bounds, scms_bounds, total_binder_target, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        rows = []

        while len(rows) < n:
            target_clinker = rng.uniform(*clinker_sum_rng)
            target_scm = total_binder_target - target_clinker

            clinker_names = list(clinker_bounds.keys())
            clinker_raw = []
            for name in clinker_names:
                lo, hi = clinker_bounds[name]
                clinker_raw.append(rng.uniform(lo, hi))

            clinker_sum = sum(clinker_raw)
            if clinker_sum == 0:
                continue
            clinker_values = [value * (target_clinker / clinker_sum) for value in clinker_raw]

            scm_names = list(scms_bounds.keys())
            scm_raw = []
            for name in scm_names:
                lo, hi = scms_bounds[name]
                scm_raw.append(rng.uniform(lo, hi))

            scm_sum = sum(scm_raw)
            if scm_sum == 0:
                continue
            scm_values = [value * (target_scm / scm_sum) for value in scm_raw]

            scm_ok = True
            for idx, value in enumerate(scm_values):
                name = scm_names[idx]
                lo, hi = scms_bounds[name]
                if value < lo - 1e-6 or value > hi + 1e-6:
                    scm_ok = False
                    break

            if not scm_ok:
                continue

            all_values = clinker_values + scm_values
            if not np.isclose(sum(all_values), total_binder_target, atol=1e-2):
                continue

            row = {}
            for idx, name in enumerate(clinker_names):
                row[name] = clinker_values[idx]
            for idx, name in enumerate(scm_names):
                row[name] = scm_values[idx]
            row["clinker_sum_%"] = target_clinker
            row["scms_sum_%"] = target_scm
            rows.append(row)

        return pd.DataFrame(rows)


class ParetoOptimizer:
    """Utility for Pareto dominance filtering."""

    @staticmethod
    def dominates(a, b, sense):
        not_worse = True
        strictly_better = False

        for key, direction in sense.items():
            if direction == "min":
                if a[key] > b[key]:
                    not_worse = False
                if a[key] < b[key]:
                    strictly_better = True
            else:
                if a[key] < b[key]:
                    not_worse = False
                if a[key] > b[key]:
                    strictly_better = True

        return not_worse and strictly_better

    @staticmethod
    def pareto_mask(records, sense):
        n = len(records)
        mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                if ParetoOptimizer.dominates(records[j], records[i], sense):
                    mask[i] = False
                    break

        return mask


class ConcreteMixProblem(Problem):
    """Pymoo problem for multi-objective concrete mix optimization."""

    def __init__(
        self,
        model,
        metrics_calc,
        clinker_bounds,
        scms_bounds,
        clinker_sum_rng,
        total_binder_target,
        time_val,
        emission_factors,
        cost_factors,
        objectives,
        sum_tol=1e-2,
        big_penalty=1e6,
    ):
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

        n_constr = 2 * len(self.scms_names) + len(self.clinker_names) + 1

        super().__init__(
            n_var=n_vars,
            n_obj=len(objectives),
            n_constr=n_constr,
            xl=np.array(xl, dtype=float),
            xu=np.array(xu, dtype=float),
        )

    def decode(self, x_values):
        x_values = np.atleast_2d(x_values)
        rows = []

        for x in x_values:
            clinker_sum = float(x[0])
            scms_sum = float(self.total_binder_target - clinker_sum)

            idx = 1
            clinker_raw = np.array(x[idx : idx + len(self.clinker_names)], dtype=float)
            idx += len(self.clinker_names)
            if clinker_raw.sum() <= 0:
                clinker_raw = np.ones_like(clinker_raw)
            clinker_vals_abs = (clinker_raw / clinker_raw.sum()) * clinker_sum

            scms_raw = np.array(x[idx : idx + len(self.scms_names)], dtype=float)
            if scms_raw.sum() <= 0:
                scms_raw = np.ones_like(scms_raw)
            scms_vals_abs = (scms_raw / scms_raw.sum()) * scms_sum

            row = {}
            for i, name in enumerate(self.clinker_names):
                row[name] = clinker_vals_abs[i]
            for i, name in enumerate(self.scms_names):
                row[name] = scms_vals_abs[i]
            row["time"] = self.time_val
            rows.append(row)

        return pd.DataFrame(rows)

    def _constraints(self, mixes: pd.DataFrame):
        g_list = []

        for name in self.scms_names:
            lo, _ = self.scms_bounds[name]
            vals = mixes[name].values
            g_list.append(np.maximum(0.0, lo - vals))
        for name in self.scms_names:
            _, hi = self.scms_bounds[name]
            vals = mixes[name].values
            g_list.append(np.maximum(0.0, vals - hi))

        clinker_min = float(self.clinker_sum_rng[0])
        for name in self.clinker_names:
            lo_pct = self.clinker_bounds[name][0]
            required_abs = lo_pct / 100.0 * clinker_min
            vals = mixes[name].values
            g_list.append(np.maximum(0.0, required_abs - vals))

        sum_vals = mixes[self.clinker_names + self.scms_names].sum(axis=1).values
        total_violation = np.maximum(0.0, np.abs(sum_vals - self.total_binder_target) - self.sum_tol)
        g_list.append(total_violation)
        return np.vstack(g_list).T

    def _evaluate(self, x_values, out, *args, **kwargs):
        mixes = self.decode(x_values)
        g = self._constraints(mixes)
        out["G"] = g

        preds = self.model.predict(mixes)
        df = self.metrics_calc.add_metrics(
            mixes,
            preds[:, 0],
            preds[:, 1],
            self.emission_factors,
            self.cost_factors,
        )

        f_cols = []
        for obj in self.objectives:
            if obj == "E_max":
                f_cols.append(-df["E"].values)
            elif obj == "CO2abs_max":
                f_cols.append(-df["CO2_abs"].values)
            elif obj == "CO2_min":
                f_cols.append(df["CO2_emission"].values)
            elif obj == "Cost_min":
                f_cols.append(df["Cost"].values)
            elif obj == "Net_min":
                f_cols.append(df["Net_emission"].values)
            else:
                raise ValueError(f"Unknown objective tag: {obj}")

        f = np.column_stack(f_cols)
        infeasible = (g > 0).any(axis=1)
        if infeasible.any():
            f[infeasible, :] = f[infeasible, :] + self.big_penalty
        out["F"] = f


def gwo_inverse_design(
    model,
    target_dict,
    active_vars,
    bounds_dict,
    fixed_params,
    pop_size=20,
    max_iter=50,
):
    """Grey Wolf Optimizer for inverse mix design."""

    dim = len(active_vars)
    lb = np.array([bounds_dict[key][0] for key in active_vars])
    ub = np.array([bounds_dict[key][1] for key in active_vars])

    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    convergence_curve = []

    def repair_constraints(pos_vector):
        pos_vector = np.clip(pos_vector, lb, ub)
        target_sum = fixed_params.get("total_binder", 96.0)
        current_sum = np.sum(pos_vector)
        if current_sum > 0:
            pos_vector = pos_vector * (target_sum / current_sum)
        return np.clip(pos_vector, lb, ub)

    def calculate_fitness(pos_vector):
        input_data = fixed_params.copy()
        for idx, var in enumerate(active_vars):
            input_data[var] = pos_vector[idx]

        preds = model.predict_single_sample(input_data)
        loss = 0.0
        for key, target_val in target_dict.items():
            pred_val = preds.get(key, 0)
            if target_val != 0:
                loss += ((pred_val - target_val) / target_val) ** 2
            else:
                loss += (pred_val - target_val) ** 2
        return loss

    for iteration in range(max_iter):
        a = 2 - iteration * (2 / max_iter)

        for i in range(pop_size):
            positions[i, :] = repair_constraints(positions[i, :])
            fitness = calculate_fitness(positions[i, :])

            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        for i in range(pop_size):
            for j in range(dim):
                r1 = np.random.random()
                r2 = np.random.random()
                a1 = 2 * a * r1 - a
                c1 = 2 * r2
                d_alpha = abs(c1 * alpha_pos[j] - positions[i, j])
                x1 = alpha_pos[j] - a1 * d_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                a2 = 2 * a * r1 - a
                c2 = 2 * r2
                d_beta = abs(c2 * beta_pos[j] - positions[i, j])
                x2 = beta_pos[j] - a2 * d_beta

                r1 = np.random.random()
                r2 = np.random.random()
                a3 = 2 * a * r1 - a
                c3 = 2 * r2
                d_delta = abs(c3 * delta_pos[j] - positions[i, j])
                x3 = delta_pos[j] - a3 * d_delta

                positions[i, j] = (x1 + x2 + x3) / 3

        convergence_curve.append(alpha_score)
        if alpha_score < 1e-4:
            break

    best_recipe = {}
    for idx, var in enumerate(active_vars):
        best_recipe[var] = alpha_pos[idx]

    return best_recipe, alpha_score, convergence_curve


__all__ = [
    "ConcreteMixProblem",
    "HAS_PYMOO",
    "MetricsCalculator",
    "ModelWrapper",
    "ParetoOptimizer",
    "Problem",
    "Sampler",
    "gwo_inverse_design",
    "project_to_bounds_with_sum",
]
