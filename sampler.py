"""
Sampling utilities for generating feasible concrete mix designs.
"""
import numpy as np
import pandas as pd


def project_to_bounds_with_sum(weights, lo, hi, total):
    """Project vector to satisfy lo <= x <= hi and sum(x) = total.
    
    Args:
        weights: Weight vector
        lo: Lower bounds
        hi: Upper bounds
        total: Target sum
        
    Returns:
        Projected vector satisfying constraints
    """
    w = np.asarray(weights, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    w = np.maximum(w, 1e-12)
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
        ww = w[mask] / w[mask].sum()
        give = np.minimum(free[mask], rem * ww)
        x[mask] += give
        free[mask] -= give
        rem -= give.sum()
    
    return np.clip(x, lo, hi)


class Sampler:
    """Handles range parsing and feasible random sampling for clinker and SCMs."""
    
    @staticmethod
    def parse_range(text: str, default=(0.0, 1.0)):
        """Parse range string like '0,10' into tuple (0.0, 10.0)."""
        try:
            a, b = [float(x.strip()) for x in str(text).split(",")]
            if a > b:
                a, b = b, a
            return (a, b)
        except Exception:
            return default
    
    @staticmethod
    def sample_group(bounds, total=100.0, rng=None):
        """Sample clinker group under bounds and total sum constraint.
        
        Args:
            bounds: Dict of {name: (lo, hi)}
            total: Target sum for the group
            rng: Random number generator
            
        Returns:
            Dict of sampled values or None if infeasible
        """
        rng = np.random.default_rng() if rng is None else rng
        names = list(bounds.keys())
        lo = np.array([bounds[k][0] for k in names])
        hi = np.array([bounds[k][1] for k in names])
        
        if lo.sum() > total or hi.sum() < total:
            return None
        
        cap = hi - lo
        rem = total - lo.sum()
        w = rng.dirichlet(np.ones_like(cap))
        x = lo + rem * w
        
        return dict(zip(names, np.clip(x, lo, hi)))
    
    @staticmethod
    def sample_mixes(n, clinker_sum_rng, clinker_bounds, scms_bounds, 
                     total_binder_target, rng=None):
        """Generate feasible mixes matching the original sampling logic.
        
        Args:
            n: Number of mixes to generate
            clinker_sum_rng: (min, max) for total clinker percentage
            clinker_bounds: Dict of clinker phase bounds (as percentages within clinker, 0-100)
            scms_bounds: Dict of SCM bounds (as absolute percentages, 0-96)
            total_binder_target: Total binder target (typically 96%)
            rng: Random number generator
            
        Returns:
            DataFrame of sampled mixes
        """
        rng = np.random.default_rng() if rng is None else rng
        rows = []
        material_keys = list(clinker_bounds.keys()) + list(scms_bounds.keys())
        
        while len(rows) < n:
            # Step 1: 随机生成目标熟料总量
            target_clinker = rng.uniform(*clinker_sum_rng)
            target_scm = total_binder_target - target_clinker
            
            # Step 2: 采样熟料组分（作为熟料内部的百分比，和为100）
            clinker_names = list(clinker_bounds.keys())
            clinker_raw = []
            for name in clinker_names:
                lo, hi = clinker_bounds[name]
                val = rng.uniform(lo, hi)
                clinker_raw.append(val)
            
            # Step 3: 归一化熟料到 target_clinker
            clinker_sum = sum(clinker_raw)
            if clinker_sum == 0:
                continue
            clinker_values = [v * (target_clinker / clinker_sum) for v in clinker_raw]
            
            # Step 4: 采样SCM组分
            scm_names = list(scms_bounds.keys())
            scm_raw = []
            for name in scm_names:
                lo, hi = scms_bounds[name]
                val = rng.uniform(lo, hi)
                scm_raw.append(val)
            
            # Step 5: 归一化SCM到 target_scm
            scm_sum = sum(scm_raw)
            if scm_sum == 0:
                continue
            scm_values = [v * (target_scm / scm_sum) for v in scm_raw]
            
            # Step 6: 检查SCM是否在bounds内
            scm_ok = True
            for i, val in enumerate(scm_values):
                name = scm_names[i]
                lo, hi = scms_bounds[name]
                if val < lo - 1e-6 or val > hi + 1e-6:
                    scm_ok = False
                    break
            
            if not scm_ok:
                continue
            
            # Step 7: 合并并检查总和
            all_values = clinker_values + scm_values
            if not np.isclose(sum(all_values), total_binder_target, atol=1e-2):
                continue
            
            # Step 8: 创建行数据
            row = {}
            for i, name in enumerate(clinker_names):
                row[name] = clinker_values[i]
            for i, name in enumerate(scm_names):
                row[name] = scm_values[i]
            row["clinker_sum_%"] = target_clinker
            row["scms_sum_%"] = target_scm
            
            rows.append(row)
        
        return pd.DataFrame(rows)
