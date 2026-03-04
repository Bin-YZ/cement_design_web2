"""
Pareto dominance utilities for multi-objective optimization.
"""
import numpy as np


class ParetoOptimizer:
    """Utility for Pareto dominance filtering."""
    
    @staticmethod
    def dominates(a, b, sense):
        """Check if solution a dominates b under min/max sense dict.
        
        Args:
            a: Solution dictionary
            b: Solution dictionary
            sense: Dict mapping objective names to 'min' or 'max'
            
        Returns:
            True if a dominates b
        """
        not_worse, strictly_better = True, False
        
        for key, direction in sense.items():
            if direction == "min":
                if a[key] > b[key]:
                    not_worse = False
                if a[key] < b[key]:
                    strictly_better = True
            else:  # max
                if a[key] < b[key]:
                    not_worse = False
                if a[key] > b[key]:
                    strictly_better = True
        
        return not_worse and strictly_better
    
    @staticmethod
    def pareto_mask(records, sense):
        """Return a boolean mask for the non-dominated front.
        
        Args:
            records: List of solution dictionaries
            sense: Dict mapping objective names to 'min' or 'max'
            
        Returns:
            Boolean numpy array marking non-dominated solutions
        """
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
