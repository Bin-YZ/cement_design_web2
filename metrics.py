import numpy as np
import pandas as pd


class MetricsCalculator:
    """Compute emissions, cost, and derived metrics with consistent units for the Streamlit app.

    Outputs:
      - CO2_emission: kg CO2 / kg binder (kg/kg)
      - CO2_abs:      kg CO2 / kg binder (kg/kg)
      - Net_emission: kg CO2 / kg binder (kg/kg)
      - Cost:         EUR / kg binder (€/kg)

    Assumptions:
      - df[materials] are grams per 100g sample (g/100g)
      - gypsum_fixed is grams per 100g sample (g/100g)
      - emission_factors are kgCO2 per kg material (kg/kg)
      - cost_factors are EUR per kg material (€/kg)
      - CO2_abs_pred is gCO2 per 100g sample (g/100g)
    """

    def __init__(self, wc_fixed: float, gypsum_fixed: float, temp_fixed: float):
        self.wc = float(wc_fixed)
        self.gypsum = float(gypsum_fixed)   # g per 100g sample
        self.temp = float(temp_fixed)

    def add_metrics(self, df, E_pred, CO2_abs_pred, emission_factors, cost_factors):
        df = df.copy()

        # --- Base predictions ---
        df["E"] = np.asarray(E_pred, dtype=float)

        # Model: g/100g -> kg/kg  (because (g/100g) * (1kg/1000g) * (1000g/kg) = g/100 => kg/kg)
        df["CO2_abs"] = np.asarray(CO2_abs_pred, dtype=float) / 100.0

        # --- Materials used for dot-product ---
        materials = [
            "C3S", "C2S", "C3A", "C4AF", "silica_fume",
            "GGBFS", "fly_ash", "calcined_clay", "limestone"
        ]

        # Safety: ensure all material columns exist
        for m in materials:
            if m not in df.columns:
                df[m] = 0.0

        emis_vec = np.array([float(emission_factors.get(m, 0.0)) for m in materials], dtype=float)  # kg/kg
        cost_vec = np.array([float(cost_factors.get(m, 0.0)) for m in materials], dtype=float)      # €/kg

        # df[materials] are grams per 100g sample -> convert to kg per 100g sample
        kg_per_100g = df[materials].to_numpy(dtype=float) / 1000.0

        # Emission / cost per 100g sample
        co2_per_100g = kg_per_100g @ emis_vec               # kg CO2 per 100g sample
        cost_per_100g = kg_per_100g @ cost_vec              # EUR per 100g sample

        # Add gypsum (also grams per 100g sample)
        gypsum_kg_per_100g = self.gypsum / 1000.0
        co2_per_100g += gypsum_kg_per_100g * float(emission_factors.get("Gypsum", 0.0))
        cost_per_100g += gypsum_kg_per_100g * float(cost_factors.get("Gypsum", 0.0))

        # Convert from per-100g-sample to per-kg-binder (kg/kg and €/kg)
        # 100g sample = 0.1 kg binder -> multiply by 10
        df["CO2_emission"] = co2_per_100g * 10.0   # kg/kg
        df["Cost"] = cost_per_100g * 10.0          # €/kg

        # Net emission in kg/kg
        df["Net_emission"] = df["CO2_emission"] - df["CO2_abs"]

        # Fixed fields (keep your original column names)
        df["w/c_fixed"] = self.wc
        df["gypsum_fixed_%"] = self.gypsum
        df["temp_fixed_C"] = self.temp

        return df
