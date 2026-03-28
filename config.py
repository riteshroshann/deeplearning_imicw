"""
config.py — Central Configuration for CFRP Research Pipeline
=============================================================
All constants, physics lookup tables, hyperparameters, and
matplotlib style settings in one place for reproducibility
and maintainability.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Unit Conversion ─────────────────────────────────────────────────────────
MPa_TO_ksi = 0.145038
GPa_TO_Msi = 0.145038

# ── Matrix Properties (Typical Aerospace Epoxy, ~120-180 °C cure) ────────────
MATRIX_MODULUS_GPa    = 3.5    # Young's modulus
MATRIX_STRENGTH_MPa   = 70.0   # Tensile strength at failure
MATRIX_DENSITY_gcc    = 1.25   # Density (g/cm³)

# ── Fibre Lookup Tables ─────────────────────────────────────────────────────
#    Sources: Hexcel, Toray, SGL Carbon data sheets (2023-2024)
FIBER_RANK = {
    "T300": 1, "Carbon Fiber": 2, "T700": 3, "HS Carbon": 4,
    "IM Carbon": 5, "T800": 6, "IM7": 7, "HM Carbon": 8,
}
FIBER_MOD_GPa = {
    "T300": 230, "Carbon Fiber": 230, "T700": 230, "HS Carbon": 240,
    "IM Carbon": 290, "T800": 294, "IM7": 276, "HM Carbon": 370,
}
FIBER_STR_MPa = {
    "T300": 3530, "Carbon Fiber": 3500, "T700": 4900, "HS Carbon": 4400,
    "IM Carbon": 5100, "T800": 5880, "IM7": 5516, "HM Carbon": 3920,
}
FIBER_DENSITY_gcc = {
    "T300": 1.76, "Carbon Fiber": 1.76, "T700": 1.80, "HS Carbon": 1.78,
    "IM Carbon": 1.79, "T800": 1.81, "IM7": 1.78, "HM Carbon": 1.73,
}

# Reverse maps from rank for GA decoding
RANK_TO_MOD = {v: FIBER_MOD_GPa[k] for k, v in FIBER_RANK.items()}
RANK_TO_STR = {v: FIBER_STR_MPa[k] for k, v in FIBER_RANK.items()}
RANK_TO_DEN = {v: FIBER_DENSITY_gcc[k] for k, v in FIBER_RANK.items()}
RANK_TO_FIB = {v: k for k, v in FIBER_RANK.items()}

# Valid aerospace fibre types
AERO_FIBERS = list(FIBER_RANK.keys())

# Manufacturing method codes
MFG_CODE = {
    "Autoclave": 1,
    "Autoclave/Vacuum Infusion": 2,
    "Vacuum Infusion": 3,
    "Hand Layup + Compression Moulding": 4,
    "Filament Winding + Hand Layup": 5,
}

# ── Feature Set ──────────────────────────────────────────────────────────────
ORIGINAL_FEATURES = [
    "fiber_rank", "fiber_tensile_modulus_GPa", "fiber_tensile_strength_MPa",
    "fiber_volume_pct", "pct_0_plies", "pct_45_plies", "pct_90_plies",
    "layup_code_num", "CNT_vol_frac_pct", "interlayer_vol_frac_pct",
    "manufacturing_code", "manufacturing_pressure_psi",
    "Tg_dry_C", "Tg_wet_C", "temperature_C", "env_code", "moisture_code",
    "test_type_code",
]

ENGINEERED_FEATURES = [
    "E_L_ROM_GPa", "E_T_HT_GPa", "orientation_efficiency",
    "sigma_L_ROM_MPa", "thermal_knockdown", "effective_Vf_pct",
    "mfg_quality_index", "anisotropy_proxy", "cnt_il_synergy",
    "zero_ply_contribution", "composite_density_gcc", "sigma_L_specific",
]

ALL_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES

TARGET_STRENGTH = "strength_MPa"
TARGET_MODULUS  = "modulus_GPa"

# ── ML Hyperparameters ───────────────────────────────────────────────────────
TEST_SIZE       = 0.20
N_CV_FOLDS      = 5
N_BOOTSTRAP     = 500

# PyTorch DL hyperparameters
DL_CONFIGS = {
    "DL1": {"layers": [128],              "activation": "silu", "dropout": 0.15, "lr": 1e-3, "epochs": 600, "patience": 50},
    "DL2": {"layers": [256, 128],         "activation": "silu", "dropout": 0.20, "lr": 1e-3, "epochs": 600, "patience": 50},
    "DL3": {"layers": [256, 128, 64],     "activation": "silu", "dropout": 0.20, "lr": 1e-3, "epochs": 800, "patience": 60},
    "DL4": {"layers": [512, 256, 128, 64],"activation": "silu", "dropout": 0.25, "lr": 5e-4, "epochs": 800, "patience": 60},
    "DL5": {"layers": [256, 128, 64, 32], "activation": "silu", "dropout": 0.20, "lr": 1e-3, "epochs": 600, "patience": 50},
}

# sklearn ensemble hyperparameters
SKLEARN_MODELS_CONFIG = {
    "RF":  {"n_estimators": 400, "max_features": "sqrt", "min_samples_leaf": 2},
    "ET":  {"n_estimators": 400, "max_features": "sqrt", "min_samples_leaf": 2},
    "GBM": {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 5,
             "min_samples_leaf": 2, "subsample": 0.85},
    "SVR": {"kernel": "rbf", "C": 200.0, "epsilon": 5.0, "gamma": "scale"},
}

# ── GA Configuration ────────────────────────────────────────────────────────
GA_POP_SIZE    = 80
GA_N_GEN       = 100
GA_MUTATE_P    = 0.12
GA_CROSSOVER_P = 0.88
GA_ELITE_K     = 6

# Design space bounds for GA
GA_BOUNDS = {
    "fiber_rank":                 (1,    8,    "int"),
    "fiber_tensile_modulus_GPa":  (230,  370,  "float"),
    "fiber_tensile_strength_MPa": (3530, 5880, "float"),
    "fiber_volume_pct":           (50,   65,   "float"),
    "pct_0_plies":                (0,    100,  "float"),
    "pct_45_plies":               (0,    100,  "float"),
    "pct_90_plies":               (0,    100,  "float"),
    "layup_code_num":             (1,    8,    "int"),
    "CNT_vol_frac_pct":           (0,    1.36, "float"),
    "interlayer_vol_frac_pct":    (0,    8.0,  "float"),
    "manufacturing_code":         (1,    5,    "int"),
    "manufacturing_pressure_psi": (0,    100,  "float"),
    "Tg_dry_C":                   (110,  220,  "float"),
    "Tg_wet_C":                   (80,   160,  "float"),
    "temperature_C":              (23,   23,   "float"),
    "env_code":                   (1,    1,    "int"),
    "moisture_code":              (0,    0,    "int"),
    "test_type_code":             (1,    10,   "int"),
    "E_L_ROM_GPa":                (50,   250,  "float"),
    "E_T_HT_GPa":                 (4,    30,   "float"),
    "orientation_efficiency":     (0,    1,    "float"),
    "sigma_L_ROM_MPa":            (500,  5000, "float"),
    "thermal_knockdown":          (0.1,  0.5,  "float"),
    "effective_Vf_pct":           (49,   65,   "float"),
    "mfg_quality_index":          (0,    5,    "float"),
    "anisotropy_proxy":           (1,    50,   "float"),
    "cnt_il_synergy":             (0,    2,    "float"),
    "zero_ply_contribution":      (0,    5000, "float"),
    "composite_density_gcc":      (1.3,  2.0,  "float"),
    "sigma_L_specific":           (200,  3500, "float"),
}

# ── Publication Color Palette ────────────────────────────────────────────────
#    Carefully chosen for accessibility, print reproduction, and colour-
#    blindness friendliness (simulated with Coblis CVD simulator).
PALETTE = {
    "primary":    "#1B4F72",
    "secondary":  "#C0392B",
    "accent1":    "#1E8449",
    "accent2":    "#B7950B",
    "accent3":    "#5B2C6F",
    "accent4":    "#1A5276",
    "neutral":    "#717D7E",
    "light":      "#D5D8DC",
    "background": "#FDFEFE",
    # Per-model colours for consistent figure styling
    "RF":  "#1A5276", "ET":  "#117A65", "GBM": "#6E2F1A",
    "SVR": "#4A235A", "STACK": "#1B4F72",
    "DL1": "#1B4F72", "DL2": "#C0392B", "DL3": "#1E8449",
    "DL4": "#B7950B", "DL5": "#5B2C6F",
}

# ── Global matplotlib RC ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "savefig.dpi":       220,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "grid.alpha":        0.3,
    "grid.linewidth":    0.6,
})

# ── Correlation Feature Subset (for heatmaps) ───────────────────────────────
CORR_FEATURES = [
    "fiber_volume_pct", "pct_0_plies", "pct_45_plies", "pct_90_plies",
    "fiber_rank", "fiber_tensile_modulus_GPa", "CNT_vol_frac_pct",
    "interlayer_vol_frac_pct", "manufacturing_pressure_psi", "Tg_dry_C",
    "temperature_C", "moisture_code", "test_type_code",
    "E_L_ROM_GPa", "E_T_HT_GPa", "orientation_efficiency",
    "sigma_L_ROM_MPa", "thermal_knockdown", "mfg_quality_index",
    TARGET_STRENGTH,
]

CORR_LABELS = [
    "Vf (%)", "0° plies", "45° plies", "90° plies",
    "Fibre rank", "Ef (GPa)", "CNT vol%",
    "Interlayer%", "Cure P (psi)", "Tg dry (°C)",
    "T_test (°C)", "Moisture", "Test type",
    "E_L_ROM", "E_T_HT", "η₀",
    "σ_L_ROM", "Thermal KD", "Mfg Quality",
    "Strength (MPa)",
]


# ── Utility ──────────────────────────────────────────────────────────────────
def savefig(name, fig=None):
    """Save figure to outputs directory and close."""
    path = os.path.join(OUT_DIR, f"{name}.png")
    if fig:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.savefig(path)
        plt.close()
    print(f"    [SAVED] {name}.png")


def section(title):
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def subsection(title):
    print(f"\n  ── {title} {'─' * max(1, 66 - len(title))}")
