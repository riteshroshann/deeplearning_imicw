"""
evaluation.py — Comprehensive Model Evaluation & Physics Benchmarks
=====================================================================
Provides regression metrics, bootstrap confidence intervals,
residual normality tests, and comparison against classical
composite mechanics models (Rule-of-Mixtures, Halpin-Tsai).
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error
)
from config import (
    MATRIX_STRENGTH_MPa, RNG, section, subsection
)


def metrics_dict(y_true, y_pred):
    """Compute a comprehensive set of regression metrics."""
    r2    = r2_score(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    resid = y_true - y_pred
    return dict(R2=r2, RMSE=rmse, MAE=mae, MAPE=mape,
                bias=resid.mean(), std_resid=resid.std())


def bootstrap_ci(y_true, y_pred, metric_fn=r2_score,
                 n_boot=500, ci=95):
    """
    Non-parametric bootstrap confidence interval for any scalar metric.

    Parameters
    ----------
    y_true, y_pred : array-like
    metric_fn      : callable(y_true, y_pred) → float
    n_boot         : number of bootstrap iterations
    ci             : confidence level (%)

    Returns
    -------
    (lower_bound, point_estimate, upper_bound)
    """
    n = len(y_true)
    boot = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        boot.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return lo, metric_fn(y_true, y_pred), hi


def residual_normality_tests(residuals):
    """
    Perform Shapiro-Wilk and D'Agostino-Pearson normality tests
    on model residuals.

    Returns
    -------
    dict with test statistics and p-values
    """
    # Shapiro-Wilk (limited to n≤5000)
    n = min(len(residuals), 5000)
    sw_stat, sw_p = stats.shapiro(residuals[:n])

    # D'Agostino-Pearson
    da_stat, da_p = stats.normaltest(residuals)

    # Skewness and Kurtosis
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)

    return {
        "shapiro_W": sw_stat, "shapiro_p": sw_p,
        "dagostino_k2": da_stat, "dagostino_p": da_p,
        "skewness": skew, "kurtosis_excess": kurt,
    }


def physics_benchmark(y_test, test_df):
    """
    Compute physics-based predictions as benchmarks:
      1. Rule-of-Mixtures (ROM) — isostrain upper bound
      2. Orientation-weighted ROM — accounts for ply angles

    These analytic baselines quantify how much the ML models
    improve over classical micromechanics.

    Parameters
    ----------
    y_test  : actual test set targets
    test_df : DataFrame with fibre properties and ply fractions

    Returns
    -------
    rom_strength   : ROM predictions (1D UD assumption)
    rom_orient     : orientation-weighted ROM predictions
    m_rom          : metrics for ROM
    m_rom_orient   : metrics for orientation-weighted ROM
    """
    subsection("Physics benchmark — ROM & Halpin-Tsai comparison")

    sigma_m = MATRIX_STRENGTH_MPa

    vf_test = test_df["fiber_volume_pct"].fillna(58).values / 100.0
    sf_test = test_df["fiber_tensile_strength_MPa"].fillna(3500).values
    p0_test = test_df["pct_0_plies"].fillna(25).values / 100.0

    # ROM: σ_L = V_f · σ_f + V_m · σ_m
    rom_strength = vf_test * sf_test + (1 - vf_test) * sigma_m

    # Orientation-weighted ROM
    rom_orient = (p0_test * vf_test * sf_test +
                  (1 - p0_test) * (1 - vf_test) * sigma_m)

    m_rom       = metrics_dict(y_test, rom_strength)
    m_rom_orient = metrics_dict(y_test, rom_orient)

    print(f"\n  Rule-of-Mixtures (UD)       R² = {m_rom['R2']:.4f}  "
          f"RMSE = {m_rom['RMSE']:.1f} MPa")
    print(f"  ROM with orientation weight  R² = {m_rom_orient['R2']:.4f}  "
          f"RMSE = {m_rom_orient['RMSE']:.1f} MPa")

    return rom_strength, rom_orient, m_rom, m_rom_orient
