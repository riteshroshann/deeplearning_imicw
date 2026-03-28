"""Model evaluation, bootstrap CI, residual normality, and physics benchmarks."""

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from config import MATRIX_STRENGTH_MPa, RNG, subsection


def metrics_dict(y_true, y_pred):
    resid = y_true - y_pred
    return dict(R2=r2_score(y_true, y_pred),
                RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
                MAE=mean_absolute_error(y_true, y_pred),
                MAPE=mean_absolute_percentage_error(y_true, y_pred)*100,
                bias=resid.mean(), std_resid=resid.std())


def bootstrap_ci(y_true, y_pred, metric_fn=r2_score, n_boot=500, ci=95):
    n = len(y_true)
    boot = [metric_fn(y_true[idx], y_pred[idx])
            for idx in (RNG.integers(0, n, n) for _ in range(n_boot))]
    lo = np.percentile(boot, (100-ci)/2)
    hi = np.percentile(boot, 100-(100-ci)/2)
    return lo, metric_fn(y_true, y_pred), hi


def residual_normality_tests(residuals):
    sw_stat, sw_p = stats.shapiro(residuals[:min(len(residuals), 5000)])
    da_stat, da_p = stats.normaltest(residuals)
    return {"shapiro_W": sw_stat, "shapiro_p": sw_p,
            "dagostino_k2": da_stat, "dagostino_p": da_p,
            "skewness": stats.skew(residuals),
            "kurtosis_excess": stats.kurtosis(residuals)}


def physics_benchmark(y_test, test_df):
    subsection("Physics benchmark — ROM comparison")
    vf = test_df["fiber_volume_pct"].fillna(58).values / 100.0
    sf = test_df["fiber_tensile_strength_MPa"].fillna(3500).values
    p0 = test_df["pct_0_plies"].fillna(25).values / 100.0

    rom_strength = vf*sf + (1-vf)*MATRIX_STRENGTH_MPa
    rom_orient = p0*vf*sf + (1-p0)*(1-vf)*MATRIX_STRENGTH_MPa

    m_rom = metrics_dict(y_test, rom_strength)
    m_rom_orient = metrics_dict(y_test, rom_orient)

    print(f"\n  ROM (UD)       R² = {m_rom['R2']:.4f}  RMSE = {m_rom['RMSE']:.1f}")
    print(f"  ROM + orient.  R² = {m_rom_orient['R2']:.4f}  RMSE = {m_rom_orient['RMSE']:.1f}")
    return rom_strength, rom_orient, m_rom, m_rom_orient
