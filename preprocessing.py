"""Physics-validated feature engineering for CFRP composites."""

import numpy as np
import pandas as pd
from config import (
    AERO_FIBERS, FIBER_RANK, FIBER_MOD_GPa, FIBER_STR_MPa,
    FIBER_DENSITY_gcc, MFG_CODE, MATRIX_MODULUS_GPa,
    MATRIX_STRENGTH_MPa, MATRIX_DENSITY_gcc,
    ALL_FEATURES, TARGET_STRENGTH, section, subsection, OUT_DIR
)


def classify_layup(row):
    p0  = row.get("pct_0_plies")
    p45 = row.get("pct_45_plies")
    p90 = row.get("pct_90_plies")
    if pd.isna(p0):
        return "Unknown", 0
    if p0 >= 90:    return "UD_0deg", 1
    elif p90 >= 90: return "UD_90deg", 2
    elif p45 >= 90: return "UD_45deg", 3
    elif abs(p0-50) < 5 and (p45 or 0) < 5: return "CrossPly_0_90", 4
    elif abs(p0-25) < 5 and abs(p45-50) < 10: return "QuasiIsotropic", 5
    elif p0 <= 15 and p45 >= 70: return "Soft_ShearDom", 6
    elif p0 >= 45 and p45 >= 35: return "Hard_Stiff", 7
    else: return "Mixed", 8


def preprocess(raw_df):
    """Full preprocessing: filter → encode → impute → engineer 12 physics features."""
    section("PHASE 2 — PREPROCESSING & FEATURE ENGINEERING")
    df = raw_df.copy()

    df = df[df["fiber_type"].isin(AERO_FIBERS)].copy()
    df = df[df["temperature_C"] >= -80].copy()
    print(f"  Records after aerospace filter : {len(df)}")

    df["fiber_rank"] = df["fiber_type"].map(FIBER_RANK).fillna(3)
    df["fiber_tensile_modulus_GPa"] = df["fiber_type"].map(FIBER_MOD_GPa).fillna(230)
    df["fiber_tensile_strength_MPa"] = df["fiber_type"].map(FIBER_STR_MPa).fillna(3500)
    df["fiber_density_gcc"] = df["fiber_type"].map(FIBER_DENSITY_gcc).fillna(1.78)
    df["manufacturing_code"] = df["manufacturing_method"].map(MFG_CODE).fillna(1)

    TEST_TYPE_CODE = {
        "Longitudinal_Tension": 1, "Transverse_Tension": 2,
        "Longitudinal_Compression": 3, "Transverse_Compression": 4,
        "InPlane_Shear": 5, "Flexural_Test": 6, "Interlaminar_Tension": 7,
    }
    df["test_type_code"] = df["test_type"].apply(
        lambda x: next((v for k, v in TEST_TYPE_CODE.items() if k in str(x)), 1))

    df[["layup_class", "layup_code_num"]] = df.apply(
        classify_layup, axis=1, result_type="expand")

    df["CNT_vol_frac_pct"]           = df["CNT_vol_frac_pct"].fillna(0.0)
    df["interlayer_vol_frac_pct"]    = df["interlayer_vol_frac_pct"].fillna(0.0)
    df["manufacturing_pressure_psi"] = df["manufacturing_pressure_psi"].fillna(85.0)
    df["fiber_volume_pct"]           = df["fiber_volume_pct"].fillna(58.0)
    df["Tg_dry_C"]                   = df["Tg_dry_C"].fillna(180.0)
    df["Tg_wet_C"]                   = df["Tg_wet_C"].fillna(df["Tg_dry_C"] - 75.0)

    subsection("Physics-based feature engineering (12 variables)")
    Vf  = df["fiber_volume_pct"] / 100.0
    Vm  = 1.0 - Vf
    E_m = MATRIX_MODULUS_GPa
    E_f = df["fiber_tensile_modulus_GPa"]

    # F1: Rule-of-Mixtures longitudinal modulus
    df["E_L_ROM_GPa"] = Vf * E_f + Vm * E_m

    # F2: Halpin-Tsai transverse modulus
    xi  = 2.0
    eta = (E_f/E_m - 1) / (E_f/E_m + xi)
    df["E_T_HT_GPa"] = E_m * (1 + xi*eta*Vf) / (1 - eta*Vf)

    # F3: Orientation efficiency factor
    p0  = df["pct_0_plies"].fillna(25) / 100.0
    p45 = df["pct_45_plies"].fillna(50) / 100.0
    p90 = df["pct_90_plies"].fillna(25) / 100.0
    df["orientation_efficiency"] = (
        p0*np.cos(np.radians(0))**4 +
        p45*np.cos(np.radians(45))**4 +
        p90*np.cos(np.radians(90))**4)

    # F4: ROM longitudinal strength
    df["sigma_L_ROM_MPa"] = Vf*df["fiber_tensile_strength_MPa"] + Vm*MATRIX_STRENGTH_MPa

    # F5: Thermal knock-down
    df["thermal_knockdown"] = df["temperature_C"] / df["Tg_dry_C"].clip(lower=1.0)

    # F6: Effective Vf with moisture correction
    df["effective_Vf_pct"] = df["fiber_volume_pct"] - df["moisture_code"]*0.5

    # F7: Manufacturing quality index
    df["mfg_quality_index"] = (
        (6.0 - df["manufacturing_code"]) *
        np.sqrt(df["manufacturing_pressure_psi"].clip(lower=0.1) / 100.0))

    # F8: Anisotropy ratio
    df["anisotropy_proxy"] = df["E_L_ROM_GPa"] / df["E_T_HT_GPa"].clip(lower=0.1)

    # F9: CNT-interlayer synergy
    df["cnt_il_synergy"] = df["CNT_vol_frac_pct"] * (1 + df["interlayer_vol_frac_pct"]/10.0)

    # F10: 0-degree ply contribution
    df["zero_ply_contribution"] = p0 * df["sigma_L_ROM_MPa"]

    # F11: Composite density
    df["composite_density_gcc"] = Vf*df["fiber_density_gcc"] + Vm*MATRIX_DENSITY_gcc

    # F12: Specific strength
    df["sigma_L_specific"] = df["sigma_L_ROM_MPa"] / df["composite_density_gcc"]

    print(f"  12 physics features added | Shape: {df.shape}")
    df.to_csv(f"{OUT_DIR}/CFRP_Engineered_Dataset.csv", index=False)
    print(f"  Saved -> CFRP_Engineered_Dataset.csv")
    return df


def prepare_ml_data(df, target=TARGET_STRENGTH, features=None):
    """Extract feature matrix X and target vector y, impute, and scale."""
    from sklearn.preprocessing import RobustScaler
    from sklearn.impute import SimpleImputer

    if features is None:
        features = ALL_FEATURES

    df_ml = df.dropna(subset=[target]).copy()
    X_raw = df_ml[features].copy()
    y = df_ml[target].values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_imp)

    print(f"\n  ML data: {len(df_ml)} samples, {len(features)} features, "
          f"range {y.min():.0f}–{y.max():.0f}")
    return X_sc, y, df_ml, imputer, scaler
