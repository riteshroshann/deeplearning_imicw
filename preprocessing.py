"""
preprocessing.py — Physics-Validated Feature Engineering
=========================================================
Transforms raw CFRP dataset into ML-ready feature matrix with
12 physics-based engineered features grounded in composite
mechanics theory (Rule-of-Mixtures, Halpin-Tsai, Classical
Lamination Theory).
"""

import numpy as np
import pandas as pd
from config import (
    AERO_FIBERS, FIBER_RANK, FIBER_MOD_GPa, FIBER_STR_MPa,
    FIBER_DENSITY_gcc, MFG_CODE, MATRIX_MODULUS_GPa,
    MATRIX_STRENGTH_MPa, MATRIX_DENSITY_gcc,
    ALL_FEATURES, TARGET_STRENGTH, section, subsection, OUT_DIR
)


def classify_layup(row):
    """
    Classify composite layup into structural categories based on
    ply angle fractions.  This categorical encoding captures the
    dominant load-carrying mechanism of the laminate.

    Categories
    ----------
    UD_0deg        : Unidirectional, fibre-dominated (axial loads)
    UD_90deg       : Unidirectional, matrix-dominated (transverse)
    UD_45deg       : Unidirectional, shear-dominated
    CrossPly_0_90  : Balanced cross-ply (biaxial loads)
    QuasiIsotropic : Near-isotropic response (general purpose)
    Soft_ShearDom  : High off-axis content (damage tolerant)
    Hard_Stiff     : High 0° content (stiffness critical)
    Mixed          : Non-standard layup
    """
    p0  = row.get("pct_0_plies")
    p45 = row.get("pct_45_plies")
    p90 = row.get("pct_90_plies")
    if pd.isna(p0):
        return "Unknown", 0
    if p0 >= 90:
        return "UD_0deg", 1
    elif p90 >= 90:
        return "UD_90deg", 2
    elif p45 >= 90:
        return "UD_45deg", 3
    elif abs(p0 - 50) < 5 and (p45 or 0) < 5:
        return "CrossPly_0_90", 4
    elif abs(p0 - 25) < 5 and abs(p45 - 50) < 10:
        return "QuasiIsotropic", 5
    elif p0 <= 15 and p45 >= 70:
        return "Soft_ShearDom", 6
    elif p0 >= 45 and p45 >= 35:
        return "Hard_Stiff", 7
    else:
        return "Mixed", 8


def preprocess(raw_df):
    """
    Full preprocessing pipeline:
      1. Aerospace fibre filter
      2. Categorical encoding (fibre rank, manufacturing code, layup class)
      3. Missing value imputation with domain-aware defaults
      4. 12 physics-based engineered features
      5. Save engineered dataset to CSV

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output from data_builder.build_dataset()

    Returns
    -------
    df : pd.DataFrame
        Feature-engineered dataset ready for EDA and ML.
    """
    section("PHASE 2 — PREPROCESSING & FEATURE ENGINEERING")

    df = raw_df.copy()

    # ── Step 1: Aerospace fibre filter ─────────────────────────────────────
    df = df[df["fiber_type"].isin(AERO_FIBERS)].copy()
    df = df[df["temperature_C"] >= -80].copy()
    print(f"  Records after aerospace filter : {len(df)}")

    # ── Step 2: Categorical encoding ───────────────────────────────────────

    # Fibre type → rank + lookup properties
    df["fiber_rank"] = df["fiber_type"].map(FIBER_RANK).fillna(3)
    df["fiber_tensile_modulus_GPa"] = (
        df["fiber_type"].map(FIBER_MOD_GPa).fillna(230))
    df["fiber_tensile_strength_MPa"] = (
        df["fiber_type"].map(FIBER_STR_MPa).fillna(3500))
    df["fiber_density_gcc"] = (
        df["fiber_type"].map(FIBER_DENSITY_gcc).fillna(1.78))

    # Manufacturing method → ordinal code
    df["manufacturing_code"] = (
        df["manufacturing_method"].map(MFG_CODE).fillna(1))

    # Test type → ordinal code (critical for predicting correct strength regime)
    TEST_TYPE_CODE = {
        "Longitudinal_Tension": 1, "Transverse_Tension": 2,
        "Longitudinal_Compression": 3, "Transverse_Compression": 4,
        "InPlane_Shear": 5, "Flexural_Test": 6,
        "Interlaminar_Tension": 7,
    }
    # Assign codes — default 1 for unrecognized test types
    df["test_type_code"] = df["test_type"].apply(
        lambda x: next((v for k, v in TEST_TYPE_CODE.items()
                        if k in str(x)), 1))

    # Layup classification
    df[["layup_class", "layup_code_num"]] = df.apply(
        classify_layup, axis=1, result_type="expand")

    # ── Step 3: Missing value imputation ───────────────────────────────────
    # Domain-aware defaults (not arbitrary; based on typical CFRP values)
    df["CNT_vol_frac_pct"]           = df["CNT_vol_frac_pct"].fillna(0.0)
    df["interlayer_vol_frac_pct"]    = df["interlayer_vol_frac_pct"].fillna(0.0)
    df["manufacturing_pressure_psi"] = df["manufacturing_pressure_psi"].fillna(85.0)
    df["fiber_volume_pct"]           = df["fiber_volume_pct"].fillna(58.0)
    df["Tg_dry_C"]                   = df["Tg_dry_C"].fillna(180.0)
    df["Tg_wet_C"]                   = df["Tg_wet_C"].fillna(
        df["Tg_dry_C"] - 75.0)

    # ── Step 4: Physics-based feature engineering ──────────────────────────
    subsection("Physics-based feature engineering (12 variables)")

    Vf = df["fiber_volume_pct"] / 100.0   # volume fraction [0,1]
    Vm = 1.0 - Vf                          # matrix volume fraction

    E_m = MATRIX_MODULUS_GPa               # matrix modulus (GPa)
    E_f = df["fiber_tensile_modulus_GPa"]  # fibre modulus (GPa)

    # ── F1: Rule-of-Mixtures longitudinal modulus ──────────────────────────
    #    E_L = V_f · E_f + V_m · E_m
    #    (Voigt upper bound — exact for isostrain condition in UD composites)
    df["E_L_ROM_GPa"] = Vf * E_f + Vm * E_m

    # ── F2: Halpin-Tsai transverse modulus ─────────────────────────────────
    #    E_T = E_m · (1 + ξ·η·V_f) / (1 - η·V_f)
    #    where η = (E_f/E_m - 1) / (E_f/E_m + ξ), ξ=2 for circular fibres
    #    Ref: Tsai & Hahn, "Introduction to Composite Materials" (1980)
    xi  = 2.0
    eta = (E_f / E_m - 1) / (E_f / E_m + xi)
    df["E_T_HT_GPa"] = E_m * (1 + xi * eta * Vf) / (1 - eta * Vf)

    # ── F3: Orientation efficiency factor ──────────────────────────────────
    #    η₀ = Σ p_θ · cos⁴(θ)   (from laminate stiffness theory)
    #    η₀ = 1.0 for UD 0°, 0.0 for 90°, 0.0625 for ±45°
    p0  = df["pct_0_plies"].fillna(25) / 100.0
    p45 = df["pct_45_plies"].fillna(50) / 100.0
    p90 = df["pct_90_plies"].fillna(25) / 100.0
    df["orientation_efficiency"] = (
        p0  * np.cos(np.radians(0))**4 +
        p45 * np.cos(np.radians(45))**4 +
        p90 * np.cos(np.radians(90))**4
    )

    # ── F4: Rule-of-Mixtures longitudinal strength ────────────────────────
    #    σ_L = V_f · σ_f + V_m · σ_m^failure
    sigma_m = MATRIX_STRENGTH_MPa
    df["sigma_L_ROM_MPa"] = (
        Vf * df["fiber_tensile_strength_MPa"] + Vm * sigma_m)

    # ── F5: Thermal knock-down factor ─────────────────────────────────────
    #    Properties degrade as T_test → T_g. Ratio T/T_g captures this.
    df["thermal_knockdown"] = (
        df["temperature_C"] / df["Tg_dry_C"].clip(lower=1.0))

    # ── F6: Effective Vf with moisture correction ─────────────────────────
    #    Moisture swells the matrix, effectively diluting fibre content.
    #    Empirical ΔV_f ≈ -0.5% per unit moisture severity.
    df["effective_Vf_pct"] = (
        df["fiber_volume_pct"] - df["moisture_code"] * 0.5)

    # ── F7: Manufacturing quality index ───────────────────────────────────
    #    Higher method rank + higher pressure → lower void content → 
    #    better property realisation.
    df["mfg_quality_index"] = (
        (6.0 - df["manufacturing_code"]) *
        np.sqrt(df["manufacturing_pressure_psi"].clip(lower=0.1) / 100.0)
    )

    # ── F8: Anisotropy ratio (E_L / E_T) ─────────────────────────────────
    #    High ratio → strongly orthotropic → fibre direction critical.
    df["anisotropy_proxy"] = (
        df["E_L_ROM_GPa"] / df["E_T_HT_GPa"].clip(lower=0.1))

    # ── F9: CNT–interlayer synergy term ───────────────────────────────────
    #    CNTs enhance matrix-dominated properties; combined with interlayer
    #    toughening, synergistic improvement is observed (S2 data).
    df["cnt_il_synergy"] = (
        df["CNT_vol_frac_pct"] *
        (1 + df["interlayer_vol_frac_pct"] / 10.0))

    # ── F10: 0° ply contribution index ────────────────────────────────────
    #    Axial load-carrying capacity proportional to 0° ply fraction ×
    #    fibre-direction strength.
    df["zero_ply_contribution"] = p0 * df["sigma_L_ROM_MPa"]

    # ── F11: Composite density (mixture rule) ─────────────────────────────
    #    ρ_c = V_f · ρ_f + V_m · ρ_m   — critical for specific properties.
    df["composite_density_gcc"] = (
        Vf * df["fiber_density_gcc"] + Vm * MATRIX_DENSITY_gcc)

    # ── F12: Specific strength proxy ──────────────────────────────────────
    #    σ_specific = σ_L_ROM / ρ_c   — key merit index for aerospace design.
    df["sigma_L_specific"] = (
        df["sigma_L_ROM_MPa"] / df["composite_density_gcc"])

    print("  Engineered features: 12 physics-based variables added")
    print(f"  Dataset shape after engineering: {df.shape}")

    # Save engineered dataset
    out_path = f"{OUT_DIR}/CFRP_Engineered_Dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved -> CFRP_Engineered_Dataset.csv")

    return df


def prepare_ml_data(df, target=TARGET_STRENGTH, features=None):
    """
    Extract feature matrix X and target vector y, impute, and scale.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset from preprocess()
    target : str
        Target column name
    features : list[str] or None
        Feature columns (defaults to ALL_FEATURES)

    Returns
    -------
    X_scaled : np.ndarray   — scaled feature matrix
    y        : np.ndarray   — target values
    df_ml    : pd.DataFrame — filtered dataframe (for metadata)
    imputer  : fitted SimpleImputer
    scaler   : fitted RobustScaler
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.impute import SimpleImputer

    if features is None:
        features = ALL_FEATURES

    df_ml = df.dropna(subset=[target]).copy()
    X_raw = df_ml[features].copy()
    y     = df_ml[target].values

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X_imp)

    print(f"\n  ML data prepared:")
    print(f"    Target   : {target}")
    print(f"    Samples  : {len(df_ml)}")
    print(f"    Features : {len(features)} ({len(features)-12} original + 12 engineered)")
    print(f"    Range    : {y.min():.1f} – {y.max():.1f}")

    return X_sc, y, df_ml, imputer, scaler
