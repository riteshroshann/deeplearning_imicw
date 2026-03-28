"""
================================================================================
  main.py — AI-DRIVEN PROPERTY OPTIMISATION OF CFRP COMPOSITES
================================================================================
  Full research pipeline orchestrator.

  Phases:
    1. Dataset construction (8 validated sources)
    2. Physics-validated preprocessing & feature engineering
    3. Exploratory data analysis (Figs 1-2)
    4. Multi-model training (sklearn + PyTorch deep learning)
    5. Stacked ensemble + bootstrap uncertainty (Figs 3-7)
    6. Multi-objective GA optimisation (Fig 8)
    7. Export all results (Fig 9 + CSVs)

  Run:
    python main.py
================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEED, OUT_DIR, ALL_FEATURES, TARGET_STRENGTH, TARGET_MODULUS,
    MPa_TO_ksi, GPa_TO_Msi, GA_BOUNDS, GA_N_GEN, GA_POP_SIZE,
    RANK_TO_FIB, section, subsection, savefig, N_BOOTSTRAP
)

# ── Phase 1: Dataset Construction ────────────────────────────────────────────
from data_builder import build_dataset
raw_df = build_dataset()

# ── Phase 2: Preprocessing & Feature Engineering ────────────────────────────
from preprocessing import preprocess, prepare_ml_data
df = preprocess(raw_df)

# ── Phase 3: EDA ─────────────────────────────────────────────────────────────
section("PHASE 3 — EXPLORATORY DATA ANALYSIS")
from visualization import fig1_eda_overview, fig2_correlation_matrix
fig1_eda_overview(df)
fig2_correlation_matrix(df)

# ── Phase 4: Model Training ──────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_sc, y, df_ml, imputer, scaler = prepare_ml_data(df)
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.20, random_state=SEED)
_, test_indices = train_test_split(
    np.arange(len(df_ml)), test_size=0.20, random_state=SEED)

print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

from training import train_all_models
results, preds, models = train_all_models(X_train, y_train, X_test, y_test)

# ── Phase 4b: Stacked Ensemble ───────────────────────────────────────────────
from ensemble import build_stacked_ensemble, bootstrap_prediction_intervals

results, preds, meta, OOF_PREDS, TEST_PREDS = build_stacked_ensemble(
    models, results, preds, X_train, y_train, X_test, y_test)

y_stack = preds["STACK"]

# ── Phase 4c: Bootstrap Uncertainty ──────────────────────────────────────────
# Use reduced bootstrap for speed (full bootstrap on sklearn models only)
sklearn_models = {k: v for k, v in models.items()
                  if not hasattr(v, 'parameters')
                  and k in ("RF", "ET", "GBM", "SVR")}

pi_lo, pi_hi, pi_cov, pi_wid, r2_ci = bootstrap_prediction_intervals(
    sklearn_models, X_train, y_train, X_test, y_test,
    OOF_PREDS[:, :len(sklearn_models)], y_stack,
    n_boot=min(N_BOOTSTRAP, 200))

# ── Phase 5: Publication Figures ─────────────────────────────────────────────
section("PHASE 5 — GENERATING PUBLICATION FIGURES")

from visualization import (
    fig3_model_performance, fig4_residual_diagnostics,
    fig5_feature_importance, fig6_learning_curves,
    fig7_physics_benchmark, fig8_ga_optimization,
    fig9_results_table
)

lims = fig3_model_performance(
    results, y_test, y_stack, pi_lo, pi_hi, pi_cov, r2_ci,
    X_train, y_train, models, OOF_PREDS)

fig4_residual_diagnostics(y_test, y_stack, df_ml, test_indices, lims)

combined_importance = fig5_feature_importance(models, X_test, y_test)

fig6_learning_curves(results)

# Physics benchmark
from evaluation import physics_benchmark
test_df = df_ml.iloc[test_indices].copy()
rom_strength, rom_orient, m_rom, m_rom_orient = physics_benchmark(
    y_test, test_df)

fig7_physics_benchmark(y_test, y_stack, results,
                       rom_strength, rom_orient,
                       m_rom, m_rom_orient, lims)

# ── Phase 6: GA Optimisation ────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from optimization import run_ga

# Train modulus surrogate
subsection("Training modulus surrogate model")
X_mod, y_mod, df_mod, _, _ = prepare_ml_data(df, target=TARGET_MODULUS)
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_mod, y_mod, test_size=0.20, random_state=SEED)
from sklearn.metrics import r2_score
surr_mod = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
surr_mod.fit(X_tr_m, y_tr_m)
print(f"  Modulus surrogate R² = {r2_score(y_te_m, surr_mod.predict(X_te_m)):.4f}")

# Use RF as strength surrogate (most stable for GA evaluation)
strength_model = models["RF"]

ga_results = run_ga(strength_model, surr_mod, imputer, scaler)

fig8_ga_optimization(ga_results)

# ── Phase 7: Export Results ──────────────────────────────────────────────────
section("PHASE 7 — EXPORTING RESULTS")

fig9_results_table(results, models)

# Model performance CSV
all_names = list(results.keys())
rows_out = []
for name in all_names:
    m = results[name]
    rows_out.append({
        "Model": name,
        "Type": ("Stacked" if name == "STACK"
                 else "DNN" if name.startswith("DL")
                 else "Classical"),
        "R2": round(m["R2"], 4),
        "RMSE_MPa": round(m["RMSE"], 2),
        "MAE_MPa": round(m["MAE"], 2),
        "MAPE_pct": round(m["MAPE"], 2),
        "Bias_MPa": round(m["bias"], 2),
        "CV_R2_mean": round(m["CV_mean"], 4),
        "CV_R2_std": round(m["CV_std"], 4),
        "Train_time_s": round(m.get("time_s", 0), 2),
    })
pd.DataFrame(rows_out).to_csv(f"{OUT_DIR}/Model_Performance_Summary.csv", index=False)

# Optimal design CSV
opt = ga_results["optimal_chrom"]
pd.DataFrame({
    "Feature": ALL_FEATURES,
    "Optimal_Value": opt,
    "Min_Bound": [v[0] for v in GA_BOUNDS.values()],
    "Max_Bound": [v[1] for v in GA_BOUNDS.values()],
}).to_csv(f"{OUT_DIR}/GA_Optimal_Design.csv", index=False)

# Pareto front CSV
pd.DataFrame({
    "predicted_strength_MPa": ga_results["pareto_strength"],
    "predicted_modulus_GPa": ga_results["pareto_modulus"],
}).to_csv(f"{OUT_DIR}/GA_Pareto_Front.csv", index=False)

# Dataset summary
src_summary = (df.groupby("source_id")
               .agg(n_records=("source_id", "count"),
                    reference=("source_ref", "first"),
                    doi=("doi_or_url", "first"),
                    data_types=("data_type", lambda x: "/".join(x.unique())))
               .reset_index())
src_summary.to_csv(f"{OUT_DIR}/Dataset_Source_Summary.csv", index=False)

# Feature importance CSV
if "RF" in models:
    from sklearn.inspection import permutation_importance
    rf_imp = models["RF"].feature_importances_
    perm = permutation_importance(models["RF"], X_test, y_test,
                                  n_repeats=15, random_state=SEED, n_jobs=-1)
    pd.DataFrame({
        "Feature": ALL_FEATURES,
        "RF_impurity_importance": rf_imp,
        "Permutation_importance": perm.importances_mean,
        "Permutation_std": perm.importances_std,
    }).sort_values("RF_impurity_importance", ascending=False).to_csv(
        f"{OUT_DIR}/Feature_Importance.csv", index=False)

# Bootstrap CI CSV
r2_lo, r2_pt, r2_hi = r2_ci
pd.DataFrame({
    "Model": ["STACK"],
    "R2_lower": [r2_lo], "R2_point": [r2_pt], "R2_upper": [r2_hi],
    "PI_coverage_pct": [pi_cov], "PI_mean_width_MPa": [pi_wid],
}).to_csv(f"{OUT_DIR}/Bootstrap_Confidence_Intervals.csv", index=False)

print(f"\n  All output files in {OUT_DIR}/:")
for fname in sorted(os.listdir(OUT_DIR)):
    p = os.path.join(OUT_DIR, fname)
    sz = os.path.getsize(p) / 1024
    print(f"    {fname:<52}  {sz:>7.1f} KB")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
section("PROJECT COMPLETE — FINAL SUMMARY")

opt_s = ga_results["opt_strength"]
opt_m = ga_results["opt_modulus"]
optimal_chrom = ga_results["optimal_chrom"]
fib_name = RANK_TO_FIB.get(int(optimal_chrom[0]), "IM7")

print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  AI-DRIVEN PROPERTY OPTIMISATION OF CFRP COMPOSITES                ║
  ║  Deep Learning & Material Informatics Research Pipeline             ║
  ╚══════════════════════════════════════════════════════════════════════╝

  DATASET
  ───────
  Sources     : 8 peer-reviewed publications + 1 FAA-validated database
  Records     : {len(df)} (aerospace-filtered)
  Data types  : Experimental / Mean_value (transparently flagged)
  Features    : 17 original + 12 physics-engineered = {len(ALL_FEATURES)} total
  Target      : Tensile Strength (MPa)
  Temp range  : -80°C to +150°C | Environments: CTD/RTD/ETD/ETW

  BEST MODEL: STACKED ENSEMBLE
  ─────────────────────────────
  Base learners   : RF + ET + GBM + SVR + 5×DNN
  Meta-learner    : Ridge regression (OOF predictions)
  Test R²         : {results['STACK']['R2']:.4f}   95% CI: [{r2_lo:.4f}, {r2_hi:.4f}]
  RMSE            : {results['STACK']['RMSE']:.1f} MPa
  MAE             : {results['STACK']['MAE']:.1f} MPa
  MAPE            : {results['STACK']['MAPE']:.2f}%
  5-Fold CV R²    : {results['STACK']['CV_mean']:.4f} ± {results['STACK']['CV_std']:.4f}
  Bootstrap PI    : {pi_cov:.1f}% coverage at 95% nominal
  Bias            : {results['STACK']['bias']:.1f} MPa

  PHYSICS BENCHMARK
  ─────────────────
  Rule-of-Mixtures R²  : {m_rom['R2']:.4f}
  ROM + Orientation R² : {m_rom_orient['R2']:.4f}
  ML improvement ΔR²   : {results['STACK']['R2'] - max(m_rom['R2'], m_rom_orient['R2']):.4f}

  GA OPTIMISATION
  ───────────────
  Approach    : NSGA-II (strength + modulus) with crowding distance
  Generations : {GA_N_GEN}  |  Population : {GA_POP_SIZE}
  Optimal:
    Fibre          : {fib_name}
    Vf             : {optimal_chrom[3]:.1f}%
    0°/45°/90°     : {optimal_chrom[4]:.0f}% / {optimal_chrom[5]:.0f}% / {optimal_chrom[6]:.0f}%
    Predicted σ    : {opt_s:.1f} MPa ({opt_s * MPa_TO_ksi:.0f} ksi)
    Predicted E    : {opt_m:.1f} GPa ({opt_m * GPa_TO_Msi:.1f} Msi)
    Pareto front   : {len(ga_results['pareto_idx'])} solutions

  OUTPUT FILES (9 figures + 7 CSVs)
  ──────────────────────────────────
  Fig1_EDA_Overview.png          Fig2_Correlation_Matrix.png
  Fig3_Model_Performance.png     Fig4_Residual_Diagnostics.png
  Fig5_Feature_Importance.png    Fig6_DNN_Learning_Curves.png
  Fig7_Physics_vs_ML.png         Fig8_GA_Optimisation.png
  Fig9_Results_Table.png
""")
print("=" * 72)
print("  ALL PHASES COMPLETE")
print("=" * 72)
