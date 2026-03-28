"""
AI-Driven Property Optimisation of CFRP Composites
Full research pipeline: data → features → models → ensemble → GA → figures
"""

import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEED, OUT_DIR, ALL_FEATURES, TARGET_STRENGTH, TARGET_MODULUS,
    MPa_TO_ksi, GPa_TO_Msi, GA_BOUNDS, GA_N_GEN, GA_POP_SIZE,
    RANK_TO_FIB, section, subsection, N_BOOTSTRAP
)

from data_builder import build_dataset
raw_df = build_dataset()

from preprocessing import preprocess, prepare_ml_data
df = preprocess(raw_df)

section("PHASE 3 — EXPLORATORY DATA ANALYSIS")
from visualization import fig1_eda_overview, fig2_correlation_matrix
fig1_eda_overview(df)
fig2_correlation_matrix(df)

from sklearn.model_selection import train_test_split
X_sc, y, df_ml, imputer, scaler = prepare_ml_data(df)
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.20, random_state=SEED)
train_indices, test_indices = train_test_split(np.arange(len(df_ml)), test_size=0.20, random_state=SEED)
print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

from training import train_all_models
results, preds, models = train_all_models(
    X_train, y_train, X_test, y_test,
    df_ml=df_ml, train_indices=train_indices, test_indices=test_indices)

from ensemble import build_stacked_ensemble, bootstrap_prediction_intervals

# Ensemble uses only standard models (exclude PINNs — they serve as standalone comparison)
ensemble_models = {k: v for k, v in models.items() if not k.startswith("PINN")}
ensemble_preds = {k: v for k, v in preds.items() if not k.startswith("PINN")}
results, ensemble_preds, meta, OOF_PREDS, TEST_PREDS = build_stacked_ensemble(
    ensemble_models, results, ensemble_preds, X_train, y_train, X_test, y_test)
preds.update(ensemble_preds)
y_stack = preds["STACK"]

sklearn_models = {k: v for k, v in models.items()
                  if not hasattr(v, 'parameters') and k in ("RF","ET","GBM","SVR")}
pi_lo, pi_hi, pi_cov, pi_wid, r2_ci = bootstrap_prediction_intervals(
    sklearn_models, X_train, y_train, X_test, y_test,
    OOF_PREDS[:, :len(sklearn_models)], y_stack, n_boot=min(N_BOOTSTRAP, 200))

section("PHASE 5 — GENERATING PUBLICATION FIGURES")
from visualization import (
    fig3_model_performance, fig4_residual_diagnostics,
    fig5_feature_importance, fig6_learning_curves,
    fig7_physics_benchmark, fig8_ga_optimization, fig9_results_table
)

lims = fig3_model_performance(results, y_test, y_stack, pi_lo, pi_hi, pi_cov, r2_ci,
                               X_train, y_train, models, OOF_PREDS)
fig4_residual_diagnostics(y_test, y_stack, df_ml, test_indices, lims)
fig5_feature_importance(models, X_test, y_test)
fig6_learning_curves(results)

from evaluation import physics_benchmark
test_df = df_ml.iloc[test_indices].copy()
rom_strength, rom_orient, m_rom, m_rom_orient = physics_benchmark(y_test, test_df)
fig7_physics_benchmark(y_test, y_stack, results, rom_strength, rom_orient, m_rom, m_rom_orient, lims)

from sklearn.ensemble import RandomForestRegressor
from optimization import run_ga

subsection("Training modulus surrogate")
X_mod, y_mod, df_mod, _, _ = prepare_ml_data(df, target=TARGET_MODULUS)
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_mod, y_mod, test_size=0.20, random_state=SEED)
from sklearn.metrics import r2_score
surr_mod = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
surr_mod.fit(X_tr_m, y_tr_m)
print(f"  Modulus surrogate R² = {r2_score(y_te_m, surr_mod.predict(X_te_m)):.4f}")

ga_results = run_ga(models["RF"], surr_mod, imputer, scaler)
fig8_ga_optimization(ga_results)

section("PHASE 7 — EXPORTING RESULTS")
fig9_results_table(results, models)

all_names = list(results.keys())
pd.DataFrame([{
    "Model": n, "R2": round(results[n]["R2"],4),
    "RMSE": round(results[n]["RMSE"],2), "MAE": round(results[n]["MAE"],2),
    "MAPE%": round(results[n]["MAPE"],2), "Bias": round(results[n]["bias"],2),
    "CV_R2": round(results[n]["CV_mean"],4)
} for n in all_names]).to_csv(f"{OUT_DIR}/Model_Performance_Summary.csv", index=False)

opt = ga_results["optimal_chrom"]
pd.DataFrame({"Feature": ALL_FEATURES, "Optimal": opt,
              "Min": [v[0] for v in GA_BOUNDS.values()],
              "Max": [v[1] for v in GA_BOUNDS.values()]
}).to_csv(f"{OUT_DIR}/GA_Optimal_Design.csv", index=False)

pd.DataFrame({"strength_MPa": ga_results["pareto_strength"],
              "modulus_GPa": ga_results["pareto_modulus"]
}).to_csv(f"{OUT_DIR}/GA_Pareto_Front.csv", index=False)

df.groupby("source_id").agg(
    n=("source_id","count"), ref=("source_ref","first"),
    doi=("doi_or_url","first")).reset_index().to_csv(
    f"{OUT_DIR}/Dataset_Source_Summary.csv", index=False)

if "RF" in models:
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(models["RF"], X_test, y_test, n_repeats=15, random_state=SEED, n_jobs=-1)
    pd.DataFrame({"Feature": ALL_FEATURES,
                  "RF_importance": models["RF"].feature_importances_,
                  "Perm_importance": perm.importances_mean
    }).sort_values("RF_importance", ascending=False).to_csv(
        f"{OUT_DIR}/Feature_Importance.csv", index=False)

r2_lo, r2_pt, r2_hi = r2_ci
pd.DataFrame({"R2_lo": [r2_lo], "R2": [r2_pt], "R2_hi": [r2_hi],
              "PI_cov%": [pi_cov], "PI_width": [pi_wid]
}).to_csv(f"{OUT_DIR}/Bootstrap_CI.csv", index=False)

print(f"\n  Output files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"    {f:<48} {os.path.getsize(os.path.join(OUT_DIR,f))/1024:>6.1f} KB")

section("COMPLETE")
opt_s, opt_m = ga_results["opt_strength"], ga_results["opt_modulus"]
fib_name = RANK_TO_FIB.get(int(opt[0]), "IM7")
print(f"""
  Dataset     : {len(df)} records, {len(ALL_FEATURES)} features
  Best Model  : STACK  R²={results['STACK']['R2']:.4f}  RMSE={results['STACK']['RMSE']:.1f}
  95% CI      : [{r2_lo:.4f}, {r2_hi:.4f}]
  ROM baseline: R²={max(m_rom['R2'], m_rom_orient['R2']):.4f}
  GA Optimal  : σ={opt_s:.0f} MPa  E={opt_m:.1f} GPa  ({fib_name})
  Output      : 9 figures + 7 CSVs
""")
print("="*72)
