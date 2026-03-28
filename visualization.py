"""
Publication-quality figures for CFRP property prediction research.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from config import (
    PALETTE, CORR_FEATURES, CORR_LABELS, ALL_FEATURES,
    TARGET_STRENGTH, savefig, OUT_DIR
)

_CB = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261",
       "#e76f51", "#606c38", "#283618", "#dda15e"]


def fig1_eda_overview(df):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("CFRP Aerospace Dataset — Exploratory Data Analysis",
                 fontsize=14, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.42,
                           top=0.93, bottom=0.06, left=0.06, right=0.97)

    # A — Records per source
    ax = fig.add_subplot(gs[0, 0])
    src = df["source_id"].value_counts().sort_index()
    bars = ax.bar(src.index, src.values, color=_CB[:len(src)],
                  edgecolor="white", linewidth=1.2, width=0.65)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.8,
                str(int(b.get_height())), ha="center", fontsize=7.5,
                fontweight="bold")
    ax.set_title("(A)  Records per Source", fontsize=10, pad=8)
    ax.set_xlabel("Source ID"); ax.set_ylabel("Count")
    ax.set_ylim(0, src.max()*1.22)

    # B — Strength distribution (single y-axis, no twin-axis overlap)
    ax = fig.add_subplot(gs[0, 1])
    s_valid = df[TARGET_STRENGTH].dropna()
    ax.hist(s_valid, bins=28, density=True, color=PALETTE["primary"],
            alpha=0.7, edgecolor="white", linewidth=0.5)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(s_valid, bw_method="scott")
    x_kde = np.linspace(s_valid.min()-50, s_valid.max()+50, 300)
    ax.plot(x_kde, kde(x_kde), color=PALETTE["secondary"], lw=2.2)
    ax.axvline(s_valid.mean(), color="#C0392B", lw=1.5, ls="--",
               label=f"μ = {s_valid.mean():.0f}")
    ax.axvline(s_valid.median(), color="#B7950B", lw=1.5, ls=":",
               label=f"Med = {s_valid.median():.0f}")
    ax.set_title("(B)  Strength Distribution", fontsize=10, pad=8)
    ax.set_xlabel("Strength (MPa)"); ax.set_ylabel("Density")
    ax.legend(fontsize=7.5, framealpha=0.9)

    # C — Test type breakdown
    ax = fig.add_subplot(gs[0, 2])
    tt = df["test_type"].value_counts().head(8)
    labels = [t.replace("_", " ")[:22] for t in tt.index[::-1]]
    ax.barh(labels, tt.values[::-1], color=PALETTE["accent1"],
            alpha=0.85, edgecolor="white", height=0.65)
    ax.set_title("(C)  Test Type Distribution", fontsize=10, pad=8)
    ax.set_xlabel("Count")
    ax.tick_params(axis="y", labelsize=7)

    # D — Temperature vs strength
    ax = fig.add_subplot(gs[1, 0])
    env_clr = {"CTD":"#5b9bd5", "RTD":"#1E8449",
               "ETD":"#B7950B", "ETW":"#C0392B"}
    for env, grp in df.dropna(subset=[TARGET_STRENGTH]).groupby("environment"):
        ax.scatter(grp["temperature_C"], grp[TARGET_STRENGTH],
                   color=env_clr.get(env,"#aaa"), label=env,
                   alpha=0.7, s=30, edgecolors="white", linewidth=0.3)
    ax.set_title("(D)  Temperature vs Strength", fontsize=10, pad=8)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Strength (MPa)")
    ax.legend(fontsize=7, title_fontsize=7.5, title="Env",
              markerscale=0.8, framealpha=0.9)

    # E — Fibre type box-plot
    ax = fig.add_subplot(gs[1, 1])
    fibre_order = ["T300","T700","T800","IM7","HS Carbon",
                   "HM Carbon","IM Carbon","Carbon Fiber"]
    fibre_data = [df[df["fiber_type"]==f][TARGET_STRENGTH].dropna().values
                  for f in fibre_order if f in df["fiber_type"].values]
    fibre_lbls = [f for f in fibre_order if f in df["fiber_type"].values]
    bp = ax.boxplot(fibre_data, labels=fibre_lbls, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.8),
                    flierprops=dict(markersize=3))
    for patch, col in zip(bp["boxes"], _CB):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_title("(E)  Strength by Fibre Type", fontsize=10, pad=8)
    ax.set_ylabel("Strength (MPa)")
    ax.tick_params(axis="x", labelsize=7, rotation=25)

    # F — Environmental degradation
    ax = fig.add_subplot(gs[1, 2])
    rtd = df[(df["env_code"]==1)&(df["moisture_code"]==0)][TARGET_STRENGTH].mean()
    wet = df[df["moisture_code"]==1][TARGET_STRENGTH].mean()
    etd = df[df["env_code"]==2][TARGET_STRENGTH].mean()
    etw = df[df["env_code"]==3][TARGET_STRENGTH].mean()
    cats = ["RTD\nDry", "Wet", "ETD", "ETW"]
    vals = [rtd, wet, etd, etw]
    bars = ax.bar(cats, vals, color=_CB[:4], edgecolor="white",
                  linewidth=1.2, width=0.55)
    for b, v in zip(bars, vals):
        if not np.isnan(v) and rtd > 0:
            ax.text(b.get_x()+b.get_width()/2, v+8,
                    f"{v/rtd*100:.0f}%", ha="center", fontsize=7.5,
                    fontweight="bold")
    ax.set_title("(F)  Mean Strength by Environment", fontsize=10, pad=8)
    ax.set_ylabel("Strength (MPa)")

    savefig("Fig1_EDA_Overview", fig)
    print("  Fig 1 saved.")


def fig2_correlation_matrix(df):
    corr_df = df[CORR_FEATURES].dropna(subset=[TARGET_STRENGTH])
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Feature Correlation — Pearson & Spearman",
                 fontsize=13, fontweight="bold", y=0.97)
    plt.subplots_adjust(wspace=0.35, top=0.90, bottom=0.12,
                        left=0.08, right=0.96)

    cmap = LinearSegmentedColormap.from_list(
        "cfrp", ["#C0392B","white","#1B4F72"])

    for ax, method, title in [
        (axes[0], "pearson",  "(A)  Pearson (Linear)"),
        (axes[1], "spearman", "(B)  Spearman (Monotonic)"),
    ]:
        cm = corr_df.corr(method=method)
        mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
        sns.heatmap(cm, mask=mask, ax=ax, cmap=cmap,
                    vmin=-1, vmax=1, center=0,
                    xticklabels=CORR_LABELS, yticklabels=CORR_LABELS,
                    annot=True, fmt=".2f", annot_kws={"size": 5.5},
                    linewidths=0.3, linecolor="#eee",
                    cbar_kws={"shrink": 0.75, "label": ""})
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.tick_params(axis="x", rotation=50, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)

    savefig("Fig2_Correlation_Matrix", fig)
    print("  Fig 2 saved.")

    target_corr = (corr_df.corr(method="spearman")[TARGET_STRENGTH]
                   .drop(TARGET_STRENGTH).abs()
                   .sort_values(ascending=False))
    print("\n  Top 8 Spearman |r|:")
    for feat, val in target_corr.head(8).items():
        print(f"    {feat:<36}  |r| = {val:.3f}")


def fig3_model_performance(results, y_test, y_stack,
                           pi_lo, pi_hi, pi_cov, r2_ci,
                           X_train, y_train, models, OOF_PREDS):
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.linear_model import Ridge

    all_names = list(results.keys())
    x = np.arange(len(all_names))
    bar_c = [PALETTE.get(m, PALETTE["neutral"]) for m in all_names]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Model Performance — CFRP Strength Prediction",
                 fontsize=13, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.06, right=0.97)

    # A — R²
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(x, [results[m]["R2"] for m in all_names],
                  color=bar_c, alpha=0.88, edgecolor="white",
                  linewidth=1.2, width=0.6)
    if "STACK" in all_names:
        bars[all_names.index("STACK")].set_edgecolor("#D4AC0D")
        bars[all_names.index("STACK")].set_linewidth(2.5)
    ax.axhline(0.90, color="green", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(all_names, fontsize=7)
    ax.set_ylim(-0.15, 1.12)
    ax.set_title("(A)  R²", fontsize=10, pad=8)
    ax.set_ylabel("R²")
    for i, m in enumerate(all_names):
        ax.text(i, results[m]["R2"]+0.02, f"{results[m]['R2']:.3f}",
                ha="center", fontsize=6.5, fontweight="bold")

    # B — RMSE
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, [results[m]["RMSE"] for m in all_names],
           color=bar_c, alpha=0.88, edgecolor="white", linewidth=1.2, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(all_names, fontsize=7)
    ax.set_title("(B)  RMSE (MPa)", fontsize=10, pad=8)
    ax.set_ylabel("RMSE")
    for i, m in enumerate(all_names):
        ax.text(i, results[m]["RMSE"]+2, f"{results[m]['RMSE']:.0f}",
                ha="center", fontsize=6.5, fontweight="bold")

    # C — MAE
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x, [results[m]["MAE"] for m in all_names],
           color=bar_c, alpha=0.88, edgecolor="white", linewidth=1.2, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(all_names, fontsize=7)
    ax.set_title("(C)  MAE (MPa)", fontsize=10, pad=8)
    ax.set_ylabel("MAE")
    for i, m in enumerate(all_names):
        ax.text(i, results[m]["MAE"]+1, f"{results[m]['MAE']:.0f}",
                ha="center", fontsize=6.5, fontweight="bold")

    # D — CV box-plot
    ax = fig.add_subplot(gs[1, 0])
    cv_data = []
    for m_name in all_names:
        if m_name == "STACK":
            cv_data.append(cross_val_score(
                Ridge(alpha=1.0), OOF_PREDS, y_train, cv=kf, scoring="r2"))
        elif m_name in models and not hasattr(models[m_name], 'parameters'):
            cv_data.append(cross_val_score(
                models[m_name], X_train, y_train, cv=kf,
                scoring="r2", n_jobs=-1))
        else:
            cv_data.append(np.array([results[m_name]["CV_mean"]]*5))
    bp = ax.boxplot(cv_data, labels=all_names, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.8),
                    flierprops=dict(markersize=3))
    for patch, col in zip(bp["boxes"], bar_c):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_title("(D)  5-Fold CV R²", fontsize=10, pad=8)
    ax.set_ylabel("CV R²")
    ax.tick_params(axis="x", labelsize=7)

    # E — Actual vs Predicted
    ax = fig.add_subplot(gs[1, 1])
    resid_c = np.abs(y_test - y_stack)
    sc = ax.scatter(y_test, y_stack, c=resid_c, cmap="RdYlGn_r",
                    s=40, alpha=0.8, edgecolors="white", linewidth=0.3,
                    vmin=0, vmax=np.percentile(resid_c, 95))
    plt.colorbar(sc, ax=ax, label="|Residual|", shrink=0.82, pad=0.02)
    lims = [min(y_test.min(), y_stack.min())-80,
            max(y_test.max(), y_stack.max())+80]
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2_lo, r2_pt, r2_hi = r2_ci
    txt = (f"R² = {results['STACK']['R2']:.4f}\n"
           f"95% CI [{r2_lo:.3f}, {r2_hi:.3f}]\n"
           f"RMSE = {results['STACK']['RMSE']:.1f}\n"
           f"PI cov = {pi_cov:.1f}%")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=7.5,
            va="top", bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="#FEF9E7", alpha=0.9, edgecolor="#ccc"))
    ax.set_title("(E)  Actual vs Predicted (Stack)", fontsize=10, pad=8)
    ax.set_xlabel("Actual (MPa)"); ax.set_ylabel("Predicted (MPa)")

    # F — MAPE
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(x, [results[m]["MAPE"] for m in all_names],
           color=bar_c, alpha=0.88, edgecolor="white", linewidth=1.2, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(all_names, fontsize=7)
    ax.set_title("(F)  MAPE (%)", fontsize=10, pad=8)
    ax.set_ylabel("MAPE")
    for i, m in enumerate(all_names):
        ax.text(i, results[m]["MAPE"]+0.3, f"{results[m]['MAPE']:.1f}%",
                ha="center", fontsize=6.5, fontweight="bold")

    savefig("Fig3_Model_Performance", fig)
    print("  Fig 3 saved.")
    return lims


def fig4_residual_diagnostics(y_test, y_stack, df_ml,
                              test_indices, lims):
    resids = y_test - y_stack
    std_res = (resids - resids.mean()) / resids.std()
    n_ = len(std_res)
    y_sorted = np.sort(std_res)
    q_th = stats.norm.ppf((np.arange(1, n_+1)-0.375) / (n_+0.25))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Residual Diagnostics — Stacked Ensemble",
                 fontsize=13, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40,
                           top=0.93, bottom=0.06, left=0.06, right=0.97)

    # A — Residuals vs predicted
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(y_stack, resids, c=PALETTE["primary"], alpha=0.7,
               s=28, edgecolors="white", linewidth=0.2)
    ax.axhline(0, color=PALETTE["secondary"], lw=1.5, ls="--")
    ax.axhline(2*resids.std(), color="#999", lw=0.8, ls=":")
    ax.axhline(-2*resids.std(), color="#999", lw=0.8, ls=":", label="±2σ")
    ax.set_xlabel("Predicted (MPa)"); ax.set_ylabel("Residual (MPa)")
    ax.set_title("(A)  Residuals vs Predicted", fontsize=10, pad=8)
    ax.legend(fontsize=7)

    # B — Q-Q
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(q_th, y_sorted, s=22, color=PALETTE["primary"],
               alpha=0.7, edgecolors="white", linewidth=0.2)
    ax.plot(q_th[[0,-1]], q_th[[0,-1]], color=PALETTE["secondary"],
            lw=1.8, ls="--")
    ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
    ax.set_title("(B)  Normal Q–Q Plot", fontsize=10, pad=8)

    # C — Residual histogram
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(resids, bins=25, color=PALETTE["primary"], alpha=0.75,
            edgecolor="white", density=True)
    x_fit = np.linspace(resids.min()-50, resids.max()+50, 200)
    ax.plot(x_fit, stats.norm.pdf(x_fit, resids.mean(), resids.std()),
            color=PALETTE["secondary"], lw=2)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Residual (MPa)"); ax.set_ylabel("Density")
    ax.set_title("(C)  Residual Distribution", fontsize=10, pad=8)

    # D — Residuals by fibre type
    ax = fig.add_subplot(gs[1, 0])
    fibre_labels = df_ml["fiber_type"].values
    test_fibre = fibre_labels[test_indices]
    unique_f = sorted(set(test_fibre))
    resid_by_f = [resids[test_fibre==f] for f in unique_f]
    bp2 = ax.boxplot(resid_by_f, labels=unique_f, patch_artist=True,
                     medianprops=dict(color="black", lw=1.8),
                     flierprops=dict(markersize=3))
    for patch, col in zip(bp2["boxes"], _CB):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.axhline(0, color=PALETTE["secondary"], lw=1.2, ls="--")
    ax.set_ylabel("Residual (MPa)")
    ax.set_title("(D)  Residuals by Fibre", fontsize=10, pad=8)
    ax.tick_params(axis="x", rotation=25, labelsize=6.5)

    # E — Residuals vs temperature
    ax = fig.add_subplot(gs[1, 1])
    test_temp = df_ml["temperature_C"].values[test_indices]
    ax.scatter(test_temp, resids, c=PALETTE["accent3"],
               alpha=0.7, s=28, edgecolors="white", linewidth=0.2)
    ax.axhline(0, color=PALETTE["secondary"], lw=1.5, ls="--")
    m_t, b_t, r_t, p_t, _ = stats.linregress(test_temp, resids)
    x_t = np.linspace(test_temp.min(), test_temp.max(), 100)
    ax.plot(x_t, m_t*x_t+b_t, color=PALETTE["secondary"],
            lw=1.8, alpha=0.7, label=f"r={r_t:.3f}")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Residual (MPa)")
    ax.set_title("(E)  Residuals vs Temperature", fontsize=10, pad=8)
    ax.legend(fontsize=7)

    # F — Actual vs Predicted by fibre
    ax = fig.add_subplot(gs[1, 2])
    fibre_cm = {"T300":"#264653", "T700":"#2a9d8f", "T800":"#e9c46a",
                "IM7":"#e76f51", "HS Carbon":"#606c38",
                "HM Carbon":"#283618", "IM Carbon":"#6E2F1A",
                "Carbon Fiber":"#717D7E"}
    for fib in unique_f:
        mask = test_fibre == fib
        ax.scatter(y_test[mask], y_stack[mask],
                   color=fibre_cm.get(fib, "#aaa"),
                   s=32, alpha=0.8, edgecolors="white",
                   linewidth=0.3, label=fib)
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual (MPa)"); ax.set_ylabel("Predicted (MPa)")
    ax.set_title("(F)  By Fibre Type", fontsize=10, pad=8)
    ax.legend(fontsize=6, ncol=2, framealpha=0.9)

    savefig("Fig4_Residual_Diagnostics", fig)
    print("  Fig 4 saved.")


def fig5_feature_importance(models, X_test, y_test):
    from sklearn.inspection import permutation_importance

    rf = models.get("RF")
    if rf is None:
        print("  Skipping Fig 5.")
        return

    rf_imp = pd.Series(rf.feature_importances_,
                       index=ALL_FEATURES).sort_values(ascending=False)
    perm = permutation_importance(rf, X_test, y_test,
                                  n_repeats=15, random_state=42, n_jobs=-1)
    perm_ser = pd.Series(perm.importances_mean,
                         index=ALL_FEATURES).sort_values(ascending=False)
    perm_std = pd.Series(perm.importances_std, index=ALL_FEATURES)

    rf_rank = rf_imp.rank(ascending=False)
    perm_rank = perm_ser.rank(ascending=False)
    combined = (rf_rank + perm_rank).sort_values().head(20)

    short = {f: f.replace("_", " ").replace("MPa","").replace("GPa","")
             .replace("pct","%").strip()[:28]
             for f in ALL_FEATURES}

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Feature Importance — Top 20",
                 fontsize=13, fontweight="bold", y=0.97)
    plt.subplots_adjust(wspace=0.50, top=0.90, bottom=0.06,
                        left=0.10, right=0.97)

    for ax, ser, err, title, color in [
        (axes[0], rf_imp.head(20).sort_values(ascending=True),
         None, "(A)  RF Impurity", PALETTE["primary"]),
        (axes[1], perm_ser.head(20).sort_values(ascending=True),
         perm_std, "(B)  Permutation", PALETTE["accent1"]),
    ]:
        feats = ser.index.tolist()
        vals = ser.values
        ax.barh(range(len(feats)), vals, color=color,
                edgecolor="white", linewidth=0.6, alpha=0.85)
        if err is not None:
            ax.errorbar(vals, range(len(feats)),
                        xerr=err[feats].values,
                        fmt="none", color="black", capsize=2, linewidth=0.8)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([short.get(f, f) for f in feats], fontsize=6.5)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        ax.set_xlabel("Score")

    ax = axes[2]
    cfeats = combined.index.tolist()
    ax.barh(range(len(cfeats)), combined.values[::-1],
            color=PALETTE["accent2"], alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(cfeats)))
    ax.set_yticklabels([short.get(f, f) for f in cfeats[::-1]], fontsize=6.5)
    ax.set_title("(C)  Combined Rank", fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel("Rank Score (lower=better)")

    savefig("Fig5_Feature_Importance", fig)
    print("  Fig 5 saved.")
    return combined


def fig6_learning_curves(results):
    dl_keys = [k for k in results if k.startswith("DL")]
    if not dl_keys:
        print("  Skipping Fig 6.")
        return

    n_plots = len(dl_keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots+2, 4.5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("DNN Training Curves",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.subplots_adjust(wspace=0.35, top=0.82, bottom=0.14,
                        left=0.06, right=0.97)

    for i, key in enumerate(dl_keys):
        ax = axes[i]
        loss = results[key].get("loss")
        vloss = results[key].get("val_loss")
        if loss:
            epochs = np.arange(1, len(loss)+1)
            ax.plot(epochs, loss, color=PALETTE.get(key, PALETTE["primary"]),
                    lw=2, label="Train", alpha=0.8)
            if vloss:
                ax.plot(epochs[:len(vloss)], vloss[:len(epochs)],
                        color=PALETTE["secondary"], lw=1.5,
                        ls="--", label="Val", alpha=0.8)
            n_iter = results[key].get("n_iter")
            if n_iter:
                ax.axvline(n_iter, color="#999", ls=":", lw=1, alpha=0.7)
        ax.set_title(f"{key}  R²={results[key]['R2']:.3f}",
                     fontsize=8.5, fontweight="bold", pad=6)
        ax.set_xlabel("Epoch", fontsize=8)
        if i == 0:
            ax.set_ylabel("Loss", fontsize=8)
        ax.set_yscale("log")
        ax.legend(fontsize=7)

    savefig("Fig6_DNN_Learning_Curves", fig)
    print("  Fig 6 saved.")


def fig7_physics_benchmark(y_test, y_stack, results,
                           rom_strength, rom_orient,
                           m_rom, m_rom_orient, lims):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("ML vs Classical Micromechanics",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.subplots_adjust(wspace=0.32, top=0.84, bottom=0.12,
                        left=0.05, right=0.97)

    plot_data = [
        (rom_strength, m_rom,
         "(A)  Rule-of-Mixtures (UD)", PALETTE["neutral"]),
        (rom_orient, m_rom_orient,
         "(B)  Orientation-Weighted ROM", PALETTE["accent2"]),
        (y_stack, results["STACK"],
         "(C)  Stacked DL Ensemble", PALETTE["primary"]),
    ]
    for ax, (y_pred, met, title, col) in zip(axes, plot_data):
        ax.scatter(y_test, y_pred, color=col, alpha=0.7,
                   s=32, edgecolors="white", linewidth=0.3)
        ax.plot(lims, lims, "k--", lw=1.2, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual (MPa)")
        ax.set_ylabel("Predicted (MPa)")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        txt = f"R² = {met['R2']:.4f}\nRMSE = {met['RMSE']:.1f}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                facecolor="#FEF9E7", alpha=0.9, edgecolor="#ccc"))

    savefig("Fig7_Physics_vs_ML", fig)
    print("  Fig 7 saved.")


def fig8_ga_optimization(ga_results):
    from config import GA_N_GEN, MPa_TO_ksi, GPa_TO_Msi

    opt_chrom = ga_results["optimal_chrom"]
    opt_s = ga_results["opt_strength"]
    opt_m = ga_results["opt_modulus"]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Multi-Objective GA Optimisation — CFRP Design",
                 fontsize=13, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40,
                           top=0.93, bottom=0.06, left=0.06, right=0.97)
    gens_ = np.arange(1, GA_N_GEN+1)

    # A — Convergence
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(gens_, ga_results["gen_best_s"], color=PALETTE["primary"],
            lw=2.2, label="Best σ")
    ax.plot(gens_, ga_results["gen_mean_s"], color=PALETTE["accent2"],
            lw=1.2, ls="--", alpha=0.7, label="Mean σ")
    ax.fill_between(gens_, ga_results["gen_mean_s"],
                    ga_results["gen_best_s"], alpha=0.08,
                    color=PALETTE["primary"])
    ax.set_title("(A)  Strength Convergence", fontsize=10, pad=8)
    ax.set_xlabel("Generation"); ax.set_ylabel("σ (MPa)")
    ax.legend(fontsize=7)

    # B — Pareto front
    ax = fig.add_subplot(gs[0, 1])
    final_fit = ga_results["final_fit"]
    ax.scatter(final_fit[:,0], final_fit[:,1], c="#ddd",
               s=20, alpha=0.5, edgecolors="none")
    ax.scatter(ga_results["pareto_strength"], ga_results["pareto_modulus"],
               color=PALETTE["secondary"], s=50, edgecolors="white",
               linewidth=1, zorder=5,
               label=f"Pareto (n={len(ga_results['pareto_idx'])})")
    ax.scatter([opt_s], [opt_m], color=PALETTE["accent1"],
               s=150, marker="*", edgecolors="white",
               linewidth=1, zorder=6, label="Optimum")
    ax.set_xlabel("σ (MPa)"); ax.set_ylabel("E (GPa)")
    ax.set_title("(B)  Pareto Front", fontsize=10, pad=8)
    ax.legend(fontsize=7)

    # C — Optimal design
    ax = fig.add_subplot(gs[0, 2])
    dvars = ["Vf%", "0°%", "45°%", "90°%", "CNT×100", "IL×10", "P/10"]
    dvals = [opt_chrom[3], opt_chrom[4], opt_chrom[5], opt_chrom[6],
             opt_chrom[8]*100, opt_chrom[9]*10, opt_chrom[11]/10]
    bars = ax.bar(dvars, dvals, color=_CB[:7], alpha=0.88,
                  edgecolor="white", linewidth=1.2, width=0.55)
    for b, v in zip(bars, dvals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"{v:.1f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_title(f"(C)  Optimal Design\nσ={opt_s:.0f} MPa  E={opt_m:.1f} GPa",
                 fontsize=9, fontweight="bold", pad=8)
    ax.set_ylabel("Value (scaled)")
    ax.tick_params(axis="x", labelsize=7)

    # D — Modulus convergence
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(gens_, ga_results["gen_best_m"], color=PALETTE["accent1"],
            lw=2.2, label="Best E")
    ax.set_title("(D)  Modulus Convergence", fontsize=10, pad=8)
    ax.set_xlabel("Generation"); ax.set_ylabel("E (GPa)")
    ax.legend(fontsize=7)

    # E — Population fitness evolution
    ax = fig.add_subplot(gs[1, 1])
    sample_gens = [0, 19, 39, 59, 79, 99]
    afits = ga_results["all_fitnesses"]
    violin_data = [afits[g][:,0] for g in sample_gens if g < len(afits)]
    gen_labels = [f"G{g+1}" for g in sample_gens if g < len(afits)]
    if violin_data:
        vp = ax.violinplot(violin_data, showmeans=True,
                           showmedians=True, showextrema=True)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(PALETTE["primary"])
            body.set_alpha(0.25 + 0.1*i)
        ax.set_xticks(range(1, len(gen_labels)+1))
        ax.set_xticklabels(gen_labels, fontsize=7.5)
    ax.set_title("(E)  Population Evolution", fontsize=10, pad=8)
    ax.set_ylabel("σ (MPa)")

    # F — Pareto front evolution
    ax = fig.add_subplot(gs[1, 2])
    snapshots = [0, 24, 49, 74, 99]
    for si, g_idx in enumerate(snapshots):
        if g_idx < len(ga_results["pareto_history"]):
            pf = ga_results["pareto_history"][g_idx]
            if len(pf) > 0:
                ax.scatter(pf[:,0], pf[:,1],
                           color=plt.cm.viridis(si/len(snapshots)),
                           s=30, alpha=0.7,
                           label=f"G{g_idx+1}")
    ax.set_xlabel("σ (MPa)"); ax.set_ylabel("E (GPa)")
    ax.set_title("(F)  Pareto Evolution", fontsize=10, pad=8)
    ax.legend(fontsize=7)

    savefig("Fig8_GA_Optimisation", fig)
    print("  Fig 8 saved.")


def fig9_results_table(results, models):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.suptitle("Model Performance Summary",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.02, right=0.98)

    all_names = list(results.keys())
    tdata = []
    for name in all_names:
        m = results[name]
        typ = ("Stack" if name == "STACK"
               else "DNN" if name.startswith("DL")
               else "Tree" if name in ("RF","ET","GBM") else "SVM")
        arch = {
            "RF":"400 trees", "ET":"400 trees", "GBM":"500 est",
            "SVR":"RBF C=200",
            "DL1":"(128)", "DL2":"(256→128)", "DL3":"(256→128→64)",
            "DL4":"(512→256→128→64)", "DL5":"(256→128→64→32)",
            "STACK":"Ridge meta",
        }.get(name, "—")
        tdata.append([name, typ, arch,
                      f"{m['R2']:.4f}", f"{m['RMSE']:.1f}",
                      f"{m['MAE']:.1f}", f"{m['MAPE']:.2f}%",
                      f"{m['CV_mean']:.4f}±{m['CV_std']:.3f}",
                      f"{m.get('bias',0):.1f}"])

    cols = ["Model","Type","Arch","R²","RMSE","MAE","MAPE","CV R²","Bias"]
    tbl = ax.table(cellText=tdata, colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.05, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D5D8DC")
        if row == 0:
            cell.set_facecolor("#1B4F72")
            cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and tdata[row-1][0] == "STACK":
            cell.set_facecolor("#FEF9E7")
            cell.set_text_props(fontweight="bold")
        elif row > 0 and tdata[row-1][1] == "DNN":
            cell.set_facecolor("#F4ECF7")
        elif row > 0 and tdata[row-1][1] == "Tree":
            cell.set_facecolor("#EAF4F4")
        elif row % 2 == 0:
            cell.set_facecolor("#F8F9FA")

    savefig("Fig9_Results_Table", fig)
    print("  Fig 9 saved.")
