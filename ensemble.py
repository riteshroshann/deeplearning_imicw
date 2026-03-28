"""
ensemble.py — Stacked Ensemble with OOF Meta-Learning
======================================================
Implements a two-level stacking architecture:
  Level-0: diverse base learners (tree ensembles, kernel, neural nets)
  Level-1: Ridge regression on out-of-fold (OOF) predictions

OOF stacking prevents target leakage that would occur if base
learners predicted on the same data used for meta-learner training.

Bootstrap prediction intervals provide calibrated uncertainty
estimates over the full ensemble.
"""

import time
import numpy as np
from config import (
    SEED, RNG, N_CV_FOLDS, N_BOOTSTRAP, subsection
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

from training import _metrics_dict, predict_pytorch
from models import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


def build_stacked_ensemble(base_models, results, preds,
                           X_train, y_train, X_test, y_test):
    """
    Train a stacked ensemble using OOF predictions from base learners.

    Procedure
    ---------
    1. For each base model, generate OOF predictions via K-fold CV.
    2. Stack OOF predictions into a meta-feature matrix.
    3. Train a Ridge meta-learner on the meta-feature matrix.
    4. Final prediction = meta_learner(base_model_predictions).

    Parameters
    ----------
    base_models : dict[str, object] — trained base learners
    results     : dict — metric dicts from training.train_all_models
    preds       : dict — test predictions from each base model
    X_train, y_train, X_test, y_test : np.ndarray

    Returns
    -------
    results     : updated with 'STACK' entry
    preds       : updated with 'STACK' predictions
    meta        : fitted Ridge meta-learner
    OOF_PREDS   : out-of-fold prediction matrix (train)
    TEST_PREDS  : test prediction matrix
    """
    subsection("Stacked Ensemble (OOF meta-learner)")

    model_names = list(base_models.keys())
    n_models    = len(model_names)
    kf          = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)

    OOF_PREDS  = np.zeros((len(X_train), n_models))
    TEST_PREDS = np.zeros((len(X_test),  n_models))

    print("  Training OOF base predictions (5-fold)...")

    for col_idx, name in enumerate(model_names):
        model = base_models[name]
        oof   = np.zeros(len(X_train))

        # OOF predictions
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                # PyTorch model — clone by creating new instance
                import copy
                clone = copy.deepcopy(model)
                # Re-fit on fold
                from training import train_pytorch_model
                clone, _ = train_pytorch_model(
                    clone, X_train[tr_idx], y_train[tr_idx],
                    X_train[val_idx], y_train[val_idx],
                    {"lr": 1e-3, "epochs": 300, "patience": 30},
                    verbose=False)
                oof[val_idx] = predict_pytorch(clone, X_train[val_idx])
            else:
                # sklearn model
                clone = type(model)(**model.get_params())
                clone.fit(X_train[tr_idx], y_train[tr_idx])
                oof[val_idx] = clone.predict(X_train[val_idx])

        OOF_PREDS[:, col_idx]  = oof
        TEST_PREDS[:, col_idx] = preds[name]

    # ── Meta-learner ─────────────────────────────────────────────────────
    meta = Ridge(alpha=1.0, fit_intercept=True)
    meta.fit(OOF_PREDS, y_train)
    y_stack = meta.predict(TEST_PREDS)

    m_stack = _metrics_dict(y_test, y_stack)
    cv_stack = cross_val_score(
        Ridge(alpha=1.0), OOF_PREDS, y_train,
        cv=kf, scoring="r2")

    results["STACK"] = {
        **m_stack, "CV_mean": cv_stack.mean(), "CV_std": cv_stack.std(),
        "n_params": None, "time_s": 0.0,
        "loss": None, "val_loss": None, "n_iter": None,
    }
    preds["STACK"] = y_stack

    # Print meta-learner weights
    weights = dict(zip(model_names, meta.coef_.round(3)))
    print(f"  {'STACK':<8} {m_stack['R2']:>7.4f} {m_stack['RMSE']:>8.1f} "
          f"{m_stack['MAE']:>8.1f} {m_stack['MAPE']:>7.2f}")
    print(f"  Meta-learner weights: {weights}")

    best_single = max(
        [k for k in results if k != "STACK"],
        key=lambda x: results[x]["R2"])
    print(f"\n  Best single model : {best_single}  "
          f"R²={results[best_single]['R2']:.4f}")
    print(f"  Stacked ensemble  : R²={results['STACK']['R2']:.4f}")

    return results, preds, meta, OOF_PREDS, TEST_PREDS


def bootstrap_prediction_intervals(base_models, X_train, y_train,
                                   X_test, y_test, OOF_PREDS,
                                   y_stack, n_boot=None):
    """
    Bootstrap prediction intervals for the stacked ensemble.

    For each bootstrap iteration:
      1. Resample training data with replacement
      2. Retrain each base learner on the bootstrap sample
      3. Fit meta-learner on resampled OOF predictions
      4. Predict on test set

    The distribution of predictions across bootstrap iterations
    provides calibrated prediction intervals.

    Returns
    -------
    pi_lo, pi_hi : np.ndarray — 95% prediction interval bounds
    pi_cov       : float — empirical coverage
    pi_wid       : float — mean interval width
    r2_ci        : tuple — (lower, point, upper) 95% CI for R²
    """
    subsection("Bootstrap prediction intervals")

    if n_boot is None:
        n_boot = N_BOOTSTRAP

    model_names = list(base_models.keys())
    boot_preds  = np.zeros((n_boot, len(X_test)))

    for b in range(n_boot):
        idx = RNG.integers(0, len(X_train), len(X_train))

        b_test = np.zeros((len(X_test), len(model_names)))
        for c, name in enumerate(model_names):
            model = base_models[name]
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                import copy
                clone = copy.deepcopy(model)
                from training import train_pytorch_model
                n_val = max(10, int(len(idx) * 0.15))
                val_i = idx[:n_val]
                tr_i  = idx[n_val:]
                clone, _ = train_pytorch_model(
                    clone, X_train[tr_i], y_train[tr_i],
                    X_train[val_i], y_train[val_i],
                    {"lr": 1e-3, "epochs": 150, "patience": 20},
                    verbose=False)
                b_test[:, c] = predict_pytorch(clone, X_test)
            else:
                cm = type(model)(**model.get_params())
                cm.fit(X_train[idx], y_train[idx])
                b_test[:, c] = cm.predict(X_test)

        meta_b = Ridge(alpha=1.0)
        meta_b.fit(OOF_PREDS, y_train)  # Use pre-computed OOF
        boot_preds[b] = meta_b.predict(b_test)

    pi_lo = np.percentile(boot_preds, 2.5, axis=0)
    pi_hi = np.percentile(boot_preds, 97.5, axis=0)
    pi_cov = np.mean((y_test >= pi_lo) & (y_test <= pi_hi)) * 100
    pi_wid = np.mean(pi_hi - pi_lo)

    # R² confidence interval
    boot_r2 = [r2_score(y_test, boot_preds[b]) for b in range(n_boot)]
    r2_lo = np.percentile(boot_r2, 2.5)
    r2_pt = r2_score(y_test, y_stack)
    r2_hi = np.percentile(boot_r2, 97.5)

    print(f"  Bootstrap 95% PI coverage: {pi_cov:.1f}%  (target ≥95%)")
    print(f"  Mean PI width: {pi_wid:.1f} MPa")
    print(f"  Bootstrap 95% CI for R²: [{r2_lo:.4f}, {r2_pt:.4f}, {r2_hi:.4f}]")

    return pi_lo, pi_hi, pi_cov, pi_wid, (r2_lo, r2_pt, r2_hi)
