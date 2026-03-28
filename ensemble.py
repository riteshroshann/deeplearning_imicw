"""Stacked ensemble with OOF meta-learning and bootstrap prediction intervals."""

import numpy as np
from config import SEED, RNG, N_CV_FOLDS, N_BOOTSTRAP, subsection
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from training import _metrics_dict, predict_pytorch
from models import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


def build_stacked_ensemble(base_models, results, preds,
                           X_train, y_train, X_test, y_test):
    subsection("Stacked Ensemble (OOF meta-learner)")
    model_names = list(base_models.keys())
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)

    OOF_PREDS  = np.zeros((len(X_train), len(model_names)))
    TEST_PREDS = np.zeros((len(X_test),  len(model_names)))

    print("  Training OOF base predictions (5-fold)...")
    for col_idx, name in enumerate(model_names):
        model = base_models[name]
        oof = np.zeros(len(X_train))
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                import copy
                clone = copy.deepcopy(model)
                from training import train_pytorch_model
                clone, _ = train_pytorch_model(
                    clone, X_train[tr_idx], y_train[tr_idx],
                    X_train[val_idx], y_train[val_idx],
                    {"lr": 1e-3, "epochs": 300, "patience": 30}, verbose=False)
                oof[val_idx] = predict_pytorch(clone, X_train[val_idx])
            else:
                clone = type(model)(**model.get_params())
                clone.fit(X_train[tr_idx], y_train[tr_idx])
                oof[val_idx] = clone.predict(X_train[val_idx])
        OOF_PREDS[:, col_idx] = oof
        TEST_PREDS[:, col_idx] = preds[name]

    meta = Ridge(alpha=1.0, fit_intercept=True)
    meta.fit(OOF_PREDS, y_train)
    y_stack = meta.predict(TEST_PREDS)

    m_stack = _metrics_dict(y_test, y_stack)
    cv_stack = cross_val_score(Ridge(alpha=1.0), OOF_PREDS, y_train, cv=kf, scoring="r2")

    results["STACK"] = {**m_stack, "CV_mean": cv_stack.mean(), "CV_std": cv_stack.std(),
                        "n_params": None, "time_s": 0.0,
                        "loss": None, "val_loss": None, "n_iter": None}
    preds["STACK"] = y_stack

    weights = dict(zip(model_names, meta.coef_.round(3)))
    print(f"  {'STACK':<8} {m_stack['R2']:>7.4f} {m_stack['RMSE']:>8.1f} "
          f"{m_stack['MAE']:>8.1f} {m_stack['MAPE']:>7.2f}")
    print(f"  Meta-learner weights: {weights}")

    best_single = max([k for k in results if k != "STACK"], key=lambda x: results[x]["R2"])
    print(f"\n  Best single: {best_single} R²={results[best_single]['R2']:.4f}")
    print(f"  Stack:       R²={results['STACK']['R2']:.4f}")

    return results, preds, meta, OOF_PREDS, TEST_PREDS


def bootstrap_prediction_intervals(base_models, X_train, y_train,
                                   X_test, y_test, OOF_PREDS,
                                   y_stack, n_boot=None):
    subsection("Bootstrap prediction intervals")
    if n_boot is None:
        n_boot = N_BOOTSTRAP

    model_names = list(base_models.keys())
    boot_preds = np.zeros((n_boot, len(X_test)))

    for b in range(n_boot):
        idx = RNG.integers(0, len(X_train), len(X_train))
        b_test = np.zeros((len(X_test), len(model_names)))
        for c, name in enumerate(model_names):
            model = base_models[name]
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                import copy
                clone = copy.deepcopy(model)
                from training import train_pytorch_model
                n_val = max(10, int(len(idx)*0.15))
                clone, _ = train_pytorch_model(
                    clone, X_train[idx[n_val:]], y_train[idx[n_val:]],
                    X_train[idx[:n_val]], y_train[idx[:n_val]],
                    {"lr": 1e-3, "epochs": 150, "patience": 20}, verbose=False)
                b_test[:, c] = predict_pytorch(clone, X_test)
            else:
                cm = type(model)(**model.get_params())
                cm.fit(X_train[idx], y_train[idx])
                b_test[:, c] = cm.predict(X_test)
        meta_b = Ridge(alpha=1.0)
        meta_b.fit(OOF_PREDS, y_train)
        boot_preds[b] = meta_b.predict(b_test)

    pi_lo = np.percentile(boot_preds, 2.5, axis=0)
    pi_hi = np.percentile(boot_preds, 97.5, axis=0)
    pi_cov = np.mean((y_test >= pi_lo) & (y_test <= pi_hi)) * 100
    pi_wid = np.mean(pi_hi - pi_lo)

    boot_r2 = [r2_score(y_test, boot_preds[b]) for b in range(n_boot)]
    r2_ci = (np.percentile(boot_r2, 2.5), r2_score(y_test, y_stack), np.percentile(boot_r2, 97.5))

    print(f"  95% PI coverage: {pi_cov:.1f}%  width: {pi_wid:.1f} MPa")
    print(f"  R² CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}, {r2_ci[2]:.4f}]")
    return pi_lo, pi_hi, pi_cov, pi_wid, r2_ci
