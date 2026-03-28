"""Training loop with LR scheduling, early stopping, and sklearn baselines."""

import time
import numpy as np
from config import SEED, DL_CONFIGS, SKLEARN_MODELS_CONFIG, N_CV_FOLDS, section, subsection
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from models import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from models import ResidualMLP


def _metrics_dict(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    resid = y_true - y_pred
    return dict(
        R2=r2_score(y_true, y_pred),
        RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
        MAE=mean_absolute_error(y_true, y_pred),
        MAPE=mean_absolute_percentage_error(y_true, y_pred)*100,
        bias=resid.mean(), std_resid=resid.std())


def train_pytorch_model(model, X_train, y_train, X_val, y_val, config, verbose=True):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_vl = torch.FloatTensor(X_val).to(device)
    y_vl = torch.FloatTensor(y_val).to(device)

    optimizer = AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.HuberLoss(delta=50.0)

    history = {"train_loss": [], "val_loss": [], "val_r2": [], "lr": []}
    best_val_loss, best_state, wait = float("inf"), None, 0
    patience = config.get("patience", 50)

    for epoch in range(1, config.get("epochs", 600)+1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            y_pred_vl = model(X_vl)
            val_loss = criterion(y_pred_vl, y_vl).item()
            val_r2 = r2_score(y_vl.cpu().numpy(), y_pred_vl.cpu().numpy())

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"      Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    history["n_epochs"] = epoch
    return model, history


def predict_pytorch(model, X):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(device)).cpu().numpy()


def build_sklearn_models():
    cfg = SKLEARN_MODELS_CONFIG
    return {
        "RF":  RandomForestRegressor(**cfg["RF"], random_state=SEED, n_jobs=-1),
        "ET":  ExtraTreesRegressor(**cfg["ET"], random_state=SEED, n_jobs=-1),
        "GBM": GradientBoostingRegressor(**cfg["GBM"], random_state=SEED),
        "SVR": SVR(**cfg["SVR"]),
    }


def train_all_models(X_train, y_train, X_test, y_test,
                     df_ml=None, train_indices=None, test_indices=None):
    section("PHASE 4 — MODEL TRAINING")
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    results, preds, models = {}, {}, {}

    print(f"\n  {'Model':<8} {'R²':>7} {'RMSE':>8} {'MAE':>8} {'MAPE%':>7} {'Time':>6}")
    print(f"  {'─'*50}")

    sklearn_models = build_sklearn_models()
    for name, model in sklearn_models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - t0
        m = _metrics_dict(y_test, y_pred)
        cv = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        results[name] = {**m, "CV_mean": cv.mean(), "CV_std": cv.std(),
                         "n_params": None, "time_s": elapsed,
                         "loss": None, "val_loss": None, "n_iter": None}
        preds[name] = y_pred
        models[name] = model
        print(f"  {name:<8} {m['R2']:>7.4f} {m['RMSE']:>8.1f} "
              f"{m['MAE']:>8.1f} {m['MAPE']:>7.2f} {elapsed:>5.1f}s")

    if TORCH_AVAILABLE:
        subsection("PyTorch Deep Learning Models")
        n_val = max(int(len(X_train)*0.15), 10)
        val_idx = np.random.RandomState(SEED).permutation(len(X_train))[:n_val]
        tr_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)
        X_tr_dl, y_tr_dl = X_train[tr_idx], y_train[tr_idx]
        X_vl_dl, y_vl_dl = X_train[val_idx], y_train[val_idx]
        input_dim = X_train.shape[1]

        for name, cfg in DL_CONFIGS.items():
            t0 = time.time()
            model = ResidualMLP(input_dim, cfg["layers"], cfg.get("dropout", 0.15))
            model, history = train_pytorch_model(
                model, X_tr_dl, y_tr_dl, X_vl_dl, y_vl_dl, cfg, verbose=False)
            y_pred = predict_pytorch(model, X_test)
            elapsed = time.time() - t0
            m = _metrics_dict(y_test, y_pred)
            n_params = sum(p.numel() for p in model.parameters())
            results[name] = {**m, "CV_mean": history["val_r2"][-1] if history["val_r2"] else 0.0,
                             "CV_std": 0.0, "n_params": n_params, "time_s": elapsed,
                             "loss": history["train_loss"], "val_loss": history["val_loss"],
                             "n_iter": history["n_epochs"]}
            preds[name] = y_pred
            models[name] = model
            print(f"  {name:<8} {m['R2']:>7.4f} {m['RMSE']:>8.1f} "
                  f"{m['MAE']:>8.1f} {m['MAPE']:>7.2f} {elapsed:>5.1f}s  "
                  f"({n_params:,} params, {history['n_epochs']} ep)")

        # Physics-Informed Neural Networks
        if df_ml is not None and train_indices is not None:
            subsection("PINN — Physics-Informed Models")
            from pinn import PINNResidualMLP, train_pinn, predict_pinn

            train_phys = df_ml.iloc[train_indices].reset_index(drop=True)
            test_phys = df_ml.iloc[test_indices].reset_index(drop=True)
            tr_phys = train_phys.iloc[tr_idx].reset_index(drop=True)
            vl_phys = train_phys.iloc[val_idx].reset_index(drop=True)

            pinn_configs = {
                "PINN1": {"hidden": (128, 64),   "dropout": 0.35, "lr": 5e-4,
                          "lambda_0": 0.05, "decay": 0.005, "epochs": 800, "patience": 60},
                "PINN2": {"hidden": (256, 128, 64), "dropout": 0.40, "lr": 3e-4,
                          "lambda_0": 0.1, "decay": 0.003, "epochs": 1000, "patience": 80},
            }
            for name, cfg in pinn_configs.items():
                try:
                    t0 = time.time()
                    model = PINNResidualMLP(input_dim, cfg["hidden"], cfg["dropout"])
                    model, history = train_pinn(
                        model, X_tr_dl, y_tr_dl, X_vl_dl, y_vl_dl,
                        tr_phys, vl_phys,
                        epochs=cfg["epochs"], lr=cfg["lr"],
                        weight_decay=1e-3,
                        lambda_0=cfg["lambda_0"], decay_rate=cfg["decay"],
                        patience=cfg["patience"])
                    y_pred = predict_pinn(model, X_test)
                    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                        print(f"  {name:<8} SKIPPED — diverged (NaN/Inf)")
                        continue
                    elapsed = time.time() - t0
                    m = _metrics_dict(y_test, y_pred)
                    n_params = sum(p.numel() for p in model.parameters())
                    results[name] = {
                        **m, "CV_mean": history["val_r2"][-1] if history["val_r2"] else 0.0,
                        "CV_std": 0.0, "n_params": n_params, "time_s": elapsed,
                        "loss": history["train_loss"], "val_loss": history["val_loss"],
                        "n_iter": history["n_epochs"],
                        "physics_loss": history.get("physics_loss"),
                        "lambda_history": history.get("lambda")}
                    preds[name] = y_pred
                    models[name] = model
                    final_lam = history["lambda"][-1] if history["lambda"] else 0
                    print(f"  {name:<8} {m['R2']:>7.4f} {m['RMSE']:>8.1f} "
                          f"{m['MAE']:>8.1f} {m['MAPE']:>7.2f} {elapsed:>5.1f}s  "
                          f"({n_params:,} params, {history['n_epochs']} ep, λ={final_lam:.3f})")
                except Exception as e:
                    print(f"  {name:<8} FAILED — {str(e)[:60]}")
    else:
        subsection("sklearn MLP (PyTorch unavailable)")
        dl_sklearn = {
            "DL1": MLPRegressor(hidden_layer_sizes=(128,), max_iter=3000, early_stopping=True, random_state=SEED),
            "DL2": MLPRegressor(hidden_layer_sizes=(256,128), max_iter=3000, early_stopping=True, random_state=SEED),
            "DL3": MLPRegressor(hidden_layer_sizes=(256,128,64), max_iter=3000, early_stopping=True, random_state=SEED),
            "DL4": MLPRegressor(hidden_layer_sizes=(512,256,128,64), max_iter=3000, early_stopping=True, learning_rate_init=5e-4, random_state=SEED),
            "DL5": MLPRegressor(hidden_layer_sizes=(256,128,64,32), activation="tanh", max_iter=3000, early_stopping=True, random_state=SEED),
        }
        for name, model in dl_sklearn.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elapsed = time.time() - t0
            m = _metrics_dict(y_test, y_pred)
            cv = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
            n_p = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
            results[name] = {**m, "CV_mean": cv.mean(), "CV_std": cv.std(),
                             "n_params": n_p, "time_s": elapsed,
                             "loss": getattr(model, "loss_curve_", None),
                             "val_loss": None, "n_iter": getattr(model, "n_iter_", None)}
            preds[name] = y_pred
            models[name] = model
            print(f"  {name:<8} {m['R2']:>7.4f} {m['RMSE']:>8.1f} "
                  f"{m['MAE']:>8.1f} {m['MAPE']:>7.2f} {elapsed:>5.1f}s")

    return results, preds, models
