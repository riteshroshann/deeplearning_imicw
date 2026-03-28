"""Physics-Informed Neural Network for CFRP strength prediction.

Embeds Rule-of-Mixtures constraints into the loss function to 
regularize DNN training on small composite datasets (~280 rows).
"""

import numpy as np

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    pass

if TORCH_AVAILABLE:

    class CFRPPhysicsDataset(Dataset):
        """Yields (X_scaled, y_target, physics_params) per sample."""
        def __init__(self, X_scaled, y, Vf, sigma_f, P0, sigma_m):
            self.X = torch.FloatTensor(X_scaled)
            self.y = torch.FloatTensor(y)
            self.physics = torch.FloatTensor(np.column_stack([
                Vf, sigma_f, P0, np.full(len(Vf), sigma_m)
            ]))

        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i], self.physics[i]


    class PhysicsInformedLoss(nn.Module):
        """L = MSE_data + λ(t) · (ROM_deviation + ceiling + non-negativity)
        
        λ decays: λ(t) = λ_0 · exp(-decay · epoch)
        """
        def __init__(self, lambda_0=0.5, decay_rate=0.005):
            super().__init__()
            self.lambda_0 = lambda_0
            self.decay_rate = decay_rate

        def get_lambda(self, epoch):
            return self.lambda_0 * np.exp(-self.decay_rate * epoch)

        def forward(self, y_pred, y_true, physics_params, epoch=0):
            Vf, sigma_f = physics_params[:, 0], physics_params[:, 1]
            P0, sigma_m = physics_params[:, 2], physics_params[:, 3]

            mse_data = F.mse_loss(y_pred, y_true)
            rom_max = Vf * sigma_f + (1 - Vf) * sigma_m
            rom_orient = P0 * Vf * sigma_f + (1 - P0) * (1 - Vf) * sigma_m

            # normalized relative deviation (scale-invariant)
            mean_target = y_true.abs().mean().clamp(min=1.0)
            physics_dev = ((y_pred - rom_orient) / mean_target).pow(2).mean()
            ceiling = (F.relu(y_pred - rom_max) / mean_target).pow(2).mean()
            negativity = (F.relu(-y_pred) / mean_target).pow(2).mean()

            lam = self.get_lambda(epoch)
            loss_physics = physics_dev + 2.0 * ceiling + negativity

            return mse_data + lam * loss_physics, {
                "mse": mse_data.item(), "physics": loss_physics.item(),
                "lambda": lam, "ceiling": ceiling.item()
            }


    class PINNResidualMLP(nn.Module):
        """Compact MLP for small datasets. Uses LayerNorm (not BatchNorm) to
        avoid crashes with batch_size=1 on small CFRP datasets."""
        def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.35):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.LayerNorm(h))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
                prev = h
            self.trunk = nn.Sequential(*layers)
            
            self.head = nn.Sequential(
                nn.Linear(prev, 32), nn.SiLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1))
            
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.head(self.trunk(self.input_norm(x))).squeeze(-1)


    def train_pinn(model, X_train, y_train, X_val, y_val,
                   train_physics, val_physics,
                   epochs=800, lr=5e-4, weight_decay=1e-3,
                   lambda_0=0.5, decay_rate=0.005, patience=60):
        from config import MATRIX_STRENGTH_MPa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        sigma_m = MATRIX_STRENGTH_MPa

        ds_train = CFRPPhysicsDataset(
            X_train, y_train,
            Vf=train_physics["fiber_volume_pct"].values / 100.0,
            sigma_f=train_physics["fiber_tensile_strength_MPa"].values,
            P0=train_physics["pct_0_plies"].values / 100.0,
            sigma_m=sigma_m)
        ds_val = CFRPPhysicsDataset(
            X_val, y_val,
            Vf=val_physics["fiber_volume_pct"].values / 100.0,
            sigma_f=val_physics["fiber_tensile_strength_MPa"].values,
            P0=val_physics["pct_0_plies"].values / 100.0,
            sigma_m=sigma_m)

        loader = DataLoader(ds_train, batch_size=min(64, len(X_train)),
                            shuffle=True, drop_last=False)
        criterion = PhysicsInformedLoss(lambda_0=lambda_0, decay_rate=decay_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

        history = {"train_loss": [], "val_loss": [], "val_r2": [],
                   "mse_data": [], "physics_loss": [], "lambda": [], "lr": []}
        best_val, best_state, wait = float("inf"), None, 0

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss, epoch_mse, epoch_phys = 0.0, 0.0, 0.0
            n_batches = 0
            for X_b, y_b, p_b in loader:
                X_b, y_b, p_b = X_b.to(device), y_b.to(device), p_b.to(device)
                optimizer.zero_grad()
                pred = model(X_b)
                loss, info = criterion(pred, y_b, p_b, epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Abort if training diverged
                if torch.isnan(loss) or torch.isinf(loss):
                    history["n_epochs"] = epoch
                    return model, history
                
                optimizer.step()
                epoch_loss += loss.item()
                epoch_mse += info["mse"]
                epoch_phys += info["physics"]
                n_batches += 1
            scheduler.step()

            history["train_loss"].append(epoch_loss / max(n_batches, 1))
            history["mse_data"].append(epoch_mse / max(n_batches, 1))
            history["physics_loss"].append(epoch_phys / max(n_batches, 1))
            history["lambda"].append(criterion.get_lambda(epoch))
            history["lr"].append(optimizer.param_groups[0]["lr"])

            model.eval()
            with torch.no_grad():
                X_v = ds_val.X.to(device)
                y_v = ds_val.y.to(device)
                p_v = ds_val.physics.to(device)
                v_pred = model(X_v)
                v_loss, _ = criterion(v_pred, y_v, p_v, epoch)
                from sklearn.metrics import r2_score
                vr2 = r2_score(y_v.cpu().numpy(), v_pred.cpu().numpy())
            history["val_loss"].append(v_loss.item())
            history["val_r2"].append(vr2)

            if v_loss.item() < best_val:
                best_val = v_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        history["n_epochs"] = epoch
        return model, history


    def predict_pinn(model, X):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(X).to(device)).cpu().numpy()
