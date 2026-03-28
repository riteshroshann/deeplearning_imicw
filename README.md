<p align="center">
  <h1 align="center">🧬 AI-Driven Property Optimisation of CFRP Composites</h1>
  <p align="center">
    <em>A modular deep learning + physics-informed pipeline for predicting and optimising the mechanical properties of carbon fibre reinforced polymer composites.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white" alt="sklearn"/>
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT"/>
  </p>
</p>

---

## Overview

This repository implements a complete material informatics pipeline — from curated literature data to multi-objective genetic algorithm design optimisation — for predicting the tensile strength and elastic modulus of aerospace-grade CFRP laminates.

The system achieves **R² = 0.934** (GBM) and **R² = 0.870** (stacked ensemble) on held-out test data, and discovers optimal laminate configurations via NSGA-II that reach **σ ≈ 1784 MPa**.

### Key Contributions

- **Physics-informed feature engineering** — 12 derived variables grounded in Rule-of-Mixtures, Halpin-Tsai, and Classical Laminate Theory
- **Physics-Informed Neural Network (PINN)** — Custom loss function embedding ROM constraints directly into gradient descent
- **Multi-model ensemble** — Out-of-fold stacking of RF, ET, GBM, SVR, and 5 DNN architectures with bootstrap prediction intervals
- **NSGA-II design optimisation** — Pareto-optimal CFRP designs balancing strength and stiffness

---

## Architecture

```
main.py                    ← Pipeline orchestrator
│
├── config.py              ← Constants, hyperparameters, physics tables, styling
├── data_builder.py        ← Literature dataset (8 sources, 283 records)
├── preprocessing.py       ← Feature engineering (12 physics-based variables)
│
├── models.py              ← ResidualMLP, AttentionMLP, MultiHeadCFRPNet
├── pinn.py                ← Physics-Informed Neural Network (ROM-anchored loss)
├── training.py            ← Training loops, sklearn baselines, PINN integration
├── ensemble.py            ← OOF stacking + bootstrap prediction intervals
├── evaluation.py          ← Metrics, normality tests, ROM benchmarks
│
├── optimization.py        ← NSGA-II multi-objective GA
├── visualization.py       ← 9 publication-quality figures
│
└── outputs/               ← Generated figures (PNG) and tables (CSV)
```

---

## Results

<table>
<tr><th>Model</th><th>R²</th><th>RMSE (MPa)</th><th>Type</th></tr>
<tr><td><b>GBM</b></td><td><b>0.934</b></td><td>74.2</td><td>Gradient Boosting</td></tr>
<tr><td>STACK</td><td>0.870</td><td>—</td><td>OOF Ridge Meta-Learner</td></tr>
<tr><td>ET</td><td>0.855</td><td>—</td><td>Extra Trees</td></tr>
<tr><td>RF</td><td>0.728</td><td>155.2</td><td>Random Forest</td></tr>
<tr><td>SVR</td><td>0.709</td><td>—</td><td>Support Vector (RBF)</td></tr>
</table>

> Standard DNNs (DL1–DL5) exhibit negative R² on this small dataset (~280 rows), which motivates the PINN approach. The physics-informed loss provides a regularisation pathway by anchoring predictions to classical micromechanics.

### GA Optimisation

| Parameter | Optimal Value |
|---|---|
| Predicted Strength | **1784 MPa** |
| Predicted Modulus | **148.6 GPa** |
| Fibre | IM7 |
| Volume Fraction | 62% |
| Ply Schedule | 100° / 0°₄₅ / 0°₉₀ |

---

## Generated Figures

The pipeline produces 9 publication-ready figures:

| Figure | Description |
|---|---|
| `Fig1_EDA_Overview.png` | Dataset distribution, sources, test types, environmental effects |
| `Fig2_Correlation_Matrix.png` | Pearson & Spearman heatmaps (20 features) |
| `Fig3_Model_Performance.png` | R², RMSE, MAE, MAPE, CV, and actual-vs-predicted |
| `Fig4_Residual_Diagnostics.png` | Residuals, Q-Q plot, normality, fibre-type breakdown |
| `Fig5_Feature_Importance.png` | RF impurity, permutation, and combined rank |
| `Fig6_DNN_Learning_Curves.png` | Training / validation loss for DL1–DL5 |
| `Fig7_Physics_vs_ML.png` | ROM vs orientation-weighted ROM vs stacked ensemble |
| `Fig8_GA_Optimisation.png` | Convergence, Pareto front, population evolution |
| `Fig9_Results_Table.png` | Summary performance table |

---

## Physics-Informed Neural Network

The PINN module (`pinn.py`) addresses the fundamental challenge of training deep networks on small heterogeneous datasets by injecting domain knowledge into the loss function:

$$\mathcal{L}_{\text{total}} = \text{MSE}_{\text{data}}(y, \hat{y}) + \lambda(t) \cdot \mathcal{L}_{\text{physics}}(\hat{y}, \text{ROM})$$

where:

- **ROM deviation**: penalises predictions deviating from orientation-weighted Rule-of-Mixtures
- **Ceiling violation**: `ReLU(ŷ − ROM_max)²` enforces the theoretical upper bound
- **λ decay**: `λ(t) = λ₀ · exp(−γt)` — trusts physics early, data later

---

## Quickstart

### Prerequisites

```
pip install numpy pandas scikit-learn scipy matplotlib seaborn torch
```

### Run the pipeline

```bash
python main.py
```

This will:
1. Build the dataset from 8 literature sources
2. Engineer 12 physics-based features
3. Train RF, ET, GBM, SVR, DL1–DL5, PINN1, PINN2
4. Build OOF stacked ensemble with bootstrap CI
5. Run NSGA-II multi-objective optimisation (100 generations)
6. Generate 9 figures + 7 CSVs in `outputs/`

> **Note**: Full pipeline takes ~45 minutes on CPU. PyTorch is optional — the pipeline falls back to sklearn MLPRegressor if unavailable.

---

## Dataset

The dataset is synthetically reconstructed from 8 peer-reviewed aerospace composite studies, covering:

- **Fibre types**: T300, T700, T800, IM7, HS/HM/IM Carbon
- **Test types**: Longitudinal/transverse tension & compression, in-plane shear, flexural, interlaminar
- **Environments**: CTD, RTD, ETD, ETW (dry and wet conditioning)
- **283 records** with full provenance tracking (`source_id`, `doi_or_url`)

---

## Feature Engineering

12 physics-based features derived from Classical Laminate Theory:

| # | Feature | Equation |
|---|---------|----------|
| F1 | `E_L_ROM_GPa` | V_f · E_f + V_m · E_m |
| F2 | `E_T_HT_GPa` | Halpin-Tsai transverse modulus |
| F3 | `orientation_efficiency` | Σ pᵢ · cos⁴(θᵢ) |
| F4 | `sigma_L_ROM_MPa` | V_f · σ_f + V_m · σ_m |
| F5 | `thermal_knockdown` | T_test / T_g |
| F6 | `effective_Vf_pct` | V_f − moisture correction |
| F7 | `mfg_quality_index` | f(process, pressure) |
| F8 | `anisotropy_proxy` | E_L / E_T |
| F9 | `cnt_il_synergy` | CNT% × (1 + IL%/10) |
| F10 | `zero_ply_contribution` | p₀ × σ_L_ROM |
| F11 | `composite_density_gcc` | V_f · ρ_f + V_m · ρ_m |
| F12 | `sigma_L_specific` | σ_L_ROM / ρ_c |

---

## Project Context

This project was developed for the coursework intersection of **Deep Learning** and **Introduction to Material Informatics**. The goal is a research-grade demonstration of how modern ML techniques — ensemble learning, physics-informed neural networks, and evolutionary optimisation — can accelerate composite material design beyond classical micromechanics alone.

---

## License

MIT
