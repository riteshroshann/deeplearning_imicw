"""
optimization.py — NSGA-II Multi-Objective Genetic Algorithm
=============================================================
Optimises CFRP design for dual objectives:
  1. Maximise tensile strength (σ_t)
  2. Maximise tensile modulus  (E_t)

Key improvements over the baseline implementation:
  - NSGA-II non-dominated sorting with crowding distance
  - Physics-aware chromosome decoding (ply normalisation,
    fibre property consistency, derived feature recalculation)
  - Tournament selection with crowding-distance tiebreaker
  - Constraint-aware mutation respecting physical bounds

Reference
---------
  Deb K. et al. (2002). A fast and elitist multiobjective genetic
  algorithm: NSGA-II. IEEE Trans. Evol. Comput. 6(2):182-197.
"""

import numpy as np
from config import (
    SEED, ALL_FEATURES, GA_BOUNDS, GA_POP_SIZE, GA_N_GEN,
    GA_MUTATE_P, GA_CROSSOVER_P, GA_ELITE_K,
    RANK_TO_MOD, RANK_TO_STR, RANK_TO_DEN, RANK_TO_FIB,
    MATRIX_MODULUS_GPa, MATRIX_STRENGTH_MPa, MATRIX_DENSITY_gcc,
    MPa_TO_ksi, GPa_TO_Msi, section, subsection
)


# ── Design Space ─────────────────────────────────────────────────────────────
MINS = np.array([v[0] for v in GA_BOUNDS.values()])
MAXS = np.array([v[1] for v in GA_BOUNDS.values()])


def decode_chromosome(g):
    """
    Decode a raw GA chromosome into physically valid design variables.

    Ensures:
      1. All values within physics bounds
      2. Ply fractions sum to 100%
      3. Fibre rank ↔ fibre properties consistency
      4. Derived features (ROM, Halpin-Tsai, etc.) recalculated
    """
    g = g.copy()
    E_m = MATRIX_MODULUS_GPa
    sigma_m = MATRIX_STRENGTH_MPa

    # Clip to bounds and round integers
    for i, (feat, (lo, hi, typ)) in enumerate(GA_BOUNDS.items()):
        g[i] = np.clip(g[i], lo, hi)
        if typ == "int":
            g[i] = round(g[i])

    # Ply fraction normalisation (must sum to 100%)
    p0, p45, p90 = g[4], g[5], g[6]
    total = p0 + p45 + p90
    if total > 0:
        g[4] = round(p0 / total * 100, 1)
        g[5] = round(p45 / total * 100, 1)
        g[6] = round(100 - g[4] - g[5], 1)

    # Force fibre property consistency
    rank  = int(g[0])
    g[1]  = RANK_TO_MOD.get(rank, 230)
    g[2]  = RANK_TO_STR.get(rank, 3500)
    rho_f = RANK_TO_DEN.get(rank, 1.78)

    Vf = g[3] / 100.0
    Vm = 1.0 - Vf

    # Recalculate all derived features
    # NOTE: indices shifted by +1 because test_type_code is at index 17
    g[18] = Vf * g[1] + Vm * E_m                          # E_L_ROM
    xi    = 2.0
    eta   = (g[1] / E_m - 1) / (g[1] / E_m + xi)
    g[19] = E_m * (1 + xi * eta * Vf) / (1 - eta * Vf)    # E_T_HT
    p0n   = g[4] / 100
    p45n  = g[5] / 100
    p90n  = g[6] / 100
    g[20] = (p0n * np.cos(np.radians(0))**4 +
             p45n * np.cos(np.radians(45))**4 +
             p90n * np.cos(np.radians(90))**4)              # orientation_eff
    g[21] = Vf * g[2] + Vm * sigma_m                       # sigma_L_ROM
    g[22] = 23.0 / max(g[12], 1.0)                          # thermal_knockdown
    g[23] = g[3]                                             # effective_Vf (dry)
    g[24] = (6.0 - g[10]) * np.sqrt(max(g[11], 0.1) / 100) # mfg_quality
    g[25] = g[18] / max(g[19], 0.1)                          # anisotropy
    g[26] = g[8] * (1 + g[9] / 10.0)                        # cnt_il_synergy
    g[27] = p0n * g[21]                                      # zero_ply_contrib
    g[28] = Vf * rho_f + Vm * MATRIX_DENSITY_gcc             # density
    g[29] = g[21] / max(g[28], 0.1)                          # specific strength
    return g


def evaluate_design(chromosome, strength_model, modulus_model,
                    imputer, scaler):
    """
    Evaluate a design point using trained surrogate models.

    Returns (predicted_strength, predicted_modulus).
    """
    g  = decode_chromosome(chromosome)
    Xg = scaler.transform(imputer.transform([g]))
    s  = float(strength_model.predict(Xg)[0])
    m  = float(modulus_model.predict(Xg)[0])
    return s, m


# ── NSGA-II Operators ────────────────────────────────────────────────────────

def non_dominated_sort(fitnesses):
    """
    Fast non-dominated sorting (NSGA-II Algorithm 1).
    Maximises both objectives.

    Returns list of fronts (each front is a list of indices).
    """
    n = len(fitnesses)
    dominated_count = np.zeros(n, dtype=int)
    dominates_set   = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(fitnesses[i], fitnesses[j]):
                dominates_set[i].append(j)
                dominated_count[j] += 1
            elif _dominates(fitnesses[j], fitnesses[i]):
                dominates_set[j].append(i)
                dominated_count[i] += 1

    # Extract fronts
    fronts = []
    current = np.where(dominated_count == 0)[0].tolist()
    while current:
        fronts.append(current)
        next_front = []
        for i in current:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current = next_front

    return fronts


def _dominates(a, b):
    """True if a Pareto-dominates b (maximise both)."""
    return (a[0] >= b[0] and a[1] >= b[1] and
            (a[0] > b[0] or a[1] > b[1]))


def crowding_distance(fitnesses, front):
    """
    Compute crowding distance for individuals in a Pareto front.
    Infinite distance for boundary solutions to preserve extremes.
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    cd = np.zeros(n)
    for obj_idx in range(2):  # 2 objectives
        vals = fitnesses[front, obj_idx]
        order = np.argsort(vals)
        cd[order[0]]  = np.inf
        cd[order[-1]] = np.inf
        val_range = vals[order[-1]] - vals[order[0]]
        if val_range < 1e-10:
            continue
        for k in range(1, n - 1):
            cd[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / val_range

    return cd


def run_ga(strength_model, modulus_model, imputer, scaler):
    """
    Run multi-objective GA optimisation.

    Returns
    -------
    results : dict with optimal design, Pareto front, convergence history
    """
    section("PHASE 6 — MULTI-OBJECTIVE GA OPTIMISATION")

    rng = np.random.default_rng(SEED)

    # Initialise population
    pop = rng.uniform(MINS, MAXS, (GA_POP_SIZE, len(ALL_FEATURES)))

    # Storage
    gen_best_s, gen_best_m, gen_mean_s = [], [], []
    all_fitnesses  = []
    pareto_history = []
    best_overall_s     = -np.inf
    best_overall_chrom = None

    print(f"\n  GA: Pop={GA_POP_SIZE}  Gens={GA_N_GEN}  "
          f"Mutation={GA_MUTATE_P}")
    print(f"  Objectives: max(σ_tensile)  AND  max(E_tensile)\n")
    print(f"  {'Gen':>5}  {'Best σ':>13}  {'Best E':>13}  {'|F₁|':>5}")
    print(f"  {'─' * 42}")

    for gen in range(GA_N_GEN):
        # Evaluate all individuals
        fit_pairs = np.array([
            evaluate_design(c, strength_model, modulus_model, imputer, scaler)
            for c in pop
        ])
        strengths = fit_pairs[:, 0]
        moduli    = fit_pairs[:, 1]
        all_fitnesses.append(fit_pairs.copy())

        # Non-dominated sorting
        fronts = non_dominated_sort(fit_pairs)
        pareto_idx = fronts[0]
        pareto_history.append(fit_pairs[pareto_idx])

        gen_best_s.append(strengths.max())
        gen_best_m.append(moduli.max())
        gen_mean_s.append(strengths.mean())

        if strengths.max() > best_overall_s:
            best_overall_s     = strengths.max()
            best_overall_chrom = pop[np.argmax(strengths)].copy()

        if gen % 10 == 0 or gen == GA_N_GEN - 1:
            print(f"  {gen+1:>5}  {strengths.max():>13.1f}  "
                  f"{moduli.max():>13.2f}  {len(pareto_idx):>5}")

        # ── Selection: NSGA-II with crowding distance ────────────────────
        # Assign each individual a rank (front index) and crowding distance
        ranks = np.zeros(len(pop), dtype=int)
        cds   = np.zeros(len(pop))
        for rank_val, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank_val
            cd = crowding_distance(fit_pairs, front)
            for i, idx in enumerate(front):
                cds[idx] = cd[i]

        # Elitism: keep top individuals from first front
        elite_idx = fronts[0][:GA_ELITE_K] if len(fronts[0]) >= GA_ELITE_K \
                    else fronts[0]
        elite = pop[elite_idx].copy()

        # Binary tournament selection (prefer lower rank, then higher CD)
        selected = []
        for _ in range(GA_POP_SIZE - len(elite)):
            i, j = rng.integers(0, len(pop), 2)
            if ranks[i] < ranks[j]:
                selected.append(pop[i])
            elif ranks[j] < ranks[i]:
                selected.append(pop[j])
            else:
                selected.append(pop[i] if cds[i] > cds[j] else pop[j])
        selected = np.array(selected)

        # Crossover (SBX-like single-point) + Gaussian mutation
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if rng.random() < GA_CROSSOVER_P:
                pt = rng.integers(1, len(ALL_FEATURES))
                c1 = np.concatenate([selected[i][:pt], selected[i+1][pt:]])
                c2 = np.concatenate([selected[i+1][:pt], selected[i][pt:]])
            else:
                c1, c2 = selected[i].copy(), selected[i+1].copy()
            for c in [c1, c2]:
                for d in range(len(c)):
                    if rng.random() < GA_MUTATE_P:
                        c[d] += rng.normal(0, (MAXS[d] - MINS[d]) * 0.06)
                offspring.append(c)

        pop = np.vstack([elite,
                         np.array(offspring[:GA_POP_SIZE - len(elite)])])

    # ── Final Pareto front ───────────────────────────────────────────────
    final_fit = np.array([
        evaluate_design(c, strength_model, modulus_model, imputer, scaler)
        for c in pop
    ])
    final_fronts = non_dominated_sort(final_fit)
    pareto_final_idx = final_fronts[0]
    pareto_str = final_fit[pareto_final_idx, 0]
    pareto_mod = final_fit[pareto_final_idx, 1]

    # Decode optimal
    optimal_chrom = decode_chromosome(best_overall_chrom)
    opt_s, opt_m = evaluate_design(
        best_overall_chrom, strength_model, modulus_model, imputer, scaler)

    # Print results
    fib_name = RANK_TO_FIB.get(int(optimal_chrom[0]), "IM7")
    print(f"\n  {'=' * 56}")
    print(f"  OPTIMAL CFRP DESIGN — GA Multi-Objective Optimisation")
    print(f"  {'=' * 56}")
    print(f"  Predicted Tensile Strength : {opt_s:.1f} MPa  "
          f"({opt_s * MPa_TO_ksi:.1f} ksi)")
    print(f"  Predicted Tensile Modulus  : {opt_m:.1f} GPa  "
          f"({opt_m * GPa_TO_Msi:.1f} Msi)")
    print(f"  Fibre Type                 : {fib_name}")
    print(f"  Fibre Volume Fraction      : {optimal_chrom[3]:.1f} %")
    print(f"  Ply Fractions              : 0°={optimal_chrom[4]:.1f}%  "
          f"45°={optimal_chrom[5]:.1f}%  90°={optimal_chrom[6]:.1f}%")
    print(f"  CNT content                : {optimal_chrom[8]:.3f} vol%")
    print(f"  Interlayer content         : {optimal_chrom[9]:.2f} vol%")
    print(f"  Cure Pressure              : {optimal_chrom[11]:.0f} psi")
    print(f"  Tg (dry)                   : {optimal_chrom[12]:.0f} °C")
    print(f"  Orientation efficiency η₀  : {optimal_chrom[20]:.3f}")
    print(f"  Composite density          : {optimal_chrom[28]:.3f} g/cm³")
    print(f"  Specific strength proxy    : {optimal_chrom[29]:.0f} MPa·cm³/g")
    print(f"  Pareto front size          : {len(pareto_final_idx)}")
    print(f"  {'=' * 56}")

    return {
        "optimal_chrom":   optimal_chrom,
        "opt_strength":    opt_s,
        "opt_modulus":     opt_m,
        "pareto_strength": pareto_str,
        "pareto_modulus":  pareto_mod,
        "pareto_idx":      pareto_final_idx,
        "final_fit":       final_fit,
        "gen_best_s":      gen_best_s,
        "gen_best_m":      gen_best_m,
        "gen_mean_s":      gen_mean_s,
        "all_fitnesses":   all_fitnesses,
        "pareto_history":  pareto_history,
        "pop":             pop,
    }
