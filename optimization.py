"""NSGA-II multi-objective GA for CFRP design optimisation."""

import numpy as np
from config import (
    SEED, ALL_FEATURES, GA_BOUNDS, GA_POP_SIZE, GA_N_GEN,
    GA_MUTATE_P, GA_CROSSOVER_P, GA_ELITE_K,
    RANK_TO_MOD, RANK_TO_STR, RANK_TO_DEN, RANK_TO_FIB,
    MATRIX_MODULUS_GPa, MATRIX_STRENGTH_MPa, MATRIX_DENSITY_gcc,
    MPa_TO_ksi, GPa_TO_Msi, section
)

MINS = np.array([v[0] for v in GA_BOUNDS.values()])
MAXS = np.array([v[1] for v in GA_BOUNDS.values()])


def decode_chromosome(g):
    g = g.copy()
    E_m, sigma_m = MATRIX_MODULUS_GPa, MATRIX_STRENGTH_MPa

    for i, (feat, (lo, hi, typ)) in enumerate(GA_BOUNDS.items()):
        g[i] = np.clip(g[i], lo, hi)
        if typ == "int": g[i] = round(g[i])

    p0, p45, p90 = g[4], g[5], g[6]
    total = p0 + p45 + p90
    if total > 0:
        g[4] = round(p0/total*100, 1)
        g[5] = round(p45/total*100, 1)
        g[6] = round(100-g[4]-g[5], 1)

    rank = int(g[0])
    g[1] = RANK_TO_MOD.get(rank, 230)
    g[2] = RANK_TO_STR.get(rank, 3500)
    rho_f = RANK_TO_DEN.get(rank, 1.78)

    Vf, Vm = g[3]/100.0, 1.0 - g[3]/100.0

    g[18] = Vf*g[1] + Vm*E_m
    xi, eta = 2.0, (g[1]/E_m-1)/(g[1]/E_m+2)
    g[19] = E_m*(1+xi*eta*Vf)/(1-eta*Vf)
    p0n, p45n = g[4]/100, g[5]/100
    g[20] = p0n*np.cos(np.radians(0))**4 + p45n*np.cos(np.radians(45))**4
    g[21] = Vf*g[2] + Vm*sigma_m
    g[22] = 23.0/max(g[12], 1.0)
    g[23] = g[3]
    g[24] = (6.0-g[10])*np.sqrt(max(g[11], 0.1)/100)
    g[25] = g[18]/max(g[19], 0.1)
    g[26] = g[8]*(1+g[9]/10.0)
    g[27] = p0n*g[21]
    g[28] = Vf*rho_f + Vm*MATRIX_DENSITY_gcc
    g[29] = g[21]/max(g[28], 0.1)
    return g


def evaluate_design(chromosome, strength_model, modulus_model, imputer, scaler):
    g = decode_chromosome(chromosome)
    Xg = scaler.transform(imputer.transform([g]))
    return float(strength_model.predict(Xg)[0]), float(modulus_model.predict(Xg)[0])


def _dominates(a, b):
    return a[0] >= b[0] and a[1] >= b[1] and (a[0] > b[0] or a[1] > b[1])


def non_dominated_sort(fitnesses):
    n = len(fitnesses)
    dominated_count = np.zeros(n, dtype=int)
    dominates_set = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if _dominates(fitnesses[i], fitnesses[j]):
                dominates_set[i].append(j); dominated_count[j] += 1
            elif _dominates(fitnesses[j], fitnesses[i]):
                dominates_set[j].append(i); dominated_count[i] += 1
    fronts, current = [], np.where(dominated_count == 0)[0].tolist()
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0: nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(fitnesses, front):
    n = len(front)
    if n <= 2: return np.full(n, np.inf)
    cd = np.zeros(n)
    for obj in range(2):
        vals = fitnesses[front, obj]
        order = np.argsort(vals)
        cd[order[0]] = cd[order[-1]] = np.inf
        rng_ = vals[order[-1]] - vals[order[0]]
        if rng_ < 1e-10: continue
        for k in range(1, n-1):
            cd[order[k]] += (vals[order[k+1]]-vals[order[k-1]])/rng_
    return cd


def run_ga(strength_model, modulus_model, imputer, scaler):
    section("PHASE 6 — MULTI-OBJECTIVE GA OPTIMISATION")
    rng = np.random.default_rng(SEED)
    pop = rng.uniform(MINS, MAXS, (GA_POP_SIZE, len(ALL_FEATURES)))

    gen_best_s, gen_best_m, gen_mean_s = [], [], []
    all_fitnesses, pareto_history = [], []
    best_overall_s, best_overall_chrom = -np.inf, None

    print(f"\n  GA: Pop={GA_POP_SIZE}  Gens={GA_N_GEN}")
    print(f"  {'Gen':>5}  {'Best σ':>10}  {'Best E':>10}  {'|F₁|':>5}")
    print(f"  {'─'*38}")

    for gen in range(GA_N_GEN):
        fit_pairs = np.array([evaluate_design(c, strength_model, modulus_model, imputer, scaler) for c in pop])
        strengths, moduli = fit_pairs[:,0], fit_pairs[:,1]
        all_fitnesses.append(fit_pairs.copy())

        fronts = non_dominated_sort(fit_pairs)
        pareto_history.append(fit_pairs[fronts[0]])
        gen_best_s.append(strengths.max())
        gen_best_m.append(moduli.max())
        gen_mean_s.append(strengths.mean())

        if strengths.max() > best_overall_s:
            best_overall_s = strengths.max()
            best_overall_chrom = pop[np.argmax(strengths)].copy()

        if gen % 10 == 0 or gen == GA_N_GEN-1:
            print(f"  {gen+1:>5}  {strengths.max():>10.1f}  {moduli.max():>10.2f}  {len(fronts[0]):>5}")

        ranks = np.zeros(len(pop), dtype=int)
        cds = np.zeros(len(pop))
        for rank_val, front in enumerate(fronts):
            for idx in front: ranks[idx] = rank_val
            cd = crowding_distance(fit_pairs, front)
            for i, idx in enumerate(front): cds[idx] = cd[i]

        elite_idx = fronts[0][:GA_ELITE_K] if len(fronts[0]) >= GA_ELITE_K else fronts[0]
        elite = pop[elite_idx].copy()

        selected = []
        for _ in range(GA_POP_SIZE - len(elite)):
            i, j = rng.integers(0, len(pop), 2)
            if ranks[i] < ranks[j]: selected.append(pop[i])
            elif ranks[j] < ranks[i]: selected.append(pop[j])
            else: selected.append(pop[i] if cds[i] > cds[j] else pop[j])
        selected = np.array(selected)

        offspring = []
        for i in range(0, len(selected)-1, 2):
            if rng.random() < GA_CROSSOVER_P:
                pt = rng.integers(1, len(ALL_FEATURES))
                c1 = np.concatenate([selected[i][:pt], selected[i+1][pt:]])
                c2 = np.concatenate([selected[i+1][:pt], selected[i][pt:]])
            else:
                c1, c2 = selected[i].copy(), selected[i+1].copy()
            for c in [c1, c2]:
                for d in range(len(c)):
                    if rng.random() < GA_MUTATE_P:
                        c[d] += rng.normal(0, (MAXS[d]-MINS[d])*0.06)
                offspring.append(c)
        pop = np.vstack([elite, np.array(offspring[:GA_POP_SIZE-len(elite)])])

    final_fit = np.array([evaluate_design(c, strength_model, modulus_model, imputer, scaler) for c in pop])
    final_fronts = non_dominated_sort(final_fit)
    pareto_final_idx = final_fronts[0]

    optimal_chrom = decode_chromosome(best_overall_chrom)
    opt_s, opt_m = evaluate_design(best_overall_chrom, strength_model, modulus_model, imputer, scaler)
    fib_name = RANK_TO_FIB.get(int(optimal_chrom[0]), "IM7")

    print(f"\n  {'='*50}")
    print(f"  OPTIMAL DESIGN")
    print(f"  σ = {opt_s:.1f} MPa ({opt_s*MPa_TO_ksi:.0f} ksi)")
    print(f"  E = {opt_m:.1f} GPa ({opt_m*GPa_TO_Msi:.1f} Msi)")
    print(f"  Fibre: {fib_name}  Vf: {optimal_chrom[3]:.1f}%")
    print(f"  Plies: 0°={optimal_chrom[4]:.0f}% 45°={optimal_chrom[5]:.0f}% 90°={optimal_chrom[6]:.0f}%")
    print(f"  Pareto front: {len(pareto_final_idx)} solutions")
    print(f"  {'='*50}")

    return {
        "optimal_chrom": optimal_chrom, "opt_strength": opt_s, "opt_modulus": opt_m,
        "pareto_strength": final_fit[pareto_final_idx, 0],
        "pareto_modulus": final_fit[pareto_final_idx, 1],
        "pareto_idx": pareto_final_idx, "final_fit": final_fit,
        "gen_best_s": gen_best_s, "gen_best_m": gen_best_m, "gen_mean_s": gen_mean_s,
        "all_fitnesses": all_fitnesses, "pareto_history": pareto_history, "pop": pop,
    }
