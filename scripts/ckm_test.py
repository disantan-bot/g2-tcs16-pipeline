#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  TEST CKM: ¿Los l_s del fit PMNS reproducen la mezcla de quarks?
═══════════════════════════════════════════════════════════════════════════

  Hipótesis: las mismas l_s sectoriales que producen PMNS deben 
  reproducir CKM, porque ambos sectores comparten los 3-ciclos 
  asociativos de X₇ y las mismas distancias geodésicas d_ij.

  Parámetros COMPARTIDOS (del fit PMNS):
    l_s₁, l_s₂, l_s₃   — longitudes de cuerda sectoriales
    d_ij                 — distancias geodésicas E₈ (fijadas)
    d_H_k                — distancias al Higgs (fijadas)

  Parámetros PROPIOS del sector quark:
    C₀_q     — prefactor instanton (puede diferir del leptónico)
    κ         — acoplamiento flux C₃ (con κ_up < 0, κ_dn > 0)
    δ_CKM    — fase CP = 64.5° (fijada por HK twist, Etapa 10b)

  Test en 3 niveles:
    Tier 1: l_s fijados de PMNS + C₀(BPS)=0.229 + κ=0.035
    Tier 2: l_s fijados de PMNS + C₀_q y κ optimizados (2 params)
    Tier 3: l_s fijados de PMNS + C₀_q, κ, NK_boost optimizados

  Fórmula (Compendio Etapa 10a):
    A_ij = C₀ exp(-d_ij/l_s_eff) ± κ · N_ij · l_s_eff² / d_ij²
    
  con N₁₂=1, N₂₃=N₁₃=2, sign(κ_up)=-1, sign(κ_dn)=+1

  Diego Santana S. — Marzo 2026
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════

# Geometric (E₈, fixed)
d_H = np.array([0.561, 0.347, 0.198])  # diagonal distances per generation
d_12, d_23, d_13 = 0.166, 0.343, 0.343  # off-diagonal distances
d_off = {(0,1): d_12, (1,2): d_23, (0,2): d_13}
N_flux = {(0,1): 1, (1,2): 2, (0,2): 2}  # flux quanta from E₈

# CKM phase (topological, Etapa 10b)
delta_CKM = np.radians(64.5)

# Quark masses at M_Z (PDG 2024, GeV)
m_up   = np.array([2.16e-3, 1.27, 172.76])   # u, c, t
m_down = np.array([4.67e-3, 0.0934, 4.18])    # d, s, b

# CKM experimental values
CKM_EXP = {
    't12': np.radians(13.02),  # θ₁₂ Cabibbo
    't23': np.radians(2.40),   # θ₂₃
    't13': np.radians(0.211),  # θ₁₃
    'delta': np.radians(65.0),
    'Vus': 0.2253,
    'Vcb': 0.0413,
    'Vub': 0.00361,
    'J': 3.08e-5,  # Jarlskog invariant
}

# Neutrino sector constants (for PMNS solution generation)
lambda_ACyl = 2.8; alpha_FHN = 0.15
t_cones = np.array([0.35, 0.50, 0.65])
v_ew = 246.22 / np.sqrt(2)
delta_PMNS = np.pi

PMNS_EXP = {'t12': np.radians(33.41), 't23': np.radians(49.1), 't13': np.radians(8.54),
            'dm2_21': 7.41e-5, 'dm2_32': 2.507e-3}

# Hitchin integrals
S_CACHE = {}
for i in range(3):
    for j in range(i+1, 3):
        def f(t, ii=i, jj=j):
            s2 = 1.0/np.cosh(lambda_ACyl*(t-0.5))**2
            th = abs(np.tanh(lambda_ACyl*(t-0.5)))
            return s2*(1+alpha_FHN*th)
        val, _ = quad(f, t_cones[i], t_cones[j])
        S_CACHE[(i,j)] = val


# ═══════════════════════════════════════════════════════════════
# PMNS SEESAW ENGINE (to regenerate solutions)
# ═══════════════════════════════════════════════════════════════

def pmns_cost(params):
    try:
        logM1,logM2,logM3, log_mu, logAD, logls1,logls2,logls3, logC0 = params
        M_diag = [10**logM1, 10**logM2, 10**logM3]
        mu = 10**log_mu; AD = 10**logAD
        ls = [10**logls1, 10**logls2, 10**logls3]; C0 = 10**logC0
        
        F = np.ones((3,3))
        for i in range(3):
            for j in range(i+1,3):
                F[i,j] = F[j,i] = np.exp(-mu * S_CACHE[(i,j)])
        MR = np.zeros((3,3))
        for i in range(3):
            MR[i,i] = M_diag[i]
            for j in range(i+1,3):
                MR[i,j] = MR[j,i] = np.sqrt(M_diag[i]*M_diag[j]) * F[i,j]
        eigs = np.linalg.eigvalsh(MR)
        if np.any(eigs <= 0): return 1e10
        
        y_D = np.array([AD*np.exp(-d_H[k]/ls[k]) for k in range(3)])
        Y = np.diag(y_D.astype(complex))
        phase = np.exp(1j*delta_PMNS)
        for (i,j) in [(0,1),(1,2),(0,2)]:
            lse = np.sqrt(ls[i]*ls[j])
            amp = C0*np.exp(-d_off[(i,j)]/lse)*np.sqrt(y_D[i]*y_D[j])
            Y[i,j] = amp*phase; Y[j,i] = amp*np.conj(phase)
        mD = Y * v_ew
        
        m_nu = -mD @ np.linalg.inv(MR.astype(complex)) @ mD.T
        H = m_nu.conj().T @ m_nu
        ev, V = np.linalg.eigh(H)
        m = np.sqrt(np.abs(ev)); idx = np.argsort(m); m = m[idx]*1e9
        U = np.abs(V[:,idx])
        s13 = np.clip(U[0,2],0,1); c13 = np.sqrt(max(1-s13**2,1e-20))
        s12 = np.clip(U[0,1]/c13,0,1); s23 = np.clip(U[1,2]/c13,0,1)
        t12 = np.arcsin(s12); t23 = np.arcsin(s23); t13 = np.arcsin(s13)
        dm21 = m[1]**2-m[0]**2; dm32 = m[2]**2-m[1]**2
        
        ea = 50*(((t12-PMNS_EXP['t12'])/PMNS_EXP['t12'])**2 +
                  ((t23-PMNS_EXP['t23'])/PMNS_EXP['t23'])**2 +
                  ((t13-PMNS_EXP['t13'])/PMNS_EXP['t13'])**2)
        if dm21 > 0 and dm32 > 0:
            ed = 20*((np.log10(dm21)-np.log10(PMNS_EXP['dm2_21']))**2 +
                      (np.log10(dm32)-np.log10(PMNS_EXP['dm2_32']))**2)
        else: ed = 200
        return ea + ed
    except: return 1e10


# ═══════════════════════════════════════════════════════════════
# CKM ENGINE
# ═══════════════════════════════════════════════════════════════

def build_quark_matrix(masses, ls_vec, C0_q, kappa, kappa_sign, nk_boost=0.0):
    """
    Build quark mass matrix with instanton + flux off-diagonals.
    
    M_ij = m_i × δ_ij + A_ij × √(m_i × m_j) × e^{i δ_CKM}
    
    A_ij = C₀_q × exp(-d_ij/l_s_eff) + kappa_sign × κ × N_ij × l_s_eff² / d_ij²
    
    NK boost: multiplies d₁₂ by (1 + nk_boost) to fine-tune θ₁₂
    """
    M = np.diag(masses.astype(complex))
    phase = np.exp(1j * delta_CKM)
    
    for (i,j) in [(0,1), (1,2), (0,2)]:
        ls_eff = np.sqrt(ls_vec[i] * ls_vec[j])
        d = d_off[(i,j)]
        if i == 0 and j == 1:
            d *= (1.0 + nk_boost)  # NK boost only on d₁₂ (Etapa 10d)
        
        # Instanton amplitude
        A_inst = C0_q * np.exp(-d / ls_eff)
        
        # Flux threading (Etapa 10a)
        A_flux = kappa * kappa_sign * N_flux[(i,j)] * ls_eff**2 / d**2
        
        A_total = A_inst + A_flux
        off = A_total * np.sqrt(masses[i] * masses[j]) * phase
        M[i,j] = off
        M[j,i] = np.conj(off)
    
    return M


def extract_ckm(M_u, M_d):
    """
    Diagonalize M_u and M_d, extract CKM = V_u† × V_d.
    Returns CKM matrix elements and angles.
    """
    # Diagonalize M†M for each sector
    H_u = M_u.conj().T @ M_u
    H_d = M_d.conj().T @ M_d
    
    _, V_u = np.linalg.eigh(H_u)
    _, V_d = np.linalg.eigh(H_d)
    
    # CKM = V_u† × V_d
    V_ckm = V_u.conj().T @ V_d
    Va = np.abs(V_ckm)
    
    # Standard parameterization
    # |V_us| ~ sin(θ₁₂), |V_cb| ~ sin(θ₂₃), |V_ub| ~ sin(θ₁₃)
    Vus = Va[0,1]; Vcb = Va[1,2]; Vub = Va[0,2]
    
    s13 = np.clip(Vub, 0, 1)
    c13 = np.sqrt(max(1 - s13**2, 1e-20))
    s12 = np.clip(Vus / c13, 0, 1)
    s23 = np.clip(Vcb / c13, 0, 1)
    
    t12 = np.arcsin(np.clip(s12, 0, 1))
    t23 = np.arcsin(np.clip(s23, 0, 1))
    t13 = np.arcsin(np.clip(s13, 0, 1))
    
    # Jarlskog invariant
    J = abs(np.imag(V_ckm[0,0]*V_ckm[1,1]*np.conj(V_ckm[0,1])*np.conj(V_ckm[1,0])))
    
    return {
        't12': t12, 't23': t23, 't13': t13,
        'Vus': Vus, 'Vcb': Vcb, 'Vub': Vub,
        'J': J, 'V': Va
    }


def ckm_cost(params_q, ls_vec):
    """
    CKM cost for given quark params and fixed l_s from PMNS.
    params_q = [log_C0q, log_kappa, nk_boost]
    """
    try:
        log_C0q, log_kappa, nk_boost = params_q
        C0_q = 10**log_C0q
        kappa = 10**log_kappa
        
        M_u = build_quark_matrix(m_up, ls_vec, C0_q, kappa, -1, nk_boost)
        M_d = build_quark_matrix(m_down, ls_vec, C0_q, kappa, +1, nk_boost)
        
        ckm = extract_ckm(M_u, M_d)
        
        # Cost: 3 CKM angles + Jarlskog
        cost = 0
        for key in ['t12', 't23', 't13']:
            cost += 30 * ((ckm[key] - CKM_EXP[key]) / CKM_EXP[key])**2
        
        # Jarlskog (logarithmic)
        if ckm['J'] > 0:
            cost += 10 * (np.log10(ckm['J']) - np.log10(CKM_EXP['J']))**2
        else:
            cost += 100
        
        return cost
    except:
        return 1e10


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

print("═" * 72)
print("  TEST CKM: Consistencia con l_s Sectoriales del Fit PMNS")
print("═" * 72)


# ─── Paso 1: Regenerar soluciones PMNS ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PASO 1: Regenerar 25 Soluciones PMNS                        ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

bounds_pmns = [(8,16),(8,16),(8,16), (0.5,3), (-4,3),
               (-2.5,-0.2),(-2.5,-0.2),(-2.5,-0.2), (-0.5,5)]

pmns_solutions = []
print(f"  Optimizando (semillas 0-39)...", flush=True)
for seed in range(10):
    res = differential_evolution(pmns_cost, bounds_pmns, seed=seed,
                                  maxiter=200, tol=1e-12, popsize=15,
                                  mutation=(0.5,1.8), recombination=0.85)
    if res.fun < 0.5:
        p = res.x
        ls = [10**p[5], 10**p[6], 10**p[7]]
        pmns_solutions.append({
            'seed': seed, 'cost': res.fun,
            'ls': ls, 'C0_pmns': 10**p[8],
            'params': p
        })

print(f"  Soluciones PMNS: {len(pmns_solutions)}/40")
print(f"  Rango l_s₁: [{min(s['ls'][0] for s in pmns_solutions):.3f}, {max(s['ls'][0] for s in pmns_solutions):.3f}]")
print(f"  Rango l_s₂: [{min(s['ls'][1] for s in pmns_solutions):.3f}, {max(s['ls'][1] for s in pmns_solutions):.3f}]")
print(f"  Rango l_s₃: [{min(s['ls'][2] for s in pmns_solutions):.3f}, {max(s['ls'][2] for s in pmns_solutions):.3f}]")


# ─── Paso 2: Test Tier 1 — Parámetros del Compendio ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PASO 2: TIER 1 — C₀(BPS)=0.229, κ=0.035 (Compendio)       ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

print(f"\n  Test: l_s del PMNS + C₀ y κ fijados del Compendio")
print(f"  Expectativa: con l_s ~ 0.3 (vs 0.044 original), las")
print(f"  amplitudes off-diagonal son mucho mayores → ¿CKM suficiente?")

C0_bps = 0.229
kappa_pheno = 0.035

print(f"\n  {'Seed':>4} {'l_s₁':>6} {'l_s₂':>6} {'l_s₃':>6} {'θ₁₂':>7} {'θ₂₃':>7} {'θ₁₃':>7} {'|Vus|':>7} {'|Vcb|':>7} {'|Vub|':>8} {'ok?':>4}")
print(f"  {'─'*78}")
print(f"  {'exp':>4} {'':>6} {'':>6} {'':>6} {'13.02':>7} {'2.40':>7} {'0.211':>7} {'0.225':>7} {'0.041':>7} {'0.0036':>8}")
print(f"  {'─'*78}")

tier1_results = []
for sol in pmns_solutions:
    ls = sol['ls']
    M_u = build_quark_matrix(m_up, ls, C0_bps, kappa_pheno, -1, 0.0)
    M_d = build_quark_matrix(m_down, ls, C0_bps, kappa_pheno, +1, 0.0)
    ckm = extract_ckm(M_u, M_d)
    
    t12d = np.degrees(ckm['t12']); t23d = np.degrees(ckm['t23']); t13d = np.degrees(ckm['t13'])
    ok_12 = abs(t12d/13.02 - 1) < 0.30
    ok_23 = abs(t23d/2.40 - 1) < 0.30
    ok_13 = abs(t13d/0.211 - 1) < 0.50
    n_ok = sum([ok_12, ok_23, ok_13])
    mark = f"{n_ok}/3" + (" ✅" if n_ok == 3 else "")
    
    tier1_results.append({**sol, 'ckm': ckm, 't12d': t12d, 't23d': t23d, 't13d': t13d, 'n_ok': n_ok})
    print(f"  {sol['seed']:>4} {ls[0]:>6.3f} {ls[1]:>6.3f} {ls[2]:>6.3f} "
          f"{t12d:>7.2f} {t23d:>7.2f} {t13d:>7.3f} "
          f"{ckm['Vus']:>7.4f} {ckm['Vcb']:>7.4f} {ckm['Vub']:>8.5f} {mark:>4}")

n_pass = sum(1 for r in tier1_results if r['n_ok'] >= 2)
print(f"\n  Tier 1 resultado: {n_pass}/{len(tier1_results)} pasan (≥2/3 ángulos CKM ±30%)")


# ─── Paso 3: Test Tier 2 — Optimizar C₀_q y κ ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PASO 3: TIER 2 — C₀_q y κ optimizados, l_s fijados PMNS   ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

print(f"\n  Ahora optimizamos C₀_q y κ (2 params + NK boost) para cada sol")
print(f"  El test es: ¿EXISTE un C₀_q, κ que reproduzca CKM?")

bounds_ckm = [(-3, 2), (-4, 1), (-0.05, 0.30)]  # log_C0q, log_kappa, nk_boost

print(f"\n  {'Seed':>4} {'C₀_q':>8} {'κ':>8} {'NK%':>5} {'θ₁₂':>7} {'θ₂₃':>7} {'θ₁₃':>7} {'|J|':>10} {'cost':>8} {'CKM':>5}")
print(f"  {'─'*78}")
print(f"  {'exp':>4} {'':>8} {'':>8} {'':>5} {'13.02':>7} {'2.40':>7} {'0.211':>7} {'3.08e-5':>10}")
print(f"  {'─'*78}")

tier2_results = []
for sol in pmns_solutions:
    ls = sol['ls']
    
    best_c = np.inf
    best_r = None
    for s in range(2):
        res = differential_evolution(ckm_cost, bounds_ckm, args=(ls,),
                                      seed=s*7+sol['seed'], maxiter=150,
                                      tol=1e-12, popsize=12, mutation=(0.5,1.5))
        if res.fun < best_c:
            best_c = res.fun
            best_r = res
    
    p = best_r.x
    C0_q = 10**p[0]; kappa = 10**p[1]; nk = p[2]
    
    M_u = build_quark_matrix(m_up, ls, C0_q, kappa, -1, nk)
    M_d = build_quark_matrix(m_down, ls, C0_q, kappa, +1, nk)
    ckm = extract_ckm(M_u, M_d)
    
    t12d = np.degrees(ckm['t12']); t23d = np.degrees(ckm['t23']); t13d = np.degrees(ckm['t13'])
    ok_all = (abs(t12d/13.02-1)<0.30 and abs(t23d/2.40-1)<0.30 and abs(t13d/0.211-1)<0.50)
    mark = "✅" if ok_all else ("⊕" if best_c < 5 else "⚠️")
    
    tier2_results.append({
        **sol, 'C0_q': C0_q, 'kappa': kappa, 'nk_boost': nk,
        'ckm': ckm, 'cost_ckm': best_c,
        't12d': t12d, 't23d': t23d, 't13d': t13d
    })
    
    print(f"  {sol['seed']:>4} {C0_q:>8.4f} {kappa:>8.5f} {nk*100:>5.1f} "
          f"{t12d:>7.2f} {t23d:>7.2f} {t13d:>7.3f} "
          f"{ckm['J']:>10.2e} {best_c:>8.3f} {mark:>5}")

# Statistics
n_pass_t2 = sum(1 for r in tier2_results if r['cost_ckm'] < 2.0)
n_good_t2 = sum(1 for r in tier2_results if r['cost_ckm'] < 0.5)
print(f"\n  Tier 2 resultado:")
print(f"    Cost < 0.5 (excelente): {n_good_t2}/{len(tier2_results)}")
print(f"    Cost < 2.0 (aceptable): {n_pass_t2}/{len(tier2_results)}")
print(f"    Cost < 5.0 (marginal):  {sum(1 for r in tier2_results if r['cost_ckm']<5)}/{len(tier2_results)}")


# ─── Paso 4: Análisis de los parámetros CKM óptimos ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PASO 4: Análisis de Parámetros CKM Óptimos                 ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

good = [r for r in tier2_results if r['cost_ckm'] < 5.0]
if good:
    C0qs = [r['C0_q'] for r in good]
    kappas = [r['kappa'] for r in good]
    nks = [r['nk_boost'] for r in good]
    t12s = [r['t12d'] for r in good]
    t23s = [r['t23d'] for r in good]
    t13s = [r['t13d'] for r in good]
    
    print(f"\n  Para las {len(good)} soluciones con cost_CKM < 5:")
    print(f"    C₀_q:  [{min(C0qs):.4f}, {max(C0qs):.4f}]  median = {np.median(C0qs):.4f}")
    print(f"    κ:     [{min(kappas):.5f}, {max(kappas):.5f}]  median = {np.median(kappas):.5f}")
    print(f"    NK%:   [{min(nks)*100:.1f}%, {max(nks)*100:.1f}%]  median = {np.median(nks)*100:.1f}%")
    print(f"    θ₁₂:  [{min(t12s):.2f}°, {max(t12s):.2f}°]  (exp: 13.02°)")
    print(f"    θ₂₃:  [{min(t23s):.2f}°, {max(t23s):.2f}°]  (exp: 2.40°)")
    print(f"    θ₁₃:  [{min(t13s):.3f}°, {max(t13s):.3f}°]  (exp: 0.211°)")
    
    # Compare C₀_q with C₀(BPS)
    print(f"\n  ─── C₀_q vs C₀(BPS) ───")
    print(f"    C₀(BPS) = 0.229 (Compendio Etapa 13)")
    print(f"    C₀_q(median) = {np.median(C0qs):.4f}")
    print(f"    Ratio: {np.median(C0qs)/0.229:.2f}×")
    
    # Compare κ with κ(pheno)
    print(f"\n  ─── κ vs κ(pheno) ───")
    print(f"    κ(pheno) = 0.035 (Compendio)")
    print(f"    κ(median) = {np.median(kappas):.5f}")
    print(f"    Ratio: {np.median(kappas)/0.035:.2f}×")
    
    # Correlation between l_s and C₀_q
    ls1s = [r['ls'][0] for r in good]
    ls2s = [r['ls'][1] for r in good]
    ls3s = [r['ls'][2] for r in good]
    
    if len(good) > 3:
        corr_c0_ls1 = np.corrcoef(C0qs, ls1s)[0,1]
        corr_c0_ls2 = np.corrcoef(C0qs, ls2s)[0,1]
        corr_k_ls1 = np.corrcoef(kappas, ls1s)[0,1]
        
        print(f"\n  ─── Correlaciones C₀_q / κ con l_s ───")
        print(f"    corr(C₀_q, l_s₁) = {corr_c0_ls1:+.3f}")
        print(f"    corr(C₀_q, l_s₂) = {corr_c0_ls2:+.3f}")
        print(f"    corr(κ, l_s₁)    = {corr_k_ls1:+.3f}")


# ─── Paso 5: Conteo de parámetros combinado ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PASO 5: Conteo de Parámetros Combinado (PMNS + CKM)        ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

print(f"""
  ANTES (solo PMNS):
    Parámetros: M₁,M₂,M₃, μ, A_D, l_s₁,l_s₂,l_s₃, C₀_PMNS = 9
    Observables: θ₁₂,θ₂₃,θ₁₃, Δm²₂₁,Δm²₃₂ = 5
    Ratio: 9:5 → subdeterminado (4 libres)
    
  AHORA (PMNS + CKM):
    Parámetros: los 9 anteriores + C₀_q, κ, NK_boost = 12
    Observables: 5 PMNS + θ₁₂,θ₂₃,θ₁₃,J del CKM = 9
    Ratio: 12:9 → aún subdeterminado (3 libres)
    
  PERO: l_s₁,l_s₂,l_s₃ aparecen en AMBOS sectores.
  Antes, las l_s tenían dispersión ~100% (frágiles).
  ¿El constraint CKM las fija?
""")

# Check: does the CKM constraint reduce l_s dispersion?
if good:
    pmns_ls1 = [s['ls'][0] for s in pmns_solutions]
    pmns_ls2 = [s['ls'][1] for s in pmns_solutions]
    pmns_ls3 = [s['ls'][2] for s in pmns_solutions]
    ckm_ls1 = [r['ls'][0] for r in good]
    ckm_ls2 = [r['ls'][1] for r in good]
    ckm_ls3 = [r['ls'][2] for r in good]
    
    def spread(arr):
        a = np.array(arr)
        return (np.percentile(a,84)-np.percentile(a,16))/np.median(a) if len(a)>2 else np.inf
    
    print(f"  Dispersión de l_s (ANTES vs DESPUÉS del filtro CKM):")
    print(f"    {'':>6} {'Solo PMNS':>14} {'PMNS+CKM':>14} {'Reducción':>12}")
    print(f"    {'─'*48}")
    print(f"    l_s₁  {spread(pmns_ls1):>14.1%} {spread(ckm_ls1):>14.1%} {1-spread(ckm_ls1)/spread(pmns_ls1):>12.0%}")
    print(f"    l_s₂  {spread(pmns_ls2):>14.1%} {spread(ckm_ls2):>14.1%} {1-spread(ckm_ls2)/spread(pmns_ls2):>12.0%}")
    print(f"    l_s₃  {spread(pmns_ls3):>14.1%} {spread(ckm_ls3):>14.1%} {1-spread(ckm_ls3)/spread(pmns_ls3):>12.0%}")

    # Does m₁ become more robust?
    # We need to check m₁ from the PMNS solutions that pass CKM
    good_seeds = set(r['seed'] for r in good)
    
    # Re-compute m₁ for good solutions
    all_m1 = []
    good_m1 = []
    for sol in pmns_solutions:
        p = sol['params']
        M_diag = [10**p[0], 10**p[1], 10**p[2]]
        mu = 10**p[3]; AD = 10**p[4]
        ls = [10**p[5], 10**p[6], 10**p[7]]; C0 = 10**p[8]
        
        F = np.ones((3,3))
        for i in range(3):
            for j in range(i+1,3):
                F[i,j] = F[j,i] = np.exp(-mu * S_CACHE[(i,j)])
        MR = np.zeros((3,3))
        for i in range(3):
            MR[i,i] = M_diag[i]
            for j in range(i+1,3):
                MR[i,j] = MR[j,i] = np.sqrt(M_diag[i]*M_diag[j]) * F[i,j]
        
        y_D = np.array([AD*np.exp(-d_H[k]/ls[k]) for k in range(3)])
        Y = np.diag(y_D.astype(complex))
        phase = np.exp(1j*delta_PMNS)
        for (i,j) in [(0,1),(1,2),(0,2)]:
            lse = np.sqrt(ls[i]*ls[j])
            amp = C0*np.exp(-d_off[(i,j)]/lse)*np.sqrt(y_D[i]*y_D[j])
            Y[i,j] = amp*phase; Y[j,i] = amp*np.conj(phase)
        mD = Y * v_ew
        
        try:
            m_nu = -mD @ np.linalg.inv(MR.astype(complex)) @ mD.T
            H = m_nu.conj().T @ m_nu
            ev = np.sort(np.linalg.eigvalsh(H))
            m = np.sqrt(np.abs(ev)) * 1e9
            all_m1.append(m[0])
            if sol['seed'] in good_seeds:
                good_m1.append(m[0])
        except:
            pass
    
    if good_m1:
        print(f"\n  Dispersión de m₁ (ANTES vs DESPUÉS del filtro CKM):")
        print(f"    Solo PMNS: [{min(all_m1):.2e}, {max(all_m1):.2e}] eV ({len(all_m1)} sol)")
        print(f"    PMNS+CKM:  [{min(good_m1):.2e}, {max(good_m1):.2e}] eV ({len(good_m1)} sol)")
        r_before = max(all_m1)/min(all_m1) if min(all_m1)>0 else np.inf
        r_after = max(good_m1)/min(good_m1) if min(good_m1)>0 else np.inf
        print(f"    Spread: {r_before:.0f}× → {r_after:.0f}× ({1-r_after/r_before:.0%} reducción)")


# ─── Paso 6: Conclusión ───
print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  CONCLUSIÓN                                                   ║
╚═══════════════════════════════════════════════════════════════╝
""")

n_total = len(pmns_solutions)
n_compat = len(good)
if tier2_results:
    n_exc = sum(1 for r in tier2_results if r['cost_ckm'] < 0.5)
    
    print(f"  De {n_total} soluciones PMNS:")
    print(f"    {n_compat}/{n_total} ({n_compat/n_total:.0%}) son compatibles con CKM (cost < 5)")
    print(f"    {n_exc}/{n_total} ({n_exc/n_total:.0%}) reproducen CKM excelentemente (cost < 0.5)")
    
    if n_compat > 0.5 * n_total:
        print(f"\n  → RESULTADO: COMPATIBLE ✅")
        print(f"    La mayoría de las soluciones PMNS admiten CKM.")
        print(f"    l_s sectoriales son consistentes entre leptones y quarks.")
    elif n_compat > 0.2 * n_total:
        print(f"\n  → RESULTADO: PARCIALMENTE COMPATIBLE ⊕")
        print(f"    Subconjunto significativo admite CKM.")
        print(f"    El filtro CKM reduce la variedad de soluciones.")
    else:
        print(f"\n  → RESULTADO: TENSION ⚠️")
        print(f"    Pocas soluciones admiten ambos PMNS y CKM.")
        print(f"    l_s(quarks) ≠ l_s(leptones) parece necesario.")

print(f"\n{'═'*72}")
print(f"  Test CKM completo.")
print(f"{'═'*72}")
