#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  DEGENERESCENCIA DEL MODELO VOLUMÉTRICO COMBINADO (PMNS + CKM)
  ¿Qué predicciones son robustas con 13 params y 9 observables?
═══════════════════════════════════════════════════════════════════════════
  Diego Santana S. — Marzo 2026
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import warnings, time
warnings.filterwarnings('ignore')

t_start = time.time()

# ═══ Constantes ═══
Vol_SU3 = 1.909798; Vol_SU2 = 0.953088; Vol_U1 = 1.121823
d_H = np.array([0.561, 0.347, 0.198])
d_12, d_23, d_13 = 0.166, 0.343, 0.343
d_off = {(0,1): d_12, (1,2): d_23, (0,2): d_13}
N_flux = {(0,1): 1, (1,2): 2, (0,2): 2}
t_cones = np.array([0.35, 0.50, 0.65])
lambda_ACyl = 2.8; alpha_FHN = 0.15
v_ew = 246.22/np.sqrt(2)
m_up = np.array([2.16e-3, 1.27, 172.76])
m_down = np.array([4.67e-3, 0.0934, 4.18])
delta_CKM = np.radians(64.5); delta_PMNS = np.pi

CKM_EXP = {'t12': np.radians(13.02), 't23': np.radians(2.40),
            't13': np.radians(0.211), 'J': 3.08e-5}
PMNS_EXP = {'t12': np.radians(33.41), 't23': np.radians(49.1),
            't13': np.radians(8.54), 'dm2_21': 7.41e-5, 'dm2_32': 2.507e-3}

# Hitchin integrals (pre-computed)
S_CACHE = {}
for i in range(3):
    for j in range(i+1, 3):
        def f(t, ii=i, jj=j):
            s2 = 1.0/np.cosh(lambda_ACyl*(t-0.5))**2
            th = abs(np.tanh(lambda_ACyl*(t-0.5)))
            return s2*(1+alpha_FHN*th)
        val, _ = quad(f, t_cones[i], t_cones[j])
        S_CACHE[(i,j)] = val


# ═══ Engine ═══
def build_mD_dual(AD, ls_D_vec, C0_D):
    ls1,ls2,ls3 = ls_D_vec
    y_D = np.array([AD*np.exp(-d_H[0]/ls1), AD*np.exp(-d_H[1]/ls2), AD*np.exp(-d_H[2]/ls3)])
    Y = np.diag(y_D.astype(complex))
    phase = np.exp(1j*delta_PMNS)
    for (i,j) in [(0,1),(1,2),(0,2)]:
        lse = np.sqrt(ls_D_vec[i]*ls_D_vec[j])
        amp = C0_D*np.exp(-d_off[(i,j)]/lse)*np.sqrt(y_D[i]*y_D[j])
        Y[i,j]=amp*phase; Y[j,i]=amp*np.conj(phase)
    return Y * v_ew

def build_MR_dual(M_diag, mu, ls_M_vec):
    ls1,ls2,ls3 = ls_M_vec
    F = np.ones((3,3))
    for i in range(3):
        for j in range(i+1,3):
            F[i,j] = F[j,i] = np.exp(-mu * S_CACHE[(i,j)])
    MR = np.zeros((3,3))
    for i in range(3):
        MR[i,i] = M_diag[i]
        for j in range(i+1,3):
            lse = np.sqrt(ls_M_vec[i]*ls_M_vec[j])
            F_extra = np.exp(-d_off[(i,j)] / lse)
            MR[i,j] = MR[j,i] = np.sqrt(M_diag[i]*M_diag[j]) * F[i,j] * F_extra
    return MR

def seesaw_full(mD, MR):
    m_nu = -mD @ np.linalg.inv(MR.astype(complex)) @ mD.T
    H = m_nu.conj().T @ m_nu
    ev, V = np.linalg.eigh(H)
    m = np.sqrt(np.abs(ev)); idx = np.argsort(m); m = m[idx]*1e9; U = np.abs(V[:,idx])
    s13 = np.clip(U[0,2],0,1); c13 = np.sqrt(max(1-s13**2,1e-20))
    s12 = np.clip(U[0,1]/c13,0,1); s23 = np.clip(U[1,2]/c13,0,1)
    return {'m': m, 'sum_m': np.sum(m),
            't12': np.arcsin(s12), 't23': np.arcsin(s23), 't13': np.arcsin(s13),
            'dm2_21': m[1]**2-m[0]**2, 'dm2_32': m[2]**2-m[1]**2,
            'ratio': (m[2]**2-m[1]**2)/(m[1]**2-m[0]**2) if m[1]**2>m[0]**2 else np.inf}

def build_quark_matrix(masses, ls_vec, C0_q, kappa, kappa_sign, nk_boost=0.0):
    M = np.diag(masses.astype(complex))
    phase = np.exp(1j * delta_CKM)
    for (i,j) in [(0,1),(1,2),(0,2)]:
        ls_eff = np.sqrt(ls_vec[i]*ls_vec[j])
        d = d_off[(i,j)] * (1.0+nk_boost if (i,j)==(0,1) else 1.0)
        A_inst = C0_q * np.exp(-d/ls_eff)
        A_flux = kappa * kappa_sign * N_flux[(i,j)] * ls_eff**2 / d**2
        off = (A_inst + A_flux) * np.sqrt(masses[i]*masses[j]) * phase
        M[i,j] = off; M[j,i] = np.conj(off)
    return M

def extract_ckm(M_u, M_d):
    _, V_u = np.linalg.eigh(M_u.conj().T @ M_u)
    _, V_d = np.linalg.eigh(M_d.conj().T @ M_d)
    V = V_u.conj().T @ V_d; Va = np.abs(V)
    s13 = np.clip(Va[0,2],0,1); c13 = np.sqrt(max(1-s13**2,1e-20))
    s12 = np.clip(Va[0,1]/c13,0,1); s23 = np.clip(Va[1,2]/c13,0,1)
    J = abs(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0])))
    return {'t12': np.arcsin(s12), 't23': np.arcsin(s23), 't13': np.arcsin(s13), 'J': J}


def decode_params(params):
    """Decode 13-vector into physical quantities."""
    (logM1,logM2,logM3, log_mu, logAD, logC0D,
     log_l0, p_vol, q_gen, log_l0M,
     logC0q, log_kappa, nk_boost) = params
    M_diag = [10**logM1, 10**logM2, 10**logM3]
    mu = 10**log_mu; AD = 10**logAD; C0_D = 10**logC0D
    l0 = 10**log_l0; l0M = 10**log_l0M
    C0_q = 10**logC0q; kappa = 10**log_kappa
    d_ref = max(d_H)
    g = np.array([(d_H[k]/d_ref)**q_gen for k in range(3)])
    ls_lep = l0 / Vol_SU2**p_vol * g
    ls_quark = l0 / Vol_SU3**p_vol * g
    ls_Maj = l0M * g
    ratio_lq = (Vol_SU3/Vol_SU2)**p_vol
    return {
        'M_diag': M_diag, 'mu': mu, 'AD': AD, 'C0_D': C0_D,
        'l0': l0, 'p_vol': p_vol, 'q_gen': q_gen, 'l0M': l0M,
        'C0_q': C0_q, 'kappa': kappa, 'nk_boost': nk_boost,
        'ls_lep': ls_lep, 'ls_quark': ls_quark, 'ls_Maj': ls_Maj,
        'ratio_lq': ratio_lq, 'g': g
    }


def combined_cost(params):
    try:
        d = decode_params(params)
        if np.any(d['ls_lep'] <= 0) or np.any(d['ls_quark'] <= 0) or np.any(d['ls_Maj'] <= 0):
            return 1e10
        
        # PMNS
        mD = build_mD_dual(d['AD'], d['ls_lep'], d['C0_D'])
        MR = build_MR_dual(d['M_diag'], d['mu'], d['ls_Maj'])
        eigs = np.linalg.eigvalsh(MR)
        if np.any(eigs <= 0): return 1e10
        r = seesaw_full(mD, MR)
        if np.any(np.isnan(r['m'])): return 1e10
        
        cost_pmns = 50*(((r['t12']-PMNS_EXP['t12'])/PMNS_EXP['t12'])**2 +
                         ((r['t23']-PMNS_EXP['t23'])/PMNS_EXP['t23'])**2 +
                         ((r['t13']-PMNS_EXP['t13'])/PMNS_EXP['t13'])**2)
        if r['dm2_21'] > 0 and r['dm2_32'] > 0:
            cost_pmns += 20*((np.log10(r['dm2_21'])-np.log10(PMNS_EXP['dm2_21']))**2 +
                              (np.log10(r['dm2_32'])-np.log10(PMNS_EXP['dm2_32']))**2)
        else: cost_pmns = 200
        
        # CKM
        M_u = build_quark_matrix(m_up, d['ls_quark'], d['C0_q'], d['kappa'], -1, d['nk_boost'])
        M_d = build_quark_matrix(m_down, d['ls_quark'], d['C0_q'], d['kappa'], +1, d['nk_boost'])
        ckm = extract_ckm(M_u, M_d)
        cost_ckm = 30*(((ckm['t12']-CKM_EXP['t12'])/CKM_EXP['t12'])**2 +
                        ((ckm['t23']-CKM_EXP['t23'])/CKM_EXP['t23'])**2 +
                        ((ckm['t13']-CKM_EXP['t13'])/CKM_EXP['t13'])**2)
        if ckm['J'] > 0:
            cost_ckm += 10*(np.log10(ckm['J'])-np.log10(CKM_EXP['J']))**2
        else: cost_ckm += 100
        
        # Physicality penalty
        MR_offdiag = abs(MR[0,1]) + abs(MR[1,2]) + abs(MR[0,2])
        MR_diag_s = abs(MR[0,0]) + abs(MR[1,1]) + abs(MR[2,2])
        offdiag_ratio = MR_offdiag / max(MR_diag_s, 1e-30)
        if offdiag_ratio < 1e-10: penalty = 50.0
        elif offdiag_ratio < 1e-3:
            penalty = 50.0 * max(0, (-np.log10(max(offdiag_ratio,1e-30)) - 3) / 7.0)
        else: penalty = 0.0
        
        return cost_pmns + cost_ckm + penalty
    except: return 1e10


def evaluate_solution(params):
    """Full evaluation of a solution — returns dict of all predictions."""
    d = decode_params(params)
    mD = build_mD_dual(d['AD'], d['ls_lep'], d['C0_D'])
    MR = build_MR_dual(d['M_diag'], d['mu'], d['ls_Maj'])
    r = seesaw_full(mD, MR)
    M_u = build_quark_matrix(m_up, d['ls_quark'], d['C0_q'], d['kappa'], -1, d['nk_boost'])
    M_d = build_quark_matrix(m_down, d['ls_quark'], d['C0_q'], d['kappa'], +1, d['nk_boost'])
    ckm = extract_ckm(M_u, M_d)
    
    MR_offdiag = abs(MR[0,1]) + abs(MR[1,2]) + abs(MR[0,2])
    MR_diag_s = abs(MR[0,0]) + abs(MR[1,1]) + abs(MR[2,2])
    MR_eigs = np.sort(np.linalg.eigvalsh(MR))
    
    return {
        'params': params, 'cost': combined_cost(params),
        # PMNS observables
        'pmns_t12': np.degrees(r['t12']), 'pmns_t23': np.degrees(r['t23']),
        'pmns_t13': np.degrees(r['t13']),
        'dm2_21': r['dm2_21'], 'dm2_32': r['dm2_32'], 'dm2_ratio': r['ratio'],
        # Masses
        'm1': r['m'][0], 'm2': r['m'][1], 'm3': r['m'][2], 'sum_m': r['sum_m'],
        'ordering': 'NO' if r['dm2_32'] > 0 else 'IO',
        # CKM observables
        'ckm_t12': np.degrees(ckm['t12']), 'ckm_t23': np.degrees(ckm['t23']),
        'ckm_t13': np.degrees(ckm['t13']), 'J_ckm': ckm['J'],
        # Volumetric ansatz
        'p_vol': d['p_vol'], 'q_gen': d['q_gen'], 'l0': d['l0'], 'l0M': d['l0M'],
        'ratio_lq': d['ratio_lq'],
        'ls_lep_1': d['ls_lep'][0], 'ls_lep_2': d['ls_lep'][1], 'ls_lep_3': d['ls_lep'][2],
        'ls_q_1': d['ls_quark'][0], 'ls_q_2': d['ls_quark'][1], 'ls_q_3': d['ls_quark'][2],
        # Internal
        'mu': d['mu'], 'C0_D': d['C0_D'], 'C0_q': d['C0_q'],
        'kappa': d['kappa'], 'nk_boost': d['nk_boost'],
        'logM1': np.log10(d['M_diag'][0]), 'logM2': np.log10(d['M_diag'][1]),
        'logM3': np.log10(d['M_diag'][2]),
        'MR_eigs': MR_eigs, 'MR_offdiag_ratio': MR_offdiag/max(MR_diag_s,1e-30),
        'MR_hierarchy': MR_eigs[2]/max(MR_eigs[0],1e-10),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN: Collect diverse solutions
# ═══════════════════════════════════════════════════════════════

bounds = [
    (8, 16), (8, 16), (8, 16),   # logM1, logM2, logM3
    (0.5, 2.5),                   # log_mu
    (-4, 3),                      # logAD
    (-2, 4),                      # logC0D
    (-2.5, 0.5),                  # log_l0
    (0.1, 3.0),                   # p_vol
    (-1.5, 2.0),                  # q_gen
    (-1.5, 0.5),                  # log_l0M
    (-3, 2),                      # logC0q
    (-4, 0),                      # log_kappa
    (-0.05, 0.25),                # nk_boost
]

print("═" * 72)
print("  ANÁLISIS DE DEGENERESCENCIA — MODELO VOLUMÉTRICO (PMNS + CKM)")
print("═" * 72)

# ─── Stage 1: Diverse DE solutions ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  ETAPA 1: Recolección de Soluciones Diversas                 ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")
print(f"\n  30 seeds × DE(maxiter=600, pop=25) + L-BFGS-B polish")
print(f"  Criterio: cost < 1.0 (9 observables dentro de ~5%)")

solutions = []
for seed in range(30):
    res = differential_evolution(combined_cost, bounds, seed=seed*13+42,
                                  maxiter=600, tol=1e-14, popsize=25,
                                  mutation=(0.4, 1.9), recombination=0.85)
    # Polish
    pol = minimize(combined_cost, res.x, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 20000, 'ftol': 1e-15})
    best = pol if pol.fun < res.fun else res
    
    if best.fun < 1.0:
        sol = evaluate_solution(best.x)
        solutions.append(sol)
        tag = "✅" if best.fun < 0.1 else ("⊕" if best.fun < 0.5 else "")
        print(f"    Seed {seed:>2}: cost = {best.fun:.4f} {tag}", flush=True)
    else:
        print(f"    Seed {seed:>2}: cost = {best.fun:.4f} (rejected)", flush=True)

N = len(solutions)
print(f"\n  Soluciones aceptadas: {N}/30")
elapsed = time.time() - t_start
print(f"  Tiempo: {elapsed:.0f}s ({elapsed/60:.1f} min)")

if N < 5:
    print("  ⚠️  Pocas soluciones — resultados pueden no ser representativos.")


# ─── Stage 2: Statistics ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  ETAPA 2: Dispersión de Predicciones                         ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

def stats(key, label, unit="", log=False, exp_val=None):
    vals = [s[key] for s in solutions if key in s and np.isfinite(s[key]) and s[key] > 0]
    if len(vals) < 2: return None
    arr = np.array(vals)
    if log:
        la = np.log10(arr)
        med = 10**np.median(la)
        lo = 10**np.percentile(la, 16); hi = 10**np.percentile(la, 84)
        spread = hi/lo
        disp = (np.percentile(la,84)-np.percentile(la,16))/abs(np.median(la)) if abs(np.median(la))>1e-30 else np.inf
        s = f"  {label:>22}: {med:.3e} [{lo:.2e}, {hi:.2e}] {unit} (spread {spread:.1f}×)"
    else:
        med = np.median(arr); lo = np.percentile(arr, 16); hi = np.percentile(arr, 84)
        disp = (hi-lo)/abs(med) if abs(med) > 1e-30 else np.inf
        s = f"  {label:>22}: {med:.4f} [{lo:.4f}, {hi:.4f}] {unit} ({disp:.1%})"
    if exp_val is not None:
        s += f"  exp={exp_val}"
    print(s)
    return {'med': med, 'lo': lo, 'hi': hi, 'disp': disp, 'label': label}

print(f"\n  ─── OBSERVABLES FITTEADOS (deben ser estables) ───")
stats('pmns_t12', 'θ₁₂ PMNS', '°', exp_val='33.41°')
stats('pmns_t23', 'θ₂₃ PMNS', '°', exp_val='49.10°')
stats('pmns_t13', 'θ₁₃ PMNS', '°', exp_val='8.54°')
stats('dm2_21', 'Δm²₂₁', 'eV²', True, exp_val='7.41e-5')
stats('dm2_32', 'Δm²₃₂', 'eV²', True, exp_val='2.507e-3')
stats('dm2_ratio', 'Δm² ratio', '', exp_val='33.8')
stats('ckm_t12', 'θ₁₂ CKM', '°', exp_val='13.02°')
stats('ckm_t23', 'θ₂₃ CKM', '°', exp_val='2.40°')
stats('ckm_t13', 'θ₁₃ CKM', '°', exp_val='0.211°')
stats('J_ckm', 'J CKM', '', True, exp_val='3.08e-5')

print(f"\n  ─── PREDICCIONES CLAVE ───")
stats('m1', 'm₁', 'eV', True)
stats('m2', 'm₂', 'eV', True)
stats('m3', 'm₃', 'eV', True)
stats('sum_m', 'Σmᵢ', 'eV')

no_count = sum(1 for s in solutions if s['ordering'] == 'NO')
print(f"\n  Ordering: NO = {no_count}/{N} ({no_count/N:.0%})" if N > 0 else "")

print(f"\n  ─── ANSATZ VOLUMÉTRICO ───")
stats('p_vol', 'p (exp. volumen)')
stats('q_gen', 'q (exp. generación)')
stats('l0', 'l₀ (escala base)', '', True)
stats('l0M', 'l₀_Maj', '', True)
stats('ratio_lq', 'l_s(lep)/l_s(quark)')

print(f"\n  ─── l_s DERIVADOS ───")
stats('ls_lep_1', 'l_s(lep, gen 1)')
stats('ls_lep_2', 'l_s(lep, gen 2)')
stats('ls_lep_3', 'l_s(lep, gen 3)')
stats('ls_q_1', 'l_s(quark, gen 1)')
stats('ls_q_2', 'l_s(quark, gen 2)')
stats('ls_q_3', 'l_s(quark, gen 3)')

print(f"\n  ─── PARÁMETROS INTERNOS ───")
stats('mu', 'μ', '', True)
stats('logM1', 'log₁₀(M₁/GeV)')
stats('logM2', 'log₁₀(M₂/GeV)')
stats('logM3', 'log₁₀(M₃/GeV)')
stats('C0_D', 'C₀_D', '', True)
stats('C0_q', 'C₀_q', '', True)
stats('kappa', 'κ', '', True)
stats('nk_boost', 'NK boost')
stats('MR_offdiag_ratio', 'M_R off/diag', '', True)
stats('MR_hierarchy', 'M_R max/min', '×', True)


# ─── Stage 3: Robustness classification ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  ETAPA 3: Clasificación de Robustez                          ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

def robustness(key, log=False):
    vals = [s[key] for s in solutions if key in s and np.isfinite(s[key]) and s[key] > 0]
    if len(vals) < 3: return np.inf
    arr = np.log10(vals) if log else np.array(vals)
    med = np.median(arr)
    lo, hi = np.percentile(arr, 16), np.percentile(arr, 84)
    return (hi - lo) / abs(med) if abs(med) > 1e-30 else np.inf

predictions = [
    # (key, label, is_log)
    ('pmns_t12', 'θ₁₂ PMNS', False), ('pmns_t23', 'θ₂₃ PMNS', False),
    ('pmns_t13', 'θ₁₃ PMNS', False), ('dm2_21', 'Δm²₂₁', True),
    ('dm2_32', 'Δm²₃₂', True), ('dm2_ratio', 'Δm² ratio', False),
    ('ckm_t12', 'θ₁₂ CKM', False), ('ckm_t23', 'θ₂₃ CKM', False),
    ('ckm_t13', 'θ₁₃ CKM', False), ('J_ckm', 'J CKM', True),
    ('m1', 'm₁', True), ('m2', 'm₂', True), ('m3', 'm₃', True),
    ('sum_m', 'Σmᵢ', False),
    ('p_vol', 'p (volumen)', False), ('q_gen', 'q (generación)', False),
    ('ratio_lq', 'ratio l_s lep/quark', False),
    ('l0', 'l₀', True), ('l0M', 'l₀_Maj', True),
    ('mu', 'μ', True),
    ('logM1', 'log₁₀(M₁)', False), ('logM2', 'log₁₀(M₂)', False),
    ('logM3', 'log₁₀(M₃)', False),
    ('C0_D', 'C₀_D', True), ('C0_q', 'C₀_q', True),
    ('kappa', 'κ', True), ('nk_boost', 'NK boost', False),
    ('MR_hierarchy', 'M_R max/min', True),
    ('ls_lep_1', 'l_s(lep,1)', False), ('ls_q_1', 'l_s(q,1)', False),
]

rob_list = [(label, robustness(key, log), key) for key, label, log in predictions]
rob_list.sort(key=lambda x: x[1])

print(f"\n  {'Predicción':>24} {'Dispersión':>12} {'Clasificación':>18} {'Antes (PMNS)':>14}")
print(f"  {'─'*72}")

# Comparison with PMNS-only degeneracy (from previous analysis)
pmns_only_disp = {
    'θ₁₂ PMNS': 0.000, 'θ₂₃ PMNS': 0.000, 'θ₁₃ PMNS': 0.000,
    'Δm²₂₁': 0.000, 'Δm²₃₂': 0.000, 'Δm² ratio': 0.000,
    'm₃': 0.001, 'm₂': 0.012, 'Σmᵢ': 0.062, 'm₁': 1.108,
    'log₁₀(M₁)': 0.278, 'log₁₀(M₂)': 0.487, 'log₁₀(M₃)': 0.503,
    'μ': 0.526, 'M_R max/min': 1.209, 'C₀_D': 4.021,
}

for label, r, key in rob_list:
    if r < 0.05: cls = "✅ ROBUSTA"
    elif r < 0.20: cls = "⊕ SEMI-ROB."
    elif r < 0.50: cls = "⚠️  MODERADA"
    else: cls = "❌ FRÁGIL"
    
    before = pmns_only_disp.get(label, None)
    if before is not None:
        change = f"{before:.3f}"
        if r < before * 0.7: change += " ↓↓"  # significant improvement
        elif r > before * 1.3: change += " ↑"  # worsened
    else:
        change = "nuevo"
    
    print(f"  {label:>24} {r:>12.4f} {cls:>18} {change:>14}")


# ─── Stage 4: PCA ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  ETAPA 4: Estructura de la Variedad de Soluciones (PCA)      ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

if N >= 5:
    P = np.array([s['params'] for s in solutions])
    param_names = ['logM1','logM2','logM3','log_mu','logAD','logC0D',
                   'log_l0','p_vol','q_gen','log_l0M','logC0q','log_kappa','nk_boost']
    
    corr = np.corrcoef(P.T)
    print(f"\n  Correlaciones fuertes (|r| > 0.7):")
    found = False
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            if abs(corr[i,j]) > 0.7:
                sign = "+" if corr[i,j] > 0 else "−"
                print(f"    {param_names[i]:>10} ↔ {param_names[j]:<10}: r = {sign}{abs(corr[i,j]):.3f}")
                found = True
    if not found: print(f"    Ninguna")
    
    P_c = P - P.mean(axis=0)
    U, S_vals, Vt = np.linalg.svd(P_c, full_matrices=False)
    
    print(f"\n  Componentes principales:")
    total_var = np.sum(S_vals**2)
    cum = 0
    for k, sv in enumerate(S_vals):
        pct = sv**2/total_var*100; cum += pct
        stiff = "◀ rígido" if pct < 1 else ""
        print(f"    PC{k+1:>2}: σ={sv:>7.3f} ({pct:>5.1f}%, cum {cum:>5.1f}%) {stiff}")
    
    n_rigid = sum(1 for sv in S_vals if sv**2/total_var < 0.01)
    n_free = len(S_vals) - n_rigid
    print(f"\n  Dimensiones libres: ~{n_free} (de 13)")
    print(f"  Dimensiones rígidas: ~{n_rigid}")
    print(f"  Esperado con 9 observables: ~4 libres")


# ─── Stage 5: Experimental predictions ───
print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  ETAPA 5: Predicciones Experimentales                        ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

if N >= 3:
    m1_vals = sorted([s['m1'] for s in solutions])
    sum_vals = sorted([s['sum_m'] for s in solutions])
    p_vals = sorted([s['p_vol'] for s in solutions])
    ratio_vals = sorted([s['ratio_lq'] for s in solutions])
    
    print(f"""
  ─── Predicciones Robustas (si dispersión < 5%) ───
  
  Normal Ordering:    {no_count}/{N} ({no_count/N:.0%})
  δ_CP(PMNS) = 180°: Topológico (no depende del fit)
  δ_CP(CKM)  = 64.5°: Topológico (HK twist)
  m₃ ≈ 0.051 eV:     Fijado por Δm²₃₂
  m₂ ≈ 0.0086 eV:    Fijado por Δm²₂₁
  
  ─── Predicciones Nuevas (del modelo volumétrico) ───
  
  p (exp. volumen):   [{min(p_vals):.2f}, {max(p_vals):.2f}]  (¿≈3?)
  ratio lep/quark:    [{min(ratio_vals):.1f}, {max(ratio_vals):.1f}]
  m₁:                 [{min(m1_vals):.2e}, {max(m1_vals):.2e}] eV
  Σmᵢ:                [{min(sum_vals):.4f}, {max(sum_vals):.4f}] eV
  
  ─── Tests Experimentales ───
  
  JUNO (~2027):       Ordering {'NO ✅' if no_count == N else f'{no_count}/{N}'}
  CMB-S4/Euclid:      Σmᵢ = [{min(sum_vals):.4f}, {max(sum_vals):.4f}] eV
  T2HK/DUNE:          δ_CP(PMNS) = 180°, δ_CP(CKM) = 64.5°
  KATRIN/Project8:     m₁ < {max(m1_vals):.2e} eV
""")


# ─── Stage 6: Comparison table ───
print(f"╔═══════════════════════════════════════════════════════════════╗")
print(f"║  ETAPA 6: Comparación PMNS-solo vs Volumétrico               ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

comparison = [
    ("Modelo", "PMNS solo (9p:5o)", "Volumétrico (13p:9o)"),
    ("Observables fitteados", "5 (3θ + 2Δm²)", "9 (+3θ_CKM + J)"),
    ("Params/Obs ratio", "9:5 = 1.80", f"13:9 = 1.44"),
]

print(f"\n  {'':>25} {'PMNS solo':>18} {'Volumétrico':>18}")
print(f"  {'─'*63}")
for label, v1, v2 in comparison:
    print(f"  {label:>25} {v1:>18} {v2:>18}")

# Summary of robustness changes
rob_improved = sum(1 for label, r, key in rob_list 
                   if label in pmns_only_disp and r < pmns_only_disp[label] * 0.7)
rob_same = sum(1 for label, r, key in rob_list
               if label in pmns_only_disp and 0.7 <= r/max(pmns_only_disp[label],1e-10) <= 1.3)

print(f"\n  Robustez vs modelo anterior:")
print(f"    Mejoradas:    {rob_improved}")
print(f"    Sin cambio:   {rob_same}")
print(f"    Nuevas:       {len(rob_list) - rob_improved - rob_same}")


# ─── Final timing ───
elapsed = time.time() - t_start
print(f"\n{'═'*72}")
print(f"  Análisis completo. {N} soluciones, {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'═'*72}")
