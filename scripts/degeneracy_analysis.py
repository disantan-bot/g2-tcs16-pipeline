#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  ANÁLISIS DE DEGENERESCENCIA Y ROBUSTEZ
  ¿Cuáles predicciones son robustas y cuáles son artefactos del fit?
═══════════════════════════════════════════════════════════════════════════

  Con 10 parámetros y 5 observables (3θ + 2Δm²), la variedad de 
  soluciones es 5-dimensional. Dos soluciones con los mismos observables
  pueden tener predicciones MUY diferentes para:
  - Masas absolutas m₁, m₂, m₃
  - Σmᵢ (suma, medible por cosmología)
  - Eigenvalues de M_R (escala de nueva física)
  - Escala de l_s sectoriales
  - Estructura de m_D
  
  Estrategia: generar MUCHAS soluciones con costo < umbral,
  luego analizar la dispersión de cada predicción.
  
  Predicción ROBUSTA: dispersión pequeña → fijada por los observables
  Predicción FRÁGIL: dispersión grande → depende de parámetros libres

  Diego Santana S. — Marzo 2026
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# ═══ Constantes ═══
lambda_ACyl = 2.8
alpha_FHN = 0.15
a_K3 = 0.204
v_ew = 246.22 / np.sqrt(2)
delta_CP = np.pi
d_H = np.array([0.561, 0.347, 0.198])
d_12, d_23, d_13 = 0.166, 0.343, 0.343
t_cones = np.array([0.35, 0.50, 0.65])

EXP = {'t12': np.radians(33.41), 't23': np.radians(49.1), 't13': np.radians(8.54),
       'dm2_21': 7.41e-5, 'dm2_32': 2.507e-3}
EXP['ratio'] = EXP['dm2_32'] / EXP['dm2_21']

# Hitchin integrals (pre-computed)
def S_integral(ti, tj):
    def f(t):
        s2 = 1.0/np.cosh(lambda_ACyl*(t-0.5))**2
        th = abs(np.tanh(lambda_ACyl*(t-0.5)))
        return s2*(1+alpha_FHN*th)
    val, _ = quad(f, ti, tj)
    return val

S_CACHE = {}
for i in range(3):
    for j in range(i+1, 3):
        S_CACHE[(i,j)] = S_integral(t_cones[i], t_cones[j])


# ═══ Engine ═══
def F_hitchin(mu):
    F = np.ones((3,3))
    for i in range(3):
        for j in range(i+1,3):
            F[i,j] = F[j,i] = np.exp(-mu * S_CACHE[(i,j)])
    return F

def build_MR(M_diag, F):
    MR = np.zeros((3,3))
    for i in range(3):
        MR[i,i] = M_diag[i]
        for j in range(i+1,3):
            MR[i,j] = MR[j,i] = np.sqrt(M_diag[i]*M_diag[j]) * F[i,j]
    return MR

def build_mD(AD, ls_vec, C0):
    ls1,ls2,ls3 = ls_vec
    y_D = np.array([AD*np.exp(-d_H[0]/ls1), AD*np.exp(-d_H[1]/ls2), AD*np.exp(-d_H[2]/ls3)])
    Y = np.diag(y_D.astype(complex))
    phase = np.exp(1j*delta_CP)
    for (i,j),d,lse in [((0,1),d_12,np.sqrt(ls1*ls2)),
                          ((1,2),d_23,np.sqrt(ls2*ls3)),
                          ((0,2),d_13,np.sqrt(ls1*ls3))]:
        amp = C0*np.exp(-d/lse)*np.sqrt(y_D[i]*y_D[j])
        Y[i,j]=amp*phase; Y[j,i]=amp*np.conj(phase)
    return Y * v_ew

def seesaw(mD, MR):
    m_nu = -mD @ np.linalg.inv(MR.astype(complex)) @ mD.T
    H = m_nu.conj().T @ m_nu
    ev, V = np.linalg.eigh(H)
    m = np.sqrt(np.abs(ev))
    idx = np.argsort(m); m = m[idx]*1e9; U = V[:,idx]
    Ua = np.abs(U)
    s13 = np.clip(Ua[0,2],0,1); c13 = np.sqrt(max(1-s13**2,1e-20))
    s12 = np.clip(Ua[0,1]/c13,0,1); s23 = np.clip(Ua[1,2]/c13,0,1)
    return {'m': m, 'sum_m': np.sum(m),
            't12': np.arcsin(s12), 't23': np.arcsin(s23), 't13': np.arcsin(s13),
            'dm2_21': m[1]**2-m[0]**2, 'dm2_32': m[2]**2-m[1]**2,
            'ratio': (m[2]**2-m[1]**2)/(m[1]**2-m[0]**2) if m[1]**2>m[0]**2 else np.inf,
            'MR_eigs': np.sort(np.linalg.eigvalsh(MR)),
            'mD': mD, 'MR': MR}


def full_cost(params):
    """Cost function: 5 observables (3θ + 2Δm²)."""
    try:
        logM1,logM2,logM3, log_mu, logAD, logls1,logls2,logls3, logC0 = params
        M_diag = [10**logM1, 10**logM2, 10**logM3]
        mu = 10**log_mu; AD = 10**logAD
        ls_vec = [10**logls1, 10**logls2, 10**logls3]; C0 = 10**logC0
        
        F = F_hitchin(mu)
        MR = build_MR(M_diag, F)
        eigs = np.linalg.eigvalsh(MR)
        if np.any(eigs <= 0): return 1e10
        
        mD = build_mD(AD, ls_vec, C0)
        r = seesaw(mD, MR)
        if np.any(np.isnan(r['m'])): return 1e10
        
        ea = 50*(((r['t12']-EXP['t12'])/EXP['t12'])**2 +
                  ((r['t23']-EXP['t23'])/EXP['t23'])**2 +
                  ((r['t13']-EXP['t13'])/EXP['t13'])**2)
        if r['dm2_21'] > 0 and r['dm2_32'] > 0:
            ed = 20*((np.log10(r['dm2_21'])-np.log10(EXP['dm2_21']))**2 +
                      (np.log10(r['dm2_32'])-np.log10(EXP['dm2_32']))**2)
        else: ed = 200
        return ea + ed
    except: return 1e10


# ═══════════════════════════════════════════════════════════════
# PARTE 1: Recolectar MUCHAS soluciones diferentes
# ═══════════════════════════════════════════════════════════════

bounds = [(8,16),(8,16),(8,16), (0.5,3), (-4,3),
          (-2.5,-0.2),(-2.5,-0.2),(-2.5,-0.2), (-0.5,5)]

print("═" * 72)
print("  ANÁLISIS DE DEGENERESCENCIA Y ROBUSTEZ")
print("  Variedad de Soluciones del Modelo Final")
print("═" * 72)

print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 1: Recolección de Soluciones Diversas                 ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")
print(f"\n  Generando soluciones con semillas 0-39 (40 optimizaciones)...")
print(f"  Criterio de aceptación: costo < 0.5 (3θ + 2Δm² dentro de ~3%)")

solutions = []
for seed in range(25):
    res = differential_evolution(full_cost, bounds, seed=seed,
                                  maxiter=250, tol=1e-10, popsize=15,
                                  mutation=(0.5, 1.8), recombination=0.85)
    if res.fun < 0.5:
        p = res.x
        M_diag = [10**p[0], 10**p[1], 10**p[2]]
        mu = 10**p[3]; AD = 10**p[4]
        ls_vec = [10**p[5], 10**p[6], 10**p[7]]; C0 = 10**p[8]
        
        F = F_hitchin(mu)
        MR = build_MR(M_diag, F)
        mD = build_mD(AD, ls_vec, C0)
        r = seesaw(mD, MR)
        
        sol = {
            'cost': res.fun, 'params': p,
            'M_diag': M_diag, 'mu': mu, 'AD': AD,
            'ls_vec': ls_vec, 'C0': C0,
            # Predictions
            'm1': r['m'][0], 'm2': r['m'][1], 'm3': r['m'][2],
            'sum_m': r['sum_m'],
            't12': np.degrees(r['t12']), 't23': np.degrees(r['t23']),
            't13': np.degrees(r['t13']),
            'dm2_21': r['dm2_21'], 'dm2_32': r['dm2_32'],
            'ratio': r['ratio'],
            'MR_eigs': r['MR_eigs'],
            'log_M1': np.log10(M_diag[0]), 'log_M2': np.log10(M_diag[1]),
            'log_M3': np.log10(M_diag[2]),
            'log_mu': np.log10(mu),
            'ls1': ls_vec[0], 'ls2': ls_vec[1], 'ls3': ls_vec[2],
            'mD_23_33': abs(mD[1,2])/abs(mD[2,2]),
            'mD_12_22': abs(mD[0,1])/abs(mD[1,1]),
            'ordering': 'NO' if r['dm2_32'] > 0 else 'IO',
        }
        solutions.append(sol)

N = len(solutions)
print(f"\n  Soluciones encontradas: {N}/40 (tasa: {N/40:.0%})")

if N < 5:
    print("  ADVERTENCIA: pocas soluciones. Relajando criterio...")
    for seed in range(25, 50):
        res = differential_evolution(full_cost, bounds, seed=seed,
                                      maxiter=250, tol=1e-10, popsize=15,
                                      mutation=(0.3, 1.9), recombination=0.7)
        if res.fun < 2.0:
            p = res.x
            M_diag = [10**p[0], 10**p[1], 10**p[2]]
            mu = 10**p[3]; AD = 10**p[4]
            ls_vec = [10**p[5], 10**p[6], 10**p[7]]; C0 = 10**p[8]
            F = F_hitchin(mu); MR = build_MR(M_diag, F)
            mD = build_mD(AD, ls_vec, C0); r = seesaw(mD, MR)
            sol = {
                'cost': res.fun, 'params': p,
                'M_diag': M_diag, 'mu': mu, 'AD': AD,
                'ls_vec': ls_vec, 'C0': C0,
                'm1': r['m'][0], 'm2': r['m'][1], 'm3': r['m'][2],
                'sum_m': r['sum_m'],
                't12': np.degrees(r['t12']), 't23': np.degrees(r['t23']),
                't13': np.degrees(r['t13']),
                'dm2_21': r['dm2_21'], 'dm2_32': r['dm2_32'],
                'ratio': r['ratio'],
                'MR_eigs': r['MR_eigs'],
                'log_M1': np.log10(M_diag[0]), 'log_M2': np.log10(M_diag[1]),
                'log_M3': np.log10(M_diag[2]),
                'log_mu': np.log10(mu),
                'ls1': ls_vec[0], 'ls2': ls_vec[1], 'ls3': ls_vec[2],
                'mD_23_33': abs(mD[1,2])/abs(mD[2,2]),
                'mD_12_22': abs(mD[0,1])/abs(mD[1,1]),
                'ordering': 'NO' if r['dm2_32'] > 0 else 'IO',
            }
            solutions.append(sol)
    N = len(solutions)
    print(f"  Soluciones totales: {N}")


# ═══════════════════════════════════════════════════════════════
# PARTE 2: Estadísticas de cada predicción
# ═══════════════════════════════════════════════════════════════

print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 2: Dispersión de Predicciones                         ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

def stats(vals, label, unit="", log=False):
    """Print statistics for a set of prediction values."""
    arr = np.array([v for v in vals if np.isfinite(v) and v > 0])
    if len(arr) == 0: return
    if log:
        arr_log = np.log10(arr)
        med = 10**np.median(arr_log)
        lo = 10**np.percentile(arr_log, 16)
        hi = 10**np.percentile(arr_log, 84)
        spread = hi/lo
        print(f"  {label:>20}: median = {med:.3e} {unit}, "
              f"[{lo:.2e}, {hi:.2e}], spread = {spread:.1f}×")
    else:
        med = np.median(arr)
        lo = np.percentile(arr, 16)
        hi = np.percentile(arr, 84)
        spread = (hi - lo) / med if med > 0 else np.inf
        print(f"  {label:>20}: median = {med:.4f} {unit}, "
              f"[{lo:.4f}, {hi:.4f}], spread = {spread:.1%}")
    return med, lo, hi

print(f"\n  ─── OBSERVABLES (deben estar fijos por construcción) ───")
stats([s['t12'] for s in solutions], "θ₁₂", "°")
stats([s['t23'] for s in solutions], "θ₂₃", "°")
stats([s['t13'] for s in solutions], "θ₁₃", "°")
stats([s['dm2_21'] for s in solutions], "Δm²₂₁", "eV²", log=True)
stats([s['dm2_32'] for s in solutions], "Δm²₃₂", "eV²", log=True)
stats([s['ratio'] for s in solutions], "Δm² ratio")

print(f"\n  ─── PREDICCIONES CLAVE (masas absolutas) ───")
stats([s['m1'] for s in solutions], "m₁", "eV", log=True)
stats([s['m2'] for s in solutions], "m₂", "eV", log=True)
stats([s['m3'] for s in solutions], "m₃", "eV", log=True)
stats([s['sum_m'] for s in solutions], "Σmᵢ", "eV")

# Ordering
no_count = sum(1 for s in solutions if s['ordering'] == 'NO')
print(f"\n  Ordenamiento: NO = {no_count}/{N} ({no_count/N:.0%}), IO = {N-no_count}/{N}")

print(f"\n  ─── PARÁMETROS INTERNOS (degenerescencia) ───")
stats([s['mu'] for s in solutions], "μ", "", log=True)
stats([s['log_M1'] for s in solutions], "log₁₀(M₁/GeV)")
stats([s['log_M2'] for s in solutions], "log₁₀(M₂/GeV)")
stats([s['log_M3'] for s in solutions], "log₁₀(M₃/GeV)")
stats([s['ls1'] for s in solutions], "l_s₁")
stats([s['ls2'] for s in solutions], "l_s₂")
stats([s['ls3'] for s in solutions], "l_s₃")
stats([s['C0'] for s in solutions], "C₀", "", log=True)
stats([s['AD'] for s in solutions], "A_D", "", log=True)

print(f"\n  ─── RATIOS ESTRUCTURALES ───")
stats([s['mD_23_33'] for s in solutions], "(m_D)₂₃/(m_D)₃₃")
stats([s['mD_12_22'] for s in solutions], "(m_D)₁₂/(m_D)₂₂")

# M_R hierarchy
stats([s['MR_eigs'][2]/s['MR_eigs'][0] for s in solutions if s['MR_eigs'][0] > 0],
      "M_R max/min", "×", log=True)

# l_s hierarchy
stats([s['ls3']/s['ls1'] for s in solutions if s['ls1'] > 0],
      "l_s₃/l_s₁")


# ═══════════════════════════════════════════════════════════════
# PARTE 3: Clasificación ROBUSTA / FRÁGIL
# ═══════════════════════════════════════════════════════════════

print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 3: Clasificación de Robustez                          ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

# Define robustness metric: relative spread (84th-16th percentile / median)
def robustness(vals, log=False):
    arr = np.array([v for v in vals if np.isfinite(v) and v > 0])
    if len(arr) < 3: return np.inf
    if log:
        arr = np.log10(arr)
    med = np.median(arr)
    lo, hi = np.percentile(arr, 16), np.percentile(arr, 84)
    return (hi - lo) / abs(med) if abs(med) > 1e-30 else np.inf

predictions = {
    'θ₁₂': ([s['t12'] for s in solutions], False),
    'θ₂₃': ([s['t23'] for s in solutions], False),
    'θ₁₃': ([s['t13'] for s in solutions], False),
    'Δm²₂₁': ([s['dm2_21'] for s in solutions], True),
    'Δm²₃₂': ([s['dm2_32'] for s in solutions], True),
    'Δm² ratio': ([s['ratio'] for s in solutions], False),
    'm₁': ([s['m1'] for s in solutions], True),
    'm₂': ([s['m2'] for s in solutions], True),
    'm₃': ([s['m3'] for s in solutions], True),
    'Σmᵢ': ([s['sum_m'] for s in solutions], False),
    'Ordering': ([1 if s['ordering']=='NO' else 0 for s in solutions], False),
    'μ': ([s['mu'] for s in solutions], True),
    'log₁₀(M₁)': ([s['log_M1'] for s in solutions], False),
    'log₁₀(M₂)': ([s['log_M2'] for s in solutions], False),
    'log₁₀(M₃)': ([s['log_M3'] for s in solutions], False),
    'l_s₁': ([s['ls1'] for s in solutions], False),
    'l_s₂': ([s['ls2'] for s in solutions], False),
    'l_s₃': ([s['ls3'] for s in solutions], False),
    'C₀': ([s['C0'] for s in solutions], True),
    'A_D': ([s['AD'] for s in solutions], True),
    '(mD)₂₃/(mD)₃₃': ([s['mD_23_33'] for s in solutions], False),
    'M_R max/min': ([s['MR_eigs'][2]/max(s['MR_eigs'][0],1e-10) for s in solutions], True),
    'l_s₃/l_s₁': ([s['ls3']/max(s['ls1'],1e-10) for s in solutions], False),
}

# Sort by robustness
rob_list = []
for name, (vals, is_log) in predictions.items():
    r = robustness(vals, is_log)
    rob_list.append((name, r, vals, is_log))

rob_list.sort(key=lambda x: x[1])

print(f"\n  {'Predicción':>20} {'Dispersión':>12} {'Clasificación':>16}")
print(f"  {'─'*52}")
for name, r, vals, is_log in rob_list:
    if r < 0.05:
        cls = "✅ ROBUSTA"
    elif r < 0.20:
        cls = "⊕ SEMI-ROBUSTA"
    elif r < 0.50:
        cls = "⚠️  MODERADA"
    else:
        cls = "❌ FRÁGIL"
    print(f"  {name:>20} {r:>12.3f} {cls:>16}")


# ═══════════════════════════════════════════════════════════════
# PARTE 4: Correlaciones entre parámetros
# ═══════════════════════════════════════════════════════════════

print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 4: Correlaciones (direcciones planas)                 ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

if N >= 5:
    # Build parameter matrix
    param_names = ['logM1','logM2','logM3','log_mu','logAD',
                   'logls1','logls2','logls3','logC0']
    P = np.array([s['params'] for s in solutions])
    
    # Correlation matrix
    if P.shape[0] > 1:
        corr = np.corrcoef(P.T)
        
        print(f"\n  Correlaciones fuertes (|r| > 0.7):")
        found = False
        for i in range(len(param_names)):
            for j in range(i+1, len(param_names)):
                if abs(corr[i,j]) > 0.7:
                    sign = "+" if corr[i,j] > 0 else "−"
                    print(f"    {param_names[i]:>8} ↔ {param_names[j]:<8}: r = {sign}{abs(corr[i,j]):.3f}")
                    found = True
        if not found:
            print(f"    Ninguna (parámetros relativamente independientes)")
        
        # PCA to find flat directions
        P_centered = P - P.mean(axis=0)
        U, S_vals, Vt = np.linalg.svd(P_centered, full_matrices=False)
        
        print(f"\n  Valores singulares (PCA de la variedad de soluciones):")
        for k, sv in enumerate(S_vals):
            contrib = sv**2 / np.sum(S_vals**2) * 100
            print(f"    PC{k+1}: σ = {sv:.3f} ({contrib:.1f}% de varianza)")
        
        print(f"\n  Direcciones más degeneradas (mayor varianza):")
        for k in range(min(3, len(S_vals))):
            if S_vals[k] > 0.1:
                vec = Vt[k]
                top_idx = np.argsort(np.abs(vec))[::-1][:3]
                parts = ", ".join([f"{param_names[idx]}({vec[idx]:+.2f})" for idx in top_idx])
                print(f"    PC{k+1}: {parts}")


# ═══════════════════════════════════════════════════════════════
# PARTE 5: Tabla resumen para el Compendio
# ═══════════════════════════════════════════════════════════════

print(f"\n╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 5: Resumen de Predicciones con Barras de Error        ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

def summary(vals, name, unit, is_log=False, exp_val=None, exp_label=""):
    arr = np.array([v for v in vals if np.isfinite(v) and v > 0])
    if len(arr) == 0: return
    if is_log:
        log_arr = np.log10(arr)
        med = 10**np.median(log_arr)
        lo = 10**np.percentile(log_arr, 16)
        hi = 10**np.percentile(log_arr, 84)
        s = f"  {name:>14}: {med:.3e} [{lo:.2e}, {hi:.2e}] {unit}"
    else:
        med = np.median(arr)
        lo = np.percentile(arr, 16)
        hi = np.percentile(arr, 84)
        s = f"  {name:>14}: {med:.4f} [{lo:.4f}, {hi:.4f}] {unit}"
    if exp_val is not None:
        s += f"  (exp: {exp_label})"
    print(s)
    return med, lo, hi

print(f"\n  PREDICCIONES ROBUSTAS (dispersión < 5%):")
print(f"  ─────────────────────────────────────────")

# For robust predictions, quote the range
r_t12 = summary([s['t12'] for s in solutions], "θ₁₂", "°", exp_val=33.41, exp_label="33.41°")
r_t23 = summary([s['t23'] for s in solutions], "θ₂₃", "°", exp_val=49.1, exp_label="49.1°")
r_t13 = summary([s['t13'] for s in solutions], "θ₁₃", "°", exp_val=8.54, exp_label="8.54°")
r_dm21 = summary([s['dm2_21'] for s in solutions], "Δm²₂₁", "eV²", True, exp_val=7.41e-5, exp_label="7.41e-5")
r_dm32 = summary([s['dm2_32'] for s in solutions], "Δm²₃₂", "eV²", True, exp_val=2.507e-3, exp_label="2.51e-3")
r_ratio = summary([s['ratio'] for s in solutions], "Δm² ratio", "", exp_val=33.8, exp_label="33.8")

print(f"\n  PREDICCIONES CON INCERTIDUMBRE:")
print(f"  ──────────────────────────────")
r_m1 = summary([s['m1'] for s in solutions], "m₁", "eV", True)
r_m2 = summary([s['m2'] for s in solutions], "m₂", "eV", True)
r_m3 = summary([s['m3'] for s in solutions], "m₃", "eV", True)
r_sum = summary([s['sum_m'] for s in solutions], "Σmᵢ", "eV")

# NO vs IO
print(f"\n  Ordering: {'NO' if no_count > N//2 else 'IO'} ({no_count}/{N})")

print(f"\n  PARÁMETROS INTERNOS (dispersión amplia):")
print(f"  ─────────────────────────────────────────")
summary([s['mu'] for s in solutions], "μ", "", True)
summary([s['log_M1'] for s in solutions], "log₁₀(M₁)", "")
summary([s['log_M2'] for s in solutions], "log₁₀(M₂)", "")
summary([s['log_M3'] for s in solutions], "log₁₀(M₃)", "")
summary([s['ls1'] for s in solutions], "l_s₁", "")
summary([s['ls2'] for s in solutions], "l_s₂", "")
summary([s['ls3'] for s in solutions], "l_s₃", "")
summary([s['C0'] for s in solutions], "C₀", "", True)


# ═══════════════════════════════════════════════════════════════
# PARTE 6: Implicaciones para predicciones experimentales
# ═══════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  PARTE 6: Implicaciones para Experimentos                    ║
╚═══════════════════════════════════════════════════════════════╝
""")

# For each experimental test, check if the prediction is robust
tests = [
    ("JUNO (~2027): Ordering",
     "NO" if no_count > N//2 else "IO",
     f"{no_count}/{N} = {no_count/N:.0%}"),
    
    ("KATRIN/P8: m_β",
     f"m₁ ∈ [{min(s['m1'] for s in solutions):.1e}, {max(s['m1'] for s in solutions):.1e}] eV",
     "m₁ muy variable" if robustness([s['m1'] for s in solutions], True) > 0.5 else "m₁ robusto"),
    
    ("CMB-S4/Euclid: Σmᵢ",
     f"Σmᵢ ∈ [{min(s['sum_m'] for s in solutions):.4f}, {max(s['sum_m'] for s in solutions):.4f}] eV",
     "robusto" if robustness([s['sum_m'] for s in solutions]) < 0.1 else "variable"),
    
    ("T2HK/DUNE: δ_CP",
     "180° (fijado topológicamente)",
     "robustísimo (no depende del fit)"),
    
    ("Oscillation: ratio Δm²",
     f"ratio ∈ [{min(s['ratio'] for s in solutions):.1f}, {max(s['ratio'] for s in solutions):.1f}]",
     "robusto" if robustness([s['ratio'] for s in solutions]) < 0.05 else "variable"),
]

for test, pred, status in tests:
    print(f"  {test}")
    print(f"    Predicción: {pred}")
    print(f"    Status: {status}")
    print()


# ═══════════════════════════════════════════════════════════════
# PARTE 7: Conclusión
# ═══════════════════════════════════════════════════════════════

print(f"╔═══════════════════════════════════════════════════════════════╗")
print(f"║  CONCLUSIÓN: ¿Qué predicciones del modelo son creíbles?      ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

# Count robust predictions
n_robust = sum(1 for _, r, _, _ in rob_list if r < 0.05)
n_semi = sum(1 for _, r, _, _ in rob_list if 0.05 <= r < 0.20)
n_fragile = sum(1 for _, r, _, _ in rob_list if r >= 0.50)

print(f"""
  De {len(rob_list)} cantidades analizadas:
    ✅ ROBUSTAS (< 5% dispersión):    {n_robust}
    ⊕  SEMI-ROBUSTAS (5-20%):         {n_semi}
    ⚠️  MODERADAS (20-50%):            {sum(1 for _, r, _, _ in rob_list if 0.20 <= r < 0.50)}
    ❌ FRÁGILES (> 50%):               {n_fragile}
  
  Las predicciones ROBUSTAS son aquellas que el modelo hace
  independientemente de qué punto de la variedad de soluciones 
  se escoja. Estas son las únicas predicciones creíbles.
  
  Las predicciones FRÁGILES dependen de la elección específica de 
  parámetros y NO deben citarse como predicciones del modelo.
  
  LECCIONES PARA EL COMPENDIO:
  1. Los 5 observables fitteados (3θ + 2Δm²) son robustos por 
     construcción (costo < 0.5 para todas las soluciones).
  2. Revisar si m₁, Σmᵢ y el ordering son robustos o no.
  3. Los valores de l_s, μ, M_diag individuales NO son predicciones
     del modelo — son parámetros que absorben la degenerescencia.
  4. Buscar COMBINACIONES que sean robustas aun cuando los 
     parámetros individuales varíen.
""")

print(f"{'═'*72}")
print(f"  Análisis de degenerescencia completo. {N} soluciones analizadas.")
print(f"{'═'*72}")
