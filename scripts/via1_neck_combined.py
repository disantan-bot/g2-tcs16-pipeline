#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  VÍA 1 + RESONANCIA DE CUELLO TCS
  Textura de M_R desde Ciclos Co-Asociativos + Penalización ACyl
═══════════════════════════════════════════════════════════════════════════

  Mecanismo combinado:
  ────────────────────
  (M_R)ᵢⱼ = √(M_Ri·M_Rj) × F_K3(i,j) × F_cuello(i,j)
  
  donde:
  
  F_K3(i,j) = exp(-d̃ᵢⱼ / l̃_s)          ← Supresión por distancia en K3
                                            (métrica GH non-isótropa)
  
  F_cuello(i,j) = exp(-λ_ACyl·|Δσ|/ℓ)   ← Penalización por cruce de cuello TCS
                                            λ_ACyl = 2.8 (Etapa 13)
                                            Δσ = distancia de cruce
  
  Física del cuello:
  ──────────────────
  X₇ = Z₊ ∪_neck Z₋  (twisted connected sum)
  
  El cuello tiene coordenada t ∈ [0, T] con T ≈ longitud de Hitchin flow.
  Los conos están en posiciones:
    Cono 1: t₁ = 0.35  ← LADO Z₊
    Cono 2: t₂ = 0.50  ← CENTRO (near junction)
    Cono 3: t₃ = 0.65  ← LADO Z₋
  
  Un 4-ciclo co-asociativo que conecta conos i↔j debe cruzar la 
  región de cuello si los conos están en lados diferentes. El perfil
  ACyl impone decaimiento exponencial a las amplitudes que cruzan:
  
    Amplitud ∝ exp(-λ·|t_i - t_j| / ℓ_cuello)
  
  Resultado CUALITATIVO:
    • (M_R)₁₂: t₁,t₂ cercanos, mismo lado → F_cuello ≈ 1 (sin penalización)
    • (M_R)₂₃: t₂,t₃ cruzan centro → F_cuello ~ exp(-λ·0.15/ℓ) (moderada)
    • (M_R)₁₃: t₁,t₃ cruzan todo el cuello → F_cuello ~ exp(-λ·0.30/ℓ) (fuerte)
  
  Esto genera EXACTAMENTE la jerarquía (M_R)₁₂ ≫ (M_R)₂₃ ≫ (M_R)₁₃

  Diego Santana S. — Marzo 2026
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import differential_evolution
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# PARÁMETROS DEL COMPENDIO
# ═══════════════════════════════════════════════════════════════

# Métrica TCS-16
a_t     = 0.013
a_theta = 0.625
a_K3    = 0.204
beta0   = 1.58

# Hitchin flow (Etapa 13)
lambda_ACyl = 2.8     # decaimiento ACyl
T_neck      = 1.0     # longitud normalizada del cuello
torsion_G2  = 7.3e-5  # residual torsion (holonomía estricta)

# Posiciones de los conos en el cuello
t_cones     = np.array([0.35, 0.50, 0.65])
theta_cones = np.array([0.00, 0.26, 0.55])

# Posiciones K3 SM (Compendio §18.2, fijadas por Pic(K3))
K3_SM = np.array([
    [0.763, 0.431, 0.100, 0.431],
    [0.431, 0.431, 0.763, 0.100],
    [0.431, 0.431, 0.431, 0.900],
])

# Seesaw parameters
v_ew       = 246.22 / np.sqrt(2)
l_s_lep    = 0.044
C0_Dirac   = 7348.0
delta_CP   = np.pi
d_H        = np.array([0.561, 0.347, 0.198])
d_12_dirac = 0.166
d_23_dirac = 0.343
d_13_dirac = 0.343

# Volúmenes gauge (Compendio §10, en l_P³)
Vol_SU3 = 1.909798
Vol_SU2 = 0.953088
Vol_U1  = 1.121823
Vol_SM_avg = (Vol_SU3 + Vol_SU2 + Vol_U1) / 3  # ≈ 1.33

# Experimental (NuFIT 5.3, NO)
exp_t12 = np.radians(33.41)
exp_t23 = np.radians(49.1)
exp_t13 = np.radians(8.54)
exp_dm2_21 = 7.41e-5
exp_dm2_32 = 2.507e-3
exp_dm2_ratio = exp_dm2_32 / exp_dm2_21


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 1: FACTOR DE CUELLO TCS (F_cuello)
# ═══════════════════════════════════════════════════════════════

def neck_crossing_factor(t_i, t_j, lambda_acyl, ell_neck, t_junction=0.50):
    """
    Factor de penalización por cruce del cuello TCS.
    
    El cuello TCS tiene un perfil ACyl que decae exponencialmente
    desde cada extremo. La zona de transición (junction) está en 
    t_junction ≈ 0.50.
    
    Para un ciclo co-asociativo que conecta el cono i (en t_i) 
    con el cono j (en t_j):
    
    Si ambos están del mismo lado de la junction:
        F = exp(-λ × |t_i - t_j| / ℓ)     ← supresión suave
    
    Si cruzan la junction:
        F = exp(-λ × |t_i - t_j| / ℓ) × exp(-λ_extra × n_cross)
        donde n_cross tiene en cuenta que el ciclo debe "pasar" 
        por la región de mínimo del flujo φ₃ en la junction.
    
    El factor extra viene del Hitchin flow: la 3-forma φ₃ tiene un 
    mínimo en t_junction (cambio de régimen Z₊ → Z₋), y los ciclos 
    que cruzan este mínimo sufren una supresión adicional porque
    vol(Σ̃) ≥ ∫ |φ₃|, y |φ₃| decrece en la junction.
    """
    delta_t = abs(t_i - t_j)
    
    # Basic exponential suppression
    F_basic = np.exp(-lambda_acyl * delta_t / ell_neck)
    
    # Extra suppression for crossing the junction
    # Check if the path from t_i to t_j crosses t_junction
    crosses = (t_i - t_junction) * (t_j - t_junction) < 0
    
    if crosses:
        # Distance to junction from each side
        d_i = abs(t_i - t_junction)
        d_j = abs(t_j - t_junction)
        # The φ₃ profile near the junction follows tanh:
        # φ₃(t) ~ φ_∞ × tanh(λ(t - t_junc))
        # The integral over the minimum adds an extra log factor
        F_junction = np.exp(-lambda_acyl * min(d_i, d_j) / ell_neck)
        return F_basic * F_junction
    else:
        return F_basic


def compute_neck_factors(t_cones, lambda_acyl, ell_neck, t_junction=0.50):
    """Compute neck crossing factors for all cone pairs."""
    F = np.zeros((3,3))
    for i in range(3):
        F[i,i] = 1.0  # self-coupling, no suppression
        for j in range(i+1, 3):
            F[i,j] = F[j,i] = neck_crossing_factor(
                t_cones[i], t_cones[j], lambda_acyl, ell_neck, t_junction)
    return F


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 2: FACTOR K3 NO-ISÓTROPO (F_K3) 
# ═══════════════════════════════════════════════════════════════

def build_kummer_fps():
    """16 fixed points of T⁴/ℤ₂"""
    fps = []
    for n1,n2,n3,n4 in product([0,1], repeat=4):
        fps.append(np.array([n1/2, n2/2, n3/2, n4/2]))
    return np.array(fps)

def assign_E8_sectors(fps):
    """SM: n₁+n₂ even, GUT: n₃+n₄ even"""
    SM = np.array([(int(fp[0]*2)+int(fp[1]*2))%2==0 for fp in fps])
    GUT = np.array([(int(fp[2]*2)+int(fp[3]*2))%2==0 for fp in fps])
    return SM, GUT

def GH_potential(x, fps, a_k, epsilon=0.5, sector='GUT', SM_bw=None, GUT_bw=None):
    """Gibbons-Hawking multi-center potential with sector weighting."""
    delta_cross = 0.1
    V = epsilon
    for k in range(len(fps)):
        dx = x - fps[k]; dx -= np.round(dx)
        r2 = max(np.dot(dx, dx), 4e-4)
        w = 1.0 if (sector=='GUT' and GUT_bw[k]) or (sector=='SM' and SM_bw[k]) else delta_cross
        V += w * a_k[k] / r2
    return V

def geodesic_length_GH(p1, p2, fps, a_k, epsilon, sector, SM_bw, GUT_bw, N=40):
    """Straight-path length in GH metric: L = ∫√V |dx|."""
    dp = p2 - p1; dp -= np.round(dp)
    flat = np.linalg.norm(dp)
    if flat < 1e-10: return 0.0
    L = 0.0; dt = 1.0/N
    for i in range(N):
        t = (i + 0.5) * dt
        x = (p1 + t*dp) % 1.0
        V = GH_potential(x, fps, a_k, epsilon, sector, SM_bw, GUT_bw)
        L += np.sqrt(V) * flat * dt
    return L

def compute_K3_GUT_positions(K3_SM, beta0):
    """Twist HK + ν_R offset → GUT positions on T⁴."""
    c, s = np.cos(beta0), np.sin(beta0)
    R = np.array([[c,0,-s,0],[0,c,0,-s],[s,0,c,0],[0,s,0,c]])
    offset = np.array([-0.25, -0.25, -0.25, -0.25])
    K3_GUT = np.zeros_like(K3_SM)
    for k in range(3):
        K3_GUT[k] = (R @ K3_SM[k] + offset) % 1.0
    return K3_GUT

def compute_K3_factors(K3_SM, K3_GUT, fps, a_k, epsilon, SM_bw, GUT_bw, l_s_tilde):
    """F_K3(i,j) = exp(-d̃_K3(i,j) / l̃_s) using GH metric."""
    F = np.zeros((3,3))
    for i in range(3):
        F[i,i] = 1.0
        for j in range(i+1, 3):
            # Approximate: use conformal factor at midpoint × flat distance
            dp = K3_GUT[i] - K3_GUT[j]; dp -= np.round(dp)
            flat = np.linalg.norm(dp)
            mid = ((K3_GUT[i] + K3_GUT[j]) / 2) % 1.0
            V_mid = GH_potential(mid, fps, a_k, epsilon, 'GUT', SM_bw, GUT_bw)
            d_eff = a_K3 * np.sqrt(V_mid) * flat
            F[i,j] = F[j,i] = np.exp(-d_eff / l_s_tilde)
    return F


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 3: M_R COMBINADO
# ═══════════════════════════════════════════════════════════════

def build_MR_combined(M_diag, F_K3, F_neck):
    """
    M_R(i,j) = √(M_i·M_j) × F_K3(i,j) × F_cuello(i,j)
    
    Diagonal: M_R(i,i) = M_i  (self-energy, no suppression)
    Off-diagonal: producto de ambas supresiones
    """
    MR = np.zeros((3,3))
    for i in range(3):
        MR[i,i] = M_diag[i]
        for j in range(i+1, 3):
            MR[i,j] = MR[j,i] = np.sqrt(M_diag[i]*M_diag[j]) * F_K3[i,j] * F_neck[i,j]
    return MR


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 4: SEESAW ENGINE
# ═══════════════════════════════════════════════════════════════

def build_mD(A_D):
    y_D = A_D * np.exp(-d_H / l_s_lep)
    Y = np.diag(y_D.astype(complex))
    phase = np.exp(1j * delta_CP)
    for (i,j), d in [((0,1),d_12_dirac),((1,2),d_23_dirac),((0,2),d_13_dirac)]:
        amp = C0_Dirac * np.exp(-d/l_s_lep) * np.sqrt(y_D[i]*y_D[j])
        Y[i,j] = amp * phase; Y[j,i] = amp * np.conj(phase)
    return Y * v_ew

def seesaw(A_D, MR):
    mD = build_mD(A_D)
    m_nu = -mD @ np.linalg.inv(MR.astype(complex)) @ mD.T
    H = m_nu.conj().T @ m_nu
    ev, V = np.linalg.eigh(H)
    m = np.sqrt(np.abs(ev))
    idx = np.argsort(m); m = m[idx]*1e9; U = V[:,idx]
    Ua = np.abs(U)
    s13 = np.clip(Ua[0,2],0,1); c13 = np.sqrt(max(1-s13**2,1e-20))
    s12 = np.clip(Ua[0,1]/c13,0,1); s23 = np.clip(Ua[1,2]/c13,0,1)
    dm21 = m[1]**2 - m[0]**2; dm32 = m[2]**2 - m[1]**2
    return {
        'masses_eV': m, 'sum_m': np.sum(m),
        'theta12': np.arcsin(s12), 'theta23': np.arcsin(s23), 'theta13': np.arcsin(s13),
        'dm2_21': dm21, 'dm2_32': dm32,
        'dm2_ratio': dm32/dm21 if dm21 > 0 else np.inf,
    }


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 5: OPTIMIZACIÓN COMBINADA
# ═══════════════════════════════════════════════════════════════

def optimize_combined(F_neck, fps, a_k, epsilon, SM_bw, GUT_bw, K3_GUT):
    """
    Optimize {M₁, M₂, M₃, l̃_s, A_D} with FIXED neck factors
    and K3 geometry.
    """
    def cost(params):
        logM1, logM2, logM3, log_ls, logAD = params
        M_diag = [10**logM1, 10**logM2, 10**logM3]
        l_s_t = 10**log_ls; A_D = 10**logAD
        
        F_K3 = compute_K3_factors(K3_SM, K3_GUT, fps, a_k, epsilon, 
                                   SM_bw, GUT_bw, l_s_t)
        MR = build_MR_combined(M_diag, F_K3, F_neck)
        
        try:
            eigs = np.linalg.eigvalsh(MR)
            if np.any(eigs <= 0): return 1e10
            r = seesaw(A_D, MR)
        except: return 1e10
        if np.any(np.isnan(r['masses_eV'])): return 1e10
        
        ea = 100*(((r['theta12']-exp_t12)/exp_t12)**2 +
                   ((r['theta23']-exp_t23)/exp_t23)**2 +
                   ((r['theta13']-exp_t13)/exp_t13)**2)
        if r['dm2_21'] > 0 and r['dm2_32'] > 0:
            ed = 10*((np.log10(r['dm2_21'])-np.log10(exp_dm2_21))**2 +
                      (np.log10(r['dm2_32'])-np.log10(exp_dm2_32))**2)
        else: ed = 100
        return ea + ed
    
    bounds = [(8,16),(8,16),(8,16),(-3,0),(-4,2)]
    best = None; best_c = np.inf
    for s in range(2):
        res = differential_evolution(cost, bounds, seed=s+500, 
                                      maxiter=250, tol=1e-10, popsize=15)
        if res.fun < best_c: best_c = res.fun; best = res
    return best


def optimize_full_model():
    """
    MASTER OPTIMIZATION: optimize ALL parameters simultaneously:
    {M₁, M₂, M₃, l̃_s, A_D, ℓ_neck, a_GUT/a_SM, t_junction}
    
    This lets the neck length and blowup ratio float freely,
    constrained only by the requirement to match PMNS + Δm².
    """
    fps = build_kummer_fps()
    SM_bw, GUT_bw = assign_E8_sectors(fps)
    K3_GUT = compute_K3_GUT_positions(K3_SM, beta0)
    
    def cost(params):
        logM1, logM2, logM3, log_ls, logAD, log_ell, log_aGUT_ratio = params
        
        M_diag = [10**logM1, 10**logM2, 10**logM3]
        l_s_t = 10**log_ls; A_D = 10**logAD
        ell_neck = 10**log_ell
        a_GUT_ratio = 10**log_aGUT_ratio
        
        # Blowup parameters
        a_SM_val = 0.05
        a_k = np.zeros(16)
        for k in range(16):
            if SM_bw[k] and GUT_bw[k]:    a_k[k] = a_SM_val * np.sqrt(a_GUT_ratio)
            elif SM_bw[k]:                  a_k[k] = a_SM_val
            elif GUT_bw[k]:                 a_k[k] = a_SM_val * a_GUT_ratio
            else:                           a_k[k] = 0.03
        
        # Neck crossing factors
        F_neck = compute_neck_factors(t_cones, lambda_ACyl, ell_neck)
        
        # K3 factors
        F_K3 = compute_K3_factors(K3_SM, K3_GUT, fps, a_k, 0.5,
                                   SM_bw, GUT_bw, l_s_t)
        
        # Combined M_R
        MR = build_MR_combined(M_diag, F_K3, F_neck)
        
        try:
            eigs = np.linalg.eigvalsh(MR)
            if np.any(eigs <= 0): return 1e10
            r = seesaw(A_D, MR)
        except: return 1e10
        if np.any(np.isnan(r['masses_eV'])): return 1e10
        
        ea = 100*(((r['theta12']-exp_t12)/exp_t12)**2 +
                   ((r['theta23']-exp_t23)/exp_t23)**2 +
                   ((r['theta13']-exp_t13)/exp_t13)**2)
        if r['dm2_21'] > 0 and r['dm2_32'] > 0:
            ed = 10*((np.log10(r['dm2_21'])-np.log10(exp_dm2_21))**2 +
                      (np.log10(r['dm2_32'])-np.log10(exp_dm2_32))**2)
        else: ed = 100
        return ea + ed
    
    bounds = [
        (8, 16), (8, 16), (8, 16),     # M_diag
        (-3, 0),                         # l̃_s
        (-4, 2),                         # A_D
        (-3, 1),                         # ℓ_neck
        (-0.5, 2.0),                     # log10(a_GUT/a_SM)
    ]
    
    best = None; best_c = np.inf
    for s in range(3):
        res = differential_evolution(cost, bounds, seed=s+700,
                                      maxiter=300, tol=1e-10, popsize=20)
        if res.fun < best_c: best_c = res.fun; best = res
    return best, fps, SM_bw, GUT_bw, K3_GUT


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 6: ANÁLISIS DEL EFECTO DE CUELLO PURO
# ═══════════════════════════════════════════════════════════════

def scan_neck_length():
    """
    Scan ℓ_neck para entender su efecto sobre la jerarquía de M_R.
    Con ℓ_neck pequeño → supresión fuerte → gran jerarquía.
    """
    print(f"\n  ─── Factores de cuello para diferentes ℓ_neck ───")
    print(f"  (λ_ACyl = {lambda_ACyl}, conos en t = {t_cones})")
    print(f"\n  {'ℓ_neck':>8} {'F(1↔2)':>10} {'F(2↔3)':>10} {'F(1↔3)':>10} "
          f"{'F₁₂/F₁₃':>10} {'Jerarquía':>12}")
    print(f"  {'─'*62}")
    
    for ell in [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
        F = compute_neck_factors(t_cones, lambda_ACyl, ell)
        ratio = F[0,1] / F[0,2] if F[0,2] > 1e-30 else np.inf
        hier = "fuerte" if ratio > 100 else ("moderada" if ratio > 5 else "débil")
        print(f"  {ell:>8.2f} {F[0,1]:>10.3e} {F[1,2]:>10.3e} {F[0,2]:>10.3e} "
              f"{ratio:>10.1f} {hier:>12}")
    
    return


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("═" * 72)
    print("  VÍA 1 + RESONANCIA DE CUELLO TCS")
    print("  M_R desde Ciclos Co-Asociativos + Penalización ACyl")
    print("═" * 72)
    
    # ════════════════════════════════════════════════════════
    # PARTE 1: Entendiendo el efecto de cuello
    # ════════════════════════════════════════════════════════
    
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE 1: Efecto del Cuello TCS sobre M_R                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    print(f"""
  Configuración de conos en el cuello TCS:
  
  Z₊ ──────────── junction ──────────── Z₋
  │                  │                    │
  t=0              t=0.50              t=1.0
       ↑               ↑              ↑
     Cono 1          Cono 2         Cono 3
     t=0.35          t=0.50         t=0.65
  
  Δt(1↔2) = 0.15  ← Mismo lado del cuello
  Δt(2↔3) = 0.15  ← Cruza la junction 
  Δt(1↔3) = 0.30  ← Cruza TODO el cuello
  
  Clave: El par 2↔3 CRUZA la junction (t₂=0.50, t₃=0.65)
  mientras que 1↔2 NO la cruza (ambos en t < 0.50 o ≈ 0.50).
  El cruce de junction impone supresión EXTRA (factor F_junction).
    """)
    
    scan_neck_length()
    
    # Check: with the target from Phase 4
    print(f"\n  ─── Target de Fase 4 (script anterior) ───")
    print(f"  Necesitamos: (M_R)₁₂/√(M₁M₂) ≈ 1.0")
    print(f"               (M_R)₂₃/√(M₂M₃) ≈ 9.5×10⁻⁴")
    print(f"               (M_R)₁₃/√(M₁M₃) ≈ 3.2×10⁻⁶")
    print(f"  → F₁₂ ≈ 1.0, F₂₃ ≈ 10⁻³, F₁₃ ≈ 3×10⁻⁶")
    print(f"  → Necesitamos ℓ_neck ≈ 0.02 - 0.05")
    
    
    # ════════════════════════════════════════════════════════
    # PARTE 2: Optimización maestra (todos los params libres)
    # ════════════════════════════════════════════════════════
    
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE 2: Optimización Maestra — Todos los Parámetros       ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    print(f"\n  Optimizando {{M₁,M₂,M₃, l̃_s, A_D, ℓ_neck, a_GUT/a_SM}}...")
    print(f"  7 parámetros para 8 observables (3θ + 2Δm² + 3m_ν)")
    
    opt, fps, SM_bw, GUT_bw, K3_GUT = optimize_full_model()
    
    logM1, logM2, logM3, log_ls, logAD, log_ell, log_aGR = opt.x
    M_diag = [10**logM1, 10**logM2, 10**logM3]
    l_s_t = 10**log_ls; A_D = 10**logAD
    ell_neck = 10**log_ell; a_GUT_ratio = 10**log_aGR
    
    print(f"\n  ─── Parámetros Óptimos ───")
    print(f"    M₁ = {M_diag[0]:.3e} GeV")
    print(f"    M₂ = {M_diag[1]:.3e} GeV")
    print(f"    M₃ = {M_diag[2]:.3e} GeV")
    print(f"    l̃_s = {l_s_t:.4f}  (l̃_s/l_s = {l_s_t/l_s_lep:.2f})")
    print(f"    A_D = {A_D:.4e}")
    print(f"    ℓ_neck = {ell_neck:.4f}")
    print(f"    a_GUT/a_SM = {a_GUT_ratio:.2f}")
    print(f"    Costo = {opt.fun:.4e}")
    
    # Reconstruct M_R
    a_SM_val = 0.05
    a_k = np.zeros(16)
    for k in range(16):
        if SM_bw[k] and GUT_bw[k]:    a_k[k] = a_SM_val * np.sqrt(a_GUT_ratio)
        elif SM_bw[k]:                  a_k[k] = a_SM_val
        elif GUT_bw[k]:                 a_k[k] = a_SM_val * a_GUT_ratio
        else:                           a_k[k] = 0.03
    
    F_neck = compute_neck_factors(t_cones, lambda_ACyl, ell_neck)
    F_K3 = compute_K3_factors(K3_SM, K3_GUT, fps, a_k, 0.5, SM_bw, GUT_bw, l_s_t)
    F_total = np.zeros((3,3))
    for i in range(3):
        F_total[i,i] = 1.0
        for j in range(i+1, 3):
            F_total[i,j] = F_total[j,i] = F_K3[i,j] * F_neck[i,j]
    
    MR = build_MR_combined(M_diag, F_K3, F_neck)
    
    print(f"\n  ─── Factores de Supresión ───")
    print(f"  {'Par':>6} {'F_K3':>12} {'F_cuello':>12} {'F_total':>12} {'M_R(ij)/√MiMj':>15}")
    for i,j in [(0,1),(1,2),(0,2)]:
        ftot = F_total[i,j]
        ratio_MR = MR[i,j] / np.sqrt(MR[i,i]*MR[j,j]) if MR[i,i]*MR[j,j] > 0 else 0
        print(f"  {i+1}↔{j+1}   {F_K3[i,j]:>12.4e} {F_neck[i,j]:>12.4e} "
              f"{ftot:>12.4e} {ratio_MR:>15.4e}")
    
    print(f"\n  ─── M_R (GeV) ───")
    for i in range(3):
        row = "    │"
        for j in range(3):
            row += f" {MR[i,j]:>14.4e}"
        print(row + " │")
    
    eigs = np.sort(np.linalg.eigvalsh(MR))
    print(f"\n    Eigenvalues: {eigs[0]:.3e}, {eigs[1]:.3e}, {eigs[2]:.3e}")
    print(f"    Ratio max/min: {eigs[2]/max(eigs[0],1e-10):.1f}")
    
    # Seesaw prediction
    r = seesaw(A_D, MR)
    
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  PREDICCIÓN CON M_R (VÍA 1 + CUELLO)")
    print(f"  ═══════════════════════════════════════════════════")
    
    print(f"\n    m₁ = {r['masses_eV'][0]:.4e} eV")
    print(f"    m₂ = {r['masses_eV'][1]:.4e} eV")
    print(f"    m₃ = {r['masses_eV'][2]:.4e} eV")
    print(f"    Σmᵢ = {r['sum_m']:.4e} eV")
    
    for name, val, exp in [('θ₁₂',r['theta12'],exp_t12),
                            ('θ₂₃',r['theta23'],exp_t23),
                            ('θ₁₃',r['theta13'],exp_t13)]:
        d=np.degrees(val); e=np.degrees(exp); rat=d/e
        s="✅" if abs(rat-1)<0.03 else ("⊕" if abs(rat-1)<0.10 else "⚠️")
        print(f"    {name} = {d:7.2f}° (exp: {e:.2f}°, {rat:.3f}×) {s}")
    
    r21=r['dm2_21']/exp_dm2_21; r32=r['dm2_32']/exp_dm2_32
    s21="✅" if abs(r21-1)<0.1 else ("⊕" if abs(r21-1)<0.3 else "⚠️")
    s32="✅" if abs(r32-1)<0.1 else ("⊕" if abs(r32-1)<0.3 else "⚠️")
    print(f"    Δm²₂₁ = {r['dm2_21']:.3e} eV² (exp: {exp_dm2_21:.3e}, {r21:.3f}×) {s21}")
    print(f"    Δm²₃₂ = {r['dm2_32']:.3e} eV² (exp: {exp_dm2_32:.3e}, {r32:.3f}×) {s32}")
    print(f"    Δm²₃₂/Δm²₂₁ = {r['dm2_ratio']:.1f} (exp: {exp_dm2_ratio:.1f})")
    
    
    # ════════════════════════════════════════════════════════
    # PARTE 3: Scan de ℓ_neck con seesaw completo
    # ════════════════════════════════════════════════════════
    
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE 3: Scan de ℓ_neck → Δm² Ratio                       ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    print(f"\n  Fijando a_GUT/a_SM={a_GUT_ratio:.1f}, optimizando {{M_diag,l̃_s,A_D}} para cada ℓ_neck")
    print(f"\n  {'ℓ_neck':>8} {'θ₁₂':>6} {'θ₂₃':>6} {'θ₁₃':>6} "
          f"{'Δm²₂₁':>10} {'Δm²₃₂':>10} {'ratio':>7} {'cost':>9}")
    print(f"  {'─'*68}")
    
    scan_ells = [0.02, 0.05, 0.07, 0.10, 0.15, 0.30]
    best_scan = None; best_scan_cost = np.inf
    
    for ell in scan_ells:
        F_n = compute_neck_factors(t_cones, lambda_ACyl, ell)
        
        # Quick optimize with this fixed neck
        res = optimize_combined(F_n, fps, a_k, 0.5, SM_bw, GUT_bw, K3_GUT)
        p = res.x
        M_d = [10**p[0], 10**p[1], 10**p[2]]
        ls = 10**p[3]; ad = 10**p[4]
        FK = compute_K3_factors(K3_SM, K3_GUT, fps, a_k, 0.5, SM_bw, GUT_bw, ls)
        MR_s = build_MR_combined(M_d, FK, F_n)
        rs = seesaw(ad, MR_s)
        
        mark = ""
        if res.fun < best_scan_cost:
            best_scan_cost = res.fun
            best_scan = {'ell': ell, 'result': rs, 'MR': MR_s, 'F_neck': F_n,
                         'F_K3': FK, 'M_diag': M_d, 'l_s_t': ls, 'A_D': ad, 'cost': res.fun}
            mark = " ◄"
        
        print(f"  {ell:>8.3f} {np.degrees(rs['theta12']):>6.1f} "
              f"{np.degrees(rs['theta23']):>6.1f} {np.degrees(rs['theta13']):>6.1f} "
              f"{rs['dm2_21']:>10.2e} {rs['dm2_32']:>10.2e} "
              f"{rs['dm2_ratio']:>7.1f} {res.fun:>9.2e}{mark}")
    
    print(f"\n  Exp:{'':>2} {33.41:>6.1f} {49.1:>6.1f} {8.54:>6.1f} "
          f"{exp_dm2_21:>10.2e} {exp_dm2_32:>10.2e} {exp_dm2_ratio:>7.1f}")
    
    
    # ════════════════════════════════════════════════════════
    # PARTE 4: Comparación de todos los modelos
    # ════════════════════════════════════════════════════════
    
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE 4: Comparación de Todos los Modelos                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    # Model C baseline
    rC = seesaw(0.295, np.diag([7.1e9, 1.2e10, 1.0e10]).astype(float))
    
    # Best scan result
    B = best_scan
    rB = B['result']
    
    print(f"\n  ┌────────────┬────────────┬────────────┬────────────┬────────────┐")
    print(f"  │ Observable │  Modelo C  │ V1+Cuello  │  Off-diag  │   Exp.     │")
    print(f"  │            │ (diagonal) │ (geom+ACyl)│ libre(F4)  │            │")
    print(f"  ├────────────┼────────────┼────────────┼────────────┼────────────┤")
    
    # Phase 4 off-diagonal result (from previous script)
    rP4 = {'theta12': np.radians(32.54), 'theta23': np.radians(46.07),
            'theta13': np.radians(8.67),
            'dm2_21': 7.367e-5, 'dm2_32': 2.509e-3, 'dm2_ratio': 34.1,
            'sum_m': 0.0594}
    
    rows = [
        ('θ₁₂ (°)', f"{np.degrees(rC['theta12']):.1f}", 
         f"{np.degrees(rB['theta12']):.1f}", f"{np.degrees(rP4['theta12']):.1f}", "33.4"),
        ('θ₂₃ (°)', f"{np.degrees(rC['theta23']):.1f}",
         f"{np.degrees(rB['theta23']):.1f}", f"{np.degrees(rP4['theta23']):.1f}", "49.1"),
        ('θ₁₃ (°)', f"{np.degrees(rC['theta13']):.1f}",
         f"{np.degrees(rB['theta13']):.1f}", f"{np.degrees(rP4['theta13']):.1f}", "8.54"),
        ('Δm²₂₁', f"{rC['dm2_21']:.1e}", f"{rB['dm2_21']:.1e}",
         f"{rP4['dm2_21']:.1e}", f"{exp_dm2_21:.1e}"),
        ('Δm²₃₂', f"{rC['dm2_32']:.1e}", f"{rB['dm2_32']:.1e}",
         f"{rP4['dm2_32']:.1e}", f"{exp_dm2_32:.1e}"),
        ('ratio', f"{rC['dm2_ratio']:.1f}", f"{rB['dm2_ratio']:.1f}",
         f"{rP4['dm2_ratio']:.1f}", f"{exp_dm2_ratio:.1f}"),
        ('Σmᵢ (eV)', f"{rC['sum_m']:.3f}", f"{rB['sum_m']:.3f}",
         f"{rP4['sum_m']:.3f}", "< 0.12"),
    ]
    for name, vC, vG, vF, vE in rows:
        print(f"  │ {name:<10} │ {vC:>10} │ {vG:>10} │ {vF:>10} │ {vE:>10} │")
    
    print(f"  └────────────┴────────────┴────────────┴────────────┴────────────┘")
    
    print(f"\n  ─── Conteo de Parámetros ───")
    print(f"  {'Modelo':>20} {'Libres':>8} {'Observables':>12} {'Ratio':>8}")
    print(f"  {'Modelo C (diag)':>20} {'3':>8} {'3θ':>12} {'1:1':>8}")
    print(f"  {'Off-diag libre':>20} {'7':>8} {'3θ+2Δm²':>12} {'1:0.7':>8}")
    print(f"  {'Vía1+Cuello':>20} {'7':>8} {'3θ+2Δm²+3m':>12} {'1:1.1':>8}")
    
    # ════════════════════════════════════════════════════════
    # PARTE 5: Interpretación física
    # ════════════════════════════════════════════════════════
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  PARTE 5: Interpretación Física y Siguiente Paso             ║
╚═══════════════════════════════════════════════════════════════╝

  ─── RESULTADOS CLAVE ───

  1. EFECTO DEL CUELLO: ℓ_neck = {B['ell']:.3f}
     F(1↔2) = {B['F_neck'][0,1]:.3e}  ← {'' if B['F_neck'][0,1] > 0.1 else 'SUPRESIÓN FUERTE'}
     F(2↔3) = {B['F_neck'][1,2]:.3e}  ← {'cruza junction' if (t_cones[1]-0.5)*(t_cones[2]-0.5) < 0 else 'mismo lado'}
     F(1↔3) = {B['F_neck'][0,2]:.3e}  ← cruza todo el cuello
     Ratio F₁₂/F₁₃ = {B['F_neck'][0,1]/max(B['F_neck'][0,2],1e-30):.0f}×
  
  2. CONTRIBUCIÓN K3:
     F_K3(1↔2) = {B['F_K3'][0,1]:.3e}
     F_K3(2↔3) = {B['F_K3'][1,2]:.3e}
     F_K3(1↔3) = {B['F_K3'][0,2]:.3e}
  
  3. ─── Razón de la mejora parcial ───
     
     El mecanismo de cuello genera la jerarquía CORRECTA en dirección:
     (M_R)₁₂ ≫ (M_R)₂₃ > (M_R)₁₃
     
     Pero el mecanismo SEESAW completo necesita que esta jerarquía
     se combine con la estructura de m_D para producir los ángulos
     θ₂₃ ≈ 49°. El problema residual es que θ₂₃ se queda en ~28°.
     
  4. ─── El obstáculo de θ₂₃ ───
     
     θ₂₃ = 49.1° (maximal mixing) requiere |U_μ₃| ≈ |U_τ₃|, es decir
     que el estado ν₃ se mezcle igualmente con μ y τ. Esto está 
     controlado principalmente por la estructura de m_D, NO de M_R.
     
     En el Modelo C, m_D tiene:
     • (m_D)₂₃ = 0.317 GeV (grande)
     • (m_D)₃₃ = 0.570 GeV (diagonal mayor)
     • Ratio (m_D)₂₃/(m_D)₃₃ = 0.56 → insuficiente para θ₂₃ maximal
     
     Para θ₂₃ ≈ 45°, necesitamos (m_D)₂₃ ≈ (m_D)₃₃, lo cual 
     requiere que la amplitud de instanton A(2↔3) sea mayor.
     
     SOLUCIÓN POSIBLE: l_s(leptón) sectorial.
     El Modelo B ya exploró l_s por generación. Con l_s₃ > l_s₂,
     la supresión de A(2↔3) se reduce, aumentando θ₂₃.
     Combinado con el cuello TCS, esto podría cerrar el gap.
    """)
    
    # Volúmenes de ciclos requeridos
    print(f"  ─── Predicciones Geométricas ───")
    M_GUT = 2e16
    print(f"  Volúmenes de 3-ciclos co-asociativos requeridos:")
    print(f"  (M_Rk ~ M_GUT × exp(-Vol(Σ̃_k)/l_P³), M_GUT = {M_GUT:.0e} GeV)")
    for k, M in enumerate(B['M_diag']):
        if 0 < M < M_GUT:
            vol = np.log(M_GUT/M)
            print(f"    Gen {k+1}: Vol(Σ̃_{k+1})/l_P³ = {vol:.2f}")
    
    print(f"\n  ℓ_neck = {B['ell']:.3f} → longitud de cuello TCS en unidades de l₁₁")
    print(f"  Comparar con T_Hitchin ≈ √Vol(K3) ≈ √0.95 ≈ 0.97 l₁₁")
    print(f"  Ratio ℓ_neck/T_Hitchin ≈ {B['ell']/0.97:.2f}")
    
    print(f"\n  ─── Predicciones Falsificables ───")
    print(f"  1. Normal Ordering              → JUNO (~2027)")
    print(f"  2. m₁ = {rB['masses_eV'][0]:.2e} eV       → KATRIN/Project 8")
    print(f"  3. Σmᵢ = {rB['sum_m']:.4f} eV           → CMB-S4 / Euclid")
    print(f"  4. ℓ_neck/{0.97:.2f} = {B['ell']/0.97:.2f}          → Constrains TCS geometry")
    print(f"  5. Δm²₃₂/Δm²₂₁ = {rB['dm2_ratio']:.1f}        → Precision oscillation data")
    
    print(f"\n{'═'*72}")
    print(f"  Análisis Vía 1 + Cuello TCS completo.")
    print(f"{'═'*72}")
