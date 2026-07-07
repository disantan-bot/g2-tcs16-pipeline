#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  l_s DESDE VOLÚMENES DE CICLOS: ¿Sale la ratio quark/leptón?
═══════════════════════════════════════════════════════════════════════════

  Datos del Compendio (§10):
    Σ₁ → SU(3)_C : Vol = 1.909798 l_P³
    Σ₂ → SU(2)_L : Vol = 0.953088 l_P³
    Σ₃ → U(1)_Y  : Vol = 1.121823 l_P³

  M-theory: la Yukawa exp(-d/l_s) viene de M2 sobre 3-ciclo:
    S_M2 = Vol(Σ_ij) / (2π l₁₁³)
    d_ij / l_s = S_M2
    → l_s depende del volumen de la fibra gauge del 3-ciclo

  Quarks acoplan a SU(3)_C, leptones a SU(2)_L. Si l_s ∝ 1/Vol(Σ):
    l_s(lep)/l_s(quark) = Vol(Σ₁)/Vol(Σ₂) = 2.005

  ¿Ese factor 2 explica por qué PMNS necesita l_s ~ 0.3 y CKM ~ 0.044?
  Respuesta rápida: no, es factor ~7. Pero exploremos sistemáticamente.

  Diego Santana S. — Marzo 2026
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# ═══ Ciclos E₈ ═══
Vol_SU3 = 1.909798   # l_P³
Vol_SU2 = 0.953088
Vol_U1  = 1.121823

# ═══ Constantes geométricas ═══
d_H = np.array([0.561, 0.347, 0.198])
d_12, d_23, d_13 = 0.166, 0.343, 0.343
d_off = {(0,1): d_12, (1,2): d_23, (0,2): d_13}
N_flux = {(0,1): 1, (1,2): 2, (0,2): 2}

# Conos
t_cones = np.array([0.35, 0.50, 0.65])
lambda_ACyl = 2.8; alpha_FHN = 0.15

# Hitchin
S_CACHE = {}
for i in range(3):
    for j in range(i+1, 3):
        def f(t, ii=i, jj=j):
            s2 = 1.0/np.cosh(lambda_ACyl*(t-0.5))**2
            th = abs(np.tanh(lambda_ACyl*(t-0.5)))
            return s2*(1+alpha_FHN*th)
        val, _ = quad(f, t_cones[i], t_cones[j])
        S_CACHE[(i,j)] = val

# Masas
v_ew = 246.22/np.sqrt(2)
m_up   = np.array([2.16e-3, 1.27, 172.76])
m_down = np.array([4.67e-3, 0.0934, 4.18])

# Experimentales
CKM_EXP = {'t12': np.radians(13.02), 't23': np.radians(2.40), 't13': np.radians(0.211),
            'J': 3.08e-5}
PMNS_EXP = {'t12': np.radians(33.41), 't23': np.radians(49.1), 't13': np.radians(8.54),
            'dm2_21': 7.41e-5, 'dm2_32': 2.507e-3}

delta_CKM = np.radians(64.5)
delta_PMNS = np.pi


print("═" * 72)
print("  l_s DESDE VOLÚMENES DE CICLOS")
print("═" * 72)


# ═══════════════════════════════════════════════════════════════
# PARTE 1: ANSÄTZE TEÓRICOS
# ═══════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  PARTE 1: Cinco Ansätze para l_s(Vol)                        ║
╚═══════════════════════════════════════════════════════════════╝

  Ciclos gauge del Compendio:
    Vol(Σ₁) = {Vol_SU3:.4f} l_P³  (SU(3)_C)
    Vol(Σ₂) = {Vol_SU2:.4f} l_P³  (SU(2)_L)
    Vol(Σ₃) = {Vol_U1:.4f} l_P³  (U(1)_Y)
    
  Ratios fundamentales:
    Vol₁/Vol₂ = {Vol_SU3/Vol_SU2:.4f}  (SU(3)/SU(2))
    Vol₃/Vol₂ = {Vol_U1/Vol_SU2:.4f}  (U(1)/SU(2))
    Vol₁/Vol₃ = {Vol_SU3/Vol_U1:.4f}  (SU(3)/U(1))
""")

# Ansatz A: l_s ∝ 1/Vol(Σ)  [acción M2 inversa]
# Motivación: S = Vol(Σ)/2πl³ → l_s = 2πl³ d/Vol(Σ) → l_s ∝ 1/Vol
# Quarks: fiber = SU(3) → l_s_q ∝ 1/Vol_SU3
# Leptons: fiber = SU(2) → l_s_l ∝ 1/Vol_SU2

# Ansatz B: l_s ∝ Vol(Σ)^(1/3)  [radio del ciclo]
# Motivación: l_s ~ "tamaño" del ciclo, Vol^(1/3) = radio

# Ansatz C: l_s ∝ Vol(Σ)^(-1/3)  [tensión efectiva]
# Motivación: T_eff ~ Vol^(1/3)/l³ → l_s ~ l³/Vol^(1/3)

# Ansatz D: cada generación tiene su propio l_s por posición en K3
# l_s_k = l_0 × f(cone_k, sector)
# donde f depende de la distancia del cono al centro del sector gauge

# Ansatz E: l_s(sector) = l₁₁ / (α_sector × Vol_sector)^(1/2)
# Motivación: string tension en la reducción dimensional

Vols = {'SU3': Vol_SU3, 'SU2': Vol_SU2, 'U1': Vol_U1}

print(f"  Ansatz                l_s_q/l_s_l   Dirección   ¿Factor ~7?")
print(f"  {'─'*60}")

# A: l_s ∝ 1/Vol
r_A = Vol_SU2/Vol_SU3  # l_s_q/l_s_l since l_s ∝ 1/Vol
print(f"  A: l_s ∝ 1/Vol          {r_A:.4f}      q < l       No ({1/r_A:.1f}×)")

# B: l_s ∝ Vol^(1/3)
r_B = (Vol_SU3/Vol_SU2)**(1/3)
print(f"  B: l_s ∝ Vol^(1/3)      {r_B:.4f}      q > l       No ({r_B:.1f}×)")

# C: l_s ∝ Vol^(-1/3)
r_C = (Vol_SU2/Vol_SU3)**(1/3)
print(f"  C: l_s ∝ Vol^(-1/3)     {r_C:.4f}      q < l       No ({1/r_C:.1f}×)")

# D: l_s ∝ 1/Vol^(2/3)
r_D = (Vol_SU2/Vol_SU3)**(2/3)
print(f"  D: l_s ∝ Vol^(-2/3)     {r_D:.4f}      q < l       No ({1/r_D:.1f}×)")

# E: l_s ∝ exp(-Vol)
r_E = np.exp(Vol_SU2 - Vol_SU3)
print(f"  E: l_s ∝ exp(-Vol)      {r_E:.4f}      q < l       No ({1/r_E:.1f}×)")

print(f"""
  RESULTADO: Ningún ansatz simple da factor ~7.
  El máximo es 2.0× (Ansatz A, inversión directa).
  
  Implicación: la diferencia l_s(lep)/l_s(quark) ~ 7 NO puede
  venir solo de Vol(Σ₁) vs Vol(Σ₂). Se necesita otro ingrediente.
""")


# ═══════════════════════════════════════════════════════════════
# PARTE 2: ¿CUÁL ES EL FACTOR FALTANTE?
# ═══════════════════════════════════════════════════════════════

print(f"╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 2: Anatomía del Factor Faltante                       ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

# CKM del Compendio usa l_s = 0.044
# PMNS fit usa l_s ~ 0.15-0.45 (mediana ~0.28)
# PERO: en el PMNS, l_s NO aparece directamente en las masas —
# aparece en m_D, que luego pasa por el seesaw.
# La amplificación seesaw (m_nu = -mD MR^{-1} mD^T) transforma
# el efecto de l_s de forma no-lineal.

# Pregunta clave: ¿cuál es el l_s "efectivo" que el seesaw produce?
# Si M_R es diagonal-dominante, m_nu ≈ m_D² / M_R
# Entonces exp(-d/l_s) en m_D se convierte en exp(-2d/l_s) en m_nu
# → l_s_eff(PMNS) = l_s(Dirac) / 2

# Verificar: si l_s(Dirac) = 0.044 (como CKM), entonces
# l_s_eff(PMNS) ~ 0.044/2 = 0.022
# Pero el fit PMNS necesita l_s ~ 0.28 >> 0.022
# → NO, la amplificación seesaw va en la dirección EQUIVOCADA

# La otra posibilidad: l_s en PMNS no es el mismo concepto que l_s en CKM.
# En PMNS: l_s controla m_D Y las off-diagonals de M_R
# En CKM: l_s solo controla las off-diagonals de Y_u y Y_d

# Veamos qué pasa si separamos:
# l_s(Dirac) para m_D → misma que quarks
# l_s(Majorana) para M_R off-diag → diferente, viene de M5 (no M2)

print(f"""
  El factor faltante no viene de volúmenes puros.
  
  Observación clave: en el PMNS hay DOS tipos de l_s:
  
  1. l_s(Dirac) en m_D = Y × v_H
     → Yukawa del mismo tipo que quarks
     → Debería ser l_s ~ 0.044 (como CKM)
     
  2. l_s(Majorana) en M_R off-diagonals  
     → Viene de M5-branas (4-ciclos co-asociativos)
     → Puede ser muy diferente de l_s(Dirac)
     
  En el fit PMNS anterior, usamos UN solo l_s para todo.
  Eso forzó l_s ~ 0.3 como compromiso entre:
  - l_s(Dirac) ~ 0.044 (necesita ser pequeño para jerarquía de masas)
  - l_s(Majorana) ~ grande (para off-diag de M_R significativos)
  
  HIPÓTESIS: Separar l_s(Dirac) = l_s(CKM) y l_s(Maj) libre.
  Esto es físicamente correcto porque M2 ≠ M5.
""")


# ═══════════════════════════════════════════════════════════════
# PARTE 3: MODELO DUAL l_s: Dirac + Majorana separados
# ═══════════════════════════════════════════════════════════════

print(f"╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 3: Modelo Dual l_s (Dirac + Majorana)                 ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

def build_mD_dual(AD, ls_D_vec, C0_D):
    """m_D con l_s de Dirac (tipo M2, como quarks)."""
    ls1,ls2,ls3 = ls_D_vec
    y_D = np.array([AD*np.exp(-d_H[0]/ls1), AD*np.exp(-d_H[1]/ls2), AD*np.exp(-d_H[2]/ls3)])
    Y = np.diag(y_D.astype(complex))
    phase = np.exp(1j*delta_PMNS)
    for (i,j) in [(0,1),(1,2),(0,2)]:
        lse = np.sqrt(ls1*ls2) if (i,j)==(0,1) else (np.sqrt(ls2*ls3) if (i,j)==(1,2) else np.sqrt(ls1*ls3))
        amp = C0_D*np.exp(-d_off[(i,j)]/lse)*np.sqrt(y_D[i]*y_D[j])
        Y[i,j]=amp*phase; Y[j,i]=amp*np.conj(phase)
    return Y * v_ew

def build_MR_dual(M_diag, mu, ls_M_vec):
    """M_R con l_s de Majorana (tipo M5, cuello TCS)."""
    ls1,ls2,ls3 = ls_M_vec
    F = np.ones((3,3))
    for i in range(3):
        for j in range(i+1,3):
            F[i,j] = F[j,i] = np.exp(-mu * S_CACHE[(i,j)])
    
    # Additional l_s modulation from Majorana sector
    MR = np.zeros((3,3))
    for i in range(3):
        MR[i,i] = M_diag[i]
        for j in range(i+1,3):
            lse = np.sqrt(ls1*ls2) if (i,j)==(0,1) else (np.sqrt(ls2*ls3) if (i,j)==(1,2) else np.sqrt(ls1*ls3))
            # M5 instanton: F_cuello × exp(-d_ij/l_s_Maj)
            F_extra = np.exp(-d_off[(i,j)] / lse)
            MR[i,j] = MR[j,i] = np.sqrt(M_diag[i]*M_diag[j]) * F[i,j] * F_extra
    return MR

def seesaw(mD, MR):
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


# ─── Ansatz Volumétrico para l_s(Dirac) ───
# l_s(Dirac)_k = l_0 × (V_ref / Vol(Σ_sector_k))
# Para quarks: sector = SU(3)_C, Vol = 1.91
# Para leptones: sector = SU(2)_L, Vol = 0.95
# Ratio: l_s_lep/l_s_quark = Vol_SU3/Vol_SU2 = 2.005

# Generational splitting from K3 local geometry:
# Each cone is at a different position in K3, with different local volume
# The E₈ lattice distances suggest:
# Cone 2 (gen 1, d=0.561): farthest → smallest local volume → largest l_s
# Cone 1 (gen 2, d=0.347): middle
# Cone 3 (gen 3, d=0.198): nearest → largest local volume → smallest l_s
# This is counter-intuitive but could work

# Simple model: l_s_k = l_0 × (d_H_k / d_H_3) for generation k
# This gives: l_s_1/l_s_3 = 0.561/0.198 = 2.83
#             l_s_2/l_s_3 = 0.347/0.198 = 1.75

# But for quarks vs leptons, we scale by gauge sector volume:
# l_s_k(quark) = l_0_q × g(k)
# l_s_k(lepton) = l_0_l × g(k)
# l_0_l / l_0_q = Vol_SU3 / Vol_SU2 = 2.005

print(f"""
  ─── Ansatz Volumétrico con Splitting Generacional ───
  
  l_s_k(sector) = l₀(sector) × g(k)
  
  donde:
    l₀(quark)  = l₀ / Vol(SU3)^p    [fibra quark más gruesa → l_s menor]
    l₀(lepton) = l₀ / Vol(SU2)^p    [fibra lepton más fina → l_s mayor]
    g(k) = (d_H_k / d_H_ref)^q      [posición generacional en K3]
  
  Parámetros: l₀, p, q (3 params → predice 6 l_s: 3 quark + 3 lepton)
  Con p=1: ratio lep/quark = Vol(SU3)/Vol(SU2) = {Vol_SU3/Vol_SU2:.3f}
  
  Escaneando p y q para encontrar compatibilidad PMNS + CKM...
""")


# ═══════════════════════════════════════════════════════════════
# PARTE 4: FIT COMBINADO PMNS + CKM CON ANSATZ VOLUMÉTRICO
# ═══════════════════════════════════════════════════════════════

print(f"╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 4: Fit Combinado con Ansatz Volumétrico               ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

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


def combined_cost(params):
    """
    Fit PMNS + CKM simultaneously con ansatz volumétrico.
    
    Params [14]:
      0-2: log(M₁), log(M₂), log(M₃)    [Majorana masses]
      3:   log(μ)                          [Hitchin coupling]
      4:   log(A_D)                        [Dirac amplitude]
      5:   log(C₀_D)                       [Dirac off-diag prefactor (leptons)]
      6:   log(l₀)                         [base string length]
      7:   p                               [volume exponent]
      8:   q                               [generational exponent]
      9:   log(l₀_Maj)                     [Majorana string length scale]
      10:  log(C₀_q)                       [quark off-diag prefactor]
      11:  log(κ)                           [flux coupling]
      12:  nk_boost                        [NK boost for θ₁₂ CKM]
    """
    try:
        (logM1,logM2,logM3, log_mu, logAD, logC0D,
         log_l0, p_vol, q_gen, log_l0M,
         logC0q, log_kappa, nk_boost) = params
        
        M_diag = [10**logM1, 10**logM2, 10**logM3]
        mu = 10**log_mu; AD = 10**logAD; C0_D = 10**logC0D
        l0 = 10**log_l0; l0M = 10**log_l0M
        C0_q = 10**logC0q; kappa = 10**log_kappa
        
        # Generational factor g(k) = (d_H_k / d_H_max)^q
        d_ref = max(d_H)
        g = np.array([(d_H[k]/d_ref)**q_gen for k in range(3)])
        
        # l_s for leptons (Dirac sector uses SU(2) fiber)
        ls_lep_D = l0 / Vol_SU2**p_vol * g
        
        # l_s for quarks (Dirac sector uses SU(3) fiber)
        ls_quark = l0 / Vol_SU3**p_vol * g
        
        # l_s for Majorana (M5 sector, independent scale)
        ls_Maj = l0M * g
        
        # Ensure all positive
        if np.any(ls_lep_D <= 0) or np.any(ls_quark <= 0) or np.any(ls_Maj <= 0):
            return 1e10
        
        # ─── PMNS ───
        mD = build_mD_dual(AD, ls_lep_D, C0_D)
        MR = build_MR_dual(M_diag, mu, ls_Maj)
        eigs = np.linalg.eigvalsh(MR)
        if np.any(eigs <= 0): return 1e10
        
        r = seesaw(mD, MR)
        if np.any(np.isnan(r['m'])): return 1e10
        
        cost_pmns = 50*(((r['t12']-PMNS_EXP['t12'])/PMNS_EXP['t12'])**2 +
                         ((r['t23']-PMNS_EXP['t23'])/PMNS_EXP['t23'])**2 +
                         ((r['t13']-PMNS_EXP['t13'])/PMNS_EXP['t13'])**2)
        if r['dm2_21'] > 0 and r['dm2_32'] > 0:
            cost_pmns += 20*((np.log10(r['dm2_21'])-np.log10(PMNS_EXP['dm2_21']))**2 +
                              (np.log10(r['dm2_32'])-np.log10(PMNS_EXP['dm2_32']))**2)
        else: cost_pmns = 200
        
        # ─── CKM ───
        M_u = build_quark_matrix(m_up, ls_quark, C0_q, kappa, -1, nk_boost)
        M_d = build_quark_matrix(m_down, ls_quark, C0_q, kappa, +1, nk_boost)
        ckm = extract_ckm(M_u, M_d)
        
        cost_ckm = 30*(((ckm['t12']-CKM_EXP['t12'])/CKM_EXP['t12'])**2 +
                        ((ckm['t23']-CKM_EXP['t23'])/CKM_EXP['t23'])**2 +
                        ((ckm['t13']-CKM_EXP['t13'])/CKM_EXP['t13'])**2)
        if ckm['J'] > 0:
            cost_ckm += 10*(np.log10(ckm['J'])-np.log10(CKM_EXP['J']))**2
        else: cost_ckm += 100
        
        # Physicality penalty: M_R must have non-trivial off-diagonals
        # (the neck mechanism must be active, not just diagonal seesaw)
        MR_offdiag = abs(MR[0,1]) + abs(MR[1,2]) + abs(MR[0,2])
        MR_diag = abs(MR[0,0]) + abs(MR[1,1]) + abs(MR[2,2])
        offdiag_ratio = MR_offdiag / max(MR_diag, 1e-30)
        if offdiag_ratio < 1e-10:
            penalty = 50.0  # diagonal M_R = trivial seesaw
        elif offdiag_ratio < 1e-3:
            # Smooth penalty: 0 at ratio=1e-3, +50 at ratio=1e-10
            penalty = 50.0 * max(0, (-np.log10(max(offdiag_ratio,1e-30)) - 3) / 7.0)
        else:
            penalty = 0.0
        
        return cost_pmns + cost_ckm + penalty
        
    except: return 1e10


# Bounds: 13 parameters
bounds = [
    (8, 16),     # logM1
    (8, 16),     # logM2
    (8, 16),     # logM3
    (0.5, 2.5),  # log_mu (μ = 3 to 316, not infinity)
    (-4, 3),     # logAD
    (-2, 4),     # logC0D
    (-2.5, 0.5), # log_l0
    (0.1, 3.0),  # p_vol (volume exponent)
    (-1.5, 2.0), # q_gen (generational exponent)
    (-1.5, 0.5), # log_l0M (Majorana l_s scale: 0.03 to 3.16, NOT zero)
    (-3, 2),     # logC0q
    (-4, 0),     # log_kappa
    (-0.05, 0.25), # nk_boost (physical range)
]

print(f"\n  Optimización combinada PMNS + CKM (13 params, 9 obs)...")
print(f"  Parámetros: M₁₋₃, μ, A_D, C₀_D, l₀, p, q, l₀_M, C₀_q, κ, NK")
print(f"  Observables: 3θ_PMNS + 2Δm² + 3θ_CKM + J_CKM = 9")

best = None; best_cost = np.inf
for seed in range(20):
    res = differential_evolution(combined_cost, bounds, seed=seed*13+42,
                                  maxiter=800, tol=1e-16, popsize=30,
                                  mutation=(0.4, 1.9), recombination=0.85)
    if res.fun < best_cost:
        best_cost = res.fun
        best = res
    print(f"    Seed {seed}: cost = {res.fun:.4f}", flush=True)

# Second-stage refinement with L-BFGS-B (RESPECTS BOUNDS)
print(f"\n  Stage 2: L-BFGS-B polish on top 5 solutions (bounded)...")
# Collect top solutions
all_results = []
for seed in range(20):
    res2 = differential_evolution(combined_cost, bounds, seed=seed*13+42,
                                   maxiter=200, tol=1e-14, popsize=20,
                                   mutation=(0.5,1.8), recombination=0.85)
    all_results.append(res2)
all_results.sort(key=lambda r: r.fun)

for i, res_i in enumerate(all_results[:5]):
    polished = minimize(combined_cost, res_i.x, method='L-BFGS-B',
                        bounds=bounds, options={'maxiter': 50000, 'ftol': 1e-15})
    if polished.fun < best_cost:
        best_cost = polished.fun
        best = polished
    print(f"    Polish {i}: {res_i.fun:.4f} → {polished.fun:.4f}", flush=True)

# Also try with p forced near integer values (1,2,3)
print(f"\n  Stage 3: Scanning p near 1, 2, 3...")
for p_fixed in [1.0, 2.0, 3.0]:
    def cost_p_fixed(params_reduced):
        full = list(params_reduced[:7]) + [p_fixed] + list(params_reduced[7:])
        return combined_cost(full)
    bounds_reduced = bounds[:7] + bounds[8:]  # remove p_vol
    for s in range(3):
        res_pf = differential_evolution(cost_p_fixed, bounds_reduced, seed=s*17+7,
                                         maxiter=500, tol=1e-14, popsize=25,
                                         mutation=(0.5,1.8))
        # Polish with bounds
        full_x = list(res_pf.x[:7]) + [p_fixed] + list(res_pf.x[7:])
        polished = minimize(combined_cost, full_x, method='L-BFGS-B',
                            bounds=bounds, options={'maxiter': 20000, 'ftol': 1e-15})
        c = polished.fun
        if c < best_cost:
            best_cost = c
            best = polished
        print(f"    p={p_fixed:.0f}, seed {s}: cost = {c:.4f}", flush=True)

print(f"\n  Mejor costo final: {best_cost:.6f}")

p = best.x
(logM1,logM2,logM3, log_mu, logAD, logC0D,
 log_l0, p_vol, q_gen, log_l0M,
 logC0q, log_kappa, nk_boost) = p

l0 = 10**log_l0; l0M = 10**log_l0M
d_ref = max(d_H)
g = np.array([(d_H[k]/d_ref)**q_gen for k in range(3)])
ls_lep_D = l0 / Vol_SU2**p_vol * g
ls_quark = l0 / Vol_SU3**p_vol * g
ls_Maj = l0M * g

print(f"\n  ─── MEJOR SOLUCIÓN (cost = {best_cost:.4f}) ───")

print(f"\n  Parámetros del ansatz volumétrico:")
print(f"    l₀ = {l0:.4f}  (escala base)")
print(f"    p  = {p_vol:.3f}  (exponente de volumen)")
print(f"    q  = {q_gen:.3f}  (exponente generacional)")
print(f"    l₀_Maj = {l0M:.4f}  (escala Majorana)")

print(f"\n  l_s derivados:")
print(f"    {'Gen':>4} {'d_H':>6} {'g(k)':>6} {'l_s(lep)':>10} {'l_s(quark)':>10} {'l_s(Maj)':>10} {'ratio l/q':>10}")
print(f"    {'─'*60}")
for k in range(3):
    r_lq = ls_lep_D[k]/ls_quark[k]
    print(f"    {k+1:>4} {d_H[k]:>6.3f} {g[k]:>6.3f} {ls_lep_D[k]:>10.4f} {ls_quark[k]:>10.4f} {ls_Maj[k]:>10.4f} {r_lq:>10.3f}")

print(f"\n    Ratio l_s(lep)/l_s(quark) = Vol(SU3)^p/Vol(SU2)^p = ({Vol_SU3}/{Vol_SU2})^{p_vol:.2f} = {(Vol_SU3/Vol_SU2)**p_vol:.3f}")

# Evaluate PMNS
M_diag = [10**logM1, 10**logM2, 10**logM3]
mu = 10**log_mu; AD = 10**logAD; C0_D = 10**logC0D
C0_q = 10**logC0q; kappa = 10**log_kappa

mD = build_mD_dual(AD, ls_lep_D, C0_D)
MR = build_MR_dual(M_diag, mu, ls_Maj)
pmns = seesaw(mD, MR)

print(f"\n  ─── PMNS ───")
for name, key in [('θ₁₂','t12'),('θ₂₃','t23'),('θ₁₃','t13')]:
    d=np.degrees(pmns[key]); e=np.degrees(PMNS_EXP[key]); rat=d/e
    s="✅" if abs(rat-1)<0.03 else ("⊕" if abs(rat-1)<0.10 else "⚠️")
    print(f"    {name} = {d:7.2f}° (exp: {e:.2f}°, {rat:.4f}×) {s}")
for name, key in [('Δm²₂₁','dm2_21'),('Δm²₃₂','dm2_32')]:
    rat=pmns[key]/PMNS_EXP[key]
    s="✅" if abs(rat-1)<0.05 else ("⊕" if abs(rat-1)<0.15 else "⚠️")
    print(f"    {name} = {pmns[key]:.4e} eV² (exp: {PMNS_EXP[key]:.3e}, {rat:.3f}×) {s}")
print(f"    Ratio = {pmns['ratio']:.2f} (exp: 33.8)")
print(f"    Σmᵢ = {pmns['sum_m']:.4e} eV")

# Evaluate CKM
M_u = build_quark_matrix(m_up, ls_quark, C0_q, kappa, -1, nk_boost)
M_d = build_quark_matrix(m_down, ls_quark, C0_q, kappa, +1, nk_boost)
ckm = extract_ckm(M_u, M_d)

print(f"\n  ─── CKM ───")
for name, key in [('θ₁₂','t12'),('θ₂₃','t23'),('θ₁₃','t13')]:
    d=np.degrees(ckm[key]); e=np.degrees(CKM_EXP[key]); rat=d/e
    s="✅" if abs(rat-1)<0.30 else "⚠️"
    print(f"    {name} = {d:7.3f}° (exp: {e:.3f}°, {rat:.3f}×) {s}")
print(f"    J = {ckm['J']:.3e} (exp: {CKM_EXP['J']:.2e}, {ckm['J']/CKM_EXP['J']:.2f}×)")
print(f"    NK boost = {nk_boost*100:.1f}%")

# Cost breakdown
cost_pmns_angles = 50*(((pmns['t12']-PMNS_EXP['t12'])/PMNS_EXP['t12'])**2 +
                        ((pmns['t23']-PMNS_EXP['t23'])/PMNS_EXP['t23'])**2 +
                        ((pmns['t13']-PMNS_EXP['t13'])/PMNS_EXP['t13'])**2)
cost_pmns_dm2 = 20*((np.log10(pmns['dm2_21'])-np.log10(PMNS_EXP['dm2_21']))**2 +
                     (np.log10(pmns['dm2_32'])-np.log10(PMNS_EXP['dm2_32']))**2)
cost_ckm_angles = 30*(((ckm['t12']-CKM_EXP['t12'])/CKM_EXP['t12'])**2 +
                       ((ckm['t23']-CKM_EXP['t23'])/CKM_EXP['t23'])**2 +
                       ((ckm['t13']-CKM_EXP['t13'])/CKM_EXP['t13'])**2)
cost_ckm_J = 10*(np.log10(ckm['J'])-np.log10(CKM_EXP['J']))**2

print(f"\n  ─── Desglose del Costo ───")
print(f"    PMNS ángulos:  {cost_pmns_angles:.4f}")
print(f"    PMNS Δm²:      {cost_pmns_dm2:.4f}")
print(f"    CKM ángulos:   {cost_ckm_angles:.4f}")
print(f"    CKM Jarlskog:  {cost_ckm_J:.4f}")
print(f"    TOTAL:         {cost_pmns_angles+cost_pmns_dm2+cost_ckm_angles+cost_ckm_J:.4f}")
print(f"    → El cuello de botella es: {'PMNS Δm²' if cost_pmns_dm2 > max(cost_pmns_angles, cost_ckm_angles) else 'CKM ángulos'}")

# M_R diagnostics
print(f"\n  ─── Diagnóstico M_R ───")
MR_eigs = np.sort(np.linalg.eigvalsh(MR))
print(f"    Eigenvalues: {MR_eigs[0]:.3e}, {MR_eigs[1]:.3e}, {MR_eigs[2]:.3e} GeV")
print(f"    Hierarchy: {MR_eigs[2]/MR_eigs[0]:.1f}×")
print(f"    M_diag: {M_diag[0]:.3e}, {M_diag[1]:.3e}, {M_diag[2]:.3e} GeV")
print(f"    μ = {mu:.2f}, l₀_Maj = {l0M:.4f}")
MR_offdiag = abs(MR[0,1]) + abs(MR[1,2]) + abs(MR[0,2])
MR_diag_sum = abs(MR[0,0]) + abs(MR[1,1]) + abs(MR[2,2])
print(f"    M_R off-diag/diag = {MR_offdiag/MR_diag_sum:.3e}")
print(f"    M_R(0,1) = {MR[0,1]:.3e}, M_R(1,2) = {MR[1,2]:.3e}, M_R(0,2) = {MR[0,2]:.3e}")

# Physicality check
warnings_list = []
if mu > 300: warnings_list.append(f"μ={mu:.0f} muy grande (>300)")
if l0M < 0.01: warnings_list.append(f"l₀_Maj={l0M:.4f} muy pequeño (<0.01)")
if abs(nk_boost) > 0.25: warnings_list.append(f"NK boost={nk_boost*100:.1f}% fuera de rango físico")
if MR_offdiag/MR_diag_sum < 1e-8: warnings_list.append("M_R es diagonal → cuello TCS inactivo")
if warnings_list:
    print(f"\n  ⚠️  ALERTAS DE FISICALIDAD:")
    for w in warnings_list:
        print(f"    • {w}")
else:
    print(f"\n  ✅ Solución dentro de rango físico")


# ═══════════════════════════════════════════════════════════════
# PARTE 5: CONTEO DE PARÁMETROS ACTUALIZADO
# ═══════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  PARTE 5: Conteo de Parámetros con Ansatz Volumétrico        ║
╚═══════════════════════════════════════════════════════════════╝

  MODELO ANTERIOR (l_s libres por sector):
    Params PMNS: M₁,M₂,M₃, μ, A_D, l_s₁,l_s₂,l_s₃, C₀ = 9
    Params CKM:  C₀_q, κ, NK, l_s₁_q,l_s₂_q,l_s₃_q = 6
    Total: 15 params, 9 obs → ratio 9:15 = 0.60
    
  MODELO VOLUMÉTRICO:
    Params compartidos: l₀, p, q = 3  (determinan 6 l_s)
    Params PMNS: M₁,M₂,M₃, μ, A_D, C₀_D, l₀_Maj = 7
    Params CKM:  C₀_q, κ, NK = 3
    Total: 13 params, 9 obs → ratio 9:13 = 0.69
    
  GANANCIA: 2 parámetros eliminados por el ansatz volumétrico.
  l₀, p, q reemplazan 6 l_s independientes (ahorro de 3) 
  pero se añade l₀_Maj (costo de 1).
  Neto: −2 parámetros.
  
  Si p y q se fijan por la geometría:
    p = 1 (inversión simple) → −1 más
    q fijo por la estructura cónica → −1 más
    Total: 11 params, 9 obs → ratio 9:11 = 0.82
""")


# ═══════════════════════════════════════════════════════════════
# PARTE 6: VERIFICAR PREDICCIONES DE p Y q
# ═══════════════════════════════════════════════════════════════

print(f"╔═══════════════════════════════════════════════════════════════╗")
print(f"║  PARTE 6: ¿Son p y q valores naturales?                      ║")
print(f"╚═══════════════════════════════════════════════════════════════╝")

print(f"\n  Valor óptimo de p = {p_vol:.3f}")
if abs(p_vol - 1.0) < 0.3:
    print(f"  → Cercano a p=1 (inversión directa l_s ∝ 1/Vol)")
    print(f"    Interpretación: S_M2 = Vol(fiber)×path / 2πl₁₁³")
    print(f"    Con Vol(fiber) ∝ Vol(Σ_gauge), l_s = 2πl₁₁³/Vol(Σ)")
elif abs(p_vol - 0.5) < 0.2:
    print(f"  → Cercano a p=0.5 (raíz del volumen)")
    print(f"    Interpretación: l_s ∝ 1/√Vol, como radio efectivo")
elif abs(p_vol - 0.333) < 0.15:
    print(f"  → Cercano a p=1/3 (radio cúbico)")
else:
    print(f"  → Valor no-trivial, podría reflejar acoplamiento mixto")

print(f"\n  Valor óptimo de q = {q_gen:.3f}")
if abs(q_gen - 1.0) < 0.3:
    print(f"  → Cercano a q=1 (lineal en d_H)")
    print(f"    Interpretación: l_s_k ∝ d_H_k, como se esperaría si")
    print(f"    el volumen local del K3 crece linealmente con la distancia")
elif abs(q_gen - 0.0) < 0.2:
    print(f"  → Cercano a q=0 (sin splitting generacional)")
    print(f"    l_s es universal para las 3 generaciones")
elif q_gen < -0.3:
    print(f"  → q negativo: generaciones lejanas tienen l_s MENOR")
    print(f"    Contraintuitivo pero posible por curvatura del cono")

# What does the optimal solution predict for the fundamental ratio?
ratio_lq = (Vol_SU3/Vol_SU2)**p_vol
print(f"\n  ─── Predicción fundamental ───")
print(f"  l_s(leptón) / l_s(quark) = (Vol_SU3/Vol_SU2)^p = {ratio_lq:.3f}")
print(f"  Esto es una PREDICCIÓN: si p se fija teóricamente,")
print(f"  la ratio quark/leptón está determinada por los volúmenes.")


print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  CONCLUSIÓN                                                   ║
╚═══════════════════════════════════════════════════════════════╝
""")

if best_cost < 2.0:
    print(f"  ✅ El ansatz volumétrico FUNCIONA (cost = {best_cost:.3f})")
    print(f"  PMNS y CKM pueden coexistir con l_s derivados de volúmenes.")
    print(f"  La ratio l_s(lep)/l_s(quark) = {ratio_lq:.2f} emerge de Vol(SU3)/Vol(SU2).")
elif best_cost < 10:
    print(f"  ⊕ El ansatz volumétrico es PARCIALMENTE exitoso (cost = {best_cost:.2f})")
    print(f"  Mejora sobre l_s completamente libres pero no perfecto.")
else:
    print(f"  ⚠️  El ansatz volumétrico NO funciona (cost = {best_cost:.1f})")
    print(f"  l_s(quark) y l_s(leptón) no se relacionan por volúmenes simples.")

print(f"\n{'═'*72}")
print(f"  Derivación l_s desde volúmenes completa.")
print(f"{'═'*72}")
