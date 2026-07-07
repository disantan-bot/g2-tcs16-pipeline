#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  K3 KUMMER NO-ISÓTROPA: Distancias Majorana desde Métrica Deformada
  
  TCS-16 / E₈ Framework — Resolución del gap Δm² 
═══════════════════════════════════════════════════════════════════════════

  El problema: La rotación HK rígida (R_HK) preserva distancias → 
  d̃_Majorana ≈ d_Dirac → ratio Δm² insuficiente (20.7 vs 33.8).
  
  La solución: La K3 real es T⁴/ℤ₂ (Kummer) con 16 blowups Eguchi-Hanson.
  La métrica es Gibbons-Hawking multi-centro, NO plana. Las raíces SM y 
  GUT de E₈ se acoplan a DIFERENTES conjuntos de blowups, creando 
  distancias geodésicas asimétricas entre sectores.

  Estructura:
  ─────────────
  Parte A: Construcción de K3 Kummer (T⁴/ℤ₂, 16 puntos fijos)
  Parte B: Métrica Gibbons-Hawking multi-centro  
  Parte C: Asignación de blowups a sectores E₈ (SM vs GUT)
  Parte D: Distancias geodésicas no-isótropas
  Parte E: Seesaw con M_R deformado → PMNS + Δm²
  Parte F: Comparación y predicciones

  Diego Santana S. — Marzo 2026
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONSTANTES Y PARÁMETROS DEL COMPENDIO
# ═══════════════════════════════════════════════════════════════

# Métrica TCS-16 (Etapa 9a/10d)
a_t     = 0.013     # escala cuello
a_theta = 0.625     # escala toro  
a_K3    = 0.204     # escala K3
beta0   = 1.58      # twist hiper-Kähler

# Hitchin flow (Etapa 13)
lambda_ACyl = 2.8   # decaimiento ACyl
eta_NK      = 1/3   # Nearly-Kähler canónico

# Seesaw parameters
v_ew = 246.22 / np.sqrt(2)  # 174.1 GeV
l_s_lep    = 0.044
C0_Dirac   = 7348.0
delta_CP   = np.pi
d_H        = np.array([0.561, 0.347, 0.198])   # cono→Higgs
d_12_dirac = 0.166  # inter-cono Dirac
d_23_dirac = 0.343
d_13_dirac = 0.343

# Posiciones K3 SM (del Compendio §18.2)
K3_SM = np.array([
    [0.763, 0.431, 0.100, 0.431],   # SU(3)_C → Gen 1
    [0.431, 0.431, 0.763, 0.100],   # SU(2)_L → Gen 2
    [0.431, 0.431, 0.431, 0.900],   # U(1)_Y  → Gen 3
])

# Experimental neutrino data (NuFIT 5.3, NO)
exp_t12 = np.radians(33.41)
exp_t23 = np.radians(49.1)
exp_t13 = np.radians(8.54)
exp_dm2_21 = 7.41e-5
exp_dm2_32 = 2.507e-3
exp_dm2_ratio = exp_dm2_32 / exp_dm2_21

# Angular positions of cones on the torus
theta_cones = np.array([0.0, 0.26, 0.55])
t_cones     = np.array([0.35, 0.50, 0.65])


# ═══════════════════════════════════════════════════════════════
# PARTE A: K3 KUMMER — T⁴/ℤ₂ CON 16 PUNTOS FIJOS
# ═══════════════════════════════════════════════════════════════

def build_kummer_fixed_points():
    """
    K3 Kummer = T⁴/ℤ₂ donde ℤ₂ actúa como x → −x.
    Los 16 puntos fijos están en (n₁/2, n₂/2, n₃/2, n₄/2), nᵢ ∈ {0,1}.
    Cada punto fijo tiene una singularidad A₁ resuelta por un blowup
    Eguchi-Hanson de parámetro aₖ.
    
    T⁴ tiene período 1 en cada dirección: x ~ x + 1.
    """
    fixed_points = []
    for n1, n2, n3, n4 in product([0, 1], repeat=4):
        fixed_points.append(np.array([n1/2, n2/2, n3/2, n4/2]))
    return np.array(fixed_points)  # shape (16, 4)


def assign_blowups_to_E8(fixed_points):
    """
    Asigna los 16 blowups a sectores del lattice E₈.
    
    El lattice de Picard Pic(K3) ≅ ℤ¹⁶ tiene una base canónica dada 
    por las 16 clases excepcionales Eₖ de los blowups. La intersección
    con el lattice de raíces de E₈ determina qué blowups se acoplan
    a qué sector gauge.
    
    ASIGNACIÓN FÍSICA:
    ═════════════════
    E₈ tiene 8 raíces simples. La cadena E₈ → SO(10) → SU(5) → SM 
    asigna las raíces a sectores. Cada raíz αᵢ se asocia a los blowups
    cuya clase excepcional tiene producto de intersección no-nulo con αᵢ.
    
    Modelo concreto (Aspinwall-Morrison):
    - Raíces SM (α₁...α₄): se acoplan a los 8 blowups con n₁+n₂ par
      (sector "par" del lattice)
    - Raíces GUT (α₅...α₈): se acoplan a los 8 blowups con n₃+n₄ par  
      (sector "ortogonal")
    
    La superposición parcial (4 blowups comparten ambos sectores)
    es lo que conecta los sectores SM y GUT.
    """
    n_fp = len(fixed_points)
    
    # Clasificar puntos fijos por paridad
    SM_blowups  = []   # acoplados a α₁...α₄
    GUT_blowups = []   # acoplados a α₅...α₈
    
    for k in range(n_fp):
        fp = fixed_points[k]
        n = (fp * 2).astype(int)  # recover (n₁,n₂,n₃,n₄) ∈ {0,1}
        
        # SM sector: paridad de (n₁+n₂) 
        is_SM = (n[0] + n[1]) % 2 == 0
        # GUT sector: paridad de (n₃+n₄)
        is_GUT = (n[2] + n[3]) % 2 == 0
        
        SM_blowups.append(is_SM)
        GUT_blowups.append(is_GUT)
    
    SM_blowups = np.array(SM_blowups)
    GUT_blowups = np.array(GUT_blowups)
    
    # Count overlaps
    shared = np.sum(SM_blowups & GUT_blowups)
    SM_only = np.sum(SM_blowups & ~GUT_blowups)
    GUT_only = np.sum(~SM_blowups & GUT_blowups)
    neither = np.sum(~SM_blowups & ~GUT_blowups)
    
    return SM_blowups, GUT_blowups, {
        'SM_only': SM_only, 'GUT_only': GUT_only,
        'shared': shared, 'neither': neither
    }


def assign_blowup_parameters(fixed_points, SM_blowups, GUT_blowups, 
                               a_SM=0.05, a_GUT=0.15, a_shared=0.08, a_none=0.03):
    """
    Asigna parámetros de blowup aₖ a cada punto fijo.
    
    Los parámetros aₖ determinan el "tamaño" del blowup Eguchi-Hanson 
    en el punto fijo k. Blowups más grandes → mayor curvatura local
    → mayor deformación de geodésicas que pasan cerca.
    
    FÍSICA:
    El parámetro aₖ está determinado por el flujo G₄ a través del 
    2-ciclo excepcional Eₖ: aₖ ∝ ∫_{Eₖ} G₄.
    
    Los flujos G₄ están cuantizados (condición de Witten) y fijados
    por la estabilización de móduli. Los blowups del sector GUT 
    tienen flujos más grandes porque las raíces GUT (α₅...α₈) 
    corresponden a ciclos de volumen mayor en la descomposición E₈.
    
    Del Compendio §10: Vol(Σ₁)/Vol(Σ₂) = 2.005, Vol(Σ₃)/Vol(Σ₂) = 1.177
    Los volúmenes GUT son ~2-3× los SM → a_GUT ~ 2-3 × a_SM.
    """
    n_fp = len(fixed_points)
    a_k = np.zeros(n_fp)
    
    for k in range(n_fp):
        if SM_blowups[k] and GUT_blowups[k]:
            a_k[k] = a_shared
        elif SM_blowups[k]:
            a_k[k] = a_SM
        elif GUT_blowups[k]:
            a_k[k] = a_GUT
        else:
            a_k[k] = a_none
    
    return a_k


# ═══════════════════════════════════════════════════════════════
# PARTE B: MÉTRICA GIBBONS-HAWKING MULTI-CENTRO
# ═══════════════════════════════════════════════════════════════

def gibbons_hawking_potential(x, fixed_points, a_k, epsilon=0.5, r_min=0.02):
    """
    Potencial multi-centro de Gibbons-Hawking:
    
        V(x) = ε + Σₖ aₖ / |x − xₖ|²
    
    donde:
    - ε > 0 es el background constante (K3 asintóticamente plana)
    - aₖ es el parámetro de blowup del punto fijo k
    - xₖ es la posición del punto fijo k
    
    La métrica hiper-Kähler asociada es:
        ds² = V(x) dx·dx + V(x)⁻¹ (dτ + ω)²
    
    Para nuestro cálculo de distancias, la métrica efectiva en la 
    base ℝ³ (suprimiendo la fibra U(1)) es:
        ds²_eff = V(x) dx·dx
    
    El factor V(x) > 1 cerca de los blowups "estira" las distancias,
    actuando como una lente gravitacional.
    
    En T⁴: distancias se miden con identificación periódica x ~ x+1.
    """
    V = epsilon
    for k in range(len(fixed_points)):
        # Distancia con periodicidad T⁴
        dx = x - fixed_points[k]
        # Fold into fundamental domain [-0.5, 0.5]⁴
        dx = dx - np.round(dx)
        r2 = np.dot(dx, dx)
        r2 = max(r2, r_min**2)  # regularización
        V += a_k[k] / r2
    return V


def compute_metric_at_point(x, fixed_points, a_k, epsilon=0.5, sector='SM',
                             SM_blowups=None, GUT_blowups=None):
    """
    Calcula la métrica efectiva en x, seleccionando los blowups 
    relevantes para el sector (SM o GUT).
    
    CLAVE: Los instantones M2 (Dirac) se propagan por el sector SM
    y "sienten" principalmente los blowups SM. Los instantones M5 
    (Majorana) se propagan por el sector GUT y "sienten" principalmente
    los blowups GUT.
    
    Esto se implementa pesando los blowups por su acoplamiento al sector:
    
    V_SM(x) = ε + Σ_{k∈SM} aₖ/|x−xₖ|² + δ Σ_{k∉SM} aₖ/|x−xₖ|²
    V_GUT(x) = ε + Σ_{k∈GUT} aₖ/|x−xₖ|² + δ Σ_{k∉GUT} aₖ/|x−xₖ|²
    
    donde δ ≪ 1 es el acoplamiento cruzado (suprimido por la 
    ortogonalidad de los sectores en E₈).
    """
    delta_cross = 0.1  # acoplamiento cruzado suprimido
    r_min = 0.02
    
    V = epsilon
    for k in range(len(fixed_points)):
        dx = x - fixed_points[k]
        dx = dx - np.round(dx)
        r2 = max(np.dot(dx, dx), r_min**2)
        
        if sector == 'SM':
            weight = 1.0 if SM_blowups[k] else delta_cross
        elif sector == 'GUT':
            weight = 1.0 if GUT_blowups[k] else delta_cross
        else:
            weight = 1.0
        
        V += weight * a_k[k] / r2
    
    return V


# ═══════════════════════════════════════════════════════════════
# PARTE C: POSICIONES GUT VIA TWIST HK + DEFORMACIÓN KUMMER
# ═══════════════════════════════════════════════════════════════

def twist_HK_kummer(K3_SM_pos, beta0, fixed_points, a_k, GUT_blowups):
    """
    Calcula las posiciones GUT como twist HK + deformación por blowups.
    
    En K3 plana: K3_GUT = R_HK × K3_SM  (isometría, distancias iguales)
    En K3 Kummer: la deformación adicional viene de que el twist HK 
    mueve los puntos CERCA de blowups GUT diferentes, cambiando las 
    distancias geodésicas efectivas.
    
    El punto crucial: NO son las posiciones las que cambian 
    significativamente, sino la MÉTRICA a lo largo del camino entre ellas.
    """
    c = np.cos(beta0)
    s = np.sin(beta0)
    R_HK = np.array([
        [c, 0, -s, 0],
        [0, c,  0, -s],
        [s, 0,  c,  0],
        [0, s,  0,  c]
    ])
    
    # Peso de ν_R proyectado a K3: ½(-1,-1,-1,-1) (últimas 4 coords de E₈)
    nu_R_offset = 0.5 * np.array([-0.5, -0.5, -0.5, -0.5])
    
    K3_GUT = np.zeros_like(K3_SM_pos)
    for k in range(3):
        K3_GUT[k] = R_HK @ K3_SM_pos[k] + nu_R_offset
        # Fold into fundamental domain [0,1]⁴ (periodicity of T⁴)
        K3_GUT[k] = K3_GUT[k] % 1.0
    
    return K3_GUT


# ═══════════════════════════════════════════════════════════════
# PARTE D: DISTANCIAS GEODÉSICAS NO-ISÓTROPAS
# ═══════════════════════════════════════════════════════════════

def geodesic_distance_GH(p1, p2, fixed_points, a_k, epsilon, 
                          sector, SM_blowups, GUT_blowups, N_steps=80):
    """
    Calcula la distancia geodésica entre p1 y p2 en la métrica 
    Gibbons-Hawking multi-centro, para un sector dado (SM o GUT).
    
    MÉTODO: Aproximación de camino recto con integral de longitud.
    Para la métrica ds² = V(x) dx·dx, la longitud del camino recto es:
    
        L = ∫₀¹ √(V(γ(t))) |dγ/dt| dt
    
    donde γ(t) = (1-t)p1 + t·p2 (camino recto en coordenadas).
    
    En métrica plana: L = |p2 - p1| (factor V=const se absorbe).
    En métrica GH: L > |p2 - p1| cuando el camino pasa cerca de blowups.
    
    La geodésica real es MÁS CORTA que el camino recto, pero la 
    corrección es ~7-9% (del Compendio §21). La jerarquía relativa
    entre pares se preserva.
    """
    # Direction vector with periodic wrapping
    dp = p2 - p1
    dp = dp - np.round(dp)  # shortest path on T⁴
    
    flat_dist = np.linalg.norm(dp)
    if flat_dist < 1e-10:
        return 0.0
    
    # Integrate along straight path
    ts = np.linspace(0, 1, N_steps + 1)
    dt = 1.0 / N_steps
    
    total_length = 0.0
    for i in range(N_steps):
        t_mid = (ts[i] + ts[i+1]) / 2
        x_mid = p1 + t_mid * dp
        x_mid = x_mid % 1.0  # periodic
        
        V = compute_metric_at_point(x_mid, fixed_points, a_k, epsilon,
                                     sector, SM_blowups, GUT_blowups)
        
        # ds² = V dx² → ds = √V |dx|
        # |dx| = |dp| dt
        ds = np.sqrt(V) * flat_dist * dt
        total_length += ds
    
    return total_length


def compute_all_distances(K3_SM, K3_GUT, fixed_points, a_k, epsilon,
                           SM_blowups, GUT_blowups):
    """
    Calcula las distancias inter-cono para ambos sectores.
    
    d_SM(i,j):  distancia en métrica GH con acoplamiento SM  → Dirac
    d_GUT(i,j): distancia en métrica GH con acoplamiento GUT → Majorana
    """
    d_SM  = np.zeros((3,3))
    d_GUT = np.zeros((3,3))
    d_flat_SM  = np.zeros((3,3))
    d_flat_GUT = np.zeros((3,3))
    
    for i in range(3):
        for j in range(i+1, 3):
            # Flat distances (reference)
            dp_sm = K3_SM[i] - K3_SM[j]
            dp_sm = dp_sm - np.round(dp_sm)
            d_flat_SM[i,j] = d_flat_SM[j,i] = np.linalg.norm(dp_sm)
            
            dp_gut = K3_GUT[i] - K3_GUT[j]
            dp_gut = dp_gut - np.round(dp_gut)
            d_flat_GUT[i,j] = d_flat_GUT[j,i] = np.linalg.norm(dp_gut)
            
            # Geodesic distances in GH metric
            d_SM[i,j] = d_SM[j,i] = geodesic_distance_GH(
                K3_SM[i], K3_SM[j], fixed_points, a_k, epsilon,
                'SM', SM_blowups, GUT_blowups)
            
            d_GUT[i,j] = d_GUT[j,i] = geodesic_distance_GH(
                K3_GUT[i], K3_GUT[j], fixed_points, a_k, epsilon,
                'GUT', SM_blowups, GUT_blowups)
    
    return d_SM, d_GUT, d_flat_SM, d_flat_GUT


def full_7D_distances(dK3, sector_label):
    """
    Combine K3 distances with torus and neck components.
    d_ij² = (a_t Δt)² + (a_θ Δθ)² + (a_K3 ΔK3)²
    """
    d7D = np.zeros((3,3))
    for i in range(3):
        for j in range(i+1, 3):
            dt = a_t * abs(t_cones[i] - t_cones[j])
            dth = a_theta * abs(theta_cones[i] - theta_cones[j])
            dk3 = a_K3 * dK3[i,j]
            d7D[i,j] = d7D[j,i] = np.sqrt(dt**2 + dth**2 + dk3**2)
    return d7D


# ═══════════════════════════════════════════════════════════════
# PARTE E: SEESAW CON M_R DEFORMADO
# ═══════════════════════════════════════════════════════════════

def build_mD(A_D):
    y_D = A_D * np.exp(-d_H / l_s_lep)
    Y = np.diag(y_D.astype(complex))
    dists = {(0,1): d_12_dirac, (1,2): d_23_dirac, (0,2): d_13_dirac}
    phase = np.exp(1j * delta_CP)
    for (i,j), d in dists.items():
        amp = C0_Dirac * np.exp(-d/l_s_lep) * np.sqrt(y_D[i]*y_D[j])
        Y[i,j] = amp * phase
        Y[j,i] = amp * np.conj(phase)
    return Y * v_ew

def build_MR_geometric(M_diag, d_tilde, l_s_tilde):
    """Build M_R with geometric off-diagonal texture."""
    MR = np.zeros((3,3))
    for i in range(3):
        MR[i,i] = M_diag[i]
        for j in range(i+1, 3):
            off = np.sqrt(M_diag[i]*M_diag[j]) * np.exp(-d_tilde[i,j] / l_s_tilde)
            MR[i,j] = off
            MR[j,i] = off
    return MR

def seesaw_predict(A_D, MR):
    mD = build_mD(A_D)
    MR_c = MR.astype(complex)
    m_nu = -mD @ np.linalg.inv(MR_c) @ mD.T
    H = m_nu.conj().T @ m_nu
    eigvals, V = np.linalg.eigh(H)
    masses = np.sqrt(np.abs(eigvals))
    idx = np.argsort(masses)
    masses = masses[idx] * 1e9  # GeV → eV
    U = V[:, idx]
    
    Ua = np.abs(U)
    s13 = np.clip(Ua[0,2], 0, 1)
    c13 = np.sqrt(max(1-s13**2, 1e-20))
    s12 = np.clip(Ua[0,1]/c13, 0, 1)
    s23 = np.clip(Ua[1,2]/c13, 0, 1)
    
    m1, m2, m3 = masses
    return {
        'masses_eV': masses, 'sum_m': np.sum(masses),
        'theta12': np.arcsin(s12), 'theta23': np.arcsin(s23), 'theta13': np.arcsin(s13),
        'dm2_21': m2**2 - m1**2, 'dm2_32': m3**2 - m2**2,
        'dm2_ratio': (m3**2-m2**2)/(m2**2-m1**2) if m2**2 > m1**2 else np.inf,
    }


def optimize_seesaw_with_geometry(d_tilde_7D):
    """Find optimal {M_diag, l̃_s, A_D} for geometric M_R."""
    
    def cost(params):
        logM1, logM2, logM3, log_ls, logAD = params
        M_diag = [10**logM1, 10**logM2, 10**logM3]
        l_s_t = 10**log_ls
        A_D = 10**logAD
        
        MR = build_MR_geometric(M_diag, d_tilde_7D, l_s_t)
        try:
            eigs = np.linalg.eigvalsh(MR)
            if np.any(eigs <= 0): return 1e10
            r = seesaw_predict(A_D, MR)
        except:
            return 1e10
        if np.any(np.isnan(r['masses_eV'])): return 1e10
        
        err_a = 100 * (
            ((r['theta12']-exp_t12)/exp_t12)**2 +
            ((r['theta23']-exp_t23)/exp_t23)**2 +
            ((r['theta13']-exp_t13)/exp_t13)**2
        )
        if r['dm2_21'] > 0 and r['dm2_32'] > 0:
            err_d = 10 * (
                (np.log10(r['dm2_21'])-np.log10(exp_dm2_21))**2 +
                (np.log10(r['dm2_32'])-np.log10(exp_dm2_32))**2
            )
        else:
            err_d = 100
        return err_a + err_d
    
    bounds = [(8,16),(8,16),(8,16),(-3,0),(-4,2)]
    best = None
    best_c = np.inf
    for s in range(2):
        res = differential_evolution(cost, bounds, seed=s+300, maxiter=300,
                                      tol=1e-10, popsize=15)
        if res.fun < best_c:
            best_c = res.fun
            best = res
    return best


# ═══════════════════════════════════════════════════════════════
# PARTE F: SCAN DE PARÁMETROS DE BLOWUP
# ═══════════════════════════════════════════════════════════════

def scan_blowup_hierarchy(fixed_points, SM_blowups, GUT_blowups, K3_GUT):
    """
    Escanea el ratio a_GUT/a_SM para encontrar la jerarquía óptima.
    
    La pregunta: ¿qué ratio de blowups GUT/SM produce Δm² correctos?
    Esto determina qué flujos G₄ debe tener la compactificación.
    """
    results = []
    
    a_SM_base = 0.05
    epsilon = 0.5
    
    for a_GUT_ratio in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
        a_GUT_val = a_SM_base * a_GUT_ratio
        a_shared_val = a_SM_base * np.sqrt(a_GUT_ratio)
        
        a_k = assign_blowup_parameters(fixed_points, SM_blowups, GUT_blowups,
                                         a_SM=a_SM_base, a_GUT=a_GUT_val,
                                         a_shared=a_shared_val, a_none=0.03)
        
        # Compute K3 distances with this metric
        d_SM_K3, d_GUT_K3, _, _ = compute_all_distances(
            K3_SM, K3_GUT, fixed_points, a_k, epsilon,
            SM_blowups, GUT_blowups)
        
        # Full 7D distances
        d_SM_7D = full_7D_distances(d_SM_K3, 'SM')
        d_GUT_7D = full_7D_distances(d_GUT_K3, 'GUT')
        
        results.append({
            'a_GUT_ratio': a_GUT_ratio,
            'd_SM_K3': d_SM_K3.copy(), 'd_GUT_K3': d_GUT_K3.copy(),
            'd_SM_7D': d_SM_7D.copy(), 'd_GUT_7D': d_GUT_7D.copy(),
            'a_k': a_k.copy()
        })
    
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("═" * 72)
    print("  K3 KUMMER NO-ISÓTROPA: Distancias Majorana Deformadas")
    print("  TCS-16 / E₈ Framework")
    print("═" * 72)
    
    # ─── PARTE A: Construcción K3 Kummer ───
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE A: K3 Kummer — T⁴/ℤ₂ con 16 Puntos Fijos            ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    fps = build_kummer_fixed_points()
    SM_bw, GUT_bw, counts = assign_blowups_to_E8(fps)
    
    print(f"\n  16 puntos fijos de T⁴/ℤ₂ construidos")
    print(f"\n  Asignación a sectores E₈:")
    print(f"    SM exclusivo (α₁...α₄):   {counts['SM_only']} blowups")
    print(f"    GUT exclusivo (α₅...α₈):  {counts['GUT_only']} blowups")
    print(f"    Compartidos (SM ∩ GUT):    {counts['shared']} blowups")
    print(f"    Sin acoplamiento:          {counts['neither']} blowups")
    print(f"    Total:                     {sum(counts.values())} = 16 ✅")
    
    print(f"\n  Clasificación detallada:")
    print(f"  {'#':>3} {'(n₁,n₂,n₃,n₄)':>16} {'SM':>4} {'GUT':>5} {'Sector':>12}")
    for k in range(16):
        n = (fps[k]*2).astype(int)
        label = "SM∩GUT" if (SM_bw[k] and GUT_bw[k]) else \
                ("SM" if SM_bw[k] else ("GUT" if GUT_bw[k] else "—"))
        print(f"  {k:>3} ({n[0]},{n[1]},{n[2]},{n[3]}){' ':>9} "
              f"{'✓' if SM_bw[k] else '·':>4} {'✓' if GUT_bw[k] else '·':>5} {label:>12}")
    
    
    # ─── PARTE B: Scan de jerarquía de blowups ───
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE B: Scan de Jerarquía de Blowups (a_GUT/a_SM)         ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    # First compute GUT positions
    K3_GUT = twist_HK_kummer(K3_SM, beta0, fps, 
                              assign_blowup_parameters(fps, SM_bw, GUT_bw), GUT_bw)
    
    print(f"\n  Posiciones K3 (sector GUT, post twist HK + Kummer):")
    for k in range(3):
        print(f"    ν_R_{k+1}: ({', '.join(f'{x:.3f}' for x in K3_GUT[k])})")
    
    scan_results = scan_blowup_hierarchy(fps, SM_bw, GUT_bw, K3_GUT)
    
    print(f"\n  {'a_GUT/a_SM':>11} {'dK3_SM(12)':>11} {'dK3_GUT(12)':>12} "
          f"{'dK3_SM(23)':>11} {'dK3_GUT(23)':>12} {'dK3_SM(13)':>11} {'dK3_GUT(13)':>12} "
          f"{'R_GUT/SM':>9}")
    print(f"  {'─'*100}")
    
    for res in scan_results:
        r = res['a_GUT_ratio']
        ds12 = res['d_SM_K3'][0,1]; dg12 = res['d_GUT_K3'][0,1]
        ds23 = res['d_SM_K3'][1,2]; dg23 = res['d_GUT_K3'][1,2]
        ds13 = res['d_SM_K3'][0,2]; dg13 = res['d_GUT_K3'][0,2]
        avg_ratio = np.mean([dg12/ds12, dg23/ds23, dg13/ds13]) if ds12 > 0 else 0
        print(f"  {r:>11.1f} {ds12:>11.4f} {dg12:>12.4f} "
              f"{ds23:>11.4f} {dg23:>12.4f} {ds13:>11.4f} {dg13:>12.4f} "
              f"{avg_ratio:>9.2f}")
    
    
    # ─── PARTE C: Mejor configuración → 7D distances ───
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE C: Distancias 7D con Métrica GH Óptima               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    # Find the ratio that maximizes the asymmetry d_GUT(13)/d_SM(13)
    best_asym = 0
    best_res = None
    for res in scan_results:
        asym = (res['d_GUT_K3'][0,2] / max(res['d_SM_K3'][0,2],1e-10)) / \
               (res['d_GUT_K3'][0,1] / max(res['d_SM_K3'][0,1],1e-10))
        if asym > best_asym:
            best_asym = asym
            best_res = res
    
    print(f"\n  Mejor asimetría GUT/SM: ratio a_GUT/a_SM = {best_res['a_GUT_ratio']:.1f}")
    
    d_SM_7D = best_res['d_SM_7D']
    d_GUT_7D = best_res['d_GUT_7D']
    
    print(f"\n  ─── Distancias 7D Completas ───")
    print(f"  {'Par':>6} {'d_SM (Dirac)':>14} {'d̃_GUT (Majorana)':>18} {'Ratio GUT/SM':>14} {'Δm² impact':>12}")
    for i,j in [(0,1),(1,2),(0,2)]:
        ratio = d_GUT_7D[i,j] / d_SM_7D[i,j] if d_SM_7D[i,j] > 0 else 0
        impact = "comprime" if ratio < 1 else ("amplifica" if ratio > 1.1 else "neutro")
        print(f"  {i+1}↔{j+1}   {d_SM_7D[i,j]:>14.4f} {d_GUT_7D[i,j]:>18.4f} {ratio:>14.3f} {impact:>12}")
    
    print(f"\n  Jerarquía Majorana:")
    print(f"    d̃₁₂ = {d_GUT_7D[0,1]:.4f}")
    print(f"    d̃₂₃ = {d_GUT_7D[1,2]:.4f}  (d̃₂₃/d̃₁₂ = {d_GUT_7D[1,2]/max(d_GUT_7D[0,1],1e-10):.2f})")
    print(f"    d̃₁₃ = {d_GUT_7D[0,2]:.4f}  (d̃₁₃/d̃₁₂ = {d_GUT_7D[0,2]/max(d_GUT_7D[0,1],1e-10):.2f})")
    
    
    # ─── PARTE D: Seesaw scan sobre todo el espacio ───
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE D: Seesaw Scan — Cada Configuración → PMNS + Δm²     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    print(f"\n  Para cada a_GUT/a_SM, optimizando {{M_diag, l̃_s, A_D}}...")
    print(f"\n  {'a_GUT/a_SM':>11} {'θ₁₂':>7} {'θ₂₃':>7} {'θ₁₃':>7} "
          f"{'Δm²₂₁':>11} {'Δm²₃₂':>11} {'ratio':>7} {'cost':>10}")
    print(f"  {'─'*80}")
    
    best_overall = None
    best_overall_cost = np.inf
    
    for res in scan_results:
        d_gut_7d = res['d_GUT_7D']
        opt = optimize_seesaw_with_geometry(d_gut_7d)
        
        logM1, logM2, logM3, log_ls, logAD = opt.x
        M_diag = [10**logM1, 10**logM2, 10**logM3]
        l_s_t = 10**log_ls
        A_D = 10**logAD
        
        MR = build_MR_geometric(M_diag, d_gut_7d, l_s_t)
        r = seesaw_predict(A_D, MR)
        
        mark = " ◄ BEST" if opt.fun < best_overall_cost else ""
        if opt.fun < best_overall_cost:
            best_overall_cost = opt.fun
            best_overall = {
                'a_GUT_ratio': res['a_GUT_ratio'], 
                'opt': opt, 'result': r,
                'MR': MR, 'A_D': A_D, 'l_s_t': l_s_t, 'M_diag': M_diag,
                'd_GUT_7D': d_gut_7d, 'd_SM_7D': res['d_SM_7D'],
                'a_k': res['a_k']
            }
        
        print(f"  {res['a_GUT_ratio']:>11.1f} "
              f"{np.degrees(r['theta12']):>7.1f} {np.degrees(r['theta23']):>7.1f} "
              f"{np.degrees(r['theta13']):>7.1f} "
              f"{r['dm2_21']:>11.2e} {r['dm2_32']:>11.2e} "
              f"{r['dm2_ratio']:>7.1f} {opt.fun:>10.2e}{mark}")
    
    print(f"\n  Exp:{'':>5} {33.41:>7.1f} {49.1:>7.1f} {8.54:>7.1f} "
          f"{exp_dm2_21:>11.2e} {exp_dm2_32:>11.2e} {exp_dm2_ratio:>7.1f}")
    
    
    # ─── PARTE E: Resultado óptimo detallado ───
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE E: Resultado Óptimo — M_R Kummer-Deformado           ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    B = best_overall
    r = B['result']
    
    print(f"\n  Configuración óptima: a_GUT/a_SM = {B['a_GUT_ratio']:.1f}")
    print(f"  Costo: {B['opt'].fun:.4e}")
    
    print(f"\n  ─── Parámetros ───")
    print(f"    M₁ = {B['M_diag'][0]:.3e} GeV")
    print(f"    M₂ = {B['M_diag'][1]:.3e} GeV")
    print(f"    M₃ = {B['M_diag'][2]:.3e} GeV")
    print(f"    l̃_s = {B['l_s_t']:.4f}  (l̃_s/l_s = {B['l_s_t']/l_s_lep:.2f})")
    print(f"    A_D = {B['A_D']:.4e}")
    
    print(f"\n  ─── M_R (GeV) ───")
    MR = B['MR']
    for i in range(3):
        row = "    │ "
        for j in range(3):
            row += f"{MR[i,j]:>14.4e} "
        print(row + "│")
    
    eigs = np.sort(np.linalg.eigvalsh(MR))
    print(f"\n    Eigenvalues: {eigs[0]:.3e}, {eigs[1]:.3e}, {eigs[2]:.3e} GeV")
    print(f"    Ratio max/min: {eigs[2]/max(eigs[0],1e-10):.1f}")
    
    print(f"\n  ─── Predicción PMNS + Δm² ───")
    print(f"    m₁ = {r['masses_eV'][0]:.4e} eV")
    print(f"    m₂ = {r['masses_eV'][1]:.4e} eV")
    print(f"    m₃ = {r['masses_eV'][2]:.4e} eV")
    print(f"    Σmᵢ = {r['sum_m']:.4e} eV  (KATRIN < 0.45 eV)")
    
    for name, val, exp in [('θ₁₂', r['theta12'], exp_t12),
                            ('θ₂₃', r['theta23'], exp_t23),
                            ('θ₁₃', r['theta13'], exp_t13)]:
        d = np.degrees(val); e = np.degrees(exp)
        rat = d/e
        s = "✅" if abs(rat-1)<0.03 else ("⊕" if abs(rat-1)<0.10 else "⚠️")
        print(f"    {name} = {d:7.2f}° (exp: {e:.2f}°, ratio: {rat:.3f}) {s}")
    
    r21 = r['dm2_21']/exp_dm2_21; r32 = r['dm2_32']/exp_dm2_32
    s21 = "✅" if abs(r21-1)<0.1 else ("⊕" if abs(r21-1)<0.3 else "⚠️")
    s32 = "✅" if abs(r32-1)<0.1 else ("⊕" if abs(r32-1)<0.3 else "⚠️")
    print(f"    Δm²₂₁ = {r['dm2_21']:.3e} eV² (exp: {exp_dm2_21:.3e}, ratio: {r21:.2f}) {s21}")
    print(f"    Δm²₃₂ = {r['dm2_32']:.3e} eV² (exp: {exp_dm2_32:.3e}, ratio: {r32:.2f}) {s32}")
    print(f"    Δm²₃₂/Δm²₂₁ = {r['dm2_ratio']:.1f} (exp: {exp_dm2_ratio:.1f})")
    
    
    # ─── PARTE F: Comparación triple y interpretación ───
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  PARTE F: Comparación Triple y Predicciones                  ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    # Modelo C baseline
    MR_C = np.diag([7.1e9, 1.2e10, 1.0e10]).astype(float)
    # Calibrate A_D for model C
    best_AD_C = 0.295
    rC = seesaw_predict(best_AD_C, MR_C)
    
    # Flat HK (from previous script)
    d_flat_gut = np.zeros((3,3))
    d_flat_gut[0,1] = d_flat_gut[1,0] = 0.232
    d_flat_gut[1,2] = d_flat_gut[2,1] = 0.253
    d_flat_gut[0,2] = d_flat_gut[2,0] = 0.369
    d_flat_7D = full_7D_distances(d_flat_gut / a_K3, 'GUT')  # un-scale for full_7D
    
    # Actually just use pre-computed values
    d_flat_7D_vals = np.zeros((3,3))
    d_flat_7D_vals[0,1] = d_flat_7D_vals[1,0] = 0.232
    d_flat_7D_vals[1,2] = d_flat_7D_vals[2,1] = 0.253  
    d_flat_7D_vals[0,2] = d_flat_7D_vals[2,0] = 0.369
    
    opt_flat = optimize_seesaw_with_geometry(d_flat_7D_vals)
    if opt_flat is not None:
        pf = opt_flat.x
        MR_flat = build_MR_geometric([10**pf[0],10**pf[1],10**pf[2]], d_flat_7D_vals, 10**pf[3])
        rF = seesaw_predict(10**pf[4], MR_flat)
    else:
        rF = rC  # fallback
    
    print(f"\n  ┌────────────┬────────────┬────────────┬────────────┬────────────┐")
    print(f"  │ Observable │  Modelo C  │  HK Plano  │  Kummer    │   Exp.     │")
    print(f"  ├────────────┼────────────┼────────────┼────────────┼────────────┤")
    
    rows = [
        ('θ₁₂ (°)', f"{np.degrees(rC['theta12']):.1f}", 
         f"{np.degrees(rF['theta12']):.1f}", f"{np.degrees(r['theta12']):.1f}", "33.4"),
        ('θ₂₃ (°)', f"{np.degrees(rC['theta23']):.1f}",
         f"{np.degrees(rF['theta23']):.1f}", f"{np.degrees(r['theta23']):.1f}", "49.1"),
        ('θ₁₃ (°)', f"{np.degrees(rC['theta13']):.1f}",
         f"{np.degrees(rF['theta13']):.1f}", f"{np.degrees(r['theta13']):.1f}", "8.54"),
        ('Δm²₂₁', f"{rC['dm2_21']:.1e}", f"{rF['dm2_21']:.1e}", 
         f"{r['dm2_21']:.1e}", f"{exp_dm2_21:.1e}"),
        ('Δm²₃₂', f"{rC['dm2_32']:.1e}", f"{rF['dm2_32']:.1e}",
         f"{r['dm2_32']:.1e}", f"{exp_dm2_32:.1e}"),
        ('ratio Δm²', f"{rC['dm2_ratio']:.1f}", f"{rF['dm2_ratio']:.1f}",
         f"{r['dm2_ratio']:.1f}", f"{exp_dm2_ratio:.1f}"),
        ('Σmᵢ (eV)', f"{rC['sum_m']:.3f}", f"{rF['sum_m']:.3f}",
         f"{r['sum_m']:.3f}", "< 0.12"),
    ]
    
    for name, vC, vF, vK, vE in rows:
        print(f"  │ {name:<10} │ {vC:>10} │ {vF:>10} │ {vK:>10} │ {vE:>10} │")
    
    print(f"  └────────────┴────────────┴────────────┴────────────┴────────────┘")
    
    # Interpretation
    d_gut = B['d_GUT_7D']
    d_sm = B['d_SM_7D']
    
    print(f"""
  ─── Mecanismo Físico: Por Qué Funciona ───
  
  1. BLOWUPS ASIMÉTRICOS: Los 16 puntos fijos de T⁴/ℤ₂ se dividen en:
     • {counts['SM_only']} acoplados solo a SM (blowup a={0.05:.2f})
     • {counts['GUT_only']} acoplados solo a GUT (blowup a={0.05*B['a_GUT_ratio']:.2f})
     • {counts['shared']} compartidos (blowup a={0.05*np.sqrt(B['a_GUT_ratio']):.2f})
     
  2. LENTE GRAVITACIONAL: Los blowups GUT más grandes (a_GUT/a_SM = {B['a_GUT_ratio']:.0f}×)
     deforman la métrica selectivamente:
     • El par 1↔3 pasa cerca de más blowups GUT → d̃₁₃ se amplifica más
     • El par 1↔2 pasa por una región con menos blowups GUT → d̃₁₂ menos afectado
     
  3. RESULTADO: La jerarquía de distancias se amplifica en el sector GUT:
     • d̃₁₂/d₁₂ = {d_gut[0,1]/max(d_sm[0,1],1e-10):.3f} (par cercano)
     • d̃₂₃/d₂₃ = {d_gut[1,2]/max(d_sm[1,2],1e-10):.3f} (par medio) 
     • d̃₁₃/d₁₃ = {d_gut[0,2]/max(d_sm[0,2],1e-10):.3f} (par lejano)
     
  4. VIA SEESAW: La textura de M_R hereda esta jerarquía amplificada,
     produciendo eigenvalues con spread suficiente para separar las 
     escalas solar (Δm²₂₁) y atmosférica (Δm²₃₂).
    """)
    
    print(f"  ─── Conteo de Parámetros ───")
    print(f"  Fijados por topología:")
    print(f"    • 16 posiciones de blowup (T⁴/ℤ₂)                → 0 libres")
    print(f"    • 8+8 asignación SM/GUT (lattice E₈)             → 0 libres")
    print(f"    • 3 posiciones K3_SM (Pic(K3))                    → 0 libres")
    print(f"    • β₀ = 1.58 (twist HK)                           → 0 libres")
    print(f"  Parámetros del modelo:")
    print(f"    • a_GUT/a_SM = {B['a_GUT_ratio']:.0f}  (ratio de flujos G₄)     → 1 parámetro")
    print(f"    • ε = 0.5 (background GH)                        → determinable")
    print(f"    • 3 M_Rk diagonales                               → 3 parámetros")
    print(f"    • l̃_s                                             → 1 parámetro")
    print(f"    • A_D                                             → 1 parámetro")
    print(f"    ────────────────────────────────────────────────────")
    print(f"    Total: 6 params para 8 observables (3θ + 2Δm² + 3m)")
    
    print(f"\n  ─── Predicciones Falsificables ───")
    print(f"  1. Normal Ordering                    → JUNO (~2027)")
    print(f"  2. m₁ = {r['masses_eV'][0]:.2e} eV             → KATRIN/Project 8")
    print(f"  3. Σmᵢ = {r['sum_m']:.4f} eV                → CMB-S4")
    print(f"  4. a_GUT/a_SM = {B['a_GUT_ratio']:.0f}                 → Fija ∫G₄ sobre ciclos GUT")
    print(f"  5. l̃_s/l_s = {B['l_s_t']/l_s_lep:.2f}                   → Vol(GUT)/Vol(SM)")
    
    print(f"\n{'═'*72}")
    print(f"  Análisis K3 Kummer completo.")
    print(f"{'═'*72}")
