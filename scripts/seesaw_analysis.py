#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  SEESAW TYPE-I: Análisis de Δm² y Refinamiento de M_R
  Framework TCS-16 / E₈ — Modelo C (Etapas 8–9)
═══════════════════════════════════════════════════════════════════
  
  Objetivo: Diagnosticar por qué los Δm² difieren ~10× del experimental
            y encontrar la estructura de M_R que los corrige sin
            destruir los ángulos PMNS (~1%).

  Diego Santana S. — Marzo 2026
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 1: PARÁMETROS DEL MODELO C (de Etapas 8-9)
# ═══════════════════════════════════════════════════════════════

# --- Constantes fundamentales ---
v_H = 246.22        # GeV, Higgs VEV
v_ew = v_H / np.sqrt(2)  # = 174.1 GeV

# --- Distancias geodésicas (Lattice E₈ + métrica anisótropa) ---
# Inter-cono (off-diagonal): fijadas por E₈
# Derivadas de Table 9, Etapas 8-9: A = exp(-d/l_s)
d_12 = 0.166   # Distancia cono 1 ↔ cono 2 (más cercanos)
d_23 = 0.343   # Distancia cono 2 ↔ cono 3
d_13 = 0.343   # Distancia cono 1 ↔ cono 3 (≈ d_23, de §7.1)

# Cono-a-Higgs (diagonal): del Compendio §16
# Asignación: Cono2→1ªgen, Cono1→2ªgen, Cono3→3ªgen
d_H = np.array([0.561, 0.347, 0.198])  # d_k para gen k=1,2,3

# --- Longitud de cuerda sectorial ---
l_s_lep = 0.044    # sector leptónico (universal para charged + Dirac)

# --- Parámetros del instanton (Modelo C) ---
C0_lep   = 13.8    # prefactor off-diagonal, leptón cargado
C0_Dirac = 7348.0  # prefactor off-diagonal, neutrino Dirac
delta_CP = np.pi    # 180° (del Modelo C; produce J≈0)

# --- M_R actual del Modelo C (GeV) ---
M_R_modelC = np.diag([7.1e9, 1.2e10, 1.0e10])

# --- Masas experimentales de leptones cargados (GeV) ---
m_e   = 5.11e-4
m_mu  = 0.1057     # PDG: 105.66 MeV
m_tau = 1.777      # PDG

# --- Datos experimentales de neutrinos (NuFIT 5.3, NO) ---
exp_theta12 = np.radians(33.41)
exp_theta23 = np.radians(49.1)
exp_theta13 = np.radians(8.54)
exp_dm2_21  = 7.41e-5   # eV² (solar)
exp_dm2_32  = 2.507e-3  # eV² (atmosférico, NO)
exp_dm2_ratio = exp_dm2_32 / exp_dm2_21  # ≈ 33.8


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 2: FUNCIONES DE CONSTRUCCIÓN
# ═══════════════════════════════════════════════════════════════

def instanton_amplitude(d_ij, l_s):
    """Amplitud de instanton M2: A = exp(-d/l_s)"""
    return np.exp(-d_ij / l_s)

def build_yukawa_matrix(y_diag, C0, l_s, delta, d12, d23, d13):
    """
    Construye la matriz de Yukawa 3×3:
      Y_ii = y_diag[i]  (diagonal, de supresión geodésica)
      Y_ij = C0 * exp(-d_ij/l_s) * sqrt(y_i*y_j) * e^(iδ)  (off-diagonal)
    """
    Y = np.diag(y_diag.astype(complex))
    
    dists = {(0,1): d12, (1,2): d23, (0,2): d13}
    phase = np.exp(1j * delta)
    
    for (i,j), d in dists.items():
        amp = C0 * instanton_amplitude(d, l_s) * np.sqrt(y_diag[i] * y_diag[j])
        Y[i,j] = amp * phase
        Y[j,i] = amp * np.conj(phase)  # Hermítica para Yukawa
    
    return Y

def build_mD_from_params(A_D, l_s, C0_D, delta, d_H_arr, d12, d23, d13):
    """
    Construye m_D (Dirac neutrino mass matrix) desde parámetros geométricos.
    
    A_D: escala global de Yukawa Dirac (parámetro libre que fija masa absoluta)
    Diagonal: y_D_i = A_D * exp(-d_H_i / l_s)
    Off-diagonal: instanton con C0_Dirac
    """
    # Yukawas diagonales desde supresión geodésica
    y_D = A_D * np.exp(-d_H_arr / l_s)
    
    # Matriz de Yukawa completa
    Y_D = build_yukawa_matrix(y_D, C0_D, l_s, delta, d12, d23, d13)
    
    # m_D = Y_D × v/√2
    mD = Y_D * v_ew
    
    return mD

def seesaw_type1(mD, MR):
    """
    Seesaw type-I: m_ν = -m_D × M_R⁻¹ × m_D^T
    
    mD: 3×3 Dirac mass matrix (complex)
    MR: 3×3 Majorana mass matrix (symmetric)
    Returns: m_nu (3×3 complex symmetric)
    """
    MR_inv = np.linalg.inv(MR)
    m_nu = -mD @ MR_inv @ mD.T
    return m_nu

def diagonalize_neutrino_mass(m_nu):
    """
    Diagonaliza la matriz de masa de neutrinos (simétrica compleja).
    
    Para una matriz simétrica compleja M, usamos la descomposición Takagi:
    M = U* × D × U†, donde D es diagonal real no-negativa.
    
    Returns: masses (3,), U_PMNS (3×3 unitary)
    """
    # Para seesaw, m_nu es simétrica compleja
    # Usamos SVD: m_nu = U × Σ × V†
    # Para matriz simétrica: U = V* (Takagi)
    
    # Método práctico: diagonalizar m_nu† × m_nu (Hermítica)
    H = m_nu.conj().T @ m_nu
    eigenvalues, V = np.linalg.eigh(H)
    
    # Masas = raíz de eigenvalues (en eV)
    masses_sq = np.abs(eigenvalues)
    masses = np.sqrt(masses_sq)
    
    # Ordenar por masa (normal ordering: m1 < m2 < m3)
    idx = np.argsort(masses)
    masses = masses[idx]
    U = V[:, idx]
    
    return masses, U

def extract_pmns_angles(U):
    """
    Extrae los ángulos de mezcla de la matriz PMNS.
    Parametrización estándar PDG:
      s13 = |U_e3|
      s12 = |U_e2| / sqrt(1 - |U_e3|²)
      s23 = |U_μ3| / sqrt(1 - |U_e3|²)
    """
    U_abs = np.abs(U)
    
    s13 = U_abs[0, 2]
    c13 = np.sqrt(1 - s13**2) if s13 < 1 else 1e-10
    
    s12 = U_abs[0, 1] / c13 if c13 > 1e-10 else 0
    s23 = U_abs[1, 2] / c13 if c13 > 1e-10 else 0
    
    # Clamp para evitar errores numéricos en arcsin
    s12 = np.clip(s12, 0, 1)
    s23 = np.clip(s23, 0, 1)
    s13 = np.clip(s13, 0, 1)
    
    theta12 = np.arcsin(s12)
    theta23 = np.arcsin(s23)
    theta13 = np.arcsin(s13)
    
    return theta12, theta23, theta13

def compute_dm2(masses_eV):
    """
    Calcula Δm² desde las masas de neutrinos.
    masses_eV: array de 3 masas en eV, ordenadas m1 < m2 < m3
    
    Returns: dm2_21 (solar), dm2_32 (atmosférico)
    """
    m1, m2, m3 = masses_eV
    dm2_21 = m2**2 - m1**2
    dm2_32 = m3**2 - m2**2
    dm2_31 = m3**2 - m1**2
    return dm2_21, dm2_32, dm2_31

def full_seesaw_prediction(A_D, MR_diag, l_s=l_s_lep, C0_D=C0_Dirac, 
                            delta=delta_CP, verbose=False):
    """
    Pipeline completo: parámetros → predicciones PMNS + Δm².
    
    A_D: escala Yukawa Dirac
    MR_diag: array [M1, M2, M3] en GeV (diagonal de M_R)
    """
    # 1. Construir m_D
    mD = build_mD_from_params(A_D, l_s, C0_D, delta, d_H, d_12, d_23, d_13)
    
    # 2. M_R
    MR = np.diag(np.array(MR_diag, dtype=complex))
    
    # 3. Seesaw
    m_nu = seesaw_type1(mD, MR)
    
    # 4. Diagonalizar (convertir a eV: m_nu está en GeV)
    masses_GeV, U = diagonalize_neutrino_mass(m_nu)
    masses_eV = masses_GeV * 1e9  # GeV → eV
    
    # 5. Ángulos PMNS
    theta12, theta23, theta13 = extract_pmns_angles(U)
    
    # 6. Δm²
    dm2_21, dm2_32, dm2_31 = compute_dm2(masses_eV)
    
    result = {
        'masses_eV': masses_eV,
        'theta12': theta12, 'theta23': theta23, 'theta13': theta13,
        'dm2_21': dm2_21, 'dm2_32': dm2_32, 'dm2_31': dm2_31,
        'dm2_ratio': dm2_32 / dm2_21 if dm2_21 > 0 else np.inf,
        'mD': mD, 'm_nu': m_nu, 'U': U
    }
    
    if verbose:
        print_prediction(result)
    
    return result

def print_prediction(r, label=""):
    """Imprime predicción formateada."""
    if label:
        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
    
    print(f"\n  Masas de neutrinos:")
    print(f"    m₁ = {r['masses_eV'][0]:.4e} eV")
    print(f"    m₂ = {r['masses_eV'][1]:.4e} eV")
    print(f"    m₃ = {r['masses_eV'][2]:.4e} eV")
    print(f"    Σmᵢ = {np.sum(r['masses_eV']):.4e} eV  (KATRIN < 0.45 eV)")
    
    print(f"\n  Ángulos PMNS:")
    for name, val, exp in [
        ('θ₁₂', r['theta12'], exp_theta12),
        ('θ₂₃', r['theta23'], exp_theta23),
        ('θ₁₃', r['theta13'], exp_theta13)
    ]:
        deg = np.degrees(val)
        exp_deg = np.degrees(exp)
        ratio = deg / exp_deg if exp_deg > 0 else 0
        status = "✅" if abs(ratio - 1) < 0.01 else ("⊕" if abs(ratio - 1) < 0.30 else "⚠️")
        print(f"    {name} = {deg:7.2f}°  (exp: {exp_deg:.2f}°, ratio: {ratio:.3f}) {status}")
    
    print(f"\n  Diferencias de masa cuadradas:")
    print(f"    Δm²₂₁ = {r['dm2_21']:.3e} eV²  (exp: {exp_dm2_21:.3e})")
    r21 = r['dm2_21'] / exp_dm2_21 if exp_dm2_21 > 0 else 0
    print(f"             ratio = {r21:.2f}×")
    
    print(f"    Δm²₃₂ = {r['dm2_32']:.3e} eV²  (exp: {exp_dm2_32:.3e})")
    r32 = r['dm2_32'] / exp_dm2_32 if exp_dm2_32 > 0 else 0
    print(f"             ratio = {r32:.2f}×")
    
    print(f"    Δm²₃₂/Δm²₂₁ = {r['dm2_ratio']:.1f}  (exp: {exp_dm2_ratio:.1f})")


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 3: FASE 1 — Reconstrucción del Modelo C
# ═══════════════════════════════════════════════════════════════

def phase1_reconstruct_modelC():
    """
    Reconstruye el Modelo C y diagnostica el problema de Δm².
    Calibra A_D para reproducir la escala correcta de masas de neutrinos.
    """
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  FASE 1: Reconstrucción del Modelo C — Diagnóstico de Δm²   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    MR_diag = [7.1e9, 1.2e10, 1.0e10]
    
    # --- Paso 1: Estructura de m_D ---
    print("\n─── Estructura del instanton ───")
    print(f"  Distancias inter-cono: d₁₂={d_12}, d₂₃={d_23}, d₁₃={d_13}")
    print(f"  Distancias cono→Higgs: d₁={d_H[0]}, d₂={d_H[1]}, d₃={d_H[2]}")
    print(f"  l_s = {l_s_lep}")
    print(f"  C₀_Dirac = {C0_Dirac}")
    
    # Supresiones diagonales
    supp = np.exp(-d_H / l_s_lep)
    print(f"\n  Supresiones diagonales exp(-d_k/l_s):")
    for k in range(3):
        print(f"    Gen {k+1}: exp(-{d_H[k]:.3f}/{l_s_lep}) = {supp[k]:.3e}")
    print(f"    Ratio sup₃/sup₁ = {supp[2]/supp[0]:.0f}× (jerarquía diagonal)")
    
    # Amplitudes off-diagonal
    for label, d in [("1↔2", d_12), ("2↔3", d_23), ("1↔3", d_13)]:
        A = instanton_amplitude(d, l_s_lep)
        print(f"    A({label}) = exp(-{d}/{l_s_lep}) = {A:.3e}")
    
    # --- Paso 2: Calibrar A_D para obtener escala correcta ---
    print("\n─── Calibración de A_D (escala Dirac) ───")
    
    # Barrido de A_D para encontrar el que da Σm_ν ~ 0.06-0.1 eV
    # (escala típica para NO con m1 ~ 0)
    best_A_D = None
    best_dm2_err = np.inf
    
    for log_A_D in np.linspace(-4, 2, 2000):
        A_D = 10**log_A_D
        try:
            r = full_seesaw_prediction(A_D, MR_diag)
            if np.any(np.isnan(r['masses_eV'])) or np.any(r['masses_eV'] < 0):
                continue
            # Minimizar error en Δm²₃₂ (escala atmosférica, mejor determinada)
            err = abs(np.log10(abs(r['dm2_32']) / exp_dm2_32))
            if err < best_dm2_err:
                best_dm2_err = err
                best_A_D = A_D
        except:
            continue
    
    print(f"  A_D óptimo (calibrado a Δm²₃₂): {best_A_D:.4e}")
    print(f"  Error log₁₀ en Δm²₃₂: {best_dm2_err:.3f}")
    
    # --- Paso 3: Predicción completa con M_R del Modelo C ---
    r_modelC = full_seesaw_prediction(best_A_D, MR_diag)
    print_prediction(r_modelC, "MODELO C RECONSTRUIDO (M_R original)")
    
    # --- Paso 4: Diagnóstico del problema ---
    print("\n─── DIAGNÓSTICO ───")
    ratio_21 = r_modelC['dm2_21'] / exp_dm2_21
    ratio_32 = r_modelC['dm2_32'] / exp_dm2_32
    ratio_dm2 = r_modelC['dm2_ratio']
    
    print(f"\n  El ratio Δm²₃₂/Δm²₂₁ predicho = {ratio_dm2:.1f}")
    print(f"  El ratio experimental          = {exp_dm2_ratio:.1f}")
    
    if abs(ratio_dm2 - exp_dm2_ratio) > 5:
        print(f"\n  ⚠️  El ratio difiere por {abs(ratio_dm2/exp_dm2_ratio - 1)*100:.0f}%")
        print(f"  → M_R casi-degenerado (M₁:M₂:M₃ ≈ 1:1.69:1.41) no genera")
        print(f"    suficiente separación entre escalas solar y atmosférica.")
    
    # Mostrar estructura de m_D
    mD = r_modelC['mD']
    print(f"\n  Estructura de m_D (GeV):")
    print(f"  ┌                                           ┐")
    for i in range(3):
        row = "  │ "
        for j in range(3):
            val = mD[i,j]
            if np.iscomplex(val) and abs(val.imag) > 1e-20:
                row += f"{val.real:+10.3e}{val.imag:+.1e}j "
            else:
                row += f"{val.real:+13.4e} "
        row += "│"
        print(row)
    print(f"  └                                           ┘")
    
    # Ratios off-diag/diag
    print(f"\n  Ratios off-diagonal / diagonal:")
    for i, j in [(0,1), (1,2), (0,2)]:
        r_ij = abs(mD[i,j]) / max(abs(mD[i,i]), abs(mD[j,j]))
        print(f"    |m_D({i+1},{j+1})| / max(diag) = {r_ij:.2f}")
    
    return best_A_D, r_modelC


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 4: FASE 2 — Problema Inverso del Seesaw
# ═══════════════════════════════════════════════════════════════

def phase2_inverse_seesaw(A_D_ref):
    """
    Problema inverso: dado m_D (fijo), encontrar M_R que reproduzca
    tanto los ángulos PMNS como los Δm² experimentales.
    """
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  FASE 2: Problema Inverso — M_R Target                      ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    # Función de costo: ángulos + Δm²
    def cost_MR(params):
        """
        params: [log10(M1), log10(M2), log10(M3), log10(A_D)]
        Minimiza error combinado en ángulos PMNS + Δm².
        """
        logM1, logM2, logM3, logAD = params
        MR_diag = [10**logM1, 10**logM2, 10**logM3]
        A_D = 10**logAD
        
        try:
            r = full_seesaw_prediction(A_D, MR_diag)
        except:
            return 1e10
        
        if np.any(np.isnan(r['masses_eV'])):
            return 1e10
        
        # Penalización por ángulos (peso alto: queremos preservar ~1%)
        w_angle = 100.0
        err_angles = (
            ((r['theta12'] - exp_theta12) / exp_theta12)**2 +
            ((r['theta23'] - exp_theta23) / exp_theta23)**2 +
            ((r['theta13'] - exp_theta13) / exp_theta13)**2
        )
        
        # Penalización por Δm²
        w_dm2 = 10.0
        if r['dm2_21'] > 0 and r['dm2_32'] > 0:
            err_dm2 = (
                (np.log10(r['dm2_21']) - np.log10(exp_dm2_21))**2 +
                (np.log10(r['dm2_32']) - np.log10(exp_dm2_32))**2
            )
        else:
            err_dm2 = 100
        
        # Penalización por masa total (KATRIN bound)
        w_mass = 1.0
        sum_m = np.sum(r['masses_eV'])
        err_mass = max(0, sum_m - 0.45)**2 * 1000  # Penalizar si > 0.45 eV
        
        return w_angle * err_angles + w_dm2 * err_dm2 + w_mass * err_mass
    
    # --- Optimización con Differential Evolution (global) ---
    print("\n─── Optimización global (Differential Evolution) ───")
    print("  Buscando M_R que minimice error en ángulos + Δm²...")
    print("  Espacio de búsqueda: M_Ri ∈ [10⁷, 10¹⁶] GeV, A_D ∈ [10⁻⁴, 10²]")
    
    bounds = [
        (7, 16),    # log10(M1)
        (7, 16),    # log10(M2)
        (7, 16),    # log10(M3)
        (-4, 2),    # log10(A_D)
    ]
    
    # Múltiples intentos para mayor robustez
    best_result = None
    best_cost = np.inf
    
    for seed in range(5):
        result = differential_evolution(
            cost_MR, bounds, 
            seed=seed + 42,
            maxiter=1000,
            tol=1e-12,
            polish=True,
            popsize=25,
            mutation=(0.5, 1.5),
            recombination=0.9
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
    
    logM1, logM2, logM3, logAD = best_result.x
    MR_target = [10**logM1, 10**logM2, 10**logM3]
    A_D_target = 10**logAD
    
    print(f"\n  Costo mínimo: {best_cost:.4e}")
    print(f"\n  M_R TARGET encontrado:")
    print(f"    M₁ = {MR_target[0]:.3e} GeV  (log₁₀ = {logM1:.2f})")
    print(f"    M₂ = {MR_target[1]:.3e} GeV  (log₁₀ = {logM2:.2f})")
    print(f"    M₃ = {MR_target[2]:.3e} GeV  (log₁₀ = {logM3:.2f})")
    print(f"    A_D = {A_D_target:.4e}")
    
    print(f"\n  Jerarquía M_R:")
    M_sorted = sorted(MR_target)
    print(f"    M₂/M₁ = {M_sorted[1]/M_sorted[0]:.1f}")
    print(f"    M₃/M₁ = {M_sorted[2]/M_sorted[0]:.1f}")
    print(f"    M₃/M₂ = {M_sorted[2]/M_sorted[1]:.1f}")
    
    # Predicción con M_R target
    r_target = full_seesaw_prediction(A_D_target, MR_target)
    print_prediction(r_target, "PREDICCIÓN CON M_R TARGET")
    
    # --- Comparación lado a lado ---
    print("\n─── COMPARACIÓN: Modelo C vs M_R Target ───")
    print(f"  {'Parámetro':<20} {'Modelo C':>15} {'M_R Target':>15} {'Experimental':>15}")
    print(f"  {'─'*65}")
    
    r_C = full_seesaw_prediction(A_D_ref, [7.1e9, 1.2e10, 1.0e10])
    
    for name, val_C, val_T, val_E in [
        ('θ₁₂ (°)', np.degrees(r_C['theta12']), np.degrees(r_target['theta12']), 33.41),
        ('θ₂₃ (°)', np.degrees(r_C['theta23']), np.degrees(r_target['theta23']), 49.1),
        ('θ₁₃ (°)', np.degrees(r_C['theta13']), np.degrees(r_target['theta13']), 8.54),
        ('Δm²₂₁ (eV²)', r_C['dm2_21'], r_target['dm2_21'], exp_dm2_21),
        ('Δm²₃₂ (eV²)', r_C['dm2_32'], r_target['dm2_32'], exp_dm2_32),
        ('Δm²₃₂/Δm²₂₁', r_C['dm2_ratio'], r_target['dm2_ratio'], exp_dm2_ratio),
    ]:
        if isinstance(val_E, float) and val_E < 0.01:
            print(f"  {name:<20} {val_C:>15.3e} {val_T:>15.3e} {val_E:>15.3e}")
        else:
            print(f"  {name:<20} {val_C:>15.2f} {val_T:>15.2f} {val_E:>15.2f}")
    
    return MR_target, A_D_target, r_target


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 5: FASE 3 — Sensibilidad y Estructura de M_R
# ═══════════════════════════════════════════════════════════════

def phase3_sensitivity(A_D_ref, MR_target):
    """
    Análisis de sensibilidad: cómo varían Δm² y ángulos
    al modificar la jerarquía de M_R.
    """
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  FASE 3: Análisis de Sensibilidad de M_R                    ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    # --- 3a: Barrido de jerarquía M₃/M₁ con M₂ fijo ---
    print("\n─── 3a: Barrido M₃/M₁ (escala fija = media geométrica) ───")
    
    M_scale = np.sqrt(MR_target[0] * MR_target[2])  # escala de referencia
    
    print(f"\n  {'M₃/M₁':>8} {'θ₁₂(°)':>8} {'θ₂₃(°)':>8} {'θ₁₃(°)':>8} "
          f"{'Δm²₂₁':>12} {'Δm²₃₂':>12} {'Ratio':>8}")
    print(f"  {'─'*72}")
    
    ratios_31 = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]
    results_scan = []
    
    for ratio in ratios_31:
        # Fijar M₃/M₁ = ratio, M₂ = media geométrica
        M1 = M_scale / np.sqrt(ratio)
        M3 = M_scale * np.sqrt(ratio)
        M2 = M_scale
        
        # Recalibrar A_D para cada configuración
        best_AD = A_D_ref
        best_err = np.inf
        for logAD in np.linspace(np.log10(A_D_ref) - 2, np.log10(A_D_ref) + 2, 500):
            AD = 10**logAD
            try:
                r = full_seesaw_prediction(AD, [M1, M2, M3])
                err = abs(np.log10(abs(r['dm2_32'])) - np.log10(exp_dm2_32))
                if err < best_err:
                    best_err = err
                    best_AD = AD
            except:
                continue
        
        r = full_seesaw_prediction(best_AD, [M1, M2, M3])
        results_scan.append((ratio, r))
        
        t12 = np.degrees(r['theta12'])
        t23 = np.degrees(r['theta23'])
        t13 = np.degrees(r['theta13'])
        
        status = ""
        if abs(r['dm2_ratio'] / exp_dm2_ratio - 1) < 0.3:
            status = " ◄─ Δm² ratio OK"
        
        print(f"  {ratio:>8.0f} {t12:>8.2f} {t23:>8.2f} {t13:>8.2f} "
              f"{r['dm2_21']:>12.3e} {r['dm2_32']:>12.3e} {r['dm2_ratio']:>8.1f}{status}")
    
    # --- 3b: Barrido 2D: M₃/M₁ vs M₂/M₁ ---
    print("\n─── 3b: Mapa 2D de Δm²₃₂/Δm²₂₁ en espacio (M₂/M₁, M₃/M₁) ───")
    print("       (Buscando la zona donde el ratio ≈ 33.8)")
    
    M1_ref = MR_target[0]
    ratios_21 = [1.0, 2.0, 5.0, 10.0, 30.0, 50.0]
    ratios_31_2d = [1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 500.0]
    
    print(f"\n  {'':>10}", end="")
    for r31 in ratios_31_2d:
        print(f" {'M₃/M₁='+str(int(r31)):>10}", end="")
    print()
    
    for r21 in ratios_21:
        print(f"  M₂/M₁={int(r21):>2}", end="")
        for r31 in ratios_31_2d:
            M1 = M1_ref
            M2 = M1 * r21
            M3 = M1 * r31
            
            # Quick calibration
            best_AD = A_D_ref
            best_err = np.inf
            for logAD in np.linspace(-3, 2, 200):
                AD = 10**logAD
                try:
                    r = full_seesaw_prediction(AD, [M1, M2, M3])
                    err = abs(np.log10(max(abs(r['dm2_32']), 1e-30)) - np.log10(exp_dm2_32))
                    if err < best_err:
                        best_err = err
                        best_AD = AD
                except:
                    continue
            
            r = full_seesaw_prediction(best_AD, [M1, M2, M3])
            ratio_val = r['dm2_ratio'] if np.isfinite(r['dm2_ratio']) else 0
            
            marker = "  ★" if abs(ratio_val / exp_dm2_ratio - 1) < 0.3 else "   "
            print(f" {ratio_val:>7.1f}{marker}", end="")
        print()
    
    print(f"\n  ★ = ratio dentro de ±30% del experimental ({exp_dm2_ratio:.1f})")
    
    return results_scan


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 6: FASE 4 — M_R Off-Diagonal (textura completa)
# ═══════════════════════════════════════════════════════════════

def phase4_offdiagonal_MR(A_D_ref):
    """
    Explora M_R no-diagonal: (M_R)_ij = M_scale × exp(-d̃_ij / l̃_s)
    donde d̃_ij son distancias entre ciclos co-asociativos.
    """
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  FASE 4: M_R Off-Diagonal (Textura Geométrica)              ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    def cost_MR_offdiag(params):
        """
        params: [logM1, logM2, logM3, logM12, logM23, logM13, logAD]
        M_R = symmetric matrix with off-diagonal entries
        """
        logM1, logM2, logM3, logM12, logM23, logM13, logAD = params
        
        MR = np.array([
            [10**logM1,  10**logM12, 10**logM13],
            [10**logM12, 10**logM2,  10**logM23],
            [10**logM13, 10**logM23, 10**logM3 ]
        ], dtype=complex)
        
        # Verificar que M_R es positiva definida
        eigvals = np.linalg.eigvalsh(MR.real)
        if np.any(eigvals <= 0):
            return 1e10
        
        A_D = 10**logAD
        mD = build_mD_from_params(A_D, l_s_lep, C0_Dirac, delta_CP, d_H, d_12, d_23, d_13)
        
        try:
            m_nu = seesaw_type1(mD, MR)
            masses_GeV, U = diagonalize_neutrino_mass(m_nu)
            masses_eV = masses_GeV * 1e9
        except:
            return 1e10
        
        if np.any(np.isnan(masses_eV)):
            return 1e10
        
        theta12, theta23, theta13 = extract_pmns_angles(U)
        dm2_21, dm2_32, _ = compute_dm2(masses_eV)
        
        err_angles = 100 * (
            ((theta12 - exp_theta12) / exp_theta12)**2 +
            ((theta23 - exp_theta23) / exp_theta23)**2 +
            ((theta13 - exp_theta13) / exp_theta13)**2
        )
        
        if dm2_21 > 0 and dm2_32 > 0:
            err_dm2 = 10 * (
                (np.log10(dm2_21) - np.log10(exp_dm2_21))**2 +
                (np.log10(dm2_32) - np.log10(exp_dm2_32))**2
            )
        else:
            err_dm2 = 100
        
        return err_angles + err_dm2
    
    print("\n  Optimizando M_R simétrico 3×3 (6 entradas + A_D)...")
    
    bounds_offdiag = [
        (8, 16), (8, 16), (8, 16),   # diagonal
        (5, 15), (5, 15), (5, 15),    # off-diagonal
        (-4, 2),                       # A_D
    ]
    
    best_result = None
    best_cost = np.inf
    
    for seed in range(5):
        result = differential_evolution(
            cost_MR_offdiag, bounds_offdiag,
            seed=seed + 100,
            maxiter=800,
            tol=1e-12,
            polish=True,
            popsize=30
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
    
    p = best_result.x
    MR_full = np.array([
        [10**p[0], 10**p[3], 10**p[5]],
        [10**p[3], 10**p[1], 10**p[4]],
        [10**p[5], 10**p[4], 10**p[2]]
    ], dtype=complex)
    A_D_opt = 10**p[6]
    
    print(f"\n  Costo mínimo: {best_cost:.4e}")
    print(f"\n  M_R Off-Diagonal TARGET (GeV):")
    print(f"  ┌                                              ┐")
    for i in range(3):
        row = "  │ "
        for j in range(3):
            row += f"{MR_full[i,j].real:>12.3e} "
        row += "│"
        print(row)
    print(f"  └                                              ┘")
    
    # Eigenvalues de M_R
    MR_eigvals = np.linalg.eigvalsh(MR_full.real)
    print(f"\n  Eigenvalues de M_R: {MR_eigvals[0]:.3e}, {MR_eigvals[1]:.3e}, {MR_eigvals[2]:.3e} GeV")
    print(f"  Ratio max/min: {MR_eigvals[2]/MR_eigvals[0]:.1f}")
    
    # Predicción
    mD = build_mD_from_params(A_D_opt, l_s_lep, C0_Dirac, delta_CP, d_H, d_12, d_23, d_13)
    m_nu = seesaw_type1(mD, MR_full)
    masses_GeV, U = diagonalize_neutrino_mass(m_nu)
    masses_eV = masses_GeV * 1e9
    theta12, theta23, theta13 = extract_pmns_angles(U)
    dm2_21, dm2_32, dm2_31 = compute_dm2(masses_eV)
    
    r_offdiag = {
        'masses_eV': masses_eV,
        'theta12': theta12, 'theta23': theta23, 'theta13': theta13,
        'dm2_21': dm2_21, 'dm2_32': dm2_32, 'dm2_31': dm2_31,
        'dm2_ratio': dm2_32 / dm2_21 if dm2_21 > 0 else np.inf,
        'mD': mD, 'm_nu': m_nu, 'U': U
    }
    
    print_prediction(r_offdiag, "PREDICCIÓN CON M_R OFF-DIAGONAL")
    
    # Interpretación geométrica de las entradas off-diagonal
    print("\n─── Interpretación Geométrica ───")
    for (i,j), label in [((0,1), "1↔2"), ((1,2), "2↔3"), ((0,2), "1↔3")]:
        M_offdiag = MR_full[i,j].real
        M_diag_avg = np.sqrt(MR_full[i,i].real * MR_full[j,j].real)
        ratio_od = M_offdiag / M_diag_avg
        # Si viene de instanton M5: (M_R)_ij ~ M_scale × exp(-d̃_ij / l̃_s)
        # → d̃_ij / l̃_s = -ln(ratio)
        if ratio_od > 0 and ratio_od < 1:
            d_tilde = -np.log(ratio_od)
            print(f"  (M_R)_{label} / √(M_{i+1}M_{j+1}) = {ratio_od:.3e}  →  d̃/l̃_s ≈ {d_tilde:.2f}")
        else:
            print(f"  (M_R)_{label} / √(M_{i+1}M_{j+1}) = {ratio_od:.3e}")
    
    return MR_full, A_D_opt


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 7: FASE 5 — Resumen y Predicciones Falsificables
# ═══════════════════════════════════════════════════════════════

def phase5_summary(r_modelC, MR_target, r_target, MR_offdiag, A_D_ref, A_D_target):
    """Resumen comparativo y predicciones."""
    
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  FASE 5: Resumen Comparativo                                ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    print(f"\n  ┌──────────────┬──────────────┬──────────────┬──────────────┐")
    print(f"  │ {'Observable':<12} │ {'Modelo C':<12} │ {'M_R Target':<12} │ {'Experiment.':<12} │")
    print(f"  ├──────────────┼──────────────┼──────────────┼──────────────┤")
    
    # Recalcular Modelo C con A_D_ref
    rC = full_seesaw_prediction(A_D_ref, [7.1e9, 1.2e10, 1.0e10])
    rT = full_seesaw_prediction(A_D_target, MR_target)
    
    rows = [
        ('θ₁₂ (°)', f"{np.degrees(rC['theta12']):.1f}", f"{np.degrees(rT['theta12']):.1f}", "33.4"),
        ('θ₂₃ (°)', f"{np.degrees(rC['theta23']):.1f}", f"{np.degrees(rT['theta23']):.1f}", "49.1"),
        ('θ₁₃ (°)', f"{np.degrees(rC['theta13']):.1f}", f"{np.degrees(rT['theta13']):.1f}", "8.54"),
        ('Δm²₂₁', f"{rC['dm2_21']:.2e}", f"{rT['dm2_21']:.2e}", f"{exp_dm2_21:.2e}"),
        ('Δm²₃₂', f"{rC['dm2_32']:.2e}", f"{rT['dm2_32']:.2e}", f"{exp_dm2_32:.2e}"),
        ('ratio', f"{rC['dm2_ratio']:.1f}", f"{rT['dm2_ratio']:.1f}", f"{exp_dm2_ratio:.1f}"),
        ('Σmᵢ (eV)', f"{np.sum(rC['masses_eV']):.3f}", f"{np.sum(rT['masses_eV']):.3f}", "< 0.12"),
    ]
    
    for name, vC, vT, vE in rows:
        print(f"  │ {name:<12} │ {vC:>12} │ {vT:>12} │ {vE:>12} │")
    
    print(f"  └──────────────┴──────────────┴──────────────┴──────────────┘")
    
    # M_R comparison
    print(f"\n  Estructura de M_R:")
    print(f"    Modelo C:  diag({7.1e9:.1e}, {1.2e10:.1e}, {1.0e10:.1e}) GeV")
    print(f"               Ratio M₃/M₁ = {1.0e10/7.1e9:.1f}  (casi-degenerado)")
    print(f"    Target:    diag({MR_target[0]:.1e}, {MR_target[1]:.1e}, {MR_target[2]:.1e}) GeV")
    M_sorted = sorted(MR_target)
    print(f"               Ratio max/min = {M_sorted[2]/M_sorted[0]:.1f}")
    
    # Predicciones falsificables
    print(f"\n─── Predicciones Falsificables (de M_R refinado) ───")
    print(f"  1. m₁ = {rT['masses_eV'][0]:.4e} eV  → testeable por KATRIN/Project 8")
    print(f"  2. Σmᵢ = {np.sum(rT['masses_eV']):.4f} eV  → testeable por cosmología CMB-S4")
    print(f"  3. Normal Ordering (NO) → confirmable por JUNO (~2027)")
    print(f"  4. M_R scale ~ {np.mean(MR_target):.1e} GeV → liga a escala GUT de E₈")
    
    # Qué volúmenes de ciclos necesitaría
    print(f"\n─── Volúmenes de Ciclos Co-Asociativos Requeridos ───")
    print(f"  Si M_Rk ~ M_GUT × exp(-Vol(Σ̃_k) / l_P³), con M_GUT = 2×10¹⁶ GeV:")
    M_GUT = 2e16
    for k, M in enumerate(MR_target):
        if M > 0 and M < M_GUT:
            vol_ratio = np.log(M_GUT / M)
            print(f"    Gen {k+1}: Vol(Σ̃_{k+1})/l_P³ = ln(M_GUT/M_{k+1}) = {vol_ratio:.2f}")
        else:
            print(f"    Gen {k+1}: M_{k+1} = {M:.1e} GeV")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 65)
    print("  SEESAW TYPE-I: Δm² Neutrinos + Refinamiento M_R")
    print("  TCS-16 / E₈ Framework — Modelo C")
    print("═" * 65)
    
    # FASE 1: Reconstrucción y diagnóstico
    A_D_ref, r_modelC = phase1_reconstruct_modelC()
    
    # FASE 2: Problema inverso — M_R target
    MR_target, A_D_target, r_target = phase2_inverse_seesaw(A_D_ref)
    
    # FASE 3: Sensibilidad
    results_scan = phase3_sensitivity(A_D_ref, MR_target)
    
    # FASE 4: M_R off-diagonal
    MR_offdiag, A_D_offdiag = phase4_offdiagonal_MR(A_D_ref)
    
    # FASE 5: Resumen
    phase5_summary(r_modelC, MR_target, r_target, MR_offdiag, A_D_ref, A_D_target)
    
    print("\n" + "═" * 65)
    print("  Análisis completo. Ver resultados arriba.")
    print("═" * 65)
