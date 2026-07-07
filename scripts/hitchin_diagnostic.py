#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  DIAGNÓSTICO: μ Ab Initio desde Volúmenes del Compendio
  ¿Se puede fijar μ = Vol(K3_GUT)/l₁₁³ desde datos existentes?
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# ═══ Constantes del Compendio ═══
lambda_ACyl = 2.8
alpha_FHN   = 0.15
a_t = 0.013; a_theta = 0.625; a_K3 = 0.204
Vol_X7    = 3.794e-3       # l₁₁⁷
Vol_K3    = 0.950          # l₁₁⁴ (from area of K3 fiber)
Vol_SU3   = 1.909798       # l_P³
Vol_SU2   = 0.953088
Vol_U1    = 1.121823

t_cones   = np.array([0.35, 0.50, 0.65])
v_ew      = 246.22/np.sqrt(2)
C0_Dirac  = 7348.0; delta_CP = np.pi
d_H       = np.array([0.561, 0.347, 0.198])
d_12, d_23, d_13 = 0.166, 0.343, 0.343
EXP = {'t12': np.radians(33.41), 't23': np.radians(49.1), 't13': np.radians(8.54),
       'dm2_21': 7.41e-5, 'dm2_32': 2.507e-3}
EXP['ratio'] = EXP['dm2_32']/EXP['dm2_21']


# ═══════════════════════════════════════════════════════════════
# 1. DIAGNÓSTICO: ¿POR QUÉ S₁₂ = S₂₃ Y S₁₃ = 2×S₁₂?
# ═══════════════════════════════════════════════════════════════

print("═" * 72)
print("  DIAGNÓSTICO: μ Ab Initio + Análisis de Regímenes")
print("═" * 72)

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  1. ¿POR QUÉ S₁₂ = S₂₃ Y S₁₃ = 2×S₁₂?                     ║
╚═══════════════════════════════════════════════════════════════╝

  Los tres conos están en t = 0.35, 0.50, 0.65.
  El centro del cuello está en t_mid = 0.50.
  
  El perfil sech²(λ(t - 0.50)) con λ = 2.8:
  
    sech²(2.8 × 0.00) = 1.000  ← cono 2 (centro exacto)
    sech²(2.8 × 0.15) = 0.851  ← conos 1 y 3 (simétricos)
    sech²(2.8 × 0.30) = 0.551  ← si hubiera cono a ±0.30
    sech²(2.8 × 0.50) = 0.248  ← extremos
    
  TODOS los conos están en la zona PLANA del perfil (sech² > 0.85).
  La escala de curvatura del sech² es 1/λ = 0.357, y los conos 
  están a ±0.15 del centro → Δt/escala = 0.15/0.357 = 0.42.
  
  En este régimen: sech²(x) ≈ 1 - x² para |x| < 1
  → ∫ sech² dt ≈ Δt (lineal) para caminos cortos.
  → S₁₃ = 2 × S₁₂ (exactamente lineal).
  → NO hay "efecto de junction" — eso era un artefacto del modelo 
    fenomenológico anterior.
""")

# Verify numerically
T = 1.0; t_mid = 0.5
def S_integral(ti, tj):
    def f(t):
        s2 = 1.0/np.cosh(lambda_ACyl*(t-t_mid))**2
        th = abs(np.tanh(lambda_ACyl*(t-t_mid)))
        return s2 * (1 + alpha_FHN * th)
    val, _ = quad(f, ti, tj)
    return val

S12 = S_integral(0.35, 0.50)
S23 = S_integral(0.50, 0.65)
S13 = S_integral(0.35, 0.65)

print(f"  Verificación numérica:")
print(f"    S₁₂ = {S12:.6f}")
print(f"    S₂₃ = {S23:.6f}")
print(f"    S₁₃ = {S13:.6f}")
print(f"    S₁₃/(S₁₂+S₂₃) = {S13/(S12+S23):.6f} (= 1.000 → aditivo)")
print(f"    S₁₃/S₁₂ = {S13/S12:.6f}")
print(f"    S₂₃/S₁₂ = {S23/S12:.6f}")


# ═══════════════════════════════════════════════════════════════
# 2. ¿CONOS ASIMÉTRICOS ROMPEN LA DEGENERACIÓN?
# ═══════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  2. POSICIONES DE CONO ALTERNATIVAS                           ║
╚═══════════════════════════════════════════════════════════════╝

  La simetría S₁₂ = S₂₃ viene de que los conos están simétricos 
  respecto al centro. ¿Qué pasa si los conos están ASIMÉTRICOS,
  o si la junction NO está en el centro?
  
  Exploramos configuraciones alternativas:
""")

print(f"  {'Config':>25} {'t₁':>6} {'t₂':>6} {'t₃':>6} {'S₁₂':>8} {'S₂₃':>8} {'S₁₃':>8} {'S₁₃/S₁₂':>9} {'S₁₂/S₂₃':>9}")
print(f"  {'─'*88}")

configs = [
    ("Original (simétrico)",     0.35, 0.50, 0.65),
    ("Asimétrico leve",          0.30, 0.45, 0.65),
    ("Asimétrico fuerte",        0.20, 0.40, 0.70),
    ("Cono 1 en borde Z₊",      0.10, 0.45, 0.65),
    ("Todos en Z₊",             0.15, 0.25, 0.40),
    ("Cono 3 en borde Z₋",      0.35, 0.55, 0.90),
    ("Spread máximo",            0.10, 0.50, 0.90),
    ("Spread extremo",           0.05, 0.50, 0.95),
]

for label, t1, t2, t3 in configs:
    s12 = S_integral(t1, t2)
    s23 = S_integral(t2, t3)
    s13 = S_integral(t1, t3)
    print(f"  {label:>25} {t1:>6.2f} {t2:>6.2f} {t3:>6.2f} "
          f"{s12:>8.4f} {s23:>8.4f} {s13:>8.4f} "
          f"{s13/s12:>9.3f} {s12/s23:>9.3f}")

# Now check: with cones at spread positions, do we get natural hierarchy?
print(f"\n  ─── Hallazgo clave ───")
print(f"  Con spread máximo (0.10, 0.50, 0.90):")
s12 = S_integral(0.10, 0.50); s23 = S_integral(0.50, 0.90); s13 = S_integral(0.10, 0.90)
print(f"    S₁₂ = {s12:.4f}, S₂₃ = {s23:.4f}, S₁₃ = {s13:.4f}")
print(f"    S₁₃/S₁₂ = {s13/s12:.3f} (vs 2.000 para simétrico)")
print(f"    Para μ=50: F₁₂ = {np.exp(-50*s12):.2e}, F₁₃ = {np.exp(-50*s13):.2e}")
print(f"    Ratio F₁₂/F₁₃ = {np.exp(-50*s12)/np.exp(-50*s13):.1e}")


# ═══════════════════════════════════════════════════════════════
# 3. DERIVACIÓN DE μ DESDE VOLÚMENES CONOCIDOS
# ═══════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  3. DERIVACIÓN DE μ DESDE EL COMPENDIO                       ║
╚═══════════════════════════════════════════════════════════════╝

  La masa Majorana desde M5-branas:
  
    (M_R)ᵢⱼ ∝ exp(-Vol(Σ̃ᵢⱼ) / (2π l₁₁³))
  
  El 4-ciclo co-asociativo Σ̃ᵢⱼ tiene topología:
    Σ̃ᵢⱼ = K3_fiber × γᵢⱼ
  donde γᵢⱼ es una curva en la base S¹×I del cuello.
  
  Su volumen es:
    Vol(Σ̃ᵢⱼ) = Vol(K3_GUT) × ∫_γ ds_base
  
  La integral ∫_γ ds_base es exactamente S_ij del Hitchin flow.
  
  Entonces: μ = Vol(K3_GUT) / (2π l₁₁³)
""")

# Estimate Vol(K3_GUT) from the Compendio data
# The gauge cycle volumes are:
# Vol(SU3) = 1.91 l_P³, Vol(SU2) = 0.95 l_P³, Vol(U1) = 1.12 l_P³
# These are 3-cycles. K3_GUT is a 4-cycle (codimension-3 in X₇).
# The relationship is: Vol(K3_GUT) ~ a_K3 × Vol(gauge_cycle)

# From the metric: a_K3 = 0.204 l₁₁
# Vol(K3) = a_K3⁴ × Vol(T⁴/ℤ₂) = a_K3⁴ × (1/2) ≈ 0.204⁴/2 ≈ 8.65×10⁻⁴ l₁₁⁴
# But the Compendio gives Vol(K3) ≈ 0.95 l₁₁⁴
# This means the "geometric" K3 has internal structure beyond the metric scale.

# The GUT 4-cycle is a K3-type cycle in the COMPLEMENTARY directions (α₅...α₈)
# Its volume includes the t-direction extent:
# Vol(K3_GUT) = Vol(K3_fiber) × L_eff(t-direction)
# But that's redundant with S_ij.

# More directly: μ × S₁₂ = effective exponent for pair 1↔2
# From the fit: μ = 51.15, so μ × S₁₂ = 51.15 × 0.146 = 7.47
# This means Vol(Σ̃₁₂)/(2πl₁₁³) = 7.47

# Can we derive μ from α_GUT⁻¹?
# At M_GUT: α_i⁻¹ = Vol(Σᵢ)/(2πl_P³)
# For the GUT sector (SO(10) / SU(5)): the relevant cycle has Vol ~ Vol(K3_GUT)
# But K3_GUT is a 4-cycle, not a 3-cycle...

# The correct relationship:
# For M5-branes (wrapping 4-cycles): the instanton action is
# S_M5 = Vol(Σ̃)/(2π)² l₁₁³ = Vol₄/(4π² l₁₁³)
# [Note: M5 has tension T_M5 = 1/(2π)² l₁₁⁶, action = T_M5 × Vol₆ = Vol₄/(4π² l₁₁²)]

# Wait - M5-brane in 11D wraps a 6-manifold of worldvolume (1 time + 5 space).
# For Euclidean M5 on a 4-cycle × worldline: action = T_M5 × Vol(4-cycle × path)
# T_M5 = 1/((2π)⁵ l_P⁵)... let me be more careful.

# Actually for M-theory instantons:
# M2 on 3-cycle: S = Vol(Σ₃)/(2π l₁₁³)  ← this is for Yukawa (m_D)
# M5 on 4-cycle × S¹: S = Vol(Σ₄)/(2π)² l₁₁³... 

# Let me use the simpler approach:
# The factor in the exponent is 2πVol(Σ̃)/l₁₁³ (standard M-theory normalization)
# So μ = 2π × Vol_fiber/l₁₁³

# From the Compendio: the relevant GUT fiber volume
# Approach 1: from gauge coupling unification
alpha_GUT_inv = 24.0  # from Compendio Table 2: all three α⁻¹ ≈ 24
# α_GUT⁻¹ = Vol(Σ_GUT)/(2πl_P³)
# → Vol(Σ_GUT) = 2π × 24 × l_P³ ≈ 150.8 l_P³

# But this is a 3-cycle volume. For M5 on 4-cycle:
# Vol(4-cycle) = Vol(3-cycle) × L_extra_direction
# The extra direction is the K3 fiber direction, with scale a_K3 = 0.204

# Approach 2: from Vol(K3) directly
# Vol(K3_GUT) ≈ Vol(K3) × (blowup factor)
# The GUT blowups have a_GUT > a_SM by some factor
# From the K3 Kummer analysis: a_GUT/a_SM was scanned 1-20

# Approach 3: dimensional analysis
# μ = Vol(K3_GUT)/(2πl₁₁³) 
# Vol(K3) = 0.95 l₁₁⁴
# Vol(K3_GUT) ~ Vol(K3)/l₁₁ ≈ 0.95 l₁₁³
# → μ ~ 0.95/(2π) ≈ 0.151 ← TOO SMALL (need 51!)

# The discrepancy factor is μ_needed/μ_naive = 51/0.15 ≈ 340

# Approach 4: from the HITCHIN FLOW normalization
# The ODE system has a natural scale. The "Vol" in the instanton action 
# is not just the naive K3 volume but includes:
# (a) The conformal factor from the G₂ holonomy metric
# (b) The flux contribution: ∫ C₃ on the cycle
# (c) The BPS correction (from Etapa 13)

# C₀(BPS) = 0.229 → this is the prefactor, not the exponent.
# The exponent is: d/l_s = d/(a_K3 × l_s_norm) 
# where l_s_norm = 0.044 (Modelo C)

# Actually, the simplest interpretation:
# μ_pheno = 51.15
# ℓ_neck^equiv = 0.0562
# Compare: a_t = 0.013 (physical t-scale)
# ℓ_neck^equiv / a_t = 0.056/0.013 = 4.3

# This means the neck is about 4.3 times the physical t-scale.
# Or: the M5 instanton "sees" the cuello as 4.3× longer than the 
# physical distance, due to the conformal warp factor.

print(f"  ─── Cuatro estimaciones de μ ───")
print()

# Estimate 1: from α_GUT
Vol_3cycle = 2 * np.pi * alpha_GUT_inv * 1.0  # l_P³ (but l_P ~ l₁₁ in 11D)
mu_est1 = Vol_3cycle * a_K3 / (2*np.pi)  # l₁₁³ × l₁₁ / l₁₁³
print(f"  (a) Desde α_GUT⁻¹ = 24:")
print(f"      Vol(Σ_3_GUT) = 2π × 24 = {Vol_3cycle:.1f} l_P³")
print(f"      Vol(K3_GUT) ≈ Vol(Σ₃) × a_K3 = {Vol_3cycle:.1f} × {a_K3} = {Vol_3cycle*a_K3:.1f}")
print(f"      μ₁ = Vol(K3_GUT)/(2πl₁₁³) = {mu_est1:.2f}")

# Estimate 2: from Vol(X₇) decomposition
# Vol(X₇) = Vol(K3) × Vol(S¹) × Vol(neck) × geometric_factor
# Vol(X₇) = 3.794×10⁻³ l₁₁⁷
# Vol(K3) ≈ a_K3⁴ × C = 0.204⁴ × C
# Vol(S¹) = 2π a_theta × r
# Vol(neck) = a_t × T
# Rough: Vol(X₇) ~ a_K3⁴ × a_theta × a_t × T × (2π)² × C
# 3.794e-3 = 0.204⁴ × 0.625 × 0.013 × 1 × 4π² × C
# C = 3.794e-3 / (1.73e-3 × 0.625 × 0.013 × 39.48)
# C = 3.794e-3 / (5.54e-4) = 6.85
Vol_K3_geom = a_K3**4 * 6.85
mu_est2 = Vol_K3_geom / (2*np.pi * a_K3)  # relative to l₁₁³ = a_K3³ × ...
print(f"\n  (b) Desde descomposición de Vol(X₇):")
print(f"      Vol(X₇) = {Vol_X7:.3e} l₁₁⁷")
print(f"      Vol(K3_geom) ≈ a_K3⁴ × C = {Vol_K3_geom:.4e} l₁₁⁴")

# Estimate 3: from the seesaw string length
# In the Dirac sector: l_s = 0.044 → this sets the string tension
# The M2-brane exponent is d/l_s = d/(a_K3 × something)
# For M5: μ × S_ij ↔ d̃_ij/l̃_s
# With l̃_s = 0.044 (same string): μ × S₁₂ = d̃₁₂/l̃_s
# → d̃₁₂ = μ × S₁₂ × l̃_s = 51.15 × 0.146 × 0.044 = 0.329
# This is comparable to d₂₃ = 0.343 → makes sense!
d_eff_12 = 51.15 * 0.146 * 0.044
d_eff_13 = 51.15 * 0.292 * 0.044
print(f"\n  (c) Consistencia con l_s de Dirac:")
print(f"      Si l̃_s ≈ l_s = 0.044:")
print(f"      d̃₁₂ = μ·S₁₂·l_s = {d_eff_12:.4f}  (cf d₂₃ = {d_23})")
print(f"      d̃₁₃ = μ·S₁₃·l_s = {d_eff_13:.4f}  (cf d₁₃ = {d_13})")
print(f"      → Las distancias GUT son ~{d_eff_12/d_12:.0f}× las distancias SM")

# Estimate 4: EXACT from the Hitchin flow normalization
# The Hitchin flow gives Vol(X₇) = 3.794×10⁻³ l₁₁⁷ 
# The NECK contributes: Vol_neck = ∫₀ᵀ Vol(K3(t)) × Vol(S¹) dt
# = ∫₀ᵀ sech²(λ(t-T/2)) × Vol(K3_∞) × 2π a_theta dt
# = Vol(K3_∞) × 2π a_theta × [tanh(λT/2)/λ - tanh(-λT/2)/λ]
# = Vol(K3_∞) × 2π a_theta × 2tanh(λ/2)/λ

neck_integral = 2 * np.tanh(lambda_ACyl * 0.5) / lambda_ACyl
Vol_neck_section = neck_integral  # ∫ sech²(λ(t-0.5)) dt from 0 to 1
print(f"\n  (d) Desde la integral del cuello:")
print(f"      ∫₀¹ sech²(λ(t-0.5)) dt = {Vol_neck_section:.4f}")
print(f"      Vol(cuello) = Vol(K3_∞)×2πa_θ × {Vol_neck_section:.4f}")

# Now: μ relates to how much of the X₇ volume is in the neck
# Vol(neck)/Vol(X₇) gives the fraction
# And μ ∝ Vol(K3_cross_section) / l₁₁³

# The KEY relationship: 
# For M5 on Σ̃ᵢⱼ: action = (2πT_M5) × Vol(Σ̃ᵢⱼ)
# T_M5 in Planck units: T_M5 = 1/(2π)²l_P³
# So: S = Vol(Σ̃ᵢⱼ)/(2π l_P³)
# Vol(Σ̃ᵢⱼ) = Vol(K3_cross) × a_t × Sᵢⱼ
# μ = Vol(K3_cross) × a_t / (2π l_P³)
# ≈ Vol(K3) × a_t / (2π)
# = 0.95 × 0.013 / (2π) ≈ 0.00197

# This gives μ ≈ 0.002, way too small!
# The resolution: l_P ≠ l₁₁ in our units.
# Actually in M-theory: l₁₁ = l_P = (ℏG₁₁/c³)^(1/9)
# The Compendio works in units where l₁₁ = 1
# And the metric parameters (a_t, a_K3 etc) are in these units

# So: Vol(K3_cross) in l₁₁ units: 
# Vol(K3) = 0.95 l₁₁⁴ from the Compendio
# Cross-section along t: Vol(K3_cross) = Vol(K3) (it IS K3)
# Vol(Σ̃ᵢⱼ) = Vol(K3) × ∫ᵢʲ (scale_factor) dt
# The scale factor along t is a_t/l₁₁ (conformal factor)

# Wait - the Sᵢⱼ integrals I computed already include the sech² profile
# But they DON'T include the PHYSICAL scale a_t.
# So: Vol(Σ̃ᵢⱼ) = Vol(K3_fiber) × a_t × Sᵢⱼ / l₁₁⁵ × l₁₁⁵
# And: μ_physical = Vol(K3_fiber)/(2π l₁₁⁴) × a_t × (1/l₁₁)... 

# Let me just do this cleanly:
# In our computation, F = exp(-μ × S)
# S is dimensionless (integral of profile along t ∈ [0,1])
# μ is dimensionless
# The physical action is: μ × S = Vol(Σ̃)/(2π l₁₁³)
# Vol(Σ̃) = Vol_fiber × L_path
# Vol_fiber = Vol(K3) = 0.95 l₁₁⁴
# L_path = a_t × S (physical length)
# So: μ × S = Vol(K3) × a_t × S / (2π l₁₁³ × l₁₁)
# Wait, that gives μ = Vol(K3) × a_t / (2π l₁₁⁴)
# = 0.95 × 0.013 / (2π) = 0.00197

# But we need μ = 51! Factor of 26,000.

# The issue: the "Vol(K3)" in the Compendio is the GAUGE volume,
# normalized differently from the M5-brane action.
# The M5-brane action includes a factor of (2π)^(-5) × g_s^(-2)...

# More practically: the ratio between M2 and M5 exponents gives μ:
# M2 on 3-cycle: exp(-d/l_s) with l_s = 0.044
# M5 on 4-cycle: exp(-μ × S)
# d/l_s = geometric distance / string length
# μ × S = Vol(K3_GUT cross-section) × path length / string_tension

# From the fit: μ × S₁₂ = 7.47
# And d₁₂/l_s = 0.166/0.044 = 3.77
# Ratio: (μ×S₁₂)/(d₁₂/l_s) = 7.47/3.77 = 1.98 ≈ 2

# This means: the M5 exponent is about 2× the M2 exponent for comparable paths!
# The factor 2 comes from the EXTRA dimension of the K3 fiber.

print(f"\n  ─── RELACIÓN M2 ↔ M5 ───")
mu_fit = 51.15
for (i,j), d, label in [((0,1), d_12, "1↔2"), ((1,2), d_23, "2↔3"), ((0,2), d_13, "1↔3")]:
    S = S_integral(t_cones[i], t_cones[j])
    l_s = 0.044
    print(f"    Par {label}: μ·S = {mu_fit*S:.3f}, d/l_s = {d/l_s:.3f}, ratio = {mu_fit*S/(d/l_s):.3f}")


# ═══════════════════════════════════════════════════════════════
# 4. μ AB INITIO: MEJOR ESTIMACIÓN
# ═══════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  4. ESTIMACIÓN AB INITIO DE μ                                ║
╚═══════════════════════════════════════════════════════════════╝

  La MEJOR estimación de μ viene de la relación:
  
  μ = (2π/l₁₁) × Vol(K3_GUT) / Vol(S_base)
  
  donde Vol(K3_GUT) es el volumen de la fibra K3 en la dirección GUT
  y Vol(S_base) normaliza el camino base.
  
  Pero más directamente, podemos estimarlo desde la relación M2↔M5:
  
  Exponente M2 (Dirac):  exp(-d_ij/l_s)         [envuelve 3-ciclo]
  Exponente M5 (Majorana): exp(-μ × S_ij)       [envuelve 4-ciclo]
  
  Si los ciclos están "emparentados" (misma región de X₇), la relación es:
  
  μ × S_ij = (d̃_ij/l̃_s) × Vol(K3_fiber)/Vol(base_3cycle)
  
  Para el par 1↔2: μ × 0.146 = (d̃₁₂/l̃_s)
  Si d̃₁₂ ≈ d₁₂ × (a_GUT/a_SM) y l̃_s ≈ l_s:
""")

# The key insight: μ can be estimated from the condition that
# the M5-brane tension is related to the M2-brane tension by
# a factor of Vol(K3_fiber)/l₁₁.

# In M-theory: T_M2 = 1/(2π)² l₁₁³, T_M5 = 1/(2π)⁵ l₁₁⁶
# Ratio: T_M5/T_M2 = 1/((2π)³ l₁₁³) 
# For a cycle with K3 fiber: Vol(4-cycle) = Vol(K3) × Vol(curve)
# Action_M5 = T_M5 × Vol(4) = Vol(K3)/(2π)³ l₁₁³ × T_M2 × Vol(curve)
# = [Vol(K3)/((2π)³ l₁₁³)] × [T_M2 × Vol(curve)]
# = [Vol(K3)/((2π)³ l₁₁³)] × exp_M2

# So: μ_theory = Vol(K3) / ((2π)³ l₁₁³)

# Wait but S_ij already includes the path, so:
# exp_M5 = -T_M5 × Vol(K3_fiber) × ∫ ds_path
# = -[Vol(K3)/(2π)³ l₁₁⁶] × a_t × S_ij × l₁₁
# = -[Vol(K3) × a_t/((2π)³ l₁₁⁵)] × S_ij

# In our units (l₁₁ = 1):
mu_theory = Vol_K3 * a_t / ((2*np.pi)**3)
print(f"  Estimación rigurosa (M-theory tensions):")
print(f"    μ_theory = Vol(K3) × a_t / (2π)³")
print(f"    = {Vol_K3:.3f} × {a_t:.3f} / {(2*np.pi)**3:.2f}")
print(f"    = {mu_theory:.6f}")
print(f"    Esto es ~{mu_fit/mu_theory:.0f}× menor que μ_fit = {mu_fit:.1f}")

# The huge discrepancy means we're missing a factor.
# Possible sources:
# 1. Vol(K3_GUT) >> Vol(K3_SM) (GUT fiber is much larger)
# 2. The normalization includes (α_GUT)⁻¹ ≈ 24 factors
# 3. The string length l_s in 11D vs 10D differs

# Most likely: the relevant cycle for Majorana is NOT just K3_fiber × curve
# It's a K3_fiber × (full S¹ of the TCS), and the S¹ factor gives 2π a_theta

mu_theory2 = Vol_K3 * a_t * 2*np.pi * a_theta / ((2*np.pi)**3)
print(f"\n  Con factor S¹: μ = Vol(K3)×a_t×2πa_θ / (2π)³")
print(f"    = {mu_theory2:.4f}")
print(f"    Aún ~{mu_fit/mu_theory2:.0f}× menor")

# Include α_GUT⁻¹ factor
mu_theory3 = mu_theory2 * alpha_GUT_inv
print(f"\n  Con factor α_GUT⁻¹ = {alpha_GUT_inv}:")
print(f"    μ = {mu_theory3:.2f}")
print(f"    Ratio μ_fit/μ_theory = {mu_fit/mu_theory3:.2f}")

# Include 4π² (M5 vs M2 extra factors)
mu_theory4 = mu_theory2 * alpha_GUT_inv * (4*np.pi**2) / (2*np.pi)
print(f"\n  Con normalización BPS (4π²/2π):")
print(f"    μ = {mu_theory4:.2f}")
print(f"    Ratio μ_fit/μ_theory = {mu_fit/mu_theory4:.2f}")

# Another approach: use the M_R scale
# M_R ~ M_GUT × exp(-μ×Vol(cone_cycle)) → fixes μ if we know M_R and M_GUT
M_GUT = 2e16  # GeV
M_R_avg = np.mean([3.15e8, 4.11e9, 1.12e14])  # from fit
Vol_cone_avg = np.mean([Vol_SU3, Vol_SU2, Vol_U1])  # ~ 1.33 l_P³
# M_R = M_GUT × exp(-2π Vol_cone / l_P³) but this is M2, not M5

print(f"\n  ─── Conclusión sobre μ ab initio ───")
print(f"  μ_fit = {mu_fit:.1f}")
print(f"  μ_theory (best) = {mu_theory4:.1f}")
print(f"  Discrepancia: {mu_fit/mu_theory4:.1f}×")
print(f"  ")
print(f"  El factor faltante probablemente viene de:")
print(f"  (a) La curvatura del cono NK (η=1/3) que aumenta el volumen efectivo")
print(f"  (b) El perfil exacto de G₄ flux sobre la fibra K3_GUT")
print(f"  (c) La contribución BPS (Etapa 13): det'(Δ) → factor adicional")


# ═══════════════════════════════════════════════════════════════
# 5. ℓ_neck PREDICHO vs ℓ_neck FENOMENOLÓGICO
# ═══════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  5. ℓ_neck: PREDICCIÓN vs FENOMENOLOGÍA                      ║
╚═══════════════════════════════════════════════════════════════╝

  El Hitchin flow FIJA la forma del perfil (sech²) pero NO la escala μ.
  
  Sin embargo, el flujo sí predice relaciones entre pares:
  
    S₁₃ = 2 × S₁₂  (exacto, por simetría de posiciones)
    S₁₂ = S₂₃       (exacto, por simetría t₁↔t₃ resp. centro)
  
  Estas relaciones significan:
    F₁₃ = (F₁₂)²    ← PREDICCIÓN del Hitchin flow
    F₁₂ = F₂₃        ← PREDICCIÓN del Hitchin flow
  
  Comparar con el modelo fenomenológico anterior (junction crossing):
    F₁₃ = (F₁₂)² × F_junction  (modelo anterior tenía factor extra)
    F₁₂ ≠ F₂₃ (modelo anterior rompía la simetría)
  
  El Hitchin flow dice: NO hay efecto de junction diferenciado.
  La jerarquía viene PURAMENTE de que S₁₃ = 2×S₁₂.
""")

# Show the consequences
print(f"  ─── Con μ_fit = {mu_fit:.1f}: ───")
print(f"    F₁₂ = exp(-{mu_fit:.1f} × {S12:.4f}) = exp(-{mu_fit*S12:.2f}) = {np.exp(-mu_fit*S12):.4e}")
print(f"    F₂₃ = exp(-{mu_fit:.1f} × {S23:.4f}) = exp(-{mu_fit*S23:.2f}) = {np.exp(-mu_fit*S23):.4e}")
print(f"    F₁₃ = exp(-{mu_fit:.1f} × {S13:.4f}) = exp(-{mu_fit*S13:.2f}) = {np.exp(-mu_fit*S13):.4e}")
print(f"    Ratio F₁₂/F₁₃ = {np.exp(-mu_fit*S12)/np.exp(-mu_fit*S13):.1f}")
print(f"    Verificar: (F₁₂)² = {np.exp(-mu_fit*S12)**2:.4e} ≈ F₁₃ = {np.exp(-mu_fit*S13):.4e} ✅")

print(f"\n  ─── Relación con ℓ_neck del modelo anterior: ───")
print(f"    El modelo anterior con junction crossing usó:")
print(f"    F₁₃ = exp(-λ·0.30/ℓ) × exp(-λ·0.15/ℓ) = exp(-λ·0.45/ℓ)")
print(f"    Con ℓ_neck = 0.015 → F₁₃ = exp(-{lambda_ACyl*0.45/0.015:.0f}) = {np.exp(-lambda_ACyl*0.45/0.015):.1e}")
print(f"    ")
print(f"    El Hitchin flow da: F₁₃ = exp(-{mu_fit*S13:.1f}) = {np.exp(-mu_fit*S13):.1e}")
print(f"    Mucho más moderado — la junction crossing era artificial.")
print(f"    ")
print(f"    ℓ_neck equivalente (del Hitchin) = λ·Δt/(μ·S) = {lambda_ACyl*0.30/(mu_fit*S13):.4f}")
print(f"    ℓ_neck del modelo anterior = 0.0147")
print(f"    El Hitchin flow predice ℓ_neck ~{lambda_ACyl*0.30/(mu_fit*S13)/0.0147:.0f}× más grande")


# ═══════════════════════════════════════════════════════════════
# 6. PREDICCIÓN ROBUSTA: RATIO Δm² COMO FUNCIÓN DE μ SOLO
# ═══════════════════════════════════════════════════════════════

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  6. ¿CUÁNTO DE μ ESTÁ REALMENTE FIJADO?                      ║
╚═══════════════════════════════════════════════════════════════╝

  Análisis de sensibilidad: ¿cómo varía el ratio Δm² con μ?
""")

def quick_seesaw_ratio(mu_val, M_diag, ls_vec, AD, C0):
    """Quick seesaw for given μ."""
    F = np.ones((3,3))
    for i in range(3):
        for j in range(i+1,3):
            S = S_integral(t_cones[i], t_cones[j])
            F[i,j] = F[j,i] = np.exp(-mu_val * S)
    MR = np.zeros((3,3))
    for i in range(3):
        MR[i,i] = M_diag[i]
        for j in range(i+1,3):
            MR[i,j] = MR[j,i] = np.sqrt(M_diag[i]*M_diag[j]) * F[i,j]
    try:
        eigs = np.linalg.eigvalsh(MR)
        if np.any(eigs <= 0): return None
        ls1,ls2,ls3 = ls_vec
        y_D = np.array([AD*np.exp(-d_H[0]/ls1), AD*np.exp(-d_H[1]/ls2), AD*np.exp(-d_H[2]/ls3)])
        Y = np.diag(y_D.astype(complex))
        phase = np.exp(1j*delta_CP)
        for (i,j),d,lse in [((0,1),d_12,np.sqrt(ls1*ls2)),((1,2),d_23,np.sqrt(ls2*ls3)),((0,2),d_13,np.sqrt(ls1*ls3))]:
            amp = C0*np.exp(-d/lse)*np.sqrt(y_D[i]*y_D[j])
            Y[i,j]=amp*phase; Y[j,i]=amp*np.conj(phase)
        mD = Y * v_ew
        m_nu = -mD @ np.linalg.inv(MR.astype(complex)) @ mD.T
        H = m_nu.conj().T @ m_nu
        ev = np.sort(np.linalg.eigvalsh(H))
        m = np.sqrt(np.abs(ev)) * 1e9
        dm21 = m[1]**2 - m[0]**2
        dm32 = m[2]**2 - m[1]**2
        return dm32/dm21 if dm21 > 0 else None
    except: return None

# Use optimal parameters from fit
M_opt = [3.153e8, 4.111e9, 1.122e14]
ls_opt = [0.2746, 0.3607, 0.4403]
AD_opt = 9.3445e-4; C0_opt = 4.5

print(f"  μ scan (otros params fijados al óptimo):")
print(f"  {'μ':>8} {'F₁₂':>12} {'F₁₃':>12} {'F₁₂/F₁₃':>10} {'Δm²ratio':>10} {'match?':>8}")
print(f"  {'─'*62}")

for mu_test in [10, 20, 30, 40, 51.15, 60, 70, 80, 100]:
    f12 = np.exp(-mu_test * S12)
    f13 = np.exp(-mu_test * S13)
    ratio_f = f12/f13 if f13 > 1e-50 else np.inf
    dm_ratio = quick_seesaw_ratio(mu_test, M_opt, ls_opt, AD_opt, C0_opt)
    dm_str = f"{dm_ratio:.1f}" if dm_ratio else "---"
    match = "✅" if dm_ratio and abs(dm_ratio - 33.8) < 2 else ""
    print(f"  {mu_test:>8.1f} {f12:>12.3e} {f13:>12.3e} {ratio_f:>10.1f} {dm_str:>10} {match:>8}")

print(f"\n  Nota: el ratio Δm² cambia porque F afecta los eigenvalues de M_R")
print(f"  y la inversión seesaw es altamente no-lineal.")

# What range of μ gives ratio within ±10%?
mu_range = []
for mu_test in np.linspace(20, 100, 200):
    r = quick_seesaw_ratio(mu_test, M_opt, ls_opt, AD_opt, C0_opt)
    if r and abs(r - 33.8) / 33.8 < 0.10:
        mu_range.append(mu_test)

if mu_range:
    print(f"\n  Rango de μ compatible con Δm² ratio ±10%:")
    print(f"    μ ∈ [{min(mu_range):.1f}, {max(mu_range):.1f}]")
    print(f"    Ancho: Δμ/μ = {(max(mu_range)-min(mu_range))/51.15:.0%}")
else:
    print(f"\n  (El ratio depende sensiblemente de todos los params, no solo μ)")


print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  RESUMEN FINAL                                                ║
╚═══════════════════════════════════════════════════════════════╝

  ─── Lo que el Hitchin flow DETERMINA: ───
  
  1. PERFIL: ψ₄(t) ~ sech²(λ(t - T/2)) con λ = 2.8
     → Fijado completamente por la Etapa 13
  
  2. RELACIONES ENTRE PARES:
     S₁₂ = S₂₃ = {S12:.6f}  (simétricos por posición)
     S₁₃ = 2 × S₁₂ = {S13:.6f}  (aditivo, sin junction)
     → F₁₃ = (F₁₂)²  ← PREDICCIÓN ROBUSTA
  
  3. NO HAY EFECTO DE JUNCTION DIFERENCIADO
     El modelo anterior (junction crossing) era un artefacto.
     La jerarquía real viene del CAMINO MÁS LARGO (S₁₃ = 2×S₁₂),
     amplificado por μ grande.
  
  ─── Lo que QUEDA LIBRE: ───
  
  4. μ = {mu_fit:.1f} (escala de acoplamiento M5-brana)
     Interpretación: Vol(K3_GUT_fiber)/normalización
     Mejor estimación teórica: μ ~ {mu_theory4:.0f} (discrepancia {mu_fit/mu_theory4:.0f}×)
     Rango compatible: μ ∈ [{min(mu_range) if mu_range else '?':.0f}, {max(mu_range) if mu_range else '?':.0f}]
  
  ─── Predicción ℓ_neck (la pregunta original): ───
  
  5. ℓ_neck^equiv = λ·Δt/(μ·S) = {lambda_ACyl*0.30/(mu_fit*S13):.4f}
     Compare: ℓ_neck(modelo anterior) = 0.0147
     El Hitchin flow predice un ℓ_neck ~4× más grande
     porque NO incluye la penalización de junction.
     
  6. REDUCCIÓN NETA DE PARÁMETROS:
     Antes: ℓ_neck + junction_model = 2 parámetros libres
     Ahora: μ = 1 parámetro libre (perfil + relaciones fijados)
     → Se elimina 1 parámetro, se gana 2 relaciones.
""")

print(f"{'═'*72}")
print(f"  Diagnóstico de ℓ_neck desde Hitchin flow completo.")
print(f"{'═'*72}")
