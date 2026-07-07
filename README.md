# G₂-TCS-16 — Pipeline computacional público

Pipeline de validación numérica del marco G₂-TCS-16 (compactificación de M-teoría sobre variedad de holonomía G₂, construcción Twisted Connected Sum) más el núcleo mínimo reproducible (MRE R16.7) del postulado de conservación informacional.

**Pre-registro de predicciones:** [DOI 10.5281/zenodo.21231245](https://doi.org/10.5281/zenodo.21231245) — publicado 2026-07-07, anterior a toda determinación experimental del ordenamiento de masas de neutrinos
**Documentación:** Compendio Tomo I v9.2, Tomo II v3, Anexo H (enlaces al depósito Zenodo)
**Autor:** Diego Santana S. (AMI Group, Santiago, Chile) — Licencia: MIT (código) / CC-BY 4.0 (documentos)
**Nota de método:** desarrollado con asistencia de IA para verificación, cómputo y edición; esto no constituye revisión por pares. Los cálculos geométricos no han sido verificados aún por especialistas independientes en geometría G₂.

## Estructura

| Carpeta | Contenido | Estado de reproducibilidad |
|---|---|---|
| `mre_r16_7/` | Implementación mínima reproducible R16.7 (dinámica reversible + observable espectral) | **Verde: corre end-to-end con solo `numpy`; 5/5 tests** |
| `scripts/` | Los 13 scripts del pipeline del Tomo II v3 + Anexo H (SHA-256 en el pre-registro) | Ámbar: requieren congelación de seeds (ver abajo). Entorno real pinneado en `environment.yml` — pipeline CPU puro (numpy/scipy/matplotlib), sin dependencias GPU |
| `herramientas/` | `calculadora_claseB.py` — fuente única de Σmᵢ, m_β, m_ββ desde NuFIT (stdlib puro) | Verde: determinista, sin dependencias |
| `resultados_referencia/` | Salidas canónicas del MRE | — |

## Reproducción inmediata (CPU, sin GPU)

```bash
pip install numpy
cd mre_r16_7 && python run_verifier.py && python -m unittest discover tests
python herramientas/calculadora_claseB.py
```

## Qué reproduce el pipeline principal (`scripts/`)

| Script | Salida | Seeds (en código) |
|---|---|---|
| `hitchin_ell_neck.py` | Tabla A.1 — flujo Hitchin, torsión G₂ = 7.3×10⁻⁵ | DE: `seed=s+100` |
| `hitchin_diagnostic.py` | Tabla A.0 — **resultado negativo C₀ ab initio (se conserva)** | **Determinista** (sin llamadas estocásticas) |
| `degeneracy_analysis.py` | Tabla C.1 | DE: `seed=0…49` (dos fases) |
| `ckm_test.py` | Descarte Tier 1 sector quarks | DE: PMNS `seed=0…39`; CKM `seed=s*7+seed_sol` |
| `ls_from_volumes_v2.py` | Tabla D.2 (9/9 observables, C₀ y κ **ajustados**) | DE: `seed=s*13+42` (×2 fases) y `s*17+7` |
| `volumetric_degeneracy.py` | Tabla E.1 | DE: `seed=s*13+42` (30 seeds) |
| `volumetric_degeneracy_v3.py` | Tabla 12 — Cuenca A (50 seeds, criterio cost<1.0) | DE: `seed=s*13+42` (50 seeds) |
| `seesaw_analysis.py`, `MR_formalization_E8.py`, `K3_kummer_nonisotropic.py`, `via1_neck_combined.py`, `final_model_neck_sectorial.py` | Vías intermedias B.1–B.4 (documentadas, incluidas por transparencia) | DE: `seed=` explícita en toda llamada (`+42/+100/+200/+300/+500/+700`) |
| `predicciones_sensibilidad_TCS16.py` | Anexo H — MC N=10,000 | `default_rng(42)` ✅ |

**Estado de seeds (auditoría 2026-07-07):** los 13 scripts tienen seeds explícitas en cada llamada estocástica o son deterministas — sin `np.random` global sin sembrar, sin paralelismo (`workers`) no determinista. La nota del pre-registro ("solo `predicciones_sensibilidad` con seed explícita") era más conservadora que el código real; se corrige aquí con la tabla exacta. Verificación empírica de determinismo (doble corrida idéntica por script) y salidas de referencia canónicas del entorno de producción: en curso — ver `REPRODUCIBILIDAD.md` §1 y §4.

## Transparencia declarada

- C₀ y κ son parámetros **ajustados a datos**, no derivados (ver `hitchin_diagnostic.py` y §5 del pre-registro; el intento ab initio dio C₀ = 27.9 vs 0.18 fenomenológico).
- El criterio de aceptación de Cuenca A (cost < 1.0) y su historial v1→v2→v3 están documentados en `REPRODUCIBILIDAD.md` §3.
- Los valores de Clase B (Σmᵢ, m_β, m_ββ) provienen exclusivamente de `herramientas/calculadora_claseB.py`; ver la nota de reconciliación de convención Δm²₃₁/Δm²₃₂ en el paquete del pre-registro.
- Resultados negativos se conservan en el repositorio: son parte del registro científico.

## CI

`.github/workflows/reproduce.yml` corre en cada push: MRE completo (verifier + tests) y calculadora Clase B, en CPU. El pipeline pesado se ejecuta manualmente (`workflow_dispatch`) con el entorno pinneado de `environment.yml` (export real del entorno de producción `g2_tcs`, 2026-07-06: Python 3.11.14, numpy 2.4.2, scipy 1.17.1 — CPU puro, verificado que ningún script importa PyTorch/CUDA).

## Cita

Ver `CITATION.cff`. Por favor cite también el DOI del pre-registro.
