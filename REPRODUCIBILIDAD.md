# REPRODUCIBILIDAD.md — Lista operativa de congelación del pipeline

Objetivo: que un tercero, sin contacto con el autor, regenere las Tablas D.2, E.1 y 12 y obtenga los mismos números. Cada ítem es verificable.

## 1. Extracción de seeds (los 13 scripts)

Para cada script, ejecutar y registrar en la tabla del README:

```bash
grep -nE "np\.random\.seed|default_rng|torch\.manual_seed|cuda\.manual_seed|seed\s*=" scripts/*.py
```

Reglas de congelación:
- Si un script usa `differential_evolution`, fijar `seed=` explícito en la llamada (no confiar en el estado global de NumPy).
- (Verificado 2026-07-06: ningún script del pipeline importa PyTorch/CUDA — es CPU puro con numpy/scipy/matplotlib. La regla siguiente queda sin objeto salvo que se añada código GPU en el futuro: `torch.manual_seed(S)`, `torch.cuda.manual_seed_all(S)`, `torch.use_deterministic_algorithms(True)`.)
- Si un script NO tiene seed, asignarla ahora, correr una vez, y esa corrida pasa a ser la referencia canónica.

## 2. Congelación de hiperparámetros (hoy implícitos en el código)

Registrar por script, en un bloque de cabecera del propio archivo:
- Bounds de cada parámetro del optimizador
- Estrategia DE (`best1bin`, etc.), `popsize`, `maxiter`, `tol`, `mutation`, `recombination`
- Tolerancias de L-BFGS-B (`ftol`, `gtol`, `maxiter`)
- Constantes físicas hardcodeadas (valores NuFIT usados, con versión: 5.3 o 6.0)
- Pesos de penalización de cada constraint (#3, #4, #5, #8)

## 3. Criterio de Cuenca A — declaración del historial (obligatoria)

Documentar en este archivo, textualmente, para blindaje ante revisores:

> v1: criterio cost < 1.0 → 18/30 seeds aceptadas.
> v2: criterio relajado a cost < 2.0 para explorar el landscape modificado → entraron 14 soluciones de la cuenca inferior (Δm² ratio ≈ 44 vs. 33.8 experimental); 4 soluciones buenas con m₁ ≈ 0.68 meV.
> v3: criterio restaurado a cost < 1.0 con reporte dual de cuencas y 50 seeds → 8/50 en Cuenca A, m₁ ≈ 0.
> Justificación física del criterio: la Cuenca B no reproduce el ratio Δm²₃₁/Δm²₂₁ experimental y se reporta como diagnóstico, no como solución.

## 4. Resultados de referencia

Tras congelar seeds e hiperparámetros, correr el pipeline completo una vez y guardar en `resultados_referencia/`:
- `tabla_D2.csv`, `tabla_E1.csv`, `tabla12.csv`
- `cuenca_A_soluciones.json` (los 8 vectores de parámetros con sus costs)
- `etapa13_hitchin_results.json` (resultado negativo, se conserva)
- Log completo de cada corrida (`*.log`) con timestamp y `nvidia-smi`/CPU info

## 5. Manifiesto y entorno

```bash
pip freeze > requirements.lock.txt
conda env export > environment.yml
sha256sum scripts/*.py resultados_referencia/* > MANIFIESTO_SHA256.txt
```

**Ejecutado 2026-07-06/07:** `environment.yml` de este repo es el export real del entorno de producción `g2_tcs`; el manifiesto (18 entradas) y el zip congelado están publicados en el pre-registro — DOI [10.5281/zenodo.21231245](https://doi.org/10.5281/zenodo.21231245), SHA-256 del zip `a09e88ae17d3675e3de22405c39d3a6366566a99027abd4daf83a717c03d8165`.

## 6. Tolerancias de verificación (para CI y para terceros)

- CPU + seed fija: igualdad exacta de los CSV.
- Otro hardware/BLAS: |Δ relativo| < 1e-4 en valores centrales de tablas; las predicciones físicas (m₁, Σmᵢ, m_β, m_ββ) deben coincidir al nivel de su incertidumbre declarada (±0.23 meV para Σmᵢ, valores corregidos 2026-07; ver pre-registro DOI 10.5281/zenodo.21231245).

## 7. Orden de publicación

1. Completar §1–§6 de esta lista.
2. Subir a GitHub (repo público) — el repo puede preceder al pulido; lo perfecto es enemigo del timestamp.
3. Conectar el repo a Zenodo (integración GitHub→Zenodo) para que el release v1.0 genere el DOI del código automáticamente.
4. Referenciar ese DOI en el pre-registro y viceversa.
