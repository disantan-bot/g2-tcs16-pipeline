#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculadora_claseB.py — Fuente única de los valores de Clase B del preregistro G2-TCS-16.

Calcula las consecuencias cinemáticas (B1: Suma m_i, B2: m_beta, B3: m_betabeta)
a partir de las predicciones estructurales de Clase A (NO, m1=0, delta=pi, alpha21=0,
alpha31=pi) y los parámetros de oscilación de NuFIT leídos desde nufit_params.json.

Motivación: los documentos del proyecto divergieron (59.41 vs 58.8 meV) porque el
pipeline del Tomo II leyó el splitting atmosférico de NuFIT (Dm2_3l) como Dm2_32,
mientras que NuFIT lo define como Dm2_31 para ordenamiento normal. Este script
calcula AMBAS convenciones y todas las versiones de NuFIT declaradas, de modo que
ningún documento vuelva a citar números de origen ambiguo.

Sin dependencias externas (solo stdlib). Determinista. Uso:

    python calculadora_claseB.py            # tabla completa
    python calculadora_claseB.py --json     # salida JSON

Fórmulas (m1 = 0, NO):
    m2 = sqrt(Dm2_21)
    m3 = sqrt(Dm2_31)            [convención NuFIT: Dm2_3l = Dm2_31 para NO]
    m3 = sqrt(Dm2_32 + m2^2)     [si el valor se interpreta como Dm2_32]
    B1: Suma = m1 + m2 + m3
    B2: m_beta = sqrt( sum_i |U_ei|^2 m_i^2 )
              = sqrt( s12^2 c13^2 Dm2_21 + s13^2 Dm2_31 )   (con m1=0)
    B3: m_bb = | c12^2 c13^2 m1 + s12^2 c13^2 m2 e^{i a21} + s13^2 m3 e^{i(a31 - 2 delta)} |
             = | s12^2 c13^2 m2 - s13^2 m3 |                 (m1=0, a21=0, a31=pi, delta=pi)

Autor: Diego Santana S. (generado en sesión de trabajo asistida, jul 2026).
Licencia: MIT.
"""
import argparse
import cmath
import json
import math
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
PARAMS_DEFAULT = os.path.join(AQUI, "nufit_params.json")

# Fases de Clase A (radianes) — predicción estructural del marco
ALPHA_21 = 0.0
ALPHA_31 = math.pi
DELTA_CP = math.pi
M1_EV = 0.0  # m1 = 0 exacto (operacionalmente < 1 meV)


def masas(dm2_21, dm2_atm, convencion):
    """Devuelve (m1, m2, m3) en eV para m1=0 según la convención del splitting."""
    m2 = math.sqrt(dm2_21)
    if convencion == "Dm2_31":
        m3 = math.sqrt(dm2_atm)
    elif convencion == "Dm2_32":
        m3 = math.sqrt(dm2_atm + m2 * m2)
    else:
        raise ValueError(convencion)
    return M1_EV, m2, m3


def observables(p, convencion):
    """Calcula B1, B2, B3 (en meV) para un set de parámetros NuFIT."""
    s12sq, s13sq = p["sin2_theta12"], p["sin2_theta13"]
    c12sq, c13sq = 1.0 - s12sq, 1.0 - s13sq
    m1, m2, m3 = masas(p["dm2_21"], p["dm2_3l"], convencion)

    suma = (m1 + m2 + m3) * 1e3

    m_beta = math.sqrt(
        c12sq * c13sq * m1 * m1 + s12sq * c13sq * m2 * m2 + s13sq * m3 * m3
    ) * 1e3

    # m_betabeta con fases explícitas (general, por si se cambian las fases de Clase A)
    termino = (
        c12sq * c13sq * m1
        + s12sq * c13sq * m2 * cmath.exp(1j * ALPHA_21)
        + s13sq * m3 * cmath.exp(1j * (ALPHA_31 - 2.0 * DELTA_CP))
    )
    m_bb = abs(termino) * 1e3

    return {
        "m1_meV": m1 * 1e3,
        "m2_meV": round(m2 * 1e3, 3),
        "m3_meV": round(m3 * 1e3, 3),
        "B1_suma_meV": round(suma, 2),
        "B2_m_beta_meV": round(m_beta, 2),
        "B3_m_betabeta_meV": round(m_bb, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Calculadora Clase B — G2-TCS-16")
    ap.add_argument("--params", default=PARAMS_DEFAULT, help="ruta a nufit_params.json")
    ap.add_argument("--json", action="store_true", help="salida JSON")
    args = ap.parse_args()

    with open(args.params, encoding="utf-8") as f:
        catalogo = json.load(f)

    resultados = {}
    for nombre, p in catalogo["fits"].items():
        resultados[nombre] = {
            conv: observables(p, conv) for conv in ("Dm2_31", "Dm2_32")
        }

    if args.json:
        json.dump(
            {"fases_claseA": {"alpha21": "0", "alpha31": "pi", "delta": "pi", "m1": "0"},
             "resultados": resultados},
            sys.stdout, indent=2, ensure_ascii=False)
        print()
        return

    print("=" * 78)
    print("Calculadora Clase B — G2-TCS-16   (m1=0, NO, delta=pi, alpha21=0, alpha31=pi)")
    print("=" * 78)
    for nombre, p in catalogo["fits"].items():
        print(f"\n### {nombre} — {p['descripcion']}")
        print(f"    Dm2_21 = {p['dm2_21']:.3e} eV^2 | Dm2_3l = {p['dm2_3l']:.3e} eV^2"
              f" | s12^2 = {p['sin2_theta12']} | s13^2 = {p['sin2_theta13']}")
        for conv, tag in (("Dm2_31", "convención NuFIT (Dm2_3l = Dm2_31)  [CORRECTA]"),
                          ("Dm2_32", "lectura como Dm2_32                 [la usada en Tomo II]")):
            r = resultados[nombre][conv]
            print(f"    {tag}")
            print(f"        m2 = {r['m2_meV']} meV, m3 = {r['m3_meV']} meV")
            print(f"        B1 Suma m_i    = {r['B1_suma_meV']} meV")
            print(f"        B2 m_beta      = {r['B2_m_beta_meV']} meV")
            print(f"        B3 m_betabeta  = {r['B3_m_betabeta_meV']} meV")
    print("\n" + "=" * 78)
    print("Nota: la fila [CORRECTA] bajo el fit vigente es la que debe citarse en")
    print("preregistro, preprint y Compendio. Ver Reconciliacion_ClaseB_59.41_vs_58.8.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
