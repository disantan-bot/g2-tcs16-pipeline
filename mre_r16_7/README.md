# TOE R16.7 — Implementación mínima reproducible

**Estado:** paquete mínimo reproducible, no una solución del TOE.

Este paquete implementa un núcleo computacional reducido para auditar una hipótesis del programa TOE:

> La información fundamental no se crea ni se destruye; una dinámica física admisible debe poder modelarse, al nivel microscópico, como una transformación reversible/biyectiva. La geometría observable se extrae como un observable efectivo sobre una estructura discreta.

La implementación busca que un referee externo pueda correr el experimento sin contexto adicional.

---

## Qué valida este MRE

1. **Dinámica reversible exacta:** implementa una regla celular binaria de segundo orden:
   \[
   x_{t+1} = F(x_t) \oplus x_{t-1}
   \]
   con inversa exacta:
   \[
   x_{t-1} = F(x_t) \oplus x_{t+1}
   \]

2. **Conservación de información en sentido finito:** para un sistema pequeño se enumera todo el espacio de estados y se verifica que la transformación global es una permutación.

3. **Recuperación exacta por evolución inversa:** para múltiples semillas aleatorias, avanzar y retroceder recupera el estado inicial bit a bit.

4. **Observable geométrico reproducible:** sobre un toro 2D discreto se calcula un laplaciano de grafo y un estimador de dimensión espectral. El resultado esperado está cercano a 2 en una ventana intermedia de escala.

---

## Qué NO valida

Este MRE **no** demuestra todavía:

- que el Modelo Estándar emerja;
- que las constantes físicas se deriven;
- que la gravedad clásica emerja;
- que haya predicciones nuevas confirmadas;
- que el TOE esté resuelto.

El objetivo de R16.7 es más modesto: dejar un núcleo ejecutable y auditable que fuerce precisión sobre el postulado de conservación informacional y el pipeline geométrico.

---

## Instalación

Requiere Python 3.10+ y `numpy`.

```bash
pip install -r requirements.txt
```

---

## Corrida principal

```bash
python run_verifier.py
```

Salida esperada:

- `outputs/results_R16_7.json`
- `outputs/report_R16_7.md`

---

## Tests opcionales

```bash
python -m unittest discover tests
```

---

## Interpretación rápida

Si todo pasa, el claim mínimo queda en **Green**:

> existe una implementación reversible finita, reproducible, con inversa exacta y verificación de permutación sobre el espacio completo para N pequeño.

El claim geométrico queda en **Amber**:

> el pipeline geométrico recupera una dimensión espectral cercana a 2 para un toro 2D plantado, pero eso todavía no prueba emergencia física desde primeros principios.

Los claims fuertes del TOE quedan **Open**.