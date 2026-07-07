# R16.7 — Implementation Notes

## Núcleo elegido

Se eligió una dinámica celular reversible de segundo orden porque separa dos cosas que a menudo se confunden:

1. **Reversibilidad microscópica:** la transformación global es invertible.
2. **Cambio macroscópico:** observables agregados, como densidad de Hamming, pueden variar.

Esto calza con la tesis informacional débil:

> la información fundamental se transforma, no se crea ni se destruye.

## Regla implementada

Sea \(x_t\) una grilla binaria \(N \times N\) con condiciones periódicas.

\[
F(x_t) = x_t \oplus roll_x^+(x_t) \oplus roll_x^-(x_t) \oplus roll_y^+(x_t) \oplus roll_y^-(x_t)
\]

La evolución es:

\[
x_{t+1}=F(x_t)\oplus x_{t-1}
\]

El estado microscópico completo es el par \((x_{t-1},x_t)\). La inversa existe de forma cerrada:

\[
x_{t-1}=F(x_t)\oplus x_{t+1}
\]

Por eso la transformación:

\[
T(x_{t-1},x_t)=(x_t,x_{t+1})
\]

es biyectiva.

## Verificación

El verificador realiza cuatro chequeos:

1. Enumeración completa del espacio de estados para \(N=2\).
2. Confirmación de 256 imágenes únicas para 256 estados.
3. Pruebas aleatorias de forward/backward para \(N=16\), 64 pasos, 25 semillas.
4. Estimación de dimensión espectral para un toro 2D plantado.

## Resultado local de esta corrida

- Permutación completa N=2: pasa.
- Entropía uniforme antes/después: 8 bits / 8 bits.
- Reversibilidad aleatoria: pasa.
- Dimensión espectral central: ~2.0969.

## Lectura científica honesta

R16.7 convierte el postulado informacional en una pieza ejecutable y auditable.
No convierte todavía el programa TOE en teoría física completa.

El próximo salto no es agregar complejidad arbitraria, sino reemplazar la geometría plantada por una geometría inducida por la dinámica, y luego buscar invariantes no triviales.