# TOE R16.7 — Claims Ledger

## Green

1. **Reversibilidad finita implementada.**  
   La regla de evolución usa una construcción reversible de segundo orden:
   \[
   (x_{t-1}, x_t) \mapsto (x_t, F(x_t)\oplus x_{t-1})
   \]
   con inversa exacta.

2. **Biyectividad verificada por enumeración completa para N=2.**  
   El espacio completo de 256 estados se transforma en 256 imágenes únicas.

3. **Recuperación exacta por inversión temporal.**  
   Para varias semillas y pasos, la evolución hacia adelante seguida de la inversa recupera el par inicial bit a bit.

## Amber

1. **Conservación de información en sentido restringido.**  
   Se verifica como reversibilidad/biyectividad de un sistema finito. Esto no equivale todavía a una derivación física completa.

2. **Pipeline geométrico auditable.**  
   El estimador de dimensión espectral devuelve un valor cercano a 2 sobre un toro 2D plantado. Esto valida el instrumento, no la emergencia de geometría desde primeros principios.

3. **Separación entre microinformación y macroobservables.**  
   La densidad de Hamming puede cambiar aunque la transformación global sea reversible. Esto evita confundir conservación informacional con conservación de cada estadística macroscópica.

## Open

1. Derivar contenido de campos del Modelo Estándar.
2. Derivar masas, acoplamientos o constantes físicas.
3. Construir un límite continuo Lorentziano.
4. Derivar gravedad clásica o cuántica.
5. Producir una predicción física nueva, falsable y cuantitativa.
6. Conectar el núcleo reversible con datos experimentales.