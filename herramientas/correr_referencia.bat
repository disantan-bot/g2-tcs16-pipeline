@echo off
REM Corrida de referencia canonica del pipeline G2-TCS-16
REM Ejecutar desde la raiz del repo, con el entorno g2_tcs activado:
REM   conda activate g2_tcs
REM   herramientas\correr_referencia.bat
REM Duracion estimada: horas. Puede dejarse corriendo durante la noche.

setlocal enabledelayedexpansion
cd /d "%~dp0.."
if not exist resultados_referencia\corrida_produccion mkdir resultados_referencia\corrida_produccion

echo Entorno: > resultados_referencia\corrida_produccion\_entorno.txt
python -c "import sys, numpy, scipy; print(sys.version); print('numpy', numpy.__version__, '| scipy', scipy.__version__)" >> resultados_referencia\corrida_produccion\_entorno.txt

for %%f in (scripts\*.py) do (
    echo [%%time%%] Corriendo %%~nf ...
    python "%%f" > "resultados_referencia\corrida_produccion\%%~nf.log" 2>&1
    if errorlevel 1 (echo   ERROR en %%~nf) else (echo   OK %%~nf)
)
echo COMPLETO. Logs en resultados_referencia\corrida_produccion\
