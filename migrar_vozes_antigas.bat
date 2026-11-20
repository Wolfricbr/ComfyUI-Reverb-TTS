@echo off
set "OLD=C:\AI\ComfyUI\ComfyUI\output\Reverb\voices"
set "NEW=C:\AI\ComfyUI-Data\models\reverb_tts\voices_custom"
if not exist "%OLD%" (echo Nenhuma biblioteca antiga encontrada. & pause & exit /b 0)
if not exist "%NEW%" mkdir "%NEW%"
echo Copiando perfis NPZ sem sobrescrever arquivos de destino...
robocopy "%OLD%" "%NEW%" *.npz /XC /XN /XO
if %ERRORLEVEL% GEQ 8 echo Falha na copia. Verifique as pastas.
echo Concluido. Arquivos originais permanecem intactos.
pause
