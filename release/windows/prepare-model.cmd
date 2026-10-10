@echo off
setlocal
rem Downloads the model ready-made into models\: qwen-nvfp4-gptq, the original Qwen3.8-Flash-Next with every expert
rem NVFP4 by this fork's GPTQ (Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata, at the revision model-files.json
rem pins, as setup.py installs it), and the image encoder. Every file is checked against its SHA-256; nothing is
rem converted. Needs Python 3.11+ on PATH and ~130 GB free on this drive. A stopped run resumes: start it again.
rem prepare-model.cmd --check only says what is there.
cd /d "%~dp0"
set "PY=.venv-model\Scripts\python.exe"
set "HF_XET_HIGH_PERFORMANCE=1"
set "PYTHONUTF8=1"

if not exist "%PY%" (
  echo Creating .venv-model and installing huggingface_hub ...
  python -m venv .venv-model || goto :fail
  "%PY%" -m pip install --quiet --upgrade pip
  "%PY%" -m pip install --quiet -r requirements-model.txt || goto :fail
)

"%PY%" tools\bundle_model.py %* || goto :fail
echo.
echo Done. Start the model with start-server.cmd.
exit /b 0

:fail
echo.
echo FAILED - see the messages above. Run prepare-model.cmd again to resume.
exit /b 1
