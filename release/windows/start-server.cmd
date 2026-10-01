@echo off
setlocal
rem The OpenAI / Anthropic-compatible server and a chat page on http://127.0.0.1:8080
rem Paths and engine flags: config\strata-nvfp4.json. Extra arguments go to serve.server (e.g. --port 9000).
cd /d "%~dp0"
if not exist models\pack\experts.bin (
  echo The model is not prepared yet: run prepare-model.cmd first.
  exit /b 1
)
set "PY=.venv-serve\Scripts\python.exe"
if not exist "%PY%" (
  echo Creating .venv-serve ...
  python -m venv .venv-serve || exit /b 1
  "%PY%" -m pip install --quiet -r requirements-serve.txt || exit /b 1
)
"%PY%" -m serve.server --engine strata --config config\strata-nvfp4.json --port 8080 %*
