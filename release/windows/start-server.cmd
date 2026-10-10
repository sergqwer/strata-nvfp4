@echo off
setlocal
rem The OpenAI / Anthropic-compatible server and a chat page on http://127.0.0.1:8080
rem Paths and engine flags: config\strata-qwen-nvfp4-gptq.json. Extra arguments go to serve.server (e.g. --port 9000).
cd /d "%~dp0"
set "CFG=config\strata-qwen-nvfp4-gptq.json"
if exist models\qwen-nvfp4-gptq\pack\experts.bin.sha256 goto :serve
if exist models\pack\experts.bin if exist config\strata-nvfp4.json goto :orca
echo The model is not prepared yet: run prepare-model.cmd first.
exit /b 1

:orca
rem a bundle from before 0.1.41-nvfp4.5 converted orca-nvfp4: it keeps working with its own config
set "CFG=config\strata-nvfp4.json"
echo Note: this folder has orca-nvfp4, which this fork withdrew in 0.1.41-nvfp4.5 - as an agent it did much worse
echo than the original Qwen. It keeps working. prepare-model.cmd downloads the replacement, the original Qwen with a
echo censorship switch, and this script then starts that one instead.

:serve
set "PY=.venv-serve\Scripts\python.exe"
if not exist "%PY%" (
  echo Creating .venv-serve ...
  python -m venv .venv-serve || exit /b 1
  "%PY%" -m pip install --quiet -r requirements-serve.txt || exit /b 1
)
"%PY%" -m serve.server --engine strata --config %CFG% --port 8080 %*
