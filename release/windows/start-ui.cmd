@echo off
setlocal
rem The new interface (http://127.0.0.1:8791) for a model server that something else runs: the tray, or start-server.cmd
rem started elsewhere. start-server.cmd starts the interface by itself; this is for the other cases.
rem   start-ui.cmd http://127.0.0.1:8092/v1     the server's address (or set STRATA_ATTACH_URL)
rem The API key never goes on a command line: set STRATA_ATTACH_KEY, or name a file holding it in STRATA_ATTACH_KEY_FILE.
rem Without an address it uses this folder's server (config\strata-qwen-nvfp4-gptq.json: its port and key).
rem The interface sends the key only while the server answers, and nothing while the page sits idle, so a tray can
rem unload its model as usual; only a chat you send may wake it. Close this window to stop the interface.
cd /d "%~dp0"
if not exist node\node.exe (
  echo The new interface needs node\node.exe, which this folder does not have - the server's own page still works.
  exit /b 1
)
if not "%~1"=="" set "STRATA_ATTACH_URL=%~1"
if not defined STRATA_ATTACH_URL if not defined STRATA_ATTACH_CONFIG set "STRATA_ATTACH_CONFIG=%~dp0config\strata-qwen-nvfp4-gptq.json"
if not defined STRATA_HOME set "STRATA_HOME=%LOCALAPPDATA%\strata-nvfp4\app-ui"
set "STRATA_MODELS_DIR=%STRATA_HOME%\models"
set "STRATA_PORT=8791"
set "STRATA_BIND=127.0.0.1"
set "STRATA_UI_MODE=browser"
set "STRATA_UI_REASON=chosen"
set "STRATA_REAL_ENGINE=1"
set "STRATA_NO_ACQUIRE=1"
set "STRATA_NO_BUNDLED_ENGINE=1"
set "STRATA_FORK_ROOT=%~dp0."
if exist .venv-serve\Scripts\python.exe set "STRATA_FORK_PYTHON=%~dp0.venv-serve\Scripts\python.exe"
if defined STRATA_ATTACH_URL (echo New interface: http://127.0.0.1:8791  ^(attached to %STRATA_ATTACH_URL%^)) else (echo New interface: http://127.0.0.1:8791  ^(attached to this folder's server^))
echo Close this window to stop it.
if not "%STRATA_APP_UI_OPEN%"=="0" start "" /b powershell -NoProfile -NonInteractive -WindowStyle Hidden -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8791/api/health | Out-Null; Start-Process 'http://127.0.0.1:8791/'; break}catch{Start-Sleep -Milliseconds 500}}"
"node\node.exe" app\host\server\index.mjs --dist app\host\dist
