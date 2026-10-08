@echo off
rem Updates this folder to the newest release of sergqwer/strata-nvfp4 - the engine, the server and the tools -
rem after checking the download against the SHA-256 GitHub publishes. config\, data\, models\ and the .venv-*
rem folders stay; the replaced program goes to .previous\. Close the server first. --check only says what is new.
rem The update replaces this file too, so the last line runs Python and ends the script in one line.
setlocal
cd /d "%~dp0"
python tools\bundle_update.py %* && (pause & exit /b 0) || (pause & exit /b 1)
