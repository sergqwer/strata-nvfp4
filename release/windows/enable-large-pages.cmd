@echo off
rem Lets this Windows account use 2 MB large pages for the 63 GiB expert arena (asks for admin).
rem Afterwards SIGN OUT AND BACK IN, or reboot. "enable-large-pages.cmd -Check" tells whether it is active,
rem "enable-large-pages.cmd -Revoke" takes it back. See README.md, "Large pages".
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\enable-large-pages.ps1" %*
