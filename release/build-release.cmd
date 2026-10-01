@echo off
rem Builds the portable release engine (STRATA_PORTABLE=ON) into build-release\, with code for RTX 20, 30, 40 and 50
rem (sm_75, 86, 89 and 120; CMake builds the one W4A4 unit for 120a itself; tested on an RTX 5090, the others through
rem STRATA_EMULATE_CC - docs/NVFP4.md).
rem release/make_windows_bundle.py runs it before taking the engine, so a bundle never ships a stale exe.
setlocal
set "ROOT=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo build-release: vswhere.exe not found - install the VS 2022 Build Tools & exit /b 1)
"%VSWHERE%" -latest -products * -property installationPath > "%TEMP%\strata-vs-path.txt" || exit /b 1
set /p VS=<"%TEMP%\strata-vs-path.txt"
if not defined VS (echo build-release: no Visual Studio installation found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
rem configured on every run: an existing build-release\ from before takes the architecture list too
cmake -G Ninja -S "%ROOT%" -B "%ROOT%\build-release" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON ^
      -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=75-real;86-real;89-real;120-real" ^
      "-DSTRATA_GGML_DIR=%ROOT%\third_party\llama.cpp" >nul || exit /b 1
cmake --build "%ROOT%\build-release" --target strata -j 16 || exit /b 1
exit /b 0
