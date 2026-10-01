@echo off
rem Builds the portable image encoder (tools/vision: llama.cpp mtmd, CPU only, AVX2) into build-vision-cpu\.
rem release/make_windows_bundle.py runs it beside build-release.cmd.  CPU only on purpose: measured against an
rem FP32 reference, the CPU encoder (FP32 weights) is 0.1% off, ggml-cuda's (flash attention in FP16) up to 11%,
rem and a GPU encoder keeps ~1.6 GB of VRAM from the expert cache (docs/NVFP4.md, "Images").
setlocal
set "ROOT=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo build-vision: vswhere.exe not found - install the VS 2022 Build Tools & exit /b 1)
"%VSWHERE%" -latest -products * -property installationPath > "%TEMP%\strata-vs-path.txt" || exit /b 1
set /p VS=<"%TEMP%\strata-vs-path.txt"
if not defined VS (echo build-vision: no Visual Studio installation found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
if not exist "%ROOT%\build-vision-cpu\CMakeCache.txt" (
  cmake -G Ninja -S "%ROOT%\tools\vision" -B "%ROOT%\build-vision-cpu" -DCMAKE_BUILD_TYPE=Release ^
        -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON "-DLLAMA_DIR=%ROOT%\third_party\llama.cpp" || exit /b 1
)
cmake --build "%ROOT%\build-vision-cpu" --target strata-vision -j 16 || exit /b 1
exit /b 0
