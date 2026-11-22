@echo off
rem build_b.cmd SRC [targets...] - a dev build of a 0.2.15-based Strata worktree into SRC\build-c: sm_120 only (the W4A4
rem unit gets 120a from CMake), the vendored strata-ggml, MMQ on, tests on; default target strata.  Run it through
rem gpu_lock.sh so a build never overlaps a benchmark (the CPU pool's timings are part of every decode number).
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
set "SRC=%~1"
shift
set "TGT="
:args
if "%~1"=="" goto done
set "TGT=%TGT% --target %~1"
shift
goto args
:done
if "%TGT%"=="" set "TGT=--target strata"
if not exist "%SRC%\build-c\build.ninja" (
  cmake -G Ninja -S "%SRC%" -B "%SRC%\build-c" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON ^
        -DCMAKE_CUDA_ARCHITECTURES=120-real "-DSTRATA_GGML_DIR=" -DSTRATA_MMQ=ON >nul || exit /b 1
)
cmake --build "%SRC%\build-c" %TGT% -j 16
exit /b %errorlevel%
