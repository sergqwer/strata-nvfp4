@echo off
rem build_w.cmd MAXT - build_dev.cmd's dev build (sm_120, MMQ on, tests on) of strata-dec3 with -DSTRATA_WINDOW_MAX_T=MAXT
rem into build-wMAXT (E1).  Run through gpu_lock.sh.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
set "SRC=D:\Projects\strata-dec3"
set "B=%SRC%\build-w%~1"
set "TGT=--target strata"
if not "%~2"=="" set "TGT=--target %~2"
if not exist "%B%\build.ninja" (
  cmake -G Ninja -S "%SRC%" -B "%B%" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON ^
        -DCMAKE_CUDA_ARCHITECTURES=120-real "-DSTRATA_GGML_DIR=" -DSTRATA_MMQ=ON -DSTRATA_WINDOW_MAX_T=%~1 >nul || exit /b 1
)
cmake --build "%B%" %TGT% -j 16
exit /b %errorlevel%
