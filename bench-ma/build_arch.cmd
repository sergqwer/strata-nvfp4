@echo off
rem build_arch.cmd ARCHS DIR [targets...] - portable release-style build of this worktree for ARCHS (e.g. 86-virtual) into ..\DIR
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
set "SRC=D:\Projects\strata-multiarch"
set "ARCHS=%~1"
set "B=%SRC%\%~2"
shift
shift
set "TGT="
:args
if "%~1"=="" goto done
set "TGT=%TGT% --target %~1"
shift
goto args
:done
if "%TGT%"=="" set "TGT=--target strata"
if not exist "%B%\build.ninja" (
  cmake -G Ninja -S "%SRC%" -B "%B%" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON ^
        -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=%ARCHS%" "-DSTRATA_GGML_DIR=" -DSTRATA_MMQ=ON || exit /b 1
)
cmake --build "%B%" %TGT% -j 16
exit /b %errorlevel%
