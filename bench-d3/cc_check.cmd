@echo off
rem cc_check.cmd FILE.cu... - compile the given kernel files for the release targets (75/86/89 PTX+SASS, 120-real), objects to bench-d3\cc
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
if not exist D:\Projects\strata-dec3\bench-d3\cc mkdir D:\Projects\strata-dec3\bench-d3\cc
set RC=0
:loop
if "%~1"=="" goto done
for %%A in (75 86 89 120) do (
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -forward-unknown-to-host-compiler -DSTRATA_HAVE_DFLASH2=1 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERIFY_MAX_T=8 -ID:\Projects\strata-dec3\third_party\ggml -ID:\Projects\strata-dec3\include -D_WINDOWS -Xcompiler=" /EHsc" -Xcompiler="-O2 -Ob2" -DNDEBUG -std=c++20 --generate-code=arch=compute_%%A,code=sm_%%A -Xcompiler=-MD --use_fast_math -x cu -c D:\Projects\strata-dec3\src\kernels\cuda\%~1 -o D:\Projects\strata-dec3\bench-d3\cc\%~n1_%%A.obj && echo OK %~1 sm_%%A || (echo FAIL %~1 sm_%%A & set RC=1)
)
shift
goto loop
:done
exit /b %RC%
