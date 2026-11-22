@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
set RC=0
for %%A in (75 86 89 120) do (
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -std=c++17 -O3 -gencode arch=compute_%%A,code=sm_%%A -ID:\Projects\strata-vision-gpu\tools\vision\own -ID:\Projects\strata-vision-gpu\include -c D:\Projects\strata-vision-gpu\tools\vision\own\vit_gpu.cu -o D:\Projects\strata-vision-gpu\bench-vision\cc\vit_%%A.obj >D:\Projects\strata-vision-gpu\bench-vision\cc\vit_%%A.log 2>&1 && (echo sm_%%A ok) || (echo sm_%%A FAILED & set RC=1)
)
exit /b %RC%
