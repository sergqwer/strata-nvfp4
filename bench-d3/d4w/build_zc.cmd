@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_120 -Xcompiler "/arch:AVX2 /EHsc" -o D:\Projects\strata-dec3\bench-d3\d4w\zc_dma.exe D:\Projects\strata-dec3\bench-d3\d4w\zc_dma.cu
