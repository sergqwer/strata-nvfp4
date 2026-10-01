@echo off
setlocal
rem Downloads the ModelOpt NVFP4 checkpoint and converts it for engine\strata.exe into models\.
rem Needs Python 3.11+ on PATH and ~300 GB free on this drive; models\checkpoint (126 GB) can go afterwards.
rem Each step is skipped when its output already exists, so a failed run can simply be started again.
cd /d "%~dp0"
set "PY=.venv-convert\Scripts\python.exe"
set "HF=.venv-convert\Scripts\hf.exe"
set "HF_XET_HIGH_PERFORMANCE=1"
set "PYTHONUTF8=1"

if not exist "%PY%" (
  echo Creating .venv-convert and installing the converter's packages ...
  python -m venv .venv-convert || goto :fail
  "%PY%" -m pip install --quiet --upgrade pip
  "%PY%" -m pip install -r requirements-convert.txt || goto :fail
)

echo [1/7] the checkpoint, 126 GB ...
"%HF%" download jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4 --local-dir models\checkpoint || goto :fail

echo [2/7] the n-gram (PLE) table in FP8, byte for byte as shipped, 51 GB ...
if not exist models\ple-fp8.gguf (
  "%PY%" tools\ple_fp8_pack.py --model models\checkpoint --out models\ple-fp8.gguf || goto :fail
)

echo [3/7] the token embedding in BF16, as shipped, 1.3 GB ...
if not exist models\token-embd-bf16.gguf (
  "%PY%" tools\embd_bf16_pack.py --model models\checkpoint --out models\token-embd-bf16.gguf || goto :fail
)

echo [4/7] the image encoder (the checkpoint's vision tower) in FP32, 1.8 GB ...
if not exist models\mmproj-f32.gguf (
  "%PY%" third_party\llama.cpp\convert_hf_to_gguf.py models\checkpoint --mmproj --outtype f32 ^
         --outfile models\mmproj-f32.gguf || goto :fail
)

echo [5/7] the GGUF: NVFP4 experts repacked without loss, 69 GB ...
if not exist models\orca-nvfp4.gguf (
  "%PY%" tools\nvfp4_convert.py --model models\checkpoint --outfile models\orca-nvfp4.gguf || goto :fail
)

echo [6/7] the pack: experts.bin 63 GB and the tokenizer ...
if not exist models\pack\experts.bin (
  "%PY%" tools\iq_pack.py --gguf models\orca-nvfp4.gguf --out models\pack || goto :fail
)

echo [7/7] the fine-tune's own MTP draft head ...
if not exist models\mtp\rt\experts.bin (
  "%PY%" tools\mtp_extract.py --model models\checkpoint --out models\mtp || goto :fail
  "%PY%" tools\mtp_pack.py --src models\mtp --experts q2_0 --out models\mtp\mtp-q2_0.gguf || goto :fail
  "%PY%" tools\mtp_rt.py --gguf models\mtp\mtp-q2_0.gguf --out models\mtp\rt || goto :fail
)
copy /y data\draft_vocab.bin models\mtp\rt\ >nul

echo.
echo Done. Start the model with start-server.cmd.
echo models\checkpoint (126 GB) is no longer needed and can be deleted.
exit /b 0

:fail
echo.
echo FAILED - see the messages above. Run prepare-model.cmd again to resume.
exit /b 1
