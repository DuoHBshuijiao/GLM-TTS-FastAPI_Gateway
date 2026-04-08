# 在启动 GLM-TTS 前 dot-source 本脚本，为当前会话注入：UTF-8、pip 安装的 CUDA/cuDNN bin 路径（若已安装）、可选 ONNX GPU。
# 用法（在仓库根目录 E:\GLM-TTS）：
#   . .\env_gpu.ps1
#
# 若尚未安装 pip 版 cuDNN / CUDA 运行库，请先执行：
#   pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12

$ErrorActionPreference = "Stop"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python 不在 PATH 中，请先激活 .venv： .\.venv\Scripts\Activate.ps1"
    return
}

$pylib = python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
foreach ($sub in @(
        "nvidia\cudnn\bin",
        "nvidia\cuda_runtime\bin",
        "nvidia\cublas\bin"
    )) {
    $p = Join-Path $pylib $sub
    if (Test-Path $p) {
        $env:PATH = "$p;$env:PATH"
    }
}

$env:PYTHONUTF8 = "1"
# 若已按文档安装 cuDNN 等到 PATH，可启用 ONNX GPU（CampPlus）；否则保持注释或设为 0。
if (-not $env:GLMTTS_ONNX_GPU) {
    $env:GLMTTS_ONNX_GPU = "1"
}

Write-Host "env_gpu.ps1: PYTHONUTF8=$env:PYTHONUTF8 GLMTTS_ONNX_GPU=$env:GLMTTS_ONNX_GPU" -ForegroundColor Green
