# Gradio Web：http://0.0.0.0:8048
# 在仓库根目录执行： .\run_gradio_gpu.ps1
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
. .\env_gpu.ps1
python -m tools.gradio_app
