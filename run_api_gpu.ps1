# 本地 HTTP API（FastAPI）：默认 http://0.0.0.0:8088
# 在仓库根目录执行： .\run_api_gpu.ps1
# 可选：$env:GLMTTS_API_HOST="127.0.0.1"; $env:GLMTTS_API_PORT="8088"
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
. .\env_gpu.ps1
python -m tools.api_server
