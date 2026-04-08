# GLM-TTS：Windows + NVIDIA GPU 完整运行说明

面向已安装 **NVIDIA 驱动** 且 `nvcc` 可用（例如 **CUDA 12.4**）的环境。PyTorch 使用官方提供的 **CUDA 12.1 预编译包（cu121）**，与 12.x 驱动兼容，无需与本机 `nvcc` 小版本完全一致。

仓库路径以下以 `E:\GLM-TTS` 为例，请按你的实际路径替换。

---

## 1. 前置条件

- Windows 10/11，**NVIDIA GPU**，驱动支持 CUDA（`nvidia-smi` 可见）。
- **Python 3.10–3.12**（与官方 README 一致）。
- 已克隆本仓库，且 **`ckpt/`** 中已有完整权重（若未下载，见第 4 节）。
- **`frontend/campplus.onnx`** 已在仓库内（随 Git 提供）。

---

## 2. 新建虚拟环境并安装 GPU 版 PyTorch

在 **PowerShell** 中执行：

```powershell
cd E:\GLM-TTS

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"
```

**先安装带 CUDA 的 PyTorch 2.3.1**（与项目 `requirements.txt` 版本一致）。若曾装过 CPU 版，请先：`pip uninstall -y torch torchaudio torchvision` 再执行：

```powershell
pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

**验证 GPU 是否被 PyTorch 识别：**

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

应输出 `cuda: True` 以及你的显卡名称。

---

## 3. 安装其余依赖（含 ONNX-GPU、Gradio、FastAPI）

若此前装过 **CPU 版** `onnxruntime`，建议先卸载再装 GPU 版：

```powershell
pip uninstall -y onnxruntime onnxruntime-gpu 2>$null
pip install -r requirements-gpu-windows.txt
```

说明：

- **`requirements-gpu-windows.txt`** 使用 **`onnxruntime-gpu`**。
- FastAPI / Uvicorn 已由 `requirements-gpu-windows.txt` 中的依赖带入；本地 API 使用 **`python -m tools.api_server`**。

---

## 4. （可选）为 ONNX Runtime GPU 安装 cuDNN / CUDA 运行库（pip）

`onnxruntime-gpu` 的 CUDA 执行提供程序需要能加载 **cuDNN 9.x**、**CUDA 12.x** 相关 DLL。可用 NVIDIA 官方 **pip 轮子** 装到当前 venv，并用脚本加入 `PATH`（见第 8 节 `env_gpu.ps1`）。

在已激活的 `.venv` 中执行：

```powershell
pip install --upgrade nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12
```

然后**每次**打开新终端准备跑 Gradio/API 前，在仓库根目录执行（或依赖 `run_gradio_gpu.ps1` / `run_api_gpu.ps1` 自动执行）：

```powershell
. .\env_gpu.ps1
```

该脚本会把 venv 下 `site-packages\nvidia\...\bin`  prepend 到当前会话的 `PATH`，并设置 `GLMTTS_ONNX_GPU=1`（让 CampPlus ONNX 尝试走 GPU）。

若仍报 DLL 加载失败，请安装最新 **VC++ x64 可再发行组件**：  
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist  

---

## 5. 下载模型权重（若尚未下载）

```powershell
cd E:\GLM-TTS
$env:PYTHONUTF8 = "1"
pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-TTS', local_dir='ckpt')"
```

国内可选 ModelScope（需先 `pip install modelscope`）：

```powershell
modelscope download --model ZhipuAI/GLM-TTS --local_dir ckpt
```

---

## 6. 命令行推理（CLI）

```powershell
cd E:\GLM-TTS
.\.venv\Scripts\Activate.ps1
. .\env_gpu.ps1
```

### 6.1 中文示例

```powershell
python glmtts_inference.py --data=example_zh --exp_name=_test --use_cache
```

输出目录：`outputs/pretrain_test/example_zh/`（含 WAV 与 JSONL）。

### 6.2 英文示例

```powershell
python glmtts_inference.py --data=example_en --exp_name=_test --use_cache
```

### 6.3 采样率与音素

```powershell
python glmtts_inference.py --data=example_zh --exp_name=_test --use_cache --sample_rate 24000
python glmtts_inference.py --data=example_zh --exp_name=_test --use_cache --use_phoneme
```

### 6.4 多 GPU

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python glmtts_inference.py --data=example_zh --exp_name=_test --use_cache
```

---

## 7. Web 界面（Gradio）

**推荐**使用仓库根目录脚本（已激活 venv、加载 `env_gpu.ps1`）：

```powershell
cd E:\GLM-TTS
.\run_gradio_gpu.ps1
```

或手动：

```powershell
cd E:\GLM-TTS
.\.venv\Scripts\Activate.ps1
. .\env_gpu.ps1
python -m tools.gradio_app
```

默认监听 **`http://0.0.0.0:8048`**，本机浏览器打开：  
[http://127.0.0.1:8048](http://127.0.0.1:8048)

---

## 8. 环境脚本说明（`env_gpu.ps1`）

| 项目 | 说明 |
|------|------|
| 位置 | 仓库根目录 `env_gpu.ps1` |
| 用法 | `cd E:\GLM-TTS` 后执行：`. .\env_gpu.ps1`（注意前面的点空格） |
| `PYTHONUTF8` | 避免部分控制台下 Unicode 问题 |
| `PATH` | 若已通过 pip 安装 `nvidia-cudnn-cu12` 等，自动把对应 `bin` 加入 PATH |
| `GLMTTS_ONNX_GPU` | 默认设为 `1`；若只想让 ONNX 用 CPU，可先执行 `$env:GLMTTS_ONNX_GPU="0"` 再 dot-source，或编辑脚本 |

---

## 9. 本地 HTTP API（FastAPI）

实现文件：`tools/api_server.py`，推理逻辑与 Gradio 共用 `tools/tts_service.py`。

### 9.1 启动服务

```powershell
cd E:\GLM-TTS
.\run_api_gpu.ps1
```

或：

```powershell
.\.venv\Scripts\Activate.ps1
. .\env_gpu.ps1
python -m tools.api_server
```

- 默认地址：**`http://127.0.0.1:8088`**
- 监听 **`0.0.0.0`**，便于局域网调用。
- 可选环境变量：`GLMTTS_API_HOST`（默认 `0.0.0.0`）、`GLMTTS_API_PORT`（默认 `8088`）。

**交互式文档（Swagger）：**  
[http://127.0.0.1:8088/docs](http://127.0.0.1:8088/docs)

### 9.2 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/tts` | `multipart/form-data`：上传参考 wav，返回合成 **WAV 二进制** |
| POST | `/api/v1/tts/json` | JSON：`prompt_audio_path` 为**服务器本机路径**，返回 base64 WAV |
| POST | `/api/v1/clear_vram` | 清空模型缓存与显存占用（与 Gradio「清显存」类似） |

### 9.3 curl：multipart 上传参考音频

将 `jiayan_zh.wav` 换为你的参考音文件路径：

```powershell
curl.exe -X POST "http://127.0.0.1:8088/api/v1/tts" `
  -F "prompt_text=他当时还跟线下其他的站姐吵架，然后，打架进局子了。" `
  -F "input_text=我最爱吃人参果，你喜欢吃吗？" `
  -F "prompt_audio=@E:\GLM-TTS\examples\prompt\jiayan_zh.wav" `
  -F "seed=42" `
  -F "sample_rate=24000" `
  -F "use_cache=true" `
  --output "out.wav"
```

### 9.4 curl：JSON（音频路径在服务端本地）

仅当 API 与文件在同一台机器、路径有效时可用：

```powershell
curl.exe -X POST "http://127.0.0.1:8088/api/v1/tts/json" `
  -H "Content-Type: application/json" `
  -d "{\"prompt_text\":\"他当时还跟线下其他的站姐吵架，然后，打架进局子了。\",\"input_text\":\"我最爱吃人参果，你喜欢吃吗？\",\"prompt_audio_path\":\"E:/GLM-TTS/examples/prompt/jiayan_zh.wav\",\"seed\":42,\"sample_rate\":24000,\"use_cache\":true}"
```

响应为 JSON，字段 `audio_wav_base64` 为整段 WAV 的 base64，可自行解码保存。

### 9.5 Python 请求示例（multipart）

```python
import pathlib
import requests

url = "http://127.0.0.1:8088/api/v1/tts"
files = {"prompt_audio": open(r"E:\GLM-TTS\examples\prompt\jiayan_zh.wav", "rb")}
data = {
    "prompt_text": "他当时还跟线下其他的站姐吵架，然后，打架进局子了。",
    "input_text": "我最爱吃人参果，你喜欢吃吗？",
    "seed": "42",
    "sample_rate": "24000",
    "use_cache": "true",
}
r = requests.post(url, files=files, data=data, timeout=600)
r.raise_for_status()
pathlib.Path("out_api.wav").write_bytes(r.content)
print("saved out_api.wav")
```

需安装：`pip install requests`。

### 9.6 Gradio 自带的 HTTP API（可选）

服务启动后，Gradio 也会暴露队列与组件相关接口；若需用 **Python 客户端** 调 Gradio 界面上的逻辑，可使用官方 **`gradio_client`**（版本需与当前 `gradio` 匹配），在浏览器打开 Web UI 后按其文档配置 `Client` 指向 `http://127.0.0.1:8048`。  
**程序化集成更推荐第 9 节的 FastAPI**，路径与字段更稳定。

---

## 10. 常见问题

| 现象 | 处理 |
|------|------|
| `torch.cuda.is_available()` 为 False | 确认安装的是 **cu121** 的 `torch`（勿被 `already satisfied` 骗成 CPU 包）；必要时先 `pip uninstall torch torchaudio torchvision` 再按第 2 节重装。 |
| ONNX：`LoadLibrary failed 126`、缺 cuDNN | 按第 3–4 节安装 pip 版 `nvidia-cudnn-cu12` 等，并执行 `. .\env_gpu.ps1`；安装 VC++ 运行库。若仍失败，可设 `$env:GLMTTS_ONNX_GPU="0"`，仅 CampPlus 用 CPU。 |
| 显存不足 | 关闭其他占显存程序；缩短文本或参考音频；或调用 `/api/v1/clear_vram`。 |
| WeTextProcessing 不可用 | Windows 常见。当前工程在无 `pynini` 时使用恒等归一化，可跑通但数字/符号读法可能变差。 |

---

## 11. 与官方 README 的对应关系

| 官方 README | 本文 |
|-------------|------|
| `pip install -r requirements.txt` | 第 2–3 节：`torch cu121` + `requirements-gpu-windows.txt` |
| `python glmtts_inference.py ...` | 第 6 节 |
| `python -m tools.gradio_app` | 第 7 节（端口 **8048**） |
| （无） | 第 9 节：本地 FastAPI（默认 **8088**） |

---

## 12. 仓库中的便捷脚本

| 文件 | 作用 |
|------|------|
| `env_gpu.ps1` | 设置 `PYTHONUTF8`、prepend pip 的 NVIDIA `bin` 到 `PATH`、`GLMTTS_ONNX_GPU` |
| `run_gradio_gpu.ps1` | 激活 venv → `env_gpu.ps1` → 启动 Gradio（8048） |
| `run_api_gpu.ps1` | 激活 venv → `env_gpu.ps1` → 启动 FastAPI（8088） |

**注意：** 必须在仓库根目录 `E:\GLM-TTS` 下执行上述脚本，且已创建 `.venv` 并完成依赖安装。
