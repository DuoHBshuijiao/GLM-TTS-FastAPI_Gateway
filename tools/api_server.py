# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Local HTTP API for GLM-TTS (FastAPI). Run from repo root:
#   python -m tools.api_server
import asyncio
import base64
import io
import os
import tempfile
import threading
import wave

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from glmtts_inference import InferenceCancelled

from .tts_service import clear_memory, run_inference_api

app = FastAPI(
    title="GLM-TTS Local API",
    description="Zero-shot TTS: upload reference wav + text, receive synthesized wav.",
    version="1.0.0",
)


async def _synthesize_until_disconnect_or_cancel(
    request: Request,
    *,
    prompt_text: str,
    prompt_audio_path: str,
    input_text: str,
    seed: int,
    sample_rate: int,
    use_cache: bool,
):
    """
    Run sync inference in a worker thread while polling for client disconnect.
    If the client disconnects first, set cancel_event so generate_long can exit
    between sentence chunks; then return None (caller should respond with 499).
    """
    cancel_event = threading.Event()

    async def _run_synth():
        return await asyncio.to_thread(
            run_inference_api,
            prompt_text,
            prompt_audio_path,
            input_text,
            seed,
            sample_rate,
            use_cache,
            cancel_event,
        )

    async def _poll_disconnect():
        while True:
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.2)

    synth_task = asyncio.create_task(_run_synth())
    poll_task = asyncio.create_task(_poll_disconnect())

    done, _pending = await asyncio.wait(
        {synth_task, poll_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    if synth_task in done:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        try:
            return await synth_task
        except InferenceCancelled:
            return None

    cancel_event.set()
    try:
        return await synth_task
    except InferenceCancelled:
        return None


@app.get("/health")
def health():
    return {"status": "ok", "service": "glm-tts"}


@app.post("/api/v1/tts")
async def tts_multipart(
    request: Request,
    prompt_text: str = Form(..., description="与参考音频对应的文本"),
    input_text: str = Form(..., description="要合成的目标文本"),
    prompt_audio: UploadFile = File(..., description="参考音色 wav"),
    seed: int = Form(42),
    sample_rate: int = Form(24000, description="24000 或 32000"),
    use_cache: bool = Form(True),
):
    """
    multipart/form-data 上传参考音频，返回 **WAV**（application/octet-stream 或 audio/wav）。
    """
    suffix = os.path.splitext(prompt_audio.filename or "")[1].lower()
    if suffix not in (".wav", ".flac", ""):
        raise HTTPException(400, "prompt_audio: use .wav or .flac")

    data = await prompt_audio.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav")
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        result = await _synthesize_until_disconnect_or_cancel(
            request,
            prompt_text=prompt_text,
            prompt_audio_path=tmp.name,
            input_text=input_text,
            seed=seed,
            sample_rate=sample_rate,
            use_cache=use_cache,
        )
        if result is None:
            return Response(status_code=499)
        sr, audio_i16 = result
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="glm_tts_out.wav"'},
    )


class TtsJsonBody(BaseModel):
    """JSON 请求体（参考音频为服务端本机路径）。"""

    prompt_text: str = Field("", description="与参考音频对应的文本")
    input_text: str = Field(..., description="要合成的目标文本")
    prompt_audio_path: str = Field(..., description="服务器上的 wav 绝对路径")
    seed: int = 42
    sample_rate: int = 24000
    use_cache: bool = True


@app.post("/api/v1/tts/json")
async def tts_json(request: Request, body: TtsJsonBody):
    """JSON 调用（`prompt_audio_path` 必须为 **API 进程所在机器** 上的文件路径）。"""
    try:
        result = await _synthesize_until_disconnect_or_cancel(
            request,
            prompt_text=body.prompt_text,
            prompt_audio_path=body.prompt_audio_path,
            input_text=body.input_text,
            seed=body.seed,
            sample_rate=body.sample_rate,
            use_cache=body.use_cache,
        )
        if result is None:
            return Response(status_code=499)
        sr, audio_i16 = result
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setframerate(sr)
        wf.setsampwidth(2)
        wf.writeframes(audio_i16.tobytes())
    buf.seek(0)
    b64 = base64.standard_b64encode(buf.read()).decode("ascii")
    return JSONResponse(
        {
            "sample_rate": sr,
            "format": "pcm_s16le_mono_wav_base64",
            "audio_wav_base64": b64,
        }
    )


@app.post("/api/v1/clear_vram")
def api_clear_vram():
    msg = clear_memory()
    return {"ok": True, "message": msg}


def main():
    host = os.environ.get("GLMTTS_API_HOST", "0.0.0.0")
    port = int(os.environ.get("GLMTTS_API_PORT", "8088"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
