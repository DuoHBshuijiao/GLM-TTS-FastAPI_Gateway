# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Shared TTS inference for Gradio UI and HTTP API (tools/api_server.py).
import logging
import os
import gc
import threading
from typing import Optional

import numpy as np
import torch
import gradio as gr

from glmtts_inference import (
    load_models,
    generate_long,
    DEVICE,
    InferenceCancelled,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_CACHE = {
    "loaded": False,
    "sample_rate": None,
    "components": None,
}


def get_models(use_phoneme=False, sample_rate=24000):
    if MODEL_CACHE["loaded"] and MODEL_CACHE["sample_rate"] == sample_rate:
        return MODEL_CACHE["components"]

    logging.info(f"Loading models with sample_rate={sample_rate}...")

    if MODEL_CACHE["components"]:
        del MODEL_CACHE["components"]
        gc.collect()
        torch.cuda.empty_cache()

    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
        use_phoneme=use_phoneme,
        sample_rate=sample_rate,
    )

    MODEL_CACHE["components"] = (frontend, text_frontend, speech_tokenizer, llm, flow)
    MODEL_CACHE["sample_rate"] = sample_rate
    MODEL_CACHE["loaded"] = True
    logging.info("Models loaded successfully.")
    return MODEL_CACHE["components"]


def _generate_audio(
    prompt_text: str,
    prompt_audio_path: str,
    input_text: str,
    seed: int,
    sample_rate: int,
    use_cache: bool,
    cancel_event: Optional[threading.Event] = None,
):
    """Core generation; raises ValueError on bad input; other exceptions propagate."""
    if not input_text or not str(input_text).strip():
        raise ValueError("input_text is empty")
    if not prompt_audio_path or not os.path.isfile(prompt_audio_path):
        raise ValueError("prompt_audio_path is missing or not a file")

    frontend, text_frontend, _, llm, flow = get_models(
        use_phoneme=True, sample_rate=sample_rate
    )

    norm_prompt_text = text_frontend.text_normalize(prompt_text) + " "
    norm_input_text = text_frontend.text_normalize(input_text)

    logging.info(f"Normalized Prompt: {norm_prompt_text}")
    logging.info(f"Normalized Input: {norm_input_text}")

    prompt_text_token = frontend._extract_text_token(norm_prompt_text)
    prompt_speech_token = frontend._extract_speech_token([prompt_audio_path])
    speech_feat = frontend._extract_speech_feat(prompt_audio_path, sample_rate=sample_rate)
    embedding = frontend._extract_spk_embedding(prompt_audio_path)

    cache_speech_token_list = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(cache_speech_token_list, dtype=torch.int32).to(DEVICE)

    cache = {
        "cache_text": [norm_prompt_text],
        "cache_text_token": [prompt_text_token],
        "cache_speech_token": cache_speech_token_list,
        "use_cache": use_cache,
    }

    tts_speech, _, _, _ = generate_long(
        frontend=frontend,
        text_frontend=text_frontend,
        llm=llm,
        flow=flow,
        text_info=["", norm_input_text],
        cache=cache,
        embedding=embedding,
        flow_prompt_token=flow_prompt_token,
        speech_feat=speech_feat,
        sample_method="ras",
        seed=int(seed),
        device=DEVICE,
        use_phoneme=False,
        cancel_event=cancel_event,
    )

    audio_data = tts_speech.squeeze().cpu().numpy()
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_data * 32767.0).astype(np.int16)
    return int(sample_rate), audio_int16


def run_inference(
    prompt_text,
    prompt_audio_path,
    input_text,
    seed,
    sample_rate,
    use_cache=True,
):
    """Gradio handler: returns (sample_rate, audio_int16)."""
    if not input_text:
        raise gr.Error("Please provide text to synthesize.")
    if not prompt_audio_path:
        raise gr.Error("Please upload a prompt audio file.")
    if not prompt_text:
        gr.Warning("Prompt text is empty. Results might be suboptimal.")

    try:
        return _generate_audio(
            prompt_text or "",
            prompt_audio_path,
            input_text,
            int(seed),
            int(sample_rate),
            bool(use_cache),
        )
    except gr.Error:
        raise
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        import traceback

        traceback.print_exc()
        raise gr.Error(f"Inference failed: {str(e)}")


def run_inference_api(
    prompt_text: str,
    prompt_audio_path: str,
    input_text: str,
    seed: int = 42,
    sample_rate: int = 24000,
    use_cache: bool = True,
    cancel_event: Optional[threading.Event] = None,
):
    """HTTP API: same return as core; raises ValueError / InferenceCancelled / Exception."""
    return _generate_audio(
        prompt_text or "",
        prompt_audio_path,
        input_text,
        int(seed),
        int(sample_rate),
        bool(use_cache),
        cancel_event=cancel_event,
    )


def clear_memory():
    global MODEL_CACHE
    if MODEL_CACHE["components"]:
        del MODEL_CACHE["components"]
    MODEL_CACHE["components"] = None
    MODEL_CACHE["loaded"] = False
    MODEL_CACHE["sample_rate"] = None
    gc.collect()
    torch.cuda.empty_cache()
    return "Memory cleared. Models will reload on next inference."
