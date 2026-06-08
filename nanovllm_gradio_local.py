from __future__ import annotations

import base64
import gc
import hmac
import logging
import os
import sys
import tempfile
import time
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

from nanovllm_voxcpm import VoxCPM

load_dotenv()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.7,max_split_size_mb:1024"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (voxcpm-nanovllm-gradio-local) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voxcpm-nanovllm-gradio-local")


MODEL_PATH = "../devuno/models/vox_cpm2"
VOICES_DIR = "./voices"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SAMPLE_RATE = 48000
TTS_API_KEY = os.getenv("TTS_API_KEY", "")

os.makedirs(VOICES_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    logger.error("Critical error: nano-vLLM requires an NVIDIA CUDA GPU.")
    sys.exit(1)

model: Any = None
LATENTS_MEM_CACHE: dict[str, bytes] = {}


def is_api_key_valid(api_key: str | None) -> bool:
    """Validate the provided API key against TTS_API_KEY from the environment."""
    if not TTS_API_KEY:
        logger.warning("TTS_API_KEY is not configured. Rejecting request.")
        return False
    return hmac.compare_digest(str(api_key or ""), TTS_API_KEY)


async def load_model() -> None:
    """Load VoxCPM into the Gradio worker loop before the first synthesis call."""
    global model

    if model is not None:
        return

    logger.info("Initializing AsyncVoxCPM2ServerPool from: %s", MODEL_PATH)
    try:
        model = VoxCPM.from_pretrained(
            model=MODEL_PATH,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            max_model_len=2048,
            gpu_memory_utilization=0.60,
            enforce_eager=False,
            devices=[0],
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Waiting for GPU server readiness...")
        await model.wait_for_ready()
        logger.info("Nano-vLLM-VoxCPM is loaded and ready on the local GPU.")
    except Exception as exc:
        logger.error("Failed to load the local accelerated engine: %s", exc)
        raise


async def stop_model() -> None:
    """Stop the distributed VoxCPM workers when Gradio unloads the app."""
    global model

    if model is None:
        return

    logger.info("Stopping VoxCPM distributed server pool...")
    await model.stop()
    model = None
    logger.info("VoxCPM workers stopped.")


def parse_float(value: Any, default: float) -> float:
    """Parse a float request field with a safe fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_seed(value: Any) -> int | None:
    """Parse an optional integer seed."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid seed received: %s. Using random generation.", value)
        return None


def write_wav_bytes(wav: np.ndarray) -> bytes:
    """Serialize a generated waveform to in-memory WAV bytes."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(tmp_fd)
        sf.write(tmp_path, wav, SAMPLE_RATE)
        with open(tmp_path, "rb") as wav_file:
            return wav_file.read()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def load_reference_latents(voice_id: str | None, ref_b64: str | None) -> bytes | None:
    """Resolve reference voice latents from memory, disk cache, voice file, or base64 audio."""
    global LATENTS_MEM_CACHE

    if voice_id:
        voice_path = os.path.join(VOICES_DIR, voice_id)
        latent_path = voice_path + ".latents"

        if voice_id in LATENTS_MEM_CACHE:
            logger.info("[Cache L1 - Memory] Using pre-encoded latents for: '%s'", voice_id)
            return LATENTS_MEM_CACHE[voice_id]

        if os.path.exists(latent_path):
            logger.info("[Cache L2 - Disk] Found saved cache for '%s'. Loading...", voice_id)
            try:
                with open(latent_path, "rb") as latent_file:
                    latents = latent_file.read()
                LATENTS_MEM_CACHE[voice_id] = latents
                logger.info("[Cache L1 - Mapped] Voice '%s' loaded into memory.", voice_id)
                return latents
            except Exception as exc:
                logger.warning("Failed to read disk cache for '%s': %s. Re-encoding...", voice_id, exc)

        if os.path.exists(voice_path):
            with open(voice_path, "rb") as voice_file:
                wav_content = voice_file.read()

            logger.info("Voice '%s' has no cache. Encoding latents for the first time...", voice_id)
            latents = await model.encode_latents(wav_content, "wav")

            try:
                with open(latent_path, "wb") as latent_file:
                    latent_file.write(latents)
                logger.info("[Cache L2 - Saved] Cache written to: %s", latent_path)
            except Exception as exc:
                logger.warning("Failed to save disk cache for '%s': %s", voice_id, exc)

            LATENTS_MEM_CACHE[voice_id] = latents
            logger.info("[Cache L1 - Created] Latents stored in memory.")
            return latents

        logger.warning("[Local Volume] Voice '%s' does not exist in %s. Ignoring clone.", voice_id, VOICES_DIR)
        return None

    if ref_b64:
        logger.info("Encoding base64 reference audio into memory latents...")
        ref_bytes = base64.b64decode(ref_b64)
        latents = await model.encode_latents(ref_bytes, "wav")
        logger.info("Base64 reference audio encoded successfully.")
        return latents

    logger.info("Zero-shot voice design mode.")
    return None


async def synthesize_audio(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate a base64 WAV response from a JSON-compatible payload.

    Args:
        payload: Request dictionary with text, control, voice_id, cfg_value, and seed fields.

    Returns:
        JSON-compatible response matching the previous FastAPI payload shape.
    """
    if model is None:
        await load_model()

    logger.info("================ RECEIVED NANO-VLLM GRADIO REQUEST ================")
    start_time = time.time()

    text = str(payload.get("text", "") or "")
    control = str(payload.get("control", "") or "")
    voice_id = payload.get("voice_id") or None
    ref_b64 = payload.get("reference_audio_base64") or None
    cfg_value = parse_float(payload.get("cfg_value", 2.0), 2.0)
    seed = parse_seed(payload.get("seed"))

    logger.info("Local Args - Voice ID: %s, Seed: %s, CFG: %s", voice_id, seed, cfg_value)

    try:
        ref_audio_latents = await load_reference_latents(voice_id, ref_b64)
    except Exception as exc:
        logger.error("Failed to encode reference audio: %s", exc)
        return {"error": f"Failed to encode reference audio: {exc}"}

    logger.info("Synthesizing audio through the async generator...")
    final_text = f"({control}){text}" if control else text

    try:
        gen_start = time.time()
        chunks = []

        async for chunk in model.generate(
            target_text=final_text,
            ref_audio_latents=ref_audio_latents,
            cfg_value=cfg_value,
            seed=seed,
        ):
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("No audio chunks were generated by the engine.")

        wav = np.concatenate(chunks, axis=0)
        logger.info("Accelerated synthesis completed in %.2fs.", time.time() - gen_start)
    except Exception as exc:
        logger.error("Accelerated audio generation failed: %s", exc)
        return {"error": f"Audio generation failed: {exc}"}

    wav_bytes = write_wav_bytes(wav)
    encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")

    total_time = time.time() - start_time
    logger.info("Request completed in the local Gradio sandbox in %.2fs.", total_time)
    logger.info("=========================================================")

    return {
        "wav_base64": encoded_audio,
        "sample_rate": SAMPLE_RATE,
        "seed_used": seed,
        "cfg_value_used": cfg_value,
    }


async def generate_api(api_key: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Gradio API endpoint compatible with gradio_client calls."""
    if not is_api_key_valid(api_key):
        logger.warning("Rejected TTS request due to invalid API key.")
        return {"error": "Invalid API key."}
    if payload is None:
        payload = {}
    return await synthesize_audio(payload)


with gr.Blocks(title="VoxCPM Nano-vLLM Local Gradio API") as demo:
    gr.Markdown("# VoxCPM Nano-vLLM Local Gradio API")
    gr.Markdown("Use the `/generate_api` endpoint with an API key and JSON payload.")

    api_key_input = gr.Textbox(label="API Key", type="password")
    request_payload = gr.JSON(
        label="Request Payload",
        value={
            "text": "Starting",
            "control": "American male executive voice. Clear studio recording.",
            "voice_id": "voz_1.wav",
            "cfg_value": 4.0,
            "seed": 44,
        },
    )
    response_payload = gr.JSON(label="Response Payload")

    gr.Button("Generate").click(
        fn=generate_api,
        inputs=[api_key_input, request_payload],
        outputs=response_payload,
        api_name="generate_api",
    )

if hasattr(demo, "unload"):
    demo.unload(stop_model)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        share=False,
    )
