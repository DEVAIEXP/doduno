import base64
import logging
import os
import sys
import tempfile
import time

import modal
from fastapi import HTTPException, Request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (voxcpm2-modal-gpu) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voxcpm2-modal-gpu")

cuda_version = "12.6.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

GITHUB_USER = "DEVAIEXP"
GITHUB_BRANCH = "local-seed"
MODAL_APP_NAME = "voxcpm2-nanovllm-service"
MODAL_VOICES_VOLUME_NAME = "voxcpm-voices"
MODAL_SECRET_NAME = "voxcpm-secrets"


def download_model() -> None:
    """Download VoxCPM2 weights into the Modal image cache."""
    from huggingface_hub import snapshot_download

    print("Downloading VoxCPM2 weights into the image cache...", flush=True)
    snapshot_download("openbmb/VoxCPM2", local_dir="models/vox_cpm2")


voxcpm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi[standard]",
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "torchcodec",
        "transformers>=4.51.0",
        "einops",
        "inflect",
        "addict",
        "wetext",
        "modelscope>=1.22.0",
        "datasets>=3,<4",
        "huggingface-hub",
        "pydantic",
        "tqdm",
        "simplejson",
        "sortedcontainers",
        "soundfile",
        "numpy",
        "librosa",
        "matplotlib",
        "triton>=3.0.0",
        "funasr",
        "spaces",
        "argbind",
        "safetensors",
        "https://huggingface.co/DEVAIEXP/wheels/resolve/main/flash_attn-2.8.2-cp310-cp310-linux_x86_64.whl",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .run_function(download_model)
    .run_commands(
        f"pip install --no-deps git+https://github.com/{GITHUB_USER}/nanovllm-voxcpm.git@{GITHUB_BRANCH} --no-cache-dir # v1"
    )
)

app = modal.App(MODAL_APP_NAME)
voices_volume = modal.Volume.from_name(MODAL_VOICES_VOLUME_NAME, create_if_missing=True)


@app.cls(
    image=voxcpm_image,
    gpu="L4",
    timeout=300,
    scaledown_window=500,
    volumes={"/voices": voices_volume},
)
class VoxCPMService:
    @modal.enter()
    async def load_model(self) -> None:
        """Initialize the async VoxCPM engine when the container starts."""
        logger.info("================ LOADING CLOUD GPU ENGINE ================")
        from nanovllm_voxcpm import VoxCPM

        self.latents_mem_cache: dict[str, bytes] = {}
        self.model = VoxCPM.from_pretrained(
            model="models/vox_cpm2",
            max_num_batched_tokens=8192,
            max_num_seqs=16,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            enforce_eager=False,
            devices=[0],
        )

        logger.info("Waiting for the GPU worker pool to become ready...")
        await self.model.wait_for_ready()
        logger.info("Nano-vLLM-VoxCPM is ready for audio requests.")
        logger.info("==========================================================")

    @modal.exit()
    async def shutdown(self) -> None:
        """Stop distributed GPU workers when the container shuts down."""
        if self.model:
            logger.info("Stopping GPU workers cleanly...")
            await self.model.stop()

    @modal.method()
    async def generate_speech(
        self,
        text: str,
        control: str = "",
        ref_bytes: bytes | None = None,
        voice_id: str | None = None,
        cfg_value: float = 2.0,
        seed: int | None = None,
    ) -> bytes:
        """Generate WAV audio bytes from text and optional reference audio."""
        import numpy as np
        import soundfile as sf

        logger.info(f"--- [Generation Start] Text: '{text[:50]}...' ---")
        logger.info(f"Parameters - Control: '{control}', Seed: {seed}, CFG: {cfg_value}")

        ref_audio_latents = None

        if voice_id:
            volume_path = f"/voices/{voice_id}"
            latent_path = volume_path + ".latents"

            if voice_id in self.latents_mem_cache:
                ref_audio_latents = self.latents_mem_cache[voice_id]
                logger.info(f"[Cache L1 - RAM] Using in-memory latents for '{voice_id}'")
            elif os.path.exists(latent_path):
                logger.info(f"[Cache L2 - Volume] Loading saved latents for '{voice_id}'")
                try:
                    with open(latent_path, "rb") as f:
                        ref_audio_latents = f.read()
                    self.latents_mem_cache[voice_id] = ref_audio_latents
                except Exception as exc:
                    logger.warning(f"Failed reading latent cache for '{voice_id}': {exc}. Re-encoding...")
                    ref_audio_latents = None

            if ref_audio_latents is None:
                if os.path.exists(volume_path):
                    with open(volume_path, "rb") as f:
                        wav_content = f.read()

                    logger.info(f"Encoding voice '{voice_id}' for the first time...")
                    ref_audio_latents = await self.model.encode_latents(wav_content, "wav")

                    try:
                        with open(latent_path, "wb") as f:
                            f.write(ref_audio_latents)
                        voices_volume.commit()
                        logger.info(f"[Cache L2 - Saved] Persisted latents at {latent_path}")
                    except Exception as exc:
                        logger.warning(f"Failed writing latent cache for '{voice_id}': {exc}")

                    self.latents_mem_cache[voice_id] = ref_audio_latents
                else:
                    logger.warning(f"[Volume] Voice '{voice_id}' was not found in /voices. Skipping voice cloning.")
        elif ref_bytes:
            logger.info("Encoding dynamic reference audio into latents...")
            ref_audio_latents = await self.model.encode_latents(ref_bytes, "wav")
            logger.info("Dynamic reference audio encoded successfully.")

        final_text = f"({control}){text}" if control else text
        try:
            chunks = []
            async for chunk in self.model.generate(
                target_text=final_text,
                ref_audio_latents=ref_audio_latents,
                cfg_value=cfg_value,
                seed=seed,
            ):
                chunks.append(chunk)

            if not chunks:
                raise RuntimeError("The engine returned no audio chunks.")

            wav = np.concatenate(chunks, axis=0)
        except Exception as exc:
            logger.error(f"Accelerated audio generation failed: {exc}")
            raise

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            sf.write(tmp_path, wav, 48000)
            os.close(tmp_fd)
            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return wav_bytes


@app.function(image=voxcpm_image, secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)])
@modal.fastapi_endpoint(method="POST")
async def generate_api(data: dict, request: Request) -> dict[str, object]:
    """HTTP endpoint used by the game TTS client."""
    logger.info("================ RECEIVED API POST REQUEST ================")
    start_time = time.time()

    expected_token = os.environ.get("TTS_API_KEY")
    if expected_token:
        auth_header = request.headers.get("Authorization")
        token = auth_header.split(" ", 1)[1] if auth_header and auth_header.startswith("Bearer ") else None

        if token != expected_token:
            logger.warning("Unauthorized request to the GPU endpoint.")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid or missing TTS_API_KEY token.",
            )
        logger.info("API key authentication succeeded.")
    else:
        logger.warning("No TTS_API_KEY is configured in Modal. Endpoint is running in public mode.")

    text = data.get("text", "").strip()
    control = data.get("control", "").strip()
    voice_id = data.get("voice_id")
    ref_b64 = data.get("reference_audio_base64")

    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")

    cfg_value = data.get("cfg_value", 2.0)
    try:
        cfg_value = float(cfg_value)
    except (ValueError, TypeError):
        cfg_value = 2.0

    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            logger.warning(f"Invalid seed received: {seed}. Falling back to random generation.")
            seed = None

    logger.info(f"API args - Voice ID: {voice_id}, Seed: {seed}, CFG: {cfg_value}")

    ref_bytes = base64.b64decode(ref_b64) if ref_b64 else None
    service = VoxCPMService()
    wav_bytes = await service.generate_speech.remote.aio(
        text,
        control,
        ref_bytes=ref_bytes,
        voice_id=voice_id,
        cfg_value=cfg_value,
        seed=seed,
    )

    encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")

    total_time = time.time() - start_time
    logger.info(f"Request completed successfully in {total_time:.2f}s.")
    logger.info("===========================================================")

    return {
        "wav_base64": encoded_audio,
        "sample_rate": 48000,
        "seed_used": seed,
        "cfg_value_used": cfg_value,
    }
