from __future__ import annotations

import json
import os
import queue
import random
import threading
from collections.abc import Mapping
from typing import Any

import gradio as gr
from gradio_client import Client
import numpy as np
from dotenv import load_dotenv

from components import Board, NeonToast
from game_manager import (
    BOT_NAME,
    LLM_API_KEY,
    APP_UI,    
    CRISES_DATABASE,
    SYNC_RATE_LEADERBOARD_SECONDS,
    SYNC_RATE_PLAYER_SECONDS,
    SYNC_RATE_SPECTATOR_SECONDS,
    TICK_LOBBY_WARMUP_SECONDS,
    TICK_RATE_SERVER_SECONDS,
    GameState,
    ServerResponse,
    get_optional_env_secret,
    global_server,
    llm_queue,
)
from inference_mapper import EndpointConfig, get_endpoint_chain, mark_endpoint_failed, mark_endpoint_success
from manual import render_how_to_play_html
from prompts import BOT_SYSTEM_PROMPT, DIRECTOR_SYSTEM_PROMPT

load_dotenv(override=True)


def sanitize_hf_token_environment() -> None:
    """Keep Gradio OAuth from using app dataset credentials or stale .env tokens."""
    os.environ.pop("HF_TOKEN", None)


sanitize_hf_token_environment()

# Highest safe seed value accepted by llama.cpp's int32 seed path.
MAX_SEED = np.iinfo(np.int32).max
# Development switch that skips model download/loading during smoke tests.
LLM_DISABLED: bool = os.getenv("DOD_DISABLE_LLM", "").lower() in {"1", "true", "yes"}
# Development switch that skips Director voice synthesis while keeping quote text.
TTS_DOWNLOAD_DISABLED: bool = os.getenv("DOD_DISABLE_TTS", "").lower() in {"1", "true", "yes"}


def randomize_seed_fn(generation_seed: int, randomize_seed: bool) -> int:
    """Return either the provided seed or a randomized int32-safe seed.

    Args:
        generation_seed: Existing seed to reuse when randomization is disabled.
        randomize_seed: Whether a new seed should be generated.

    Returns:
        Seed value suitable for llama.cpp generation calls.
    """
    if randomize_seed:
        generation_seed = random.randint(0, MAX_SEED)
    return generation_seed


tts_audio_queue: queue.Queue[dict[str, Any]] = queue.Queue()
HF_AUTH_PLAYER_NAMES: set[str] = set()


def create_llm_client(
    endpoint: EndpointConfig,
    timeout_override: float | None = None,
    ip_token: str = "",
) -> Client:
    """Create an isolated Gradio client for one LLM request.

    Args:
        endpoint: Resolved endpoint configuration from the mapper.
        timeout_override: Optional HTTP timeout for warmup calls.
        ip_token: Hugging Face ZeroGPU IP token forwarded from the user request.

    Returns:
        Gradio client connected to the endpoint URL.
    """
    url = endpoint["url"]
    timeout = float(timeout_override if timeout_override is not None else endpoint.get("timeout", 120.0))
    headers = {"x-ip-token": ip_token} if ip_token else None
    hf_space_token = get_optional_env_secret("HF_SPACE_TOKEN")
    print(
        f"[LLM Client] Connecting to {endpoint.get('name', 'endpoint')}: {url} "
        f"(zero_gpu_token={'yes' if ip_token else 'no'})",
        flush=True,
    )
    return Client(url, token=hf_space_token or None, headers=headers, httpx_kwargs={"timeout": timeout})


def predict_llm(
    system_prompt: str,
    user_payload: str,
    temperature: float,
    grammar_schema: str,
    use_warmup_timeout: bool = False,
    ip_token: str = "",
) -> str:
    """Call the mapped LLM primary endpoint and fallback endpoints.

    Args:
        system_prompt: System prompt sent to the inference service.
        user_payload: Serialized user payload.
        temperature: Generation temperature.
        grammar_schema: Serialized JSON schema passed to the service.
        ip_token: Hugging Face ZeroGPU IP token forwarded from the user request.

    Returns:
        Raw model response string.
    """
    last_error: Exception | None = None

    for endpoint in get_endpoint_chain("llm"):
        mode = endpoint.get("mode", "gradio")
        url = endpoint.get("url", "")
        api_name = endpoint.get("api_name") or "/generate_inference"

        if mode != "gradio":
            print(f"[LLM Client] Skipping unsupported LLM mode '{mode}' for {url}", flush=True)
            continue

        try:
            timeout_override = float(endpoint.get("warmup_timeout", endpoint.get("timeout", 120.0))) if use_warmup_timeout else None
            client = create_llm_client(endpoint, timeout_override, ip_token)
            print(f"[LLM Client] Calling {endpoint.get('name', 'endpoint')} via Gradio: {url}", flush=True)
            result = client.predict(
                LLM_API_KEY,
                system_prompt,
                user_payload,
                temperature,
                grammar_schema,
                api_name=api_name,
            )
            mark_endpoint_success("llm", endpoint)
            return result
        except Exception as exc:
            last_error = exc
            mark_endpoint_failed("llm", endpoint, str(exc))
            print(f"[LLM Client] Endpoint failed ({url}): {exc}", flush=True)

    raise RuntimeError(f"No LLM endpoint succeeded: {last_error}")


GLOBAL_CSS = """

html {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background-color: #070913 !important;
    background: #070913 !important;
    margin: 0 !important;
    padding: 0 !important;
}


body {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background-color: transparent !important;
    background: transparent !important;
    margin: 0 !important;
    padding: 0 !important;
}


.gradio-container {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: auto !important;
    background-color: transparent !important;
    background: transparent !important;
    scrollbar-width: thin;
    scrollbar-color: #44345d transparent;
}


.gradio-container::-webkit-scrollbar {
    width: 6px;
}
.gradio-container::-webkit-scrollbar-track {
    background: transparent;
}
.gradio-container::-webkit-scrollbar-thumb {
    background: #44345d;
    border-radius: 4px;
}
#bg_canvas {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -1 !important;
    pointer-events: none !important;
}

.glass-lobby {
    background: rgba(20, 15, 30, 0.6) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 30px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;

    max-width: 600px !important;
    width: 100% !important;
    margin: 40px auto 20px auto !important;
    box-sizing: border-box !important;
}

.lobby-logo {    
    max-width: 150px;
    margin: 0 auto 20px auto;
    display: block;
    /*filter: drop-shadow(0 0 10px rgba(0, 243, 255, 0.5));*/
}

#global_lang_bar {
    position: relative !important;
    z-index: 50 !important;
    width: min(100%, 1220px) !important;
    max-width: 1220px !important;
    margin: 6px auto -2px auto !important;
    padding: 0 18px !important;
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    box-sizing: border-box !important;
    background: transparent !important;
    border: 0 !important;
    min-height: 0 !important;
}

#global_lang_bar > div,
#global_lang_bar .form {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: max-content !important;
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}

#global_lang_bar .language-switch {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: max-content !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

#global_lang_bar .language-switch > label,
#global_lang_bar .language-switch .block-label,
#global_lang_bar .language-switch .label-container,
#global_lang_bar .language-switch .label,
#global_lang_bar .language-switch .label-wrap,
#global_lang_bar .language-switch .svelte-1gfkn6j,
#global_lang_bar .language-switch legend {
    display: none !important;
}

#global_lang_bar .language-switch fieldset,
#global_lang_bar .language-switch .container,
#global_lang_bar .language-switch .input-container,
#global_lang_bar .language-switch .wrap {
    width: auto !important;
    min-width: 0 !important;
    max-width: max-content !important;
    margin: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#global_lang_bar .language-switch .wrap,
#global_lang_bar .language-switch .radio-group {
    display: flex !important;
    align-items: center !important;
    gap: 0 !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: max-content !important;
    padding: 3px !important;
    border: 1px solid rgba(0, 243, 255, 0.38) !important;
    border-radius: 999px !important;
    background: rgba(7, 20, 38, 0.82) !important;
    box-shadow: 0 0 16px rgba(0, 243, 255, 0.14) !important;
    overflow: hidden !important;
}

#global_lang_bar .language-switch label {
    margin: 0 !important;
    border-radius: 999px !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

#global_lang_bar .language-switch input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

#global_lang_bar .language-switch label span {
    min-width: 58px !important;
    padding: 6px 11px !important;
    border-radius: 999px !important;
    color: #cbd5e0 !important;
    font-size: 12px !important;
    font-weight: 900 !important;
    text-align: center !important;
    text-transform: uppercase !important;
}

#global_lang_bar .language-switch input:checked + span,
#global_lang_bar .language-switch label:has(input:checked) span,
#global_lang_bar .language-switch label.selected span {
    background: linear-gradient(180deg, #00f3ff 0%, #00a8ff 100%) !important;
    color: #070913 !important;
    box-shadow: 0 0 12px rgba(0, 243, 255, 0.38) !important;
}

#lobby-title,
#lobby-title h1,
#lobby-title span,
#lobby-title .prose {
    text-align: center !important;
    display: block !important;
    width: 100% !important;
    margin: 0 auto !important;
}

/* 1. Primary Game Button (Chunky 3D Bevel with physical active click) */
.glass-lobby button.primary {
    background: linear-gradient(180deg, #00f3ff 0%, #00a8ff 100%) !important;
    color: #070913 !important;
    font-family: "Segoe UI", Arial, Helvetica, sans-serif !important;
    font-size: 14px !important;
    font-weight: 900 !important;
    line-height: 1.15 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    text-rendering: geometricPrecision !important;
    
    border-top: 3px solid #ffffff !important;
    border-left: 3px solid #00f3ff !important;
    border-bottom: 6px solid #005c8a !important;
    border-right: 6px solid #004566 !important;
    border-radius: 8px !important;
    
    box-shadow: 0 6px 15px rgba(0, 243, 255, 0.35) !important;
    transition: all 0.1s ease !important;
    transform: translateY(0px) !important;
}

.glass-lobby button.primary:active {
    transform: translateY(3px) !important;
    border-bottom: 2px solid #005c8a !important;
    border-right: 2px solid #004566 !important;
    box-shadow: 0 2px 5px rgba(0, 243, 255, 0.25) !important;
}

.manual-page {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 18px !important;
    color: #ffffff !important;
    font-family: "Segoe UI", Arial, Helvetica, sans-serif !important;
}

.manual-hero {
    border: 1.5px solid rgba(0, 243, 255, 0.5) !important;
    border-radius: 10px !important;
    padding: 22px !important;
    background: radial-gradient(circle at top, rgba(0, 243, 255, 0.14), rgba(7, 20, 38, 0.9) 55%, rgba(5, 6, 11, 0.95)) !important;
    box-shadow: 0 0 22px rgba(0, 243, 255, 0.15), inset 0 0 18px rgba(0, 0, 0, 0.45) !important;
}

.manual-kicker {
    color: #00f3ff !important;
    font-size: 12px !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

.manual-hero h2 {
    margin: 6px 0 8px 0 !important;
    color: #ffffff !important;
    font-size: 28px !important;
}

.manual-hero p,
.manual-panel p {
    color: #cbd5e0 !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
}

.manual-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)) !important;
    gap: 12px !important;
    margin-top: 12px !important;
}

.manual-panel {
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    background: rgba(7, 20, 38, 0.82) !important;
    padding: 16px !important;
    box-shadow: inset 0 0 16px rgba(0, 0, 0, 0.32) !important;
}

.manual-wide {
    margin-top: 12px !important;
}

.manual-panel h3 {
    margin: 0 0 9px 0 !important;
    color: #00f3ff !important;
    font-size: 17px !important;
}

.manual-list {
    margin: 0 !important;
    padding-left: 20px !important;
    color: #cbd5e0 !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
}

.manual-stack-row {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
}

.manual-stack-chip {
    display: inline-flex !important;
    align-items: center !important;
    min-height: 28px !important;
    padding: 4px 10px !important;
    border: 2px solid rgba(255, 255, 255, 0.75) !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    font-size: 12px !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.85) !important;
}

.manual-card-gallery {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 12px !important;
    align-items: flex-start !important;
    margin-top: 14px !important;
}

.manual-card-wrap {
    width: 116px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 6px !important;
}

.manual-card {
    position: relative !important;
    width: 100px !important;
    height: 140px !important;
    border-radius: 8px !important;
    border: 3.5px solid #ffffff !important;
    padding: 8px 6px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.35) !important;
}

.manual-card-green { background-color: #2ecc71 !important; }
.manual-card-blue { background-color: #3498db !important; }
.manual-card-red { background-color: #e74c3c !important; }
.manual-card-yellow { background-color: #f1c40f !important; color: #111115 !important; }
.manual-card-wild { background-color: #111115 !important; }

.manual-card-badge {
    position: absolute !important;
    top: 4px !important;
    left: 4px !important;
    background: rgba(0, 0, 0, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 4px !important;
    color: #ffffff !important;
    font-size: 10px !important;
    font-weight: 900 !important;
    padding: 2px 5px !important;
}

.manual-card-stack {
    text-align: right !important;
    min-height: 10px !important;
    font-size: 7px !important;
    font-weight: 900 !important;
    color: rgba(255, 255, 255, 0.78) !important;
    text-transform: uppercase !important;
}

.manual-card-yellow .manual-card-stack {
    color: rgba(17, 17, 21, 0.7) !important;
}

.manual-card-diamond {
    width: 50px !important;
    height: 50px !important;
    margin: 6px auto 5px auto !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    transform: rotate(45deg) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.2) !important;
}

.manual-card-diamond span {
    transform: rotate(-45deg) !important;
    font-size: 23px !important;
    color: #111115 !important;
}

.manual-card-title {
    min-height: 22px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    font-size: 8px !important;
    font-weight: 900 !important;
    line-height: 1.08 !important;
    text-transform: uppercase !important;
    overflow-wrap: anywhere !important;
}

.manual-text-light {
    color: #ffffff !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.95), 0 0 2px rgba(0, 0, 0, 0.8) !important;
}

.manual-text-dark {
    color: #111115 !important;
    text-shadow: 0 1px 0 rgba(255, 255, 255, 0.35) !important;
}

.manual-card-stats {
    margin-top: auto !important;
    display: flex !important;
    gap: 2px !important;
    border-top: 1.5px dashed rgba(255, 255, 255, 0.45) !important;
    padding-top: 3px !important;
    font-size: 8px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    white-space: nowrap !important;
}

.manual-card-stats span {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    padding: 2px 1px !important;
    border-radius: 3px !important;
    background: rgba(0, 0, 0, 0.62) !important;
    text-align: center !important;
}

.manual-stat-good {
    color: #d9ff5a !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 1) !important;
}

.manual-stat-bad {
    color: #ffc7d1 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 1) !important;
}

.manual-card-note {
    color: #cbd5e0 !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    text-align: center !important;
}

.manual-tip {
    margin-top: 12px !important;
    padding: 12px 14px !important;
    border-left: 3px solid #00f3ff !important;
    border-radius: 6px !important;
    background: rgba(0, 243, 255, 0.08) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* 2. Text inputs styled as engraved CRT computer terminals */
.glass-lobby input[type="text"] {
    background-color: #05060b !important;
    color: #00f3ff !important;
    font-family: "Courier New", Courier, monospace !important;
    font-weight: bold !important;
    border-radius: 6px !important;
    
    border-top: 2.5px solid #070913 !important;
    border-left: 2.5px solid #070913 !important;
    border-bottom: 2.5px solid #3b426f !important;
    border-right: 2.5px solid #3b426f !important;
    
    box-shadow: inset 4px 4px 10px rgba(0, 0, 0, 0.85) !important;
}

.glass-lobby input[type="text"]:focus {
    border-color: #00f3ff !important;
    box-shadow: inset 4px 4px 10px rgba(0, 0, 0, 0.85), 0 0 8px rgba(0, 243, 255, 0.4) !important;
}

/* 3. Top Navigation Tabs styled as hardware console selector buttons */
#main_tabs > .tab-nav {
    background-color: #0b0d19 !important;
    border-bottom: 3px solid #232844 !important;
    padding: 5px 15px 0 15px !important;
    display: flex !important;
    gap: 6px !important;
}

#main_tabs > .tab-nav > button {
    background-color: #111424 !important;
    color: #a0aec0 !important;
    border: 2px solid #232844 !important;
    border-bottom: none !important;
    border-top-left-radius: 8px !important;
    border-top-right-radius: 8px !important;
    padding: 8px 18px !important;
    font-weight: bold !important;
    font-size: 13px !important;
    box-shadow: inset 0 -4px 8px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.2s ease !important;
}

#main_tabs > .tab-nav > button:hover {
    color: #ffffff !important;
    background-color: #161a35 !important;
}

#main_tabs > .tab-nav > button.selected {
    background-color: #1a1e3a !important;
    color: #00f3ff !important;
    border-color: #00f3ff !important;
    text-shadow: 0 0 8px rgba(0, 243, 255, 0.5) !important;
    box-shadow: none !important;
    transform: translateY(-2px) !important;
    position: relative !important;
    z-index: 5 !important;
}

#leave_queue_btn {
    max-width: 180px !important;
    margin: 8px auto 0 auto !important;
    background-color: transparent !important;
    color: #ff6b9a !important;
    border: 1px solid #ff0055 !important;
    border-radius: 6px !important;
    font-weight: bold !important;
}

#leave_queue_btn:hover {
    background-color: #ff0055 !important;
    color: #ffffff !important;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
    
.game-spinner {
    width: 16px;
    height: 16px;
    border: 3px solid rgba(0, 243, 255, 0.15);
    border-top: 3px solid #00f3ff; /* Cyan active indicator */
    border-radius: 50%;
    animation: spin 1s linear infinite !important;
    display: inline-block;
    flex-shrink: 0;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(0, 243, 255, 0.4);
    margin-left: 10px;
    margin-right: 8px;
}
"""

GLOBAL_JS = """
(n, l, h) => {
    window.dodAudioMuted = true;
    localStorage.setItem('dod_audio_muted', 'true');
    window.dodLobbyMusic = window.dodLobbyMusic || {
        audio: null,
        lastState: null,
        shouldPlay: true,
        autoplayWarningShown: false,
        init() {
            if (this.audio) return;
            this.audio = new Audio("/gradio_api/file=assets/main.ogg");
            this.audio.loop = true;
            this.audio.preload = "auto";
            this.audio.volume = 0.28;
        },
        canPlayInState(state) {
            if (!state) return true;
            if (state.restart_countdown) return false;
            const players = Array.isArray(state.players) ? state.players : [];
            const viewerId = state.viewer_id || localStorage.getItem("uno_name") || "";
            const viewerInMatch = viewerId !== "" && players.indexOf(viewerId) !== -1;
            const tabButtons = document.querySelectorAll("#main_tabs > .tab-nav > button");
            const matchTabSelected = !!(tabButtons && tabButtons[1] && tabButtons[1].classList.contains("selected"));
            return !(state.game_started && (viewerInMatch || matchTabSelected));
        },
        sync(state) {
            this.init();
            this.lastState = state || null;
            this.shouldPlay = this.canPlayInState(state);
            if (window.dodAudioMuted || !this.shouldPlay) {
                this.pause();
                return;
            }
            this.play();
        },
        play() {
            this.init();
            if (!this.audio || window.dodAudioMuted || !this.shouldPlay) return;
            const promise = this.audio.play();
            if (promise && typeof promise.catch === "function") {
                promise.catch((error) => {
                    if (!this.autoplayWarningShown) {
                        console.warn("Lobby music autoplay is waiting for a browser interaction.", error);
                        this.autoplayWarningShown = true;
                    }
                });
            }
        },
        pause() {
            if (this.audio && !this.audio.paused) {
                this.audio.pause();
            }
        }
    };
    window.dodLobbyMusic.sync(null);
    window.addEventListener("click", () => {
        if (window.dodLobbyMusic) {
            window.dodLobbyMusic.sync(window.dodLobbyMusic.lastState);
        }
    }, { passive: true });
    window.gameAudio = {
        ctx: null,
        init() {
            if (this.ctx) return;
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (AudioContext) {
                this.ctx = new AudioContext();
            }
        },
        play(type) {
            if (window.dodAudioMuted) return;
            this.init();
            if (!this.ctx) return;
            if (this.ctx.state === 'suspended') {
                this.ctx.resume();
            }
            const now = this.ctx.currentTime;

            try {
                switch(type) {
                    case 'play': {
                        let osc = this.ctx.createOscillator();
                        let gain = this.ctx.createGain();
                        osc.type = 'triangle';
                        osc.frequency.setValueAtTime(500, now);
                        osc.frequency.exponentialRampToValueAtTime(150, now + 0.08);
                        gain.gain.setValueAtTime(0.12, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
                        osc.connect(gain);
                        gain.connect(this.ctx.destination);
                        osc.start(now);
                        osc.stop(now + 0.08);
                        break;
                    }
                    case 'attack': {
                        let osc = this.ctx.createOscillator();
                        let gain = this.ctx.createGain();
                        osc.type = 'sawtooth';
                        osc.frequency.setValueAtTime(880, now);
                        osc.frequency.exponentialRampToValueAtTime(110, now + 0.22);
                        gain.gain.setValueAtTime(0.07, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
                        osc.connect(gain);
                        gain.connect(this.ctx.destination);
                        osc.start(now);
                        osc.stop(now + 0.22);
                        break;
                    }
                    case 'draw': {
                        let osc = this.ctx.createOscillator();
                        let gain = this.ctx.createGain();
                        osc.type = 'sine';
                        osc.frequency.setValueAtTime(120, now);
                        osc.frequency.exponentialRampToValueAtTime(450, now + 0.15);
                        gain.gain.setValueAtTime(0.1, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
                        osc.connect(gain);
                        gain.connect(this.ctx.destination);
                        osc.start(now);
                        osc.stop(now + 0.15);
                        break;
                    }
                    case 'shout': {
                        let osc1 = this.ctx.createOscillator();
                        let osc2 = this.ctx.createOscillator();
                        let gain = this.ctx.createGain();
                        osc1.type = 'sine';
                        osc1.frequency.setValueAtTime(523.25, now);
                        osc2.type = 'sine';
                        osc2.frequency.setValueAtTime(659.25, now + 0.08);
                        gain.gain.setValueAtTime(0.08, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
                        osc1.connect(gain);
                        osc2.connect(gain);
                        gain.connect(this.ctx.destination);
                        osc1.start(now);
                        osc1.stop(now + 0.4);
                        osc2.start(now + 0.08);
                        osc2.stop(now + 0.4);
                        break;
                    }
                    case 'tick': {
                        let osc = this.ctx.createOscillator();
                        let gain = this.ctx.createGain();
                        osc.type = 'sine';
                        osc.frequency.setValueAtTime(900, now);
                        gain.gain.setValueAtTime(0.03, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.02);
                        osc.connect(gain);
                        gain.connect(this.ctx.destination);
                        osc.start(now);
                        osc.stop(now + 0.02);
                        break;
                    }
                    case 'warning': {
                        let osc = this.ctx.createOscillator();
                        let gain = this.ctx.createGain();
                        osc.type = 'sawtooth';
                        osc.frequency.setValueAtTime(170, now);
                        osc.frequency.linearRampToValueAtTime(160, now + 0.22);
                        gain.gain.setValueAtTime(0.05, now);
                        gain.gain.linearRampToValueAtTime(0.001, now + 0.22);
                        osc.connect(gain);
                        gain.connect(this.ctx.destination);
                        osc.start(now);
                        osc.stop(now + 0.22);
                        break;
                    }
                    case 'victory': {
                        const notes = [523.25, 659.25, 783.99, 1046.50];
                        notes.forEach((freq, idx) => {
                            let osc = this.ctx.createOscillator();
                            let gain = this.ctx.createGain();
                            osc.type = 'square';
                            osc.frequency.setValueAtTime(freq, now + idx * 0.09);
                            gain.gain.setValueAtTime(0.05, now + idx * 0.09);
                            gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.09 + 0.22);
                            osc.connect(gain);
                            gain.connect(this.ctx.destination);
                            osc.start(now + idx * 0.09);
                            osc.stop(now + idx * 0.09 + 0.22);
                        });
                        break;
                    }
                    case 'game_over': {
                        let osc = this.ctx.createOscillator();
                        let gain = this.ctx.createGain();
                        osc.type = 'sawtooth';
                        osc.frequency.setValueAtTime(220, now);
                        osc.frequency.exponentialRampToValueAtTime(80, now + 0.9);
                        gain.gain.setValueAtTime(0.07, now);
                        gain.gain.linearRampToValueAtTime(0.001, now + 0.9);
                        osc.connect(gain);
                        gain.connect(this.ctx.destination);
                        osc.start(now);
                        osc.stop(now + 0.9);
                        break;
                    }
                }
            } catch (e) {
                console.warn("Audio play failed:", e);
            }
        }
    };
    
    function applyTransparency() {
        const body = document.body;
        const gradioApp = document.querySelector('gradio-app');
        const container = document.querySelector('.gradio-container');

        if (body) {
            body.style.backgroundColor = 'transparent';
            body.style.margin = '0';
        }
        if (gradioApp) {
            gradioApp.style.background = 'transparent';
        }
        if (container) {
            container.style.background = 'transparent';
            container.style.boxShadow = 'none';
        }
    }

    applyTransparency();

    function initAnimation() {
        const canvas = document.getElementById('bg_canvas');
        if (!canvas) {
            requestAnimationFrame(initAnimation);
            return;
        }

        const ctx = canvas.getContext('2d');
        let w, h;
        let animationId;

        function resize() {
            const rawW = window.innerWidth;
            const rawH = window.innerHeight;

            w = rawW;
            h = Math.min(rawH, 1080);

            const ratio = window.devicePixelRatio || 1;
            canvas.width = Math.round(w * ratio);
            canvas.height = Math.round(h * ratio);

            canvas.style.width = '100vw';
            canvas.style.height = '100vh';

            ctx.scale(ratio, ratio);
        }

        window.addEventListener('resize', resize);
        resize();

        const colors = {
            "green": "rgba(46, 204, 113, ",
            "blue": "rgba(52, 152, 219, ",
            "red": "rgba(231, 76, 60, ",
            "yellow": "rgba(241, 196, 15, "
        };
        const imageFiles = [
            "icon_super.png", "icon_fix.png", "icon_refactor.png", "icon_tech_debt.png",
            "icon_docs.png", "icon_patch.png", "icon_stack_overflow.png", "icon_spaghetti.png",
            "icon_blind_pr.png", "icon_bug.png", "icon_skip.png", "icon_reverse.png",
            "icon_attack.png", "icon_wild.png", "icon_nuke.png", "icon_gradio.png", "icon_nvidia.png",
            "icon_openbmb.png", "icon_openai.png", "icon_modal.png", "icon_huggingface.png"
        ];

        const fallbackSymbols = [
            "🚀", "🩹", "🧹", "💸", "📝", "🔨", "📋", "🍝", "🙈", "🐛", "🛑", "🔄", "➕", "🎨", "☢️",
            "🤗", "💚"
        ];

        const loadedImages = [];
        let imagesReady = false;
        let loadedCount = 0;

        imageFiles.forEach((filename, idx) => {
            const img = new Image();
            img.src = "/gradio_api/file=assets/" + filename;
            img.onload = () => {
                loadedCount++;
                if (loadedCount === imageFiles.length) {
                    imagesReady = true;
                }
            };
            img.onerror = () => {
                loadedCount++;
                if (loadedCount === imageFiles.length) {
                    imagesReady = true;
                }
            };
            loadedImages[idx] = img;
        });

        function drawRoundedRect(ctx, x, y, width, height, radius) {
            ctx.beginPath();
            ctx.moveTo(x + radius, y);
            ctx.lineTo(x + width - radius, y);
            ctx.arcTo(x + width, y, x + width, y + radius, radius);
            ctx.lineTo(x + width, y + height - radius);
            ctx.arcTo(x + width, y + height, x + width - radius, y + height, radius);
            ctx.lineTo(x + radius, y + height);
            ctx.arcTo(x, y + height, x, y + height - radius, radius);
            ctx.lineTo(x, y + radius);
            ctx.arcTo(x, y, x + radius, y, radius);
            ctx.closePath();
        }
         const assetColors = [
            "red",
            "green",
            "blue",
            "yellow",
            "green",
            "red",
            "blue",
            "yellow",
            "red",
            "green",
            "yellow",
            "blue",
            "red",
            "blue",
            "red",
            "yellow",
            "green",
            "blue",
            "black",
            "green",
            "black",
        ];
        class FloatingCard {
            constructor() {
                this.reset(true);
            }
            reset(initial = false) {
                this.w = 60;
                this.h = 84;
                this.x = Math.random() * w;
                this.y = initial ? Math.random() * h : h + 100;
                this.vx = (Math.random() - 0.5) * 0.4;
                this.vy = -(0.3 + Math.random() * 0.6);
                this.angle = Math.random() * Math.PI * 2;
                this.rotSpeed = (Math.random() - 0.5) * 0.005;
                this.assetIndex = Math.floor(Math.random() * imageFiles.length);
                this.symbol = fallbackSymbols[this.assetIndex];
                this.colorKey = assetColors[this.assetIndex];

                this.opacity = 0;
                this.fadeSpeed = 0.005 + Math.random() * 0.005;
                this.isFadingIn = true;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                this.angle += this.rotSpeed;

                if (this.isFadingIn) {
                    this.opacity += this.fadeSpeed;
                    if (this.opacity >= 0.6) {
                        this.opacity = 0.6;
                        this.isFadingIn = false;
                    }
                } else if (this.y < -100 || this.x < -100 || this.x > w + 100) {
                    this.reset();
                } else if (this.y < h * 0.2) {
                    this.opacity -= this.fadeSpeed;
                    if (this.opacity <= 0) this.reset();
                }
            }
            draw(ctx) {
                ctx.save();
                ctx.translate(this.x, this.y);
                ctx.rotate(this.angle);
                if (imagesReady && loadedImages[this.assetIndex]) {
                    ctx.save();
                    ctx.strokeStyle = colors[this.colorKey] + (this.opacity * 0.5) + ")";
                    ctx.lineWidth = 10;
                    drawRoundedRect(ctx, -this.w/2 - 2, -this.h/2 - 2, this.w + 4, this.h + 4, 8);
                    ctx.stroke();
                    ctx.restore();
                    ctx.save();
                    ctx.globalAlpha = this.opacity * 1.5;
                    const imgEl = loadedImages[this.assetIndex];
                    ctx.drawImage(imgEl, -this.w/2, -this.h/2, this.w, this.h);
                    ctx.restore();
                } else {
                    ctx.fillStyle = "rgba(11, 14, 28, 0.75)";
                    drawRoundedRect(ctx, -this.w/2, -this.h/2, this.w, this.h, 8);
                    ctx.fill();

                    ctx.strokeStyle = colors[this.colorKey] + (this.opacity * 0.3) + ")";
                    ctx.lineWidth = 6;
                    ctx.stroke();

                    ctx.strokeStyle = colors[this.colorKey] + (this.opacity * 0.75) + ")";
                    ctx.lineWidth = 3;
                    ctx.stroke();

                    ctx.strokeStyle = colors[this.colorKey] + (this.opacity * 1.0) + ")";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();

                    ctx.fillStyle = "rgba(255, 255, 255, " + (this.opacity * 1.8) + ")";
                    ctx.font = "bold 22px system-ui, -apple-system, sans-serif";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.shadowColor = "rgba(255, 255, 255, 0.5)";
                    ctx.shadowBlur = 4;
                    ctx.fillText(this.symbol, 0, 0);
                }

                ctx.restore();
            }
        }

        let cards = [];
        for (let i = 0; i < 21; i++) cards.push(new FloatingCard());

        function animate() {
            ctx.clearRect(0, 0, w, h);
            cards.forEach(card => {
                card.update();
                card.draw(ctx);
            });
            animationId = requestAnimationFrame(animate);
        }

        animate();
    }
    initAnimation();
    const savedLangRaw = localStorage.getItem('uno_lang') || 'EN-US';
    const savedLang = (savedLangRaw.includes('Portugu') || savedLangRaw === 'PT-BR') ? 'PT-BR' : 'EN-US';
    return [localStorage.getItem('uno_name') || '', savedLang, h];
}
"""


game_theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="yellow",
    neutral_hue="neutral"
).set(

    body_background_fill="*neutral_900",
    body_background_fill_dark="*neutral_900",
    background_fill_secondary="*neutral_900",
    background_fill_secondary_dark="*neutral_900",
    body_text_color="*neutral_200",
    body_text_color_dark="*neutral_200",
    block_background_fill="*neutral_800",
    block_background_fill_dark="*neutral_800",
    checkbox_label_background_fill="neutral_800",
    checkbox_label_background_fill_dark="neutral_800",
    checkbox_label_background_fill_selected="neutral_800",
    checkbox_label_background_fill_selected_dark="*neutral_800",
    body_text_size="1.1em",
    code_background_fill="*neutral_800",
    code_background_fill_dark="*neutral_800",
    shadow_drop="2px 2px 4px rgba(0, 0, 0, 0.5)",
    block_label_background_fill="*neutral_800",
    block_label_background_fill_dark="*neutral_800",
    block_label_text_color="*neutral_200",
    block_label_text_color_dark="*neutral_200",
    block_title_text_color="*primary_300",
    block_title_text_color_dark="*primary_300",
    panel_background_fill="*neutral_950",
    panel_background_fill_dark="*neutral_950",
    panel_border_color="*neutral_800",
    panel_border_color_dark="*neutral_800",
    checkbox_border_color="*neutral_700",
    checkbox_border_color_dark="*neutral_700",
    input_background_fill="*neutral_850",
    input_background_fill_dark="*neutral_850",
    input_border_color="*neutral_700",
    input_border_color_dark="*neutral_700",
    slider_color="*primary_400",
    slider_color_dark="*primary_400",


    button_primary_background_fill="*primary_600",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_background_fill_hover_dark="*primary_700",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",

    button_secondary_background_fill="*secondary_500",
    button_secondary_background_fill_dark="*secondary_500",
    button_secondary_background_fill_hover="*secondary_600",
    button_secondary_background_fill_hover_dark="*secondary_600",
    button_secondary_text_color="*neutral_950",
    button_secondary_text_color_dark="*neutral_950",

    button_cancel_background_fill="*neutral_800",
    button_cancel_background_fill_dark="*neutral_800",
    button_cancel_background_fill_hover="*neutral_700",
    button_cancel_background_fill_hover_dark="*neutral_700",
    button_cancel_text_color="*neutral_200",
    button_cancel_text_color_dark="*neutral_200",
)


def play_card(data: dict[str, Any] | None = None) -> ServerResponse:
    """Server bridge for card-play events emitted by the custom HTML board.

    Args:
        data: Event payload containing player index, card index, and caller id.
    """
    payload = data or {}
    return global_server.play_card(payload["player"], payload["card"], payload["caller"])


def draw_card(data: dict[str, Any] | None = None) -> ServerResponse:
    """Server bridge for draw-pile clicks emitted by the custom HTML board.

    Args:
        data: Event payload containing the caller id.
    """
    payload = data or {}
    return global_server.draw_card(payload["caller"])


def select_wild_color(data: dict[str, Any] | None = None) -> ServerResponse:
    """Server bridge for wild-color selections emitted by the custom HTML board.

    Args:
        data: Event payload containing selected color and caller id.
    """
    payload = data or {}
    return global_server.select_wild_color(payload["color"], payload["caller"])


def accuse_player(data: dict[str, Any] | None = None) -> ServerResponse:
    """Server bridge for accusation clicks emitted by the custom HTML board.

    Args:
        data: Event payload containing target index and caller id.
    """
    payload = data or {}
    return global_server.accuse_player(payload["target"], payload["caller"])


def pass_turn_manual(data: dict[str, Any] | None = None) -> ServerResponse:
    """Server bridge for pass-turn clicks emitted by the custom HTML board.

    Args:
        data: Event payload containing the caller id.
    """
    payload = data or {}
    return global_server.pass_turn_manual(payload["caller"])


def shout_deploy(data: dict[str, Any] | None = None) -> ServerResponse:
    """Server bridge for Deploy shout clicks emitted by the custom HTML board.

    Args:
        data: Event payload containing the caller id.
    """
    payload = data or {}
    return global_server.shout_deploy(payload["caller"])


def leave_game(data: dict[str, Any] | None = None) -> ServerResponse:
    """Server bridge for leave-room clicks emitted by the custom HTML board.

    Args:
        data: Event payload containing the caller id.
    """
    payload = data or {}
    return global_server.leave_game(payload["caller"])


def receive_toast(evt: gr.EventData) -> str:
    """Extract toast text from a custom HTML event.

    Args:
        evt: Gradio event data emitted by `trigger('show_toast', ...)`.
    """
    return evt._data["msg"]


def do_tick() -> None:
    """Advance the backend game clock."""
    global_server.tick_countdown()


def fetch_state_for_player(uid: str) -> tuple[Any, Any]:
    """Fetch a personalized game state payload for a logged-in player.

    Args:
        uid: Player name stored in Gradio state.
    """

    if not uid or uid.strip() == "":
        return gr.update(), gr.update(visible=False)
    state = global_server.get_state(uid)
    state["viewer_id"] = uid
    return state, gr.update(visible=uid in global_server.queue)


def fetch_state_for_spectator() -> GameState:
    """Fetch the public spectator state payload."""
    state = global_server.get_state("")
    state["viewer_id"] = ""
    return state


def get_lang_code(lang_choice: str) -> str:
    """Return the internal language code for a lobby language label.

    Args:
        lang_choice: Language label selected in the lobby.

    Returns:
        `pt` for Portuguese labels, otherwise `en`.
    """
    normalized_choice = (lang_choice or "").upper()
    return "pt" if "PORTUG" in normalized_choice or normalized_choice == "PT-BR" else "en"


def get_hf_userinfo(request: gr.Request | None) -> dict[str, Any]:
    """Read Hugging Face OAuth user info from a Gradio request.

    Args:
        request: Gradio request object injected into event handlers.

    Returns:
        OAuth user info dictionary, or an empty dictionary when unavailable.
    """
    if request is None:
        return {}
    try:
        raw_request = getattr(request, "request", None)
        session = getattr(request, "session", None) or getattr(raw_request, "session", None)
        if not session:
            return {}
        oauth_info = session.get("oauth_info", {})
        userinfo = oauth_info.get("userinfo", {})
        if isinstance(userinfo, Mapping):
            return dict(userinfo)
        if hasattr(userinfo, "model_dump"):
            return dict(userinfo.model_dump())
        if hasattr(userinfo, "dict"):
            return dict(userinfo.dict())
        if hasattr(userinfo, "items"):
            return dict(userinfo.items())
        return {}
    except Exception:
        return {}


def get_zero_gpu_ip_token(request: gr.Request | None) -> str:
    """Read the Hugging Face ZeroGPU IP token from the incoming Gradio request."""
    if request is None:
        return ""
    try:
        headers = getattr(request, "headers", None)
        raw_request = getattr(request, "request", None)
        if headers is None and raw_request is not None:
            headers = getattr(raw_request, "headers", None)
        if not headers:
            return ""
        token = headers.get("x-ip-token") or headers.get("X-IP-Token") or ""
        return str(token).strip()
    except Exception:
        return ""


def get_hf_username(request: gr.Request | None) -> str:
    """Read the Hugging Face OAuth username from a Gradio request.

    Args:
        request: Gradio request object injected into event handlers.

    Returns:
        Authenticated Hugging Face username, or an empty string when unavailable.
    """
    try:
        userinfo = get_hf_userinfo(request)
        username = userinfo.get("preferred_username") or userinfo.get("name") or ""
        return str(username).strip()
    except Exception:
        return ""


def get_hf_picture_url(request: gr.Request | None) -> str:
    """Read the Hugging Face OAuth profile image URL from a Gradio request."""
    try:
        picture_url = get_hf_userinfo(request).get("picture", "")
        return str(picture_url).strip()
    except Exception:
        return ""


def resolve_hf_identity(request: gr.Request | None = None, *known_values: str) -> str:
    """Resolve the Hugging Face username from request, state, or hidden component values.

    Args:
        request: Gradio request object injected into event handlers.
        known_values: Previously stored Hugging Face usernames.

    Returns:
        First non-empty Hugging Face username found.
    """
    request_username = get_hf_username(request)
    if request_username:
        return request_username
    for value in known_values:
        username = (value or "").strip()
        if username:
            return username
    return ""


def get_hf_login_button_update(lang_code: str, hf_username: str = "") -> Any:
    """Return localized labels for the Hugging Face login button.

    Args:
        lang_code: Internal app language code.
        hf_username: Authenticated username, when available.

    Returns:
        Gradio update for the login button labels.
    """
    t = APP_UI[lang_code]
    if hf_username:
        return gr.update(value=t["hf_logout_button"].format(hf_username), logout_value=t["hf_logout_button"])
    return gr.update(value=t["hf_login_button"], logout_value=t["hf_logout_button"])


def format_lobby_player_status(player_name: str, lang_code: str) -> str:
    """Return the lobby status shown for a player waiting in the active room."""
    t = APP_UI.get(lang_code, APP_UI["en"])
    message = t["welcome_play"].replace("{name}", player_name)
    return f'<span style="color: #2ecc71; font-weight: 800;">{message}</span>'


def fetch_leaderboard_for_player(uid: str, lang_choice: str) -> str:
    """Render leaderboard HTML using the player or lobby language preference.

    Args:
        uid: Player name used to resolve language preference.
        lang_choice: Current language radio label used before a player joins.
    """
    fallback_lang = get_lang_code(lang_choice)
    lang = global_server.player_langs.get(uid, fallback_lang)
    return global_server.render_leaderboard_html(lang)


def execute_leave_ui(hf_uid: str = "", uid: str = "", request: gr.Request | None = None) -> tuple[str, str, Any, Any, Any, Any, Any, Any, str]:
    """Reset visible Gradio tabs after the custom board forces a leave action."""
    hf_identity = resolve_hf_identity(request, hf_uid)
    name_update = gr.update(value=hf_identity, visible=False) if hf_identity else gr.update(visible=True)
    return "", "", gr.update(visible=False), gr.update(), gr.update(selected="tab_lobby"), gr.update(visible=False), gr.update(interactive=True), name_update, hf_identity


def leave_queue_from_lobby(uid: str, lang_choice: str) -> tuple[Any, ...]:
    """Remove a queued user from the lobby without entering the player board.

    Args:
        uid: Player name stored in Gradio state.
        lang_choice: Current language radio label.

    Returns:
        Gradio output tuple that resets the lobby to the anonymous state.
    """
    fallback_lang = get_lang_code(lang_choice)
    t = APP_UI[fallback_lang]

    if uid in global_server.players:
        msg = t["welcome_play"].replace("{name}", uid)
        selected_tab = "tab_player" if global_server.game_started else "tab_lobby"
        return uid, msg, gr.update(), gr.update(visible=True), gr.update(selected=selected_tab), gr.update(), gr.update(visible=False), gr.update(interactive=False)

    if not uid or uid not in global_server.queue:
        return "", t["status"], gr.update(), gr.update(visible=False), gr.update(selected="tab_lobby"), gr.update(), gr.update(visible=False), gr.update(interactive=True)

    result = global_server.leave_game(uid)
    state = result.get("state") or global_server.get_state("")
    state["viewer_id"] = ""
    return "", result.get("toast", t["status"]), gr.update(value=state), gr.update(visible=False), gr.update(selected="tab_lobby"), gr.update(), gr.update(visible=False), gr.update(interactive=True)


def change_lang_ui(choice: str, uid: str, hf_uid: str = "", request: gr.Request | None = None) -> tuple[Any, ...]:
    """Update lobby labels and leaderboard HTML after a language change.

    Args:
        choice: Label from the language radio component.
        uid: Player name stored in Gradio state.
        hf_uid: Hugging Face username stored from the login status refresh.
        request: Gradio request used to read an optional Hugging Face OAuth username.
    """
    lang = get_lang_code(choice)
    t = APP_UI[lang]
    is_registered = bool(uid and (uid in global_server.players or uid in global_server.queue))
    is_queued = bool(uid and uid in global_server.queue)
    if uid and is_registered:
        global_server.player_langs[uid] = lang
        global_server.set_player_ip_token(uid, get_zero_gpu_ip_token(request))
        if BOT_NAME in global_server.players and uid in global_server.players:
            global_server.player_langs[BOT_NAME] = lang
    hf_username = resolve_hf_identity(request, hf_uid)
    if hf_username:
        HF_AUTH_PLAYER_NAMES.add(hf_username)
    hf_status = t["hf_login_authenticated"].replace("{name}", hf_username) if hf_username else t["hf_login_guest"]
    name_update = gr.update(label=t["name_label"], value=hf_username, visible=False) if hf_username else gr.update(label=t["name_label"], visible=True)

    styled_title = f'<h1 style="text-align: center !important; color: #ffffff !important; text-shadow: 0 0 10px rgba(0, 243, 255, 0.45); font-size: 26px; font-weight: bold; margin: 0; width: 100%;">{t["title"]}</h1>'


    styled_sub = f'<p style="text-align: center !important; color: #cbd5e0 !important; font-size: 14px; margin: 5px 0 20px 0; width: 100%;">{t["subtitle"]}</p>'

    return (
        styled_title, styled_sub,
        gr.update(label=t["lang_label"]), name_update, gr.update(value=t["btn_join"], interactive=not is_registered), gr.update(value=t["btn_leave_queue"], visible=is_queued),
        t["status"],
        gr.update(label=t["tab_lobby"]), gr.update(label=t["tab_player"]),
        gr.update(label=t["tab_leaderboard"]),
        global_server.render_leaderboard_html(lang),
        gr.update(label=t["tab_manual"]),
        render_how_to_play_html(lang),
        hf_status,
        get_hf_login_button_update(lang, hf_username),
    )


def join_match(player_name: str, lang_choice: str, current_uid: str = "", hf_uid: str = "", request: gr.Request | None = None) -> tuple[Any, ...]:
    """Join a player to the match or queue from the lobby form.

    Args:
        player_name: Name typed by the user.
        lang_choice: UI language radio label.
        current_uid: Existing player name already assigned to this browser tab.
        hf_uid: Hugging Face username stored from the login status refresh.
        request: Gradio request used to read an optional Hugging Face OAuth username.

    Returns:
        Gradio output tuple for user id, status, board state, tab visibility, tab selection, and lobby visibility.
    """
    lang_code = get_lang_code(lang_choice)
    hf_username = resolve_hf_identity(request, hf_uid)
    if hf_username:
        HF_AUTH_PLAYER_NAMES.add(hf_username)
    name = hf_username or (player_name or "").strip()
    known_hf_username = hf_username
    hf_state = known_hf_username or hf_uid
    name_update = gr.update(value=known_hf_username, visible=False) if known_hf_username else gr.update(value=name, visible=True)
    t = APP_UI[lang_code]

    if not name:
        return "", t["invalid_name"], gr.update(), gr.update(), gr.update(), gr.update(), gr.update(visible=False), gr.update(interactive=True), gr.update(visible=True), hf_state

    current_uid = (current_uid or "").strip()
    if current_uid in global_server.players or current_uid in global_server.queue:
        res_name = current_uid
        global_server.player_langs[res_name] = lang_code
    else:
        res_name = global_server.join_lobby(name, lang_code)

    if res_name == "DUPLICATE_REJECT":
        return "", t["duplicate_name"], gr.update(), gr.update(), gr.update(), gr.update(), gr.update(visible=False), gr.update(interactive=True), name_update, hf_state

    global_server.set_player_ip_token(res_name, get_zero_gpu_ip_token(request))
    is_hf_identity = bool(known_hf_username and res_name == known_hf_username)
    global_server.set_player_authenticated(res_name, is_hf_identity)
    if hf_username and res_name == hf_username:
        global_server.set_player_picture(res_name, get_hf_picture_url(request))

    new_state = global_server.get_state(res_name)

    new_state["viewer_id"] = res_name

    is_active_room_player = res_name in global_server.players

    if is_active_room_player and global_server.can_start_lobby_match():
        if not global_server.modal_is_warm and not global_server.modal_is_warming_up:
            # Thread-Lock: Set warming up immediately on the main thread
            global_server.modal_is_warming_up = True
            threading.Thread(target=async_modal_warmup, daemon=True).start()
        
        # Keep players in lobby with a beautiful progress warning instead of redirecting them immediately
        msg = f'<div style="display: inline-flex; align-items: center; justify-content: center; width: 100%; color: #00f3ff; font-weight: bold;"><div class="game-spinner"></div> {t["warmup_status"]}</div>'
            
        return res_name, msg, gr.update(value=new_state), gr.update(visible=True), gr.update(selected="tab_lobby"), gr.update(), gr.update(visible=False), gr.update(interactive=False), name_update, hf_state
    
    if is_active_room_player:
        msg = format_lobby_player_status(res_name, lang_code)
    else:
        pos = global_server.queue.index(res_name) + 1
        msg = f"⏳ {t['welcome_queue'].replace('{pos}', str(pos))}"
        return res_name, msg, gr.update(value=new_state), gr.update(visible=True), gr.update(selected="tab_lobby"), gr.update(), gr.update(visible=True), gr.update(interactive=False), name_update, hf_state
        
    return res_name, msg, gr.update(value=new_state), gr.update(visible=True), gr.update(selected="tab_lobby"), gr.update(), gr.update(visible=False), gr.update(interactive=False), name_update, hf_state



def check_auto_login(saved_name: str, saved_lang: str, hf_uid: str = "", request: gr.Request | None = None) -> tuple[Any, ...]:
    """Restore a saved localStorage session and resolve Hugging Face identity once."""
    lang_code = get_lang_code(saved_lang)
    lang_value = "PT-BR" if lang_code == "pt" else "EN-US"
    t = APP_UI[lang_code]
    hf_username = resolve_hf_identity(request, hf_uid)
    if hf_username:
        HF_AUTH_PLAYER_NAMES.add(hf_username)
    hf_status = t["hf_login_authenticated"].replace("{name}", hf_username) if hf_username else t["hf_login_guest"]
    hf_button_update = get_hf_login_button_update(lang_code, hf_username)

    if saved_name and saved_name.strip() and (saved_name in global_server.players or saved_name in global_server.queue):
        join_outputs = join_match(saved_name, saved_lang, "", hf_username, request)
        return tuple(list(join_outputs) + [hf_status, hf_button_update, gr.update(value=lang_value)])

    name_update = gr.update(value=hf_username, visible=False) if hf_username else gr.update(visible=True)
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(visible=False),
        gr.update(interactive=True),
        name_update,
        hf_username,
        hf_status,
        hf_button_update,
        gr.update(value=lang_value),
    )

def get_name_input_visibility_update(
    hf_uid: str = "",
    request: gr.Request | None = None,
    uid: str = "",
) -> Any:
    """Keep the manual name field hidden while Hugging Face identity is active.

    Args:
        hf_uid: Hugging Face username stored from the login status refresh.
        request: Gradio request used to read an optional Hugging Face OAuth username.
        uid: Current browser-tab player identity. This is not used to infer login state.

    Returns:
        Gradio update for the lobby name textbox visibility.
    """
    hf_username = resolve_hf_identity(request, hf_uid)
    if hf_username:
        HF_AUTH_PLAYER_NAMES.add(hf_username)
        return gr.update(value=hf_username, visible=False)
    return gr.update(visible=True)


def lobby_sync_check(uid: str, lang_choice: str, request: gr.Request | None = None) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Move a logged-in user from the lobby to the player tab once the match starts.

    Args:
        uid: Player name stored in Gradio state.
        request: Gradio request used to keep Hugging Face identity controls in sync.

    Returns:
        Gradio updates for player tab visibility, selected tab, lobby visibility, queue-leave button visibility, join button state, player board state, and lobby status text.
    """
    fallback_lang = get_lang_code(lang_choice)
    lang = global_server.player_langs.get(uid, fallback_lang) if uid else fallback_lang
    t = APP_UI.get(lang, APP_UI["en"])

    if global_server.can_start_lobby_match():
        if global_server.modal_is_warm:
            global_server.init_game()
        elif not global_server.modal_is_warming_up:
            global_server.modal_is_warming_up = True
            threading.Thread(target=async_modal_warmup, daemon=True).start()

    if uid and uid.strip() != "":
        global_server.set_player_ip_token(uid, get_zero_gpu_ip_token(request))
        global_server.touch_presence(uid)
        if uid in global_server.players and global_server.game_started:
            state = global_server.get_state(uid)
            state["viewer_id"] = uid
            return gr.update(visible=True), gr.update(selected="tab_player"), gr.update(), gr.update(visible=False), gr.update(interactive=False), gr.update(value=state), ""
        if uid in global_server.players:
            if global_server.restart_countdown > 0 or global_server.game_end_reason:
                return gr.update(), gr.update(), gr.update(), gr.update(visible=False), gr.update(interactive=False), gr.update(), t["status"]
            if global_server.modal_is_warming_up:
                msg = f'<div style="display: inline-flex; align-items: center; justify-content: center; width: 100%; color: #00f3ff; font-weight: bold;"><div class="game-spinner"></div> {t["warmup_status"]}</div>'
                return gr.update(visible=True), gr.update(selected="tab_lobby"), gr.update(), gr.update(visible=False), gr.update(interactive=False), gr.update(), msg
            return gr.update(visible=True), gr.update(selected="tab_lobby"), gr.update(), gr.update(visible=False), gr.update(interactive=False), gr.update(), format_lobby_player_status(uid, lang)
        if uid in global_server.queue:
            state = global_server.get_state(uid)
            state["viewer_id"] = uid
            pos = global_server.queue.index(uid) + 1
            return gr.update(visible=True), gr.update(selected="tab_lobby"), gr.update(), gr.update(visible=True), gr.update(interactive=False), gr.update(value=state), t["welcome_queue"].replace("{pos}", str(pos))
        return gr.update(visible=False), gr.update(), gr.update(), gr.update(visible=False), gr.update(interactive=True), gr.update(), t["status"]
    return gr.update(visible=False), gr.update(), gr.update(), gr.update(visible=False), gr.update(interactive=True), gr.update(), t["status"]

STANDARD_STACKS = ["green", "blue", "red", "yellow"]


def choose_dominant_stack(cards: list[dict[str, Any]]) -> str:
    """Choose the most common playable stack color from a hand.

    Args:
        cards: Cards to inspect.

    Returns:
        Dominant stack color, or a random standard stack when no colored cards exist.
    """
    stack_colors = [card["stack"] for card in cards if card.get("stack") in STANDARD_STACKS]
    if stack_colors:
        return max(set(stack_colors), key=stack_colors.count)
    return random.choice(STANDARD_STACKS)


def apply_bot_card_play(bot_name: str, player_index: int, card_index: int | None, chosen_color: str | None = None) -> bool:
    """Play a bot card when the selected index is still valid.

    Args:
        bot_name: Bot player name.
        player_index: Current bot index in the players list.
        card_index: Index selected by the LLM or fallback logic.
        chosen_color: Optional wild-card color selected by the LLM.

    Returns:
        True when a card was played, otherwise False.
    """
    hand = global_server.hands.get(bot_name, [])
    if card_index is None or card_index < 0 or card_index >= len(hand):
        return False

    card = hand[card_index]
    if not global_server.is_valid_play(card):
        return False

    print(f"[Bot Decision] Playing card: '{card['name'].get('en', '')}' at index {card_index}", flush=True)
    global_server.play_card(player_index, card_index, bot_name)

    if card["stack"] == "wild":
        remaining_hand = global_server.hands.get(bot_name, [])
        color_to_apply = chosen_color if chosen_color in STANDARD_STACKS else choose_dominant_stack(remaining_hand)
        print(f"[Bot Decision] Applying wild color: {color_to_apply}", flush=True)
        global_server.select_wild_color(color_to_apply, bot_name)

    return True


def apply_bot_draw_flow(bot_name: str, player_index: int) -> None:
    """Draw for the bot and immediately play the drawn card when legal.

    Args:
        bot_name: Bot player name.
        player_index: Current bot index in the players list.
    """
    print("[Bot Decision] Drawing card from deck...", flush=True)
    previous_card_ids = {id(card) for card in global_server.hands.get(bot_name, [])}
    global_server.draw_card(bot_name)

    updated_hand = global_server.hands.get(bot_name, [])
    if not updated_hand:
        global_server.pass_turn_manual(bot_name)
        return

    drawn_card_index = next((idx for idx, card in enumerate(updated_hand) if id(card) not in previous_card_ids), None)
    if drawn_card_index is None:
        global_server.pass_turn_manual(bot_name)
        return

    drawn_card = updated_hand[drawn_card_index]
    if global_server.is_valid_play(drawn_card):
        print(f"[Bot Decision] Playing newly drawn card: '{drawn_card['name'].get('en', '')}'", flush=True)
        global_server.play_card(player_index, drawn_card_index, bot_name)

        if drawn_card["stack"] == "wild":
            remaining_hand = global_server.hands.get(bot_name, [])
            color_to_apply = choose_dominant_stack(remaining_hand)
            print(f"[Bot Decision] Applying drawn wild color: {color_to_apply}", flush=True)
            global_server.select_wild_color(color_to_apply, bot_name)
        return

    global_server.pass_turn_manual(bot_name)


def apply_local_bot_fallback(bot_name: str, player_index: int) -> None:
    """Use deterministic local rules when remote bot inference is unavailable.

    Args:
        bot_name: Bot player name.
        player_index: Current bot index in the players list.
    """
    playable_indices = [
        idx
        for idx, card in enumerate(global_server.hands.get(bot_name, []))
        if global_server.is_valid_play(card)
    ]

    if playable_indices:
        apply_bot_card_play(bot_name, player_index, playable_indices[0])
        return

    apply_bot_draw_flow(bot_name, player_index)


def extract_json_payload(raw_text: str) -> str:
    """Extract a JSON object from common model wrappers.

    Args:
        raw_text: Raw model response.

    Returns:
        Candidate JSON string after removing markdown or thinking wrappers.
    """
    result = raw_text.strip()
    if "```" in result:
        parts = result.split("```")
        for part in parts:
            part_clean = part.strip()
            if part_clean.startswith("json"):
                part_clean = part_clean[4:].strip()
            if part_clean.startswith("{") and part_clean.endswith("}"):
                return part_clean
    if "</think>" in result:
        return result.split("</think>")[-1].strip()
    return result


def process_queued_bot_turn(bot_name: str, ip_token: str = "") -> None:
    """Execute a queued bot decision through remote inference with local fallback.

    Args:
        bot_name: Name of the bot player whose turn should be processed.
        ip_token: Hugging Face ZeroGPU IP token to forward to the LLM Space.
    """
    print(f"[Bot Decision] Initiating turn evaluation for: {bot_name}", flush=True)
    
    if not global_server.game_started or bot_name not in global_server.players:
        print(f"[Bot Decision] Aborted: game_started={global_server.game_started}", flush=True)
        return
        
    p_idx = global_server.players.index(bot_name)
    if p_idx != global_server.active_player:
        print(f"[Bot Decision] Aborted: p_idx={p_idx} is not active_player={global_server.active_player}", flush=True)
        return

    hand = global_server.hands.get(bot_name, [])
    formatted_hand = []
    for idx, c in enumerate(hand):
        playable = global_server.is_valid_play(c)
        formatted_hand.append({
            "index": idx, 
            "stack": c["stack"], 
            "category": c["category"], 
            "playable": playable, 
            "res": c["res"], 
            "panic": c["panic"]
        })
    has_playable_card = any(card["playable"] for card in formatted_hand)
    
    active = global_server.active_card or {"stack": "wild", "category": "WILD"}
    
    state_payload = {
        "active_card": {"stack": active["stack"], "category": active["category"]},
        "metrics": {"resolution": global_server.resolution, "panic": global_server.panic},
        "hand": formatted_hand
    }

    try:
        print("[Bot Decision] Dispatching external API call to mapped LLM endpoint...", flush=True)

        bot_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["PLAY", "DRAW"]},
                "card_index": {"type": ["integer", "null"]},
                "chosen_color": {"type": "string", "enum": ["green", "blue", "red", "yellow", "none"]}
            },
            "required": ["action", "card_index", "chosen_color"]
        }

        result_str = predict_llm(
            BOT_SYSTEM_PROMPT,
            json.dumps(state_payload),
            0.1,
            json.dumps(bot_schema),
            ip_token=ip_token,
        )

        print(f"[Bot Decision] Raw Response from LLM: '{result_str}'", flush=True)

        result_str = extract_json_payload(result_str)
        if not (result_str.startswith("{") and result_str.endswith("}")):
            raise RuntimeError(f"LLM returned a non-JSON error response: {result_str}")

        decision = json.loads(result_str)
        action = decision.get("action")
        card_idx = decision.get("card_index")
        chosen_color = decision.get("chosen_color")
        print(f"[Bot Decision] Parsed Strategic LLM Decision: {decision}", flush=True)

        if action == "PLAY":
            if apply_bot_card_play(bot_name, p_idx, card_idx, chosen_color):
                return
            print("[Bot Decision] LLM selected an invalid card. Using local fallback rules.", flush=True)
            apply_local_bot_fallback(bot_name, p_idx)
            return

        if action == "DRAW":
            if has_playable_card:
                print("[Bot Decision] LLM chose DRAW despite playable cards. Using local fallback rules.", flush=True)
                apply_local_bot_fallback(bot_name, p_idx)
                return
            print("[Bot Decision] LLM chose DRAW. Using safe draw flow.", flush=True)
            apply_bot_draw_flow(bot_name, p_idx)
            return

        raise RuntimeError(f"Unsupported bot action: {action}")

    except Exception as e:
        print(f"[Bot Decision] Remote inference failed ({e}). Activating local fallback.", flush=True)
        try:
            apply_local_bot_fallback(bot_name, p_idx)
        except Exception as fe:
            print(f"[Bot Fallback] Critical failure in fallback runner: {fe}. Forcing pass.", flush=True)
            global_server.draw_card(bot_name)
            global_server.pass_turn_manual(bot_name)


def get_current_crisis_data() -> dict[str, Any]:
    """Return the active crisis configuration used by Director quote generation."""
    if not CRISES_DATABASE:
        return {}
    return CRISES_DATABASE[global_server.current_crisis_idx % len(CRISES_DATABASE)]


def build_director_crisis_context() -> dict[str, Any]:
    """Build the crisis payload sent to the Director LLM."""
    crisis_data = get_current_crisis_data()
    return {
        "title_en": crisis_data.get("title", {}).get("en", ""),
        "title_pt": crisis_data.get("title", {}).get("pt", ""),
        "description_en": crisis_data.get("desc", {}).get("en", ""),
        "description_pt": crisis_data.get("desc", {}).get("pt", ""),
    }


def build_recent_director_context(event_id: float, limit: int = 2) -> list[dict[str, str]]:
    """Return a compact history of recent Director reactions before the current event."""
    recent_quotes = []
    for evt in reversed(global_server.events):
        kwargs = evt.get("kwargs", {})
        if evt.get("key") != "play" or kwargs.get("quote_id") == event_id:
            continue

        quote = kwargs.get("quote")
        if not quote:
            continue

        if isinstance(quote, dict):
            quote_text = quote.get("en") or quote.get("pt") or ""
        else:
            quote_text = str(quote)

        card_name = kwargs.get("card", {})
        if isinstance(card_name, dict):
            card_text = card_name.get("en") or card_name.get("pt") or ""
        else:
            card_text = str(card_name)

        recent_quotes.append({
            "card_played": card_text,
            "director_quote": quote_text,
        })
        if len(recent_quotes) >= limit:
            break
    return list(reversed(recent_quotes))


FORBIDDEN_DIRECTOR_QUOTE_PHRASES = (
    "deploy da frente",
    "codigo de pao",
    "código de pão",
    "ddos",
    "front do ddos",
    "front do ddo",
    "a revertão",
    "a revertao",
    "base de dados",
    "banco desaparecido",
    "teclas aws",
    "fusão de código",
    "fusao de codigo",
    "produção frontal",
    "producao frontal",
    "a fixa",
    "o fixa",
)


def validate_director_quote(quote: dict[str, str], card_played: str) -> None:
    """Reject Director quotes that contain known bad translations or card drift."""
    combined_text = f"{quote.get('en', '')} {quote.get('pt', '')}".lower()
    for phrase in FORBIDDEN_DIRECTOR_QUOTE_PHRASES:
        if phrase in combined_text:
            raise RuntimeError(f"Director quote failed lexical guard: {phrase}")

    if "Deploy Friday 6PM (Backend)" in card_played and "front-end" in combined_text:
        raise RuntimeError("Director quote mentioned front-end for a Backend deploy.")
    if "Deploy Friday 6PM (Frontend)" in card_played and "back-end" in combined_text:
        raise RuntimeError("Director quote mentioned back-end for a Frontend deploy.")


def choose_director_fallback_quote(card_type: str) -> dict[str, str]:
    """Choose a localized Director quote from the active crisis fallback pool."""
    crisis_data = get_current_crisis_data()
    quotes = crisis_data.get("quotes", {})
    quote_pool = quotes.get(card_type) or quotes.get("bad") or []
    if quote_pool:
        return random.choice(quote_pool)
    return {
        "en": "We need this fixed now!",
        "pt": "Precisamos corrigir isso agora!",
    }


def apply_director_quote_to_event(event_id: float, quote: dict[str, str]) -> bool:
    """Attach a Director quote to the exact play log event that requested it."""
    for evt in reversed(global_server.events):
        if evt["key"] == "play" and evt["kwargs"].get("quote_id") == event_id:
            evt["kwargs"]["quote"] = quote
            return True
    return False


def queue_director_audio_task(
    quote: dict[str, str],
    event_id: float,
    cache_key: str,
    active_langs: set[str],
    card_type: str,
    audio_generation_id: int,
    is_fallback: bool = False,
) -> None:
    """Queue Director speech synthesis for active player languages."""
    if TTS_DOWNLOAD_DISABLED:
        print(f"[TTS Queue] Audio download disabled; keeping director text only for event {event_id}", flush=True)
        return

    tts_audio_queue.put({
        "quote_en": quote.get("en", ""),
        "quote_pt": quote.get("pt", ""),
        "event_id": event_id,
        "cache_key": cache_key,
        "active_langs": active_langs,
        "players": list(global_server.players),
        "card_type": card_type,
        "audio_generation_id": audio_generation_id,
        "is_fallback": is_fallback,
    })


def is_current_audio_generation(audio_generation_id: int | None) -> bool:
    """Return whether a queued audio task still belongs to the active match."""
    return (
        audio_generation_id is not None
        and global_server.game_started
        and audio_generation_id == global_server.audio_generation_id
    )


def process_queued_director_quote(card_played, card_type, event_id, card_context=None, audio_generation_id=None, ip_token: str = ""):
    """Generates the IT Director quote, overwrites the exact log event, and queues player audios."""
    if not is_current_audio_generation(audio_generation_id):
        print(f"[Director Quote] Skipping stale director task for event {event_id}", flush=True)
        return

    print(f"[Director Quote] Initiating generation for card: {card_played} ({card_type})", flush=True)
    
    active_langs = {global_server.player_langs.get(p, "en") for p in global_server.players}
    cache_key = str(event_id)

    if cache_key not in global_server.audio_cache:
        global_server.audio_cache[cache_key] = {"en": "", "pt": ""}

    state_payload = {
        "card_played": card_played,
        "type": card_type,
        "card_effect": card_context or {},
        "recent_director_quotes": build_recent_director_context(event_id),
        "crisis": build_director_crisis_context(),
    }

    try:
        print("[Director Quote] Calling mapped LLM endpoint...", flush=True)

        director_schema = {
            "type": "object",
            "properties": {
                "quote_en": {"type": "string", "minLength": 10, "maxLength": 100},
                "quote_pt": {"type": "string", "minLength": 10, "maxLength": 100}
            },
            "required": ["quote_en", "quote_pt"]
        }

        result_str = predict_llm(
            DIRECTOR_SYSTEM_PROMPT,
            json.dumps(state_payload),
            0.75,
            json.dumps(director_schema),
            ip_token=ip_token,
        )

        print(f"[Director Quote] Raw Response from LLM: '{result_str}'", flush=True)

        result_str = extract_json_payload(result_str)
        if not (result_str.startswith("{") and result_str.endswith("}")):
            raise RuntimeError(f"LLM returned a non-JSON error response: {result_str}")

        result = json.loads(result_str)

        quote_en = result.get("quote_en", "")
        quote_pt = result.get("quote_pt", "")
        print(f"[Director Quote] Text generated: EN='{quote_en}' | PT='{quote_pt}'", flush=True)

        quote = {"en": quote_en, "pt": quote_pt}
        validate_director_quote(quote, card_played)
        apply_director_quote_to_event(event_id, quote)

        if not is_current_audio_generation(audio_generation_id):
            print(f"[Director Quote] Dropping stale audio task after text generation for event {event_id}", flush=True)
            return

        queue_director_audio_task(quote, event_id, cache_key, active_langs, card_type, audio_generation_id)
        print(f"[TTS Queue] Queued director audio for event {event_id}", flush=True)

    except Exception as e:
        print(f"[Director Quote] Text generation failed: {e}", flush=True)
        try:
            if not is_current_audio_generation(audio_generation_id):
                print(f"[Director Quote] Dropping stale fallback quote for event {event_id}", flush=True)
                return
            fallback_quote = choose_director_fallback_quote(card_type)
            apply_director_quote_to_event(event_id, fallback_quote)
            queue_director_audio_task(
                fallback_quote,
                event_id,
                cache_key,
                active_langs,
                card_type,
                audio_generation_id,
                is_fallback=True,
            )
        except Exception as fe:
            print(f"[Director Quote] Critical failure applying fallback: {fe}", flush=True)

def async_modal_warmup():
    """Triggers a background non-blocking wakeup call to both TTS and LLM services.
       to handle GPU cold starts in parallel while players wait in the lobby."""
    print("[Warmup] Initiating background wakeup handshake to cloud GPU services...", flush=True)
    global_server.modal_is_warming_up = True

    warmup_results = {"modal_ready": False, "llm_ready": False}

    def warm_tts_endpoint() -> None:
        if TTS_DOWNLOAD_DISABLED:
            print("[Warmup] TTS warmup skipped because DOD_DISABLE_TTS=True.", flush=True)
            warmup_results["modal_ready"] = True
            return
        print("[Warmup] Sending wakeup ping to Modal (Audio Server)...", flush=True)
        warmup_results["modal_ready"] = global_server.download_tts_language(
            "0",
            "Starting",
            "en",
            False,
            use_warmup_timeout=True,
        )

    def warm_llm_endpoint() -> None:
        print("[Warmup] Sending wakeup ping to LLM Server...", flush=True)
        try:
            result_str = predict_llm(
                "Warmup ping",
                '{"ping": true}',
                0.1,
                "",
                use_warmup_timeout=True,
                ip_token=global_server.get_player_ip_token(),
            )
            if result_str and not result_str.startswith("❌"):
                warmup_results["llm_ready"] = True
                print("[Warmup] LLM inference server successfully warmed up!", flush=True)
        except Exception as e:
            print(f"[Warmup] LLM wakeup failed: {e}", flush=True)

    tts_thread = threading.Thread(target=warm_tts_endpoint, daemon=True)
    llm_thread = threading.Thread(target=warm_llm_endpoint, daemon=True)
    tts_thread.start()
    llm_thread.start()
    tts_thread.join()
    llm_thread.join()

    modal_ready = warmup_results["modal_ready"]
    llm_ready = warmup_results["llm_ready"]

    if modal_ready and llm_ready:
        global_server.modal_is_warm = True
        global_server.modal_is_warming_up = False
        print("[Warmup] ALL cloud GPU services are fully active! Launching match...", flush=True)

        if global_server.can_start_lobby_match():
            global_server.init_game()
    else:
        print(f"[Warmup] Warning: Warmup incomplete. TTS={modal_ready}, LLM={llm_ready}. Retrying on next join.", flush=True)
        global_server.modal_is_warm = False
        global_server.modal_is_warming_up = False
        return
        
BOARD_SERVER_FUNCTIONS = [
    play_card,
    draw_card,
    select_wild_color,
    accuse_player,
    pass_turn_manual,
    shout_deploy,
    leave_game,
]

def tts_audio_queue_worker() -> None:
    """Download director audio sequentially so the local TTS API is not overloaded."""
    while True:
        task = tts_audio_queue.get()
        try:
            event_id = task["event_id"]
            cache_key = task["cache_key"]
            active_langs = set(task["active_langs"])
            audio_generation_id = task.get("audio_generation_id")

            if not is_current_audio_generation(audio_generation_id):
                print(f"[TTS Queue] Skipping stale audio task before synthesis for event {event_id}", flush=True)
                continue

            print(f"[TTS Queue] Starting audio synthesis for event {event_id}", flush=True)

            def queue_audio_for_language(lang: str, audio_cache_key: str) -> None:
                cached_audio = global_server.audio_cache.get(audio_cache_key, {})
                if not cached_audio.get(lang):
                    return
                for player_name in task["players"]:
                    if player_name not in global_server.players:
                        continue
                    if global_server.player_langs.get(player_name, "en") != lang:
                        continue
                    if player_name not in global_server.pending_audios:
                        global_server.pending_audios[player_name] = []
                    global_server.pending_audios[player_name].append({"id": event_id, "cache_key": audio_cache_key})

            def synthesize_and_queue(lang: str, text: str, audio_cache_key: str) -> bool:
                global_server.audio_cache.setdefault(audio_cache_key, {"en": "", "pt": ""})
                if not text:
                    return False
                if not is_current_audio_generation(audio_generation_id):
                    print(f"[TTS Queue] Skipping stale {lang} synthesis before request for event {event_id}", flush=True)
                    return False
                if not global_server.download_tts_language(audio_cache_key, text, lang):
                    return False
                if not is_current_audio_generation(audio_generation_id):
                    print(f"[TTS Queue] Discarding late {lang} audio response for event {event_id}", flush=True)
                    return False
                queue_audio_for_language(lang, audio_cache_key)
                return True

            audio_ready = False

            if "en" in active_langs:
                audio_ready |= synthesize_and_queue("en", task["quote_en"], cache_key)
            if "pt" in active_langs:
                audio_ready |= synthesize_and_queue("pt", task["quote_pt"], cache_key)

            if not audio_ready and not task.get("is_fallback", False):
                if not is_current_audio_generation(audio_generation_id):
                    print(f"[TTS Queue] Skipping stale fallback synthesis for event {event_id}", flush=True)
                    continue
                fallback_quote = choose_director_fallback_quote(task.get("card_type", "bad"))
                fallback_cache_key = f"{cache_key}:fallback"
                apply_director_quote_to_event(event_id, fallback_quote)
                print(f"[TTS Queue] Generated audio failed; trying crisis fallback for event {event_id}", flush=True)
                if "en" in active_langs:
                    audio_ready |= synthesize_and_queue("en", fallback_quote["en"], fallback_cache_key)
                if "pt" in active_langs:
                    audio_ready |= synthesize_and_queue("pt", fallback_quote["pt"], fallback_cache_key)

            if not audio_ready:
                print(f"[TTS Queue] No playable audio was produced for event {event_id}", flush=True)

            print(f"[TTS Queue] Completed audio synthesis for event {event_id}", flush=True)
        except Exception as e:
            print(f"[TTS Queue] Error processing director audio: {e}", flush=True)
        finally:
            tts_audio_queue.task_done()

def llm_queue_worker() -> None:
    """Processes Bot decisions and IT Director Quotes sequentially to prevent CPU bottlenecks."""
    while True:
        task = llm_queue.get()
        try:
            if task["type"] == "bot_decision":
                process_queued_bot_turn(task["bot_name"], task.get("ip_token", ""))
            elif task["type"] == "director_quote":
                process_queued_director_quote(
                    task["card_played"],
                    task["card_type"],
                    task["event_id"],
                    task.get("card_context"),
                    task.get("audio_generation_id"),
                    task.get("ip_token", ""),
                )
        except Exception as e:
            print(f"Error executing queued LLM task: {e}")
        finally:
            llm_queue.task_done()
            
with gr.Blocks() as demo:
    gr.HTML('<canvas id="bg_canvas"></canvas>')
    user_id = gr.State("")
    hf_user_id = gr.State("")
    toast_ui = NeonToast()
    initial_ui = APP_UI["en"]

    with gr.Row(elem_id="global_lang_bar"):
        lang_input = gr.Radio(
            choices=["EN-US", "PT-BR"],
            value="EN-US",
            label=initial_ui["lang_label"],
            show_label=False,
            elem_classes=["language-switch"],
        )

    with gr.Tabs(elem_id="main_tabs") as main_tabs:
        with gr.Tab(initial_ui["tab_lobby"], id="tab_lobby") as lobby_tab:
            with gr.Column(elem_classes="glass-lobby", visible=True) as login_box:
                gr.HTML('<img src="/gradio_api/file=assets/logo.jpeg" class="lobby-logo" style="border-radius: 12px; max-width: 180px; display: block; margin: 0 auto 20px auto;">')
                title_html = gr.HTML(f'<h1 style="text-align: center !important; color: #ffffff !important; text-shadow: 0 0 10px rgba(0, 243, 255, 0.45); font-size: 26px; font-weight: bold; margin: 0; width: 100%;">{initial_ui["title"]}</h1>')
                sub_html = gr.HTML(f'<p style="text-align: center !important; color: #cbd5e0 !important; font-size: 14px; margin: 5px 0 20px 0; width: 100%;">{initial_ui["subtitle"]}</p>')
                hf_login_btn = gr.LoginButton(value=initial_ui["hf_login_button"], logout_value=initial_ui["hf_logout_button"])
                auth_status = gr.Markdown(initial_ui["hf_login_guest"])
                name_input = gr.Textbox(label=initial_ui["name_label"], elem_id="manual_name_input")
                join_btn = gr.Button(initial_ui["btn_join"], variant="primary")
                status_msg = gr.Markdown(initial_ui["status"])
                leave_queue_btn = gr.Button(initial_ui["btn_leave_queue"], visible=False, elem_id="leave_queue_btn")

            init_state = global_server.get_state("")
            init_state["viewer_id"] = ""
            spectator_board = Board(value=init_state, server_functions=BOARD_SERVER_FUNCTIONS)

        with gr.Tab(initial_ui["tab_player"], id="tab_player", visible=False) as player_tab:
            player_board = Board(value=init_state, server_functions=BOARD_SERVER_FUNCTIONS)

        with gr.Tab(initial_ui["tab_leaderboard"], id="tab_leaderboard") as leaderboard_tab:
            leaderboard_board = gr.HTML(value=global_server.render_leaderboard_html("en"))

        with gr.Tab(initial_ui["tab_manual"], id="tab_manual") as manual_tab:
            manual_board = gr.HTML(value=render_how_to_play_html("en"))

    lang_input.change(
        fn=change_lang_ui,
        inputs=[lang_input, user_id, hf_user_id],

        outputs=[title_html, sub_html, lang_input, name_input, join_btn, leave_queue_btn, status_msg, lobby_tab, player_tab, leaderboard_tab, leaderboard_board, manual_tab, manual_board, auth_status, hf_login_btn],
        show_progress=False,
    )

    join_btn.click(
        fn=join_match,
        inputs=[name_input, lang_input, user_id, hf_user_id],
        outputs=[user_id, status_msg, player_board, player_tab, main_tabs, login_box, leave_queue_btn, join_btn, name_input, hf_user_id],
        js="(n, l, u, h) => { localStorage.setItem('uno_name', u || h || n); localStorage.setItem('uno_lang', l); return [n, l, u, h]; }",
        show_progress=False,
    )

    leave_queue_btn.click(
        fn=leave_queue_from_lobby,
        inputs=[user_id, lang_input],
        outputs=[user_id, status_msg, player_board, player_tab, main_tabs, login_box, leave_queue_btn, join_btn],
        js="(u, l) => { localStorage.removeItem('uno_name'); localStorage.removeItem('uno_lang'); return [u, l]; }",
        show_progress=False,
    )

    demo.load(
        fn=check_auto_login,
        inputs=[name_input, lang_input, hf_user_id],
        outputs=[user_id, status_msg, player_board, player_tab, main_tabs, login_box, leave_queue_btn, join_btn, name_input, hf_user_id, auth_status, hf_login_btn, lang_input],
        js=GLOBAL_JS,
        show_progress=False,
    )

    player_board.show_toast(fn=receive_toast, inputs=None, outputs=toast_ui, show_progress=False)
    spectator_board.show_toast(fn=receive_toast, inputs=None, outputs=toast_ui, show_progress=False)

    player_board.force_leave_ui(fn=execute_leave_ui, inputs=[hf_user_id, user_id], outputs=[user_id, status_msg, player_tab, login_box, main_tabs, leave_queue_btn, join_btn, name_input, hf_user_id], show_progress=False)


    tick_timer = gr.Timer(TICK_RATE_SERVER_SECONDS)
    tick_timer.tick(fn=do_tick, inputs=[], outputs=[], show_progress=False)


    player_sync_timer = gr.Timer(SYNC_RATE_PLAYER_SECONDS)
    player_sync_timer.tick(fn=fetch_state_for_player, inputs=[user_id], outputs=[player_board, leave_queue_btn], show_progress=False)


    spectator_sync_timer = gr.Timer(SYNC_RATE_SPECTATOR_SECONDS)
    spectator_sync_timer.tick(fn=fetch_state_for_spectator, inputs=[], outputs=[spectator_board], show_progress=False)


    leaderboard_timer = gr.Timer(SYNC_RATE_LEADERBOARD_SECONDS)
    leaderboard_timer.tick(fn=fetch_leaderboard_for_player, inputs=[user_id, lang_input], outputs=[leaderboard_board], show_progress=False)
    lobby_timer = gr.Timer(TICK_LOBBY_WARMUP_SECONDS)
    lobby_timer.tick(
        fn=lobby_sync_check,
        inputs=[user_id, lang_input],
        outputs=[player_tab, main_tabs, login_box, leave_queue_btn, join_btn, player_board, status_msg],
        show_progress=False,
    )

threading.Thread(target=llm_queue_worker, daemon=True).start()
threading.Thread(target=tts_audio_queue_worker, daemon=True).start()
os.makedirs("assets", exist_ok=True)
demo.launch(allowed_paths=["./assets"], server_name="0.0.0.0", css=GLOBAL_CSS, theme=game_theme)
