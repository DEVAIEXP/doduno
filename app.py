from __future__ import annotations

import json
import os
import random
import threading
from typing import Any

import gradio as gr
from gradio_client import Client
import numpy as np
from dotenv import load_dotenv

from components import Board, NeonToast
from game_manager import (
    HF_TOKEN,
    MAX_PLAYERS,
    SPACE_B_API_KEY,
    SPACE_B_URL,
    APP_UI,    
    CRISES_DATABASE,
    SYNC_RATE_LEADERBOARD_SECONDS,
    SYNC_RATE_PLAYER_SECONDS,
    SYNC_RATE_SPECTATOR_SECONDS,
    TICK_LOBBY_WARMUP_SECONDS,
    TICK_RATE_SERVER_SECONDS,
    GameState,
    ServerResponse,
    global_server,
    llm_queue,
)
from prompts import BOT_SYSTEM_PROMPT, DIRECTOR_SYSTEM_PROMPT

load_dotenv()

# Highest safe seed value accepted by llama.cpp's int32 seed path.
MAX_SEED = np.iinfo(np.int32).max
# Development switch that skips model download/loading during smoke tests.
LLM_DISABLED: bool = os.getenv("DOD_DISABLE_LLM", "").lower() in {"1", "true", "yes"}


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

print(f"[Space B Client] Connecting to: {SPACE_B_URL}", flush=True)
space_b_client = Client(SPACE_B_URL, token=HF_TOKEN)


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
    font-weight: 900 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    
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
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(0, 243, 255, 0.4);
    margin-right: 8px;
}
"""

GLOBAL_JS = """
() => {
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
    return [localStorage.getItem('uno_name') || '', localStorage.getItem('uno_lang') || 'English (US)'];
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


def fetch_state_for_player(uid: str) -> GameState | dict[str, Any]:
    """Fetch a personalized game state payload for a logged-in player.

    Args:
        uid: Player name stored in Gradio state.
    """

    if not uid or uid.strip() == "":
        return gr.update()
    state = global_server.get_state(uid)
    state["viewer_id"] = uid
    return state


def fetch_state_for_spectator() -> GameState:
    """Fetch the public spectator state payload."""
    state = global_server.get_state("")
    state["viewer_id"] = ""
    return state


def fetch_leaderboard_for_player(uid: str) -> str:
    """Render leaderboard HTML using the player's language preference.

    Args:
        uid: Player name used to resolve language preference.
    """
    lang = global_server.player_langs.get(uid, "en")
    return global_server.render_leaderboard_html(lang)


def execute_leave_ui() -> tuple[str, str, Any, Any, Any]:
    """Reset visible Gradio tabs after the custom board forces a leave action."""
    return "", "", gr.update(visible=False), gr.update(visible=True), gr.update(selected="tab_lobby")


def change_lang_ui(choice: str) -> tuple[Any, ...]:
    """Update lobby labels and leaderboard HTML after a language change.

    Args:
        choice: Label from the language radio component.
    """
    lang = "pt" if "Português" in choice else "en"
    t = APP_UI[lang]

    styled_title = f'<h1 style="text-align: center !important; color: #ffffff !important; text-shadow: 0 0 10px rgba(0, 243, 255, 0.45); font-size: 26px; font-weight: bold; margin: 0; width: 100%;">{t["title"]}</h1>'


    styled_sub = f'<p style="text-align: center !important; color: #cbd5e0 !important; font-size: 14px; margin: 5px 0 20px 0; width: 100%;">{t["subtitle"]}</p>'

    return (
        styled_title, styled_sub,
        gr.update(label=t["lang_label"]), gr.update(label=t["name_label"]), gr.update(value=t["btn_join"]),
        t["status"],
        gr.update(label=t["tab_lobby"]), gr.update(label=t["tab_player"]),
        gr.update(label=t["tab_leaderboard"]),
        global_server.render_leaderboard_html(lang)
    )


def join_match(player_name: str, lang_choice: str) -> tuple[Any, ...]:
    """Join a player to the match or queue from the lobby form.

    Args:
        player_name: Name typed by the user.
        lang_choice: UI language radio label.

    Returns:
        Gradio output tuple for user id, status, board state, tab visibility, tab selection, and lobby visibility.
    """
    lang_code = "pt" if "Português" in lang_choice else "en"
    name = player_name.strip()

    if not name:
        msg = "Digite um nome válido! / Enter a valid name!"
        return "", msg, gr.update(), gr.update(), gr.update(), gr.update()

    res_name = global_server.join_lobby(name, lang_code)

    if res_name == "DUPLICATE_REJECT":
        msg = "⚠️ Este nome já está ativo em outra aba! / This name is already active in another tab!"
        return "", msg, gr.update(), gr.update(), gr.update(), gr.update()

    new_state = global_server.get_state(res_name)

    new_state["viewer_id"] = res_name

    t = APP_UI[lang_code]
    if len(global_server.players) == MAX_PLAYERS and not global_server.game_started:
        if not global_server.modal_is_warm and not global_server.modal_is_warming_up:
            # Thread-Lock: Set warming up immediately on the main thread
            global_server.modal_is_warming_up = True
            threading.Thread(target=async_modal_warmup, daemon=True).start()
        
        # Keep players in lobby with a beautiful progress warning instead of redirecting them immediately
        if lang_code == "pt":
            msg = '<div style="display: inline-flex; align-items: center; justify-content: center; width: 100%; color: #00f3ff; font-weight: bold;"><div class="game-spinner"></div> DOD UNO: Cozinhando os assets de áudio na nuvem... Aguarde cerca de 30-40 segundos!</div>'
        else:
            msg = '<div style="display: inline-flex; align-items: center; justify-content: center; width: 100%; color: #00f3ff; font-weight: bold;"><div class="game-spinner"></div> DOD UNO: Cooking cloud audio assets... Please wait about 30-40 seconds!</div>'
            
        return res_name, msg, gr.update(value=new_state), gr.update(visible=True), gr.update(selected="tab_lobby"), gr.update(visible=True)    
    
    if res_name in global_server.players:
        msg = f"✅ {t['welcome_play'].replace('{name}', res_name)}"
    else:
        pos = global_server.queue.index(res_name) + 1
        msg = f"⏳ {t['welcome_queue'].replace('{pos}', str(pos))}"
        
    return res_name, msg, gr.update(value=new_state), gr.update(visible=True), gr.update(selected="tab_player"), gr.update(visible=False)



def check_auto_login(saved_name: str, saved_lang: str) -> tuple[Any, ...]:
    """Restore a saved localStorage session when it still exists on the server.

    Args:
        saved_name: Name restored from browser localStorage.
        saved_lang: Language label restored from browser localStorage.
    """
    if not saved_name or not saved_name.strip():
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    if saved_name in global_server.players or saved_name in global_server.queue:
        return join_match(saved_name, saved_lang)


    return gr.skip(), gr.skip(), gr.update(), gr.update(), gr.update(), gr.update()


def lobby_sync_check(uid: str) -> tuple[Any, Any]:
    """Move a logged-in user from the lobby to the player tab once the match starts.

    Args:
        uid: Player name stored in Gradio state.

    Returns:
        Gradio updates for the selected tab and lobby visibility.
    """
    if uid and uid.strip() != "" and global_server.game_started:
        return gr.update(selected="tab_player"), gr.update(visible=False)
    return gr.update(), gr.update()

def choose_dominant_stack(cards):
    """Choose the most common playable stack color from a hand.

    Args:
        cards: Cards to inspect.

    Returns:
        Dominant stack color, or a random standard stack when no colored cards exist.
    """
    stack_colors = [card["stack"] for card in cards if card.get("stack") in ["green", "blue", "red", "yellow"]]
    if stack_colors:
        return max(set(stack_colors), key=stack_colors.count)
    return random.choice(["green", "blue", "red", "yellow"])

def process_queued_bot_turn(bot_name: str) -> None:
    """Execute a queued bot decision using llama.cpp JSON constraints.

    Args:
        bot_name: Name of the bot player whose turn should be processed.
    """
    print(f"[Bot Decision] Initiating turn evaluation for: {bot_name}", flush=True)
    
    if not global_server.game_started or bot_name not in global_server.players:
        print(f"[Bot Decision] Aborted: game_started={global_server.game_started}", flush=True)
        return
        
    p_idx = global_server.players.index(bot_name)
    if p_idx != global_server.active_player:
        print(f"[Bot Decision] Aborted: p_idx={p_idx} is not active_player={global_server.active_player}", flush=True)
        return  # Safety check: ensure it's still their turn

    # Format hand state with "playable" flag (Cognitive Scaffold)
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
    
    active = global_server.active_card or {"stack": "wild", "category": "WILD"}
    
    state_payload = {
        "active_card": {"stack": active["stack"], "category": active["category"]},
        "metrics": {"resolution": global_server.resolution, "panic": global_server.panic},
        "hand": formatted_hand
    }

    try:
        # Check if space B URL is provided, otherwise fallback to local CPU inference
        if SPACE_B_URL:
            print("[Bot Decision] Dispatching external API call to Space B via Gradio Client...", flush=True)
            
            # The exact schema constraints for the Bot decision JSON
            bot_schema = {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["PLAY", "DRAW"]},
                    "card_index": {"type": ["integer", "null"]},
                    "chosen_color": {"type": "string", "enum": ["green", "blue", "red", "yellow", "none"]}
                },
                "required": ["action", "card_index", "chosen_color"]
            }
            
            # Parameters are passed strictly as positional arguments, including the 5th parameter (schema)
            result_str = space_b_client.predict(
                SPACE_B_API_KEY,                         # Parameter 1: api_key
                BOT_SYSTEM_PROMPT,                       # Parameter 2: system_prompt
                json.dumps(state_payload),               # Parameter 3: user_payload
                0.1,                                     # Parameter 4: temperature
                json.dumps(bot_schema),                  # Parameter 5: grammar_schema (serialized)
                api_name="/generate_inference"           # Target Gradio API endpoint
            )
            
            print(f"[Bot Decision] Raw Response from Space B: '{result_str}'", flush=True)
            
            # Iron-clad Markdown/Preamble shield. Extracts raw JSON even if the LLM 
            # outputs thinking blocks or code blocks like ```json ... ```
            result_str = result_str.strip()
            if "```" in result_str:
                parts = result_str.split("```")
                for part in parts:
                    part_clean = part.strip()
                    if part_clean.startswith("json"):
                        part_clean = part_clean[4:].strip()
                    if part_clean.startswith("{") and part_clean.endswith("}"):
                        result_str = part_clean
                        break
            elif "</think>" in result_str:
                result_str = result_str.split("</think>")[-1].strip()

            if not (result_str.startswith("{") and result_str.endswith("}")):
                raise RuntimeError(f"Space B returned a non-JSON error response: {result_str}")
                
            decision = json.loads(result_str)
            action = decision.get("action")
            card_idx = decision.get("card_index")
            chosen_color = decision.get("chosen_color")
            print(f"[Bot Decision] Parsed Strategic LLM Decision: {decision}", flush=True)

            # Apply move with safety validation
            if action == "PLAY" and card_idx is not None:
                card = hand[card_idx]
                if global_server.is_valid_play(card):
                    global_server.play_card(p_idx, card_idx, bot_name)
                    if card["stack"] == "wild":
                        color_to_apply = chosen_color
                        # Fallback check: If the LLM returned "none" or an invalid option, 
                        # we choose the color the Bot has the most of in its hand.
                        if color_to_apply not in ["green", "blue", "red", "yellow"]:
                            color_to_apply = choose_dominant_stack(hand)
                        
                        global_server.select_wild_color(color_to_apply, bot_name)
                else:
                    action = "DRAW"

            if action == "DRAW":
                # Fallback to local draw-play logic inside the try block
                raise RuntimeError("Bot chose DRAW, falling back to safe local CPU evaluation.")

        else:
            raise RuntimeError("Space B URL not configured.")

    except Exception as e:
        # FIX: The Circuit Breaker / Graceful Degradation pattern!
        # If the LLM API times out, runs out of quota, or fails, the Bot immediately
        # switches to this fast local rule-based CPU algorithm. The game remains 100% playable.
        print(f"[Bot Decision] LLM API Offline/Failed ({e}). Activating Local CPU Fallback...", flush=True)
        try:
            # Find all mathematically playable card indices in the Bot's hand
            playable_indices = [idx for idx, c in enumerate(hand) if global_server.is_valid_play(c)]
            
            if playable_indices:
                # Rule: Play the first playable card in hand
                chosen_idx = playable_indices[0]
                card = hand[chosen_idx]
                print(f"[Bot Fallback] Playing valid card: '{card['name'].get('en', '')}' at index {chosen_idx}", flush=True)
                global_server.play_card(p_idx, chosen_idx, bot_name)
                
                # Handle color selection if it was a wild card
                if card["stack"] == "wild":
                    color_to_apply = choose_dominant_stack(hand)
                    global_server.select_wild_color(color_to_apply, bot_name)
            else:
                # Rule: Draw a card and try to play it immediately
                print("[Bot Fallback] No playable cards. Drawing card from deck...", flush=True)
                global_server.draw_card(bot_name)
                
                updated_hand = global_server.hands.get(bot_name, [])
                if updated_hand:
                    drawn_card = updated_hand[-1]
                    if global_server.is_valid_play(drawn_card):
                        drawn_card_index = len(updated_hand) - 1
                        print(f"[Bot Fallback] Playing newly drawn card: '{drawn_card['name'].get('en', '')}'", flush=True)
                        global_server.play_card(p_idx, drawn_card_index, bot_name)
                        
                        if drawn_card["stack"] == "wild":
                            color_to_apply = choose_dominant_stack(updated_hand)
                            global_server.select_wild_color(color_to_apply, bot_name)
                    else:
                        global_server.pass_turn_manual(bot_name)
                else:
                    global_server.pass_turn_manual(bot_name)
                    
        except Exception as fe:
            print(f"[Bot Fallback] Critical failure in fallback runner: {fe}. Forcing pass.", flush=True)
            global_server.draw_card(bot_name)
            global_server.pass_turn_manual(bot_name)


def process_queued_director_quote(card_played, card_type, event_id):
    """Generates the IT Director quote, overwrites the exact log event, and queues player audios."""
    print(f"[Director Quote] Initiating generation for card: {card_played} ({card_type})", flush=True)
    
    active_langs = {global_server.player_langs.get(p, "en") for p in global_server.players}
    cache_key = str(event_id)

    if cache_key not in global_server.audio_cache:
        global_server.audio_cache[cache_key] = {"en": "", "pt": ""}

    state_payload = {"card_played": card_played, "type": card_type}

    try:
        if SPACE_B_URL:
            print(f"[Director Quote] Calling Space B API via Gradio Client...", flush=True)
            
            director_schema = {
                "type": "object",
                "properties": {
                    "quote_en": {"type": "string", "minLength": 10, "maxLength": 100},
                    "quote_pt": {"type": "string", "minLength": 10, "maxLength": 100}
                },
                "required": ["quote_en", "quote_pt"]
            }
            
            result_str = space_b_client.predict(
                SPACE_B_API_KEY,                         
                DIRECTOR_SYSTEM_PROMPT,                  
                json.dumps(state_payload),               
                0.75,                                    
                json.dumps(director_schema),             
                api_name="/generate_inference"           
            )
            
            print(f"[Director Quote] Raw Response from Space B: '{result_str}'", flush=True)
            
            # Iron-clad Markdown/Preamble shield
            result_str = result_str.strip()
            if "```" in result_str:
                parts = result_str.split("```")
                for part in parts:
                    part_clean = part.strip()
                    if part_clean.startswith("json"):
                        part_clean = part_clean[4:].strip()
                    if part_clean.startswith("{") and part_clean.endswith("}"):
                        result_str = part_clean
                        break
            elif "</think>" in result_str:
                result_str = result_str.split("</think>")[-1].strip()

            if not (result_str.startswith("{") and result_str.endswith("}")):
                raise RuntimeError(f"Space B returned a non-JSON error response: {result_str}")
                
            result = json.loads(result_str)
        else:
            raise RuntimeError("Space B URL not configured.")

        quote_en = result.get("quote_en", "")
        quote_pt = result.get("quote_pt", "")
        print(f"[Director Quote] Text generated: EN='{quote_en}' | PT='{quote_pt}'", flush=True)

        # Overwrite the SPECIFIC logged event matching our unique event_id
        for evt in reversed(global_server.events):
            if evt["key"] == "play" and evt["kwargs"].get("quote_id") == event_id:
                evt["kwargs"]["quote"] = {"en": quote_en, "pt": quote_pt}
                break

        # No more duplicate HTTP requests! Runs inside a separate non-blocking thread to manage cold starts.
        def async_audio_downloader_task(
            en_text: str,
            pt_text: str,
            ev_id: float,
            c_key: str,
            langs: set[str],
        ) -> None:
            """Download director quote audio without blocking the LLM worker.

            Args:
                en_text: English director quote to synthesize.
                pt_text: Portuguese director quote to synthesize.
                ev_id: Log-event identifier linked to this audio payload.
                c_key: Audio cache key shared with the client state payload.
                langs: Active player language codes that need audio.
            """
            print(f"[Async TTS] Background downloader started for event {ev_id}", flush=True)
            
            # We call the central method directly with the correct string-based 'c_key'!
            if "en" in langs:
                global_server.download_tts_language(c_key, en_text, "en")
            if "pt" in langs:
                global_server.download_tts_language(c_key, pt_text, "pt")

            # Append the new audio task to every player's pending playlist queue
            for p in global_server.players:
                if p not in global_server.pending_audios:
                    global_server.pending_audios[p] = []
                global_server.pending_audios[p].append({"id": ev_id, "cache_key": c_key})
            print(f"[Async TTS] Background downloader completed for event {ev_id}", flush=True)

        # Launch the non-blocking worker thread
        threading.Thread(
            target=async_audio_downloader_task,
            args=(quote_en, quote_pt, event_id, cache_key, active_langs),
            daemon=True
        ).start()

    except Exception as e:
        print(f"[Director Quote] Text generation failed: {e}", flush=True)
        try:
            crisis_data = CRISES_DATABASE[global_server.current_crisis_idx]
            q_list = crisis_data["quotes"].get(card_type, crisis_data["quotes"]["bad"])
            fallback_quote = random.choice(q_list)
            
            for evt in reversed(global_server.events):
                if evt["key"] == "play" and evt["kwargs"].get("quote_id") == event_id:
                    evt["kwargs"]["quote"] = fallback_quote
                    break
        except Exception as fe:
            print(f"[Director Quote] Critical failure applying fallback: {fe}", flush=True)

def async_modal_warmup():
    """Triggers a background non-blocking wakeup call to BOTH Modal (TTS) and Space B (LLM) 
       to handle GPU cold starts in parallel while players wait in the lobby."""
    print("[Warmup] Initiating background wakeup handshake to cloud GPU services...", flush=True)
    global_server.modal_is_warming_up = True
    
    # Track successful wakeups for both microservices
    modal_ready = False
    space_b_ready = False
    
    print("[Warmup] Sending wakeup ping to Modal (Audio Server)...", flush=True)    
    modal_ready = global_server.download_tts_language("0", "Starting", "en", False)
    
    # 2. WAKE UP LLM Server
    try:
        print("[Warmup] Sending wakeup ping to LLM Server...", flush=True)
        # We perform a lightweight, safe dummy prediction to trigger GGUF CUDA Graph compilation
        result_str = space_b_client.predict(
            SPACE_B_API_KEY,                         # Parameter 1: api_key
            "Warmup ping",                           # Parameter 2: system_prompt
            '{"ping": true}',                        # Parameter 3: user_payload
            0.1,                                     # Parameter 4: temperature
            "",                                      # Parameter 5: empty grammar_schema
            api_name="/generate_inference"           # Target Gradio API endpoint
        )
        if result_str and not result_str.startswith("❌"):
            space_b_ready = True
            print("[Warmup] Space B (Inference Server) successfully warmed up!", flush=True)
    except Exception as e:
        print(f"[Warmup] Space B wakeup failed: {e}", flush=True)

    # 3. IF BOTH ARE WARM, OPEN THE GATE AND START THE GAME!
    if modal_ready and space_b_ready:
        global_server.modal_is_warm = True
        print("[Warmup] ALL cloud GPU services are fully active! Launching match...", flush=True)
                
        if len(global_server.players) == MAX_PLAYERS and not global_server.game_started:
            global_server.init_game()
    else:
        print(f"[Warmup] Warning: Warmup incomplete. Modal={modal_ready}, SpaceB={space_b_ready}. Retrying on next join.", flush=True)
        global_server.modal_is_warm = False
        global_server.modal_is_warming_up = False
        
BOARD_SERVER_FUNCTIONS = [
    play_card,
    draw_card,
    select_wild_color,
    accuse_player,
    pass_turn_manual,
    shout_deploy,
    leave_game,
]

def llm_queue_worker() -> None:
    """Processes Bot decisions and IT Director Quotes sequentially to prevent CPU bottlenecks."""
    while True:
        task = llm_queue.get()
        try:
            if task["type"] == "bot_decision":
                process_queued_bot_turn(task["bot_name"])
            elif task["type"] == "director_quote":
                process_queued_director_quote(task["card_played"], task["card_type"], task["event_id"])
        except Exception as e:
            print(f"Error executing queued LLM task: {e}")
        finally:
            llm_queue.task_done()
            
with gr.Blocks() as demo:
    gr.HTML('<canvas id="bg_canvas"></canvas>')
    user_id = gr.State("")
    toast_ui = NeonToast()

    with gr.Tabs(elem_id="main_tabs") as main_tabs:
        with gr.Tab("🎮 Lobby & Spectator", id="tab_lobby") as lobby_tab:
            with gr.Column(elem_classes="glass-lobby", visible=True) as login_box:
                gr.HTML('<img src="/gradio_api/file=assets/logo.jpeg" class="lobby-logo" style="border-radius: 12px; max-width: 180px; display: block; margin: 0 auto 20px auto;">')
                title_html = gr.HTML('<h1 style="text-align: center !important; color: #ffffff !important; text-shadow: 0 0 10px rgba(0, 243, 255, 0.45); font-size: 26px; font-weight: bold; margin: 0; width: 100%;">DOD: Deploy or Draw! UNO GAME 🚀</h1>')
                sub_html = gr.HTML('<p style="text-align: center !important; color: #cbd5e0 !important; font-size: 14px; margin: 5px 0 20px 0; width: 100%;">Select language, enter your name and join the queue.</p>')
                lang_input = gr.Radio(choices=["English (US)", "Português (BR)"], value="English (US)", label=APP_UI['en']['lang_label'])
                name_input = gr.Textbox(label=APP_UI['en']['name_label'])
                join_btn = gr.Button(APP_UI['en']['btn_join'], variant="primary")
                status_msg = gr.Markdown(APP_UI['en']['status'])

            init_state = global_server.get_state("")
            init_state["viewer_id"] = ""
            spectator_board = Board(value=init_state, server_functions=BOARD_SERVER_FUNCTIONS)

        with gr.Tab("💻 Your Game", id="tab_player", visible=False) as player_tab:
            player_board = Board(value=init_state, server_functions=BOARD_SERVER_FUNCTIONS)

        with gr.Tab("🏆 Leaderboard", id="tab_leaderboard") as leaderboard_tab:
            leaderboard_board = gr.HTML(value=global_server.render_leaderboard_html("en"))

    lang_input.change(
        fn=change_lang_ui,
        inputs=[lang_input],

        outputs=[title_html, sub_html, lang_input, name_input, join_btn, status_msg, lobby_tab, player_tab, leaderboard_tab, leaderboard_board]
    )

    join_btn.click(
        fn=join_match,
        inputs=[name_input, lang_input],
        outputs=[user_id, status_msg, player_board, player_tab, main_tabs, login_box],
        js="(n, l) => { localStorage.setItem('uno_name', n); localStorage.setItem('uno_lang', l); return [n, l]; }"
    )

    demo.load(
        fn=check_auto_login,
        inputs=[name_input, lang_input],
        outputs=[user_id, status_msg, player_board, player_tab, main_tabs, login_box],
        js=GLOBAL_JS
    )

    player_board.show_toast(fn=receive_toast, inputs=None, outputs=toast_ui)
    spectator_board.show_toast(fn=receive_toast, inputs=None, outputs=toast_ui)

    player_board.force_leave_ui(fn=execute_leave_ui, inputs=None, outputs=[user_id, status_msg, player_tab, login_box, main_tabs])


    tick_timer = gr.Timer(TICK_RATE_SERVER_SECONDS)
    tick_timer.tick(fn=do_tick, inputs=[], outputs=[])


    player_sync_timer = gr.Timer(SYNC_RATE_PLAYER_SECONDS)
    player_sync_timer.tick(fn=fetch_state_for_player, inputs=[user_id], outputs=[player_board])


    spectator_sync_timer = gr.Timer(SYNC_RATE_SPECTATOR_SECONDS)
    spectator_sync_timer.tick(fn=fetch_state_for_spectator, inputs=[], outputs=[spectator_board])


    leaderboard_timer = gr.Timer(SYNC_RATE_LEADERBOARD_SECONDS)
    leaderboard_timer.tick(fn=fetch_leaderboard_for_player, inputs=[user_id], outputs=[leaderboard_board])
    lobby_timer = gr.Timer(TICK_LOBBY_WARMUP_SECONDS)
    lobby_timer.tick(
        fn=lobby_sync_check,
        inputs=[user_id],
        outputs=[main_tabs, login_box]
    )

threading.Thread(target=llm_queue_worker, daemon=True).start()
os.makedirs("assets", exist_ok=True)
demo.launch(allowed_paths=["./assets"], css=GLOBAL_CSS, theme=game_theme)
