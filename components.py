from __future__ import annotations

from typing import Any, Callable

import gradio as gr


class NeonToast(gr.HTML):
    """Custom HTML toast that reacts to value changes through Gradio props."""

    def __init__(self, value: str = "", **kwargs: Any) -> None:
        """Create the toast component.

        Args:
            value: Initial toast text.
            **kwargs: Additional arguments forwarded to `gr.HTML`.
        """
        html_template = """
        <div id="toast-container" class="toast-hidden">
            <span id="toast-msg">${value}</span>
        </div>
        """
        css_template = """
        #toast-container {
            position: fixed; top: -100px; left: 50%; transform: translateX(-50%);
            background-color: #161122 !important; border: 2px solid #ff0055 !important; color: #ffffff !important;
            padding: 12px 25px; border-radius: 8px; font-weight: bold;
            box-shadow: 0 0 20px rgba(255, 0, 85, 0.6); z-index: 10000;
            transition: top 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            font-size: 14px; letter-spacing: 0.5px;
            display: flex; align-items: center; gap: 10px;
            pointer-events: none;
        }
        #toast-container.toast-show { top: 20px; }

        #toast-msg { color: #ffffff !important; }
        """
        js_on_load = """
        let toastTimeout;
        let toastShowTimer;
        let toastToken = 0;

        window.dodClearToast = () => {
            toastToken += 1;
            clearTimeout(toastTimeout);
            clearTimeout(toastShowTimer);
            const toastEl = element.querySelector('#toast-container');
            if (!toastEl) return;
            toastEl.querySelector('#toast-msg').innerText = "";
            toastEl.classList.remove('toast-show');
        };

        watch('value', () => {
            const toastEl = element.querySelector('#toast-container');
            if (!toastEl) return;

            const message = (typeof props.value === "string") ? props.value.trim() : "";
            if (!message) {
                window.dodClearToast();
                return;
            }

            const currentToken = toastToken + 1;
            toastToken = currentToken;
            clearTimeout(toastTimeout);
            clearTimeout(toastShowTimer);
            toastEl.classList.remove('toast-show');

            toastShowTimer = setTimeout(() => {
                if (toastToken !== currentToken) return;
                toastEl.querySelector('#toast-msg').innerText = props.value;
                toastEl.classList.add('toast-show');
                toastTimeout = setTimeout(() => {
                    if (toastToken !== currentToken) return;
                    window.dodClearToast();
                }, 5000);
            }, 50);
        });
        """
        super().__init__(value=value, html_template=html_template, css_template=css_template, js_on_load=js_on_load, **kwargs)

class Board(gr.HTML):
    """Custom reactive DOD UNO board component backed by Gradio HTML templates."""

    def __init__(
        self,
        value: dict[str, Any] | None = None,
        server_functions: list[Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a board bound to the DOD UNO custom HTML/CSS/JS templates.

        Args:
            value: Initial board state payload.
            server_functions: Python functions exposed to the browser-side component.
            **kwargs: Additional arguments forwarded to `gr.HTML`.
        """
        css_template = """
    .game-layout {
        display: flex !important;
        gap: 15px !important;
        width: 100% !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        color: #ffffff !important;
        height: 750px !important;
        overflow: hidden;
        background-color: #0f1117 !important;
        padding: 15px !important;
        border-radius: 12px !important;
        box-sizing: border-box !important;
    }
    .spectator-mode .card, .spectator-mode button, .spectator-mode .draw-pile-btn, .spectator-mode .color-btn { pointer-events: none !important; user-select: none !important; }
    .spectator-mode .audio-toggle-btn { pointer-events: auto !important; user-select: auto !important; }

    .left-col { flex: 3 !important; display: flex !important; flex-direction: column !important; gap: 8px !important; height: 100%; overflow-y: auto; overflow-x: hidden; scrollbar-width: thin; scrollbar-color: #44345d transparent; padding-top: 2px !important;}
    .left-col::-webkit-scrollbar { width: 4px; }
    .left-col::-webkit-scrollbar-track { background: transparent; }
    .left-col::-webkit-scrollbar-thumb { background: #44345d; border-radius: 4px; }
    .right-col { flex: 1 !important; display: flex !important; flex-direction: column !important; background-color: #0b0812 !important; border-radius: 12px !important; border: 1.5px solid #44345d !important; padding: 15px !important; box-sizing: border-box !important; height: 100%; }
    .right-col h3 { padding-right: 46px !important; min-height: 40px !important; box-sizing: border-box !important; }

    .crisis-box { background-color: #1a1525 !important; border: 2px solid #ff0055 !important; border-radius: 10px !important; padding: 10px 15px !important; box-shadow: 0 4px 15px rgba(255, 0, 85, 0.25) !important; color: #ffffff !important; width: 100% !important; box-sizing: border-box !important;}
    .alert-title { color: #ff0055 !important; font-weight: bold !important; text-shadow: 0 0 5px #ff0055 !important; }


    .status-container { margin-top: 4px !important; }
    .bar-label { display: flex !important; justify-content: space-between !important; font-size: 11px !important; color: #ffffff !important; font-weight: bold !important; }


    .progress-bar { background-color: #2a2235 !important; border-radius: 6px !important; height: 12px !important; overflow: hidden !important; border: 1px solid rgba(255, 255, 255, 0.15) !important; }
    .progress-fill { height: 100% !important; transition: width 0.4s ease !important; }
    .res-fill { background-color: #2ecc71 !important; box-shadow: 0 0 10px #2ecc71 !important; }
    .panic-fill { background-color: #e74c3c !important; box-shadow: 0 0 10px #e74c3c !important; }

    .board-container { display: flex !important; flex-direction: column !important; gap: 8px !important; height: 100%;}


    .opponents-row { display: flex !important; gap: 12px !important; justify-content: center !important; margin-bottom: 2px !important; flex-wrap: wrap !important; padding: 5px !important;}
    .opponent-badge { position: relative; background: rgba(0,0,0,0.4) !important; border-radius: 10px !important; padding: 6px 12px !important; text-align: center !important; color: white !important; border: 2px solid #44345d !important; transition: all 0.3s !important; min-width: 100px !important;}


    .opponent-badge.active {
        border-color: #00f3ff !important;
        animation: pulse-glow 2.5s infinite ease-in-out !important;
        transform: scale(1.03) !important;
        background: rgba(0,243,255,0.1) !important;
    }
    .opp-name { font-weight: bold !important; font-size: 13px !important; margin-bottom: 3px !important; color: #cbd5e0 !important; }
    .opp-cards { font-size: 16px !important; font-weight: bold !important; color: #2ecc71 !important; display: flex !important; align-items: center !important; justify-content: center !important; gap: 5px !important; }
    .opp-avatar { width: 24px !important; height: 24px !important; border-radius: 50% !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; overflow: hidden !important; background: rgba(10, 14, 28, 0.9) !important; border: 1px solid rgba(255, 255, 255, 0.25) !important; box-shadow: 0 1px 4px rgba(0,0,0,0.35) !important; flex-shrink: 0 !important; }
    .opp-avatar-img { width: 100% !important; height: 100% !important; object-fit: cover !important; display: block !important; }
    .opp-avatar-default { width: 15px !important; height: 15px !important; fill: #cbd5e0 !important; }
    .opp-risk { position: absolute; top: -8px; right: -8px; font-size: 10px !important; font-weight: bold !important; color: #fff !important; background: #ff0055; padding: 1px 5px; border-radius: 8px; animation: blink 1.5s infinite !important; }

    .my-hand-panel { margin-top: 2px !important; }


    .cards-list-horizontal { display: flex !important; overflow-x: auto !important; gap: 10px !important; padding: 5px !important; scrollbar-width: thin !important; scrollbar-color: #44345d transparent !important; background-color: rgba(0,0,0,0.2) !important; border-radius: 8px !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; min-height: 145px; align-items: center;}
    .cards-list-horizontal::-webkit-scrollbar { height: 6px; }
    .cards-list-horizontal::-webkit-scrollbar-thumb { background: #44345d; border-radius: 4px; }

    .queue-banner { background: rgba(241, 196, 15, 0.2) !important; border: 2px solid #f1c40f !important; color: #f1c40f !important; padding: 10px !important; text-align: center !important; font-weight: bold !important; border-radius: 8px !important; font-size: 16px !important; margin-top: 10px !important;}
    .restart-banner { background: rgba(0, 243, 255, 0.2) !important; border: 2px solid #00f3ff !important; color: #00f3ff !important; padding: 10px !important; text-align: center !important; font-weight: bold !important; border-radius: 8px !important; font-size: 16px !important; margin-bottom: 10px !important;}
    .board-toolbar { position: absolute !important; top: 8px !important; right: 8px !important; z-index: 50 !important; display: flex !important; gap: 6px !important; }
    .audio-toggle-btn { width: 34px !important; height: 34px !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; border-radius: 50% !important; border: 1px solid rgba(0, 243, 255, 0.75) !important; background: rgba(7, 20, 38, 0.82) !important; color: #00f3ff !important; font-size: 17px !important; cursor: pointer !important; box-shadow: 0 0 12px rgba(0, 243, 255, 0.25) !important; transition: transform 0.15s ease, background 0.15s ease, color 0.15s ease !important; }
    .audio-toggle-btn:hover { transform: translateY(-1px) scale(1.04) !important; background: rgba(0, 243, 255, 0.18) !important; color: #ffffff !important; }
    .audio-toggle-btn.is-muted { border-color: rgba(255, 0, 85, 0.85) !important; color: #ff6b9a !important; box-shadow: 0 0 12px rgba(255, 0, 85, 0.25) !important; }


    .table-board { display: flex !important; width: 100% !important; height: 200px !important; justify-content: center !important; gap: 40px !important; align-items: center !important; background: radial-gradient(circle, #1a365d 0%, #071426 100%) !important; border: 1.5px solid rgba(255, 255, 255, 0.08) !important; border-radius: 12px !important; padding: 20px !important; position: relative !important; box-sizing: border-box !important; box-shadow: inset 0 0 20px rgba(0,0,0,0.5) !important; margin-top: 4px !important;}
    .picker-box { position: absolute !important; top: 0 !important; left: 0 !important; width: 100% !important; height: 100% !important; background-color: rgba(7, 20, 38, 0.98) !important; border-radius: 12px !important; flex-direction: column !important; justify-content: center !important; align-items: center !important; z-index: 9999 !important; border: 2px solid #00f3ff !important; }
    .color-btns { display: flex !important; gap: 10px !important; margin-top: 10px !important; align-items: center !important; justify-content: center !important; }
    .color-btn { display: inline-flex !important; align-items: center !important; justify-content: center !important; height: 38px !important; padding: 0 15px !important; border: 1px solid #cbd5e0 !important; border-radius: 6px !important; font-weight: bold !important; cursor: pointer !important; color: #ffffff !important; box-sizing: border-box !important; line-height: 1 !important; margin: 0 !important; align-self: center !important; vertical-align: middle !important; }

    .draw-pile-btn { background-color: #f4f4f9 !important; border: 4px solid #ffffff !important; border-radius: 10px !important; display: flex !important; flex-direction: column !important; justify-content: center !important; align-items: center !important; cursor: pointer !important; color: #111115 !important; font-weight: bold !important; text-align: center !important; box-shadow: 2px 2px 0px #ffffff, 4px 4px 0px #1a1525, 6px 6px 0px #ffffff, 8px 8px 15px rgba(0,0,0,0.6), inset 0 0 20px rgba(0,0,0,0.12) !important; margin-right: 8px !important; margin-bottom: 8px !important; overflow: hidden !important; }
    .draw-pile-btn.card-large { padding: 0 !important; }
    .draw-pile-btn:hover { transform: translateY(-5px) !important; box-shadow: 2px 7px 0px #ffffff, 4px 9px 0px #1a1525, 6px 11px 0px #ffffff, 8px 13px 20px rgba(0,0,0,0.8) !important; }

    .draw-pile-card-face { width: 100% !important; height: 100% !important; display: flex !important; align-items: center !important; justify-content: center !important; background: #f4f4f9 !important; border-radius: 6px !important; position: relative !important; overflow: hidden !important; }
    .draw-pile-logo { width: 100% !important; height: 100% !important; object-fit: cover !important; display: block !important; }
    .draw-pile-label { position: absolute !important; left: 50% !important; bottom: 8px !important; transform: translateX(-50%) !important; max-width: calc(100% - 12px) !important; padding: 3px 7px !important; border-radius: 5px !important; background: rgba(244, 244, 249, 0.86) !important; color: #111115 !important; font-size: 11px !important; font-weight: 900 !important; letter-spacing: 0.5px !important; line-height: 1 !important; text-transform: uppercase !important; box-shadow: 0 2px 6px rgba(0,0,0,0.25) !important; white-space: nowrap !important; z-index: 1 !important; }

    .action-btn { padding: 8px 12px !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; border-radius: 6px !important; cursor: pointer !important; font-weight: bold !important; font-size: 11px !important; width: 120px !important; margin-top: 8px !important; font-family: inherit !important; text-align: center; }
    .pass-btn { background-color: #00f3ff !important; color: black !important; box-shadow: 0 0 10px rgba(0, 243, 255, 0.5) !important; border: none !important; }
    .shout-btn { background-color: #ff0055 !important; color: white !important; box-shadow: 0 0 10px rgba(255, 0, 85, 0.5) !important; border: none !important; animation: blink 1.5s infinite; }
    .btn-leave { background-color: transparent !important; color: #ff0055 !important; border: 1px solid #ff0055 !important; padding: 4px 10px !important; border-radius: 6px !important; font-size: 12px !important; cursor: pointer !important; font-weight: bold !important; transition: all 0.2s !important; }
    .btn-leave:hover { background-color: #ff0055 !important; color: #fff !important; }

    .card { flex-shrink: 0; position: relative !important; width: 100px !important; height: 140px !important; border-radius: 8px !important; background-color: #ffffff !important; display: inline-flex !important; flex-direction: column !important; justify-content: flex-start !important; gap: 5px !important; padding: 8px 6px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important; cursor: pointer !important; transition: transform 0.2s, box-shadow 0.2s !important; box-sizing: border-box !important; margin: 4px !important; border: 3.5px solid #ffffff !important; }
    .card:hover { transform: translateY(-8px) scale(1.04) !important; box-shadow: 0 8px 16px rgba(0,0,0,0.5) !important; z-index: 10 !important; position: relative; }
    .card-badge { position: absolute !important; top: 4px !important; left: 4px !important; background-color: rgba(0,0,0,0.65) !important; color: white !important; padding: 2px 5px !important; border-radius: 4px !important; font-weight: bold !important; font-size: 10px !important; border: 1px solid rgba(255,255,255,0.4) !important; box-shadow: 0 2px 4px rgba(0,0,0,0.5) !important; }


    .card-large { width: 105px !important; height: 150px !important; border-radius: 10px !important; padding: 8px 6px !important; }
    .card-large .card-diamond { width: 44px !important; height: 44px !important; margin: 5px auto 3px auto !important; flex-shrink: 0 !important; display: flex !important; justify-content: center !important; align-items: center !important;}
    .card-large .card-symbol { font-size: 22px !important; }
    .card-large .card-title-text { font-size: 10px !important; margin-top: 2px !important; }
    .card-large .card-stats { font-size: 8px !important; }

    .card-bg-green  { background-color: #2ecc71 !important; }
    .card-bg-blue   { background-color: #3498db !important; }
    .card-bg-red    { background-color: #e74c3c !important; }
    .card-bg-yellow { background-color: #f1c40f !important; }
    .card-bg-wild   { background-color: #111115 !important; }

    .card-back { background-color: #f4f4f9 !important; border: 4px solid #ffffff !important; box-shadow: inset 0 0 20px rgba(0,0,0,0.15), 0 4px 8px rgba(0,0,0,0.3) !important; justify-content: center !important; align-items: center !important; padding: 0 !important; }
    .card-back-inner { border-radius: 4px !important; display: flex !important; justify-content: center !important; align-items: center !important; box-shadow: inset 0 0 10px rgba(0,0,0,0.05) !important; }
    .card-back img { filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2)); opacity: 0.85; width: 70% !important;}

    .card-diamond { width: 50px !important; height: 50px !important; background-color: #ffffff !important; transform: rotate(45deg) !important; display: flex !important; justify-content: center !important; align-items: center !important; margin: 8px auto 6px auto !important; box-shadow: inset 0 2px 5px rgba(0,0,0,0.2) !important; border-radius: 8px !important; }
    .card-symbol { transform: rotate(-45deg) !important; font-size: 24px !important; font-weight: bold !important; }

    .card-title-text { font-size: 8px !important; font-weight: 900 !important; text-align: center !important; overflow-wrap: anywhere !important; line-height: 1.08 !important; text-transform: uppercase !important; min-height: 20px !important; display: flex !important; align-items: center !important; justify-content: center !important; }
    .text-light { color: #ffffff !important; text-shadow: 0 1px 3px rgba(0,0,0,0.95), 0 0 2px rgba(0,0,0,0.8) !important; }
    .text-dark  { color: #111115 !important; text-shadow: 0 1px 0 rgba(255,255,255,0.35) !important; }

    .card-stats { margin-top: auto !important; display: flex !important; justify-content: space-between !important; gap: 2px !important; font-size: 8px !important; font-weight: 900 !important; border-top: 1.5px dashed rgba(255,255,255,0.45) !important; padding-top: 3px !important; line-height: 1 !important; white-space: nowrap !important; }
    .card-stats span { display: inline-flex !important; align-items: center !important; justify-content: center !important; flex: 1 1 0 !important; min-width: 0 !important; padding: 2px 1px !important; border-radius: 3px !important; background: rgba(0,0,0,0.62) !important; box-shadow: 0 1px 2px rgba(0,0,0,0.45) !important; white-space: nowrap !important; }
    .stat-good { color: #d9ff5a !important; text-shadow: 0 1px 2px rgba(0,0,0,1) !important; }
    .stat-bad  { color: #ffc7d1 !important; text-shadow: 0 1px 2px rgba(0,0,0,1) !important; }
    .faded { opacity: 0.3 !important; }

    .log-box::-webkit-scrollbar { width: 4px; }
    .log-box::-webkit-scrollbar-track { background: transparent; }
    .log-box::-webkit-scrollbar-thumb { background: #44345d; border-radius: 4px; }
    .log-box { width: 100% !important; flex-grow: 1 !important; overflow-y: auto !important; color: #2ecc71 !important; font-size: 13px !important; line-height: 1.4 !important; padding-right: 5px !important; max-height: calc(100vh - 80px); scrollbar-width: thin; scrollbar-color: #44345d transparent; }
    @keyframes pulse-glow {
        0% {
            border-color: #00f3ff;
            box-shadow: 0 0 8px rgba(0, 243, 255, 0.2);
        }
        50% {
            border-color: #00a8ff;
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.6);
        }
        100% {
            border-color: #00f3ff;
            box-shadow: 0 0 8px rgba(0, 243, 255, 0.2);
        }
    }
    .panel-hand { background: radial-gradient(circle, #1a365d 0%, #071426 100%) !important; border: 1.5px solid rgba(255, 255, 255, 0.08) !important; border-radius: 10px !important; padding: 10px !important; opacity: 0.45 !important; transition: opacity 0.3s !important; color: #ffffff !important;}
    .panel-active {
        opacity: 1 !important;
        border: 1.5px solid #00f3ff !important;
        animation: pulse-glow 2.5s infinite ease-in-out !important;
    }
    .hand-title { margin: 0 0 8px 0 !important; color: #00f3ff !important; font-size: 13px !important; display: flex !important; justify-content: space-between !important; align-items: center !important; font-weight: bold !important;}
    .accuse-btn { background-color: #ff0055 !important; color: white !important; border: none !important; padding: 4px 10px !important; border-radius: 4px !important; font-size: 11px !important; cursor: pointer !important; font-weight: bold; box-shadow: 0 0 8px #ff0055 !important; margin-top: 5px;}
    #bg_canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        pointer-events: none;
    }
    
"""
        html_template = """
${(function() {
    const maxP = (value && value.max_players) ? value.max_players : 4;
    const initialI18n = (value && value.i18n) ? value.i18n : {};
    const muted = window.dodAudioMuted === undefined ? true : window.dodAudioMuted === true;
    const audioIcon = muted ? "🔇" : "🔊";
    const audioTitle = muted ? (initialI18n.unmute_audio || "Unmute audio") : (initialI18n.mute_audio || "Mute audio");
    const audioClass = muted ? "audio-toggle-btn is-muted" : "audio-toggle-btn";
    const audioToggleHtml = "<div class='board-toolbar'><button id='btn-audio-toggle' class='" + audioClass + "' title='" + audioTitle + "' aria-label='" + audioTitle + "'>" + audioIcon + "</button></div>";

    if (!value || (!value.game_started && !value.restart_countdown)) {
        const waitingStr = (value && value.i18n) ? value.i18n.waiting : "Waiting for {num} players to start...";
        const txt = waitingStr.replace("{num}", maxP);
        const warmingUp = value && value.is_warming_up;
        const startCountdown = (value && value.lobby_start_countdown) ? value.lobby_start_countdown : 0;
        const countdownStr = (value && value.i18n && value.i18n.start_countdown) ? value.i18n.start_countdown : "Starting with current players in {sec}s...";
        const warmupStr = (value && value.i18n && value.i18n.warmup_status) ? value.i18n.warmup_status : "DOD UNO: Cooking cloud audio assets... Please wait about 30-50 seconds!";
        const warmupHtml = warmingUp
            ? "<div style='display: inline-flex; align-items: center; justify-content: center; gap: 8px; margin-top: 12px; font-size: 14px; color: #00f3ff !important; text-align: center;'><div class='game-spinner'></div>" + warmupStr + "</div>"
            : "";
        const countdownHtml = (!warmingUp && startCountdown > 0)
            ? "<div style='margin-top: 10px; font-size: 14px; color: #00f3ff !important; text-align: center;'>" + countdownStr.replace("{sec}", startCountdown) + "</div>"
            : "";
        return "<div class='game-layout' style='justify-content: center; align-items: center; font-size: 20px; color: white !important; flex-direction: column; position: relative;'>" + audioToggleHtml + txt + countdownHtml + warmupHtml + "</div>";
    }

    const t = value.i18n;
    const stackLabels = {
        green: t.stack_green || "FRONTEND",
        blue: t.stack_blue || "BACKEND",
        red: t.stack_red || "DEVOPS",
        yellow: t.stack_yellow || "A.I."
    };
    const res = value.resolution;
    const panic = value.panic;
    const activeCard = value.active_card;
    const activePlayer = value.active_player;
    const isPickingColor = value.is_picking_color;
    const hasDrawn = value.has_drawn_this_turn;
    const waitingShout = value.waiting_for_shout;
    const hands = value.hands;
    const shoutedDeploy = value.has_shouted_deploy;
    const log = value.game_log;
    const players = value.players;
    const playerPictures = value.player_pictures || {};
    const queue = value.queue || [];
    const crisis = value.current_crisis;
    const drawPileIcons = [
        "/gradio_api/file=assets/icon_huggingface.png",
        "/gradio_api/file=assets/icon_modal.png",
        "/gradio_api/file=assets/icon_nvidia.png",
        "/gradio_api/file=assets/icon_openbmb.png",
        "/gradio_api/file=assets/icon_openai.png",
        "/gradio_api/file=assets/icon_gradio.png"
    ];
    if (!window.__dodDrawPileIcon) {
        window.__dodDrawPileIcon = drawPileIcons[Math.floor(Math.random() * drawPileIcons.length)];
    }
    const escapeAttr = (raw) => String(raw || "").replaceAll("&", "&amp;").replaceAll("'", "&#39;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

    const myId = value.viewer_id !== undefined ? value.viewer_id : "";
    const pIdx = players.indexOf(myId);
    const isSpectator = (pIdx === -1);
    const inQueue = (queue.indexOf(myId) !== -1);
    const queuePos = inQueue ? queue.indexOf(myId) + 1 : 0;

    const amIActive = (pIdx !== -1 && pIdx === activePlayer);
    const activeHand = isSpectator ? [] : hands[pIdx];

    const showPicker = isPickingColor && amIActive;
    const pickerDisplay = showPicker ? 'flex !important' : 'none !important';
    const isTurnStartShout = value.is_turn_start_shout || false;
    const passBtnDisplay = (amIActive && hasDrawn && !waitingShout) ? 'block' : 'none';
    const canShout = (activeHand.length === 1);
    const countdown = value.shout_countdown || 0;
    const shoutBtnDisplay = (amIActive && canShout && !shoutedDeploy[pIdx] && countdown > 0) ? 'block' : 'none';

    const layoutClass = isSpectator ? "game-layout spectator-mode" : "game-layout";
    let html = "<div class='" + layoutClass + "' style='position: relative;'>" + audioToggleHtml;

    html += "<div class='left-col'><div class='board-container'>";

    if (!value.game_started && value.restart_countdown > 0) {
        html += "<div class='restart-banner'>" + t.restarting.replace("{sec}", value.restart_countdown) + "</div>";
    }
    let opponentsHtml = "<div class='opponents-row'>";
    players.forEach((pName, i) => {
        if (i === pIdx) return;
        const isActive = (activePlayer === i);
        const pOneCard = (hands[i].length === 1);
        const pShouted = shoutedDeploy[i];

        let riskHtml = "";
        if (pOneCard && !pShouted && !waitingShout) {
            riskHtml = "<div class='opp-risk'>" + t.risk + "</div>";
        }

        const showAccuse = (pOneCard && !pShouted && !isSpectator && !waitingShout && !value.pending_wild_shout);
        let accuseHtml = showAccuse ? "<button class='accuse-btn' data-target='" + i + "'>" + t.accuse + "</button>" : "";

        const isBot = (pName.toLowerCase() === "nemotron");
        const avatarUrl = playerPictures[pName] || (isBot ? "/gradio_api/file=assets/nemotron.jpg" : "");
        const avatarImg = avatarUrl
            ? "<span class='opp-avatar'><img class='opp-avatar-img' src='" + escapeAttr(avatarUrl) + "' alt='" + escapeAttr(pName) + "'></span>"
            : "<span class='opp-avatar'><svg class='opp-avatar-default' viewBox='0 0 24 24'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg></span>";

        opponentsHtml += "<div class='opponent-badge " + (isActive ? "active" : "") + "'>" +
                         "<div class='opp-name'>" + pName + "</div>" +
                         "<div class='opp-cards'>" + avatarImg + " x" + hands[i].length + "</div>" +
                         riskHtml + accuseHtml + "</div>";
    });
    opponentsHtml += "</div>";

    html += opponentsHtml;
    const activeBadgeHtml = (activeCard && activeCard.badge) ? "<div class='card-badge'>" + activeCard.badge + "</div>" : "";
    const activeStack = activeCard ? activeCard.stack : "wild";
    const activeSymbol = activeCard ? activeCard.categorySymbol : "🚀";
    const activeName = activeCard ? activeCard.name : (t.final_deploy || "Final Deploy");
    const activeRes = activeCard ? activeCard.res : 0;
    const activePanic = activeCard ? activeCard.panic : 0;
    html += "<div class='crisis-box'>" +
            "  <div style='color: #ffffff !important; font-weight: bold;'>" + t.status + " <span class='alert-title'>" + crisis.title + "</span></div>" +
            "  <div style='margin-top: 5px; font-size: 13px; color: #cbd5e0 !important; font-weight: bold; margin-bottom: 12px;'>" + crisis.desc + "</div>" +
            "  <div class='status-container'>" +
            "    <div class='bar-label'><span style='color: #ffffff !important;'>" + t.res + "</span><span style='color: #ffffff !important;'>" + res + "%</span></div>" +
            "    <div class='progress-bar'><div class='progress-fill res-fill' style='width: " + res + "%'></div></div>" +
            "  </div>" +
            "  <div class='status-container' style='margin-top:10px;'>" +
            "    <div class='bar-label'><span style='color: #ffffff !important;'>" + t.panic + "</span><span style='color: #ffffff !important;'>" + panic + "%</span></div>" +
            "    <div class='progress-bar'><div class='progress-fill panic-fill' style='width: " + panic + "%'></div></div>" +
            "  </div>" +
            "</div>";
    html += "<div class='table-board'>" +
            "  <div class='picker-box' style='display: " + pickerDisplay + ";'>" +
            "    <div style='font-weight: bold; color: #00f3ff; font-size: 16px; letter-spacing: 1px;'>" + t.pick + "</div>" +
            "    <div class='color-btns'>" +
            "      <button class='color-btn' style='background-color: #2ecc71; color: #000;' data-color='green'>" + stackLabels.green + "</button>" +
            "      <button class='color-btn' style='background-color: #3498db;' data-color='blue'>" + stackLabels.blue + "</button>" +
            "      <button class='color-btn' style='background-color: #e74c3c;' data-color='red'>" + stackLabels.red + "</button>" +
            "      <button class='color-btn' style='background-color: #f1c40f; color: #000;' data-color='yellow'>" + stackLabels.yellow + "</button>" +
            "    </div>" +
            "  </div>" +
            "  <div style='display: flex; flex-direction: column; align-items: center; gap: 8px;'>" +
            "    <div style='font-size: 12px; color: #cbd5e0; font-weight: bold;'>" + t.draw + "</div>" +
            "    <div id='draw-pile' class='draw-pile-btn card-large'>" +
            "      <div class='draw-pile-card-face'>" +
            "        <img id='draw-pile-icon' class='draw-pile-logo' src='" + window.__dodDrawPileIcon + "' alt='" + (t.draw_pile_alt || "Draw pile provider") + "'>" +
            "        <span class='draw-pile-label'>" + t.draw_btn + "</span>" +
            "      </div>" +
            "    </div>" +
            "  </div>" +
            "  <div style='display: flex; flex-direction: column; justify-content: center; gap: 10px; width: 120px;'>" +
            "    <button id='btn-pass' class='action-btn pass-btn' style='display: " + passBtnDisplay + "'>" + t.pass + "</button>" +
            "    <button id='btn-shout' class='action-btn shout-btn' style='display: " + shoutBtnDisplay + "' data-txt='" + t.shout + "'>" + t.shout + " (" + countdown + "s)</button>" +
            "  </div>" +
            "  <div style='display: flex; flex-direction: column; align-items: center; gap: 8px;'>" +
            "    <div style='font-size: 12px; color: #cbd5e0; font-weight: bold;'>" + t.table + "</div>";

    html += "    <div class='card card-large card-bg-" + activeStack + "' style='margin: 0;'>" + activeBadgeHtml +
            "      <div class='card-diamond'><span class='card-symbol' style='color: var(--" + activeStack + ")'>" + activeSymbol + "</span></div>" +
            "      <div class='card-title-text " + (activeStack === 'yellow' ? 'text-dark' : 'text-light') + "'>" + activeName + "</div>" +
            "      <div class='card-stats'>" +
            "        <span class='" + (activeRes >= 0 ? 'stat-good' : 'stat-bad') + "'>Res: " + (activeRes >= 0 ? '+' : '') + activeRes + "%</span>" +
            "        <span class='" + (activePanic <= 0 ? 'stat-good' : 'stat-bad') + "'>Pan: " + (activePanic >= 0 ? '+' : '') + activePanic + "%</span>" +
            "      </div>" +
            "    </div>" +
            "  </div>" +
            "</div>";
    if (!isSpectator) {
        const isActive = (activePlayer === pIdx);
        const pOneCard = (activeHand.length === 1);
        const pShouted = shoutedDeploy[pIdx];
        let badgeText = "";
        if (pOneCard) {
            if (pShouted) badgeText = t.deploy_saved;
            else if (!waitingShout) badgeText = t.risk;
        }

        const myAvatarUrl = "/gradio_api/file=assets/icon_gradio.png";
        const myAvatarImg = "<img src='" + myAvatarUrl + "' style='width: 16px; height: 16px; vertical-align: middle; display: inline-block; margin-right: 4px; border-radius: 3px; object-fit: contain;'>";

        html += "<div class='my-hand-panel'>" +
                "  <div id='panel-p" + pIdx + "' class='panel-hand " + (isActive ? 'panel-active' : '') + "'>" +
                "    <h3 class='hand-title'>" +
                "      <span style='color:#ffffff !important;'>" + myId + " " + t.you + " <span style='color: #00f3ff; font-size: 12px; margin-left: 5px; font-weight: bold;'>(" + myAvatarImg + "x" + activeHand.length + ")</span><span style='color: #ff0055;'> " + badgeText + "</span></span>" +
                "      <button id='btn-leave' class='btn-leave'>🚪 " + t.leave + "</button>" +
                "    </h3>" +
                "    <div class='cards-list-horizontal'>";

        activeHand.forEach((card, cIdx) => {
            const resColor = card.res >= 0 ? 'stat-good' : 'stat-bad';
            const panicColor = card.panic <= 0 ? 'stat-good' : 'stat-bad';
            const textClass = card.stack === 'yellow' ? 'text-dark' : 'text-light';
            const cardBadgeHtml = card.badge ? "<div class='card-badge'>" + card.badge + "</div>" : "";

            html += "<div class='card card-bg-" + card.stack + " " + (!isActive ? 'faded' : '') + "' data-player='" + pIdx + "' data-card='" + cIdx + "'>" + cardBadgeHtml +
                    "  <div class='card-diamond'><span class='card-symbol' style='color: var(--" + card.stack + ")'>" + card.categorySymbol + "</span></div>" +
                    "  <div class='card-title-text " + textClass + "'>" + card.name + "</div>" +
                    "  <div class='card-stats'>" +
                    "    <span class='" + resColor + "'>Res: " + (card.res >= 0 ? '+' : '') + card.res + "%</span>" +
                    "    <span class='" + panicColor + "'>Pan: " + (card.panic >= 0 ? '+' : '') + card.panic + "%</span>" +
                    "  </div>" +
                    "</div>";
        });
        html += "  </div></div></div>";
    } else if (inQueue) {
        const qMsg = t.queue_msg.replace("{pos}", queuePos);
        html += "<div class='queue-banner'>" + qMsg + "</div>";
    }

    html += "</div></div>";
    const timerText = value.game_started ? " (" + (t.turn || "Turn") + ": " + value.turn_left + "s)" : "";

    html += "<div class='right-col'>" +
            "  <h3 style='color: #2ecc71 !important; margin-top: 0; font-size: 16px; border-bottom: 2px solid #44345d; padding-bottom: 10px;'>" + t.log + timerText + "</h3>" +
            "  <div class='log-box' id='auto-scroll-log'>" + log + "</div>" +
            "</div></div>";

    return html;
})()}
"""
        js_on_load = """
    const getMyId = () => {
        if (props.value && props.value.viewer_id !== undefined) return props.value.viewer_id;
        return "";
    };

    const selectLobbyTab = () => {
        setTimeout(() => {
            const tabButtons = document.querySelectorAll('#main_tabs > .tab-nav > button');
            if (tabButtons && tabButtons[0]) tabButtons[0].click();
        }, 0);
    };

    const returnToLobbyUi = () => {
        if (window._returningToLobby) return;
        window._returningToLobby = true;
        localStorage.removeItem('uno_name');
        localStorage.removeItem('uno_lang');
        selectLobbyTab();
        trigger('force_leave_ui');
        setTimeout(selectLobbyTab, 150);
        setTimeout(() => { window._returningToLobby = false; }, 1200);
    };

    const scheduleEndGameLobbyReturn = () => {
        if (window.endGameReturnTimer) {
            clearTimeout(window.endGameReturnTimer);
        }
        const restartSeconds = props.value && props.value.restart_countdown ? props.value.restart_countdown : 0;
        window.endGameReturnTimer = setTimeout(() => {
            if (props.value && !props.value.game_started) {
                returnToLobbyUi();
            }
            window.endGameReturnTimer = null;
        }, Math.max(0, restartSeconds * 1000) + 1200);
    };

    if (window.dodAudioMuted === undefined) {
        window.dodAudioMuted = true;
        localStorage.setItem("dod_audio_muted", "true");
    }

    const syncAudioToggle = () => {
        const btn = element.querySelector('#btn-audio-toggle');
        if (!btn) return;
        const isMuted = window.dodAudioMuted === true;
        const i18n = (props.value && props.value.i18n) ? props.value.i18n : {};
        btn.textContent = isMuted ? "🔇" : "🔊";
        btn.title = isMuted ? (i18n.unmute_audio || "Unmute audio") : (i18n.mute_audio || "Mute audio");
        btn.setAttribute("aria-label", btn.title);
        btn.classList.toggle("is-muted", isMuted);
    };

    const drawPileIcons = [
        "/gradio_api/file=assets/icon_huggingface.png",
        "/gradio_api/file=assets/icon_modal.png",
        "/gradio_api/file=assets/icon_nvidia.png",
        "/gradio_api/file=assets/icon_openbmb.png",
        "/gradio_api/file=assets/icon_openai.png",
        "/gradio_api/file=assets/icon_gradio.png"
    ];

    const rotateDrawPileIcon = () => {
        const currentIcon = window.__dodDrawPileIcon || "";
        const candidates = drawPileIcons.filter((icon) => icon !== currentIcon);
        const nextIcon = candidates[Math.floor(Math.random() * candidates.length)] || drawPileIcons[0];
        window.__dodDrawPileIcon = nextIcon;

        const iconEl = element.querySelector('#draw-pile-icon');
        if (iconEl) {
            iconEl.src = nextIcon;
        }
    };

    const handleServerResponse = (response) => {
        if (response) {
            if (response.toast && response.toast !== "") trigger('show_toast', {"msg": response.toast});
            if (response.state) {
                response.state.viewer_id = getMyId();
                props.value = response.state;
            }
        }
    };

    element.addEventListener('click', async (e) => {
        if (window.gameAudio) {
            window.gameAudio.init();
        }

        const myId = getMyId();

        if (e.target.id === 'btn-audio-toggle') {
            window.dodAudioMuted = !(window.dodAudioMuted === true);
            localStorage.setItem("dod_audio_muted", window.dodAudioMuted ? "true" : "false");
            if (window.dodAudioMuted) {
                stopDirectorAudioQueue(false);
            }
            syncAudioToggle();
            if (window.dodLobbyMusic) {
                window.dodLobbyMusic.sync(props.value || null);
            }
            return;
        }

        if (!myId || myId === "") return;

        if (e.target.id === 'btn-leave') {
            const res = await server.leave_game({ caller: myId });
            returnToLobbyUi();
            handleServerResponse(res);
            return;
        }

        if (props.value && props.value.game_started && (props.value.turn_handoff_left || 0) > 0) {
            return;
        }

        const drawPileBtn = e.target.closest('#draw-pile');
        if (drawPileBtn) {
            const players = (props.value && props.value.players) ? props.value.players : [];
            const myIndex = players.indexOf(myId);
            const canAttemptDraw = (
                props.value &&
                props.value.game_started &&
                myIndex === props.value.active_player &&
                !props.value.waiting_for_shout &&
                !props.value.is_picking_color
            );
            const res = await server.draw_card({ caller: myId });
            if (canAttemptDraw && res && !res.toast) {
                if (window.gameAudio) window.gameAudio.play('draw');
                rotateDrawPileIcon();
            }
            handleServerResponse(res);
            return;
        }

        const cardEl = e.target.closest('.card');
        if (cardEl && cardEl.dataset.player !== undefined) {
            const pIdx = parseInt(cardEl.dataset.player);
            const res = await server.play_card({ player: pIdx, card: parseInt(cardEl.dataset.card), caller: myId });
            handleServerResponse(res);
            return;
        }

        const accuseBtn = e.target.closest('.accuse-btn');
        if (accuseBtn) {
            if (window.gameAudio) window.gameAudio.play('warning');
            const res = await server.accuse_player({ target: parseInt(accuseBtn.dataset.target), caller: myId });
            handleServerResponse(res);
            return;
        }

        const colorBtn = e.target.closest('.color-btn');
        if (colorBtn) {
            if (window.gameAudio) window.gameAudio.play('play');
            const res = await server.select_wild_color({ color: colorBtn.dataset.color, caller: myId });
            handleServerResponse(res);
            return;
        }

        if (e.target.id === 'btn-pass') {
            if (window.gameAudio) window.gameAudio.play('play');
            const res = await server.pass_turn_manual({ caller: myId });
            handleServerResponse(res);
            return;
        }

        if (e.target.id === 'btn-shout') {
            if (window.gameAudio) window.gameAudio.play('shout');
            const res = await server.shout_deploy({ caller: myId });
            handleServerResponse(res);
            return;
        }
    });

    let shoutTimer = null;
    let countdownInterval = null;
    let isWaitingShoutLocal = false;
    let timeLeft = (props.value && props.value.shout_countdown) ? props.value.shout_countdown : 3;

    function stopDirectorAudioQueue(resetLastId = true) {
        if (window._activeDirectorAudio) {
            window._activeDirectorAudio.pause();
            window._activeDirectorAudio.currentTime = 0;
            window._activeDirectorAudio = null;
        }
        window._audioQueue = [];
        window._isAudioPlaying = false;
        if (resetLastId) {
            window._lastDirectorAudioId = null;
        }
    }

    watch('value', () => {
        window.dodAudioMuted = localStorage.getItem("dod_audio_muted") !== "false";
        syncAudioToggle();
        if (window.dodLobbyMusic) {
            window.dodLobbyMusic.sync(props.value || null);
        }
        const myId = getMyId();
        if (!props.value) return;
        if (myId && myId !== "") {
            const players = props.value.players || [];
            const queue = props.value.queue || [];
            const stillInGame = (players.indexOf(myId) !== -1 || queue.indexOf(myId) !== -1);

            if (!stillInGame) {
                if (!window.kickTimer) {
                    window.kickTimer = setTimeout(() => {
                        returnToLobbyUi();
                        window.kickTimer = null;
                    }, 1500);
                }
                return;
            } else {
                if (window.kickTimer) {
                    clearTimeout(window.kickTimer);
                    window.kickTimer = null;
                }
            }
        }

        if (!props.value.players) return;
        const pIdx = props.value.players.indexOf(myId);
        const amIActive = (pIdx !== -1 && pIdx === props.value.active_player);
        const isWaiting = props.value.waiting_for_shout;
        
        const wasStarted = window._lastGameStarted;
        const isStarted = props.value.game_started;
        const localName = localStorage.getItem('uno_name') || "";
        if (isStarted && localName !== "") {
            const players = props.value.players || [];
            if (players.indexOf(localName) !== -1) {
                const tabButtons = document.querySelectorAll('#main_tabs > .tab-nav > button');
                if (tabButtons && tabButtons[1] && !tabButtons[1].classList.contains('selected')) {
                    tabButtons[1].click(); // Redirects to 'Your Game' tab instantly!
                    if (window.dodLobbyMusic) {
                        window.dodLobbyMusic.sync(props.value || null);
                    }
                }
            }
        }

        if (isStarted && wasStarted !== true) {
            if (window.endGameReturnTimer) {
                clearTimeout(window.endGameReturnTimer);
                window.endGameReturnTimer = null;
            }
            window._lastEndToastKey = null;
            window._returningToLobby = false;
            if (window.dodClearToast) window.dodClearToast();
            if (wasStarted === false && window.gameAudio) window.gameAudio.play('shout');
        } else if (wasStarted === true && !isStarted) {
            stopDirectorAudioQueue();
            
            const endReason = props.value.game_end_reason || "";
            const endToastKey = [
                endReason || "ended",
                props.value.resolution || 0,
                props.value.panic || 0
            ].join(":");
            if (window._lastEndToastKey !== endToastKey) {
                window._lastEndToastKey = endToastKey;
                if (endReason === "victory") {
                    if (window.gameAudio) window.gameAudio.play('victory');
                    trigger('show_toast', {"msg": props.value.i18n.toast_victory});
                } else if (endReason === "abandon") {
                    if (window.gameAudio) window.gameAudio.play('game_over');
                    trigger('show_toast', {"msg": props.value.i18n.toast_game_over_abandon});
                } else if (endReason === "game_over" || endReason === "timeout") {
                    if (window.gameAudio) window.gameAudio.play('game_over');
                    trigger('show_toast', {"msg": props.value.i18n.toast_game_over});
                } else if (props.value.panic >= 100) {
                    if (window.gameAudio) window.gameAudio.play('game_over');
                    trigger('show_toast', {"msg": props.value.i18n.toast_game_over});
                } else if (props.value.resolution >= 100) {
                    if (window.gameAudio) window.gameAudio.play('victory');
                    trigger('show_toast', {"msg": props.value.i18n.toast_victory});
                }
            }
            scheduleEndGameLobbyReturn();
        }
        window._lastGameStarted = isStarted;
        if (!isStarted) {
            stopDirectorAudioQueue();
        }
        if (!window._playAudioQueue) {
            window._playAudioQueue = function() {
                if (window.dodAudioMuted) {
                    stopDirectorAudioQueue(false);
                    return;
                }
                if (window._isAudioPlaying) {
                    return; // Wait for the active audio to finish speaking
                }
                if (!window._audioQueue || window._audioQueue.length === 0) {
                    return; // No pending quotes in playlist
                }
                
                const nextItem = window._audioQueue.shift();
                try {
                    const dirAudio = new Audio("data:audio/wav;base64," + nextItem.b64);
                    dirAudio.volume = 0.85;
                    window._activeDirectorAudio = dirAudio;
                    window._isAudioPlaying = true;
                    
                    // Trigger next audio recursively when current audio ends
                    dirAudio.onended = function() {
                        window._isAudioPlaying = false;
                        window._activeDirectorAudio = null;
                        window._playAudioQueue(); 
                    };
                    
                    dirAudio.onerror = function() {
                        window._isAudioPlaying = false;
                        window._activeDirectorAudio = null;
                        window._playAudioQueue(); 
                    };
                    
                    dirAudio.play().catch(e => {
                        console.warn("Audio playback blocked by browser:", e);
                        window._isAudioPlaying = false;
                        window._activeDirectorAudio = null;
                        window._playAudioQueue();
                    });
                } catch (e) {
                    console.error("Failed to initialize HTML5 Audio element:", e);
                    window._isAudioPlaying = false;
                    window._playAudioQueue();
                }
            };
        }
        if (isStarted && props.value.active_card) {
            const currentCardId = props.value.active_card.id;
            if (window._lastCardId !== undefined && window._lastCardId !== currentCardId) {
                const card = props.value.active_card;
                const isSpecialAttack = card.drawTwo || card.drawFour || card.skip || card.reverse ||
                                       card.category === 'NUKE' || card.category === 'ATTACK' ||
                                       card.category === 'SKIP' || card.category === 'REVERSE';

                if (window.gameAudio) {
                    if (isSpecialAttack) {
                        window.gameAudio.play('attack');
                    } else {
                        window.gameAudio.play('play');
                    }
                }
            }
            window._lastCardId = currentCardId;
        }
        if (props.value.panic >= 80 && isStarted) {
            const nowTime = Date.now();
            if (!window._lastWarnTime || nowTime - window._lastWarnTime > 8000) {
                if (window.gameAudio) window.gameAudio.play('warning');
                window._lastWarnTime = nowTime;
            }
        }
        if (isStarted && props.value.director_audio && props.value.director_audio.b64 !== "") {
            const audioId = props.value.director_audio.id;
            if (window._lastDirectorAudioId !== audioId) {
                window._lastDirectorAudioId = audioId;
                if (window.dodAudioMuted) {
                    stopDirectorAudioQueue(false);
                    return;
                }
                try {
                    if (!window._audioQueue) {
                        window._audioQueue = [];
                    }
                    
                    // Push the newly received audio to the queue
                    window._audioQueue.push({
                        id: audioId,
                        b64: props.value.director_audio.b64
                    });
                    
                    // Trigger the queue player execution
                    window._playAudioQueue();
                } catch(e) {
                    console.error("Audio Queue push error", e);
                }
            }
        }
        
        if (amIActive && isWaiting && !isWaitingShoutLocal) {
            isWaitingShoutLocal = true;
            timeLeft = props.value.shout_countdown || 3;

            const btnShoutInit = element.querySelector('#btn-shout');
            let shoutTxt = btnShoutInit ? btnShoutInit.dataset.txt : props.value.i18n.shout;
            if (btnShoutInit) btnShoutInit.innerText = shoutTxt + " (" + timeLeft + "s)";
            if (window.gameAudio) window.gameAudio.play('warning');

            countdownInterval = setInterval(() => {
                timeLeft--;
                if (window.gameAudio) window.gameAudio.play('tick');
                const currentBtn = element.querySelector('#btn-shout');
                if (currentBtn) currentBtn.innerText = shoutTxt + " (" + timeLeft + "s)";
            }, 1000);

            shoutTimer = setTimeout(async () => {
                clearInterval(countdownInterval);
                isWaitingShoutLocal = false;
                const res = await server.pass_turn_manual({ caller: myId });
                handleServerResponse(res);
            }, (props.value.shout_countdown || 3) * 1000);
        }
        else if (!isWaiting || !amIActive) {
            if (isWaitingShoutLocal) {
                clearTimeout(shoutTimer);
                clearInterval(countdownInterval);
                isWaitingShoutLocal = false;
            }
        }

        if (isWaitingShoutLocal) {
            const currentBtn = element.querySelector('#btn-shout');
            let shoutTxt = currentBtn ? currentBtn.dataset.txt : props.value.i18n.shout;
            if (currentBtn) currentBtn.innerText = shoutTxt + " (" + timeLeft + "s)";
        }

        setTimeout(() => {
            const logBox = element.querySelector('#auto-scroll-log');
            if (logBox) logBox.scrollTop = logBox.scrollHeight;
        }, 150);
    });
"""
        super().__init__(
            value=value,
            html_template=html_template,
            css_template=css_template,
            js_on_load=js_on_load,
            server_functions=server_functions,
            **kwargs,
        )
