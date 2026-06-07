# AGENTS.md - Technical Specification: DOD UNO (Draw or Deploy) - Reference-Aligned v2.1

This file is the agent-facing source of truth for maintaining **DOD UNO** in this repository. The implementation in `app.py` must follow the manually authored reference in `app_ref.py`. Do not replace it with a conventional Gradio button/textbox bridge or a separately rendered HTML string.

---

## 1. Reference Implementation Contract

* `app_ref.py` is the behavioral and UI reference. `app.py` should preserve its architecture, event flow, assets, and game logic.
* The UI is built with the new Gradio custom HTML component pattern:
  * `gr.HTML(value=state, html_template=HTML_TEMPLATE, css_template=CSS_TEMPLATE, js_on_load=JS_ON_LOAD, server_functions=[...])`
  * JavaScript reads and mutates `props.value`.
  * JavaScript uses `watch('value', ...)` for reactive state changes.
  * JavaScript uses `trigger('event_name', data)` for custom Gradio events such as `show_toast` and `force_leave_ui`.
  * JavaScript calls Python directly through the `server` object exposed by `server_functions`.
* Do not convert the custom `gr.HTML` board into a hand-built string renderer, hidden textbox command bus, or ordinary Gradio component layout.
* Keep `NeonToast(gr.HTML)` as a reusable custom HTML component class. This follows Gradio's documented custom component-class pattern.

---

## 2. Runtime & Dependency Rules

* Use the existing `.venv` environment through `uv`.
* Install dependencies with:
  * `uv pip install --system-certs --python .\.venv\Scripts\python.exe -r requirements.txt`
* The app targets Python 3.10+ and Gradio 6.x.
* Required runtime libraries include `gradio`, `llama-cpp-python`, `huggingface_hub`, `python-dotenv`, `hf_xet`, `requests`, `numpy`, and the CUDA PyTorch packages already pinned in `requirements.txt`.
* The local LLM is `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`, file `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`.
* The Llama constructor must not receive a static seed. Dynamic seeds belong inside each `create_chat_completion(...)` call through `seed=randomize_seed_fn(-1, True)`.
* `DOD_DISABLE_LLM=1` may be used only for local syntax/launch smoke tests. Normal runtime should load the GGUF model.

---

## 3. Application Architecture

* The backend is a synchronized Python `GameServer` state machine.
* Global state lives in `global_server`.
* The server exposes board operations through thin wrapper functions:
  * `py_play_card`
  * `py_draw_card`
  * `py_select_wild_color`
  * `py_accuse_player`
  * `py_pass_turn_manual`
  * `py_shout_deploy`
  * `py_leave_game`
* Keep the shared `BOARD_SERVER_FUNCTIONS` list and pass it to both spectator and player boards.
* State updates are pushed by Gradio timers:
  * server tick every 1 second
  * player sync every 1 second
  * spectator sync every 2 seconds
  * leaderboard sync every 15 seconds
* The UI state payload must include `viewer_id`, localized strings, active card, players, queue, countdowns, metrics, pending audio, and leaderboard-facing data needed by the HTML template.

---

## 4. Multiplayer & Lobby Rules

* Exactly two active players are supported.
* Human players join through the lobby. The AI bot is named `Nemotron`.
* Additional users enter the spectator/queue path rather than replacing active players.
* Duplicate-name protection must remain active.
* Heartbeat and room inactivity cleanup must remain active.
* Leaving the game must update backend state and trigger the `force_leave_ui` custom HTML event flow when needed.

---

## 5. Deck & Game Mechanics

The game is UNO-inspired and software-engineering themed.

* Metrics:
  * `resolution` reaches 100% for victory.
  * `panic` reaches 100% for game over.
* Cards use stack/category semantics from the reference implementation.
* The standard deck must remain 108 cards:
  * 76 base cards across green/frontend, blue/backend, red/devops, and yellow/AI.
  * 24 action cards: skip, reverse, attack.
  * 8 wild cards: wild and nuke.
* A new game must clear historical event leakage and must start on a non-wild active card.
* Wild color selection is a two-step flow. Do not collapse it in a way that breaks the template's `select_wild_color` call path.

---

## 6. Bot & LLM Queue

* All LLM calls must run through the single FIFO `llm_queue`.
* `llm_queue_worker` must remain a daemon thread and must process bot decisions and director quotes sequentially.
* Bot decisions use temperature `0.1` and JSON response constraints.
* Director quotes use temperature `0.75` and return `quote_en` and `quote_pt`.
* Before sending the bot hand to the LLM, the backend must calculate `playable` with `global_server.is_valid_play(card)` and inject that boolean per card.
* The bot prompt must instruct the model to consider only cards where `"playable": true`.
* If the bot decides to draw, the server must immediately inspect the drawn card:
  * play it if valid
  * select a wild color if needed
  * otherwise call `pass_turn_manual(...)`

---

## 7. Timers, Deploy Shout, and Accusations

* Player turn time limit: 30 seconds.
* Shout Deploy buffer: 6 seconds.
* Drawing a card grants a `+10s` grace bonus capped at 30 seconds.
* If a player reaches one card, they become vulnerable until they shout Deploy or the buffer expires.
* The bot has a 90% chance per tick to auto-shout when it is vulnerable.
* The bot has a 25% chance per second to accuse a vulnerable human after the shout buffer expires.

---

## 8. UI/UX Requirements

* Preserve the skeuomorphic arcade-console design from `app_ref.py`.
* The page must include `#bg_canvas` and render the animated floating-card background.
* Use the updated assets in `assets/`; do not regenerate placeholder card art over them.
* The lobby uses `assets/logo.jpeg`.
* Bot/player avatars use the relevant provider icons already referenced by the HTML template.
* The main tab container must keep `elem_id="main_tabs"`.
* The lobby must keep `elem_classes="glass-lobby"`.
* Lobby title and subtitle are intentionally `gr.HTML`, not `gr.Markdown`, to avoid Gradio/Svelte alignment and truncation issues.
* Keep the `750px` locked layout and hidden-overflow guard that prevents Hugging Face iframe resizing loops.
* Keep the 3D join button, engraved terminal inputs, tactile tabs, neon toast, and audio-interruption behavior.

---

## 9. Audio, Toasts, and End Game Sync

* Director quote generation may queue bilingual audio through `TTS_API_URL`.
* Pending audio must be cleared when game over/victory happens.
* The JavaScript client must pause and clear `window._activeDirectorAudio` before playing victory or defeat sounds.
* End-game toasts must be emitted from the `watch('value')` state transition when `game_started` changes from true to false.
* Backend play functions should return an empty toast on final victory/game-over transitions to avoid duplicate toasts.

---

## 10. Refactoring Guardrails

* Refactors are welcome only when they preserve the reference behavior.
* Prefer standardizing names, extracting repeated lists/constants, and reducing duplicated wrapper data.
* Do not move game actions out of the `server_functions` pathway.
* Do not remove `html_template`, `css_template`, or `js_on_load` from the board components.
* Do not simplify away i18n dictionaries, leaderboard rendering, TTS queueing, lobby queueing, heartbeat cleanup, or spectator sync.
* Keep code comments and function names understandable in English where new code is added, but do not churn existing working strings or localized text unnecessarily.

---

## 11. Validation Checklist

Run validation through `uv` and the existing `.venv`.

* Syntax:
  * `.\.venv\Scripts\python.exe -m py_compile app.py`
* Dependency check:
  * `uv pip list --python .\.venv\Scripts\python.exe`
* Fast launch smoke without model download:
  * `DOD_DISABLE_LLM=1` on POSIX or `$env:DOD_DISABLE_LLM='1'` on PowerShell before launching.
* Normal launch:
  * `.\.venv\Scripts\python.exe app.py`
* Verify `/gradio_api/file=assets/logo.jpeg`, `/gradio_api/file=assets/icon_nvidia.png`, and at least one card icon route return image responses.
