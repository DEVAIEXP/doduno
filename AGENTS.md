# AGENTS.md - Technical Specification: DOD UNO (Deploy or Draw)

This file is the agent-facing source of truth for maintaining **DOD UNO** in this repository. Keep it aligned with the live implementation in `app.py`, `components.py`, `game_manager.py`, and `prompts.py`.

Do not replace the custom Gradio HTML component architecture with a conventional button/textbox bridge, a string-only HTML renderer, or an ordinary Gradio component layout. The browser board is a real custom `gr.HTML` component that talks directly to Python through `server_functions`.

---

## 1. Current Runtime Files

* `app.py` is the Gradio entrypoint. It owns:
  * `GLOBAL_CSS` for page, lobby, tab, background, and app-level styling.
  * `GLOBAL_JS` for browser audio, localStorage restore, and floating-card background initialization.
  * The Gradio `Blocks` layout, timers, server bridge functions, LLM client calls, queue worker, and cloud warmup flow.
* `components.py` owns reusable custom Gradio components:
  * `NeonToast(gr.HTML)` for reactive toast display.
  * `Board(gr.HTML)` for the full game board, including its board-scoped CSS, HTML template, JavaScript event handlers, `watch('value', ...)`, custom triggers, and server calls.
* `game_manager.py` owns the synchronized backend state machine:
  * `GameManager`
  * `global_server`
  * `llm_queue`
  * deck generation, lobby/queue state, gameplay rules, heartbeat cleanup, leaderboard persistence, TTS download/cache, and bot queueing.
* `prompts.py` owns:
  * `BOT_SYSTEM_PROMPT`
  * `DIRECTOR_SYSTEM_PROMPT`

---

## 2. Custom Gradio Component Contract

* The board must stay implemented through the new Gradio custom HTML component pattern:
  * `Board(value=state, server_functions=BOARD_SERVER_FUNCTIONS)`
  * `Board` must call `gr.HTML(..., html_template=..., css_template=..., js_on_load=..., server_functions=...)`.
  * JavaScript reads and mutates `props.value`.
  * JavaScript uses `watch('value', ...)` for reactive state changes.
  * JavaScript uses `trigger('show_toast', data)` and `trigger('force_leave_ui')` for custom Gradio events.
  * JavaScript calls Python through the `server` object exposed by `server_functions`.
* Keep `NeonToast(gr.HTML)` and `Board(gr.HTML)` as reusable component classes in `components.py`.
* Board-specific selectors belong in `Board`'s `css_template`.
* Page, lobby, tab, body, `#bg_canvas`, and `elem_classes="glass-lobby"` styles belong in `GLOBAL_CSS` in `app.py`.
* Do not remove `html_template`, `css_template`, or `js_on_load` from `Board` or `NeonToast`.

---

## 3. Runtime and Dependency Rules

* Use the existing `.venv` through `uv`.
* Install main app dependencies with:
  * `uv pip install --system-certs --python .\.venv\Scripts\python.exe -r requirements.txt`
* The main app targets Python 3.10+ and Gradio 6.x.
* Main app runtime dependencies include `gradio[oauth]`, `gradio_client`, `huggingface_hub`, `python-dotenv`, `hf_xet`, `requests`, `numpy`, `llama-cpp-python`, and the pinned CUDA PyTorch packages in `requirements.txt`.
* `requirements_nanovllml.txt` is for the separate NanoVLLM / VoxCPM service path and includes `nanovllm-voxcpm`, `soundfile`, and platform-specific `flash-attn` wheels.
* The current app uses `gradio_client.Client` to call the external LLM inference endpoint configured by:
  * `LLM_URL`
  * `LLM_API_KEY`
  * `HF_TOKEN`
  * `LLM_URL_PRIORITY` with `primary` or `fallback`
* TTS is provided through `TTS_API_URL` and optional `TTS_API_KEY`.
  * `TTS_URL_PRIORITY` with `primary` or `fallback`
  * `DOD_DISABLE_TTS=True` skips TTS warmup/downloads during development while keeping Director quote text in the match log.
* When `USE_LOCAL=True`, `LLM_URL` and `TTS_API_URL` from `.env` are used directly.
* When `USE_LOCAL=False`, LLM/TTS endpoint chains must come from the mapper dataset. Local `.env` URLs are not appended as fallbacks.
* Dataset locations are configured through:
  * `DOD_INFERENCE_MAPPER_DATASET_REPO_ID`
  * `DOD_INFERENCE_MAPPER_DATASET_REVISION`
  * `DOD_INFERENCE_MAPPER_DATASET_PATH`
  * optional full override `DOD_INFERENCE_MAPPER_URL`
  * `DOD_LEADERBOARD_DATASET_REPO_ID`
  * `DOD_LEADERBOARD_DATASET_PATH`
* `DOD_DISABLE_LLM` exists in `app.py`, but the current production inference path is the external LLM Gradio API flow, not an in-process llama.cpp engine in `app.py`.

---

## 4. Application Architecture

* The backend is `GameManager`.
* Global game state lives in `global_server = GameManager()`.
* All bot and director LLM work runs through the single FIFO `llm_queue`.
* `llm_queue_worker()` is started as one daemon thread after the Gradio layout is declared.
* The shared `BOARD_SERVER_FUNCTIONS` list must be passed to both `spectator_board` and `player_board`.
* Current board server bridge function names in `app.py` are:
  * `play_card`
  * `draw_card`
  * `select_wild_color`
  * `accuse_player`
  * `pass_turn_manual`
  * `shout_deploy`
  * `leave_game`
* Do not reintroduce the old `py_` prefixes. Board JavaScript calls `server.play_card(...)`, `server.draw_card(...)`, and the other names above.
* Other important app helpers:
  * `receive_toast`
  * `do_tick`
  * `fetch_state_for_player`
  * `fetch_state_for_spectator`
  * `fetch_leaderboard_for_player`
  * `execute_leave_ui`
  * `get_lang_code`
  * `get_hf_username`
  * `change_lang_ui`
  * `join_match`
  * `check_auto_login`
  * `lobby_sync_check`
  * `choose_dominant_stack`
  * `build_recent_director_context`
  * `process_queued_bot_turn`
  * `process_queued_director_quote`
  * `async_modal_warmup`
  * `llm_queue_worker`

---

## 5. Gradio Layout and Timers

* `gr.Blocks()` must keep:
  * `gr.HTML('<canvas id="bg_canvas"></canvas>')`
  * `gr.Tabs(elem_id="main_tabs")`
  * lobby `gr.Column(elem_classes="glass-lobby")`
  * lobby title and subtitle as `gr.HTML`, not `gr.Markdown`
  * `gr.LoginButton()` for optional Hugging Face login
  * `NeonToast()`
  * `Board(...)` for both spectator and player boards
  * leaderboard as `gr.HTML`
* The page launch must keep:
  * `demo.launch(allowed_paths=["./assets"], css=GLOBAL_CSS, theme=game_theme)`
* Timer rules:
  * `TICK_RATE_SERVER_SECONDS = 1`: `do_tick`
  * `SYNC_RATE_PLAYER_SECONDS = 1`: `fetch_state_for_player`
  * `SYNC_RATE_SPECTATOR_SECONDS = 2`: `fetch_state_for_spectator`
  * `SYNC_RATE_LEADERBOARD_SECONDS = 15`: `fetch_leaderboard_for_player`
  * `TICK_LOBBY_WARMUP_SECONDS = 1`: `lobby_sync_check`
* `demo.load` must continue to call `check_auto_login` with `GLOBAL_JS` so localStorage restore and audio/background initialization work.

---

## 6. Multiplayer and Lobby Rules

* `MAX_PLAYERS = 2`.
* A full match is one human player plus the mandatory AI bot `Nemotron`.
* `MIN_PLAYERS_TO_START` controls the minimum active room size required to start the lobby countdown.
* `LOBBY_START_COUNTDOWN_SECONDS` controls how long the lobby waits for more players after the minimum active room size is reached.
* Human players join through `join_match(...)`, which delegates to `global_server.join_lobby(...)`.
* The lobby supports either a manually typed player name or an optional Hugging Face OAuth login through `gr.LoginButton()`.
* The Hugging Face login button labels must be localized through `APP_UI`.
* `get_hf_username(request)` reads `request.request.session["oauth_info"]["userinfo"]["preferred_username"]`; `join_match(...)` uses that identity when the manual name field is empty.
* When a Hugging Face username is available, `check_auto_login(...)` hides the manual name field and locks the session identity for matches and leaderboard entries.
* The `hf_user_id` Gradio state stores the detected Hugging Face username so click/timer events keep the manual name field hidden even when `gr.Request` is not populated on a later event.
* When the user logs out of Hugging Face, the manual name field becomes visible again.
* When the first human joins and the room is not started, `GameManager.join_lobby(...)` automatically adds `Nemotron`.
* `Nemotron` is always required for active matches; queue rotation must promote the next human and then complete the room with `Nemotron`.
* Before a match starts, humans may join the active room until `MAX_PLAYERS`; after `global_server.game_started` becomes true, late arrivals must enter `global_server.queue` for the next match.
* The lobby start countdown appears only after the active room reaches `MIN_PLAYERS_TO_START`; when it expires, `lobby_sync_check(...)` starts warmup and the match begins through `async_modal_warmup()`.
* Additional users enter `global_server.queue`.
* During warmup, the first human plus `Nemotron` reserve the active room; later humans must remain in the queue and must not receive the warmup/player-room UI.
* Queued users keep lobby focus with the queue message and a lobby-level leave-queue button until they are promoted into the active match.
* The player tab and personalized board state may stay prepared for queued users, but the selected tab must remain the lobby until promotion.
* The lobby join button must stay disabled for a browser tab that already owns a player or queue slot, and must be re-enabled only after leaving the queue, leaving the match, or being removed from the finished match.
* The lobby leave-queue button must be explicitly hidden for anonymous, removed, active-player, and promoted-player states; avoid `gr.skip()`/plain `gr.update()` paths that preserve a stale visible button.
* `lobby_sync_check(...)` must refresh presence for queued users; otherwise they can be removed by heartbeat cleanup before the active match rotates.
* When a queued user is promoted, `lobby_sync_check(...)` must make the player tab visible and push a personalized `player_board` state with the promoted user's `viewer_id`.
* Queue rotation prepares the next room but does not call `init_game()` directly; `lobby_sync_check(...)` is responsible for detecting the full room, starting warmup, and letting `async_modal_warmup()` launch the match.
* Duplicate-name protection must remain active, including the short active-tab rejection window.
* Player and queue heartbeat cleanup must remain active through `GameManager.tick_countdown()`.
* A full room does not immediately start unless cloud services are warmed:
  * `join_match(...)` starts `async_modal_warmup()` when needed.
  * Players stay in the lobby with the warmup message while TTS and LLM inference wake up.
  * `lobby_sync_check(...)` moves joined users to the player tab once `global_server.game_started` becomes true.
* Leaving the game must update backend state and trigger the `force_leave_ui` custom HTML flow when needed.

---

## 7. Deck and Game Mechanics

The game is UNO-inspired and software-engineering themed.

* Victory and failure metrics:
  * `resolution >= 100` means victory.
  * `panic >= 100` means game over.
* A new game must clear historical event leakage and start on a non-wild active card when possible.
* `generate_full_deck()` returns 108 cards:
  * 100 colored cards across green/frontend, blue/backend, red/devops, and yellow/A.I.
  * Each colored stack has:
    * 1 `SUPER`
    * 2 each of `FIX`, `REFACTOR`, `TECH_DEBT`, `DOCS`, `PATCH`, `STACK_OVERFLOW`, `SPAGHETTI`, `BLIND_PR`, `BUG`, `SKIP`, `REVERSE`, and `ATTACK`
  * 8 wild cards:
    * 4 `WILD`
    * 4 `NUKE`
* Valid play logic must remain in `GameManager.is_valid_play(...)`.
* Wild color selection is a two-step flow:
  * play the wild/NUKE card
  * then call `select_wild_color`
* Do not collapse wild color handling in a way that breaks the template's color picker or `server.select_wild_color(...)` path.
* Drawing a card grants a `+10s` turn grace bonus capped at the normal turn limit.
* A drawn card should be playable by the bot immediately when valid; otherwise the bot passes.

---

## 8. Bot and Director LLM Rules

* All queued LLM tasks must pass through `llm_queue`.
* `llm_queue_worker()` must process tasks sequentially:
  * `bot_decision` -> `process_queued_bot_turn`
  * `director_quote` -> `process_queued_director_quote`
* Bot decisions:
  * use `BOT_SYSTEM_PROMPT`
  * call the mapped LLM endpoint through `predict_llm(...)`
  * pass `LLM_API_KEY`, prompt, JSON payload, temperature `0.1`, and serialized grammar schema
  * inject a `playable` boolean for every hand card using `global_server.is_valid_play(card)`
  * require raw JSON with `action`, `card_index`, and `chosen_color`
  * fall back to a local rule-based bot when the LLM is unavailable, returns invalid JSON, or chooses draw
* Director quotes:
  * use `DIRECTOR_SYSTEM_PROMPT`
  * call the mapped LLM endpoint with temperature `0.75`
  * include the current crisis title/description in the LLM payload so the quote matches both the played card and the active incident
  * include compact card effects, localized card name/feedback, and up to two recent Director quotes so the model avoids repeating that the crisis was already solved
  * validate generated quotes against known bad translation phrases before accepting them; invalid quotes must use the crisis fallback path
  * require raw JSON with `quote_en` and `quote_pt`
  * update only the matching log event whose `quote_id` equals the queued `event_id`
  * fall back to `CRISES_DATABASE` quotes if generation fails
* Keep the Markdown/thinking-block JSON extraction guards in both LLM paths unless replacing them with a safer parser.

---

## 9. TTS, Audio, Toasts, and End-Game Sync

* `GameManager.download_tts_language(cache_key, text, lang, store_cache=True)` is the central TTS download/cache function.
* It must send:
  * `control` from `TTS_CONTROLS`
  * `text`
  * `cfg_value`
  * `voice_id = TTS_VOICE_ID`
  * `seed = TTS_VOICE_SEED`
  * optional `Authorization: Bearer <TTS_API_KEY>` when set
* `process_queued_director_quote(...)` must start a daemon audio downloader thread after text generation so the LLM queue is not blocked by TTS cold starts.
* If Director text generation fails, the active crisis quote pool is used and still queued for TTS.
* If generated Director TTS produces no playable audio, the worker may try a one-shot active-crisis fallback quote; if all TTS endpoints fail, no broken audio event should be queued.
* Queued Director audio tasks must carry the current `audio_generation_id`; stale tasks must be discarded before calling TTS and late TTS responses must not be delivered after a match ends.
* Per-player audio delivery uses `global_server.pending_audios` and `director_audio` in `GameManager.get_state(...)`.
* The JavaScript client must pause and clear `window._activeDirectorAudio` before victory or defeat sounds.
* End-game toasts are emitted from `watch('value')` when `game_started` changes from true to false.
* Backend play functions should return an empty toast on final victory/game-over transitions to avoid duplicate toasts.
* `NeonToast` must remain the central toast display component.

---

## 10. Deploy Shout, Accusations, and Turn Timers

* Player turn time limit: `PLAYER_TURN_TIME_LIMIT_SECONDS = 30`.
* Deploy shout buffer: `SERVER_SHOUT_WINDOW_BUFFER_SECONDS = 6`.
* If a player reaches one card, they become vulnerable until they shout Deploy or the buffer expires.
* The bot has a high-probability auto-shout path when vulnerable.
* The bot can accuse a vulnerable human after the shout buffer expires.
* `check_turn_start_deploy`, `start_shout_window`, `shout_deploy`, `accuse_player`, `pass_turn_manual`, and `pass_turn` must stay consistent with the Board JavaScript countdown and auto-pass behavior.

---

## 11. Leaderboard Rules

* Leaderboard state is persisted through the Hugging Face dataset configured by `DOD_LEADERBOARD_DATASET_REPO_ID`.
* `load_leaderboard_from_hf()` loads the CSV configured by `DOD_LEADERBOARD_DATASET_PATH`.
* `async_save_leaderboard_to_hf()` saves updates asynchronously.
* `render_leaderboard_html(lang)` returns localized HTML for the leaderboard tab.
* Do not remove leaderboard cache fields from the state machine or the 15-second Gradio leaderboard timer.

---

## 12. UI and Asset Requirements

* Preserve the skeuomorphic arcade-console design.
* The page must include `#bg_canvas` and render the animated floating-card background through `GLOBAL_JS`.
* Use assets from `assets/`; do not regenerate placeholder card art over them.
* The lobby uses `assets/logo.jpeg`.
* Card categories and provider/player visuals rely on the icons referenced by `Board`'s template.
* The active crisis title uses a slow emergency-style pulse in the board CSS.
* Keep the locked `750px` board layout and hidden-overflow guard that prevents iframe resizing loops.
* Keep the 3D join button, engraved terminal input styling, tactile tabs, neon toast, and audio interruption behavior.
* Keep player/spectator behavior distinct:
  * spectator board is read-only
  * player board can call server functions
  * queue and restart banners render from the Board template

---

## 13. Refactoring Guardrails

* Refactors are welcome only when they preserve the current behavior and custom component event flow.
* Prefer standardizing names, extracting constants, and reducing repeated wrapper logic.
* Do not move game actions out of the `server_functions` pathway.
* Do not simplify away:
  * i18n dictionaries
  * leaderboard rendering/persistence
  * TTS queueing and per-player pending audio
  * lobby queueing
  * heartbeat cleanup
  * spectator sync
  * TTS/LLM warmup gate
  * local bot fallback
* Keep comments, docstrings, variable names, and function names in English for new code.
* Do not churn localized UI strings or gameplay copy unless the task explicitly asks for it.
* Existing mojibake-looking text may be an encoding display issue; verify file encoding before changing user-facing strings.

---

## 14. Git Attribution Rule

* This repository uses `.githooks/prepare-commit-msg` to append:
  * `Co-authored-by: Codex <noreply@openai.com>`
* The local checkout must have:
  * `git config core.hooksPath .githooks`
* Keep this hook unless the user explicitly asks to remove it. It exists for hackathon Codex attribution.

---

## 15. Validation Checklist

Run validation through `uv` and the existing `.venv`.

* Syntax:
  * `.\.venv\Scripts\python.exe -m py_compile app.py components.py game_manager.py prompts.py`
* Dependency check:
  * `uv pip list --python .\.venv\Scripts\python.exe`
* Fast local launch:
  * ensure required environment variables are available for LLM and TTS if testing full gameplay
  * `.\.venv\Scripts\python.exe app.py`
* Smoke routes:
  * `/gradio_api/file=assets/logo.jpeg`
  * `/gradio_api/file=assets/icon_nvidia.png`
  * at least one card icon route from `assets/`
* Before pushing:
  * `git log --pretty=fuller -1` should show `Co-authored-by: Codex <noreply@openai.com>` in the latest commit message.
