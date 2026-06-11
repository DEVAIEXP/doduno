<p align="center">
  <img src="assets/logo.jpeg" alt="DOD - Deploy or Draw logo" width="180">
</p>

<h1 align="center">DOD - Deploy or Draw</h1>

<p align="center">
  <strong>A Multiplayer UNO Game where production incidents become chaos, comedy, and AI-powered table drama.</strong>
</p>

DOD - (Deploy or Draw) is a software-engineering themed multiplayer UNO game built for the **Build Small Hackathon**. Players race to resolve a live production crisis before the Director's panic meter explodes, while an AI opponent makes strategic card decisions and an AI Director reacts to every play with short, dramatic, bilingual voice lines.

This is not just a card game with AI sprinkled on top. The AI is load-bearing: Nemotron plays as a mandatory rival, the Director improvises crisis-aware commentary, and VoxCPM2 turns those reactions into arcade-style table banter. The result is a strange little deploy-night toy: part UNO, part incident war room, part absurd corporate theater.

Built with Gradio custom HTML components, Hugging Face Spaces, NVIDIA Nemotron Nano 4B, and VoxCPM2.

## Prerequisites

- Python 3.10 exactly. The local LLM and NanoVLLM requirements use prebuilt `cp310-cp310` wheels to avoid native Windows builds.
- Git with submodule support
- `uv`
- NVIDIA CUDA GPU for the local NanoVLLM/VoxCPM TTS service
- Hugging Face account/token only if you use private remote datasets, want authenticated leaderboard persistence, or deploy with OAuth login in a Space

Install `uv` if needed:

```bash
pip install uv
```

If dependency installation reports that a `cp310-cp310` wheel is incompatible, check that the virtual environment is using Python 3.10. Those wheels are intentionally pinned so Windows users do not need Visual Studio Build Tools for native compilation.

## Hardware Requirements

Local inference requires an NVIDIA CUDA GPU.

- Local LLM server only: use a GPU with at least 4 GB VRAM.
- Local LLM server plus local NanoVLLM TTS: use at least 12 GB VRAM for a good gameplay experience.
- Windows with 8 GB VRAM can sometimes run both services because shared GPU memory may spill over into system RAM, but expect slower audio generation and delayed Director voice playback.
- If GPU memory is saturated, TTS can become slow enough that queued audio arrives late, and the LLM server can also become slower because both services are competing for memory.
- Linux generally does not provide the same practical shared-memory spillover behavior for this workload, so 8 GB VRAM is not recommended for running both local services. Use 12 GB VRAM or more.

If your GPU has limited VRAM, use one of these lighter setups:

- run only the local LLM and set `DOD_DISABLE_TTS=True`
- run the local LLM and use Modal for TTS
- use remote endpoints for both LLM and TTS

## Clone

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/DEVAIEXP/doduno.git
cd doduno
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

The `dod-llm-server` submodule is required when running the LLM service locally.

## Environment File

Create your local environment file from the example:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

For a fully local development setup, use this shape:

```env
DOD_USE_LOCAL_DATA=True
DOD_DISABLE_TTS=False
DOD_DISABLE_LOGS=False
DOD_USE_LOCAL_API=True
DOD_MAX_PLAYERS=2
DOD_MIN_PLAYERS_TO_START=2
DOD_LOBBY_START_COUNTDOWN_SECONDS=30

TTS_API_URL=http://127.0.0.1:8000
TTS_API_MODE=gradio
LLM_URL=http://127.0.0.1:7880

TTS_API_KEY=your_local_tts_key
LLM_API_KEY=your_local_llm_key
# Optional for private remote datasets only:
# HF_TOKEN_DATASET=your_huggingface_dataset_token
```

For local-only development, `TTS_API_KEY` and `LLM_API_KEY` are mainly pass-through values used by the game when calling the local APIs. They can be any value as long as the game and the local service agree on the same value. Treat them as real secrets only when the service is exposed remotely, through a public tunnel, or deployed outside your machine.

`DOD_MAX_PLAYERS` controls the active room size and includes the mandatory Nemotron bot. For example, `DOD_MAX_PLAYERS=3` means up to two human players plus Nemotron. `DOD_MIN_PLAYERS_TO_START` controls when the lobby countdown can begin, and `DOD_LOBBY_START_COUNTDOWN_SECONDS` controls how long the lobby waits for more players before starting.

Set `DOD_DISABLE_TTS=True` if you want faster development runs without calling the TTS service. Director lines will still appear as text in the match log, but they will not be audible.

Set `DOD_DISABLE_LOGS=True` to hide app-authored operational console logs such as warmup, mapper, TTS, and connection chatter. Errors and compact bot decisions still print. This does not affect the in-game server log shown inside the match UI.

`HF_TOKEN_DATASET` is not required for the public model downloads used by the local services. Configure it only when your inference mapper or leaderboard datasets are private, or when your deployment environment needs authenticated Hugging Face Hub access. Create this token from your Hugging Face account settings page under **Access Tokens**, then paste it as `HF_TOKEN_DATASET` in `.env`.

For local OAuth testing, Gradio uses the Hugging Face credentials available on your machine. Keep `HF_TOKEN` out of `.env`; this app uses `HF_TOKEN_DATASET` for private datasets so it does not override `hf auth login`. If you see a `401 Unauthorized` error from `whoami-v2`, remove invalid `HF_TOKEN` values from your shell environment or run `hf auth login` with a valid account.

The local LLM server runs from inside the `dod-llm-server` submodule, so it needs its own `.env` file. Copy its example and use the same `LLM_API_KEY` value configured in the root `.env`:

Windows PowerShell:

```powershell
Copy-Item dod-llm-server\.env.example dod-llm-server\.env
```

Linux/macOS:

```bash
cp dod-llm-server/.env.example dod-llm-server/.env
```

When `DOD_USE_LOCAL_DATA=True`, the app reads local data from:

- Windows: `%USERPROFILE%\.dod\inference_map.json`
- Windows: `%USERPROFILE%\.dod\leaderboard.csv`
- Linux/macOS: `~/.dod/inference_map.json`
- Linux/macOS: `~/.dod/leaderboard.csv`

The local `inference_map.json` is only needed when `DOD_USE_LOCAL_API=False`, because in that mode the game resolves LLM/TTS endpoints through the mapper. When `DOD_USE_LOCAL_API=True`, the game uses `LLM_URL` and `TTS_API_URL` directly and does not need `inference_map.json`.

The app does not create `inference_map.json` automatically. Create it yourself when using local data plus mapper-based endpoint resolution. A `fallback` endpoint is optional: if you provide only `primary`, the app will run with one endpoint and simply has no backup if that endpoint fails.

When `DOD_USE_LOCAL_API=True`, the app uses `LLM_URL` and `TTS_API_URL` directly instead of the remote inference mapper.

## Install the Main Game

Windows PowerShell:

```powershell
uv venv .venv --python 3.10
uv pip install --system-certs --python .\.venv\Scripts\python.exe -r requirements.txt
```

Linux/macOS:

```bash
uv venv .venv --python 3.10
uv pip install --system-certs --python .venv/bin/python -r requirements.txt
```

## Install the Local LLM Server

The local LLM server lives in the `dod-llm-server` submodule and uses `requirements_local.txt` outside Hugging Face Spaces.

Windows PowerShell:

```powershell
cd dod-llm-server
uv venv .venv --python 3.10
uv pip install --system-certs --python .\.venv\Scripts\python.exe -r requirements_local.txt
cd ..
```

Linux/macOS:

```bash
cd dod-llm-server
uv venv .venv --python 3.10
uv pip install --system-certs --python .venv/bin/python -r requirements_local.txt
cd ..
```

The local LLM server starts on port `7880`.

## Install the Local NanoVLLM TTS Server

Use a separate virtual environment for NanoVLLM/VoxCPM so its CUDA and audio dependencies do not collide with the main game environment.

Windows PowerShell:

```powershell
uv venv .venv-nanovllm --python 3.10
uv pip install --system-certs --python .\.venv-nanovllm\Scripts\python.exe -r requirements_nanovllm.txt
```

Linux/macOS:

```bash
uv venv .venv-nanovllm --python 3.10
uv pip install --system-certs --python .venv-nanovllm/bin/python -r requirements_nanovllm.txt
```

The local TTS server starts on port `8000` and exposes the Gradio API endpoint `/generate_api`.

If you prefer to run TTS on Modal instead of your local GPU, follow [Modal TTS Setup](MODAL_TTS_SETUP.md).

## Prepare Local Data

Create this data before starting the main game when both values are true:

- `DOD_USE_LOCAL_DATA=True`
- `DOD_USE_LOCAL_API=False`

In that mode, the app reads endpoint routing from the local mapper file. If `DOD_USE_LOCAL_API=True`, you can skip `inference_map.json` because the game uses `LLM_URL` and `TTS_API_URL` directly.

Create the local data directory:

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force $HOME\.dod
```

Linux/macOS:

```bash
mkdir -p ~/.dod
```

Local mapper path:

- Windows: `%USERPROFILE%\.dod\inference_map.json`
- Linux/macOS: `~/.dod/inference_map.json`

Example `inference_map.json`:

```json
{
  "llm": {
    "primary": {
      "name": "local-llm",
      "url": "http://127.0.0.1:7880",
      "mode": "gradio",
      "timeout": 30,
      "warmup_timeout": 75,
      "cooldown_seconds": 180
    },
    "fallback": {
      "name": "backup-llm",
      "url": "https://your-backup-llm.example.com",
      "mode": "gradio",
      "timeout": 30,
      "warmup_timeout": 75,
      "cooldown_seconds": 180
    }
  },
  "tts": {
    "primary": {
      "name": "local-tts",
      "url": "http://127.0.0.1:8000",
      "mode": "gradio",
      "timeout": 25,
      "warmup_timeout": 75,
      "cooldown_seconds": 180
    },
    "fallback": {
      "name": "backup-tts",
      "url": "https://your-backup-tts.example.com",
      "mode": "rest",
      "timeout": 25,
      "warmup_timeout": 75,
      "cooldown_seconds": 180
    }
  }
}
```

The `fallback` entries are optional. If you provide only `primary`, the app will run with one endpoint and no backup.

The mapper builds one endpoint chain for `llm` and another for `tts`. By default, the game tries `primary` first. If that endpoint fails or times out, it is temporarily placed on cooldown and the game tries `fallback` next. Additional fallback endpoints can be listed under `fallbacks`.

You can change which endpoint is tried first without editing the JSON:

```env
LLM_URL_PRIORITY=primary
TTS_URL_PRIORITY=primary
```

Use `fallback` when you want to test the backup endpoint first:

```env
LLM_URL_PRIORITY=fallback
TTS_URL_PRIORITY=fallback
```

These priority variables are independent, so you can test fallback TTS while keeping primary LLM, or the opposite. They only apply when the chain has more than one endpoint. If `DOD_USE_LOCAL_API=True`, the mapper is skipped and these priority variables are not used.

Local leaderboard path:

- Windows: `%USERPROFILE%\.dod\leaderboard.csv`
- Linux/macOS: `~/.dod/leaderboard.csv`

The leaderboard file is optional. If it does not exist, the app starts with an empty local leaderboard and creates the CSV when it saves results.

Example `leaderboard.csv`:

```csv
player_name,wins,losses,xp,games_played,picture_url
Nemotron,0,0,0,0,assets/nemotron.jpg
```

## Optional Remote Datasets

Use this mode when you want the inference mapper and leaderboard to live in Hugging Face Dataset repositories instead of local files.

Create two Hugging Face repositories with the **Dataset** type:

- one dataset for `inference_map.json`
- one dataset for `leaderboard.csv`

Then configure the root `.env` like this:

```env
DOD_USE_LOCAL_DATA=False
DOD_INFERENCE_MAPPER_DATASET_REPO_ID=your-user-or-org/your-inference-mapper-dataset
DOD_INFERENCE_MAPPER_DATASET_REVISION=main
DOD_LEADERBOARD_DATASET_REPO_ID=your-user-or-org/your-leaderboard-dataset
```

If either dataset is private, create an access token from your Hugging Face account settings page under **Access Tokens** and set:

```env
HF_TOKEN_DATASET=your_huggingface_dataset_token
```

The inference mapper dataset must contain `inference_map.json`. The leaderboard dataset uses `leaderboard.csv`; if it does not exist yet, the app starts with an empty leaderboard and creates it when saving results.

## Run Locally

Start each service in a separate terminal.

### Terminal 1: LLM Server

Windows PowerShell:

```powershell
cd dod-llm-server
.\.venv\Scripts\python.exe app.py
```

Linux/macOS:

```bash
cd dod-llm-server
.venv/bin/python app.py
```

Expected local URL:

```text
http://127.0.0.1:7880
```

The local LLM server binds to `0.0.0.0` so it can be tunneled when needed, but from the same machine you should access it through `http://127.0.0.1:7880`.

### Terminal 2: TTS Server

Windows PowerShell:

```powershell
.\.venv-nanovllm\Scripts\python.exe nanovllm_gradio_local.py
```

Linux/macOS:

```bash
.venv-nanovllm/bin/python nanovllm_gradio_local.py
```

Expected local URL:

```text
http://127.0.0.1:8000
```

The local TTS server also binds to `0.0.0.0` for tunneling, but the local game should call it through `http://127.0.0.1:8000`.

### Terminal 3: Main Game

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe app.py
```

Linux/macOS:

```bash
.venv/bin/python app.py
```

Open the Gradio URL printed in the terminal.

## Validation

Run a syntax check from the repository root:

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py components.py game_manager.py prompts.py inference_mapper.py
```

Linux/macOS:

```bash
.venv/bin/python -m py_compile app.py components.py game_manager.py prompts.py inference_mapper.py
```

## Notes

- The main game uses custom Gradio `gr.HTML` components. Do not replace the board with ordinary Gradio buttons or a static HTML string.
- Hugging Face OAuth works fully inside a Hugging Face Space. Locally, Gradio can mock the login if your machine is authenticated with Hugging Face.
- Use `DOD_DISABLE_TTS=True` when you want to test gameplay without waiting for audio synthesis.
- Use `DOD_USE_LOCAL_API=True` when running the LLM and TTS services on your own machine.

## Credits

Voice generation uses VoxCPM2 by OpenBMB. Local and remote LLM gameplay inference use NVIDIA Nemotron Nano 4B. Development was assisted by OpenAI Codex with GPT-5.5. Built with Gradio and Hugging Face Spaces for the Build Small Hackathon.

## Powered By

<p align="center">
  <img src="assets/gradio.png" alt="Gradio" height="42">
  &nbsp;&nbsp;
  <img src="assets/huggingface.png" alt="Hugging Face" height="42">
  &nbsp;&nbsp;
  <img src="assets/modal.png" alt="Modal" height="42">
  &nbsp;&nbsp;
  <img src="assets/nvidia.png" alt="NVIDIA" height="42">
  &nbsp;&nbsp;
  <img src="assets/openbmb.png" alt="OpenBMB" height="42">
</p>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
