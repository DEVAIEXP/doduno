# DOD UNO

DOD UNO is a software-engineering themed UNO-style game built with Gradio custom HTML components, an external LLM decision service, and an optional local NanoVLLM/VoxCPM TTS service.

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
DOD_USE_LOCAL_API=True

TTS_API_URL=http://127.0.0.1:8000
TTS_API_MODE=gradio
LLM_URL=http://127.0.0.1:7880

TTS_API_KEY=your_local_tts_key
LLM_API_KEY=your_local_llm_key
# Optional for private remote datasets only:
# HF_TOKEN_DATASET=your_huggingface_dataset_token
```

For local-only development, `TTS_API_KEY` and `LLM_API_KEY` are mainly pass-through values used by the game when calling the local APIs. They can be any value as long as the game and the local service agree on the same value. Treat them as real secrets only when the service is exposed remotely, through a public tunnel, or deployed outside your machine.

Set `DOD_DISABLE_TTS=True` if you want faster development runs without calling the TTS service. Director lines will still appear as text in the match log, but they will not be audible.

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

## Local Data Examples

Create the local data directory:

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force $HOME\.dod
```

Linux/macOS:

```bash
mkdir -p ~/.dod
```

Example local mapper path:

- Windows: `%USERPROFILE%\.dod\inference_map.json`
- Linux/macOS: `~/.dod/inference_map.json`

Example `inference_map.json`:

```json
{
  "llm": {
    "primary": {
      "name": "local-llm",
      "url": "http://127.0.0.1:7880",
      "mode": "gradio"
    },
    "fallback": {
      "name": "backup-llm",
      "url": "https://your-backup-llm.example.com",
      "mode": "gradio"
    }
  },
  "tts": {
    "primary": {
      "name": "local-tts",
      "url": "http://127.0.0.1:8000",
      "mode": "gradio"
    },
    "fallback": {
      "name": "backup-tts",
      "url": "https://your-backup-tts.example.com",
      "mode": "rest"
    }
  }
}
```

Example local leaderboard path:

- Windows: `%USERPROFILE%\.dod\leaderboard.csv`
- Linux/macOS: `~/.dod/leaderboard.csv`

Example `leaderboard.csv`:

```csv
player_name,wins,losses,xp,games_played,picture_url
Nemotron,0,0,0,0,assets/nemotron.jpg
```

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
