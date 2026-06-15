# Configuração do TTS na Modal

Use este caminho opcional quando você não quiser rodar `nanovllm_gradio_local.py` na sua própria máquina. O app da Modal expõe o mesmo formato de API REST de TTS consumido pelo jogo.

## Configurar o App da Modal

Abra `nanovllm_app_modal.py` e revise estas constantes perto do início do arquivo:

```python
MODAL_APP_NAME = "voxcpm2-nanovllm-service"
MODAL_VOICES_VOLUME_NAME = "voxcpm-voices"
MODAL_SECRET_NAME = "voxcpm-secrets"
```

Altere esses valores se quiser usar outro nome para o app da Modal, para o volume de vozes ou para o secret.

## Instalar e Autenticar a Modal

Instale a CLI da Modal no ambiente que você usa para deploy:

```bash
pip install modal
```

Autentique sua máquina na Modal:

```bash
modal setup
```

## Criar o Secret

O app lê `TTS_API_KEY` a partir de um secret da Modal. Crie o secret usando o mesmo valor que o jogo enviará em `TTS_API_KEY`:

```bash
modal secret create voxcpm-secrets TTS_API_KEY=your_tts_api_key
```

Se você alterou `MODAL_SECRET_NAME`, use esse mesmo nome no lugar de `voxcpm-secrets`.

## Criar o Volume de Voz

Crie o volume da Modal usado para armazenar as vozes de referência:

```bash
modal volume create voxcpm-voices
```

Se você alterou `MODAL_VOICES_VOLUME_NAME`, use esse mesmo nome no lugar de `voxcpm-voices`.

## Enviar a Voz de Referência

Envie o arquivo local de voz de referência para a raiz do volume da Modal:

```bash
modal volume put voxcpm-voices ./voices/voz_1.wav voz_1.wav
```

O app espera `voice_id="voz_1.wav"`, que aponta para `/voices/voz_1.wav` dentro do container da Modal.

Você pode inspecionar o conteúdo do volume com:

```bash
modal volume ls voxcpm-voices
```

## Fazer o Deploy

Faça o deploy do app da Modal:

```bash
modal deploy nanovllm_app_modal.py
```

Depois do deploy, a Modal exibirá o endpoint público da função `generate_api`. Use essa URL como endpoint de TTS na configuração do jogo.

## Configurar o Jogo

Para uso direto no `.env` com `DOD_USE_LOCAL_API=True`:

```env
DOD_DISABLE_TTS=False
DOD_USE_LOCAL_API=True
TTS_API_URL=https://your-modal-generate-api-url
TTS_API_MODE=rest
TTS_API_KEY=your_tts_api_key
```

Para uso com o inference mapper, adicione o endpoint ao seu `inference_map.json`:

```json
{
  "tts": {
    "primary": {
      "name": "modal-tts",
      "url": "https://your-modal-generate-api-url",
      "mode": "rest"
    }
  }
}
```

Mantenha `TTS_API_KEY` no ambiente do jogo quando o secret da Modal estiver configurado, porque o jogo envia esse valor como bearer token para o endpoint da Modal.
