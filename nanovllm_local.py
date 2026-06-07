import gc
import os
import sys
import base64
import logging
import tempfile
import time
import torch
import numpy as np
import soundfile as sf
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Importa o VoxCPM oficial do motor acelerado
from nanovllm_voxcpm import VoxCPM

# --- CONFIGURAÇÃO DE LOGS ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (voxcpm-nanovllm-local) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("voxcpm-nanovllm-local")

# --- CONFIGURAÇÕES DE CAMINHOS LOCAIS ---
MODEL_PATH = "./models/vox_cpm2" 
VOICES_DIR = "./voices" # Emula o volume /voices do Modal localmente

os.makedirs(VOICES_DIR, exist_ok=True)

# Detectar se há placa de vídeo NVIDIA disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    logger.error("Erro Crítico: O motor nano-vllm exige uma placa de vídeo NVIDIA CUDA para rodar!")
    sys.exit(1)

# Inicializamos a variável global do modelo que será preenchida pelo Lifespan
model = None

# --- ARQUITETURA DE CACHE DE LATENTES (L1 - Memória RAM) ---
# Dicionário global mapeando o "voice_id" para os bytes dos latentes pré-calculados
LATENTS_MEM_CACHE = {}

# --- GERENCIADOR DE CONTEXTO LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==================== STARTUP (Inicialização) ====================
    global model
    logger.info(f"Inicializando o motor acelerado AsyncVoxCPM2ServerPool a partir de: {MODEL_PATH}...")
    try:
        model = VoxCPM.from_pretrained(
            model=MODEL_PATH,
            max_num_batched_tokens=4096,
            max_num_seqs=4,
            max_model_len=4096,
            gpu_memory_utilization=0.90, # 98% para dar folga ao Windows WDDM
            enforce_eager=False,          # Nossa alteração no model_runner forçará eager=True no Windows de forma segura
            devices=[0],
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Aguardando o servidor de GPU ficar pronto (wait_for_ready)...")
        await model.wait_for_ready()
        logger.info("Motor Nano-vLLM-VoxCPM carregado e PRONTO na GPU local!")
        
    except Exception as e:
        logger.error(f"Falha ao carregar o motor acelerado localmente: {e}")
        sys.exit(1)
        
    yield
    
    # ==================== SHUTDOWN (Encerramento) ====================
    if model:
        logger.info("Encerrando o pool de servidores distribuídos do VoxCPM...")
        await model.stop()
        logger.info("Servidores parados com sucesso.")

# --- INICIALIZAÇÃO DO FASTAPI PASSANDO O LIFESPAN ---
app = FastAPI(title="VoxCPM Nano-vLLM Local Sandbox API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINT DE GERAÇÃO ASSÍNCRONO ---
@app.post("/generate_api")
@app.post("/")
async def generate_api(request: Request):
    global model, LATENTS_MEM_CACHE
    data = await request.json()
    logger.info("================ RECEBIDA REQUISIÇÃO NANO-VLLM POST ================")
    start_time = time.time()
    
    text = data.get("text", "")
    control = data.get("control", "")
    voice_id = data.get("voice_id", None)
    ref_b64 = data.get("reference_audio_base64", None)
    
    # --- CONVERSÃO DEFENSIVA DO CFG_VALUE ---
    cfg_value = data.get("cfg_value", 2.0)
    try:
        cfg_value = float(cfg_value)
    except (ValueError, TypeError):
        cfg_value = 2.0
        
    # --- CONVERSÃO DEFENSIVA DA SEED ---
    seed = data.get("seed", None)
    if seed is not None:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            logger.warning(f"Seed fornecida inválida: {seed}. Ignorando e usando aleatório.")
            seed = None
            
    logger.info(f"Local Args - Voice ID: {voice_id}, Seed: {seed}, CFG: {cfg_value}")
    
    ref_bytes = base64.b64decode(ref_b64) if ref_b64 else None
    ref_audio_latents = None
    
    # 1. Tratar o cache e carregamento de voz persistente do disco local
    if voice_id:
        volume_path = os.path.join(VOICES_DIR, voice_id)
        latent_path = volume_path + ".latents" # Caminho do cache L2 no disco
        
        # --- VERIFICAÇÃO DO CACHE DUPLO ---
        
        # Passo A: Verificar se já está na memória RAM (Cache L1)
        if voice_id in LATENTS_MEM_CACHE:
            ref_audio_latents = LATENTS_MEM_CACHE[voice_id]
            logger.info(f"[Cache L1 - Memória] Usando latentes pré-codificados para: '{voice_id}' (0ms delay)")
            
        # Passo B: Se não estiver na RAM, verificar se já foi salvo no disco anteriormente (Cache L2)
        elif os.path.exists(latent_path):
            logger.info(f"[Cache L2 - Disco] Encontrado cache salvo para '{voice_id}'. Carregando...")
            try:
                with open(latent_path, "rb") as f:
                    ref_audio_latents = f.read()
                # Salva na memória RAM para as próximas chamadas ficarem instantâneas
                LATENTS_MEM_CACHE[voice_id] = ref_audio_latents
                logger.info(f"[Cache L1 - Mapeado] Voz '{voice_id}' carregada na RAM com sucesso.")
            except Exception as e:
                logger.warning(f"Falha ao ler o cache de disco de '{voice_id}': {e}. Recodificando...")
                ref_audio_latents = None
                
        # Passo C: Se não houver cache em lugar nenhum, executa o encoder e salva o cache nos dois níveis
        if ref_audio_latents is None:
            if os.path.exists(volume_path):
                with open(volume_path, "rb") as f:
                    wav_content = f.read()
                
                logger.info(f"Voz '{voice_id}' não possui cache. Codificando pela primeira vez (pode demorar alguns ms)...")
                ref_audio_latents = await model.encode_latents(wav_content, "wav")
                
                # Salva no disco (Cache L2 - SSD)
                try:
                    with open(latent_path, "wb") as f:
                        f.write(ref_audio_latents)
                    logger.info(f"[Cache L2 - Salvo] Cache gravado com sucesso em: {latent_path}")
                except Exception as e:
                    logger.warning(f"Falha ao salvar cache de disco para '{voice_id}': {e}")
                
                # Salva na memória RAM (Cache L1)
                LATENTS_MEM_CACHE[voice_id] = ref_audio_latents
                logger.info(f"[Cache L1 - Criado] Latentes guardados na memória RAM.")
            else:
                logger.warning(f"[Volume Local] Voz '{voice_id}' NÃO existe em {VOICES_DIR}. Ignorando clonagem.")
            
    # 2. Se o usuário enviou o áudio dinamicamente via API em Base64
    elif ref_bytes:
        logger.info("Codificando áudio recebido via Base64 para latentes em memória...")
        try:
            ref_audio_latents = await model.encode_latents(ref_bytes, "wav")
            logger.info("Áudio em Base64 codificado com sucesso em latentes.")
        except Exception as e:
            logger.error(f"Falha ao codificar áudio Base64: {e}")
            raise HTTPException(status_code=400, detail=f"Falha ao decodificar áudio de referência: {e}")
    else:
        logger.info("Modo Zero-Shot (Design de voz).")
    
    logger.info("Sintetizando áudio através do gerador assíncrono...")
    final_text = f"({control}){text}" if control else text
    try:
        gen_start = time.time()
        buf = []
        
        # Consumo do gerador assíncrono oficial passando os LATENTES REAIS codificados
        async for chunk in model.generate(
            target_text=final_text,            
            ref_audio_latents=ref_audio_latents,
            cfg_value=cfg_value,            
            seed=seed
        ):
            buf.append(chunk)
            
        if not buf:
            raise RuntimeError("Nenhum pedaço de áudio foi gerado pelo motor.")
            
        # Concatena os pedaços de áudio gerados na GPU
        wav = np.concatenate(buf, axis=0)
        logger.info(f"Síntese acelerada concluída com sucesso em {time.time() - gen_start:.2f}s.")
    except Exception as e:
        logger.error(f"Falha na geração de áudio acelerada: {e}")
        raise e

    # Converter o resultado numpy em WAV bytes em memória (48kHz - Seguro para Windows)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        sf.write(tmp_path, wav, 48000)
        os.close(tmp_fd)
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")
    
    total_time = time.time() - start_time
    logger.info(f"Requisição concluída no Sandbox Local em {total_time:.2f}s!")
    logger.info("=========================================================")

    # Retorna exatamente a mesma payload gerada pelo Modal
    return {
        "wav_base64": encoded_audio,
        "sample_rate": 48000,
        "seed_used": seed,
        "cfg_value_used": cfg_value
    }


if __name__ == "__main__":
    # Sobe o servidor local na porta 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)