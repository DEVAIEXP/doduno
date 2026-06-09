from __future__ import annotations

import json
import os
import queue
import random
import threading
import time
from typing import Any

import requests
from dotenv import load_dotenv
from gradio_client import Client
from huggingface_hub import hf_hub_download, upload_file

from inference_mapper import EndpointConfig, get_endpoint_chain, mark_endpoint_failed, mark_endpoint_success

load_dotenv(override=True)

Card = dict[str, Any]
GameState = dict[str, Any]
ServerResponse = dict[str, Any]

# Maximum seconds an active player has to act during a normal turn.
PLAYER_TURN_TIME_LIMIT_SECONDS = 30
# Server-side grace window for the required Deploy shout.
SERVER_SHOUT_WINDOW_BUFFER_SECONDS = 6
# Seconds to show end-game state before rotating/restarting the room.
GAME_RESTART_COUNTDOWN_SECONDS = 10
# Minimum active seats required before the lobby can start a match countdown.
MIN_PLAYERS_TO_START = int(os.getenv("DOD_MIN_PLAYERS_TO_START", "2"))
# Seconds to wait for more players once the minimum active seats are present.
LOBBY_START_COUNTDOWN_SECONDS = int(os.getenv("DOD_LOBBY_START_COUNTDOWN_SECONDS", "30"))
# Seconds without heartbeat before a player is considered inactive.
PLAYER_HEARTBEAT_KICK_LIMIT_SECONDS = 45.0
# Seconds without any table action before the room is force-closed.
MAX_ROOM_INACTIVITY_TIMEOUT_SECONDS = 180.0
# Seconds during which duplicate active names are rejected.
DUPLICATE_CHECK_WINDOW_SECONDS = 5.0
# Sync lobby during warmup
TICK_LOBBY_WARMUP_SECONDS = 1
# Backend tick interval for turn timers and room cleanup.
TICK_RATE_SERVER_SECONDS = 1
# Player board polling interval.
SYNC_RATE_PLAYER_SECONDS = 1
# Spectator board polling interval.
SYNC_RATE_SPECTATOR_SECONDS = 2
# Leaderboard polling interval.
SYNC_RATE_LEADERBOARD_SECONDS = 15
# Maximum active players supported by the room.
MAX_PLAYERS = 3

# Built-in AI opponent name.
BOT_NAME = "Nemotron"
# Huggingface Authentication Token
HF_TOKEN=os.getenv("HF_TOKEN", "")
# Nemotron LLM server URL.
LLM_URL = os.getenv("LLM_URL", "https://elismasilva-voxcpm2-nanovllm-service.hf.space")
# Nemotron LLM server API key.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# Hugging Face dataset that stores leaderboard persistence.
LEADERBOARD_DATASET_REPO_ID = os.getenv("DOD_LEADERBOARD_DATASET_REPO_ID", "elismasilva/dod-leaderboard")
# CSV path inside the leaderboard dataset repository.
LEADERBOARD_DATASET_PATH = os.getenv("DOD_LEADERBOARD_DATASET_PATH", "leaderboard.csv")
# External TTS endpoint used for director voice audio.
TTS_API_URL = os.getenv("TTS_API_URL", "http://127.0.0.1:8000/generate_api")
# Voice-control prompts for the TTS service.
TTS_CONTROLS = {
    "pt": "Brazilian male broadcaster, speaking with a native Brazilian Portuguese accent. Deep but crisp voice. He sounds like an experienced, highly professional manager. 'No-nonsense', authoritative, and strategic tone. Clear studio recording.",
    "en": "American male executive, speaking with a standard US English accent. Deep but crisp voice. He sounds like an experienced, highly professional manager. 'No-nonsense', authoritative, and strategic tone. Clear studio recording."
}
# Reference voice id expected by the TTS service.
TTS_VOICE_ID = "voz_1.wav"
TTS_VOICE_SEED = 44

# Shared FIFO channel used by the app-level LLM worker.
llm_queue: queue.Queue[dict[str, Any]] = queue.Queue()
tts_gradio_clients: dict[str, Client] = {}


def get_tts_gradio_client(endpoint: EndpointConfig, timeout_override: float | None = None) -> Client:
    """Return a cached Gradio client for a TTS endpoint."""
    url = endpoint["url"]
    timeout = float(timeout_override if timeout_override is not None else endpoint.get("timeout", 120.0))
    cache_key = f"{url}|{timeout}"
    if cache_key not in tts_gradio_clients:
        print(f"[TTS] Connecting Gradio client to {endpoint.get('name', 'endpoint')}: {url}", flush=True)
        tts_gradio_clients[cache_key] = Client(url, token=HF_TOKEN or None, httpx_kwargs={"timeout": timeout})
    return tts_gradio_clients[cache_key]

APP_UI = {
    "en": {
        "title": "DOD: Deploy or Draw! UNO GAME 🚀",
        "subtitle": "Select language, enter your name and join the queue.",
        "lang_label": "Language",
        "name_label": "Your Name",
        "btn_join": "Join Game",
        "btn_leave_queue": "Leave Queue",
        "hf_login_button": "Sign in with Hugging Face",
        "hf_logout_button": "Logout ({})",
        "hf_login_guest": "Optional: sign in with Hugging Face or enter a name below.",
        "hf_login_authenticated": "Signed in with Hugging Face as **{name}**. This identity will be used for matches and leaderboard.",
        "status": "Waiting for action...",
        "tab_lobby": "🎮 Lobby & Spectator",
        "tab_player": "💻 Your Game",
        "welcome_play": "Welcome, {name}! You are in the game.",
        "welcome_queue": "Room is full. You are #{pos} in the queue.",
        "tab_leaderboard": "🏆 Leaderboard",
        "invalid_name": "⚠️ Enter a valid name!",
        "duplicate_name": "⚠️ This name is already active in another tab!",
        "warmup_status": "DOD UNO: Cooking cloud audio assets... Please wait about 30-50 seconds!"
    },
    "pt": {
        "title": "DOD: Deploy or Draw! JOGO UNO 🚀",
        "subtitle": "Selecione o idioma, digite seu nome e entre na fila.",
        "lang_label": "Idioma",
        "name_label": "Seu Nome",
        "btn_join": "Entrar na Partida",
        "btn_leave_queue": "Sair da Fila",
        "hf_login_button": "Entrar com Hugging Face",
        "hf_logout_button": "Sair ({})",
        "hf_login_guest": "Opcional: entre com Hugging Face ou digite um nome abaixo.",
        "hf_login_authenticated": "Conectado ao Hugging Face como **{name}**. Esta identidade será usada nas partidas e na classificação.",
        "status": "Aguardando ação...",
        "tab_lobby": "🎮 Lobby & Espectador",
        "tab_player": "💻 Sua Partida",
        "welcome_play": "Bem-vindo(a), {name}! Você está no jogo.",
        "welcome_queue": "Partida cheia. Você é o #{pos} na fila.",
        "tab_leaderboard": "🏆 Classificação",
        "invalid_name": "⚠️ Digite um nome válido!",
        "duplicate_name": "⚠️ Este nome já está ativo em outra aba!",
        "warmup_status": "DOD UNO: Cozinhando os assets de áudio na nuvem... Aguarde cerca de 30-50 segundos!"
    }
}

UI_I18N = {
    "en": {
        "waiting": "Waiting for {num} players to start...", "status": "SERVER STATUS:", "res": "Crisis Resolution", "panic": "IT Director Panic",
        "pick": "CHOOSE THE NEXT STACK (COLOR)", "draw": "DRAW PILE", "draw_btn": "DRAW", "table": "CARD ON TABLE",
        "pass": "PASS TURN", "shout": "SHOUT DEPLOY!", "log": "📜 SERVER LOG", "you": "(YOU)", "leave": "LEAVE MATCH",
        "deploy_saved": "📢 DEPLOY SAVED!", "risk": "⚠️ RISK!", "accuse": "🚨 ACCUSE!", "player": "Player",
        "director": "Director", "queue_msg": "⏳ You are #{pos} in the queue.", "restarting": "🔄 Next match starting in {sec}s...",
        "toast_not_turn": "⛔ Not your turn!", "toast_wait_shout": "⚠️ 1 card left! Click SHOUT DEPLOY!",
        "toast_wait_color": "⚠️ Choose the Stack before continuing.", "toast_invalid": "❌ Invalid Move! Card doesn't match.",
        "toast_game_over": "💀 Director had a heart attack! GAME OVER.", "toast_victory": "🏆 VICTORY! Deploy completed successfully!",
        "toast_pick_shout": "🎨 Choose Stack. Then you can shout Deploy!", "toast_alert_deploy": "🚨 DEPLOY ALERT!",
        "toast_cant_accuse_self": "⛔ You cannot accuse yourself!", "toast_wait_accuse": "⏳ Player still has a chance to shout Deploy.",
        "toast_accuse_success": "🚨 You successfully accused {name}!", "toast_accuse_invalid": "❌ Invalid Accusation!",
        "toast_shout_protected": "📢 DEPLOY! You are protected from accusations.", "toast_shout_invalid": "⚠️ You can only shout DEPLOY! with 1 card left.",
        "toast_left_queue": "🚪 You left the queue.", "toast_left_game": "🚪 You abandoned the match.",
        "toast_game_over_abandon": "💀 Match ended due to lack of players.",
        "toast_game_not_started": "⚠️ The match has not started yet!",
        "lb_title": "🏆 PLAYERS LEADERBOARD",
        "start_countdown": "Starting with current players in {sec}s...",
        "warmup_status": "DOD UNO: Cooking cloud audio assets... Please wait about 30-50 seconds!",
        "lb_rank": "Rank",
        "lb_developer": "Player",
        "lb_wins": "Wins",
        "lb_games": "Games",
        "lb_xp": "Career XP",
        "lb_role": "Title",
        "role_intern": "Intern 👶",
        "role_junior": "Junior 🩹",
        "role_mid": "Mid-Level 🔨",
        "role_senior": "Senior 🚀",
        "role_lead": "Tech Lead 👑"
    },
    "pt": {
        "waiting": "Aguardando {num} jogadores para iniciar...", "status": "STATUS DO SERVIDOR:", "res": "Resolução da Crise", "panic": "Pânico do Diretor de TI",
        "pick": "ESCOLHA A PRÓXIMA STACK (COR)", "draw": "MONTE", "draw_btn": "COMPRAR", "table": "CARTA NA MESA",
        "pass": "PASSAR VEZ", "shout": "GRITAR DEPLOY!", "log": "📜 LOG DO SERVIDOR", "you": "(VOCÊ)", "leave": "SAIR DO JOGO",
        "deploy_saved": "📢 DEPLOY SALVO!", "risk": "⚠️ RISCO!", "accuse": "🚨 ACUSAR!", "player": "Jogador",
        "director": "Diretor", "queue_msg": "⏳ Você é o #{pos} na fila de espera.", "restarting": "🔄 Nova partida iniciando em {sec}s...",
        "toast_not_turn": "⛔ Não é a sua vez!", "toast_wait_shout": "⚠️ Você ficou com 1 carta! Confirme clicando em GRITAR DEPLOY!",
        "toast_wait_color": "⚠️ Escolha a Stack antes de continuar.", "toast_invalid": "❌ Jogada Inválida! A carta não combina na mesa.",
        "toast_game_over": "💀 O Diretor infartou! FIM DE JOGO.", "toast_victory": "🏆 VITÓRIA! Deploy concluído com sucesso!",
        "toast_pick_shout": "🎨 Escolha a Stack. Depois você poderá gritar Deploy!", "toast_alert_deploy": "🚨 ALERTA DE DEPLOY!",
        "toast_cant_accuse_self": "⛔ Você não pode acusar a si mesmo!", "toast_wait_accuse": "⏳ O jogador ainda tem chance de gritar Deploy.",
        "toast_accuse_success": "🚨 Você acusou {name} com sucesso!", "toast_accuse_invalid": "❌ Acusação inválida!",
        "toast_shout_protected": "📢 DEPLOY! Você está protegido contra acusações.", "toast_shout_invalid": "⚠️ Você só pode gritar DEPLOY! se tiver 1 carta na mão.",
        "toast_left_queue": "🚪 Você saiu da fila.", "toast_left_game": "🚪 Você abandonou a partida.",
        "toast_game_over_abandon": "💀 Partida encerrada por falta de jogadores.",
        "toast_game_not_started": "⚠️ A partida ainda não começou!",
        "lb_title": "🏆 CLASSIFICAÇÃO DOS DEVS",
        "start_countdown": "Iniciando com os jogadores atuais em {sec}s...",
        "warmup_status": "DOD UNO: Cozinhando os assets de áudio na nuvem... Aguarde cerca de 30-50 segundos!",
        "lb_rank": "Rank",
        "lb_developer": "Jogador",
        "lb_wins": "Vitórias",
        "lb_games": "Partidas",
        "lb_xp": "XP de Carreira",
        "lb_role": "Cargo",
        "role_intern": "Estagiário 👶",
        "role_junior": "Júnior 🩹",
        "role_mid": "Pleno 🔨",
        "role_senior": "Sênior 🚀",
        "role_lead": "Tech Lead 👑"
    }
}

LOG_I18N = {
    "en": {
        "lobby_wait": "[System]: Waiting for {num} players in the Lobby...", "connected": "🔌 {name} joined the server.",
        "queued": "⏳ {name} joined the waiting list.",
        "new_game": "✨ [New Game]: {crisis}! Solve the crisis!", "draw": "📥 [{name}] drew a card.",
        "play": "✅ [{name}] played '{card}': {feedback} (Res: {res}%, Pan: {pan}%){quote}", "win": "🏆 [VICTORY]: {name} executed the FINAL DEPLOY!",
        "empty_punish": "⚠️ [{name}] ran out of cards, but crisis continues. +2 tasks penalty!", "skip": "🛑 [Block]: {name} lost their turn in a call!",
        "reverse": "🔄 [Revert]: Game direction reversed!", "attack": "⚔️ [Attack]: {name} received +2 tasks and lost their turn!",
        "wild_pick": "🎨 [Wild]: Stack changed to {color}.", "wild_attack": "🔥 [Merge Conflict]: {name} got +4 conflicts and lost their turn!",
        "shout_alert": "⚠️ [{name}] has 1 card left and didn't shout Deploy!", "shout_success": "📢 [{name}] shouted DEPLOY!",
        "shout_fail": "🤐 [{name}] failed to shout Deploy. Vulnerable to accusations!", "accuse_success": "🚨 [ACCUSATION]: {name} didn't shout Deploy! Penalty: +2 cards.",
        "pass_turn": "⏭️ [{name}] ended their turn.", "game_over": "💀 [Game Over]: Panic reached 100%. Team fired.",
        "left_lobby": "🚪 [{name}] left the lobby.", "left_game": "🚪 [{name}] abandoned the match!", "game_over_abandon": "💀 [Game Over]: Match ended due to lack of players (W.O.).",
        "game_over_timeout": "💀 [Game Over]: Match expired due to 3 minutes of inactivity.",
        "afk_skip": "⏳ {name} is AFK. Passing turn...",
        "turn_timeout": "⏳ {name} took too long! Drew a card and lost their turn."
    },
    "pt": {
        "lobby_wait": "[Sistema]: Aguardando {num} jogadores no Lobby...", "connected": "🔌 {name} conectou ao servidor.",
        "queued": "⏳ {name} entrou na fila de espera.",
        "new_game": "✨ [Novo Jogo]: {crisis}! Resolvam a crise!", "draw": "📥 [{name}] comprou uma carta.",
        "play": "✅ [{name}] jogou '{card}': {feedback} (Res: {res}%, Pan: {pan}%){quote}", "win": "🏆 [VITÓRIA]: {name} realizou o DEPLOY FINAL!",
        "empty_punish": "⚠️ [{name}] ficou sem cartas, mas a crise continua. +2 tarefas de punição!", "skip": "🛑 [Bloqueio]: {name} perdeu a vez na call!",
        "reverse": "🔄 [Revert]: A ordem do jogo foi invertida!", "attack": "⚔️ [Ataque]: {name} recebeu +2 tarefas e perdeu a vez!",
        "wild_pick": "🎨 [Coringa]: Stack alterada para {color}.", "wild_attack": "🔥 [Merge Conflict]: {name} recebeu +4 conflitos e perdeu a vez!",
        "shout_alert": "⚠️ [{name}] ainda tem 1 carta e não gritou Deploy!", "shout_success": "📢 [{name}] gritou DEPLOY!",
        "shout_fail": "🤐 [{name}] não gritou Deploy a tempo. Alvo vulnerável!", "accuse_success": "🚨 [ACUSAÇÃO]: {name} não gritou Deploy! Punição: +2 cartas.",
        "pass_turn": "⏭️ [{name}] encerrou o turno.", "game_over": "💀 [Fim de Jogo]: Pânico atingiu 100%. Equipe demitida.",
        "left_lobby": "🚪 [{name}] saiu do lobby.", "left_game": "🚪 [{name}] abandonou a partida!", "game_over_abandon": "💀 [Fim de Jogo]: Partida encerrada por falta de jogadores (W.O.).",
        "game_over_timeout": "💀 [Fim de Jogo]: Partida expirada por 3 minutos de inatividade.",
        "afk_skip": "⏳ {name} ficou ausente por muito tempo. Passando o turno...",
        "turn_timeout": "⏳ {name} demorou muito! Comprou uma carta e perdeu a vez."
    }
}


CRISES_DATABASE = [
    {
        "title": {"en": "CRITICAL INCIDENT: DB CORRUPTED", "pt": "CRITICAL INCIDENT: DB CORROMPIDO"},
        "desc": {"en": "MySQL is down. Front makes warning screen, Back cries, DevOps tries backup.", "pt": "O MySQL caiu de madrugada. Front faz tela de aviso, Back chora, DevOps tenta backup."},
        "quotes": {
            "good": [{"en": "Phew! DB is coming back!", "pt": "Ufa!! O banco tá voltando!"}, {"en": "Good job! Saved my job.", "pt": "Boa, garoto! Salvaram meu emprego."}, {"en": "Amen! A decent query!", "pt": "Amém! Finalmente uma query decente!"}],
            "bad": [{"en": "Holy shit! Did you DROP the wrong table?!", "pt": "Puta que o pariu! Vocês deram DROP na tabela errada?!"}, {"en": "CEO is calling, fix it now!", "pt": "Caralho! O CEO tá me ligando, volta esse banco logo!"}, {"en": "Are you crazy? Clients disappeared!", "pt": "Vocês são loucos? Meus clientes sumiram do sistema!"}],
            "mixed": [{"en": "Bro... this workaround will explode tomorrow, but works today.", "pt": "Mano... essa gambiarra no banco vai explodir amanhã, mas serve por hoje."}, {"en": "Indexed wrong, CPU at 99%, but it loaded. Phew.", "pt": "Indexou tudo errado e a CPU tá em 99%, mas a tela carregou. Ufa."}]
        }
    },
    {
        "title": {"en": "DDoS ON HUGGING FACE HUB", "pt": "DDoS NO HUGGING FACE HUB"},
        "desc": {"en": "Insane traffic! Models frozen, APIs bursting request limits.", "pt": "Tráfego insano! Modelos travados, APIs estourando limite de requests."},
        "quotes": {
            "good": [{"en": "Phew! Rate limit saved the server.", "pt": "Ufa! O rate limit salvou o servidor."}, {"en": "Yes! Scale the pods, fast!", "pt": "Isso! Escalem os pods, rápido!"}, {"en": "Traffic is normalizing!", "pt": "O tráfego tá normalizando, continuem assim!"}],
            "bad": [{"en": "Are you crazy?! AWS will charge a fortune for bandwidth!", "pt": "Vocês são loucos?! A AWS vai cobrar uma fortuna de banda!"}, {"en": "Shit, the cluster died again!", "pt": "Puta merda, o cluster foi pro saco de novo!"}, {"en": "Damn, bots took down the entire home page!", "pt": "Caralho, os bots derrubaram a home page inteira!"}],
            "mixed": [{"en": "Did you cache the frontend poorly? At least it didn't crash.", "pt": "Fizeram cache no front-end de qualquer jeito? Tá, pelo menos não cai o servidor."}, {"en": "Queue is huge, but no more 502 Errors. Acceptable.", "pt": "A fila tá gigante, mas parou de dar Erro 502. Aceitável."}]
        }
    },
    {
        "title": {"en": "AWS KEYS LEAKED", "pt": "VAZAMENTO DE CHAVES AWS"},
        "desc": {"en": "Someone committed the .env to a public repo. Bots mining crypto on our account!", "pt": "Alguém commitou a .env no repositório público. Bots estão minerando cripto na nossa conta!"},
        "quotes": {
            "good": [{"en": "Amen! Key revoked. Cold sweat here.", "pt": "Amém! Chave revogada. Suor frio aqui."}, {"en": "Great! Block billing before it hits $1M.", "pt": "Ótimo! Bloqueiam o billing antes de bater 1 milhão de dólares."}, {"en": "Phew! AWS account is safe for now.", "pt": "Ufa! A conta da AWS tá salva por enquanto."}],
            "bad": [{"en": "Holy crap, did they spin up 50 GPU instances in Asia?!", "pt": "Puta que pariu, já subiram 50 instâncias de GPU na Ásia?!"}, {"en": "Damn, which intern pushed this?!", "pt": "Caralho, quem foi o estagiário jumento que deu git push nisso?!"}, {"en": "We're bankrupt! Shut this all down now!", "pt": "Tamo falido! Desliga essa porra toda agora!"}],
            "mixed": [{"en": "Deleted the whole repo to hide the leak? My God...", "pt": "Deletaram o repositório inteiro pra esconder o vazamento? Meu Deus..."}, {"en": "Account is locked, but at least they stopped mining bitcoin.", "pt": "A conta tá travada por segurança, mas pelo menos pararam de minerar bitcoin."}]
        }
    },
    {
        "title": {"en": "A.I. HALLUCINATING IN PROD", "pt": "I.A. ALUCINANDO NO PROD"},
        "desc": {"en": "Support chatbot is swearing at clients. Turn it off now!", "pt": "O chatbot de suporte está xingando os clientes. Desliguem isso agora!"},
        "quotes": {
            "good": [{"en": "Phew! Bot stopped swearing.", "pt": "Ufa! O bot parou de falar palavrão."}, {"en": "Adjusted the temperature? Finally sane answers.", "pt": "Ajustaram a temperatura? Finalmente respostas sãs."}, {"en": "Thank God they limited the model's prompt.", "pt": "Graças a Deus limitaram o prompt do modelo."}],
            "bad": [{"en": "Shit, the bot just insulted our biggest investor!", "pt": "Puta merda, o bot acabou de ofender o nosso maior investidor!"}, {"en": "Are you crazy? Who left this unfiltered model in prod?!", "pt": "Vocês são loucos? Quem deixou esse modelo sem filtro em produção?!"}, {"en": "Damn, AI is promising free products in chat!", "pt": "Caralho, a IA tá prometendo produto de graça no chat!"}],
            "mixed": [{"en": "Bot is answering in Latin now, but at least no insults.", "pt": "O bot tá dando respostas em latim agora, mas pelo menos não tá ofendendo ninguém."}, {"en": "Turned off AI and used a 1990s if/else? F*ck it, it works.", "pt": "Desligaram a IA e botaram um if/else de 1990? Foda-se, tá funcionando."}]
        }
    }
]

def generate_full_deck() -> list[Card]:
    """Build and shuffle-ready the complete 108-card DOD UNO deck.

    Returns:
        List of card dictionaries containing stack, category, effects, and i18n text.
    """
    deck = []
    card_id = 1

    stacks = [
        {"color": "green", "name_en": "Frontend", "name_pt": "Frontend"},
        {"color": "blue", "name_en": "Backend", "name_pt": "Backend"},
        {"color": "red", "name_en": "DevOps", "name_pt": "DevOps"},
        {"color": "yellow", "name_en": "A.I.", "name_pt": "I.A."}
    ]

    for stack in stacks:
        c = stack["color"]


        deck.append({
            "id": card_id, "stack": c, "category": "SUPER", "categorySymbol": "🚀", "badge": "⭐",
            "res": 25, "panic": -10,
            "name": {"en": f"Deploy Friday 6PM ({stack['name_en']})", "pt": f"Deploy Sexta 18h ({stack['name_pt']})"},
            "feedback": {"en": "Worked on first try! A miracle!", "pt": "Funcionou de primeira! Um milagre!"}
        })
        card_id += 1


        for _ in range(2):

            deck.append({"id": card_id, "stack": c, "category": "FIX", "categorySymbol": "🩹", "res": 15, "panic": -5, "name": {"en": "Fix Nasty Bug", "pt": "Corrigir Bug"}, "feedback": {"en": "Bug squashed successfully.", "pt": "Bug esmagado com sucesso."}}); card_id+=1
            deck.append({"id": card_id, "stack": c, "category": "REFACTOR", "categorySymbol": "🧹", "res": 10, "panic": -5, "name": {"en": "Refactor Code", "pt": "Refatorar Código"}, "feedback": {"en": "Clean and readable now.", "pt": "Código limpo e legível."}}); card_id+=1
            deck.append({"id": card_id, "stack": c, "category": "TECH_DEBT", "categorySymbol": "💸", "res": 10, "panic": -10, "name": {"en": "Clear Tech Debt", "pt": "Pagar Dívida Técnica"}, "feedback": {"en": "System is breathing better.", "pt": "O sistema está respirando melhor."}}); card_id+=1
            deck.append({"id": card_id, "stack": c, "category": "DOCS", "categorySymbol": "📝", "res": 5, "panic": -15, "name": {"en": "Update Docs", "pt": "Atualizar Docs"}, "feedback": {"en": "Director sighed in relief.", "pt": "Diretor respirou aliviado."}}); card_id+=1


            deck.append({"id": card_id, "stack": c, "category": "PATCH", "categorySymbol": "🔨", "res": 15, "panic": 10, "name": {"en": "Quick Workaround", "pt": "Gambiarra Rápida"}, "feedback": {"en": "It works... don't look at it.", "pt": "Funciona... só não olhe o código."}}); card_id+=1
            deck.append({"id": card_id, "stack": c, "category": "STACK_OVERFLOW", "categorySymbol": "📋", "res": 10, "panic": 5, "name": {"en": "StackOverflow Copy", "pt": "Copiar StackOverflow"}, "feedback": {"en": "No idea why it works, but it does.", "pt": "Não sei porque funciona, mas funciona."}}); card_id+=1


            deck.append({"id": card_id, "stack": c, "category": "SPAGHETTI", "categorySymbol": "🍝", "res": 5, "panic": 10, "name": {"en": "Spaghetti Code", "pt": "Código Espaguete"}, "feedback": {"en": "Nobody understands this architecture.", "pt": "Ninguém entende essa arquitetura."}}); card_id+=1
            deck.append({"id": card_id, "stack": c, "category": "BLIND_PR", "categorySymbol": "🙈", "res": 5, "panic": 15, "name": {"en": "Blind PR Merge", "pt": "Merge sem Review"}, "feedback": {"en": "Merged directly to main! Risky!", "pt": "Merge direto na main! Arriscado!"}}); card_id+=1
            deck.append({"id": card_id, "stack": c, "category": "BUG", "categorySymbol": "🐛", "res": -10, "panic": 15, "name": {"en": "Code Regression", "pt": "Regressão no Código"}, "feedback": {"en": "Broke what was already working!", "pt": "Quebrou o que já estava funcionando!"}}); card_id+=1


            deck.append({"id": card_id, "stack": c, "category": "SKIP", "categorySymbol": "🛑", "badge": "⭐", "res": 0, "panic": 0, "skip": True, "name": {"en": "Surprise Meeting", "pt": "Reunião Surpresa"}, "feedback": {"en": "Next dev pulled into a call, lost turn!", "pt": "Próximo perdeu a vez na call!"}}); card_id+=1
            deck.append({"id": card_id, "stack": c, "category": "REVERSE", "categorySymbol": "🔄", "badge": "⭐", "res": 5, "panic": 0, "reverse": True, "name": {"en": "Git Revert", "pt": "Git Revert"}, "feedback": {"en": "Project flow inverted!", "pt": "Fluxo invertido!"}}); card_id+=1
            deck.append({"id": card_id, "stack": c, "category": "ATTACK", "categorySymbol": "➕", "badge": "+2", "res": 5, "panic": 5, "drawTwo": True, "name": {"en": "Code Review Hell", "pt": "Code Review Chato"}, "feedback": {"en": "Next dev redoes everything! (+2)", "pt": "Próximo dev refaz tudo! (+2)"}}); card_id+=1


    for _ in range(4):
        deck.append({"id": card_id, "stack": "wild", "category": "WILD", "categorySymbol": "🎨", "badge": "🎨", "res": 10, "panic": -10, "name": {"en": "Senior Agile Consultant", "pt": "Consultor Ágil Sênior"}, "feedback": {"en": "Sprint reorganized! Choose the Stack.", "pt": "Sprint reorganizada! Escolha a Stack."}}); card_id+=1
        deck.append({"id": card_id, "stack": "wild", "category": "NUKE", "categorySymbol": "☢️", "badge": "+4", "res": -10, "panic": 30, "drawFour": True, "name": {"en": "Drop Prod Database", "pt": "Apagar Banco Prod"}, "feedback": {"en": "Total chaos! Next dev suffers! (+4)", "pt": "Caos total! Próximo dev sofre! (+4)"}}); card_id+=1

    return deck


class GameManager:
    """Thread-shared game state machine for lobby, match, bot, audio, and leaderboard state."""

    def __init__(self) -> None:
        """Initialize lobby, timers, caches, and persisted leaderboard state."""
        self.players = []
        self.queue = []
        self.player_langs = {}
        self.last_seen = {}
        self.game_started = False
        self.game_end_reason = ""
        self.restart_countdown = 0
        self.lobby_start_countdown = -1
        self.turn_start_shout_shown = False
        self.is_turn_start_shout = False
        self.direction = 1
        self.current_crisis_idx = 0
        self.last_move_time = 0.0

        self.resolution = 0
        self.panic = 20
        self.active_card = None
        self.active_player = 0


        self.hands = {}
        self.has_shouted_deploy = {}

        self.draw_pile = []
        self.shout_countdown = SERVER_SHOUT_WINDOW_BUFFER_SECONDS
        self.turn_time_left = PLAYER_TURN_TIME_LIMIT_SECONDS
        self.has_drawn_this_turn = False
        self.is_picking_color = False
        self.wild_draw_four_pending = False
        self.waiting_for_shout = False
        self.pending_skip_on_shout = False
        self._last_tick_time = 0.0
        self._last_turn_tick = 0.0
        self.events = [{"key": "lobby_wait", "kwargs": {"num": MAX_PLAYERS}}]
        self.audio_cache = {}
        self.latest_director_audio = {"id": 0, "b64": ""}
        self.leaderboard_cache = {}
        self.match_stats = {}
        self.pending_audios = {}
        self.repo_id = LEADERBOARD_DATASET_REPO_ID
        self.leaderboard_path = LEADERBOARD_DATASET_PATH
        self.load_leaderboard_from_hf()
        self.modal_is_warm = False
        self.modal_is_warming_up = False

    def init_game(self) -> None:
        """Start a fresh match for the current active players."""
        self.game_started = True
        self.game_end_reason = ""
        self.restart_countdown = 0
        self.lobby_start_countdown = -1
        self.direction = 1
        self.current_crisis_idx = random.randint(0, len(CRISES_DATABASE)-1)
        self.resolution = 0
        self.panic = 20
        self.active_player = 0
        self.has_drawn_this_turn = False
        self.is_picking_color = False
        self.wild_draw_four_pending = False
        self.pending_skip_on_shout = False
        self.last_move_time = time.time()
        self.turn_time_left = PLAYER_TURN_TIME_LIMIT_SECONDS

        self.hands = {}
        self.has_shouted_deploy = {}
        self.discard_pile = []
        self.events = []
        for p_name in self.players:
            self.log_event("connected", name=p_name)
        self.shout_countdown = 0
        self.waiting_for_shout = False

        crisis_data = CRISES_DATABASE[self.current_crisis_idx]
        self.log_event("new_game", crisis=crisis_data["title"])

        self.draw_pile = generate_full_deck()
        random.shuffle(self.draw_pile)

        self.match_stats = {}
        for p_name in self.players:
            self.match_stats[p_name] = {"res_contrib": 0, "panic_mitigation": 0}
            self.hands[p_name] = self.get_unique_starting_hand(p_name, 7)
            self.sort_hand(p_name)
            self.has_shouted_deploy[p_name] = False
            self.last_seen[p_name] = time.time()


        starting_card = None
        for idx, card in enumerate(self.draw_pile):
            if card["stack"] != "wild":
                starting_card = self.draw_pile.pop(idx)
                break
        if not starting_card:
            starting_card = self.draw_pile.pop(0)
        self.active_card = starting_card


        self.trigger_bot_if_active()

    def load_leaderboard_from_hf(self) -> None:
        """Load persisted leaderboard rows from the configured Hugging Face dataset."""
        try:

            filepath = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.leaderboard_path,
                repo_type="dataset",
                token=os.getenv("HF_TOKEN")
            )
            import csv
            with open(filepath, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.leaderboard_cache[row["player_name"]] = {
                        "wins": int(row["wins"]),
                        "losses": int(row["losses"]),
                        "xp": int(row["xp"]),
                        "games": int(row["games_played"])
                    }
        except Exception:

            self.leaderboard_cache = {}

    def log_event(self, key: str, **kwargs: Any) -> None:
        """Append a localized log event descriptor.

        Args:
            key: Event identifier resolved through `LOG_I18N`.
            **kwargs: Event formatting values.
        """
        self.events.append({"key": key, "kwargs": kwargs})

    def touch_presence(self, viewer_id: str) -> None:
        """Refresh the heartbeat for a player or queued user.

        Args:
            viewer_id: Player or queued user name associated with a browser tab.
        """
        if viewer_id and (viewer_id in self.players or viewer_id in self.queue):
            self.last_seen[viewer_id] = time.time()

    def join_lobby(self, name: str, lang_code: str) -> str:
        """Join a player to the active room or queue.

        Args:
            name: User-facing player name.
            lang_code: Language code, usually `en` or `pt`.

        Returns:
            Accepted canonical player name, or `DUPLICATE_REJECT`.
        """
        name_clean = name.strip()
        name_lower = name_clean.lower()

        if name_lower == BOT_NAME.lower():
            return "DUPLICATE_REJECT"

        self.pending_audios[name_clean] = []
        existing_players_lower = [p.lower() for p in self.players]
        existing_queue_lower = [q.lower() for q in self.queue]

        if name_lower in existing_players_lower or name_lower in existing_queue_lower:
            matched_name = next((p for p in self.players if p.lower() == name_lower), None)
            if not matched_name:
                matched_name = next((q for q in self.queue if q.lower() == name_lower), None)

            now = time.time()

            if self.game_started and now - self.last_seen.get(matched_name, 0) < DUPLICATE_CHECK_WINDOW_SECONDS:
                return "DUPLICATE_REJECT"

            self.player_langs[matched_name] = lang_code
            return matched_name

        self.player_langs[name_clean] = lang_code
        if self.game_started or len(self.players) >= MAX_PLAYERS:
            self.queue.append(name_clean)
            self.log_event("queued", name=name_clean)
            return name_clean

        if BOT_NAME in self.players:
            bot_index = self.players.index(BOT_NAME)
            self.players.insert(bot_index, name_clean)
        else:
            self.players.append(name_clean)
        self.log_event("connected", name=name_clean)

        if BOT_NAME not in self.players and len(self.players) < MAX_PLAYERS:
            self.players.append(BOT_NAME)
            self.player_langs[BOT_NAME] = "pt" if lang_code == "pt" else "en"
            self.log_event("connected", name=BOT_NAME)

        self.update_lobby_start_countdown(advance=False)
        return name_clean

    def rotate_queue_and_restart(self) -> None:
        """Promote queued players into the next match or reset the room to lobby state."""

        for p_name in list(self.players):
            self.player_langs.pop(p_name, None)
            self.last_seen.pop(p_name, None)

        human_slots = max(1, MAX_PLAYERS - 1)
        promoted_players = self.queue[:human_slots]
        self.queue = self.queue[human_slots:]

        self.players = promoted_players
        if self.players and len(self.players) < MAX_PLAYERS:
            self.players.append(BOT_NAME)
            first_human = promoted_players[0]
            self.player_langs[BOT_NAME] = "pt" if self.player_langs.get(first_human, "en") == "pt" else "en"

        self.game_started = False
        self.game_end_reason = ""
        self.reset_turn_flags()

        self.resolution = 0
        self.panic = 20
        self.active_card = None
        self.hands = {}
        self.has_shouted_deploy = {}
        self.discard_pile = []
        self.draw_pile = []
        self.events = [{"key": "lobby_wait", "kwargs": {"num": MAX_PLAYERS}}]
        self.pending_audios = {}
        self.last_move_time = time.time()
        self.modal_is_warm = False
        self.modal_is_warming_up = False
        self.lobby_start_countdown = 0 if len(self.players) >= MIN_PLAYERS_TO_START else -1

    def can_start_lobby_match(self) -> bool:
        """Return whether the current lobby is ready to warm up and start."""
        return (
            not self.game_started
            and self.restart_countdown == 0
            and len(self.players) >= MIN_PLAYERS_TO_START
            and self.lobby_start_countdown == 0
        )

    def update_lobby_start_countdown(self, advance: bool = True) -> None:
        """Advance or reset the pre-match countdown based on active room size."""
        if self.game_started or self.restart_countdown > 0:
            return

        if len(self.players) < MIN_PLAYERS_TO_START:
            self.lobby_start_countdown = -1
            return

        if self.lobby_start_countdown < 0:
            self.lobby_start_countdown = LOBBY_START_COUNTDOWN_SECONDS
        elif advance and self.lobby_start_countdown > 0:
            self.lobby_start_countdown -= 1

    def tick_countdown(self) -> None:
        """Advance server timers, inactivity cleanup, shout windows, and bot accusations."""
        now = time.time()
        if now - self._last_tick_time < 0.9: return
        self._last_tick_time = now


        if self.game_started and now - self.last_move_time > MAX_ROOM_INACTIVITY_TIMEOUT_SECONDS:
            self.log_event("game_over_timeout")
            self.handle_game_over("timeout")
            return


        inactive_limit = PLAYER_HEARTBEAT_KICK_LIMIT_SECONDS
        for p_name in list(self.players):
            if p_name == BOT_NAME:
                continue
            if p_name not in self.last_seen:
                self.last_seen[p_name] = now
            elif now - self.last_seen[p_name] > inactive_limit:
                self.leave_game(p_name)

        for q_name in list(self.queue):
            if q_name not in self.last_seen:
                self.last_seen[q_name] = now
            elif now - self.last_seen[q_name] > inactive_limit:
                self.leave_game(q_name)


        if not self.game_started and self.restart_countdown > 0:
            self.restart_countdown -= 1
            if self.restart_countdown == 0:
                self.rotate_queue_and_restart()
            return

        self.update_lobby_start_countdown()

        if self.waiting_for_shout:


            if self.active_player < len(self.players) and self.players[self.active_player] == BOT_NAME:
                if random.random() < 0.90:
                    self.shout_deploy(BOT_NAME)
                    return

            if self.shout_countdown > 0:
                self.shout_countdown -= 1
            if self.shout_countdown == 0:
                if self.active_player < len(self.players):
                    p_name = self.players[self.active_player]
                    self.log_event("shout_fail", name=p_name)
                self.waiting_for_shout = False
                was_turn_start = self.is_turn_start_shout
                self.is_turn_start_shout = False
                if not was_turn_start:
                    skip = getattr(self, "pending_skip_on_shout", False)
                    self.pending_skip_on_shout = False
                    self.pass_turn(skip)
            return


        if self.game_started:

            for idx, p_name in enumerate(self.players):
                if p_name != BOT_NAME and p_name in self.hands:


                    is_vulnerable = (
                        len(self.hands[p_name]) == 1 and
                        not self.has_shouted_deploy.get(p_name, False) and
                        not (self.waiting_for_shout and self.active_player == idx)
                    )
                    if is_vulnerable:

                        if random.random() < 0.25:
                            self.accuse_player(idx, BOT_NAME)
                            break


        if self.game_started and not self.is_picking_color:
            if self.active_player < len(self.players) and self.players[self.active_player] == BOT_NAME:
                self.turn_time_left = PLAYER_TURN_TIME_LIMIT_SECONDS
                return
            if self.turn_time_left > 0:
                self.turn_time_left -= 1
            if self.turn_time_left == 0:
                if self.active_player < len(self.players):
                    p_name = self.players[self.active_player]
                    self.log_event("turn_timeout", name=p_name)
                    card = self.get_unique_card(p_name)
                    if card and p_name in self.hands:
                        self.hands[p_name].append(card)
                        self.sort_hand(p_name)
                self.pass_turn()

    def start_shout_window(self) -> None:
        """Open the short Deploy shout window for the active player."""
        self.waiting_for_shout = True
        self.shout_countdown = SERVER_SHOUT_WINDOW_BUFFER_SECONDS

    def get_unique_starting_hand(self, player_name: str, size: int) -> list[Card]:
        """Draw the initial hand from the shuffled deck.

        Args:
            player_name: Player receiving the cards.
            size: Number of cards to draw.

        Returns:
            Cards drawn from the top of the deck.
        """


        hand = []
        for _ in range(size):
            if self.draw_pile:
                hand.append(self.draw_pile.pop(0))
        return hand

    def get_unique_card(self, player_name: str) -> Card | None:
        """Draw one card, rebuilding the draw pile when needed.

        Args:
            player_name: Player requesting the draw.

        Returns:
            The drawn card, or `None` if no card can be produced.
        """
        if not self.draw_pile:
            if self.discard_pile:
                self.draw_pile = list(self.discard_pile)
                self.discard_pile = []
                random.shuffle(self.draw_pile)
            else:
                self.draw_pile = generate_full_deck()
                random.shuffle(self.draw_pile)


        return self.draw_pile.pop(0)

    def draw_card(self, caller_id: str) -> ServerResponse:
        """Handle an active player's manual draw action.

        Args:
            caller_id: Player name making the draw request.

        Returns:
            State/toast response consumed by the custom HTML component.
        """
        lang = self.player_langs.get(caller_id, "en")
        if not self.game_started: return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_game_not_started"]}
        if caller_id not in self.players: return {"state": None, "toast": ""}
        p_idx = self.players.index(caller_id)

        if p_idx != self.active_player: return {"state": None, "toast": UI_I18N[lang]["toast_not_turn"]}
        if self.waiting_for_shout: return {"state": None, "toast": UI_I18N[lang]["toast_wait_shout"]}
        if self.is_picking_color: return {"state": None, "toast": UI_I18N[lang]["toast_wait_color"]}

        card = self.get_unique_card(caller_id)
        if card:
            self.hands[caller_id].append(card)
            self.sort_hand(caller_id)
            self.has_drawn_this_turn = True
            self.has_shouted_deploy[caller_id] = False

            self.turn_time_left = min(PLAYER_TURN_TIME_LIMIT_SECONDS, self.turn_time_left + 10)

            self.log_event("draw", name=caller_id)
            self.last_move_time = time.time()
        return {"state": self.get_state(caller_id), "toast": ""}

    def is_valid_play(self, card: Card) -> bool:
        """Check whether a card can be played on the current active card.

        Args:
            card: Candidate card from a player's hand.

        Returns:
            True when the card matches by stack/category or is wild.
        """
        if not self.active_card: return True
        if card["stack"] == "wild" or self.active_card["stack"] == "wild": return True
        return card["stack"] == self.active_card["stack"] or card["category"] == self.active_card["category"]

    def render_leaderboard_html(self, lang: str = "en") -> str:
        """Render the leaderboard as standalone HTML.

        Args:
            lang: Language code for labels.

        Returns:
            HTML string for the leaderboard tab.
        """
        if lang not in ["en", "pt"]:
            lang = "en"

        t = UI_I18N[lang]

        sorted_board = sorted(
            self.leaderboard_cache.items(),
            key=lambda x: (x[1]["wins"], x[1]["xp"]),
            reverse=True
        )

        html = f"""
        <style>
            .lb-card {{
                background: radial-gradient(circle at top, #10152e 0%, #070915 100%) !important;
                border: 2px solid #44345d !important;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(0, 243, 255, 0.05) !important;
                border-radius: 16px !important;
                padding: 25px !important;
                max-width: 750px;
                margin: 0 auto;
                font-family: inherit;
                text-align: center;
                box-sizing: border-box;
            }}
            .lb-header {{
                background: linear-gradient(90deg, #1e3a8a 0%, #1d4ed8 100%) !important;
                border: 2.5px solid #00f3ff !important;
                border-radius: 30px !important;
                padding: 10px 40px !important;
                display: inline-block;
                color: #00f3ff !important;
                font-weight: 900 !important;
                font-size: 18px !important;
                letter-spacing: 2px !important;
                text-transform: uppercase;
                box-shadow: 0 0 20px rgba(0, 243, 255, 0.4) !important;
                margin-bottom: 25px;
                text-shadow: 0 0 8px rgba(0, 243, 255, 0.6) !important;
            }}
            .lb-row {{
                display: flex !important;
                align-items: center !important;
                background: rgba(20, 26, 46, 0.6) !important;
                border: 1.5px solid rgba(255, 255, 255, 0.06) !important;
                border-radius: 50px !important;
                padding: 10px 25px !important;
                margin-bottom: 12px !important;
                gap: 15px !important;
                transition: all 0.25s ease !important;
                box-sizing: border-box;
            }}
            .lb-row:hover {{
                transform: translateY(-2px) !important;
                background: rgba(0, 243, 255, 0.06) !important;
                border-color: #00f3ff !important;
                box-shadow: 0 0 15px rgba(0, 243, 255, 0.15) !important;
            }}
            .lb-rank-col {{
                width: 40px !important;
                font-size: 22px !important;
                font-weight: 900 !important;
                color: #cbd5e0;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }}
            .lb-avatar {{
                width: 44px !important;
                height: 44px !important;
                border-radius: 50% !important;
                background: #1c1d2e !important;
                border: 2px solid #4a5568 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                overflow: hidden !important;
                flex-shrink: 0 !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3) !important;
            }}
            .lb-avatar-svg {{
                width: 24px;
                height: 24px;
                fill: #cbd5e0;
            }}
            .lb-name-pill {{
                flex-grow: 1 !important;
                background: rgba(10, 14, 28, 0.8) !important;
                border: 1.5px solid rgba(255, 255, 255, 0.06) !important;
                border-radius: 25px !important;
                padding: 8px 20px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                gap: 10px !important;
                box-sizing: border-box;
            }}
            .lb-player-name {{
                font-weight: bold !important;
                font-size: 15px !important;
                color: #ffffff !important;
            }}
            .lb-stars {{
                color: #f1c40f !important;
                font-size: 11px !important;
                letter-spacing: 1px !important;
                text-shadow: 0 0 5px rgba(241, 196, 15, 0.4);
            }}
            .lb-score-col {{
                font-size: 18px !important;
                font-weight: 900 !important;
                color: #2ecc71 !important;
                text-shadow: 0 0 8px rgba(46, 204, 113, 0.4) !important;
                min-width: 110px !important;
                text-align: right !important;
                flex-shrink: 0;
            }}
            .lb-row-gold {{ border-color: rgba(255, 215, 0, 0.35) !important; background: rgba(255, 215, 0, 0.03) !important; }}
            .lb-row-gold .lb-avatar {{ border-color: #ffd700 !important; box-shadow: 0 0 8px rgba(255, 215, 0, 0.4) !important; }}
            .lb-row-gold .lb-avatar-svg {{ fill: #ffd700; }}
            .lb-row-gold .lb-name-pill {{ border-color: rgba(255, 215, 0, 0.2) !important; }}

            .lb-row-silver {{ border-color: rgba(192, 192, 192, 0.35) !important; background: rgba(192, 192, 192, 0.03) !important; }}
            .lb-row-silver .lb-avatar {{ border-color: #c0c0c0 !important; box-shadow: 0 0 8px rgba(192, 192, 192, 0.4) !important; }}
            .lb-row-silver .lb-avatar-svg {{ fill: #c0c0c0; }}

            .lb-row-bronze {{ border-color: rgba(205, 127, 50, 0.35) !important; background: rgba(205, 127, 50, 0.03) !important; }}
            .lb-row-bronze .lb-avatar {{ border-color: #cd7f32 !important; box-shadow: 0 0 8px rgba(205, 127, 50, 0.4) !important; }}
            .lb-row-bronze .lb-avatar-svg {{ fill: #cd7f32; }}
        </style>

        <div class="lb-card">
            <div class="lb-header">{t["lb_title"]}</div>
            <div style="display: flex; flex-direction: column;">
        """

        trophies = {1: "🥇", 2: "🥈", 3: "🥉"}
        star_ratings = {
            "role_intern": "⭐",
            "role_junior": "⭐⭐",
            "role_mid": "⭐⭐⭐",
            "role_senior": "⭐⭐⭐⭐",
            "role_lead": "⭐⭐⭐⭐⭐"
        }

        for idx, (p_name, stats) in enumerate(sorted_board[:20], 1):
            rank_display = trophies.get(idx, f"{idx}")

            xp_val = stats["xp"]
            if xp_val < 500:
                role_key = "role_intern"
            elif xp_val < 1500:
                role_key = "role_junior"
            elif xp_val < 4000:
                role_key = "role_mid"
            elif xp_val < 8000:
                role_key = "role_senior"
            else:
                role_key = "role_lead"

            role_title = t[role_key]
            stars_visual = star_ratings[role_key]

            row_class = ""
            if idx == 1: row_class = "lb-row-gold"
            elif idx == 2: row_class = "lb-row-silver"
            elif idx == 3: row_class = "lb-row-bronze"

            wins_sub_en = f"({stats['wins']} wins / {stats['games']} matches)"
            wins_sub_pt = f"({stats['wins']} vitórias / {stats['games']} partidas)"
            wins_sub = wins_sub_en if lang == "en" else wins_sub_pt

            html += f"""
            <div class="lb-row {row_class}">
                <div class="lb-rank-col">{rank_display}</div>
                <div class="lb-avatar">
                    <svg class="lb-avatar-svg" viewBox="0 0 24 24">
                        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                    </svg>
                </div>
                <div class="lb-name-pill">
                    <span class="lb-player-name">{p_name} <span style="font-size: 11px; color: #a0aec0; font-weight: normal; margin-left: 5px;">{wins_sub}</span></span>
                    <span class="lb-stars" title="{role_title}">{stars_visual}</span>
                </div>
                <div class="lb-score-col">{xp_val} XP</div>
            </div>
            """

        html += """
            </div>
        </div>
        """
        return html

    def async_save_leaderboard_to_hf(self) -> None:
        """Persist the in-memory leaderboard to the configured Hugging Face dataset."""
        try:
            import csv
            temp_path = "./leaderboard.csv"
            with open(temp_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["player_name", "wins", "losses", "xp", "games_played"])
                for p_name, stats in self.leaderboard_cache.items():
                    writer.writerow([p_name, stats["wins"], stats["losses"], stats["xp"], stats["games"]])

            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=self.leaderboard_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                token=os.getenv("HF_TOKEN"),
                commit_message="Update Leaderboard Career Stats"
            )
            os.remove(temp_path)
        except Exception as e:
            print(f"Leaderboard sync failed: {e}")

    def sort_hand(self, player_name: str) -> None:
        """Sort a player's hand in-place by stack, category priority, then card id.

        Args:
            player_name: Player whose hand should be sorted.
        """
        if player_name in self.hands:
            color_order = {"green": 1, "blue": 2, "red": 3, "yellow": 4, "wild": 5}


            category_order = {
                "SUPER": 1,
                "FIX": 2,
                "REFACTOR": 3,
                "TECH_DEBT": 4,
                "DOCS": 5,
                "PATCH": 6,
                "STACK_OVERFLOW": 7,
                "SPAGHETTI": 8,
                "BLIND_PR": 9,
                "BUG": 10,
                "SKIP": 11,
                "REVERSE": 12,
                "ATTACK": 13,
                "WILD": 14,
                "NUKE": 15
            }

            self.hands[player_name].sort(
                key=lambda c: (
                    color_order.get(c.get("stack", "wild"), 99),
                    category_order.get(c.get("category", ""), 99),
                    c.get("id", 0)
                )
            )

    def handle_game_over(self, reason: str | None = None) -> None:
        """Finalize the current match, update scores, and schedule the next room rotation.

        Args:
            reason: End-state reason used by clients to select the right toast and audio.
        """
        self.game_started = False
        self.restart_countdown = GAME_RESTART_COUNTDOWN_SECONDS
        self.pending_audios = {}
        self.modal_is_warm = False
        winner_name = None
        if self.resolution >= 100:
            for p_name in self.players:
                if len(self.hands.get(p_name, [])) == 0:
                    winner_name = p_name
                    break

        if reason:
            self.game_end_reason = reason
        elif self.panic >= 100:
            self.game_end_reason = "game_over"
        elif winner_name:
            self.game_end_reason = "victory"
        else:
            self.game_end_reason = "game_over"

        for p_name in self.players:
            if p_name not in self.leaderboard_cache:
                self.leaderboard_cache[p_name] = {"wins": 0, "losses": 0, "xp": 0, "games": 0}

            stats = self.leaderboard_cache[p_name]
            m_stats = self.match_stats.get(p_name, {"res_contrib": 0, "panic_mitigation": 0})


            base_xp = 50
            win_loss_xp = 0
            if self.panic >= 100:

                stats["losses"] += 1
                win_loss_xp = 0
            elif winner_name:
                if p_name == winner_name:
                    stats["wins"] += 1
                    win_loss_xp = 150
                else:
                    stats["losses"] += 1
                    win_loss_xp = 25

            contrib_xp = int(m_stats["res_contrib"] * 1.0 + m_stats["panic_mitigation"] * 1.5)
            total_earned_xp = base_xp + win_loss_xp + contrib_xp

            stats["xp"] += total_earned_xp
            stats["games"] += 1


        threading.Thread(target=self.async_save_leaderboard_to_hf).start()

    def play_card(self, player_index: int, card_index: int, caller_id: str) -> ServerResponse:
        """Play one card from a player's hand.

        Args:
            player_index: Index of the player whose hand is being used.
            card_index: Index of the card inside that player's hand.
            caller_id: Player name initiating the action.

        Returns:
            State/toast response consumed by the custom HTML component.
        """
        lang = self.player_langs.get(caller_id, "en")
        if not self.game_started: return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_game_not_started"]}
        if caller_id not in self.players: return {"state": None, "toast": ""}
        p_idx = self.players.index(caller_id)

        if p_idx != player_index: return {"state": None, "toast": UI_I18N[lang]["toast_not_turn"]}
        if self.is_picking_color: return {"state": None, "toast": UI_I18N[lang]["toast_wait_color"]}
        if player_index != self.active_player: return {"state": None, "toast": UI_I18N[lang]["toast_not_turn"]}

        p_name = self.players[player_index]
        card = self.hands[p_name][card_index]
        if not self.is_valid_play(card): return {"state": None, "toast": UI_I18N[lang]["toast_invalid"]}

        if self.active_card:
            self.discard_pile.append(json.loads(json.dumps(self.active_card)))

        new_res = min(100, max(0, self.resolution + card["res"]))
        new_panic = min(100, max(0, self.panic + card["panic"]))
        contrib_res = max(0, card["res"])
        mitigation_panic = abs(min(0, card["panic"]))

        if caller_id in self.match_stats:
            self.match_stats[caller_id]["res_contrib"] += contrib_res
            self.match_stats[caller_id]["panic_mitigation"] += mitigation_panic

        self.resolution = new_res
        self.panic = new_panic
        self.active_card = json.loads(json.dumps(card))
        self.hands[p_name].pop(card_index)

        if len(self.hands[p_name]) > 1:
            self.has_shouted_deploy[p_name] = False

        res_sign = '+' if card['res'] >= 0 else ''
        panic_sign = '+' if card['panic'] >= 0 else ''
        has_quote = False
        card_type = "good"

        if card["category"] == "SKIP":
            has_quote = True
            card_type = "bad"
        elif card["res"] == 0 and card["panic"] == 0 and card["category"] not in ["NUKE", "SUPER"]:
            pass
        elif card["res"] < 0 or card["category"] == "NUKE":
            has_quote = True
            card_type = "bad"
        elif card["res"] > 0 and card["panic"] > 0:
            has_quote = True
            card_type = "bad"
        elif card["res"] >= 0 or card["panic"] < 0:
            has_quote = True
            card_type = "good"

        quote_id = time.time()

        self.log_event("play", name=caller_id, card=card['name'], feedback=card['feedback'],
            res=res_sign+str(card['res']), pan=panic_sign+str(card['panic']),
            has_quote=has_quote, quote=None, quote_id=quote_id)

        director_quote_task = None
        if has_quote:
            director_quote_task = {
                "type": "director_quote",
                "card_played": card['name'].get("en", ""),
                "card_type": card_type,
                "event_id": quote_id
            }

        def enqueue_director_quote() -> None:
            if director_quote_task and self.game_started:
                llm_queue.put(director_quote_task)

        self.last_move_time = time.time()

        if self.panic >= 100:
            self.log_event("game_over")
            self.handle_game_over("game_over")
            return {"state": self.get_state(caller_id), "toast": ""}

        if len(self.hands[p_name]) == 0:
            if self.resolution == 100:
                self.log_event("win", name=caller_id)
                self.handle_game_over("victory")
                return {"state": self.get_state(caller_id), "toast": ""}
            else:
                self.draw_cards_for_player(player_index, 2)
                self.log_event("empty_punish", name=caller_id)
                self.pass_turn()
                enqueue_director_quote()
                return {"state": self.get_state(caller_id), "toast": ""}

        if len(self.hands[p_name]) == 1:
            if card["stack"] == "wild":
                self.is_picking_color = True
                self.wild_draw_four_pending = card.get("drawFour", False)
                self.waiting_for_shout = False
                enqueue_director_quote()
                return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_pick_shout"]}
            else:
                self.start_shout_window()
                self.has_shouted_deploy[p_name] = False
                enqueue_director_quote()
                return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_alert_deploy"]}

        skip_next = False
        if card.get("skip"):
            next_player = (self.active_player + self.direction) % len(self.players)
            self.log_event("skip", name=self.players[next_player])
            skip_next = True

        if card.get("reverse"):
            self.direction *= -1
            self.log_event("reverse")
            if len(self.players) == 2:
                next_player = (self.active_player + self.direction) % len(self.players)
                self.log_event("skip", name=self.players[next_player])
                skip_next = True

        if card.get("drawTwo"):
            next_player = (self.active_player + self.direction) % len(self.players)
            self.draw_cards_for_player(next_player, 2)
            self.log_event("attack", name=self.players[next_player])
            skip_next = True

        if len(self.hands[p_name]) == 1:
            if card["stack"] == "wild":
                self.is_picking_color = True
                self.wild_draw_four_pending = card.get("drawFour", False)
                self.waiting_for_shout = False
                enqueue_director_quote()
                return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_pick_shout"]}
            else:
                self.start_shout_window()
                self.has_shouted_deploy[p_name] = False
                self.pending_skip_on_shout = skip_next
                enqueue_director_quote()
                return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_alert_deploy"]}

        if card["stack"] == "wild":
            self.is_picking_color = True
            self.wild_draw_four_pending = card.get("drawFour", False)
            enqueue_director_quote()
            return {"state": self.get_state(caller_id), "toast": ""}

        self.pass_turn(skip_next)
        enqueue_director_quote()
        return {"state": self.get_state(caller_id), "toast": ""}

    def reset_turn_flags(self) -> None:
        """Clear transient state associated with the current turn."""
        self.has_drawn_this_turn = False
        self.waiting_for_shout = False
        self.shout_countdown = 0
        self.is_picking_color = False
        self.turn_start_shout_shown = False
        self.is_turn_start_shout = False
        self.pending_skip_on_shout = False

    def leave_game(self, caller_id: str) -> ServerResponse:
        """Remove a player from queue or active match.

        Args:
            caller_id: Player name leaving the room.

        Returns:
            State/toast response consumed by the custom HTML component.
        """
        lang = self.player_langs.get(caller_id, "en")
        self.last_seen.pop(caller_id, None)

        if caller_id in self.queue:
            self.queue.remove(caller_id)
            return {"state": self.get_state(""), "toast": UI_I18N[lang]["toast_left_queue"]}

        if caller_id not in self.players:
            return {"state": None, "toast": ""}

        p_idx = self.players.index(caller_id)

        if not self.game_started:
            self.players.pop(p_idx)
            self.log_event("left_lobby", name=caller_id)
            return {"state": self.get_state(""), "toast": UI_I18N[lang]["toast_left_queue"]}

        self.players.pop(p_idx)
        hand = self.hands.pop(caller_id, [])
        self.has_shouted_deploy.pop(caller_id, None)

        self.draw_pile.extend(hand)
        random.shuffle(self.draw_pile)

        self.log_event("left_game", name=caller_id)

        if len(self.players) < 2:
            self.log_event("game_over_abandon")
            self.handle_game_over("abandon")
            return {"state": self.get_state(""), "toast": UI_I18N[lang]["toast_game_over_abandon"]}

        if self.active_player == p_idx:
            self.active_player = self.active_player % len(self.players)
            self.reset_turn_flags()
            self.turn_time_left = PLAYER_TURN_TIME_LIMIT_SECONDS
            self.check_turn_start_deploy()
            self.trigger_bot_if_active()
        elif self.active_player > p_idx:
            self.active_player -= 1

        self.last_move_time = time.time()
        return {"state": self.get_state(""), "toast": UI_I18N[lang]["toast_left_game"]}

    def accuse_player(self, target_idx: int, caller_id: str) -> ServerResponse:
        """Accuse another player of missing the Deploy shout.

        Args:
            target_idx: Index of the accused player.
            caller_id: Player name making the accusation.

        Returns:
            State/toast response consumed by the custom HTML component.
        """
        lang = self.player_langs.get(caller_id, "en")
        if not self.game_started: return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_game_not_started"]}
        if caller_id not in self.players: return {"state": None, "toast": ""}
        p_idx = self.players.index(caller_id)

        if p_idx == target_idx: return {"state": None, "toast": UI_I18N[lang]["toast_cant_accuse_self"]}
        if self.waiting_for_shout and self.active_player == target_idx: return {"state": None, "toast": UI_I18N[lang]["toast_wait_accuse"]}

        t_name = self.players[target_idx]
        if len(self.hands[t_name]) == 1 and not self.has_shouted_deploy[t_name]:
            self.draw_cards_for_player(target_idx, 2)
            self.log_event("accuse_success", name=t_name)
            self.last_move_time = time.time()
            return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_accuse_success"].replace("{name}", t_name)}
        return {"state": None, "toast": UI_I18N[lang]["toast_accuse_invalid"]}

    def draw_cards_for_player(self, player_idx: int, count: int) -> None:
        """Draw penalty cards into a player's hand.

        Args:
            player_idx: Target player index.
            count: Number of cards to draw.
        """
        if 0 <= player_idx < len(self.players):
            p_name = self.players[player_idx]
            for _ in range(count):
                card = self.get_unique_card(p_name)
                if card: self.hands[p_name].append(card)
            self.sort_hand(p_name)
            self.has_shouted_deploy[p_name] = False

    def select_wild_color(self, color_name: str, caller_id: str) -> ServerResponse:
        """Resolve a pending wild-card color choice.

        Args:
            color_name: Selected stack color.
            caller_id: Player name choosing the color.

        Returns:
            State/toast response consumed by the custom HTML component.
        """
        lang = self.player_langs.get(caller_id, "en")
        if not self.game_started: return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_game_not_started"]}
        if caller_id not in self.players: return {"state": None, "toast": ""}
        p_idx = self.players.index(caller_id)

        if p_idx != self.active_player: return {"state": None, "toast": UI_I18N[lang]["toast_not_turn"]}

        self.active_card["stack"] = color_name
        self.is_picking_color = False
        self.log_event("wild_pick", color=color_name.upper())
        self.last_move_time = time.time()


        skip_next = False
        if self.wild_draw_four_pending:
            next_player = (self.active_player + self.direction) % len(self.players)
            self.draw_cards_for_player(next_player, 4)
            n_name = self.players[next_player]
            self.log_event("wild_attack", name=n_name)
            skip_next = True
            self.wild_draw_four_pending = False


        if len(self.hands[caller_id]) == 1:
            self.start_shout_window()
            self.has_shouted_deploy[caller_id] = False
            self.pending_skip_on_shout = skip_next
            return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_alert_deploy"]}

        self.pass_turn(skip_next)
        return {"state": self.get_state(caller_id), "toast": ""}

    def check_turn_start_deploy(self) -> bool:
        """Open a Deploy shout window if the incoming active player is vulnerable."""
        p = self.active_player
        if p >= len(self.players):
            return False
        p_name = self.players[p]
        if p_name not in self.hands:
            return False
        if (len(self.hands[p_name]) == 1 and not self.has_shouted_deploy[p_name] and not self.waiting_for_shout and not self.turn_start_shout_shown):
            self.turn_start_shout_shown = True
            self.is_turn_start_shout = True
            self.start_shout_window()
            self.log_event("shout_alert", name=p_name)
            return True
        return False

    def shout_deploy(self, caller_id: str) -> ServerResponse:
        """Protect the active player from accusation by shouting Deploy.

        Args:
            caller_id: Active player name.

        Returns:
            State/toast response consumed by the custom HTML component.
        """
        lang = self.player_langs.get(caller_id, "en")
        if not self.game_started: return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_game_not_started"]}
        if caller_id not in self.players: return {"state": None, "toast": ""}
        p_idx = self.players.index(caller_id)

        if p_idx != self.active_player: return {"state": None, "toast": UI_I18N[lang]["toast_not_turn"]}

        if len(self.hands[caller_id]) <= 2:
            self.has_shouted_deploy[caller_id] = True
            self.is_turn_start_shout = False
            self.log_event("shout_success", name=caller_id)
            self.last_move_time = time.time()
            if self.waiting_for_shout:
                self.waiting_for_shout = False
                self.shout_countdown = 0
                if not self.is_turn_start_shout:

                    skip = getattr(self, "pending_skip_on_shout", False)
                    self.pending_skip_on_shout = False
                    self.pass_turn(skip)
            return {"state": self.get_state(caller_id), "toast": UI_I18N[lang]["toast_shout_protected"]}
        return {"state": None, "toast": UI_I18N[lang]["toast_shout_invalid"]}

    def pass_turn_manual(self, caller_id: str) -> ServerResponse:
        """Let the active player manually pass after drawing or missing a shout.

        Args:
            caller_id: Active player name.

        Returns:
            State/toast response consumed by the custom HTML component.
        """
        lang = self.player_langs.get(caller_id, "en")
        if caller_id not in self.players: return {"state": None, "toast": ""}
        p_idx = self.players.index(caller_id)

        if p_idx != self.active_player: return {"state": None, "toast": UI_I18N[lang]["toast_not_turn"]}

        if self.waiting_for_shout:
            self.waiting_for_shout = False
            p_name = self.players[self.active_player]
            self.log_event("shout_fail", name=p_name)

            skip = getattr(self, "pending_skip_on_shout", False)
            self.pending_skip_on_shout = False
            self.pass_turn(skip)
            return {"state": self.get_state(caller_id), "toast": ""}

        if self.has_drawn_this_turn:
            self.log_event("pass_turn", name=caller_id)
            self.pass_turn(False)
            self.last_move_time = time.time()
        return {"state": self.get_state(caller_id), "toast": ""}

    def pass_turn(self, skip_next: bool = False) -> None:
        """Advance the active player pointer.

        Args:
            skip_next: Whether to skip the next player in turn order.
        """
        self.has_drawn_this_turn = False
        self.waiting_for_shout = False
        self.shout_countdown = 0
        self.turn_start_shout_shown = False
        self.turn_time_left = PLAYER_TURN_TIME_LIMIT_SECONDS

        if not self.players:
            return

        base_step = 2 if skip_next else 1
        step = base_step * self.direction
        self.active_player = (self.active_player + step) % len(self.players)

        self.check_turn_start_deploy()

        self.trigger_bot_if_active()

    def localize_card(self, card: Card | None, lang: str) -> Card | None:
        """Return a copy of a card localized for the requested language.

        Args:
            card: Card to localize.
            lang: Language code used to select localized fields.
        """
        if not card: return None
        c = card.copy()
        c["name"] = c["name"].get(lang, c["name"]["en"])
        c["feedback"] = c["feedback"].get(lang, c["feedback"]["en"])
        return c

    def render_event(self, evt: dict[str, Any], lang: str) -> str:
        """Render a localized log event to HTML-ready text.

        Args:
            evt: Event descriptor from `self.events`.
            lang: Language code used to format the event.
        """
        kwargs = {}
        for k, v in evt["kwargs"].items():
            if isinstance(v, dict): kwargs[k] = v.get(lang, v.get("en", ""))
            else: kwargs[k] = v

        if evt["key"] == "play":
            quote_html = ""

            if evt["kwargs"].get("has_quote") and evt["kwargs"].get("quote") is not None:
                q_text = kwargs["quote"]
                director = UI_I18N[lang]["director"]
                quote_html = f"<br>&nbsp;&nbsp;&nbsp;↳ 👨‍💼 <span style='color: #e2e8f0 !important;'><b style='color: #ffffff !important;'>{director}:</b> <i style='color: #e2e8f0 !important;'>\"{q_text}\"</i></span>"
            kwargs["quote"] = quote_html

        return LOG_I18N[lang][evt["key"]].format(**kwargs)

    def get_state(self, viewer_id: str = "") -> GameState:
        """Build the reactive state payload consumed by the custom HTML boards.

        Args:
            viewer_id: Player name for personalized hand, audio, and language state.

        Returns:
            Full board state dictionary.
        """
        lang = self.player_langs.get(viewer_id, "en")

        if viewer_id and viewer_id != "":
            self.last_seen[viewer_id] = time.time()

        rendered_logs = [self.render_event(e, lang) for e in self.events]
        formatted_log = "<br><br>".join(rendered_logs)

        crisis_source = CRISES_DATABASE[self.current_crisis_idx] if self.game_started else {"title": {"en": "IDLE", "pt": "OCIOSO"}, "desc": {"en": "Waiting...", "pt": "Aguardando..."}}
        localized_crisis = {
            "title": crisis_source["title"].get(lang, crisis_source["title"]["en"]),
            "desc": crisis_source["desc"].get(lang, crisis_source["desc"]["en"])
        }

        ordered_hands = [self.hands.get(p, []) for p in self.players]
        ordered_shouted = [self.has_shouted_deploy.get(p, False) for p in self.players]

        active_player_name = self.players[self.active_player] if self.active_player < len(self.players) else ""
        hand_length = len(self.hands.get(active_player_name, [])) if active_player_name else 0


        inactivity_left = int(MAX_ROOM_INACTIVITY_TIMEOUT_SECONDS  - (time.time() - self.last_move_time)) if self.game_started else 0


        localized_audio = {"id": 0, "b64": ""}
        player_queue = self.pending_audios.get(viewer_id, [])
        if player_queue:
            next_audio = player_queue.pop(0)
            c_key = next_audio["cache_key"]
            if c_key in self.audio_cache:
                localized_audio["id"] = next_audio["id"]
                localized_audio["b64"] = self.audio_cache[c_key].get(lang, "")

        return {
            "game_started": self.game_started,
            "game_end_reason": self.game_end_reason,
            "players": self.players,
            "queue": self.queue,
            "max_players": MAX_PLAYERS,
            "current_crisis": localized_crisis,
            "resolution": self.resolution,
            "panic": self.panic,
            "active_card": self.localize_card(self.active_card, lang),
            "active_player": self.active_player,
            "hands": [[self.localize_card(c, lang) for c in hand] for hand in ordered_hands],
            "game_log": formatted_log,
            "has_drawn_this_turn": self.has_drawn_this_turn,
            "is_picking_color": self.is_picking_color,
            "has_shouted_deploy": ordered_shouted,
            "waiting_for_shout": self.waiting_for_shout,
            "shout_countdown": self.shout_countdown,
            "restart_countdown": self.restart_countdown,
            "lobby_start_countdown": max(0, self.lobby_start_countdown),
            "is_warming_up": self.modal_is_warming_up,
            "inactivity_left": inactivity_left,
            "turn_left": self.turn_time_left,
            "pending_wild_shout": self.is_picking_color and hand_length == 1,
            "is_turn_start_shout": self.is_turn_start_shout,
            "i18n": UI_I18N[lang],
            "director_audio": localized_audio
        }

    def download_tts_language(
        self,
        cache_key: str,
        text: str,
        lang: str,
        store_cache: bool = True,
        use_warmup_timeout: bool = False,
    ) -> bool:
        """Download one TTS language variant into the shared audio cache.

        Args:
            cache_key: Audio cache bucket for the generated quote.
            text: Quote text to synthesize.
            lang: Language code for voice controls.
            store_cache: Keeps audio in local cache.
            use_warmup_timeout: Uses a longer timeout for cold-start warmup calls.

        Returns:
            True if the audio was successfully downloaded and cached, False otherwise.
        """
        if store_cache and cache_key not in self.audio_cache:
            self.audio_cache[cache_key] = {"en": "", "pt": ""}
        if store_cache and self.audio_cache[cache_key].get(lang):
            return True
        if lang not in TTS_CONTROLS:
            return False

        payload = {
            "control": TTS_CONTROLS[lang],
            "text": text,
            "cfg_value": 4.0,
            "voice_id": TTS_VOICE_ID,
            "seed": TTS_VOICE_SEED
        }
                
        headers = {}
        tts_api_key = os.getenv("TTS_API_KEY")
        if tts_api_key:
            headers["Authorization"] = f"Bearer {tts_api_key}"

        for endpoint in get_endpoint_chain("tts"):
            url = endpoint.get("url", "")
            mode = endpoint.get("mode", "rest")
            api_name = endpoint.get("api_name") or "/generate_api"
            timeout = float(endpoint.get("warmup_timeout" if use_warmup_timeout else "timeout", 120.0))

            try:
                print(f"[TTS] Requesting {lang} audio from {endpoint.get('name', 'endpoint')} ({mode}): {url}", flush=True)
                if mode == "gradio":
                    client = get_tts_gradio_client(endpoint, timeout)
                    result = client.predict(tts_api_key or "", payload, api_name=api_name)
                else:
                    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                    if resp.status_code != 200:
                        print(f"[TTS] REST endpoint failed with status code: {resp.status_code}", flush=True)
                        mark_endpoint_failed("tts", endpoint, f"status {resp.status_code}")
                        continue
                    result = resp.json()

                if isinstance(result, str):
                    result = json.loads(result)
                b64_audio = result.get("wav_base64", "") if isinstance(result, dict) else ""
                if b64_audio:
                    if store_cache:
                        self.audio_cache[cache_key][lang] = b64_audio
                    mark_endpoint_success("tts", endpoint)
                    return True
                print(f"[TTS] Endpoint returned no audio payload: {url}", flush=True)
                mark_endpoint_failed("tts", endpoint, "empty audio payload")
            except Exception as e:
                mark_endpoint_failed("tts", endpoint, str(e))
                print(f"[TTS] Endpoint error ({lang}) at {url}: {e}", flush=True)

        print(f"[TTS] All mapped endpoints failed for language: {lang}", flush=True)
        return False

    def fetch_tts_async(self, quote_en: str, quote_pt: str, event_id: float) -> None:
        """Fetch director TTS audio for languages used by active players.

        Args:
            quote_en: English quote text.
            quote_pt: Portuguese quote text.
            event_id: Log-event identifier linked to the audio payload.
        """
        cache_key = quote_en

        active_langs = {self.player_langs.get(p, "en") for p in self.players}

        if "en" in active_langs:
            self.download_tts_language(cache_key, quote_en, "en")
        if "pt" in active_langs:
            self.download_tts_language(cache_key, quote_pt, "pt")


        self.latest_director_audio = {"id": event_id, "cache_key": cache_key}

    def trigger_bot_if_active(self) -> None:
        """Queue a bot decision when the active player is Nemotron."""
        if not self.game_started or not self.players:
            return
        if self.active_player < len(self.players):
            current_p = self.players[self.active_player]
            if current_p == BOT_NAME:
                llm_queue.put({
                    "type": "bot_decision",
                    "bot_name": current_p
                })

global_server = GameManager()
