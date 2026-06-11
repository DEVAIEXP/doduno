from __future__ import annotations

import html
from typing import Any

from game_manager import generate_full_deck


Card = dict[str, Any]

MANUAL_I18N = {
    "en": {
        "title": "How to Play",
        "kicker": "Manual",
        "intro": "DOD - Deploy or Draw is a multiplayer UNO-inspired crisis game. Match cards by stack or action type, survive the Director's panic, and complete the final deploy before production melts down.",
        "objective_title": "Objective",
        "objective": "Raise Crisis Resolution to 100 before IT Director Panic reaches 100. If panic hits 100, the team loses. If the deploy succeeds, the table wins.",
        "deck_title": "Deck",
        "deck_text": "The deck has 108 cards: 100 colored stack cards and 8 wild cards.",
        "stacks_title": "Stacks",
        "rules_title": "Basic UNO Rules",
        "rules": [
            "On your turn, play a card that matches the active stack color or the active card category. The category is the symbol/icon in the center of the card, such as bug, revert, meeting, or +2.",
            "Wild cards can be played on any stack, then you choose the next stack.",
            "If you cannot or do not want to play, draw a card from the pile.",
            "After drawing, you may pass your turn when the drawn card is not played.",
            "The AI player Nemotron is always part of the match and plays by LLM decisions with a local fallback.",
        ],
        "deploy_title": "Deploy Shout",
        "deploy_text": "When you reach one card, shout Deploy. If you forget, another player can accuse you and force a +2 card penalty.",
        "metrics_title": "Resolution and Panic",
        "metrics_text": "Good cards increase Resolution or reduce Panic. Risky cards may still help the deploy but can make the Director panic.",
        "special_title": "Special Cards",
        "special_text": "Special cards change the flow of the match: skip a player, reverse the direction, force extra draws, or choose the next stack.",
        "ai_title": "AI Director",
        "ai_text": "After major plays, the AI Director reacts to the card and the active crisis. If TTS is enabled, the line is also spoken aloud.",
        "examples_title": "Card Examples",
        "stack_green": "Frontend",
        "stack_blue": "Backend",
        "stack_red": "DevOps",
        "stack_yellow": "A.I.",
        "stack_wild": "Wild",
        "res": "Res",
        "pan": "Pan",
        "match": "Match stack or category",
        "same_stack": "Same stack",
        "same_category": "Same category",
        "wild_anytime": "Any stack",
        "draw_effect": "+2 / +4 pressure",
        "skip_effect": "Skip next player",
        "reverse_effect": "Reverse direction",
        "final_tip": "Tip: keep an eye on both your hand and the crisis meters. The funniest play is not always the safest deploy.",
    },
    "pt": {
        "title": "Como Jogar",
        "kicker": "Manual",
        "intro": "DOD - Deploy or Draw é um jogo multiplayer inspirado em UNO. Combine cartas por stack ou tipo de ação, sobreviva ao pânico do Diretor e conclua o deploy final antes da produção derreter.",
        "objective_title": "Objetivo",
        "objective": "Aumente a Resolução da Crise até 100 antes que o Pânico do Diretor de TI chegue a 100. Se o pânico chegar a 100, o time perde. Se o deploy for concluído, a mesa vence.",
        "deck_title": "Baralho",
        "deck_text": "O baralho tem 108 cartas: 100 cartas coloridas de stack e 8 cartas coringa.",
        "stacks_title": "Stacks",
        "rules_title": "Regras Básicas do UNO",
        "rules": [
            "No seu turno, jogue uma carta que combine com a stack ativa ou com a categoria da carta na mesa. A categoria é o símbolo/ícone no centro da carta, como bug, revert, reunião ou +2.",
            "Cartas coringa podem ser jogadas em qualquer stack; depois você escolhe a próxima stack.",
            "Se não puder ou não quiser jogar, compre uma carta do monte.",
            "Depois de comprar, você pode passar a vez quando a carta comprada não for jogada.",
            "O jogador de IA Nemotron sempre participa da partida e decide jogadas com LLM, usando fallback local quando necessário.",
        ],
        "deploy_title": "Gritar Deploy",
        "deploy_text": "Quando ficar com uma carta, grite Deploy. Se esquecer, outro jogador pode acusar você e aplicar uma penalidade de +2 cartas.",
        "metrics_title": "Resolução e Pânico",
        "metrics_text": "Cartas boas aumentam a Resolução ou reduzem o Pânico. Cartas arriscadas ainda podem ajudar o deploy, mas deixam o Diretor mais desesperado.",
        "special_title": "Cartas Especiais",
        "special_text": "Cartas especiais mudam o fluxo da partida: pulam jogador, invertem a direção, forçam compras extras ou escolhem a próxima stack.",
        "ai_title": "Diretor de IA",
        "ai_text": "Depois das jogadas importantes, o Diretor de IA reage à carta e à crise ativa. Se o TTS estiver ligado, a frase também é falada em voz alta.",
        "examples_title": "Exemplos de Cartas",
        "stack_green": "Frontend",
        "stack_blue": "Backend",
        "stack_red": "DevOps",
        "stack_yellow": "I.A.",
        "stack_wild": "Coringa",
        "res": "Res",
        "pan": "Pan",
        "match": "Combine stack ou categoria",
        "same_stack": "Mesma stack",
        "same_category": "Mesma categoria",
        "wild_anytime": "Qualquer stack",
        "draw_effect": "Pressão +2 / +4",
        "skip_effect": "Pula próximo jogador",
        "reverse_effect": "Inverte a direção",
        "final_tip": "Dica: observe sua mão e os medidores da crise. A jogada mais engraçada nem sempre é o deploy mais seguro.",
    },
}

STACK_CLASS = {
    "green": "manual-card-green",
    "blue": "manual-card-blue",
    "red": "manual-card-red",
    "yellow": "manual-card-yellow",
    "wild": "manual-card-wild",
}

STACK_KEYS = {
    "green": "stack_green",
    "blue": "stack_blue",
    "red": "stack_red",
    "yellow": "stack_yellow",
    "wild": "stack_wild",
}


def _escape(value: Any) -> str:
    """Escape a value for safe HTML output."""
    return html.escape(str(value or ""), quote=True)


def _localized(value: Any, lang: str) -> str:
    """Return localized card field text."""
    if isinstance(value, dict):
        return str(value.get(lang) or value.get("en") or "")
    return str(value or "")


def _find_card(deck: list[Card], category: str, stack: str | None = None) -> Card:
    """Find one representative card by category and optional stack."""
    for card in deck:
        if card.get("category") == category and (stack is None or card.get("stack") == stack):
            return card
    raise ValueError(f"Missing manual card sample for {category}")


def _render_card(card: Card, lang: str, label: str = "") -> str:
    """Render a compact static card matching the game board style."""
    t = MANUAL_I18N[lang]
    stack = str(card.get("stack", "wild"))
    stack_class = STACK_CLASS.get(stack, "manual-card-wild")
    name = _localized(card.get("name"), lang)
    symbol = card.get("categorySymbol", "")
    badge = card.get("badge", "")
    res = int(card.get("res", 0))
    panic = int(card.get("panic", 0))
    text_class = "manual-text-dark" if stack == "yellow" else "manual-text-light"
    stack_label = t.get(STACK_KEYS.get(stack, "stack_wild"), stack.upper())
    badge_html = f"<div class='manual-card-badge'>{_escape(badge)}</div>" if badge else ""
    label_html = f"<div class='manual-card-note'>{_escape(label)}</div>" if label else ""
    res_prefix = "+" if res >= 0 else ""
    panic_prefix = "+" if panic >= 0 else ""

    return (
        f"<div class='manual-card-wrap'>"
        f"  <div class='manual-card {stack_class}'>"
        f"    {badge_html}"
        f"    <div class='manual-card-stack'>{_escape(stack_label)}</div>"
        f"    <div class='manual-card-diamond'><span>{_escape(symbol)}</span></div>"
        f"    <div class='manual-card-title {text_class}'>{_escape(name)}</div>"
        f"    <div class='manual-card-stats'>"
        f"      <span class='manual-stat-good'>{_escape(t['res'])}: {res_prefix}{res}%</span>"
        f"      <span class='{'manual-stat-good' if panic <= 0 else 'manual-stat-bad'}'>{_escape(t['pan'])}: {panic_prefix}{panic}%</span>"
        f"    </div>"
        f"  </div>"
        f"  {label_html}"
        f"</div>"
    )


def _render_rule_list(rules: list[str]) -> str:
    """Render manual rules as list items."""
    items = "".join(f"<li>{_escape(rule)}</li>" for rule in rules)
    return f"<ul class='manual-list'>{items}</ul>"


def render_how_to_play_html(lang: str = "en") -> str:
    """Render the localized How to Play page."""
    lang = lang if lang in MANUAL_I18N else "en"
    t = MANUAL_I18N[lang]
    deck = generate_full_deck()
    samples = [
        (_find_card(deck, "FIX", "green"), t["same_stack"]),
        (_find_card(deck, "BUG", "blue"), t["same_category"]),
        (_find_card(deck, "SKIP", "red"), t["skip_effect"]),
        (_find_card(deck, "REVERSE", "yellow"), t["reverse_effect"]),
        (_find_card(deck, "ATTACK", "red"), t["draw_effect"]),
        (_find_card(deck, "WILD"), t["wild_anytime"]),
        (_find_card(deck, "NUKE"), t["draw_effect"]),
    ]
    cards_html = "".join(_render_card(card, lang, label) for card, label in samples)
    stack_chips = "".join(
        f"<span class='manual-stack-chip {STACK_CLASS[color]}'>{_escape(t[key])}</span>"
        for color, key in STACK_KEYS.items()
        if color != "wild"
    )

    return f"""
    <div class="manual-page">
      <div class="manual-hero">
        <div class="manual-kicker">{_escape(t["kicker"])}</div>
        <h2>{_escape(t["title"])}</h2>
        <p>{_escape(t["intro"])}</p>
      </div>
      <div class="manual-grid">
        <section class="manual-panel">
          <h3>{_escape(t["objective_title"])}</h3>
          <p>{_escape(t["objective"])}</p>
        </section>
        <section class="manual-panel">
          <h3>{_escape(t["deck_title"])}</h3>
          <p>{_escape(t["deck_text"])}</p>
        </section>
        <section class="manual-panel">
          <h3>{_escape(t["stacks_title"])}</h3>
          <div class="manual-stack-row">{stack_chips}</div>
        </section>
        <section class="manual-panel">
          <h3>{_escape(t["metrics_title"])}</h3>
          <p>{_escape(t["metrics_text"])}</p>
        </section>
      </div>
      <section class="manual-panel manual-wide">
        <h3>{_escape(t["rules_title"])}</h3>
        {_render_rule_list(t["rules"])}
      </section>
      <section class="manual-panel manual-wide">
        <h3>{_escape(t["special_title"])}</h3>
        <p>{_escape(t["special_text"])}</p>
        <div class="manual-card-gallery">{cards_html}</div>
      </section>
      <div class="manual-grid">
        <section class="manual-panel">
          <h3>{_escape(t["deploy_title"])}</h3>
          <p>{_escape(t["deploy_text"])}</p>
        </section>
        <section class="manual-panel">
          <h3>{_escape(t["ai_title"])}</h3>
          <p>{_escape(t["ai_text"])}</p>
        </section>
      </div>
      <div class="manual-tip">{_escape(t["final_tip"])}</div>
    </div>
    """
