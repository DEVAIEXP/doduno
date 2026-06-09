from __future__ import annotations

import os
import threading
import time
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv(override=True)

EndpointConfig = dict[str, Any]

MAPPER_URL = os.getenv(
    "DOD_INFERENCE_MAPPER_URL",
    "https://huggingface.co/datasets/elismasilva/dod-inference-mapper/raw/main/inference_map.json",
)
MAPPER_CACHE_TTL_SECONDS = float(os.getenv("DOD_INFERENCE_MAPPER_TTL_SECONDS", "60"))
ENDPOINT_FAILURE_COOLDOWN_SECONDS = float(os.getenv("DOD_ENDPOINT_FAILURE_COOLDOWN_SECONDS", "180"))
ENDPOINT_WARMUP_TIMEOUT_SECONDS = float(os.getenv("DOD_ENDPOINT_WARMUP_TIMEOUT_SECONDS", "75"))
_mapper_lock = threading.Lock()
_cached_mapper: dict[str, Any] | None = None
_last_mapper_update = 0.0
_endpoint_cooldowns: dict[tuple[str, str], float] = {}


def _refresh_env() -> None:
    """Reload local .env values so development flags override stale shell values."""
    load_dotenv(override=True)


def _env_enabled(name: str, fallback_name: str | None = None) -> bool:
    """Return whether an environment flag is truthy."""
    _refresh_env()
    value = os.getenv(name)
    if value is None and fallback_name:
        value = os.getenv(fallback_name, "")
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _default_endpoint(service: str) -> EndpointConfig:
    """Return the local environment fallback endpoint for a service."""
    _refresh_env()
    if service == "space_b":
        return {
            "name": "env-space-b",
            "url": os.getenv("SPACE_B_URL", "https://elismasilva-voxcpm2-nanovllm-service.hf.space"),
            "mode": "gradio",
            "api_name": "/generate_inference",
        }
    if service == "tts":
        tts_url = os.getenv("TTS_API_URL", "http://127.0.0.1:8000/generate_api")
        tts_mode = os.getenv("TTS_API_MODE", "rest")
        if tts_mode == "gradio" and tts_url.rstrip("/").endswith("/generate_api"):
            tts_url = tts_url.rstrip("/")[: -len("/generate_api")]
        return {
            "name": "env-tts",
            "url": tts_url,
            "mode": tts_mode,
            "api_name": "/generate_api",
        }
    return {"name": f"env-{service}", "url": "", "mode": "rest"}


def _normalize_endpoint(raw_endpoint: Any, service: str, role: str) -> EndpointConfig | None:
    """Normalize one mapper entry into a consistent endpoint dictionary."""
    if isinstance(raw_endpoint, str):
        raw_endpoint = {"url": raw_endpoint}
    if not isinstance(raw_endpoint, dict):
        return None

    default = _default_endpoint(service)
    mode = str(raw_endpoint.get("mode", default.get("mode", "rest"))).strip().lower()
    url = str(raw_endpoint.get("url") or raw_endpoint.get("space") or raw_endpoint.get("src") or "").strip()
    is_http_url = url.startswith("http")
    is_gradio_space_id = mode == "gradio" and "/" in url and " " not in url
    if not is_http_url and not is_gradio_space_id:
        return None

    api_name = str(raw_endpoint.get("api_name", default.get("api_name", ""))).strip()
    if api_name and not api_name.startswith("/"):
        api_name = f"/{api_name}"
    if mode == "gradio" and is_http_url and api_name and url.rstrip("/").endswith(api_name):
        url = url.rstrip("/")[: -len(api_name)]

    timeout = float(raw_endpoint.get("timeout", 120.0))
    warmup_timeout = float(raw_endpoint.get("warmup_timeout", max(timeout, ENDPOINT_WARMUP_TIMEOUT_SECONDS)))

    return {
        "name": str(raw_endpoint.get("name", role)),
        "url": url,
        "mode": mode,
        "api_name": api_name,
        "timeout": timeout,
        "warmup_timeout": warmup_timeout,
        "cooldown_seconds": float(raw_endpoint.get("cooldown_seconds", ENDPOINT_FAILURE_COOLDOWN_SECONDS)),
    }


def _extract_service_endpoints(mapper: dict[str, Any], service: str) -> list[EndpointConfig]:
    """Extract primary and fallback endpoints from mapper JSON."""
    service_config = mapper.get(service, {})
    endpoints: list[EndpointConfig] = []

    if isinstance(service_config, list):
        raw_entries = service_config
    elif isinstance(service_config, dict):
        raw_entries = []
        if "primary" in service_config:
            raw_entries.append(service_config["primary"])
        if "fallback" in service_config:
            raw_entries.append(service_config["fallback"])
        raw_entries.extend(service_config.get("fallbacks", []))
        if "url" in service_config:
            raw_entries.insert(0, service_config)
    else:
        raw_entries = [service_config]

    seen_urls = set()
    for idx, raw_entry in enumerate(raw_entries):
        endpoint = _normalize_endpoint(raw_entry, service, "primary" if idx == 0 else f"fallback-{idx}")
        if not endpoint:
            print(f"[Mapper] Ignored invalid {service} endpoint entry: {raw_entry}", flush=True)
            continue
        if endpoint["url"] in seen_urls:
            continue
        seen_urls.add(endpoint["url"])
        endpoints.append(endpoint)

    return endpoints


def _fetch_mapper() -> dict[str, Any]:
    """Fetch the remote mapper JSON with a short timeout."""
    try:
        _refresh_env()
        hf_token = os.getenv("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        response = requests.get(MAPPER_URL, headers=headers, timeout=3.0)
        if response.status_code == 200:
            mapper = response.json()
            if isinstance(mapper, dict):
                print(f"[Mapper] Loaded inference mapper from {MAPPER_URL}", flush=True)
                return mapper
            print("[Mapper] Remote mapper is not a JSON object. Using environment defaults.", flush=True)
        else:
            print(f"[Mapper] Remote mapper failed with status {response.status_code}.", flush=True)
    except Exception as exc:
        print(f"[Mapper] Failed fetching inference mapper, using defaults: {exc}", flush=True)
    return {}


def get_inference_mapper() -> dict[str, Any]:
    """Return cached mapper JSON, refreshing it after the configured TTL."""
    global _cached_mapper, _last_mapper_update

    if _env_enabled("USE_LOCA", "USE_LOCAL"):
        return {}

    now = time.time()
    with _mapper_lock:
        if _cached_mapper is not None and now - _last_mapper_update < MAPPER_CACHE_TTL_SECONDS:
            return _cached_mapper

        _cached_mapper = _fetch_mapper()
        _last_mapper_update = now
        return _cached_mapper


def mark_endpoint_failed(service: str, endpoint: EndpointConfig, reason: str) -> None:
    """Temporarily skip an endpoint after a runtime failure.

    Args:
        service: Service name, such as space_b or tts.
        endpoint: Endpoint configuration that failed.
        reason: Short failure reason for logs.
    """
    url = endpoint.get("url", "")
    if not url:
        return

    cooldown = float(endpoint.get("cooldown_seconds", ENDPOINT_FAILURE_COOLDOWN_SECONDS))
    retry_at = time.time() + cooldown
    with _mapper_lock:
        _endpoint_cooldowns[(service, url)] = retry_at
    print(f"[Mapper] Disabled {service} endpoint for {cooldown:.0f}s after failure: {url} ({reason})", flush=True)


def mark_endpoint_success(service: str, endpoint: EndpointConfig) -> None:
    """Clear a previously marked endpoint failure after a successful call."""
    url = endpoint.get("url", "")
    if not url:
        return

    with _mapper_lock:
        _endpoint_cooldowns.pop((service, url), None)


def get_endpoint_chain(service: str) -> list[EndpointConfig]:
    """Return available endpoints for a service, always keeping a last-resort retry path."""
    if _env_enabled("USE_LOCA", "USE_LOCAL"):
        endpoint = _default_endpoint(service)
        if endpoint.get("url"):
            print(f"[Mapper] USE_LOCA=True. Using local {service} endpoint: {endpoint['url']}", flush=True)
            return [endpoint]
        return []

    mapper = get_inference_mapper()
    endpoints = _extract_service_endpoints(mapper, service) if mapper else []

    if _env_enabled("PRIORITIZE_FALLBACK_URL") and len(endpoints) > 1:
        print(f"[Mapper] PRIORITIZE_FALLBACK_URL=True. Trying mapped fallback before primary for {service}.", flush=True)
        endpoints = endpoints[1:] + endpoints[:1]

    default_endpoint = _default_endpoint(service)
    if default_endpoint.get("url") and default_endpoint["url"] not in {endpoint["url"] for endpoint in endpoints}:
        endpoints.append(default_endpoint)

    now = time.time()
    available = [
        endpoint
        for endpoint in endpoints
        if now >= _endpoint_cooldowns.get((service, endpoint["url"]), 0.0)
    ]
    skipped_count = len(endpoints) - len(available)
    if skipped_count:
        print(f"[Mapper] Skipping {skipped_count} cooling-down {service} endpoint(s).", flush=True)

    selected = available or endpoints
    if selected:
        names = ", ".join(f"{endpoint.get('name', 'endpoint')}={endpoint['url']}" for endpoint in selected)
        print(f"[Mapper] Active {service} endpoint chain: {names}", flush=True)
    return selected
