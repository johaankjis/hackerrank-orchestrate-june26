"""
llm_client – async LLM call infrastructure with retry, caching, and logging.

This module provides:
  • call_llm()  – the single async entry-point for every LLM call in the
                  pipeline.  It handles concurrency, retries, disk caching,
                  and per-call logging.
  • Provider adapters for OpenAI-compatible chat completions. Direct OpenAI
    and Azure OpenAI are supported; more providers can be added behind the
    same call_llm() boundary.

Design decisions:
  - asyncio.Semaphore bounds concurrency (default 8, from config).
  - Exponential backoff on 429 / 5xx (3 attempts, base delay 2 s).
  - Disk cache under code/.cache/ keyed on SHA-256 of
    (stage_name, prompt_version, user_content, sorted image hashes).
  - One CSV row per call appended to evaluation/call_log.csv.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import json
import mimetypes
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Optional

import config  # project config module

# ── Concurrency semaphore ─────────────────────────────────────────────────────

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazy-init a module-level semaphore (must be created inside a running
    event loop)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)
    return _semaphore


# ── Disk cache helpers ────────────────────────────────────────────────────────

PROMPT_VERSION = "v0"  # bump when prompts change meaningfully


def _hash_file(path: pathlib.Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(
    stage_name: str,
    user_content: str,
    image_paths: list[pathlib.Path] | None,
) -> str:
    """Compute a deterministic cache key from inputs."""
    parts: list[str] = [stage_name, PROMPT_VERSION, user_content]
    if image_paths:
        # Sort so order doesn't matter for cache hits
        parts.extend(sorted(_hash_file(p) for p in image_paths))
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> pathlib.Path:
    return config.CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> dict | None:
    """Return cached response dict, or None on miss."""
    p = _cache_path(key)
    if p.is_file():
        try:
            return json.loads(p.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_cache(key: str, data: dict) -> None:
    """Persist a response dict to the disk cache."""
    p = _cache_path(key)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ── Call log ──────────────────────────────────────────────────────────────────

_LOG_HEADER = [
    "timestamp",
    "stage",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "cache_hit",
    "image_count",
]
_log_lock = asyncio.Lock() if False else None  # replaced at runtime


def _ensure_log_lock() -> asyncio.Lock:
    global _log_lock
    if _log_lock is None:
        _log_lock = asyncio.Lock()
    return _log_lock


async def _append_call_log(
    stage: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    cache_hit: bool,
    image_count: int,
) -> None:
    """Append one row to evaluation/call_log.csv (async-safe)."""
    lock = _ensure_log_lock()
    async with lock:
        log_path = config.CALL_LOG_PATH
        write_header = not log_path.is_file() or log_path.stat().st_size == 0
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(_LOG_HEADER)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                stage,
                model,
                prompt_tokens,
                completion_tokens,
                round(latency_ms, 1),
                cache_hit,
                image_count,
            ])


# ── Provider adapters ─────────────────────────────────────────────────────────


class LLMResponseFormatError(Exception):
    """Raised when a model response is not a strict JSON object."""

    def __init__(self, message: str, raw_content: Any = None):
        super().__init__(message)
        self.raw_content = raw_content


def _parse_json_object(raw_content: Any) -> dict[str, Any]:
    """
    Parse model content as a strict JSON object.

    Deliberately does not strip Markdown fences or recover embedded JSON; the
    pipeline stages use a correction retry when a provider returns anything
    other than a bare JSON object.
    """
    if isinstance(raw_content, dict):
        return raw_content
    if not isinstance(raw_content, str):
        raise LLMResponseFormatError(
            f"Expected JSON object text, got {type(raw_content).__name__}.",
            raw_content=raw_content,
        )

    text = raw_content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseFormatError(
            f"Model response was not valid JSON: {exc}",
            raw_content=raw_content,
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMResponseFormatError(
            f"Expected a JSON object, got {type(parsed).__name__}.",
            raw_content=raw_content,
        )
    return parsed


def _image_to_data_url(path: pathlib.Path) -> str:
    """Read a local image and return an OpenAI-compatible data URL."""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _build_user_parts(
    user_content: str,
    images: list[pathlib.Path] | None,
) -> list[dict[str, Any]]:
    """Build OpenAI/Azure multimodal user content blocks."""
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": user_content}]
    for image_path in images or []:
        user_parts.append({
            "type": "image_url",
            "image_url": {
                "url": _image_to_data_url(image_path),
                "detail": "high",
            },
        })
    return user_parts


def _usage_dict(usage: Any) -> dict[str, int]:
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }


async def _call_openai_provider(
    model: str,
    system_prompt: str,
    user_content: str,
    images: list[pathlib.Path] | None = None,
) -> dict[str, Any]:
    """Call OpenAI Chat Completions with optional image inputs."""
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or code/.env, "
            "or set LLM_PROVIDER to another implemented provider."
        )

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    response = await client.chat.completions.create(
        model=model,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_user_parts(user_content, images)},
        ],
    )

    content = response.choices[0].message.content or ""
    return {
        "content": content,
        "usage": _usage_dict(getattr(response, "usage", None)),
    }


async def _call_azure_openai_provider(
    model: str,
    system_prompt: str,
    user_content: str,
    images: list[pathlib.Path] | None = None,
) -> dict[str, Any]:
    """Call Azure OpenAI Chat Completions with optional image inputs."""
    required = {
        "AZURE_OPENAI_API_KEY": config.AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": config.AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_DEPLOYMENT": config.AZURE_OPENAI_DEPLOYMENT,
        "AZURE_OPENAI_API_VERSION": config.AZURE_OPENAI_API_VERSION,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing Azure OpenAI environment variable(s): "
            + ", ".join(missing)
            + ". Set them in the environment or code/.env."
        )

    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )

    def _sync_call() -> Any:
        return client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_user_parts(user_content, images)},
            ],
        )

    response = await asyncio.to_thread(_sync_call)
    content = response.choices[0].message.content or ""
    return {
        "content": content,
        "usage": _usage_dict(getattr(response, "usage", None)),
    }


async def _call_provider(
    model: str,
    system_prompt: str,
    user_content: str,
    images: list[pathlib.Path] | None = None,
) -> dict[str, Any]:
    """
    Dispatch to the configured LLM provider.

    Provider adapters return raw model content plus token usage. call_llm()
    performs strict JSON parsing so every pipeline stage gets the same
    cache/log/retry behavior.
    """
    provider = config.LLM_PROVIDER
    if provider == "openai":
        return await _call_openai_provider(model, system_prompt, user_content, images)
    if provider == "azure_openai":
        return await _call_azure_openai_provider(
            model,
            system_prompt,
            user_content,
            images,
        )

    raise NotImplementedError(
        f"LLM_PROVIDER={provider!r} is not implemented in llm_client yet. "
        "The public call_llm() interface is provider-agnostic; add a thin "
        "adapter in _call_provider() for the selected SDK."
    )


# ── Retry wrapper ─────────────────────────────────────────────────────────────


class LLMRetryError(Exception):
    """Raised when all retry attempts are exhausted."""


def _is_retryable_exception(exc: Exception) -> bool:
    """Best-effort retry classifier for common SDK/network failures."""
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    if isinstance(status, int):
        return status in {408, 409, 429} or 500 <= status <= 599

    name = type(exc).__name__.lower()
    retry_terms = (
        "ratelimit",
        "timeout",
        "connection",
        "internalserver",
        "serviceunavailable",
        "apierror",
    )
    return any(term in name for term in retry_terms)


async def _call_with_retry(
    model: str,
    system_prompt: str,
    user_content: str,
    images: list[pathlib.Path] | None,
) -> dict[str, Any]:
    """
    Call _call_provider with exponential backoff on retryable errors
    (HTTP 429 / 5xx or provider-specific rate-limit exceptions).
    """
    last_exc: Exception | None = None
    for attempt in range(config.LLM_RETRY_ATTEMPTS):
        try:
            return await _call_provider(model, system_prompt, user_content, images)
        except NotImplementedError:
            # Don't retry a stub – let it bubble up immediately.
            raise
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_exception(exc):
                raise
            delay = config.LLM_RETRY_BASE_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)

    raise LLMRetryError(
        f"All {config.LLM_RETRY_ATTEMPTS} attempts failed."
    ) from last_exc


# ── Public API ────────────────────────────────────────────────────────────────


async def call_llm(
    stage_name: str,
    model: str,
    system_prompt: str,
    user_content: str,
    images: list[pathlib.Path] | None = None,
) -> dict:
    """
    Single entry-point for every LLM call in the pipeline.

    Flow:
      1. Check disk cache; return immediately on hit.
      2. Acquire concurrency semaphore.
      3. Call provider with retry.
      4. Log the call to evaluation/call_log.csv.
      5. Write result to disk cache.
      6. Return the parsed JSON dict from the model.

    Parameters
    ----------
    stage_name : str
        Pipeline stage label, e.g. "extract_claim", "verify_images".
    model : str
        Model identifier, e.g. "gpt-4o", "gemini-2.0-flash".
    system_prompt : str
        The system-level instruction.
    user_content : str
        The user-turn text (may include the claim conversation, etc.).
    images : list[Path] | None
        Absolute paths to images to include in the request.

    Returns
    -------
    dict
        The parsed JSON object response from the model.
    """
    # ── 1. Cache check ────────────────────────────────────────────────────
    key = _cache_key(stage_name, user_content, images)
    cached = _read_cache(key)
    if cached is not None:
        await _append_call_log(
            stage=stage_name,
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
            cache_hit=True,
            image_count=len(images or []),
        )
        return cached

    # ── 2. Acquire semaphore ──────────────────────────────────────────────
    sem = _get_semaphore()
    async with sem:
        # ── 3. Call with retry ────────────────────────────────────────────
        t0 = time.perf_counter()
        result = await _call_with_retry(model, system_prompt, user_content, images)
        latency_ms = (time.perf_counter() - t0) * 1000

    # ── 4. Log ────────────────────────────────────────────────────────────
    usage = result.get("usage", {})
    await _append_call_log(
        stage=stage_name,
        model=model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        cache_hit=False,
        image_count=len(images or []),
    )

    # ── 5. Cache ──────────────────────────────────────────────────────────
    content = _parse_json_object(result.get("content", result))
    _write_cache(key, content)

    # ── 6. Return ─────────────────────────────────────────────────────────
    return content
