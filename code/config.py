"""
config – centralised configuration for the Multi-Modal Evidence Review system.

Reads environment variables (from .env via python-dotenv) and exposes them as
module-level constants.  Provider-specific API keys are validated lazily by
llm_client so evaluation/reporting code can import config without requiring a
specific vendor key up front.
"""

from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv

# ── Load .env from the code/ directory (or repo root) ────────────────────────
_CODE_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent

# Try code/.env first, then repo root .env
for _candidate in (_CODE_DIR / ".env", _REPO_ROOT / ".env"):
    if _candidate.is_file():
        load_dotenv(_candidate)
        break

# ── Provider configuration ───────────────────────────────────────────────────

LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()

OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY", "").strip() or None
GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY", "").strip() or None
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None

# ── Tuning knobs ──────────────────────────────────────────────────────────────

LLM_CONCURRENCY: int = int(os.environ.get("LLM_CONCURRENCY", "8"))
LLM_RETRY_ATTEMPTS: int = int(os.environ.get("LLM_RETRY_ATTEMPTS", "3"))
LLM_RETRY_BASE_DELAY: float = float(os.environ.get("LLM_RETRY_BASE_DELAY", "2.0"))
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "1400"))

# Pricing assumptions used only by evaluation/evaluation_report.md. Override
# these if the selected provider/model uses different pricing.
LLM_INPUT_COST_PER_1M: float = float(os.environ.get("LLM_INPUT_COST_PER_1M", "0.15"))
LLM_OUTPUT_COST_PER_1M: float = float(os.environ.get("LLM_OUTPUT_COST_PER_1M", "0.60"))

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_DIR: pathlib.Path = _REPO_ROOT / "dataset"
IMAGES_DIR: pathlib.Path = DATASET_DIR / "images"
CACHE_DIR: pathlib.Path = _CODE_DIR / ".cache"
CALL_LOG_PATH: pathlib.Path = _CODE_DIR / "evaluation" / "call_log.csv"

# Ensure mutable dirs exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CALL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Output column order (from problem_statement.md) ──────────────────────────

OUTPUT_COLUMNS: list[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# ── Allowed value enums (from problem_statement.md) ──────────────────────────

ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

ALLOWED_OBJECT_PARTS: dict[str, set[str]] = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid",
        "corner", "port", "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal",
        "label", "contents", "item", "unknown",
    },
}
