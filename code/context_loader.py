"""
context_loader – load and join all dataset CSVs for a given claim row.

Responsibilities:
  1. Load the four dataset CSVs (claims, sample_claims, user_history,
     evidence_requirements) with pandas.
  2. For a claim row, resolve image_paths (semicolon-separated) to absolute
     paths under dataset/images/{sample,test}/…, verify each file exists and
     is readable, and return a list of (image_id, filepath) pairs plus a
     valid_image bool and reason string when any are missing.
  3. Join user_id against user_history.csv → return that row as a dict
     (or None with a flag if the user is not found).
  4. Given claim_object and an issue_family string, filter
     evidence_requirements.csv for matching rows (claim_object match OR "all")
     → return matching requirement text.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import DATASET_DIR, IMAGES_DIR

# ── Lazy singleton DataFrames ─────────────────────────────────────────────────

_claims_df: pd.DataFrame | None = None
_sample_claims_df: pd.DataFrame | None = None
_user_history_df: pd.DataFrame | None = None
_evidence_requirements_df: pd.DataFrame | None = None


def load_claims(path: pathlib.Path | None = None) -> pd.DataFrame:
    """Load and cache the claims CSV (input-only rows for test)."""
    global _claims_df
    if _claims_df is None or path is not None:
        p = path or DATASET_DIR / "claims.csv"
        _claims_df = pd.read_csv(p)
    return _claims_df


def load_sample_claims(path: pathlib.Path | None = None) -> pd.DataFrame:
    """Load and cache sample_claims CSV (inputs + expected outputs)."""
    global _sample_claims_df
    if _sample_claims_df is None or path is not None:
        p = path or DATASET_DIR / "sample_claims.csv"
        _sample_claims_df = pd.read_csv(p)
    return _sample_claims_df


def load_user_history() -> pd.DataFrame:
    """Load and cache user_history CSV."""
    global _user_history_df
    if _user_history_df is None:
        _user_history_df = pd.read_csv(DATASET_DIR / "user_history.csv")
    return _user_history_df


def load_evidence_requirements() -> pd.DataFrame:
    """Load and cache evidence_requirements CSV."""
    global _evidence_requirements_df
    if _evidence_requirements_df is None:
        _evidence_requirements_df = pd.read_csv(DATASET_DIR / "evidence_requirements.csv")
    return _evidence_requirements_df


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ImageInfo:
    """Metadata for a single resolved image."""

    image_id: str  # e.g. "img_1"
    filepath: pathlib.Path  # absolute path
    exists: bool
    reason: str = ""  # non-empty only when exists is False


@dataclass
class ResolvedImages:
    """Result of resolving all image paths for a claim row."""

    images: list[ImageInfo] = field(default_factory=list)
    valid_image: bool = True
    reason: str = ""  # summary reason when valid_image is False


@dataclass
class ClaimContext:
    """Everything the pipeline needs for one claim row."""

    # Raw input row
    user_id: str
    image_paths_raw: str  # the original semicolon-separated string
    user_claim: str
    claim_object: str  # "car" | "laptop" | "package"

    # Resolved enrichments
    resolved_images: ResolvedImages = field(default_factory=ResolvedImages)
    user_history: dict | None = None  # None → user not found
    user_history_found: bool = False
    evidence_requirements: list[dict] = field(default_factory=list)


# ── Image resolution ─────────────────────────────────────────────────────────


def resolve_images(image_paths_raw: str) -> ResolvedImages:
    """
    Parse semicolon-separated image paths and resolve each to an absolute path
    under the dataset/images/ directory.

    Parameters
    ----------
    image_paths_raw : str
        e.g. "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg"

    Returns
    -------
    ResolvedImages
        Contains a list of ImageInfo items plus an aggregate valid_image flag.
    """
    result = ResolvedImages()
    if not image_paths_raw or pd.isna(image_paths_raw):
        result.valid_image = False
        result.reason = "No image paths provided."
        return result

    missing_reasons: list[str] = []

    for raw_path in image_paths_raw.split(";"):
        raw_path = raw_path.strip()
        if not raw_path:
            continue

        # The CSV stores paths like "images/test/case_001/img_1.jpg".
        # The dataset root already has dataset/, and images/ is inside it.
        # So we resolve relative to the repo's dataset/ parent.
        abs_path = (DATASET_DIR / raw_path).resolve()
        image_id = pathlib.Path(raw_path).stem  # "img_1"

        exists = abs_path.is_file()
        info = ImageInfo(
            image_id=image_id,
            filepath=abs_path,
            exists=exists,
            reason="" if exists else f"File not found: {abs_path}",
        )
        result.images.append(info)

        if not exists:
            missing_reasons.append(f"{image_id}: {abs_path}")

    if missing_reasons:
        result.valid_image = False
        result.reason = (
            f"{len(missing_reasons)} image(s) missing or unreadable: "
            + "; ".join(missing_reasons)
        )

    if not result.images:
        result.valid_image = False
        result.reason = "No image paths could be parsed."

    return result


# ── User history lookup ──────────────────────────────────────────────────────


def get_user_history(user_id: str) -> tuple[dict | None, bool]:
    """
    Look up a user's claim history from user_history.csv.

    Returns
    -------
    (history_dict, found)
        history_dict is None when the user is not found; found is the bool flag.
    """
    df = load_user_history()
    matches = df[df["user_id"] == user_id]
    if matches.empty:
        return None, False
    # Take the first (and presumably only) matching row.
    row = matches.iloc[0].to_dict()
    return row, True


# ── Evidence requirements lookup ─────────────────────────────────────────────


def get_evidence_requirements(
    claim_object: str,
    issue_family: str = "",
) -> list[dict]:
    """
    Filter evidence_requirements.csv for rows that apply to the given
    claim_object (exact match OR "all").  When *issue_family* is provided,
    further filter to rows whose ``applies_to`` contains *issue_family*
    (case-insensitive substring match) OR whose ``claim_object`` is "all".

    Parameters
    ----------
    claim_object : str
        One of "car", "laptop", "package".
    issue_family : str, optional
        e.g. "dent or scratch", "crack", "crushed".

    Returns
    -------
    list[dict]
        Each dict has keys: requirement_id, claim_object, applies_to,
        minimum_image_evidence.
    """
    df = load_evidence_requirements()

    # Match claim_object == row's claim_object OR row's claim_object == "all"
    mask = (df["claim_object"] == claim_object) | (df["claim_object"] == "all")
    filtered = df[mask]

    if issue_family:
        issue_lower = issue_family.lower()
        # Keep rows where applies_to overlaps with the issue_family string
        # or the row is a universal ("all") rule
        further = filtered[
            filtered["applies_to"].str.lower().str.contains(issue_lower, na=False)
            | (filtered["claim_object"] == "all")
        ]
        # If the further filter removed everything, fall back to the broader set
        if not further.empty:
            filtered = further

    return filtered.to_dict(orient="records")


# ── Full context builder ─────────────────────────────────────────────────────


def build_claim_context(row: pd.Series) -> ClaimContext:
    """
    Given a single row from the claims CSV, resolve images, look up user
    history, and gather evidence requirements.  Returns a ClaimContext
    ready for pipeline stages.

    Parameters
    ----------
    row : pd.Series
        Must contain: user_id, image_paths, user_claim, claim_object.
    """
    user_id = str(row["user_id"])
    image_paths_raw = str(row["image_paths"])
    user_claim = str(row["user_claim"])
    claim_object = str(row["claim_object"])

    resolved = resolve_images(image_paths_raw)
    history, found = get_user_history(user_id)
    requirements = get_evidence_requirements(claim_object)

    return ClaimContext(
        user_id=user_id,
        image_paths_raw=image_paths_raw,
        user_claim=user_claim,
        claim_object=claim_object,
        resolved_images=resolved,
        user_history=history,
        user_history_found=found,
        evidence_requirements=requirements,
    )
