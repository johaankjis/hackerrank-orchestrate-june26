#!/usr/bin/env python3
"""
main.py – CLI entry-point for the Multi-Modal Evidence Review pipeline.

Usage examples:
    # Full run (two-stage strategy)
    python main.py --input ../dataset/claims.csv --output ../output.csv

    # Quick smoke-test on 5 rows
    python main.py --input ../dataset/sample_claims.csv --output sample_out.csv --limit 5

    # Single-pass strategy
    python main.py --strategy single_pass --input ../dataset/claims.csv --output ../output.csv
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time

import pandas as pd

# Project modules  (config is imported first so it validates env vars early)
import config
from context_loader import build_claim_context, load_claims
from pipeline_stages import decide, extract_claim, single_pass, verify_images

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Evidence Review – damage-claim verification pipeline.",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=config.DATASET_DIR / "claims.csv",
        help="Path to the input claims CSV (default: dataset/claims.csv).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=config._REPO_ROOT / "output.csv",
        help="Path for the output predictions CSV (default: repo_root/output.csv).",
    )
    parser.add_argument(
        "--strategy",
        choices=["two_stage", "single_pass"],
        default="two_stage",
        help=(
            "Pipeline strategy. 'two_stage' runs extraction then verification "
            "then decision. 'single_pass' merges into one LLM call. "
            "(default: two_stage)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (for quick testing).",
    )
    return parser.parse_args(argv)


# ── Row processor ─────────────────────────────────────────────────────────────


async def process_row(row: pd.Series, strategy: str) -> dict:
    """
    Run the full pipeline for one claim row and return a dict matching
    config.OUTPUT_COLUMNS.

    Parameters
    ----------
    row : pd.Series
        One row from the claims CSV.
    strategy : str
        "two_stage" or "single_pass".

    Returns
    -------
    dict
        Keys are exactly config.OUTPUT_COLUMNS.
    """
    ctx = build_claim_context(row)

    if strategy == "two_stage":
        # Stage 1: Extract the claim from the conversation
        extracted = await extract_claim(ctx)

        # Optionally refine evidence requirements using extracted issue family
        issue_family = extracted.get("issue_family", "")
        if issue_family:
            from context_loader import get_evidence_requirements
            ctx.evidence_requirements = get_evidence_requirements(
                ctx.claim_object, issue_family
            )

        # Stage 2: Verify images against the extracted claim
        verification = await verify_images(ctx, extracted)

        # Stage 3: Final decision
        output_row = await decide(ctx, extracted, verification)

    elif strategy == "single_pass":
        # In single_pass, extraction + verification + draft decision come from
        # one multimodal LLM call. The final decide() step remains pure Python.
        extracted, verification = await single_pass(ctx)

        issue_family = extracted.get("issue_family", "")
        if issue_family:
            from context_loader import get_evidence_requirements
            ctx.evidence_requirements = get_evidence_requirements(
                ctx.claim_object, issue_family
            )

        output_row = await decide(ctx, extracted, verification)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return output_row


# ── Main loop ─────────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> None:
    """Load input CSV, process each row, write output CSV."""
    print(f"Loading input from {args.input} …")
    df = load_claims(args.input)

    if args.limit is not None:
        df = df.head(args.limit)
        print(f"  (limited to first {args.limit} rows)")

    print(f"Processing {len(df)} claim(s) with strategy={args.strategy!r} …")

    results: list[dict] = []
    t0 = time.perf_counter()

    for idx, row in df.iterrows():
        try:
            result = await process_row(row, args.strategy)
            results.append(result)
        except NotImplementedError as exc:
            # Expected during scaffolding – print and skip
            print(f"  Row {idx}: STUB – {exc}")
            # Append a skeleton row so the output CSV shape is correct
            results.append(_skeleton_row(row))
        except Exception as exc:
            print(f"  Row {idx}: ERROR – {exc}", file=sys.stderr)
            results.append(_skeleton_row(row))

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s. Writing output to {args.output} …")

    out_df = pd.DataFrame(results, columns=config.OUTPUT_COLUMNS)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


def _skeleton_row(row: pd.Series) -> dict:
    """Return a placeholder output row when a stage is stubbed / errored."""
    return {
        "user_id": str(row.get("user_id", "")),
        "image_paths": str(row.get("image_paths", "")),
        "user_claim": str(row.get("user_claim", "")),
        "claim_object": str(row.get("claim_object", "")),
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "Pipeline stage not implemented yet.",
        "risk_flags": "none",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Pipeline stage not implemented yet.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


# ── Entry-point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
