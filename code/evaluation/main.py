#!/usr/bin/env python3
"""
evaluation/main.py – run and evaluate sample-set predictions.

By default this script runs both strategies against dataset/sample_claims.csv,
writes per-strategy prediction CSVs under code/evaluation/, scores the exact
match fields requested in problem_statement.md, and writes
code/evaluation/evaluation_report.md.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import pathlib
import sys
from typing import Any

# Add parent dir to sys.path so we can import project modules
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

import config
import main as pipeline_main


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run both sample strategies and evaluate predictions against "
            "sample_claims.csv gold labels."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional existing predictions CSV to score. When omitted, the "
            "script runs the configured strategies on sample_claims.csv."
        ),
    )
    parser.add_argument(
        "--gold",
        type=pathlib.Path,
        default=config.DATASET_DIR / "sample_claims.csv",
        help="Path to the gold-standard CSV (default: dataset/sample_claims.csv).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["two_stage", "single_pass"],
        default=["two_stage", "single_pass"],
        help="Strategies to run when --predictions is not supplied.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Informational label when scoring --predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run/evaluate only the first N rows.",
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=config._CODE_DIR / "evaluation" / "evaluation_report.md",
        help="Path for the Markdown evaluation report.",
    )
    return parser.parse_args(argv)


# ── Scoring columns ──────────────────────────────────────────────────────────

EXACT_MATCH_COLUMNS = [
    "claim_status",
    "issue_type",
    "object_part",
    "evidence_standard_met",
    "valid_image",
]

SET_OVERLAP_COLUMNS = [
    "risk_flags",
    "supporting_image_ids",
]


def _norm(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return text


def _split_set(value: Any) -> set[str]:
    text = _norm(value)
    if not text or text == "none":
        return set()
    return {part.strip() for part in text.split(";") if part.strip()}


def score_exact_match(pred: pd.DataFrame, gold: pd.DataFrame, column: str) -> float:
    """Compute exact-match accuracy for a single column."""
    if column not in pred.columns or column not in gold.columns or len(gold) == 0:
        return 0.0
    pred_values = pred[column].map(_norm)
    gold_values = gold[column].map(_norm)
    return float((pred_values == gold_values).mean())


def score_set_overlap(pred: pd.DataFrame, gold: pd.DataFrame, column: str) -> float:
    """Compute average set-overlap F1 for a semicolon-separated column."""
    if column not in pred.columns or column not in gold.columns or len(gold) == 0:
        return 0.0

    scores: list[float] = []
    for pred_value, gold_value in zip(pred[column], gold[column], strict=False):
        pred_set = _split_set(pred_value)
        gold_set = _split_set(gold_value)
        if not pred_set and not gold_set:
            scores.append(1.0)
            continue
        if not pred_set or not gold_set:
            scores.append(0.0)
            continue
        overlap = len(pred_set & gold_set)
        precision = overlap / len(pred_set)
        recall = overlap / len(gold_set)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _align(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align by row order; the pipeline preserves sample_claims.csv order."""
    rows = min(len(pred_df), len(gold_df))
    return pred_df.head(rows).reset_index(drop=True), gold_df.head(rows).reset_index(drop=True)


async def _run_strategy(
    strategy: str,
    gold_path: pathlib.Path,
    limit: int | None,
) -> pathlib.Path:
    pred_path = config._CODE_DIR / "evaluation" / f"sample_predictions_{strategy}.csv"
    args = argparse.Namespace(
        input=gold_path,
        output=pred_path,
        strategy=strategy,
        limit=limit,
    )
    await pipeline_main.run(args)
    return pred_path


def _score_predictions(
    strategy: str,
    pred_path: pathlib.Path,
    gold_df: pd.DataFrame,
    limit: int | None,
) -> dict[str, Any]:
    pred_df = pd.read_csv(pred_path)
    if limit is not None:
        pred_df = pred_df.head(limit)
    pred_df, aligned_gold = _align(pred_df, gold_df)

    exact_scores = {
        column: score_exact_match(pred_df, aligned_gold, column)
        for column in EXACT_MATCH_COLUMNS
    }
    set_scores = {
        column: score_set_overlap(pred_df, aligned_gold, column)
        for column in SET_OVERLAP_COLUMNS
    }
    mean_exact = (
        sum(exact_scores.values()) / len(exact_scores) if exact_scores else 0.0
    )
    return {
        "strategy": strategy,
        "path": pred_path,
        "rows": len(pred_df),
        "exact_scores": exact_scores,
        "set_scores": set_scores,
        "mean_exact": mean_exact,
    }


def _read_call_log() -> pd.DataFrame:
    if not config.CALL_LOG_PATH.is_file():
        return pd.DataFrame(columns=[
            "timestamp",
            "stage",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "latency_ms",
            "cache_hit",
            "image_count",
        ])
    return pd.read_csv(config.CALL_LOG_PATH)


def _strategy_for_stage(stage: str) -> str:
    if stage in {"claim_extraction", "vision_verification"}:
        return "two_stage"
    if stage == "single_pass":
        return "single_pass"
    return "other"


def _operational_rows(call_log: pd.DataFrame) -> list[dict[str, Any]]:
    if call_log.empty:
        return []

    log = call_log.copy()
    for column in ["prompt_tokens", "completion_tokens", "latency_ms", "image_count"]:
        if column not in log.columns:
            log[column] = 0
        log[column] = pd.to_numeric(log[column], errors="coerce").fillna(0)
    if "cache_hit" not in log.columns:
        log["cache_hit"] = False
    log["cache_hit"] = log["cache_hit"].astype(str).str.lower().isin({"true", "1"})
    log["strategy"] = log["stage"].map(_strategy_for_stage)

    rows: list[dict[str, Any]] = []
    group_cols = ["strategy", "stage", "model"]
    for (strategy, stage, model), group in log.groupby(group_cols, dropna=False):
        prompt_tokens = int(group["prompt_tokens"].sum())
        completion_tokens = int(group["completion_tokens"].sum())
        est_cost = (
            prompt_tokens * config.LLM_INPUT_COST_PER_1M
            + completion_tokens * config.LLM_OUTPUT_COST_PER_1M
        ) / 1_000_000
        rows.append({
            "strategy": strategy,
            "stage": stage,
            "model": model,
            "calls": int(len(group)),
            "billable_calls": int((~group["cache_hit"]).sum()),
            "cache_hits": int(group["cache_hit"].sum()),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "images": int(group["image_count"].sum()),
            "avg_latency_ms": float(group["latency_ms"].mean()),
            "est_cost_usd": est_cost,
        })
    return rows


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _write_report(
    report_path: pathlib.Path,
    gold_path: pathlib.Path,
    results: list[dict[str, Any]],
    operational: list[dict[str, Any]],
    limit: int | None,
) -> None:
    result_rows = []
    for result in results:
        scores = result["exact_scores"]
        result_rows.append([
            result["strategy"],
            result["rows"],
            _pct(scores["claim_status"]),
            _pct(scores["issue_type"]),
            _pct(scores["object_part"]),
            _pct(scores["evidence_standard_met"]),
            _pct(scores["valid_image"]),
            _pct(result["mean_exact"]),
        ])

    set_rows = []
    for result in results:
        scores = result["set_scores"]
        set_rows.append([
            result["strategy"],
            _pct(scores["risk_flags"]),
            _pct(scores["supporting_image_ids"]),
            result["path"],
        ])

    op_rows = [
        [
            row["strategy"],
            row["stage"],
            row["model"],
            row["calls"],
            row["billable_calls"],
            row["cache_hits"],
            row["prompt_tokens"],
            row["completion_tokens"],
            row["images"],
            f"{row['avg_latency_ms']:.1f}",
            f"${row['est_cost_usd']:.6f}",
        ]
        for row in operational
    ]

    if not op_rows:
        op_rows.append(["n/a", "n/a", "n/a", 0, 0, 0, 0, 0, 0, "0.0", "$0.000000"])

    report = [
        "# Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Gold file: `{gold_path}`",
        f"Limit: `{limit}`" if limit is not None else "Limit: full sample set",
        "",
        "## Exact-Match Results",
        "",
        _markdown_table(
            [
                "strategy",
                "rows",
                "claim_status",
                "issue_type",
                "object_part",
                "evidence_standard_met",
                "valid_image",
                "mean_exact",
            ],
            result_rows,
        ),
        "",
        "## Set-Overlap Diagnostics",
        "",
        _markdown_table(
            ["strategy", "risk_flags_f1", "supporting_image_ids_f1", "predictions"],
            set_rows,
        ),
        "",
        "## Operational Analysis",
        "",
        _markdown_table(
            [
                "strategy",
                "stage",
                "model",
                "calls",
                "billable_calls",
                "cache_hits",
                "prompt_tokens",
                "completion_tokens",
                "images",
                "avg_latency_ms",
                "est_cost_usd",
            ],
            op_rows,
        ),
        "",
        (
            "Cost estimate uses "
            f"${config.LLM_INPUT_COST_PER_1M:.4f}/1M input tokens and "
            f"${config.LLM_OUTPUT_COST_PER_1M:.4f}/1M output tokens. "
            "Override LLM_INPUT_COST_PER_1M and LLM_OUTPUT_COST_PER_1M if the "
            "selected provider/model uses different pricing."
        ),
        "",
        (
            "TPM/RPM notes: the two_stage strategy makes one text call and one "
            "vision call per claim; single_pass makes one vision call per claim. "
            "llm_client uses an asyncio semaphore, exponential retry for 429/5xx "
            "style failures, SHA-256 disk caching keyed by prompt and image "
            "content, and logs cache hits so repeated evaluation runs avoid "
            "unnecessary billable model calls."
        ),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> None:
    gold_df = pd.read_csv(args.gold)
    if args.limit is not None:
        gold_df = gold_df.head(args.limit)

    if args.predictions is not None:
        if not args.predictions.is_file():
            print(f"ERROR: Predictions file not found: {args.predictions}", file=sys.stderr)
            sys.exit(1)
        strategy = args.strategy or "predictions"
        results = [_score_predictions(strategy, args.predictions, gold_df, args.limit)]
    else:
        prediction_paths: dict[str, pathlib.Path] = {}

        async def _run_all() -> None:
            for strategy in args.strategies:
                prediction_paths[strategy] = await _run_strategy(
                    strategy=strategy,
                    gold_path=args.gold,
                    limit=args.limit,
                )

        asyncio.run(_run_all())
        results = [
            _score_predictions(strategy, path, gold_df, args.limit)
            for strategy, path in prediction_paths.items()
        ]

    operational = _operational_rows(_read_call_log())
    _write_report(args.report, args.gold, results, operational, args.limit)

    print(f"Gold: {args.gold}")
    print(f"Report: {args.report}")
    for result in results:
        print(
            f"{result['strategy']}: rows={result['rows']} "
            f"mean_exact={_pct(result['mean_exact'])}"
        )


# ── Entry-point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
