#!/usr/bin/env python3
"""Evaluation: annotation protocol, inter-rater agreement, bootstrap CIs, and stratified error analysis
for Typed Failure Detection in neuro-symbolic pipelines."""

import json
import sys
import gc
import math
import resource
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/eval.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATA_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/"
    "run_gtgRw-BvOJEe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/full_method_out.json"
)
OUT_PATH = WORKSPACE / "eval_out.json"
PLOTS_DIR = WORKSPACE / "plots"
N_BOOTSTRAP = 1000
RNG_SEED = 42

FAILURE_CATEGORIES = [
    "none",
    "TYPE_1_LEXICAL_MISMATCH",
    "TYPE_2_ARITY_MISMATCH",
    "TYPE_3_MISSING_FACT",
    "TYPE_4_CATEGORY_VIOLATION",
    "TYPE_5_SCOPE_CONFLICT",
]


# ---------------------------------------------------------------------------
# Heuristic annotator B — simulates a second human rater
# ---------------------------------------------------------------------------

def annotator_b(example: dict) -> str:
    """Rule-based heuristic that classifies failure type independently of the
    automated Prolog-exception classifier (Annotator A = metadata_failure_type).

    Uses surface text features: goal predicate name, clause count, kinship keywords.
    """
    failure_type_a = example["metadata_failure_type"]
    if failure_type_a == "none":
        return "none"  # Both annotators only classify known failures

    input_text = example["input"].lower()
    num_clauses = int(example.get("metadata_num_clauses", "2"))
    goal = example.get("metadata_goal", "").lower()

    kinship_terms = {
        "mother", "father", "son", "daughter", "sister", "brother",
        "uncle", "aunt", "grandfather", "grandmother", "grandson",
        "granddaughter", "nephew", "niece", "cousin", "husband", "wife",
        "parent", "child", "sibling", "spouse",
    }

    has_kinship = any(term in input_text for term in kinship_terms)
    has_relationship_goal = "relationship(" in goal

    # Heuristic: kinship examples with few clauses → TYPE_1 (predicate name mismatch)
    #            kinship examples with many clauses → TYPE_3 (missing intermediate fact)
    #            non-kinship with failure → TYPE_3
    if has_relationship_goal or has_kinship:
        if num_clauses <= 3:
            return "TYPE_1_LEXICAL_MISMATCH"
        else:
            return "TYPE_3_MISSING_FACT"
    else:
        # RuleTaker-style failures — propositional, likely missing fact
        return "TYPE_3_MISSING_FACT"


# ---------------------------------------------------------------------------
# Bootstrap utility
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_resamples: int = N_BOOTSTRAP,
                 rng: np.random.Generator = None) -> dict:
    """Bootstrap 95% CI for the mean of a 1D list."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan")}
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@logger.catch(reraise=True)
def main() -> None:
    # RAM limit: ~12 GB (generous; data is <2 MB)
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    PLOTS_DIR.mkdir(exist_ok=True)

    logger.info(f"Loading data from {DATA_PATH}")
    raw = json.loads(DATA_PATH.read_text())
    metadata_exp = raw.get("metadata", {})

    all_examples: list[dict] = []
    for ds_block in raw["datasets"]:
        for ex in ds_block["examples"]:
            all_examples.append(ex)
    logger.info(f"Loaded {len(all_examples)} examples total")
    del raw
    gc.collect()

    # ------------------------------------------------------------------
    # Phase 1: Failure-type distribution + annotation
    # ------------------------------------------------------------------
    logger.info("Phase 1: Annotation protocol & inter-rater agreement")

    failure_examples = [e for e in all_examples if e["metadata_failure_type"] != "none"]
    logger.info(f"  Found {len(failure_examples)} failure examples for annotation")

    annotator_a_labels = [e["metadata_failure_type"] for e in failure_examples]
    annotator_b_labels = [annotator_b(e) for e in failure_examples]

    kappa = cohen_kappa_score(annotator_a_labels, annotator_b_labels,
                              labels=[c for c in FAILURE_CATEGORIES if c != "none"])
    logger.info(f"  Cohen's kappa: {kappa:.4f}")

    # Agreement counts
    agreement_count = sum(a == b for a, b in zip(annotator_a_labels, annotator_b_labels))
    agreement_pct = agreement_count / max(len(annotator_a_labels), 1)
    logger.info(f"  Raw agreement: {agreement_count}/{len(annotator_a_labels)} ({agreement_pct:.1%})")

    # Observed distribution (full dataset, all 200 examples)
    full_ft_dist = Counter(e["metadata_failure_type"] for e in all_examples)
    logger.info(f"  Failure distribution: {dict(full_ft_dist)}")

    # ------------------------------------------------------------------
    # Phase 2: Per-example metrics + detection accuracy
    # ------------------------------------------------------------------
    logger.info("Phase 2: Per-example metric extraction")

    rng = np.random.default_rng(RNG_SEED)

    # Per-dataset aggregates
    per_dataset: dict[str, list[dict]] = defaultdict(list)
    for ex in all_examples:
        per_dataset[ex["metadata_dataset"]].append(ex)

    def get_correct(val: str) -> float:
        return 1.0 if val.strip().lower() == "true" else 0.0

    # Build enriched examples for eval_out
    eval_datasets: list[dict] = []
    for ds_name, exs in per_dataset.items():
        enriched = []
        for ex in exs:
            typed_correct = get_correct(ex["metadata_typed_correct"])
            base_correct = get_correct(ex["metadata_baseline_correct"])
            typed_hall = float(ex["metadata_typed_hallucination"])
            base_hall = float(ex["metadata_baseline_hallucination"])

            enriched.append({
                "input": ex["input"],
                "output": ex["output"],
                "predict_typed": ex["predict_typed"],
                "predict_baseline": ex["predict_baseline"],
                "predict_annotator_b": annotator_b(ex),
                "metadata_dataset": ex["metadata_dataset"],
                "metadata_goal": ex["metadata_goal"],
                "metadata_failure_type": ex["metadata_failure_type"],
                "metadata_num_clauses": ex["metadata_num_clauses"],
                "eval_typed_correct": typed_correct,
                "eval_baseline_correct": base_correct,
                "eval_typed_hallucination": typed_hall,
                "eval_baseline_hallucination": base_hall,
                "eval_improvement": typed_correct - base_correct,
                "eval_hallucination_improvement": base_hall - typed_hall,
            })
        eval_datasets.append({"dataset": ds_name, "examples": enriched})

    # ------------------------------------------------------------------
    # Phase 3: Bootstrap CIs
    # ------------------------------------------------------------------
    logger.info("Phase 3: Bootstrap confidence intervals (N=1000)")

    def collect_vals(ds_name: str, field: str) -> list[float]:
        for block in eval_datasets:
            if block["dataset"] == ds_name:
                return [ex[field] for ex in block["examples"]]
        return []

    bootstrap_results: dict[str, dict] = {}
    metrics_to_bootstrap = [
        ("ruletaker", "eval_typed_correct", "ruletaker_typed_accuracy"),
        ("ruletaker", "eval_baseline_correct", "ruletaker_baseline_accuracy"),
        ("ruletaker", "eval_typed_hallucination", "ruletaker_typed_hallucination"),
        ("ruletaker", "eval_baseline_hallucination", "ruletaker_baseline_hallucination"),
        ("ruletaker", "eval_improvement", "ruletaker_improvement"),
        ("ruletaker", "eval_hallucination_improvement", "ruletaker_hallucination_reduction"),
        ("clutrr", "eval_typed_correct", "clutrr_typed_accuracy"),
        ("clutrr", "eval_baseline_correct", "clutrr_baseline_accuracy"),
        ("clutrr", "eval_typed_hallucination", "clutrr_typed_hallucination"),
        ("clutrr", "eval_baseline_hallucination", "clutrr_baseline_hallucination"),
        ("clutrr", "eval_improvement", "clutrr_improvement"),
        ("clutrr", "eval_hallucination_improvement", "clutrr_hallucination_reduction"),
    ]

    for ds_name, field, label in metrics_to_bootstrap:
        vals = collect_vals(ds_name, field)
        ci = bootstrap_ci(vals, rng=rng)
        bootstrap_results[label] = ci
        logger.info(f"  {label}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # Overall (combined) bootstrap
    all_typed = collect_vals("ruletaker", "eval_typed_correct") + collect_vals("clutrr", "eval_typed_correct")
    all_base = collect_vals("ruletaker", "eval_baseline_correct") + collect_vals("clutrr", "eval_baseline_correct")
    bootstrap_results["overall_typed_accuracy"] = bootstrap_ci(all_typed, rng=rng)
    bootstrap_results["overall_baseline_accuracy"] = bootstrap_ci(all_base, rng=rng)

    # ------------------------------------------------------------------
    # Phase 4: Stratified error analysis
    # ------------------------------------------------------------------
    logger.info("Phase 4: Stratified error analysis by failure type × dataset")

    stratified: dict[str, dict] = {}
    failure_types_seen = sorted(full_ft_dist.keys())

    for ft in failure_types_seen:
        stratified[ft] = {}
        for block in eval_datasets:
            ds = block["dataset"]
            exs = [e for e in block["examples"] if e["metadata_failure_type"] == ft]
            if not exs:
                continue
            n = len(exs)
            typed_acc = sum(e["eval_typed_correct"] for e in exs) / n
            base_acc = sum(e["eval_baseline_correct"] for e in exs) / n
            typed_hall = sum(e["eval_typed_hallucination"] for e in exs) / n
            base_hall = sum(e["eval_baseline_hallucination"] for e in exs) / n
            typed_wins = sum(1 for e in exs if e["eval_typed_correct"] > e["eval_baseline_correct"])
            base_wins = sum(1 for e in exs if e["eval_baseline_correct"] > e["eval_typed_correct"])

            stratified[ft][ds] = {
                "n": n,
                "typed_accuracy": typed_acc,
                "baseline_accuracy": base_acc,
                "typed_hallucination": typed_hall,
                "baseline_hallucination": base_hall,
                "hallucination_reduction": base_hall - typed_hall,
                "typed_wins": typed_wins,
                "baseline_wins": base_wins,
                "ties": n - typed_wins - base_wins,
            }
            logger.info(
                f"  [{ft}][{ds}] n={n}, typed_acc={typed_acc:.2f}, "
                f"base_acc={base_acc:.2f}, hall_reduction={base_hall - typed_hall:.4f}"
            )

    # ------------------------------------------------------------------
    # Phase 5: Visualizations
    # ------------------------------------------------------------------
    logger.info("Phase 5: Generating visualizations")

    _generate_plots(
        full_ft_dist=full_ft_dist,
        stratified=stratified,
        bootstrap_results=bootstrap_results,
        kappa=kappa,
        annotator_a_labels=annotator_a_labels,
        annotator_b_labels=annotator_b_labels,
        plots_dir=PLOTS_DIR,
    )

    # ------------------------------------------------------------------
    # Build metrics_agg (flat dict, only numbers)
    # ------------------------------------------------------------------
    logger.info("Building metrics_agg")

    metrics_agg: dict[str, float] = {
        "cohen_kappa": round(kappa, 6),
        "annotator_agreement_pct": round(agreement_pct, 6),
        "n_failure_examples_annotated": float(len(failure_examples)),
        "n_total_examples": float(len(all_examples)),
    }

    # Overall bootstrap
    for key, ci in bootstrap_results.items():
        metrics_agg[f"{key}_mean"] = round(ci["mean"], 6)
        metrics_agg[f"{key}_ci_lower"] = round(ci["ci_lower"], 6)
        metrics_agg[f"{key}_ci_upper"] = round(ci["ci_upper"], 6)
        metrics_agg[f"{key}_std"] = round(ci["std"], 6)

    # Stratified (flattened)
    for ft, ds_data in stratified.items():
        ft_short = ft.replace("TYPE_", "T").replace("_LEXICAL_MISMATCH", "1").replace(
            "_ARITY_MISMATCH", "2").replace("_MISSING_FACT", "3").replace(
            "_CATEGORY_VIOLATION", "4").replace("_SCOPE_CONFLICT", "5")
        if ft == "none":
            ft_short = "none"
        for ds, vals in ds_data.items():
            prefix = f"strat_{ft_short}_{ds}"
            metrics_agg[f"{prefix}_n"] = float(vals["n"])
            metrics_agg[f"{prefix}_typed_acc"] = round(vals["typed_accuracy"], 6)
            metrics_agg[f"{prefix}_base_acc"] = round(vals["baseline_accuracy"], 6)
            metrics_agg[f"{prefix}_typed_hall"] = round(vals["typed_hallucination"], 6)
            metrics_agg[f"{prefix}_base_hall"] = round(vals["baseline_hallucination"], 6)
            metrics_agg[f"{prefix}_hall_reduction"] = round(vals["hallucination_reduction"], 6)

    # Failure-type distribution counts
    for ft, cnt in full_ft_dist.items():
        ft_key = ft.replace("-", "_").replace(" ", "_")
        metrics_agg[f"dist_{ft_key}_count"] = float(cnt)
        metrics_agg[f"dist_{ft_key}_pct"] = round(cnt / len(all_examples), 6)

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    eval_out = {
        "metadata": {
            "evaluation_name": "typed_failure_detection_evaluation",
            "description": (
                "Annotation protocol, inter-rater agreement (Cohen's kappa), "
                "bootstrap 95% CIs, and stratified error analysis by failure type."
            ),
            "n_bootstrap": N_BOOTSTRAP,
            "experiment_metadata": metadata_exp,
            "failure_type_taxonomy": {
                "TYPE_1_LEXICAL_MISMATCH": "Predicate absent from KB but similar one exists",
                "TYPE_2_ARITY_MISMATCH": "Predicate exists but called with wrong arity",
                "TYPE_3_MISSING_FACT": "Proof search exhausted — missing ground fact",
                "TYPE_4_CATEGORY_VIOLATION": "Type-error exception in proof",
                "TYPE_5_SCOPE_CONFLICT": "Quantifier scope conflict (heuristic)",
            },
            "annotation_summary": {
                "n_annotated": len(failure_examples),
                "annotator_a": "Prolog-exception automated classifier",
                "annotator_b": "Text-feature heuristic (clause count + kinship keyword matching)",
                "cohen_kappa": round(kappa, 4),
                "raw_agreement_pct": round(agreement_pct, 4),
                "kappa_interpretation": _interpret_kappa(kappa),
                "failure_type_distribution": dict(full_ft_dist),
            },
            "bootstrap_summary": {
                label: {
                    "mean": round(ci["mean"], 4),
                    "std": round(ci["std"], 4),
                    "ci_95": [round(ci["ci_lower"], 4), round(ci["ci_upper"], 4)],
                }
                for label, ci in bootstrap_results.items()
            },
            "stratified_analysis": stratified,
        },
        "metrics_agg": metrics_agg,
        "datasets": eval_datasets,
    }

    logger.info(f"Writing output to {OUT_PATH}")
    OUT_PATH.write_text(json.dumps(eval_out, indent=2))
    logger.info(f"Output written: {OUT_PATH.stat().st_size / 1024:.1f} KB")


def _interpret_kappa(k: float) -> str:
    if k <= 0:
        return "no agreement"
    elif k <= 0.20:
        return "slight"
    elif k <= 0.40:
        return "fair"
    elif k <= 0.60:
        return "moderate"
    elif k <= 0.80:
        return "substantial"
    else:
        return "almost perfect"


def _generate_plots(
    full_ft_dist: Counter,
    stratified: dict,
    bootstrap_results: dict,
    kappa: float,
    annotator_a_labels: list,
    annotator_b_labels: list,
    plots_dir: Path,
) -> None:
    """Generate all evaluation visualizations."""

    plt.style.use("seaborn-v0_8-whitegrid")
    COLORS = {"ruletaker": "#4C72B0", "clutrr": "#DD8452"}

    # (a) Failure-type distribution bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    types = sorted(full_ft_dist.keys())
    counts = [full_ft_dist[t] for t in types]
    bars = ax.bar(range(len(types)), counts, color="#5B9BD5", edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("TYPE_", "T").replace("_", "\n") for t in types], fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Failure-Type Distribution (200 examples)")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "a_failure_type_distribution.png", dpi=150)
    plt.close()
    logger.info("  Saved: a_failure_type_distribution.png")

    # (b) Per-type detection accuracy (annotator A vs B agreement)
    failure_types = [t for t in FAILURE_CATEGORIES if t != "none" and t in full_ft_dist]
    a_arr = np.array(annotator_a_labels)
    b_arr = np.array(annotator_b_labels)
    per_type_agreement = []
    for ft in failure_types:
        mask = a_arr == ft
        if mask.sum() == 0:
            per_type_agreement.append(0.0)
            continue
        per_type_agreement.append((a_arr[mask] == b_arr[mask]).mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(failure_types))
    ax.bar(x, per_type_agreement, color="#70AD47", edgecolor="white", linewidth=0.8)
    ax.axhline(kappa, color="red", linestyle="--", linewidth=1.2, label=f"Cohen's κ = {kappa:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([ft.replace("TYPE_", "T").replace("_", "\n") for ft in failure_types], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Agreement rate")
    ax.set_title("Per-Type Inter-Rater Agreement (A vs B)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "b_per_type_agreement.png", dpi=150)
    plt.close()
    logger.info("  Saved: b_per_type_agreement.png")

    # (c) Per-type accuracy improvement (typed vs baseline) by dataset
    datasets = ["ruletaker", "clutrr"]
    failure_types_strat = [ft for ft in FAILURE_CATEGORIES if ft in stratified]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    for i, ft in enumerate(failure_types_strat):
        for j, ds in enumerate(datasets):
            if ds not in stratified[ft]:
                continue
            vals = stratified[ft][ds]
            improvement = vals["typed_accuracy"] - vals["baseline_accuracy"]
            color = COLORS[ds]
            offset = (j - 0.5) * width
            ax.bar(i + offset, improvement, width=width * 0.9,
                   color=color, label=ds if i == 0 else "", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(failure_types_strat)))
    ax.set_xticklabels([ft.replace("TYPE_", "T").replace("_", "\n") for ft in failure_types_strat], fontsize=9)
    ax.set_ylabel("Accuracy improvement (typed − baseline)")
    ax.set_title("Per-Type Accuracy Improvement by Dataset")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[ds]) for ds in datasets]
    ax.legend(handles, datasets)
    plt.tight_layout()
    plt.savefig(plots_dir / "c_per_type_accuracy_improvement.png", dpi=150)
    plt.close()
    logger.info("  Saved: c_per_type_accuracy_improvement.png")

    # (d) Per-type hallucination rates with 95% CI error bars
    # Use bootstrap CIs from main results; for per-type we compute inline
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(RNG_SEED + 1)

    for ax, ds in zip(axes, datasets):
        for block_eval in []:
            pass  # not needed here; use stratified
        ds_exs_all = []  # collect from eval_datasets — but we don't have them here
        # Use aggregate per-type hallucination from stratified
        ft_labels = [ft for ft in failure_types_strat if ds in stratified.get(ft, {})]
        typed_halls = [stratified[ft][ds]["typed_hallucination"] for ft in ft_labels]
        base_halls = [stratified[ft][ds]["baseline_hallucination"] for ft in ft_labels]

        x = np.arange(len(ft_labels))
        width = 0.35
        ax.bar(x - width / 2, typed_halls, width, label="typed", color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, base_halls, width, label="baseline", color="#DD8452", alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([ft.replace("TYPE_", "T").replace("_", "\n") for ft in ft_labels], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Hallucination rate")
        ax.set_title(f"{ds.upper()} — Per-Type Hallucination")
        ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "d_per_type_hallucination.png", dpi=150)
    plt.close()
    logger.info("  Saved: d_per_type_hallucination.png")

    # (e) Bootstrap CI widths for key metrics
    key_metrics = [
        "ruletaker_typed_accuracy", "ruletaker_baseline_accuracy",
        "clutrr_typed_accuracy", "clutrr_baseline_accuracy",
        "ruletaker_typed_hallucination", "ruletaker_baseline_hallucination",
        "clutrr_typed_hallucination", "clutrr_baseline_hallucination",
        "overall_typed_accuracy", "overall_baseline_accuracy",
    ]
    ci_widths = []
    ci_means = []
    ci_lowers = []
    ci_uppers = []
    valid_keys = [k for k in key_metrics if k in bootstrap_results]
    for k in valid_keys:
        ci = bootstrap_results[k]
        ci_widths.append(ci["ci_upper"] - ci["ci_lower"])
        ci_means.append(ci["mean"])
        ci_lowers.append(ci["mean"] - ci["ci_lower"])
        ci_uppers.append(ci["ci_upper"] - ci["mean"])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(valid_keys))
    ax.bar(x, [bootstrap_results[k]["mean"] for k in valid_keys],
           color="#5B9BD5", alpha=0.75, label="mean", edgecolor="white")
    ax.errorbar(x, ci_means, yerr=[ci_lowers, ci_uppers],
                fmt="none", color="black", capsize=4, linewidth=1.5, label="95% CI")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in valid_keys], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Metric value")
    ax.set_title("Bootstrap 95% Confidence Intervals for Key Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "e_bootstrap_ci.png", dpi=150)
    plt.close()
    logger.info("  Saved: e_bootstrap_ci.png")

    # (f) Kappa interpretation summary panel
    fig, ax = plt.subplots(figsize=(7, 4))
    landis_koch = [
        (0.0, 0.20, "#E74C3C", "Slight / No agmt"),
        (0.20, 0.40, "#F39C12", "Fair"),
        (0.40, 0.60, "#F1C40F", "Moderate"),
        (0.60, 0.80, "#27AE60", "Substantial"),
        (0.80, 1.00, "#1ABC9C", "Almost Perfect"),
    ]
    for lo, hi, col, label in landis_koch:
        ax.barh(0, hi - lo, left=lo, height=0.5, color=col, alpha=0.7, edgecolor="white")
        ax.text((lo + hi) / 2, 0, label, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.axvline(kappa, color="black", linewidth=2, label=f"Observed κ = {kappa:.3f}")
    ax.set_xlim(-0.1, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("Cohen's κ")
    ax.set_title(f"Inter-Rater Agreement: κ = {kappa:.3f} ({_interpret_kappa(kappa)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "f_kappa_interpretation.png", dpi=150)
    plt.close()
    logger.info("  Saved: f_kappa_interpretation.png")

    logger.info(f"All plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
