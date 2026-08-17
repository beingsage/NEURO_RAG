#!/usr/bin/env python3
"""
Sweep DG->CA3 bridge plasticity while holding the rest of the model fixed.

This runner varies only `dg_bridge_lr` so the false-retrieval effect can be
tested cleanly across many seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import _reference_corpus
from spiking_multimodal_memory import MultiModalMemory


RUN_ROOT = Path("outputs/bridge_lr_sweeps")
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_BRIDGE_LRS = (0.0, 0.01, 0.02, 0.04, 0.06)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def _bootstrap_ci(values_a: Sequence[float], values_b: Sequence[float], *, n_resamples: int = 5000,
                  seed: int = 0) -> Tuple[float, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = a - b
    if diff.size == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_resamples):
        idx = rng.randint(0, diff.size, size=diff.size)
        boot.append(float(np.mean(diff[idx])))
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def _paired_stats(values_a: Sequence[float], values_b: Sequence[float], *, seed: int = 0) -> Dict[str, Any]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size != b.size:
        raise ValueError("paired stats require equal-length inputs")
    if a.size == 0:
        return {
            "mean_delta": 0.0,
            "paired_t_stat": 0.0,
            "paired_t_p": 1.0,
            "bootstrap_ci95": [0.0, 0.0],
        }

    t_stat, p_value = stats.ttest_rel(a, b, nan_policy="omit")
    if not np.isfinite(t_stat):
        t_stat = 0.0
    if not np.isfinite(p_value):
        p_value = 1.0
    ci_low, ci_high = _bootstrap_ci(a, b, seed=seed)
    return {
        "mean_delta": float(np.mean(a - b)),
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(p_value),
        "bootstrap_ci95": [ci_low, ci_high],
    }


def _parse_floats(raw: str) -> List[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_ints(raw: str) -> List[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _partial_sql_row(mem: MultiModalMemory, row: Dict[str, Any], fraction: float) -> Dict[str, Any]:
    return mem._partial_sql_row(row, fraction)


def _relation_accuracy_under_partial_sql(mem: MultiModalMemory, cue_fraction: float, duration: float) -> Dict[str, Any]:
    per_episode: List[float] = []
    per_prediction: List[str] = []

    for record in mem.episode_records:
        partial_sql = _partial_sql_row(mem, record.sql_row, cue_fraction)
        retrieved = mem.retrieve(sql_cue=partial_sql, duration=duration)
        graph_metrics = mem.compute_graph_retrieval_accuracy(retrieved, record.graph_edge)
        per_episode.append(float(graph_metrics.get("relation_accuracy", 0.0)))
        per_prediction.append(str(graph_metrics.get("relation_prediction", None)))

    return {
        "per_episode": per_episode,
        "mean": float(np.mean(per_episode)) if per_episode else 0.0,
        "std": float(np.std(per_episode, ddof=1)) if len(per_episode) > 1 else 0.0,
        "predictions": per_prediction,
    }


@dataclass(frozen=True)
class SweepConfig:
    cue_fraction: float = 0.4
    duration_ms: float = 50.0
    use_text: bool = True
    ca3_exc: int = 240
    ca3_inh: int = 60
    dg_bridge_fanout: int = 12
    dg_output_dim: int = 2000
    dg_target_sparsity: float = 0.01
    ca1_n: int = 320
    ca1_train_epochs: int = 12
    ca1_train_lr: float = 0.03
    ca1_relation_lr: float = 0.05
    ca1_relation_train_cue_fraction: float = 0.4


def _build_memory(seed: int, bridge_lr: float, cfg: SweepConfig) -> MultiModalMemory:
    return MultiModalMemory(
        use_text=cfg.use_text,
        seed=seed,
        ca3_exc=cfg.ca3_exc,
        ca3_inh=cfg.ca3_inh,
        dg_bridge_fanout=cfg.dg_bridge_fanout,
        dg_bridge_lr=bridge_lr,
        dg_output_dim=cfg.dg_output_dim,
        dg_target_sparsity=cfg.dg_target_sparsity,
        ca1_n=cfg.ca1_n,
        ca1_train_epochs=cfg.ca1_train_epochs,
        ca1_train_lr=cfg.ca1_train_lr,
        ca1_relation_lr=cfg.ca1_relation_lr,
        ca1_relation_train_cue_fraction=cfg.ca1_relation_train_cue_fraction,
    )


def _evaluate_seed(seed: int, bridge_lr: float, cfg: SweepConfig) -> Dict[str, Any]:
    mem = _build_memory(seed, bridge_lr, cfg)
    for idx, (sql_row, graph_edge, text) in enumerate(_reference_corpus(use_text=cfg.use_text)):
        mem.encode_episode(sql_row, graph_edge, text=text, episode_time=float(idx), consolidate=True)

    relation = _relation_accuracy_under_partial_sql(mem, cue_fraction=cfg.cue_fraction, duration=cfg.duration_ms)
    separation = mem.evaluate_separation(duration=cfg.duration_ms)
    false_retrieval = mem.evaluate_false_retrieval_rate(duration=cfg.duration_ms)
    interference = mem.evaluate_interference()

    return {
        "seed": int(seed),
        "bridge_lr": float(bridge_lr),
        "relation_accuracy": float(relation["mean"]),
        "relation_predictions": relation["predictions"],
        "separation_margin_mean": float(separation["separation_margin_mean"]),
        "false_retrieval_rate": float(false_retrieval["false_retrieval_rate"]),
        "mean_retention": float(interference["mean_retention"]),
        "target_overlap_mean": float(separation["target_overlap_mean"]),
        "best_impostor_overlap_mean": float(separation["best_impostor_overlap_mean"]),
        "pairwise_overlap": float(separation["mean_pairwise_assembly_overlap"]),
        "oldest_retention": float(interference["oldest_retention"]),
        "newest_retention": float(interference["newest_retention"]),
    }


def _summarize_bridge(bridge_value: float, seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        "relation_accuracy",
        "separation_margin_mean",
        "false_retrieval_rate",
        "mean_retention",
        "target_overlap_mean",
        "best_impostor_overlap_mean",
        "pairwise_overlap",
        "oldest_retention",
        "newest_retention",
    ]
    payload: Dict[str, Any] = {}
    for key in metrics:
        vals = [float(item[key]) for item in seed_metrics]
        mean, std = _mean_std(vals)
        payload[key] = {
            "per_seed": vals,
            "mean": mean,
            "std": std,
        }
    payload["bridge_lr"] = float(bridge_value)
    return payload


def _average_per_seed(metrics: Sequence[Dict[str, Any]], key: str) -> List[float]:
    return [float(item[key]) for item in metrics]


def _plot_results(summary: Dict[str, Any], out_dir: Path) -> None:
    bridge_lrs = [float(x) for x in summary["bridge_lrs"]]
    bridge_lrs_sorted = sorted(bridge_lrs)
    results = summary["results"]

    relation_means = [results[str(x)]["relation_accuracy"]["mean"] for x in bridge_lrs_sorted]
    relation_stds = [results[str(x)]["relation_accuracy"]["std"] for x in bridge_lrs_sorted]
    false_means = [results[str(x)]["false_retrieval_rate"]["mean"] for x in bridge_lrs_sorted]
    false_stds = [results[str(x)]["false_retrieval_rate"]["std"] for x in bridge_lrs_sorted]
    margin_means = [results[str(x)]["separation_margin_mean"]["mean"] for x in bridge_lrs_sorted]
    margin_stds = [results[str(x)]["separation_margin_mean"]["std"] for x in bridge_lrs_sorted]
    retention_means = [results[str(x)]["mean_retention"]["mean"] for x in bridge_lrs_sorted]
    retention_stds = [results[str(x)]["mean_retention"]["std"] for x in bridge_lrs_sorted]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    flat = axes.ravel()

    flat[0].errorbar(bridge_lrs_sorted, relation_means, yerr=relation_stds, marker="o", capsize=4)
    flat[0].set_xlabel("bridge_lr")
    flat[0].set_ylabel("Relation accuracy")
    flat[0].set_ylim(0.0, 1.05)
    flat[0].grid(True, alpha=0.3)

    flat[1].errorbar(bridge_lrs_sorted, false_means, yerr=false_stds, marker="o", capsize=4, color="#d62728")
    flat[1].set_xlabel("bridge_lr")
    flat[1].set_ylabel("False retrieval rate")
    flat[1].set_ylim(0.0, 1.05)
    flat[1].grid(True, alpha=0.3)

    flat[2].errorbar(bridge_lrs_sorted, margin_means, yerr=margin_stds, marker="o", capsize=4, color="#2ca02c")
    flat[2].set_xlabel("bridge_lr")
    flat[2].set_ylabel("Separation margin")
    flat[2].grid(True, alpha=0.3)

    flat[3].errorbar(bridge_lrs_sorted, retention_means, yerr=retention_stds, marker="o", capsize=4, color="#9467bd")
    flat[3].set_xlabel("bridge_lr")
    flat[3].set_ylabel("Mean retention")
    flat[3].set_ylim(0.0, 1.05)
    flat[3].grid(True, alpha=0.3)

    for ax in flat:
        ax.axvline(summary["baseline_bridge_lr"], color="black", linestyle="--", linewidth=1, alpha=0.6)

    fig.suptitle("DG->CA3 bridge plasticity sweep")
    fig.savefig(out_dir / "bridge_sweep_metrics.png", dpi=160)
    plt.close(fig)

    baseline = str(summary["baseline_bridge_lr"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    deltas = {
        "false_retrieval_rate": [results[str(x)]["vs_baseline"]["false_retrieval_rate"]["per_seed_delta"] for x in bridge_lrs_sorted if str(x) != baseline],
        "separation_margin_mean": [results[str(x)]["vs_baseline"]["separation_margin_mean"]["per_seed_delta"] for x in bridge_lrs_sorted if str(x) != baseline],
        "mean_retention": [results[str(x)]["vs_baseline"]["mean_retention"]["per_seed_delta"] for x in bridge_lrs_sorted if str(x) != baseline],
    }
    labels = [str(x) for x in bridge_lrs_sorted if str(x) != baseline]
    for ax, (metric, series) in zip(axes, deltas.items()):
        try:
            ax.boxplot(series, tick_labels=labels, vert=True, patch_artist=True)
        except TypeError:
            ax.boxplot(series, labels=labels, vert=True, patch_artist=True)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("bridge_lr")
        ax.set_ylabel(f"Delta {metric} vs baseline")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Per-seed deltas vs baseline bridge_lr={summary['baseline_bridge_lr']}")
    fig.savefig(out_dir / "bridge_sweep_deltas.png", dpi=160)
    plt.close(fig)


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Bridge LR Sweep",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Timestamp UTC: `{summary['timestamp_utc']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Bridge LRs: `{summary['bridge_lrs']}`",
        f"- Baseline bridge_lr: `{summary['baseline_bridge_lr']}`",
        f"- Cue fraction: `{summary['cue_fraction']}`",
        f"- Duration ms: `{summary['duration_ms']}`",
        "",
        "## Summary",
        "",
        "| bridge_lr | Relation acc | False retrieval | Separation margin | Mean retention | Best impostor | Pairwise overlap |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for bridge_lr in summary["bridge_lrs"]:
        payload = summary["results"][str(bridge_lr)]
        lines.append(
            f"| `{bridge_lr}` | "
            f"{payload['relation_accuracy']['mean']:.3f} ± {payload['relation_accuracy']['std']:.3f} | "
            f"{payload['false_retrieval_rate']['mean']:.3f} ± {payload['false_retrieval_rate']['std']:.3f} | "
            f"{payload['separation_margin_mean']['mean']:.3f} ± {payload['separation_margin_mean']['std']:.3f} | "
            f"{payload['mean_retention']['mean']:.3f} ± {payload['mean_retention']['std']:.3f} | "
            f"{payload['best_impostor_overlap_mean']['mean']:.3f} ± {payload['best_impostor_overlap_mean']['std']:.3f} | "
            f"{payload['pairwise_overlap']['mean']:.3f} ± {payload['pairwise_overlap']['std']:.3f} |"
        )

    lines.extend([
        "",
        "## Paired Tests vs Baseline",
        "",
        "| bridge_lr | Metric | Mean delta | Paired t p | Bootstrap 95% CI |",
        "| --- | --- | --- | --- | --- |",
    ])

    for bridge_lr in summary["bridge_lrs"]:
        if bridge_lr == summary["baseline_bridge_lr"]:
            continue
        payload = summary["results"][str(bridge_lr)]["vs_baseline"]
        for metric in ["false_retrieval_rate", "separation_margin_mean", "mean_retention", "best_impostor_overlap_mean"]:
            stats_entry = payload[metric]
            lines.append(
                f"| `{bridge_lr}` | `{metric}` | {stats_entry['mean_delta']:.3f} | "
                f"{stats_entry['paired_t_p']:.4g} | "
                f"[{stats_entry['bootstrap_ci95'][0]:.3f}, {stats_entry['bootstrap_ci95'][1]:.3f}] |"
            )

    return "\n".join(lines) + "\n"


def run_sweep(
    seeds: Sequence[int],
    bridge_lrs: Sequence[float],
    *,
    baseline_bridge_lr: float,
    cfg: SweepConfig,
) -> Dict[str, Any]:
    bridge_lrs = [float(x) for x in bridge_lrs]
    if baseline_bridge_lr not in bridge_lrs:
        raise ValueError("baseline bridge_lr must be included in bridge_lrs")

    summary: Dict[str, Any] = {
        "run_id": _utc_run_id(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [int(s) for s in seeds],
        "bridge_lrs": bridge_lrs,
        "baseline_bridge_lr": float(baseline_bridge_lr),
        "cue_fraction": float(cfg.cue_fraction),
        "duration_ms": float(cfg.duration_ms),
        "config": {
            "use_text": bool(cfg.use_text),
            "ca3_exc": int(cfg.ca3_exc),
            "ca3_inh": int(cfg.ca3_inh),
            "dg_bridge_fanout": int(cfg.dg_bridge_fanout),
            "dg_output_dim": int(cfg.dg_output_dim),
            "dg_target_sparsity": float(cfg.dg_target_sparsity),
            "ca1_n": int(cfg.ca1_n),
            "ca1_train_epochs": int(cfg.ca1_train_epochs),
            "ca1_train_lr": float(cfg.ca1_train_lr),
            "ca1_relation_lr": float(cfg.ca1_relation_lr),
            "ca1_relation_train_cue_fraction": float(cfg.ca1_relation_train_cue_fraction),
        },
        "results": {},
    }

    for bridge_lr in bridge_lrs:
        print(f"[bridge] bridge_lr={bridge_lr}", flush=True)
        seed_metrics: List[Dict[str, Any]] = []
        for seed in seeds:
            print(f"  [seed] {seed}", flush=True)
            seed_metrics.append(_evaluate_seed(seed, bridge_lr, cfg))

        payload = _summarize_bridge(bridge_lr, seed_metrics)
        payload["seed_metrics"] = seed_metrics
        summary["results"][str(bridge_lr)] = payload

    base = summary["results"][str(baseline_bridge_lr)]
    base_metrics = {
        metric: base[metric]["per_seed"]
        for metric in ["false_retrieval_rate", "separation_margin_mean", "mean_retention", "best_impostor_overlap_mean"]
    }
    for bridge_lr in bridge_lrs:
        key = str(bridge_lr)
        if bridge_lr == baseline_bridge_lr:
            summary["results"][key]["vs_baseline"] = {
                metric: {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                    "per_seed_delta": [0.0 for _ in base_metrics[metric]],
                }
                for metric in base_metrics
            }
            continue

        summary["results"][key]["vs_baseline"] = {}
        cur = summary["results"][key]
        for metric in base_metrics:
            cur_vals = cur[metric]["per_seed"]
            base_vals = base_metrics[metric]
            stats_entry = _paired_stats(cur_vals, base_vals, seed=17)
            stats_entry["per_seed_delta"] = [float(a - b) for a, b in zip(cur_vals, base_vals)]
            summary["results"][key]["vs_baseline"][metric] = stats_entry

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bridge_lr sweep with 10+ seeds.")
    parser.add_argument("--seeds", type=_parse_ints, default=list(DEFAULT_SEEDS), help="Comma-separated seed list.")
    parser.add_argument("--bridge-lrs", type=_parse_floats, default=list(DEFAULT_BRIDGE_LRS),
                        help="Comma-separated bridge_lr values.")
    parser.add_argument("--baseline-bridge-lr", type=float, default=0.02,
                        help="Bridge_lr used as the paired-test baseline.")
    parser.add_argument("--cue-fraction", type=float, default=0.4)
    parser.add_argument("--duration-ms", type=float, default=50.0)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--dg-output-dim", type=int, default=2000)
    parser.add_argument("--dg-target-sparsity", type=float, default=0.01)
    parser.add_argument("--dg-bridge-fanout", type=int, default=12)
    parser.add_argument("--ca3-exc", type=int, default=240)
    parser.add_argument("--ca3-inh", type=int, default=60)
    parser.add_argument("--ca1-n", type=int, default=320)
    parser.add_argument("--ca1-train-epochs", type=int, default=12)
    parser.add_argument("--ca1-train-lr", type=float, default=0.03)
    parser.add_argument("--ca1-relation-lr", type=float, default=0.05)
    parser.add_argument("--ca1-relation-train-cue-fraction", type=float, default=0.4)
    args = parser.parse_args()

    cfg = SweepConfig(
        cue_fraction=args.cue_fraction,
        duration_ms=args.duration_ms,
        use_text=True,
        ca3_exc=args.ca3_exc,
        ca3_inh=args.ca3_inh,
        dg_bridge_fanout=args.dg_bridge_fanout,
        dg_output_dim=args.dg_output_dim,
        dg_target_sparsity=args.dg_target_sparsity,
        ca1_n=args.ca1_n,
        ca1_train_epochs=args.ca1_train_epochs,
        ca1_train_lr=args.ca1_train_lr,
        ca1_relation_lr=args.ca1_relation_lr,
        ca1_relation_train_cue_fraction=args.ca1_relation_train_cue_fraction,
    )

    run = run_sweep(
        seeds=args.seeds,
        bridge_lrs=args.bridge_lrs,
        baseline_bridge_lr=args.baseline_bridge_lr,
        cfg=cfg,
    )

    run_dir = args.output_root / run["run_id"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run["run_dir"] = str(run_dir.resolve())

    _write_json(run_dir / "summary.json", run)
    _write_text(run_dir / "summary.md", _render_markdown(run))
    _plot_results(run, fig_dir)

    print(json.dumps(_jsonable({
        "run_dir": run["run_dir"],
        "summary_md": str((run_dir / "summary.md").resolve()),
        "figures": sorted(str(p.resolve()) for p in fig_dir.glob("*.png")),
    }), indent=2))


if __name__ == "__main__":
    main()
