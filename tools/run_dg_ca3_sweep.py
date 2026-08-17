#!/usr/bin/env python3
"""
Sweep DG separation knobs with a fixed bridge learning rate.

This runner holds the best observed bridge setting fixed and varies:
  - DG target sparsity
  - DG->CA3 bridge fanout

The goal is to reduce false retrieval / engram overlap while preserving
relation accuracy and retention under the same 10-seed protocol used for the
bridge sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import _reference_corpus
from spiking_multimodal_memory import MultiModalMemory


RUN_ROOT = Path("outputs/dg_ca3_sweeps")
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_SPARSITIES = (0.005, 0.01, 0.015, 0.02, 0.03)
DEFAULT_FANOUTS = (4, 8, 12, 16, 24)


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


@dataclass(frozen=True)
class BaseConfig:
    cue_fraction: float = 0.4
    duration_ms: float = 50.0
    use_text: bool = True
    bridge_lr: float = 0.01
    dg_output_dim: int = 2000
    dg_target_sparsity: float = 0.01
    dg_bridge_fanout: int = 12
    ca3_exc: int = 240
    ca3_inh: int = 60
    ca1_n: int = 320
    ca1_train_epochs: int = 12
    ca1_train_lr: float = 0.03
    ca1_relation_lr: float = 0.05
    ca1_relation_train_cue_fraction: float = 0.4


def _build_memory(seed: int, cfg: BaseConfig) -> MultiModalMemory:
    return MultiModalMemory(
        use_text=cfg.use_text,
        seed=seed,
        ca3_exc=cfg.ca3_exc,
        ca3_inh=cfg.ca3_inh,
        dg_bridge_fanout=cfg.dg_bridge_fanout,
        dg_bridge_lr=cfg.bridge_lr,
        dg_output_dim=cfg.dg_output_dim,
        dg_target_sparsity=cfg.dg_target_sparsity,
        ca1_n=cfg.ca1_n,
        ca1_train_epochs=cfg.ca1_train_epochs,
        ca1_train_lr=cfg.ca1_train_lr,
        ca1_relation_lr=cfg.ca1_relation_lr,
        ca1_relation_train_cue_fraction=cfg.ca1_relation_train_cue_fraction,
    )


def _evaluate_seed(seed: int, cfg: BaseConfig) -> Dict[str, Any]:
    corpus = _reference_corpus(use_text=cfg.use_text)
    mem = _build_memory(seed, cfg)

    for idx, (sql_row, graph_edge, text) in enumerate(corpus):
        mem.encode_episode(
            sql_row,
            graph_edge,
            text=text,
            episode_time=float(idx),
            consolidate=True,
        )

    relation_scores: List[float] = []
    relation_predictions: List[str] = []
    for record in mem.episode_records:
        partial_sql = _partial_sql_row(mem, record.sql_row, cfg.cue_fraction)
        retrieved = mem.retrieve(sql_cue=partial_sql, duration=cfg.duration_ms)
        graph_metrics = mem.compute_graph_retrieval_accuracy(retrieved, record.graph_edge)
        relation_scores.append(float(graph_metrics.get("relation_accuracy", 0.0)))
        relation_predictions.append(str(graph_metrics.get("relation_prediction", "")))

    relation_mean, relation_std = _mean_std(relation_scores)
    separation = mem.evaluate_separation(duration=cfg.duration_ms)
    retention = mem.evaluate_interference()

    return {
        "seed": int(seed),
        "bridge_lr": float(cfg.bridge_lr),
        "dg_output_dim": int(cfg.dg_output_dim),
        "dg_target_sparsity": float(cfg.dg_target_sparsity),
        "dg_bridge_fanout": int(cfg.dg_bridge_fanout),
        "relation_accuracy": {
            "per_episode": relation_scores,
            "predictions": relation_predictions,
            "mean": relation_mean,
            "std": relation_std,
        },
        "false_retrieval_rate": float(separation["false_retrieval_rate"]),
        "target_overlap_mean": float(separation["target_overlap_mean"]),
        "best_impostor_overlap_mean": float(separation["best_impostor_overlap_mean"]),
        "separation_margin_mean": float(separation["separation_margin_mean"]),
        "separation_top1_accuracy": float(separation["separation_top1_accuracy"]),
        "mean_pairwise_overlap": float(separation["mean_pairwise_assembly_overlap"]),
        "mean_retention": float(retention["mean_retention"]),
        "oldest_retention": float(retention["oldest_retention"]),
        "newest_retention": float(retention["newest_retention"]),
    }


def _summarize_family(value_key: str, values: Sequence[Any], seed_metrics: Mapping[Any, List[Dict[str, Any]]],
                      baseline_value: Any) -> Dict[str, Any]:
    metric_names = [
        "relation_accuracy",
        "false_retrieval_rate",
        "separation_margin_mean",
        "mean_retention",
        "best_impostor_overlap_mean",
        "mean_pairwise_overlap",
        "target_overlap_mean",
        "oldest_retention",
        "newest_retention",
    ]

    results: Dict[str, Any] = {}
    for value in values:
        metrics = seed_metrics[float(value)]
        payload: Dict[str, Any] = {}
        for metric in metric_names:
            if metric == "relation_accuracy":
                vals = [float(item[metric]["mean"]) for item in metrics]
            else:
                vals = [float(item[metric]) for item in metrics]
            mean, std = _mean_std(vals)
            payload[metric] = {
                "per_seed": vals,
                "mean": mean,
                "std": std,
            }
        results[str(value)] = payload

    baseline = results[str(baseline_value)]
    for value in values:
        key = str(value)
        if float(value) == float(baseline_value):
            results[key]["vs_baseline"] = {
                "relation_accuracy": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                    "per_seed_delta": [0.0 for _ in baseline["relation_accuracy"]["per_seed"]],
                },
                "false_retrieval_rate": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                    "per_seed_delta": [0.0 for _ in baseline["false_retrieval_rate"]["per_seed"]],
                },
                "separation_margin_mean": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                    "per_seed_delta": [0.0 for _ in baseline["separation_margin_mean"]["per_seed"]],
                },
                "mean_retention": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                    "per_seed_delta": [0.0 for _ in baseline["mean_retention"]["per_seed"]],
                },
                "best_impostor_overlap_mean": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                    "per_seed_delta": [0.0 for _ in baseline["best_impostor_overlap_mean"]["per_seed"]],
                },
                "mean_pairwise_overlap": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                    "per_seed_delta": [0.0 for _ in baseline["mean_pairwise_overlap"]["per_seed"]],
                },
            }
            continue

        cur = results[key]
        results[key]["vs_baseline"] = {
            "relation_accuracy": _paired_stats(cur["relation_accuracy"]["per_seed"], baseline["relation_accuracy"]["per_seed"], seed=7),
            "false_retrieval_rate": _paired_stats(cur["false_retrieval_rate"]["per_seed"], baseline["false_retrieval_rate"]["per_seed"], seed=11),
            "separation_margin_mean": _paired_stats(cur["separation_margin_mean"]["per_seed"], baseline["separation_margin_mean"]["per_seed"], seed=13),
            "mean_retention": _paired_stats(cur["mean_retention"]["per_seed"], baseline["mean_retention"]["per_seed"], seed=17),
            "best_impostor_overlap_mean": _paired_stats(cur["best_impostor_overlap_mean"]["per_seed"], baseline["best_impostor_overlap_mean"]["per_seed"], seed=19),
            "mean_pairwise_overlap": _paired_stats(cur["mean_pairwise_overlap"]["per_seed"], baseline["mean_pairwise_overlap"]["per_seed"], seed=23),
        }

        for metric in results[key]["vs_baseline"].keys():
            cur_vals = results[key][metric]["per_seed"]
            base_vals = baseline[metric]["per_seed"]
            results[key]["vs_baseline"][metric]["per_seed_delta"] = [
                float(a - b) for a, b in zip(cur_vals, base_vals)
            ]

    return {
        "parameter": value_key,
        "values": list(values),
        "baseline_value": baseline_value,
        "results": results,
    }


def _run_family(name: str, value_key: str, values: Sequence[Any], baseline_value: Any,
                base_cfg: BaseConfig, seeds: Sequence[int]) -> Dict[str, Any]:
    if baseline_value not in values:
        raise ValueError(f"baseline {baseline_value} must be included in {name} values")

    seed_metrics: Dict[Any, List[Dict[str, Any]]] = {}
    for value in values:
        print(f"[sweep:{name}] {value_key}={value}", flush=True)
        per_seed: List[Dict[str, Any]] = []
        for seed in seeds:
            print(f"  [seed] {seed}", flush=True)
            cfg_value = int(value) if value_key == "dg_bridge_fanout" else float(value)
            cfg = replace(base_cfg, **{value_key: cfg_value})
            per_seed.append(_evaluate_seed(seed, cfg))
        seed_metrics[value] = per_seed

    return _summarize_family(value_key, values, seed_metrics, baseline_value)


def _plot_family(summary: Dict[str, Any], family_name: str, out_dir: Path) -> None:
    values = list(summary["values"])
    results = summary["results"]
    x = values

    rel_means = [results[str(v)]["relation_accuracy"]["mean"] for v in values]
    rel_stds = [results[str(v)]["relation_accuracy"]["std"] for v in values]
    false_means = [results[str(v)]["false_retrieval_rate"]["mean"] for v in values]
    false_stds = [results[str(v)]["false_retrieval_rate"]["std"] for v in values]
    margin_means = [results[str(v)]["separation_margin_mean"]["mean"] for v in values]
    margin_stds = [results[str(v)]["separation_margin_mean"]["std"] for v in values]
    retention_means = [results[str(v)]["mean_retention"]["mean"] for v in values]
    retention_stds = [results[str(v)]["mean_retention"]["std"] for v in values]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    axes[0].errorbar(x, rel_means, yerr=rel_stds, marker="o", capsize=4)
    axes[0].set_xlabel(summary["parameter"])
    axes[0].set_ylabel("Relation accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(x, false_means, yerr=false_stds, marker="o", capsize=4, color="#d62728")
    axes[1].set_xlabel(summary["parameter"])
    axes[1].set_ylabel("False retrieval rate")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(x, margin_means, yerr=margin_stds, marker="o", capsize=4, color="#2ca02c")
    axes[2].set_xlabel(summary["parameter"])
    axes[2].set_ylabel("Separation margin")
    axes[2].grid(True, alpha=0.3)

    axes[3].errorbar(x, retention_means, yerr=retention_stds, marker="o", capsize=4, color="#9467bd")
    axes[3].set_xlabel(summary["parameter"])
    axes[3].set_ylabel("Mean retention")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"{family_name} sweep")
    fig.savefig(out_dir / f"{family_name}_metrics.png", dpi=160)
    plt.close(fig)

    baseline = str(summary["baseline_value"])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    delta_metrics = [
        "false_retrieval_rate",
        "separation_margin_mean",
        "mean_retention",
        "best_impostor_overlap_mean",
    ]
    labels = [str(v) for v in values if str(v) != baseline]
    for ax, metric in zip(axes, delta_metrics):
        series = [
            results[str(v)]["vs_baseline"][metric]["per_seed_delta"]
            for v in values if str(v) != baseline
        ]
        try:
            ax.boxplot(series, tick_labels=labels, orientation="vertical", patch_artist=True)
        except TypeError:
            ax.boxplot(series, labels=labels, vert=True, patch_artist=True)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel(summary["parameter"])
        ax.set_ylabel(f"Delta {metric}")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Per-seed deltas vs baseline {summary['parameter']}={summary['baseline_value']}")
    fig.savefig(out_dir / f"{family_name}_deltas.png", dpi=160)
    plt.close(fig)


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# DG / CA3 Sweep",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Timestamp UTC: `{summary['timestamp_utc']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Fixed bridge_lr: `{summary['fixed_config']['bridge_lr']}`",
        f"- Fixed DG output_dim: `{summary['fixed_config']['dg_output_dim']}`",
        f"- Fixed CA1 epochs/lr: `{summary['fixed_config']['ca1_train_epochs']} / {summary['fixed_config']['ca1_train_lr']}`",
        "",
    ]

    for family_name, family in summary["families"].items():
        lines.extend([
            f"## {family_name}",
            "",
            f"- Parameter: `{family['parameter']}`",
            f"- Baseline: `{family['baseline_value']}`",
            "",
            "| Value | Relation acc | False retrieval | Separation margin | Mean retention | Best impostor | Pairwise overlap |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for value in family["values"]:
            payload = family["results"][str(value)]
            lines.append(
                f"| `{value}` | "
                f"{payload['relation_accuracy']['mean']:.3f} ± {payload['relation_accuracy']['std']:.3f} | "
                f"{payload['false_retrieval_rate']['mean']:.3f} ± {payload['false_retrieval_rate']['std']:.3f} | "
                f"{payload['separation_margin_mean']['mean']:.3f} ± {payload['separation_margin_mean']['std']:.3f} | "
                f"{payload['mean_retention']['mean']:.3f} ± {payload['mean_retention']['std']:.3f} | "
                f"{payload['best_impostor_overlap_mean']['mean']:.3f} ± {payload['best_impostor_overlap_mean']['std']:.3f} | "
                f"{payload['mean_pairwise_overlap']['mean']:.3f} ± {payload['mean_pairwise_overlap']['std']:.3f} |"
            )

        lines.extend([
            "",
            "### Paired Tests vs Baseline",
            "",
            "| Value | Metric | Mean delta | Paired t p | Bootstrap 95% CI |",
            "| --- | --- | --- | --- | --- |",
        ])
        for value in family["values"]:
            if float(value) == float(family["baseline_value"]):
                continue
            payload = family["results"][str(value)]["vs_baseline"]
            for metric in [
                "false_retrieval_rate",
                "separation_margin_mean",
                "mean_retention",
                "best_impostor_overlap_mean",
            ]:
                stats_entry = payload[metric]
                lines.append(
                    f"| `{value}` | `{metric}` | {stats_entry['mean_delta']:.3f} | "
                    f"{stats_entry['paired_t_p']:.4g} | "
                    f"[{stats_entry['bootstrap_ci95'][0]:.3f}, {stats_entry['bootstrap_ci95'][1]:.3f}] |"
                )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DG sparsity and CA3 fanout sweeps.")
    parser.add_argument("--seeds", type=_parse_ints, default=list(DEFAULT_SEEDS),
                        help="Comma-separated seeds, e.g. 0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--sparsities", type=_parse_floats, default=list(DEFAULT_SPARSITIES),
                        help="Comma-separated dg_target_sparsity values.")
    parser.add_argument("--fanouts", type=_parse_ints, default=list(DEFAULT_FANOUTS),
                        help="Comma-separated dg_bridge_fanout values.")
    parser.add_argument("--baseline-sparsity", type=float, default=0.01,
                        help="Baseline dg_target_sparsity for paired tests.")
    parser.add_argument("--baseline-fanout", type=int, default=12,
                        help="Baseline dg_bridge_fanout for paired tests.")
    parser.add_argument("--bridge-lr", type=float, default=0.01,
                        help="Fixed dg_bridge_lr used for all runs.")
    parser.add_argument("--dg-output-dim", type=int, default=2000)
    parser.add_argument("--ca3-exc", type=int, default=240)
    parser.add_argument("--ca3-inh", type=int, default=60)
    parser.add_argument("--ca1-n", type=int, default=320)
    parser.add_argument("--ca1-train-epochs", type=int, default=12)
    parser.add_argument("--ca1-train-lr", type=float, default=0.03)
    parser.add_argument("--ca1-relation-lr", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()

    base_cfg = BaseConfig(
        bridge_lr=args.bridge_lr,
        dg_output_dim=args.dg_output_dim,
        dg_target_sparsity=float(args.baseline_sparsity),
        dg_bridge_fanout=int(args.baseline_fanout),
        ca3_exc=args.ca3_exc,
        ca3_inh=args.ca3_inh,
        ca1_n=args.ca1_n,
        ca1_train_epochs=args.ca1_train_epochs,
        ca1_train_lr=args.ca1_train_lr,
        ca1_relation_lr=args.ca1_relation_lr,
    )

    run = {
        "run_id": _utc_run_id(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(int(s) for s in args.seeds),
        "fixed_config": asdict(base_cfg),
        "families": {},
    }

    sparsity_family = _run_family(
        name="dg_sparsity",
        value_key="dg_target_sparsity",
        values=args.sparsities,
        baseline_value=float(args.baseline_sparsity),
        base_cfg=base_cfg,
        seeds=args.seeds,
    )
    fanout_family = _run_family(
        name="ca3_fanout",
        value_key="dg_bridge_fanout",
        values=args.fanouts,
        baseline_value=int(args.baseline_fanout),
        base_cfg=base_cfg,
        seeds=args.seeds,
    )

    run["families"] = {
        "dg_sparsity": sparsity_family,
        "ca3_fanout": fanout_family,
    }

    run_dir = args.output_root / run["run_id"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run["run_dir"] = str(run_dir.resolve())

    _write_json(run_dir / "summary.json", run)
    _write_text(run_dir / "summary.md", _render_markdown(run))

    _plot_family(sparsity_family, "dg_sparsity", fig_dir)
    _plot_family(fanout_family, "ca3_fanout", fig_dir)

    print(json.dumps(_jsonable({
        "run_dir": run["run_dir"],
        "summary_md": str((run_dir / "summary.md").resolve()),
        "figures": sorted(str(p.resolve()) for p in fig_dir.glob("*.png")),
    }), indent=2))


if __name__ == "__main__":
    main()
