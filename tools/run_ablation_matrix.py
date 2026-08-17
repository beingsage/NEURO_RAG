#!/usr/bin/env python3
"""
Run the hippocampal ablation matrix across multiple seeds.

This sweep evaluates:
  - relation accuracy under a partial SQL cue
  - CERA (cross-modal episodic retrieval accuracy) matrix

Conditions:
  - no delay
  - random delay
  - no STDP
  - no DG
  - no CA3 recurrence
  - full model
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import _reference_corpus
from spiking_multimodal_memory import MultiModalMemory


RUN_ROOT = Path("outputs/ablation_runs")
MODALITIES = ("sql", "graph", "text")


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    graph_delay_mode: str = "structured"  # structured | zero | random
    use_dg: bool = True
    use_recurrence: bool = True
    disable_stdp: bool = False


CONDITIONS: "OrderedDict[str, ConditionSpec]" = OrderedDict(
    [
        ("no_delay", ConditionSpec(name="no_delay", graph_delay_mode="zero")),
        ("random_delay", ConditionSpec(name="random_delay", graph_delay_mode="random")),
        ("no_stdp", ConditionSpec(name="no_stdp", disable_stdp=True)),
        ("no_dg", ConditionSpec(name="no_dg", use_dg=False)),
        ("no_ca3_recurrence", ConditionSpec(name="no_ca3_recurrence", use_recurrence=False)),
        ("full", ConditionSpec(name="full")),
    ]
)


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


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _spikes_to_counts(spikes: Mapping[int, Sequence[float]], dim: int) -> np.ndarray:
    counts = np.zeros(dim, dtype=float)
    for nid, times in spikes.items():
        if 0 <= int(nid) < dim:
            counts[int(nid)] = float(len(times))
    return counts


def _partial_sql_row(mem: MultiModalMemory, row: Dict[str, Any], fraction: float) -> Dict[str, Any]:
    return mem._partial_sql_row(row, fraction)


def _top_k_overlap_from_scores(score_map: Mapping[int, float], support_counts: np.ndarray) -> float:
    support_idx = set(int(i) for i in np.flatnonzero(support_counts > 0))
    if not support_idx:
        return 0.0

    k = max(1, len(support_idx))
    vec = np.array([float(score_map.get(i, 0.0)) for i in range(len(support_counts))], dtype=float)
    top = set(int(i) for i in np.argsort(vec)[-k:])
    return float(len(top & support_idx) / max(1, len(support_idx)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


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
        raise ValueError("paired stats require the same number of samples")
    if a.size == 0:
        return {
            "mean_delta": 0.0,
            "paired_t_stat": 0.0,
            "paired_t_p": 1.0,
            "bootstrap_ci95": [0.0, 0.0],
        }

    diff = a - b
    t_stat, p_value = stats.ttest_rel(a, b, nan_policy="omit")
    if not np.isfinite(t_stat):
        t_stat = 0.0
    if not np.isfinite(p_value):
        p_value = 1.0
    ci_low, ci_high = _bootstrap_ci(a, b, seed=seed)
    return {
        "mean_delta": float(np.mean(diff)),
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(p_value),
        "bootstrap_ci95": [ci_low, ci_high],
    }


def _apply_condition(mem: MultiModalMemory, spec: ConditionSpec, seed: int) -> None:
    rng = np.random.RandomState(seed + 10_000)

    if spec.graph_delay_mode == "zero":
        mem.graph_enc.RELATION_DELAYS = {key: 0.0 for key in mem.graph_enc.RELATION_DELAYS}
    elif spec.graph_delay_mode == "random":
        rels = list(mem.graph_enc.RELATION_DELAYS.keys())
        mem.graph_enc.RELATION_DELAYS = {
            rel: float(rng.uniform(0.0, 8.0)) for rel in rels
        }

    if not spec.use_dg:
        mem.dg.encode = lambda _x: set()  # type: ignore[method-assign]
        mem.dg.to_spikes = lambda _active, t_offset=0.0: {}  # type: ignore[method-assign]
        mem.dg.update = lambda _x, _active: None  # type: ignore[method-assign]
        mem.dg.W[:] = 0.0

    if not spec.use_recurrence:
        _strip_ca3_recurrence(mem)

    if spec.disable_stdp:
        _disable_ca3_stdp(mem)


def _strip_ca3_recurrence(mem: MultiModalMemory) -> None:
    input_syn_ids = {
        id(syn)
        for syn_list in mem.ca3.input_synapses.values()
        for syn in syn_list
    }
    kept_synapses = []
    incoming: Dict[int, List[Any]] = defaultdict(list)
    outgoing: Dict[int, List[Any]] = defaultdict(list)

    for syn in mem.ca3.synapses:
        if id(syn) not in input_syn_ids:
            continue
        kept_synapses.append(syn)
        incoming[syn.post].append(syn)

    mem.ca3.synapses = kept_synapses
    mem.ca3.incoming = incoming
    mem.ca3.outgoing = outgoing
    mem.ca3._active_synapses = {syn for syn in mem.ca3._active_synapses if id(syn) in input_syn_ids}
    mem.ca3.assembly_bias[:] = 0.0
    mem.ca3.reinforce_assembly = lambda *args, **kwargs: None  # type: ignore[method-assign]


def _disable_ca3_stdp(mem: MultiModalMemory) -> None:
    for syn in mem.ca3.synapses:
        syn.stdp_update = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mem.ca3.reinforce_assembly = lambda *args, **kwargs: None  # type: ignore[method-assign]
    mem._reinforce_dg_bridge = lambda *args, **kwargs: None  # type: ignore[method-assign]


def _build_memory(
    seed: int,
    spec: ConditionSpec,
    *,
    use_text: bool = True,
    cue_fraction: float = 0.4,
    ca3_exc: int = 240,
    ca3_inh: int = 60,
    dg_bridge_fanout: int = 12,
    dg_bridge_lr: float = 0.02,
    dg_output_dim: int = 1200,
    dg_target_sparsity: float = 0.02,
    ca1_n: int = 320,
    ca1_train_epochs: int = 12,
    ca1_train_lr: float = 0.03,
    ca1_relation_lr: float = 0.05,
) -> MultiModalMemory:
    mem = MultiModalMemory(
        use_text=use_text,
        seed=seed,
        ca3_exc=ca3_exc,
        ca3_inh=ca3_inh,
        dg_bridge_fanout=dg_bridge_fanout,
        dg_bridge_lr=dg_bridge_lr,
        dg_output_dim=dg_output_dim,
        dg_target_sparsity=dg_target_sparsity,
        ca1_n=ca1_n,
        ca1_train_epochs=ca1_train_epochs,
        ca1_train_lr=ca1_train_lr,
        ca1_relation_lr=ca1_relation_lr,
    )
    _apply_condition(mem, spec, seed)

    for idx, (sql_row, graph_edge, text) in enumerate(_reference_corpus(use_text=use_text)):
        mem.encode_episode(
            sql_row,
            graph_edge,
            text=text,
            episode_time=float(idx),
            consolidate=True,
        )

    return mem


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


def _cera_matrix(mem: MultiModalMemory, duration: float) -> Dict[str, Any]:
    modalities = ["sql", "graph"]
    if mem.use_text:
        modalities.append("text")

    matrix: Dict[str, Dict[str, List[float]]] = {
        cue: {target: [] for target in modalities if target != cue}
        for cue in modalities
    }

    for record in mem.episode_records:
        sql_support = _spikes_to_counts(record.sql_spikes, 100)
        graph_support = _spikes_to_counts(record.graph_spikes, 80)
        text_support = _spikes_to_counts(record.text_spikes, 100) if mem.use_text else np.zeros(0)

        for cue in modalities:
            cue_args = {"sql_cue": None, "graph_cue": None, "text_cue": None}
            if cue == "sql":
                cue_args["sql_cue"] = record.sql_row
            elif cue == "graph":
                cue_args["graph_cue"] = record.graph_edge
            elif cue == "text":
                cue_args["text_cue"] = record.text

            retrieved = mem.retrieve(duration=duration, **cue_args)
            scores = _cera_scores_for_record(mem, record, retrieved, sql_support, graph_support, text_support)
            for target in matrix[cue].keys():
                matrix[cue][target].append(float(scores[target]))

    cell_means = {
        cue: {target: float(np.mean(vals)) if vals else 0.0 for target, vals in targets.items()}
        for cue, targets in matrix.items()
    }
    all_values = [val for targets in cell_means.values() for val in targets.values()]
    return {
        "matrix": cell_means,
        "per_record_matrix": matrix,
        "offdiag_mean": float(np.mean(all_values)) if all_values else 0.0,
    }


def _cera_scores_for_record(
    mem: MultiModalMemory,
    record: Any,
    retrieved: Dict[str, Any],
    sql_support: np.ndarray,
    graph_support: np.ndarray,
    text_support: np.ndarray,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    sql_recon = retrieved.get("sql_reconstruction", {})
    scores["sql"] = _top_k_overlap_from_scores(sql_recon, sql_support)

    graph_metrics = mem.compute_graph_retrieval_accuracy(retrieved, record.graph_edge)
    scores["graph"] = float(graph_metrics.get("edge_accuracy", 0.0))

    text_recon = retrieved.get("text_reconstruction", {})
    if mem.use_text and text_support.size > 0:
        pred = np.array([float(text_recon.get(i, 0.0)) for i in range(len(text_support))], dtype=float)
        tgt = np.array(text_support[: len(pred)], dtype=float)
        scores["text"] = _cosine(pred, tgt)
    else:
        scores["text"] = 0.0

    return scores


def _summarize_condition(seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    relation_values = [float(item["relation_accuracy"]["mean"]) for item in seed_metrics]
    cera_values = [float(item["cera"]["offdiag_mean"]) for item in seed_metrics]
    sql_to_graph = [float(item["cera"]["matrix"]["sql"]["graph"]) for item in seed_metrics]

    relation_mean, relation_std = _mean_std(relation_values)
    cera_mean, cera_std = _mean_std(cera_values)
    sql_graph_mean, sql_graph_std = _mean_std(sql_to_graph)

    cera_matrix_means = _average_matrices([item["cera"]["matrix"] for item in seed_metrics])

    return {
        "relation_accuracy": {
            "per_seed": relation_values,
            "mean": relation_mean,
            "std": relation_std,
        },
        "cera_mean": {
            "per_seed": cera_values,
            "mean": cera_mean,
            "std": cera_std,
        },
        "sql_to_graph_cera": {
            "per_seed": sql_to_graph,
            "mean": sql_graph_mean,
            "std": sql_graph_std,
        },
        "cera_matrix_mean": cera_matrix_means,
    }


def _average_matrices(matrices: Sequence[Mapping[str, Mapping[str, float]]]) -> Dict[str, Dict[str, float]]:
    if not matrices:
        return {}
    cues = list(matrices[0].keys())
    targets_by_cue = {cue: list(matrices[0][cue].keys()) for cue in cues}
    out: Dict[str, Dict[str, float]] = {}
    for cue in cues:
        out[cue] = {}
        for target in targets_by_cue[cue]:
            vals = [float(matrix[cue][target]) for matrix in matrices]
            out[cue][target] = float(np.mean(vals)) if vals else 0.0
    return out


def _plot_metric_bars(summary: Dict[str, Any], out_dir: Path) -> None:
    conditions = list(summary["conditions"].keys())
    rel_means = [summary["conditions"][c]["relation_accuracy"]["mean"] for c in conditions]
    rel_stds = [summary["conditions"][c]["relation_accuracy"]["std"] for c in conditions]
    cera_means = [summary["conditions"][c]["cera_mean"]["mean"] for c in conditions]
    cera_stds = [summary["conditions"][c]["cera_mean"]["std"] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(x, rel_means, width=width, yerr=rel_stds, capsize=4, color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=30, ha="right")
    axes[0].set_ylabel("Relation accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Partial-SQL graph relation accuracy")

    axes[1].bar(x, cera_means, width=width, yerr=cera_stds, capsize=4, color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=30, ha="right")
    axes[1].set_ylabel("Mean CERA")
    axes[1].set_ylim(0.0, max(1.0, max(cera_means + cera_stds) + 0.1))
    axes[1].set_title("Mean off-diagonal CERA")

    fig.suptitle("Ablation matrix across seeds")
    fig.savefig(out_dir / "ablation_metric_bars.png", dpi=160)
    plt.close(fig)


def _plot_cera_heatmaps(summary: Dict[str, Any], out_dir: Path) -> None:
    conditions = list(summary["conditions"].keys())
    n_cols = 3
    n_rows = math.ceil(len(conditions) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    for idx, condition in enumerate(conditions):
        ax = axes[idx // n_cols][idx % n_cols]
        matrix = summary["conditions"][condition]["cera_matrix_mean"]
        cues = list(matrix.keys())
        targets = list(next(iter(matrix.values())).keys()) if matrix else []
        data = np.array([[float(matrix[cue].get(target, 0.0)) for target in targets] for cue in cues], dtype=float)
        sns.heatmap(data, annot=True, fmt=".2f", cmap="magma", vmin=0.0, vmax=max(1e-6, float(np.max(data)) if data.size else 1.0),
                    xticklabels=targets, yticklabels=cues, ax=ax)
        ax.set_title(condition)
        ax.set_xlabel("Target")
        ax.set_ylabel("Cue")

    # Hide unused axes
    for idx in range(len(conditions), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Mean CERA matrices by condition")
    fig.savefig(out_dir / "cera_heatmaps.png", dpi=160)
    plt.close(fig)


def _plot_seed_deltas(summary: Dict[str, Any], out_dir: Path) -> None:
    if "full" not in summary["conditions"]:
        return

    base_rel = np.asarray(summary["conditions"]["full"]["relation_accuracy"]["per_seed"], dtype=float)
    base_cera = np.asarray(summary["conditions"]["full"]["cera_mean"]["per_seed"], dtype=float)

    conditions = [c for c in summary["conditions"].keys() if c != "full"]
    rel_deltas = []
    cera_deltas = []
    labels = []
    for condition in conditions:
        rel = np.asarray(summary["conditions"][condition]["relation_accuracy"]["per_seed"], dtype=float)
        cera = np.asarray(summary["conditions"][condition]["cera_mean"]["per_seed"], dtype=float)
        rel_deltas.append(rel - base_rel)
        cera_deltas.append(cera - base_cera)
        labels.append(condition)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].boxplot(rel_deltas, vert=True, patch_artist=True)
    axes[0].set_xticks(np.arange(1, len(labels) + 1))
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("Delta relation accuracy vs full")

    axes[1].boxplot(cera_deltas, vert=True, patch_artist=True)
    axes[1].set_xticks(np.arange(1, len(labels) + 1))
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Delta mean CERA vs full")

    fig.suptitle("Per-seed deltas relative to the full model")
    fig.savefig(out_dir / "ablation_deltas_vs_full.png", dpi=160)
    plt.close(fig)


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Ablation Matrix",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Timestamp UTC: `{summary['timestamp_utc']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Partial SQL cue fraction: `{summary['cue_fraction']}`",
        f"- Retrieval duration: `{summary['duration_ms']} ms`",
        "",
        "## Summary",
        "",
        "| Condition | Relation acc (mean ± std) | Mean CERA (mean ± std) | SQL→Graph CERA (mean ± std) |",
        "| --- | --- | --- | --- |",
    ]

    for condition, payload in summary["conditions"].items():
        rel = payload["relation_accuracy"]
        cera = payload["cera_mean"]
        sg = payload["sql_to_graph_cera"]
        lines.append(
            f"| `{condition}` | "
            f"{_format_float(rel['mean'])} ± {_format_float(rel['std'])} | "
            f"{_format_float(cera['mean'])} ± {_format_float(cera['std'])} | "
            f"{_format_float(sg['mean'])} ± {_format_float(sg['std'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired Tests vs Full",
            "",
            "| Condition | Metric | Mean delta | Paired t p | Bootstrap 95% CI |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    full = summary["conditions"]["full"]
    for condition, payload in summary["conditions"].items():
        if condition == "full":
            continue
        stats_rel = payload["stats_vs_full"]["relation_accuracy"]
        stats_cera = payload["stats_vs_full"]["cera_mean"]
        lines.append(
            f"| `{condition}` | relation accuracy | {_format_float(stats_rel['mean_delta'])} | "
            f"{stats_rel['paired_t_p']:.4g} | "
            f"[{_format_float(stats_rel['bootstrap_ci95'][0])}, {_format_float(stats_rel['bootstrap_ci95'][1])}] |"
        )
        lines.append(
            f"| `{condition}` | mean CERA | {_format_float(stats_cera['mean_delta'])} | "
            f"{stats_cera['paired_t_p']:.4g} | "
            f"[{_format_float(stats_cera['bootstrap_ci95'][0])}, {_format_float(stats_cera['bootstrap_ci95'][1])}] |"
        )

    return "\n".join(lines) + "\n"


def run_ablation_suite(
    seeds: Sequence[int],
    *,
    cue_fraction: float = 0.4,
    duration_ms: float = 50.0,
    use_text: bool = True,
    ca3_exc: int = 240,
    ca3_inh: int = 60,
    dg_bridge_fanout: int = 12,
    dg_bridge_lr: float = 0.02,
    dg_output_dim: int = 1200,
    dg_target_sparsity: float = 0.02,
    ca1_n: int = 320,
    ca1_train_epochs: int = 12,
    ca1_train_lr: float = 0.03,
    ca1_relation_lr: float = 0.05,
) -> Dict[str, Any]:
    corpus = _reference_corpus(use_text=use_text)
    if not corpus:
        raise RuntimeError("reference corpus is empty")

    results: Dict[str, Any] = {
        "run_id": _utc_run_id(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(int(s) for s in seeds),
        "cue_fraction": float(cue_fraction),
        "duration_ms": float(duration_ms),
        "config": {
            "use_text": bool(use_text),
            "ca3_exc": int(ca3_exc),
            "ca3_inh": int(ca3_inh),
            "dg_bridge_fanout": int(dg_bridge_fanout),
            "dg_bridge_lr": float(dg_bridge_lr),
            "dg_output_dim": int(dg_output_dim),
            "dg_target_sparsity": float(dg_target_sparsity),
            "ca1_n": int(ca1_n),
            "ca1_train_epochs": int(ca1_train_epochs),
            "ca1_train_lr": float(ca1_train_lr),
            "ca1_relation_lr": float(ca1_relation_lr),
        },
        "conditions": {},
    }

    for condition_name, spec in CONDITIONS.items():
        print(f"[ablation] condition={condition_name}", flush=True)
        seed_metrics = []
        for seed in seeds:
            print(f"  [seed] {seed}", flush=True)
            mem = _build_memory(
                seed,
                spec,
                use_text=use_text,
                cue_fraction=cue_fraction,
                ca3_exc=ca3_exc,
                ca3_inh=ca3_inh,
                dg_bridge_fanout=dg_bridge_fanout,
                dg_bridge_lr=dg_bridge_lr,
                dg_output_dim=dg_output_dim,
                dg_target_sparsity=dg_target_sparsity,
                ca1_n=ca1_n,
                ca1_train_epochs=ca1_train_epochs,
                ca1_train_lr=ca1_train_lr,
                ca1_relation_lr=ca1_relation_lr,
            )
            relation = _relation_accuracy_under_partial_sql(mem, cue_fraction=cue_fraction, duration=duration_ms)
            cera = _cera_matrix(mem, duration=duration_ms)
            seed_metrics.append(
                {
                    "seed": int(seed),
                    "relation_accuracy": relation,
                    "cera": cera,
                }
            )

        summary = _summarize_condition(seed_metrics)
        summary["seed_metrics"] = seed_metrics
        results["conditions"][condition_name] = summary

    full_relation = results["conditions"]["full"]["relation_accuracy"]["per_seed"]
    full_cera = results["conditions"]["full"]["cera_mean"]["per_seed"]
    for condition in results["conditions"]:
        if condition == "full":
            results["conditions"][condition]["stats_vs_full"] = {
                "relation_accuracy": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                },
                "cera_mean": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                },
            }
            continue

        rel = results["conditions"][condition]["relation_accuracy"]["per_seed"]
        cera = results["conditions"][condition]["cera_mean"]["per_seed"]
        results["conditions"][condition]["stats_vs_full"] = {
            "relation_accuracy": _paired_stats(rel, full_relation, seed=42),
            "cera_mean": _paired_stats(cera, full_cera, seed=99),
        }

    return results


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ablation matrix over several seeds.")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"),
                        help="Comma-separated seed list, e.g. 0,1,2,3,4")
    parser.add_argument("--cue-fraction", type=float, default=0.4, help="SQL cue fraction for relation accuracy")
    parser.add_argument("--duration-ms", type=float, default=50.0, help="Retrieval duration")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT, help="Directory for ablation outputs")
    parser.add_argument("--no-text", action="store_true", help="Disable text modality for this sweep")
    parser.add_argument("--ca3-exc", type=int, default=240)
    parser.add_argument("--ca3-inh", type=int, default=60)
    parser.add_argument("--dg-bridge-fanout", type=int, default=12)
    parser.add_argument("--dg-bridge-lr", type=float, default=0.02)
    parser.add_argument("--dg-output-dim", type=int, default=1200)
    parser.add_argument("--dg-target-sparsity", type=float, default=0.02)
    parser.add_argument("--ca1-n", type=int, default=320)
    parser.add_argument("--ca1-train-epochs", type=int, default=12)
    parser.add_argument("--ca1-train-lr", type=float, default=0.03)
    parser.add_argument("--ca1-relation-lr", type=float, default=0.05)
    args = parser.parse_args()

    run = run_ablation_suite(
        args.seeds,
        cue_fraction=args.cue_fraction,
        duration_ms=args.duration_ms,
        use_text=not args.no_text,
        ca3_exc=args.ca3_exc,
        ca3_inh=args.ca3_inh,
        dg_bridge_fanout=args.dg_bridge_fanout,
        dg_bridge_lr=args.dg_bridge_lr,
        dg_output_dim=args.dg_output_dim,
        dg_target_sparsity=args.dg_target_sparsity,
        ca1_n=args.ca1_n,
        ca1_train_epochs=args.ca1_train_epochs,
        ca1_train_lr=args.ca1_train_lr,
        ca1_relation_lr=args.ca1_relation_lr,
    )

    run_dir = args.output_root / run["run_id"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run["run_dir"] = str(run_dir.resolve())

    _write_json(run_dir / "summary.json", run)
    _write_text(run_dir / "summary.md", _render_markdown(run))

    _plot_metric_bars(run, fig_dir)
    _plot_cera_heatmaps(run, fig_dir)
    _plot_seed_deltas(run, fig_dir)

    _write_json(run_dir / "seed_metrics.json", {
        condition: payload["seed_metrics"] for condition, payload in run["conditions"].items()
    })

    print(json.dumps(
        _jsonable({
            "run_dir": run["run_dir"],
            "summary_md": str((run_dir / "summary.md").resolve()),
            "figures": sorted(str(p.resolve()) for p in fig_dir.glob("*.png")),
        }),
        indent=2,
    ))


if __name__ == "__main__":
    main()
