from __future__ import annotations

import argparse
import json
import math
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from spiking_multimodal_memory import MultiModalMemory


RELATIONS = ("WORKS_AT", "FRIENDS_WITH", "MANAGES", "REPORTS_TO")
NAMES = (
    "Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Hana",
    "Ivan", "Jules", "Kira", "Liam", "Mina", "Noah", "Omar", "Pia",
)
CITIES = ("NYC", "Seattle", "Austin", "Boston", "Denver", "Chicago")
ROLES = {
    "WORKS_AT": "Engineer",
    "FRIENDS_WITH": "Analyst",
    "MANAGES": "Manager",
    "REPORTS_TO": "Lead",
}
DEPTS = {
    "WORKS_AT": "AI",
    "FRIENDS_WITH": "Research",
    "MANAGES": "Ops",
    "REPORTS_TO": "Cloud",
}
TEXT_TEMPLATES = {
    "WORKS_AT": "{src} works at {org}",
    "FRIENDS_WITH": "{src} collaborates with {dst}",
    "MANAGES": "{src} manages {dst}",
    "REPORTS_TO": "{src} reports to {dst}",
}

DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/research_experiments") if Path("/kaggle/working").exists() else Path("outputs/research_experiments")

PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "ca3_exc": 240,
        "ca3_inh": 60,
        "dg_bridge_fanout": 12,
        "dg_bridge_lr": 0.02,
        "dg_output_dim": 1200,
        "dg_target_sparsity": 0.02,
        "dg_weight_scale": 0.08,
        "ca1_n": 320,
        "ca1_train_epochs": 12,
        "ca1_train_lr": 0.03,
        "ca1_relation_lr": 0.05,
        "ca1_relation_bins": 12,
        "ca1_relation_train_cue_fraction": 0.4,
        "ca1_relation_probe_fractions": (1.0, 0.4),
    },
    "tuned": {
        "ca3_exc": 288,
        "ca3_inh": 72,
        "dg_bridge_fanout": 16,
        "dg_bridge_lr": 0.03,
        "dg_output_dim": 1600,
        "dg_target_sparsity": 0.015,
        "dg_weight_scale": 0.06,
        "ca1_n": 360,
        "ca1_train_epochs": 20,
        "ca1_train_lr": 0.04,
        "ca1_relation_lr": 0.08,
        "ca1_relation_bins": 16,
        "ca1_relation_train_cue_fraction": 0.6,
        "ca1_relation_probe_fractions": (1.0, 0.6, 0.4),
    },
    "fast": {
        "ca3_exc": 160,
        "ca3_inh": 40,
        "dg_bridge_fanout": 10,
        "dg_bridge_lr": 0.02,
        "dg_output_dim": 800,
        "dg_target_sparsity": 0.015,
        "dg_weight_scale": 0.06,
        "ca1_n": 280,
        "ca1_train_epochs": 6,
        "ca1_train_lr": 0.05,
        "ca1_relation_lr": 0.06,
        "ca1_relation_bins": 12,
        "ca1_relation_train_cue_fraction": 0.5,
        "ca1_relation_probe_fractions": (1.0, 0.5, 0.35),
    },
}


@dataclass(frozen=True)
class EpisodeSpec:
    sql_row: Dict[str, Any]
    graph_edge: Tuple[int, str, int]
    text: Optional[str]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
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


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _bootstrap_ci(values: Sequence[float], confidence: float = 0.95, resamples: int = 2000,
                  seed: int = 0) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.RandomState(seed)
    samples = np.empty(resamples, dtype=float)
    for i in range(resamples):
        draw = rng.choice(arr, size=arr.size, replace=True)
        samples[i] = float(np.mean(draw))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _paired_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size == 0 or y.size == 0:
        return {"mean_diff": 0.0, "t_stat": 0.0, "p_value": 1.0}
    n = min(x.size, y.size)
    diff = x[:n] - y[:n]
    mean_diff = float(np.mean(diff)) if diff.size else 0.0
    if diff.size < 2:
        return {"mean_diff": mean_diff, "t_stat": 0.0, "p_value": 1.0}
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        return {"mean_diff": mean_diff, "t_stat": float("inf") if mean_diff != 0 else 0.0, "p_value": 0.0 if mean_diff != 0 else 1.0}
    t_stat = mean_diff / (sd / math.sqrt(diff.size))
    p_value = 1.0
    try:
        from scipy import stats  # type: ignore

        p_value = float(stats.t.sf(abs(t_stat), df=diff.size - 1) * 2.0)
    except Exception:
        pass
    return {"mean_diff": mean_diff, "t_stat": float(t_stat), "p_value": float(p_value)}


def _make_episode(index: int, rng: np.random.RandomState, use_text: bool = True) -> EpisodeSpec:
    relation = RELATIONS[index % len(RELATIONS)]
    src = index % 4
    tgt = (src + 1 + int(rng.randint(0, 3))) % 4
    name = NAMES[index % len(NAMES)]
    partner = NAMES[(index + 1) % len(NAMES)]
    org = f"{name.lower()}-{index % 997:03d}"
    sql_row = {
        "age": int(22 + ((index * 7) + rng.randint(0, 9)) % 45),
        "salary": int(rng.randint(45_000, 220_000)),
        "city": CITIES[(index + int(rng.randint(0, len(CITIES)))) % len(CITIES)],
        "role": ROLES[relation],
        "dept": DEPTS[relation],
    }
    text = TEXT_TEMPLATES[relation].format(src=name, dst=partner, org=org) if use_text else None
    return EpisodeSpec(sql_row=sql_row, graph_edge=(src, relation, tgt), text=text)


def _iter_corpus(size: int, seed: int = 42, use_text: bool = True) -> Iterable[EpisodeSpec]:
    rng = np.random.RandomState(seed)
    for index in range(size):
        yield _make_episode(index, rng, use_text=use_text)


def _memory_kwargs(preset: str = "baseline") -> Dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    return dict(PRESETS[preset])


def _coerce_override_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) > 1:
            return tuple(_coerce_override_value(part) for part in parts)
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_overrides(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Empty key.")
        overrides[key] = _coerce_override_value(raw_value)
    return overrides


def build_memory(seed: int = 42, use_text: bool = True, preset: str = "baseline",
                 text_use_pretrained: bool = False, keep_episode_history: bool = True,
                 enable_stdp: bool = True, use_dg_bridge: bool = True,
                 ca3_recurrent_scale: float = 1.0, **overrides: Any) -> MultiModalMemory:
    config = _memory_kwargs(preset)
    config.update(overrides)
    config.update({
        "use_text": use_text,
        "seed": seed,
        "text_use_pretrained": text_use_pretrained,
        "keep_episode_history": keep_episode_history,
        "enable_stdp": enable_stdp,
        "use_dg_bridge": use_dg_bridge,
        "ca3_recurrent_scale": ca3_recurrent_scale,
    })
    return MultiModalMemory(**config)


def _bridge_synapses(memory: MultiModalMemory):
    return [syn for pre_id, group in memory.ca3.input_synapses.items() if str(pre_id).startswith("dg:") for syn in group]


def _apply_bridge_delay_mode(memory: MultiModalMemory, mode: str, seed: int = 0) -> None:
    synapses = _bridge_synapses(memory)
    if not synapses:
        return
    if mode == "zero":
        for syn in synapses:
            syn.delay = 0.0
        return
    if mode == "random":
        rng = np.random.RandomState(seed)
        delays = np.asarray([syn.delay for syn in synapses], dtype=float)
        shuffled = rng.permutation(delays)
        for syn, delay in zip(synapses, shuffled):
            syn.delay = float(delay)
        return


def _sql_topk_score(memory: MultiModalMemory, retrieved: Dict[str, Any], record) -> float:
    sql_recon = retrieved.get("sql_reconstruction", {})
    target_key = getattr(record, "engram_id", None)
    if target_key is None:
        target_key = record.episode_id
    sql_support = memory.episode_targets.get(target_key, {}).get("sql_support", None)
    if sql_support is None:
        return 0.0
    sql_support = np.asarray(sql_support, dtype=float)
    k = int(max(1, (sql_support > 0).sum()))
    if k <= 0:
        return 0.0
    top = set(int(i) for i in np.argsort(list(sql_recon.values()))[-k:]) if sql_recon else set()
    support_idx = set(int(i) for i, v in enumerate(sql_support) if v > 0)
    return float(len(top & support_idx) / max(1, len(support_idx)))


def _text_cosine_score(retrieved: Dict[str, Any], memory: MultiModalMemory, record) -> float:
    text_recon = retrieved.get("text_reconstruction", {})
    target_key = getattr(record, "engram_id", None)
    if target_key is None:
        target_key = record.episode_id
    text_support = memory.episode_targets.get(target_key, {}).get("text_support", None)
    if text_support is None or len(text_support) == 0:
        return 0.0
    pred = np.array([text_recon.get(i, 0.0) for i in range(len(text_support))], dtype=float)
    tgt = np.array(text_support[: len(pred)], dtype=float)
    denom = (np.linalg.norm(pred) * np.linalg.norm(tgt)) + 1e-9
    return float(np.dot(pred, tgt) / denom) if denom > 0 else 0.0


def compute_cera(memory: MultiModalMemory, records: Sequence[Any], cue_fraction: float = 0.4) -> Dict[str, Any]:
    modalities = ["sql", "graph"]
    if memory.use_text:
        modalities.append("text")

    matrix: Dict[str, Dict[str, List[float]]] = {cue: {t: [] for t in modalities if t != cue} for cue in modalities}

    for record in records:
        for cue in modalities:
            cue_args = {"sql_cue": None, "graph_cue": None, "text_cue": None}
            if cue == "sql":
                cue_args["sql_cue"] = memory._partial_sql_row(record.sql_row, cue_fraction)
            elif cue == "graph":
                cue_args["graph_cue"] = record.graph_edge
            elif cue == "text":
                cue_args["text_cue"] = record.text

            retrieved = memory.retrieve(**cue_args, duration=50.0)
            graph_acc = memory.compute_graph_retrieval_accuracy(retrieved, record.graph_edge).get("edge_accuracy", 0.0)
            sql_score = _sql_topk_score(memory, retrieved, record)
            text_score = _text_cosine_score(retrieved, memory, record)
            target_scores = {
                "sql": sql_score,
                "graph": float(graph_acc),
                "text": text_score,
            }
            for target in matrix[cue]:
                matrix[cue][target].append(float(target_scores[target]))

    matrix_mean = {
        cue: {target: float(np.mean(values)) if values else 0.0 for target, values in targets.items()}
        for cue, targets in matrix.items()
    }
    off_diag = [
        score
        for cue, targets in matrix_mean.items()
        for target, score in targets.items()
        if cue != target
    ]
    return {
        "cera_matrix": matrix_mean,
        "cera_mean": float(np.mean(off_diag)) if off_diag else 0.0,
        "cue_fraction": float(cue_fraction),
    }


def _evaluate_record_set(memory: MultiModalMemory, records: Sequence[Any], cue_fraction: float = 0.4) -> Dict[str, Any]:
    relation_hits: List[float] = []
    graph_edge_hits: List[float] = []
    for record in records:
        partial_sql = memory._partial_sql_row(record.sql_row, cue_fraction)
        retrieved = memory.retrieve(sql_cue=partial_sql, duration=50.0)
        relation_hits.append(float(retrieved.get("relation_prediction") == record.graph_edge[1]))
        graph_edge_hits.append(float(memory.compute_graph_retrieval_accuracy(retrieved, record.graph_edge).get("edge_accuracy", 0.0)))

    cera = compute_cera(memory, records, cue_fraction=cue_fraction)
    separation = memory.evaluate_separation(duration=50.0)
    false_retrieval = memory.evaluate_false_retrieval_rate(duration=50.0)
    interference = memory.evaluate_interference()
    return {
        "relation_accuracy": float(np.mean(relation_hits)) if relation_hits else 0.0,
        "graph_edge_accuracy": float(np.mean(graph_edge_hits)) if graph_edge_hits else 0.0,
        "cera_mean": float(cera["cera_mean"]),
        "cera_matrix": cera["cera_matrix"],
        "separation_margin_mean": float(separation.get("separation_margin_mean", 0.0)),
        "false_retrieval_rate": float(false_retrieval.get("false_retrieval_rate", false_retrieval.get("false_retrieval_count", 0.0))),
        "mean_retention": float(interference.get("mean_retention", 0.0)),
        "oldest_retention": float(interference.get("oldest_retention", 0.0)),
        "newest_retention": float(interference.get("newest_retention", 0.0)),
    }


def _sample_indices(size: int, sample_size: int, seed: int) -> List[int]:
    if size <= 0:
        return []
    sample_size = min(max(1, sample_size), size)
    rng = np.random.RandomState(seed)
    return sorted(int(i) for i in rng.choice(size, size=sample_size, replace=False))


def run_capacity_experiment(checkpoints: Sequence[int], preset: str = "fast", seed: int = 42,
                            use_text: bool = True, text_use_pretrained: bool = False,
                            sample_size: int = 64, consolidate: bool = False,
                            memory_overrides: Optional[Dict[str, Any]] = None,
                            output_dir: Optional[Path] = None) -> Dict[str, Any]:
    checkpoints = sorted(int(c) for c in checkpoints if int(c) > 0)
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")
    max_n = checkpoints[-1]
    memory_config = {
        "seed": seed,
        "use_text": use_text,
        "preset": preset,
        "text_use_pretrained": text_use_pretrained,
        "keep_episode_history": False,
    }
    memory_config.update(memory_overrides or {})
    memory = build_memory(**memory_config)

    sample_sets = {checkpoint: set(_sample_indices(checkpoint, sample_size, seed + checkpoint)) for checkpoint in checkpoints}
    anchor_banks: Dict[int, Dict[int, Dict[str, Any]]] = {checkpoint: {} for checkpoint in checkpoints}
    results: Dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "preset": preset,
            "use_text": use_text,
            "text_use_pretrained": text_use_pretrained,
            "sample_size": sample_size,
            "consolidate": consolidate,
            "checkpoints": checkpoints,
        },
        "checkpoints": {},
    }

    checkpoint_set = set(checkpoints)
    for index, episode in enumerate(_iter_corpus(max_n, seed=seed, use_text=use_text)):
        encoded = memory.encode_episode(
            episode.sql_row,
            episode.graph_edge,
            text=episode.text,
            episode_time=float(index),
            consolidate=consolidate,
        )
        for checkpoint in checkpoints:
            if index in sample_sets[checkpoint]:
                anchor_banks[checkpoint][index] = {
                    "episode_index": index,
                    "sql_row": dict(episode.sql_row),
                    "graph_edge": tuple(episode.graph_edge),
                    "text": episode.text,
                    "ca3_assembly": set(encoded["ca3_active"]),
                }

        if (index + 1) in checkpoint_set:
            checkpoint = index + 1
            anchors = [anchor_banks[checkpoint][idx] for idx in sorted(anchor_banks[checkpoint])]
            metrics = _evaluate_capacity_anchors(memory, anchors)
            results["checkpoints"][str(checkpoint)] = metrics

    if output_dir is not None:
        _write_json(output_dir / "capacity.json", results)
    return results


def _evaluate_capacity_anchors(memory: MultiModalMemory, anchors: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not anchors:
        return {
            "anchor_count": 0,
            "mean_retention": 0.0,
            "std_retention": 0.0,
            "mean_best_impostor_overlap": 0.0,
            "mean_margin": 0.0,
            "false_retrieval_rate": 0.0,
            "relation_accuracy": 0.0,
            "oldest_retention": 0.0,
            "newest_retention": 0.0,
        }

    anchors = sorted(anchors, key=lambda a: a["episode_index"])
    retention_scores: List[float] = []
    impostor_scores: List[float] = []
    relation_hits: List[float] = []
    for idx, anchor in enumerate(anchors):
        retrieved = memory.retrieve(sql_cue=anchor["sql_row"], graph_cue=anchor["graph_edge"], text_cue=anchor["text"], duration=50.0)
        target_overlap = _jaccard(retrieved["ca3_active"], anchor["ca3_assembly"])
        best_impostor = 0.0
        for jdx, other in enumerate(anchors):
            if jdx == idx:
                continue
            best_impostor = max(best_impostor, _jaccard(retrieved["ca3_active"], other["ca3_assembly"]))
        retention_scores.append(float(target_overlap))
        impostor_scores.append(float(best_impostor))
        relation_hits.append(float(retrieved.get("relation_prediction") == anchor["graph_edge"][1]))

    retention = np.asarray(retention_scores, dtype=float)
    impostors = np.asarray(impostor_scores, dtype=float)
    return {
        "anchor_count": int(len(anchors)),
        "mean_retention": float(retention.mean()),
        "std_retention": float(retention.std(ddof=1)) if retention.size > 1 else 0.0,
        "mean_best_impostor_overlap": float(impostors.mean()),
        "mean_margin": float(np.mean(retention - impostors)),
        "false_retrieval_rate": float(np.mean(impostors >= retention)),
        "relation_accuracy": float(np.mean(relation_hits)) if relation_hits else 0.0,
        "oldest_retention": float(retention[0]),
        "newest_retention": float(retention[-1]),
    }


def _condition_config(name: str) -> Dict[str, Any]:
    conditions: Dict[str, Dict[str, Any]] = {
        "full_model": {"enable_stdp": True, "use_dg_bridge": True, "ca3_recurrent_scale": 1.0, "bridge_delay_mode": "structured"},
        "no_delay": {"enable_stdp": True, "use_dg_bridge": True, "ca3_recurrent_scale": 1.0, "bridge_delay_mode": "zero"},
        "random_delay": {"enable_stdp": True, "use_dg_bridge": True, "ca3_recurrent_scale": 1.0, "bridge_delay_mode": "random"},
        "no_stdp": {"enable_stdp": False, "use_dg_bridge": True, "ca3_recurrent_scale": 1.0, "bridge_delay_mode": "structured"},
        "no_dg": {"enable_stdp": True, "use_dg_bridge": False, "ca3_recurrent_scale": 1.0, "bridge_delay_mode": "structured"},
        "no_ca3_recurrence": {"enable_stdp": True, "use_dg_bridge": True, "ca3_recurrent_scale": 0.0, "bridge_delay_mode": "structured"},
    }
    if name not in conditions:
        raise ValueError(f"Unknown ablation condition: {name}")
    return dict(conditions[name])


def run_ablation_suite(seeds: Sequence[int], corpus_size: int = 64, preset: str = "tuned",
                       use_text: bool = True, text_use_pretrained: bool = False,
                       consolidate: bool = True, cue_fraction: float = 0.4,
                       conditions: Sequence[str] = ("full_model", "no_delay", "random_delay", "no_stdp", "no_dg", "no_ca3_recurrence"),
                       memory_overrides: Optional[Dict[str, Any]] = None,
                       output_dir: Optional[Path] = None) -> Dict[str, Any]:
    seeds = [int(s) for s in seeds]
    corpus_size = int(corpus_size)
    per_condition: Dict[str, List[Dict[str, Any]]] = {name: [] for name in conditions}

    for seed in seeds:
        corpus = list(_iter_corpus(corpus_size, seed=seed, use_text=use_text))
        for condition in conditions:
            config = _condition_config(condition)
            memory_config = {
                "seed": seed,
                "use_text": use_text,
                "preset": preset,
                "text_use_pretrained": text_use_pretrained,
                "keep_episode_history": True,
                "enable_stdp": config.pop("enable_stdp"),
                "use_dg_bridge": config.pop("use_dg_bridge"),
                "ca3_recurrent_scale": config.pop("ca3_recurrent_scale"),
            }
            memory_config.update(memory_overrides or {})
            memory = build_memory(**memory_config)
            delay_mode = config.pop("bridge_delay_mode", "structured")
            if delay_mode != "structured":
                _apply_bridge_delay_mode(memory, delay_mode, seed=seed + 101)

            for episode_index, episode in enumerate(corpus):
                memory.encode_episode(
                    episode.sql_row,
                    episode.graph_edge,
                    text=episode.text,
                    episode_time=float(episode_index),
                    consolidate=consolidate,
                )

            eval_records = list(memory.episode_records)
            metrics = _evaluate_record_set(memory, eval_records, cue_fraction=cue_fraction)
            metrics["seed"] = seed
            per_condition[condition].append(metrics)

    summary: Dict[str, Any] = {
        "metadata": {
            "seeds": seeds,
            "corpus_size": corpus_size,
            "preset": preset,
            "use_text": use_text,
            "text_use_pretrained": text_use_pretrained,
            "consolidate": consolidate,
            "cue_fraction": cue_fraction,
            "conditions": list(conditions),
        },
        "conditions": {},
        "paired_tests": {},
    }

    baseline = per_condition.get("full_model", [])
    baseline_relation = [float(item["relation_accuracy"]) for item in baseline]
    baseline_cera = [float(item["cera_mean"]) for item in baseline]

    for condition, runs in per_condition.items():
        relation = [float(item["relation_accuracy"]) for item in runs]
        cera = [float(item["cera_mean"]) for item in runs]
        false_retrieval = [float(item["false_retrieval_rate"]) for item in runs]
        separation = [float(item["separation_margin_mean"]) for item in runs]
        retention = [float(item["mean_retention"]) for item in runs]
        summary["conditions"][condition] = {
            "per_seed": runs,
            "relation_accuracy": _summarise_distribution(relation, seed=17),
            "cera_mean": _summarise_distribution(cera, seed=29),
            "false_retrieval_rate": _summarise_distribution(false_retrieval, seed=41),
            "separation_margin_mean": _summarise_distribution(separation, seed=53),
            "mean_retention": _summarise_distribution(retention, seed=67),
        }
        if condition != "full_model":
            summary["paired_tests"][condition] = {
                "relation_vs_full": _paired_t_test(relation, baseline_relation),
                "cera_vs_full": _paired_t_test(cera, baseline_cera),
            }

    if output_dir is not None:
        _write_json(output_dir / "ablation.json", summary)
    return summary


def _summarise_distribution(values: Sequence[float], seed: int = 0) -> Dict[str, Any]:
    mean, std = _mean_std(values)
    ci_low, ci_high = _bootstrap_ci(values, seed=seed)
    return {
        "mean": mean,
        "std": std,
        "ci95": [ci_low, ci_high],
        "n": int(len(values)),
    }


def _run_writeup(capacity: Dict[str, Any], ablation: Dict[str, Any], output_dir: Path) -> Path:
    lines: List[str] = []
    lines.append("# Novelty and Causality Report")
    lines.append("")
    lines.append(f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Host: `{platform.platform()}`")
    lines.append("")
    lines.append("## Claim Framing")
    lines.append("")
    lines.append("This report is generated from the pure EC->DG->CA3->CA1 path with direct CA3 shortcuts removed.")
    lines.append("It is intended to support, not overstate, the mechanistic story.")
    lines.append("")
    lines.append("## Capacity")
    lines.append("")
    lines.append("| Episodes | Mean retention | Mean relation acc. | False retrieval |")
    lines.append("| --- | --- | --- | --- |")
    for checkpoint, metrics in capacity.get("checkpoints", {}).items():
        lines.append(
            f"| {checkpoint} | {metrics['mean_retention']:.3f} | {metrics['relation_accuracy']:.3f} | {metrics['false_retrieval_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Ablation Summary")
    lines.append("")
    conditions = ablation.get("conditions", {})
    if conditions:
        lines.append("| Condition | Relation acc. | CERA | False retrieval | Separation margin |")
        lines.append("| --- | --- | --- | --- | --- |")
        for condition, payload in conditions.items():
            rel = payload["relation_accuracy"]
            cera = payload["cera_mean"]
            false_ret = payload["false_retrieval_rate"]
            sep = payload["separation_margin_mean"]
            lines.append(
                f"| {condition} | {rel['mean']:.3f} ± {rel['std']:.3f} | {cera['mean']:.3f} ± {cera['std']:.3f} | {false_ret['mean']:.3f} ± {false_ret['std']:.3f} | {sep['mean']:.3f} ± {sep['std']:.3f} |"
            )
        lines.append("")
        lines.append("## Significance")
        lines.append("")
        for condition, payload in ablation.get("paired_tests", {}).items():
            rel = payload["relation_vs_full"]
            cera = payload["cera_vs_full"]
            lines.append(
                f"- `{condition}` vs `full_model`: relation diff {rel['mean_diff']:.3f}, t={rel['t_stat']:.3f}, p={rel['p_value']:.4g}; CERA diff {cera['mean_diff']:.3f}, t={cera['t_stat']:.3f}, p={cera['p_value']:.4g}."
            )
        lines.append("")
    else:
        lines.append("Ablation results were not included in this run.")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If `no_delay` or `random_delay` underperform `full_model`, the temporal bridge is contributing causally rather than cosmetically.")
    lines.append("- If `no_dg` and `no_ca3_recurrence` drop sharply, separation and attractor completion are doing real work.")
    lines.append("- If the tuned condition improves relation accuracy and CERA over the baseline preset, the readout and bridge are stronger, not just more parameterized.")
    lines.append("")
    report = "\n".join(lines)
    path = output_dir / "novelty_causality_report.md"
    _write_text(path, report)
    return path


def _plot_capacity(capacity: Dict[str, Any], output_dir: Path) -> None:
    checkpoints = [int(k) for k in capacity.get("checkpoints", {}).keys()]
    checkpoints.sort()
    if not checkpoints:
        return
    retention = [capacity["checkpoints"][str(c)]["mean_retention"] for c in checkpoints]
    relation = [capacity["checkpoints"][str(c)]["relation_accuracy"] for c in checkpoints]
    false_ret = [capacity["checkpoints"][str(c)]["false_retrieval_rate"] for c in checkpoints]
    plt.figure(figsize=(7, 4))
    plt.plot(checkpoints, retention, marker="o", label="Retention")
    plt.plot(checkpoints, relation, marker="o", label="Relation acc.")
    plt.plot(checkpoints, false_ret, marker="o", label="False retrieval")
    plt.xscale("log")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("Capacity / interference scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "capacity_curve.png", dpi=160)
    plt.close()


def _plot_ablation(ablation: Dict[str, Any], output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    conditions = list(ablation.get("conditions", {}).keys())
    if not conditions:
        return
    relation = [ablation["conditions"][c]["relation_accuracy"] for c in conditions]
    cera = [ablation["conditions"][c]["cera_mean"] for c in conditions]
    false_ret = [ablation["conditions"][c]["false_retrieval_rate"] for c in conditions]
    separation = [ablation["conditions"][c]["separation_margin_mean"] for c in conditions]

    def _bar(values: List[Dict[str, Any]], filename: str, title: str, ylabel: str) -> None:
        means = [v["mean"] for v in values]
        errs = [[v["mean"] - v["ci95"][0], v["ci95"][1] - v["mean"]] for v in values]
        yerr = np.array(errs).T if errs else None
        plt.figure(figsize=(10, 4))
        plt.bar(conditions, means, yerr=yerr, capsize=4)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()

    _bar(relation, "ablation_relation_accuracy.png", "Relation accuracy by ablation", "Accuracy")
    _bar(cera, "ablation_cera.png", "CERA by ablation", "CERA")
    plt.figure(figsize=(10, 4))
    plt.bar(conditions, [v["mean"] for v in false_ret])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("False retrieval")
    plt.title("False retrieval by ablation")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_false_retrieval.png", dpi=160)
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.bar(conditions, [v["mean"] for v in separation])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Separation margin")
    plt.title("Separation margin by ablation")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_separation.png", dpi=160)
    plt.close()


def _make_run_dir(output_root: Path, mode: str) -> Path:
    run_id = _utc_run_id()
    run_dir = output_root / f"{run_id}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "LATEST.txt").write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaling, ablation, and report generation experiments.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_capacity = subparsers.add_parser("capacity", help="Run retention / interference scaling experiments.")
    p_capacity.add_argument("--checkpoints", nargs="+", type=int, default=[100, 100000], help="Episode counts to evaluate.")
    p_capacity.add_argument("--preset", type=str, default="fast", choices=sorted(PRESETS.keys()))
    p_capacity.add_argument("--seed", type=int, default=42)
    p_capacity.add_argument("--sample-size", type=int, default=64)
    p_capacity.add_argument("--use-text", action="store_true", default=True)
    p_capacity.add_argument("--no-text", action="store_true", help="Disable text modality for speed.")
    p_capacity.add_argument("--text-pretrained", action="store_true", help="Enable pretrained text encoder if available.")
    p_capacity.add_argument("--consolidate", action="store_true", default=False, help="Enable replay consolidation during encoding.")
    p_capacity.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                            help="Override a memory config field, e.g. dg_bridge_lr=0.01 or ca1_relation_probe_fractions=1.0,0.5,0.25.")
    p_capacity.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p_ablation = subparsers.add_parser("ablation", help="Run ablation matrix with seeds and confidence intervals.")
    p_ablation.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p_ablation.add_argument("--corpus-size", type=int, default=64)
    p_ablation.add_argument("--preset", type=str, default="tuned", choices=sorted(PRESETS.keys()))
    p_ablation.add_argument("--sample-cue-fraction", type=float, default=0.4)
    p_ablation.add_argument("--use-text", action="store_true", default=True)
    p_ablation.add_argument("--no-text", action="store_true", help="Disable text modality for speed.")
    p_ablation.add_argument("--text-pretrained", action="store_true", help="Enable pretrained text encoder if available.")
    p_ablation.add_argument("--consolidate", action="store_true", default=True)
    p_ablation.add_argument("--no-consolidate", action="store_true", help="Disable replay consolidation.")
    p_ablation.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                            help="Override a memory config field, e.g. dg_bridge_lr=0.01 or dg_target_sparsity=0.01.")
    p_ablation.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p_report = subparsers.add_parser("report", help="Generate a markdown report from prior experiment JSON outputs.")
    p_report.add_argument("--input-root", type=Path, required=True, help="Directory containing capacity.json and ablation.json.")
    p_report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p_all = subparsers.add_parser("all", help="Run capacity, ablation, plots, and report generation.")
    p_all.add_argument("--capacity-checkpoints", nargs="+", type=int, default=[100, 100000])
    p_all.add_argument("--capacity-preset", type=str, default="fast", choices=sorted(PRESETS.keys()))
    p_all.add_argument("--capacity-seed", type=int, default=42)
    p_all.add_argument("--capacity-sample-size", type=int, default=64)
    p_all.add_argument("--capacity-consolidate", action="store_true", default=False)
    p_all.add_argument("--ablation-seeds", nargs="+", type=int, default=list(range(10)))
    p_all.add_argument("--ablation-corpus-size", type=int, default=64)
    p_all.add_argument("--ablation-preset", type=str, default="tuned", choices=sorted(PRESETS.keys()))
    p_all.add_argument("--ablation-consolidate", action="store_true", default=True)
    p_all.add_argument("--ablation-no-consolidate", action="store_true")
    p_all.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_all.add_argument("--use-text", action="store_true", default=True)
    p_all.add_argument("--no-text", action="store_true")
    p_all.add_argument("--text-pretrained", action="store_true")
    p_all.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                       help="Override a memory config field for both capacity and ablation runs.")

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(output_root, args.mode)
    memory_overrides = _parse_overrides(getattr(args, "override", []))
    manifest = {
        "run_id": run_dir.name,
        "mode": args.mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "seed": getattr(args, "seed", None),
        "memory_overrides": memory_overrides,
    }

    if args.mode == "capacity":
        use_text = bool(args.use_text) and not bool(args.no_text)
        capacity = run_capacity_experiment(
            checkpoints=args.checkpoints,
            preset=args.preset,
            seed=args.seed,
            use_text=use_text,
            text_use_pretrained=bool(args.text_pretrained),
            sample_size=args.sample_size,
            consolidate=bool(args.consolidate),
            memory_overrides=memory_overrides,
            output_dir=run_dir,
        )
        _plot_capacity(capacity, run_dir)
        report_path = _run_writeup(capacity, {"conditions": {}, "paired_tests": {}}, run_dir)
        _write_json(run_dir / "manifest.json", manifest)
        _write_text(run_dir / "summary.md", report_path.read_text(encoding="utf-8"))
        print(json.dumps({"run_dir": str(run_dir), "capacity": capacity["checkpoints"]}, indent=2))
        return

    if args.mode == "ablation":
        use_text = bool(args.use_text) and not bool(args.no_text)
        consolidate = bool(args.consolidate) and not bool(args.no_consolidate)
        ablation = run_ablation_suite(
            seeds=args.seeds,
            corpus_size=args.corpus_size,
            preset=args.preset,
            use_text=use_text,
            text_use_pretrained=bool(args.text_pretrained),
            consolidate=consolidate,
            cue_fraction=args.sample_cue_fraction,
            memory_overrides=memory_overrides,
            output_dir=run_dir,
        )
        _plot_ablation(ablation, run_dir)
        report_path = _run_writeup({"checkpoints": {}}, ablation, run_dir)
        _write_json(run_dir / "manifest.json", manifest)
        _write_text(run_dir / "summary.md", report_path.read_text(encoding="utf-8"))
        print(json.dumps({"run_dir": str(run_dir), "ablation_conditions": list(ablation["conditions"].keys())}, indent=2))
        return

    if args.mode == "report":
        input_root = Path(args.input_root)
        capacity_path = input_root / "capacity.json"
        ablation_path = input_root / "ablation.json"
        capacity = json.loads(capacity_path.read_text(encoding="utf-8")) if capacity_path.exists() else {"checkpoints": {}}
        ablation = json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.exists() else {"conditions": {}, "paired_tests": {}}
        report_path = _run_writeup(capacity, ablation, run_dir)
        print(json.dumps({"run_dir": str(run_dir), "report": str(report_path)}, indent=2))
        return

    if args.mode == "all":
        use_text = bool(args.use_text) and not bool(args.no_text)
        capacity = run_capacity_experiment(
            checkpoints=args.capacity_checkpoints,
            preset=args.capacity_preset,
            seed=args.capacity_seed,
            use_text=use_text,
            text_use_pretrained=bool(args.text_pretrained),
            sample_size=args.capacity_sample_size,
            consolidate=bool(args.capacity_consolidate),
            memory_overrides=memory_overrides,
            output_dir=run_dir,
        )
        consolidate = bool(args.ablation_consolidate) and not bool(args.ablation_no_consolidate)
        ablation = run_ablation_suite(
            seeds=args.ablation_seeds,
            corpus_size=args.ablation_corpus_size,
            preset=args.ablation_preset,
            use_text=use_text,
            text_use_pretrained=bool(args.text_pretrained),
            consolidate=consolidate,
            memory_overrides=memory_overrides,
            output_dir=run_dir,
        )
        _plot_capacity(capacity, run_dir)
        _plot_ablation(ablation, run_dir)
        report_path = _run_writeup(capacity, ablation, run_dir)
        _write_json(run_dir / "manifest.json", manifest)
        _write_text(run_dir / "summary.md", report_path.read_text(encoding="utf-8"))
        print(json.dumps({"run_dir": str(run_dir), "capacity": capacity["checkpoints"], "ablation_conditions": list(ablation["conditions"].keys())}, indent=2))
        return


if __name__ == "__main__":
    main()
