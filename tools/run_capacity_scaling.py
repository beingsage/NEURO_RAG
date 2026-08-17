#!/usr/bin/env python3
"""
Scale the corpus size and measure retention/interference.

Protocol:
  - exact model at N=100
  - streaming proxy at N>=1000, including N=100000

The benchmark uses partial-SQL cueing so interference becomes measurable as
the corpus grows. Results are written as versioned JSON/Markdown plus plots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spiking_multimodal_memory import DentateGyrus, GraphEncoder, MultiModalMemory, SQLEncoder, TextEncoder


RUN_ROOT = Path("outputs/capacity_runs")
DEFAULT_SIZES = (100, 1_000, 10_000, 100_000)


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


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


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


@dataclass(frozen=True)
class EpisodeSpec:
    sql_row: Dict[str, Any]
    graph_edge: Tuple[int, str, int]
    text: Optional[str]


class SyntheticCorpusFactory:
    """Generate a deterministic high-diversity synthetic corpus."""

    def __init__(self, seed: int = 0, use_text: bool = True):
        self.seed = seed
        self.use_text = use_text
        self.relations = tuple(GraphEncoder.RELATION_DELAYS.keys())
        self.city_count = 512
        self.role_count = 512
        self.dept_count = 512

    def make(self, idx: int) -> EpisodeSpec:
        age = int(20 + ((idx * 7 + self.seed * 13) % 46))
        salary = int(35_000 + ((idx * 7_919 + self.seed * 101) % 215_000))
        city = f"city_{(idx * 11 + self.seed) % self.city_count}"
        role = f"role_{(idx * 13 + self.seed * 3) % self.role_count}"
        dept = f"dept_{(idx * 17 + self.seed * 5) % self.dept_count}"
        sql_row = {
            "age": age,
            "salary": salary,
            "city": city,
            "role": role,
            "dept": dept,
        }

        src = int((idx + self.seed) % 4)
        rel = self.relations[(idx * 3 + self.seed) % len(self.relations)]
        tgt = int((src + 1 + ((idx // 7) + self.seed) % 3) % 4)
        graph_edge = (src, rel, tgt)

        if not self.use_text:
            text = None
        else:
            text_templates = {
                "WORKS_AT": "Agent {idx} works at org {org}",
                "FRIENDS_WITH": "Agent {idx} is friends with agent {peer}",
                "MANAGES": "Agent {idx} manages team {team}",
                "REPORTS_TO": "Agent {idx} reports to lead {lead}",
            }
            template = text_templates.get(rel, "Episode {idx} relates to {peer}")
            text = template.format(
                idx=idx,
                org=f"org_{(idx * 19 + self.seed) % 2048}",
                peer=f"peer_{(idx * 23 + self.seed) % 2048}",
                team=f"team_{(idx * 29 + self.seed) % 2048}",
                lead=f"lead_{(idx * 31 + self.seed) % 2048}",
            )

        return EpisodeSpec(sql_row=sql_row, graph_edge=graph_edge, text=text)


def _partial_sql_row(sql_row: Dict[str, Any], fraction: float) -> Dict[str, Any]:
    fields = ["age", "salary", "city", "role", "dept"]
    keep_n = max(1, int(round(len(fields) * float(np.clip(fraction, 0.0, 1.0)))))
    partial = dict(sql_row)
    for field in fields[keep_n:]:
        partial.pop(field, None)
    return partial


def _spikes_to_counts(spikes: Mapping[int, Sequence[float]], dim: int) -> np.ndarray:
    counts = np.zeros(dim, dtype=np.uint8)
    for nid, times in spikes.items():
        if 0 <= int(nid) < dim and times:
            counts[int(nid)] = 1
    return counts


def _support_signature(bits: np.ndarray, *, seed: int = 0) -> int:
    payload = bits.tobytes() + seed.to_bytes(4, byteorder="little", signed=False)
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def _assembly_from_signature(signature: int, *, ca3_size: int = 240, active_k: int = 24) -> np.ndarray:
    rng = np.random.RandomState(signature)
    k = max(1, min(ca3_size, active_k))
    return np.asarray(sorted(rng.choice(ca3_size, size=k, replace=False)), dtype=np.uint16)


def _memory_footprint_mb(num_episodes: int, *, sql_dim: int = 100, graph_dim: int = 80,
                         text_dim: int = 100, ca3_size: int = 240,
                         assembly_k: int = 24) -> float:
    bits = sql_dim + graph_dim + text_dim + ca3_size
    bytes_est = num_episodes * (bits / 8.0 + assembly_k * 2.0)
    return float(bytes_est / 1e6)


class FastCapacityProxy:
    """Bucketed streaming proxy for large-N capacity tests.

    The proxy keeps the same encoders and DG stage, but replaces the full CA3
    simulation with a stable sparse assembly hash so 100k-scale runs remain
    tractable.
    """

    def __init__(self, seed: int = 0, use_text: bool = True,
                 dg_output_dim: int = 1200, dg_target_sparsity: float = 0.02,
                 bucket_bits: int = 11, ca3_size: int = 240, assembly_k: int = 24):
        self.seed = seed
        self.use_text = use_text
        self.bucket_count = 1 << int(bucket_bits)
        self.ca3_size = int(ca3_size)
        self.assembly_k = int(assembly_k)
        self.sql_enc = SQLEncoder(seed=seed)
        self.graph_enc = GraphEncoder(seed=seed)
        self.text_enc = TextEncoder(seed=seed + 17, use_pretrained=False) if use_text else None
        ec_dim = 280 if use_text else 180
        self.dg = DentateGyrus(
            input_dim=ec_dim,
            output_dim=dg_output_dim,
            target_sparsity=dg_target_sparsity,
            seed=seed + 31,
        )
        self.records: List[Dict[str, Any]] = []
        self.buckets: Dict[int, List[int]] = defaultdict(list)

    def _encode_support(self, sql_row: Dict[str, Any], graph_edge: Tuple[int, str, int],
                        text: Optional[str], cue_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
        partial_sql = _partial_sql_row(sql_row, cue_fraction)
        sql_spikes = self.sql_enc.encode(partial_sql)
        graph_spikes = self.graph_enc.encode(graph_edge)
        text_spikes: Dict[int, List[float]] = {}
        if text and self.text_enc:
            text_spikes = self.text_enc.encode(text)

        x_sql = _spikes_to_counts(sql_spikes, 100)
        x_graph = _spikes_to_counts(graph_spikes, 80)
        x_text = _spikes_to_counts(text_spikes, 100) if (text_spikes and self.use_text) else np.zeros(100, dtype=np.uint8)
        support = np.concatenate([x_sql, x_graph, x_text]).astype(np.uint8)
        dg_active = self.dg.encode(support.astype(float))
        return support, np.asarray(sorted(dg_active), dtype=np.uint16)

    def _bucket(self, dg_active: np.ndarray) -> int:
        payload = dg_active.tobytes() + self.seed.to_bytes(4, byteorder="little", signed=False)
        return int(hashlib.sha1(payload).hexdigest()[:8], 16) % self.bucket_count

    def _assembly(self, sql_row: Dict[str, Any], graph_edge: Tuple[int, str, int], text: Optional[str]) -> np.ndarray:
        payload = repr((tuple(sorted(sql_row.items())), tuple(graph_edge), text or "", self.seed))
        sig = int(hashlib.sha1(payload.encode()).hexdigest()[:8], 16)
        return _assembly_from_signature(sig, ca3_size=self.ca3_size, active_k=self.assembly_k)

    def store(self, idx: int, episode: EpisodeSpec, cue_fraction: float) -> None:
        partial_support, dg_active = self._encode_support(episode.sql_row, episode.graph_edge, episode.text, cue_fraction)
        bucket = self._bucket(dg_active)
        self.records.append({
            "episode_id": idx,
            "sql_row": dict(episode.sql_row),
            "graph_edge": tuple(episode.graph_edge),
            "text": episode.text,
            "partial_support": partial_support,
            "dg_active": dg_active,
            "bucket": bucket,
            "assembly": self._assembly(episode.sql_row, episode.graph_edge, episode.text),
        })
        self.buckets[bucket].append(idx)

    def retrieve(self, episode: EpisodeSpec, cue_fraction: float) -> Dict[str, Any]:
        query_support, dg_active = self._encode_support(episode.sql_row, episode.graph_edge, episode.text, cue_fraction)
        bucket = self._bucket(dg_active)
        candidate_ids = self.buckets.get(bucket, [])
        if not candidate_ids:
            return {
                "retrieved_id": None,
                "retrieved_assembly": np.zeros(0, dtype=np.uint16),
                "target_overlap": 0.0,
                "best_impostor_overlap": 0.0,
                "false_retrieval": True,
            }

        best_idx = None
        best_score = -1.0
        best_overlap = 0.0
        best_impostor = 0.0

        q_support = query_support.astype(bool)
        for record_idx in candidate_ids:
            record = self.records[record_idx]
            cand_support = record["partial_support"].astype(bool)
            inter = int(np.logical_and(q_support, cand_support).sum())
            union = int(np.logical_or(q_support, cand_support).sum())
            score = inter / union if union > 0 else 0.0
            if score > best_score or (abs(score - best_score) <= 1e-12 and (best_idx is None or record_idx > best_idx)):
                if best_idx is not None:
                    best_impostor = max(best_impostor, best_score)
                best_score = score
                best_idx = record_idx
                best_overlap = score
            else:
                best_impostor = max(best_impostor, score)

        retrieved = self.records[best_idx] if best_idx is not None else None
        retrieved_assembly = retrieved["assembly"] if retrieved is not None else np.zeros(0, dtype=np.uint16)
        target_assembly = self.records[len(self.records) - 1]["assembly"] if False else None
        # Target assembly is resolved by the caller; we only return the candidate info here.
        return {
            "retrieved_id": best_idx,
            "retrieved_assembly": retrieved_assembly,
            "best_score": float(best_score),
            "best_impostor_overlap": float(best_impostor),
            "false_retrieval": bool(best_idx is None),
            "candidate_count": len(candidate_ids),
        }


def _evaluate_exact_model(n: int, seed: int, cue_fraction: float, use_text: bool,
                          ca3_exc: int, ca3_inh: int, ca1_n: int,
                          ca1_train_epochs: int, ca1_train_lr: float, ca1_relation_lr: float,
                          dg_output_dim: int, dg_target_sparsity: float,
                          dg_bridge_fanout: int, dg_bridge_lr: float) -> Dict[str, Any]:
    factory = SyntheticCorpusFactory(seed=seed, use_text=use_text)
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

    for idx in range(n):
        ep = factory.make(idx)
        mem.encode_episode(ep.sql_row, ep.graph_edge, text=ep.text, episode_time=float(idx), consolidate=True)

    target_overlaps: List[float] = []
    best_impostor_overlaps: List[float] = []
    margins: List[float] = []
    hits = 0

    for idx, record in enumerate(mem.episode_records):
        partial_sql = mem._partial_sql_row(record.sql_row, cue_fraction)
        retrieved = mem.retrieve(sql_cue=partial_sql, duration=50.0)
        target_overlap = _jaccard(retrieved["ca3_active"], record.ca3_assembly)
        impostor_scores = [
            _jaccard(retrieved["ca3_active"], other.ca3_assembly)
            for other_idx, other in enumerate(mem.episode_records)
            if other_idx != idx
        ]
        best_impostor = max(impostor_scores) if impostor_scores else 0.0
        target_overlaps.append(target_overlap)
        best_impostor_overlaps.append(best_impostor)
        margins.append(target_overlap - best_impostor)
        hits += int(target_overlap >= best_impostor)

    retention = np.asarray(target_overlaps, dtype=float)
    return {
        "mode": "exact",
        "n": int(n),
        "seed": int(seed),
        "cue_fraction": float(cue_fraction),
        "mean_retention": float(np.mean(retention)) if retention.size else 0.0,
        "std_retention": float(np.std(retention, ddof=1)) if retention.size > 1 else 0.0,
        "oldest_retention": float(retention[0]) if retention.size else 0.0,
        "newest_retention": float(retention[-1]) if retention.size else 0.0,
        "mean_best_impostor_overlap": float(np.mean(best_impostor_overlaps)) if best_impostor_overlaps else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "false_retrieval_rate": float(1.0 - (hits / max(1, len(mem.episode_records)))),
        "anchor_count": int(len(mem.episode_records)),
        "memory_footprint_mb": _memory_footprint_mb(len(mem.episode_records), ca3_size=ca3_exc),
        "retention_by_quartile": _quartile_means(retention),
    }


def _select_anchors(n: int, max_anchors: int) -> np.ndarray:
    if n <= max_anchors:
        return np.arange(n, dtype=int)
    anchors = np.linspace(0, n - 1, num=max_anchors, dtype=int)
    anchors[0] = 0
    anchors[-1] = n - 1
    return np.unique(anchors)


def _quartile_means(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    splits = np.array_split(arr, 4)
    return [float(np.mean(chunk)) if chunk.size else 0.0 for chunk in splits]


def _evaluate_proxy_model(n: int, seed: int, cue_fraction: float, use_text: bool,
                          dg_output_dim: int, dg_target_sparsity: float,
                          bucket_bits: int, ca3_size: int, assembly_k: int,
                          max_anchors: int = 512) -> Dict[str, Any]:
    factory = SyntheticCorpusFactory(seed=seed, use_text=use_text)
    proxy = FastCapacityProxy(
        seed=seed,
        use_text=use_text,
        dg_output_dim=dg_output_dim,
        dg_target_sparsity=dg_target_sparsity,
        bucket_bits=bucket_bits,
        ca3_size=ca3_size,
        assembly_k=assembly_k,
    )

    for idx in range(n):
        proxy.store(idx, factory.make(idx), cue_fraction=cue_fraction)

    anchors = _select_anchors(n, max_anchors=max_anchors)
    target_overlaps: List[float] = []
    best_impostor_overlaps: List[float] = []
    margins: List[float] = []
    hits = 0

    for idx in anchors:
        episode = factory.make(int(idx))
        retrieved = proxy.retrieve(episode, cue_fraction=cue_fraction)
        if retrieved["retrieved_id"] is None:
            target_overlaps.append(0.0)
            best_impostor_overlaps.append(0.0)
            margins.append(0.0)
            continue

        target_assembly = proxy.records[int(idx)]["assembly"]
        retrieved_assembly = retrieved["retrieved_assembly"]
        target_overlap = _jaccard(retrieved_assembly, target_assembly)

        # Compute best impostor in the same candidate bucket.
        q_support, q_dg = proxy._encode_support(episode.sql_row, episode.graph_edge, episode.text, cue_fraction)
        bucket = proxy._bucket(q_dg)
        candidate_ids = proxy.buckets.get(bucket, [])
        q_support_bool = q_support.astype(bool)
        best_impostor = 0.0
        for ridx in candidate_ids:
            if ridx == int(idx):
                continue
            cand = proxy.records[ridx]["partial_support"].astype(bool)
            inter = int(np.logical_and(q_support_bool, cand).sum())
            union = int(np.logical_or(q_support_bool, cand).sum())
            score = inter / union if union > 0 else 0.0
            best_impostor = max(best_impostor, score)

        target_overlaps.append(target_overlap)
        best_impostor_overlaps.append(best_impostor)
        margins.append(target_overlap - best_impostor)
        hits += int(int(retrieved["retrieved_id"]) == int(idx))

    retention = np.asarray(target_overlaps, dtype=float)
    return {
        "mode": "proxy",
        "n": int(n),
        "seed": int(seed),
        "cue_fraction": float(cue_fraction),
        "mean_retention": float(np.mean(retention)) if retention.size else 0.0,
        "std_retention": float(np.std(retention, ddof=1)) if retention.size > 1 else 0.0,
        "oldest_retention": float(retention[0]) if retention.size else 0.0,
        "newest_retention": float(retention[-1]) if retention.size else 0.0,
        "mean_best_impostor_overlap": float(np.mean(best_impostor_overlaps)) if best_impostor_overlaps else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "false_retrieval_rate": float(1.0 - (hits / max(1, len(anchors)))),
        "anchor_count": int(len(anchors)),
        "memory_footprint_mb": _memory_footprint_mb(n, ca3_size=ca3_size),
        "retention_by_quartile": _quartile_means(retention),
    }


def _run_one(n: int, seed: int, cue_fraction: float, use_text: bool,
             exact_max_n: int, ca3_exc: int, ca3_inh: int, ca1_n: int,
             ca1_train_epochs: int, ca1_train_lr: float, ca1_relation_lr: float,
             dg_output_dim: int, dg_target_sparsity: float,
             dg_bridge_fanout: int, dg_bridge_lr: float,
             bucket_bits: int, ca3_size: int, assembly_k: int) -> Dict[str, Any]:
    if n <= exact_max_n:
        return _evaluate_exact_model(
            n=n,
            seed=seed,
            cue_fraction=cue_fraction,
            use_text=use_text,
            ca3_exc=ca3_exc,
            ca3_inh=ca3_inh,
            ca1_n=ca1_n,
            ca1_train_epochs=ca1_train_epochs,
            ca1_train_lr=ca1_train_lr,
            ca1_relation_lr=ca1_relation_lr,
            dg_output_dim=dg_output_dim,
            dg_target_sparsity=dg_target_sparsity,
            dg_bridge_fanout=dg_bridge_fanout,
            dg_bridge_lr=dg_bridge_lr,
        )

    return _evaluate_proxy_model(
        n=n,
        seed=seed,
        cue_fraction=cue_fraction,
        use_text=use_text,
        dg_output_dim=dg_output_dim,
        dg_target_sparsity=dg_target_sparsity,
        bucket_bits=bucket_bits,
        ca3_size=ca3_size,
        assembly_k=assembly_k,
    )


def _summarize_size(seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "mean_retention",
        "false_retrieval_rate",
        "mean_best_impostor_overlap",
        "mean_margin",
        "memory_footprint_mb",
    ]
    summary: Dict[str, Any] = {}
    for key in keys:
        vals = [float(item[key]) for item in seed_metrics]
        mean, std = _mean_std(vals)
        summary[key] = {
            "per_seed": vals,
            "mean": mean,
            "std": std,
        }
    summary["oldest_retention"] = {
        "per_seed": [float(item["oldest_retention"]) for item in seed_metrics],
        "mean": float(np.mean([float(item["oldest_retention"]) for item in seed_metrics])) if seed_metrics else 0.0,
        "std": float(np.std([float(item["oldest_retention"]) for item in seed_metrics], ddof=1)) if len(seed_metrics) > 1 else 0.0,
    }
    summary["newest_retention"] = {
        "per_seed": [float(item["newest_retention"]) for item in seed_metrics],
        "mean": float(np.mean([float(item["newest_retention"]) for item in seed_metrics])) if seed_metrics else 0.0,
        "std": float(np.std([float(item["newest_retention"]) for item in seed_metrics], ddof=1)) if len(seed_metrics) > 1 else 0.0,
    }
    summary["retention_by_quartile"] = {
        "mean": np.mean([item["retention_by_quartile"] for item in seed_metrics], axis=0).tolist() if seed_metrics else [0.0, 0.0, 0.0, 0.0],
        "std": np.std([item["retention_by_quartile"] for item in seed_metrics], axis=0, ddof=1).tolist() if len(seed_metrics) > 1 else [0.0, 0.0, 0.0, 0.0],
    }
    return summary


def _plot_results(summary: Dict[str, Any], out_dir: Path) -> None:
    sizes = [int(k) for k in summary["sizes"]]
    sizes_sorted = sorted(sizes)
    results = summary["results"]

    means_ret = [results[str(n)]["mean_retention"]["mean"] for n in sizes_sorted]
    std_ret = [results[str(n)]["mean_retention"]["std"] for n in sizes_sorted]
    means_false = [results[str(n)]["false_retrieval_rate"]["mean"] for n in sizes_sorted]
    std_false = [results[str(n)]["false_retrieval_rate"]["std"] for n in sizes_sorted]
    means_margin = [results[str(n)]["mean_margin"]["mean"] for n in sizes_sorted]
    std_margin = [results[str(n)]["mean_margin"]["std"] for n in sizes_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].errorbar(sizes_sorted, means_ret, yerr=std_ret, marker="o", capsize=4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Corpus size N")
    axes[0].set_ylabel("Mean retention")
    axes[0].set_title("Retention vs N")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(sizes_sorted, means_false, yerr=std_false, marker="o", capsize=4, color="#d62728")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Corpus size N")
    axes[1].set_ylabel("False retrieval rate")
    axes[1].set_title("False retrieval vs N")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(sizes_sorted, means_margin, yerr=std_margin, marker="o", capsize=4, color="#2ca02c")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Corpus size N")
    axes[2].set_ylabel("Mean margin")
    axes[2].set_title("Interference margin vs N")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Capacity scaling sweep")
    fig.savefig(out_dir / "capacity_scaling_summary.png", dpi=160)
    plt.close(fig)

    # Quartile retention traces
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    for n in sizes_sorted:
        q = results[str(n)]["retention_by_quartile"]["mean"]
        ax.plot(quartiles, q, marker="o", label=f"N={n}")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean retention")
    ax.set_title("Retention by corpus quartile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "capacity_quartiles.png", dpi=160)
    plt.close(fig)


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Capacity Scaling",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Timestamp UTC: `{summary['timestamp_utc']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Cue fraction: `{summary['cue_fraction']}`",
        f"- Exact cutoff: `{summary['exact_max_n']}`",
        "",
        "## Summary",
        "",
        "| N | Mode | Mean retention | False retrieval | Mean margin | Memory footprint MB |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for n in summary["sizes"]:
        res = summary["results"][str(n)]
        lines.append(
            f"| `{n}` | `{res['mode']}` | "
            f"{res['mean_retention']['mean']:.3f} ± {res['mean_retention']['std']:.3f} | "
            f"{res['false_retrieval_rate']['mean']:.3f} ± {res['false_retrieval_rate']['std']:.3f} | "
            f"{res['mean_margin']['mean']:.3f} ± {res['mean_margin']['std']:.3f} | "
            f"{res['memory_footprint_mb']['mean']:.1f} ± {res['memory_footprint_mb']['std']:.1f} |"
        )

    lines.extend([
        "",
        "## Paired Tests vs N=100",
        "",
        "| N | Metric | Mean delta | Paired t p | Bootstrap 95% CI |",
        "| --- | --- | --- | --- | --- |",
    ])
    base = summary["results"]["100"]
    for n in summary["sizes"]:
        if n == 100:
            continue
        cur = summary["results"][str(n)]
        for metric in ["mean_retention", "false_retrieval_rate", "mean_margin"]:
            stats_entry = cur["vs_100"][metric]
            lines.append(
                f"| `{n}` | `{metric}` | {stats_entry['mean_delta']:.3f} | {stats_entry['paired_t_p']:.4g} | "
                f"[{stats_entry['bootstrap_ci95'][0]:.3f}, {stats_entry['bootstrap_ci95'][1]:.3f}] |"
            )
    return "\n".join(lines) + "\n"


def run_capacity_suite(
    seeds: Sequence[int],
    sizes: Sequence[int],
    *,
    cue_fraction: float = 0.4,
    use_text: bool = True,
    exact_max_n: int = 100,
    ca3_exc: int = 240,
    ca3_inh: int = 60,
    ca1_n: int = 320,
    ca1_train_epochs: int = 12,
    ca1_train_lr: float = 0.03,
    ca1_relation_lr: float = 0.05,
    dg_output_dim: int = 1200,
    dg_target_sparsity: float = 0.02,
    dg_bridge_fanout: int = 12,
    dg_bridge_lr: float = 0.02,
    bucket_bits: int = 11,
    ca3_size: int = 240,
    assembly_k: int = 24,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "run_id": _utc_run_id(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(int(s) for s in seeds),
        "sizes": [int(s) for s in sizes],
        "cue_fraction": float(cue_fraction),
        "exact_max_n": int(exact_max_n),
        "config": {
            "use_text": bool(use_text),
            "ca3_exc": int(ca3_exc),
            "ca3_inh": int(ca3_inh),
            "ca1_n": int(ca1_n),
            "ca1_train_epochs": int(ca1_train_epochs),
            "ca1_train_lr": float(ca1_train_lr),
            "ca1_relation_lr": float(ca1_relation_lr),
            "dg_output_dim": int(dg_output_dim),
            "dg_target_sparsity": float(dg_target_sparsity),
            "dg_bridge_fanout": int(dg_bridge_fanout),
            "dg_bridge_lr": float(dg_bridge_lr),
            "bucket_bits": int(bucket_bits),
            "ca3_size": int(ca3_size),
            "assembly_k": int(assembly_k),
        },
        "results": {},
    }

    for n in sizes:
        print(f"[capacity] N={n}", flush=True)
        seed_metrics: List[Dict[str, Any]] = []
        mode = "exact" if n <= exact_max_n else "proxy"
        for seed in seeds:
            print(f"  [seed] {seed} ({mode})", flush=True)
            metric = _run_one(
                n=n,
                seed=seed,
                cue_fraction=cue_fraction,
                use_text=use_text,
                exact_max_n=exact_max_n,
                ca3_exc=ca3_exc,
                ca3_inh=ca3_inh,
                ca1_n=ca1_n,
                ca1_train_epochs=ca1_train_epochs,
                ca1_train_lr=ca1_train_lr,
                ca1_relation_lr=ca1_relation_lr,
                dg_output_dim=dg_output_dim,
                dg_target_sparsity=dg_target_sparsity,
                dg_bridge_fanout=dg_bridge_fanout,
                dg_bridge_lr=dg_bridge_lr,
                bucket_bits=bucket_bits,
                ca3_size=ca3_size,
                assembly_k=assembly_k,
            )
            seed_metrics.append(metric)

        payload = _summarize_size(seed_metrics)
        payload["mode"] = mode
        payload["seed_metrics"] = seed_metrics
        summary["results"][str(n)] = payload

    base_ret = summary["results"]["100"]["mean_retention"]["per_seed"]
    base_false = summary["results"]["100"]["false_retrieval_rate"]["per_seed"]
    base_margin = summary["results"]["100"]["mean_margin"]["per_seed"]
    for n in sizes:
        if n == 100:
            summary["results"][str(n)]["vs_100"] = {
                "mean_retention": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                },
                "false_retrieval_rate": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                },
                "mean_margin": {
                    "mean_delta": 0.0,
                    "paired_t_stat": 0.0,
                    "paired_t_p": 1.0,
                    "bootstrap_ci95": [0.0, 0.0],
                },
            }
            continue
        cur = summary["results"][str(n)]
        cur["vs_100"] = {
            "mean_retention": _paired_stats(cur["mean_retention"]["per_seed"], base_ret, seed=11),
            "false_retrieval_rate": _paired_stats(cur["false_retrieval_rate"]["per_seed"], base_false, seed=13),
            "mean_margin": _paired_stats(cur["mean_margin"]["per_seed"], base_margin, seed=17),
        }

    return summary


def parse_ints(raw: str) -> List[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the corpus scaling benchmark.")
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("0,1,2,3,4"), help="Comma-separated seeds.")
    parser.add_argument("--sizes", type=parse_ints, default=list(DEFAULT_SIZES), help="Comma-separated corpus sizes.")
    parser.add_argument("--cue-fraction", type=float, default=0.4, help="SQL cue fraction for partial retrieval.")
    parser.add_argument("--exact-max-n", type=int, default=100, help="Use the exact model up to this corpus size.")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT, help="Directory for output artifacts.")
    parser.add_argument("--no-text", action="store_true", help="Disable the text modality.")
    parser.add_argument("--ca3-exc", type=int, default=240)
    parser.add_argument("--ca3-inh", type=int, default=60)
    parser.add_argument("--ca1-n", type=int, default=320)
    parser.add_argument("--ca1-train-epochs", type=int, default=12)
    parser.add_argument("--ca1-train-lr", type=float, default=0.03)
    parser.add_argument("--ca1-relation-lr", type=float, default=0.05)
    parser.add_argument("--dg-output-dim", type=int, default=1200)
    parser.add_argument("--dg-target-sparsity", type=float, default=0.02)
    parser.add_argument("--dg-bridge-fanout", type=int, default=12)
    parser.add_argument("--dg-bridge-lr", type=float, default=0.02)
    parser.add_argument("--bucket-bits", type=int, default=11)
    parser.add_argument("--ca3-size", type=int, default=240)
    parser.add_argument("--assembly-k", type=int, default=24)
    args = parser.parse_args()

    run = run_capacity_suite(
        seeds=args.seeds,
        sizes=args.sizes,
        cue_fraction=args.cue_fraction,
        use_text=not args.no_text,
        exact_max_n=args.exact_max_n,
        ca3_exc=args.ca3_exc,
        ca3_inh=args.ca3_inh,
        ca1_n=args.ca1_n,
        ca1_train_epochs=args.ca1_train_epochs,
        ca1_train_lr=args.ca1_train_lr,
        ca1_relation_lr=args.ca1_relation_lr,
        dg_output_dim=args.dg_output_dim,
        dg_target_sparsity=args.dg_target_sparsity,
        dg_bridge_fanout=args.dg_bridge_fanout,
        dg_bridge_lr=args.dg_bridge_lr,
        bucket_bits=args.bucket_bits,
        ca3_size=args.ca3_size,
        assembly_k=args.assembly_k,
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
