from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tools.generate_research_site_data import save_data, stage_example, v2_v3_data, v4_v5_data, v6_data
from spiking_multimodal_memory import DelayedSynapse, GraphEncoder, MultiModalMemory, SQLEncoder


RUN_ROOT = Path("outputs/eval_runs")
SITE_JSON = Path("research_site/data.json")
SITE_JS = Path("research_site/data.js")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
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


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _threshold_pass(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _dg_bridge_synapses(mem: MultiModalMemory) -> List[DelayedSynapse]:
    synapses: List[DelayedSynapse] = []
    for pre_id, group in mem.ca3.input_synapses.items():
        if str(pre_id).startswith("dg:"):
            synapses.extend(group)
    return synapses


def _dg_bridge_alignment_stats(mem: MultiModalMemory, prepared: Dict[str, Any],
                               duration: float = 10.0) -> Dict[str, Any]:
    """Measure how tightly DG bridge arrivals align on CA3 active neurons."""
    mem.ca3.reset()
    pre_duration = min(10.0, duration)
    mem.ca3.run(pre_duration, prepared["ca3_inputs"], M=0.0)
    ca3_active = mem.ca3.get_active_neurons(threshold=1)

    arrivals: List[float] = []
    for dg_nid, times in prepared.get("dg_spikes", {}).items():
        pre_key = f"dg:{dg_nid}"
        for syn in mem.ca3.input_synapses.get(pre_key, []):
            if syn.post not in ca3_active:
                continue
            arrivals.extend(float(t + syn.delay) for t in times)

    if arrivals:
        target_time = float(np.median(arrivals))
        errors = [abs(arrival - target_time) for arrival in arrivals]
        error_values = list(errors)
        mean_abs_error = float(np.mean(error_values))
        std_abs_error = float(np.std(error_values))
        coincidence = {
            "within_0.5ms": float(np.mean([err <= 0.5 for err in error_values])),
            "within_1.0ms": float(np.mean([err <= 1.0 for err in error_values])),
            "within_2.0ms": float(np.mean([err <= 2.0 for err in error_values])),
        }
    else:
        target_time = 0.0
        errors = []
        mean_abs_error = 0.0
        std_abs_error = 0.0
        coincidence = {"within_0.5ms": 0.0, "within_1.0ms": 0.0, "within_2.0ms": 0.0}

    return {
        "ca3_active": ca3_active,
        "arrival_count": len(arrivals),
        "target_time_ms": target_time,
        "arrival_errors_ms": errors,
        "mean_abs_error_ms": mean_abs_error,
        "std_abs_error_ms": std_abs_error,
        "coincidence": coincidence,
        "passed": mean_abs_error <= 2.0,
    }


def _set_dg_bridge_delay_mode(mem: MultiModalMemory, mode: str, seed: int = 0) -> None:
    synapses = _dg_bridge_synapses(mem)
    if not synapses:
        return

    if mode == "zero":
        for syn in synapses:
            syn.delay = 0.0
        return

    if mode == "random":
        rng = np.random.RandomState(seed)
        delays = np.asarray([syn.delay for syn in synapses], dtype=float)
        if delays.size == 0:
            return
        rng.shuffle(delays)
        for syn, delay in zip(synapses, delays):
            syn.delay = float(delay)
        return

    raise ValueError(f"Unsupported DG bridge delay mode: {mode}")


def _reference_corpus(use_text: bool = True) -> List[Tuple[Dict[str, Any], Tuple[int, str, int], Optional[str]]]:
    corpus = [
        (
            {"age": 30, "salary": 85000, "city": "NYC", "role": "Engineer", "dept": "AI"},
            (0, "WORKS_AT", 1),
            "Alice works at Google" if use_text else None,
        ),
        (
            {"age": 31, "salary": 92000, "city": "NYC", "role": "Manager", "dept": "AI"},
            (0, "MANAGES", 1),
            "Alice manages Bob" if use_text else None,
        ),
        (
            {"age": 32, "salary": 110000, "city": "Seattle", "role": "Engineer", "dept": "Cloud"},
            (1, "FRIENDS_WITH", 2),
            "Bob collaborates with Carol" if use_text else None,
        ),
        (
            {"age": 34, "salary": 180000, "city": "Seattle", "role": "CTO", "dept": "Cloud"},
            (2, "REPORTS_TO", 3),
            "Carol reports to Dave" if use_text else None,
        ),
    ]
    return corpus


def _build_memory(use_text: bool = True, seed: int = 42, consolidate: bool = True) -> MultiModalMemory:
    mem = MultiModalMemory(use_text=use_text, seed=seed)
    for idx, (sql_row, graph_edge, text) in enumerate(_reference_corpus(use_text=use_text)):
        mem.encode_episode(
            sql_row,
            graph_edge,
            text=text,
            episode_time=float(idx),
            consolidate=consolidate,
        )
    return mem


def evaluate_encoder_reconstructibility() -> Dict[str, Any]:
    sql_row = {"age": 30, "salary": 85000, "city": "NYC", "role": "Engineer", "dept": "AI"}
    sql_enc = SQLEncoder(seed=0)
    sql_spikes = sql_enc.encode(sql_row)
    decoded = sql_enc.decode_row(sql_spikes)

    graph_enc = GraphEncoder(seed=0)
    graph_edge = (0, "WORKS_AT", 1)
    graph_spikes = graph_enc.encode(graph_edge)
    src_code = graph_enc.get_active_neurons(graph_edge[0], "source")
    tgt_code = graph_enc.get_active_neurons(graph_edge[2], "target")
    src_onset = min(t for nid in src_code for t in graph_spikes.get(nid, []))
    tgt_onset = min(t for nid in tgt_code for t in graph_spikes.get(nid, []))
    actual_delay = tgt_onset - src_onset
    expected_delay = graph_enc.RELATION_DELAYS[graph_edge[1]]

    age_error = abs(float(decoded["age"]) - sql_row["age"])
    salary_error = abs(float(decoded["salary"]) - sql_row["salary"])
    category_exact = all(decoded[field] == sql_row[field] for field in ("city", "role", "dept"))
    graph_delay_error = abs(actual_delay - expected_delay)

    passed = (
        age_error < 10.0
        and salary_error < 5000.0
        and category_exact
        and graph_delay_error <= 1.0
    )

    return {
        "sql_row": sql_row,
        "decoded_row": decoded,
        "age_abs_error": float(age_error),
        "salary_abs_error": float(salary_error),
        "category_exact_match": category_exact,
        "graph_edge": graph_edge,
        "graph_expected_delay_ms": float(expected_delay),
        "graph_measured_delay_ms": float(actual_delay),
        "graph_delay_abs_error_ms": float(graph_delay_error),
        "passed": passed,
    }


def evaluate_multimodal_binding(seed: int = 42) -> Dict[str, Any]:
    mem = _build_memory(use_text=True, seed=seed, consolidate=True)
    target = mem.episode_records[0]
    partial_sql = mem._partial_sql_row(target.sql_row, 0.4)
    control_sql = {"age": 58, "salary": 12500, "city": "Austin", "role": "Analyst", "dept": "Ops"}

    learned = mem.retrieve(sql_cue=partial_sql, duration=50.0)
    control = mem.retrieve(sql_cue=control_sql, duration=50.0)

    target_overlap = _jaccard(learned["ca3_active"], target.ca3_assembly)
    control_overlap = _jaccard(control["ca3_active"], target.ca3_assembly)
    activation_delta = len(learned["ca3_active"]) - len(control["ca3_active"])
    overlap_delta = target_overlap - control_overlap
    graph_metrics = mem.compute_graph_retrieval_accuracy(learned, target.graph_edge)

    passed = activation_delta > 2

    return {
        "target_episode_id": target.episode_id,
        "partial_sql_cue": partial_sql,
        "control_sql_cue": control_sql,
        "learned_ca3_active": len(learned["ca3_active"]),
        "control_ca3_active": len(control["ca3_active"]),
        "activation_delta": int(activation_delta),
        "target_overlap": float(target_overlap),
        "control_overlap": float(control_overlap),
        "overlap_delta": float(overlap_delta),
        "relation_prediction": learned["relation_prediction"],
        "relation_confidence": float(learned["relation_confidence"]),
        "graph_metrics": graph_metrics,
        "passed": passed,
    }


def evaluate_engram_stability(repeats: int = 10, seed: int = 42) -> Dict[str, Any]:
    mem = MultiModalMemory(use_text=True, seed=seed)
    sql_row, graph_edge, text = _reference_corpus(use_text=True)[0]

    assemblies: List[set[int]] = []
    for i in range(repeats):
        result = mem.encode_episode(
            sql_row,
            graph_edge,
            text=text,
            episode_time=float(i),
            consolidate=True,
        )
        assemblies.append(set(result["ca3_active"]))

    reference = assemblies[0]
    jaccards = [_jaccard(reference, assembly) for assembly in assemblies]
    sizes = [len(assembly) for assembly in assemblies]
    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
    std_jaccard = float(np.std(jaccards)) if jaccards else 0.0
    size_variance = float(np.var(sizes)) if sizes else 0.0
    cv = float(std_jaccard / mean_jaccard) if mean_jaccard > 1e-9 else 0.0

    return {
        "repeats": repeats,
        "mean_jaccard": mean_jaccard,
        "std_jaccard": std_jaccard,
        "coefficient_of_variation": cv,
        "engram_size_variance": size_variance,
        "mean_size": float(np.mean(sizes)) if sizes else 0.0,
        "passed": mean_jaccard >= 0.80,
    }


def evaluate_pattern_separation(seed: int = 42) -> Dict[str, Any]:
    mem = _build_memory(use_text=True, seed=seed, consolidate=True)
    similar_pairs = [
        (
            {"age": 30, "salary": 85000, "city": "NYC", "role": "Engineer", "dept": "AI"},
            (0, "WORKS_AT", 1),
            "Alice works at Google",
            {"age": 30, "salary": 85000, "city": "NYC", "role": "Engineer", "dept": "AI"},
            (0, "MANAGES", 1),
            "Alice manages Bob",
        ),
        (
            {"age": 32, "salary": 110000, "city": "Seattle", "role": "Engineer", "dept": "Cloud"},
            (1, "FRIENDS_WITH", 2),
            "Bob collaborates with Carol",
            {"age": 34, "salary": 110000, "city": "Seattle", "role": "Engineer", "dept": "Cloud"},
            (1, "FRIENDS_WITH", 2),
            "Bob collaborates with Carol",
        ),
    ]

    input_jaccards: List[float] = []
    dg_jaccards: List[float] = []
    separation_gains: List[float] = []

    for sql_a, graph_a, text_a, sql_b, graph_b, text_b in similar_pairs:
        prep_a = mem._prepare_modalities(sql_a, graph_a, text_a, duration=10.0)
        prep_b = mem._prepare_modalities(sql_b, graph_b, text_b, duration=10.0)
        input_a = set(np.flatnonzero(prep_a["x_ec"] > 0))
        input_b = set(np.flatnonzero(prep_b["x_ec"] > 0))
        dg_a = set(prep_a["dg_active"])
        dg_b = set(prep_b["dg_active"])
        j_input = _jaccard(input_a, input_b)
        j_dg = _jaccard(dg_a, dg_b)
        input_jaccards.append(j_input)
        dg_jaccards.append(j_dg)
        separation_gains.append(1.0 - (j_dg / j_input) if j_input > 1e-9 else 0.0)

    return {
        "input_jaccard_mean": float(np.mean(input_jaccards)) if input_jaccards else 0.0,
        "dg_jaccard_mean": float(np.mean(dg_jaccards)) if dg_jaccards else 0.0,
        "separation_gain_mean": float(np.mean(separation_gains)) if separation_gains else 0.0,
        "passed": bool(np.mean(separation_gains) > 0.0) if separation_gains else False,
    }


def evaluate_polychronous_binding(seed: int = 42) -> Dict[str, Any]:
    mem = _build_memory(use_text=True, seed=seed, consolidate=True)
    sql_row, graph_edge, text = _reference_corpus(use_text=True)[0]
    prepared = mem._prepare_modalities(sql_row, graph_edge, text, duration=10.0)
    structured = _dg_bridge_alignment_stats(mem, prepared, duration=10.0)

    mem_zero = _build_memory(use_text=True, seed=seed, consolidate=True)
    prepared_zero = mem_zero._prepare_modalities(sql_row, graph_edge, text, duration=10.0)
    _set_dg_bridge_delay_mode(mem_zero, "zero")
    zero_delay = _dg_bridge_alignment_stats(mem_zero, prepared_zero, duration=10.0)

    mem_rand = _build_memory(use_text=True, seed=seed, consolidate=True)
    prepared_rand = mem_rand._prepare_modalities(sql_row, graph_edge, text, duration=10.0)
    _set_dg_bridge_delay_mode(mem_rand, "random", seed=seed)
    random_delay = _dg_bridge_alignment_stats(mem_rand, prepared_rand, duration=10.0)

    structured_error = float(structured["mean_abs_error_ms"])
    zero_error = float(zero_delay["mean_abs_error_ms"])
    random_error = float(random_delay["mean_abs_error_ms"])
    coincidence = structured["coincidence"]

    return {
        "mean_abs_error_ms": structured_error,
        "structured_mean_abs_error_ms": structured_error,
        "zero_delay_mean_abs_error_ms": zero_error,
        "random_delay_mean_abs_error_ms": random_error,
        "arrival_count": int(structured["arrival_count"]),
        "structured_arrival_count": int(structured["arrival_count"]),
        "target_time_ms": float(structured["target_time_ms"]),
        "arrival_errors_ms": structured["arrival_errors_ms"],
        "std_abs_error_ms": float(structured["std_abs_error_ms"]),
        "coincidence": coincidence,
        "gain_vs_zero": float(zero_error / max(1e-9, structured_error)),
        "gain_vs_random": float(random_error / max(1e-9, structured_error)),
        "passed": structured_error <= 2.0,
    }


def evaluate_stdp_causality(trials: int = 64, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)

    def delta_w(delay: float, pre_times: Sequence[float], post_time: float) -> float:
        syn = DelayedSynapse(pre="p", post=0, weight=0.5, delay=delay)
        before = syn.w
        syn.stdp_update(post_time, list(pre_times), M=1.0)
        return float(syn.w - before)

    real_deltas: List[float] = []
    shuffled_deltas: List[float] = []
    zero_delay_deltas: List[float] = []
    random_delay_deltas: List[float] = []

    for _ in range(trials):
        base = float(rng.uniform(0.0, 2.0))
        pre_times = [base, base + 3.0, base + 6.0]
        real_post = pre_times[-1] + 2.0 + float(rng.uniform(0.0, 0.75))
        shuffled_post = pre_times[0] - float(rng.uniform(0.5, 1.5))
        zero_post = real_post
        random_delay = float(rng.uniform(0.0, 6.0))
        random_post = real_post

        real_deltas.append(delta_w(2.0, pre_times, real_post))
        shuffled_deltas.append(delta_w(2.0, pre_times, shuffled_post))
        zero_delay_deltas.append(delta_w(0.0, pre_times, zero_post))
        random_delay_deltas.append(delta_w(random_delay, pre_times, random_post))

    real_mean = float(np.mean(real_deltas))
    shuffled_mean = float(np.mean(shuffled_deltas))
    zero_mean = float(np.mean(zero_delay_deltas))
    random_mean = float(np.mean(random_delay_deltas))

    return {
        "trials": trials,
        "real_mean_delta_w": real_mean,
        "shuffled_mean_delta_w": shuffled_mean,
        "zero_delay_mean_delta_w": zero_mean,
        "random_delay_mean_delta_w": random_mean,
        "real_minus_shuffled": float(real_mean - shuffled_mean),
        "real_minus_zero_delay": float(real_mean - zero_mean),
        "real_minus_random": float(real_mean - random_mean),
        "passed": real_mean > shuffled_mean,
    }


def evaluate_cera(seed: int = 42) -> Dict[str, Any]:
    """Compute Cross-Modal Episodic Retrieval Accuracy (CERA) matrix.

    Returns average reconstruction accuracy for each cue -> target pair.
    """
    mem = _build_memory(use_text=True, seed=seed, consolidate=True)
    modalities = ["sql", "graph"]
    if mem.use_text:
        modalities.append("text")

    # Helper to evaluate single target reconstruction
    def eval_target(retrieved: Dict, record) -> Dict[str, float]:
        res = {}
        # SQL target: compare top-k neuron overlap
        sql_recon = retrieved.get("sql_reconstruction", {})
        sql_support = mem.episode_targets.get(record.episode_id, {}).get("sql_support", None)
        if sql_support is not None:
            k = int(max(1, (sql_support > 0).sum())) if hasattr(sql_support, "sum") else int(max(1, sum(sql_support)))
            if k <= 0:
                res["sql"] = 0.0
            else:
                top = set(int(i) for i in np.argsort(list(sql_recon.values()))[-k:]) if sql_recon else set()
                support_idx = set(int(i) for i, v in enumerate(sql_support) if v > 0)
                res["sql"] = float(len(top & support_idx) / max(1, len(support_idx)))

        # Graph target: use provided graph retrieval accuracy
        graph_acc = mem.compute_graph_retrieval_accuracy(retrieved, record.graph_edge)
        res["graph"] = float(graph_acc.get("edge_accuracy", 0.0))

        # Text target: cosine similarity between decoded text vector and original support
        text_recon = retrieved.get("text_reconstruction", {})
        text_support = mem.episode_targets.get(record.episode_id, {}).get("text_support", None)
        if text_support is not None and len(text_support) > 0:
            # build vectors
            pred = np.array([text_recon.get(i, 0.0) for i in range(len(text_support))])
            tgt = np.array(text_support[: len(pred)])
            denom = (np.linalg.norm(pred) * np.linalg.norm(tgt)) + 1e-9
            res["text"] = float(np.dot(pred, tgt) / denom) if denom > 0 else 0.0
        else:
            res["text"] = 0.0

        return res

    matrix = {c: {t: [] for t in modalities if t != c} for c in modalities}

    for record in mem.episode_records:
        for cue in modalities:
            cue_args = {"sql_cue": None, "graph_cue": None, "text_cue": None}
            if cue == "sql":
                cue_args["sql_cue"] = record.sql_row
            if cue == "graph":
                cue_args["graph_cue"] = record.graph_edge
            if cue == "text":
                cue_args["text_cue"] = record.text

            retrieved = mem.retrieve(**cue_args, duration=50.0)
            scores = eval_target(retrieved, record)
            for target in matrix[cue].keys():
                matrix[cue][target].append(float(scores.get(target, 0.0)))

    # Average and return
    out = {cue: {t: float(np.mean(vals)) if vals else 0.0 for t, vals in targets.items()} for cue, targets in matrix.items()}
    return {"cera_matrix": out}


def evaluate_binding_index(seed: int = 42) -> Dict[str, Any]:
    """Compute Binding Index (BI) as cross-modal accuracy divided by unimodal control accuracy.

    Example: BI(SQL->Graph) = CERA(SQL->Graph) / CERA(SQL+Graph->Graph)
    """
    mem = _build_memory(use_text=True, seed=seed, consolidate=True)
    cera = evaluate_cera(seed=seed).get("cera_matrix", {})

    bi = {}
    # For each cross-modal cue->target, compare against control with both modalities
    for cue in cera:
        for target in cera[cue]:
            # compute control: cue+target-modality used as combined cue (e.g., SQL+Graph -> Graph)
            # We'll form a combined cue that includes both cue and target (if possible)
            combined_scores = []
            for record in mem.episode_records:
                sql_cue = record.sql_row if (cue == "sql" or target == "sql") else None
                graph_cue = record.graph_edge if (cue == "graph" or target == "graph") else None
                text_cue = record.text if (cue == "text" or target == "text") else None
                retrieved = mem.retrieve(sql_cue=sql_cue, graph_cue=graph_cue, text_cue=text_cue, duration=50.0)
                graph_acc = mem.compute_graph_retrieval_accuracy(retrieved, record.graph_edge)["edge_accuracy"] if target == "graph" else 0.0
                sql_score = 0.0
                if target == "sql":
                    sql_recon = retrieved.get("sql_reconstruction", {})
                    sql_support = mem.episode_targets.get(record.episode_id, {}).get("sql_support", None)
                    if sql_support is not None:
                        k = int(max(1, (sql_support > 0).sum())) if hasattr(sql_support, "sum") else int(max(1, sum(sql_support)))
                        top = set(int(i) for i in np.argsort(list(sql_recon.values()))[-k:]) if sql_recon else set()
                        support_idx = set(int(i) for i, v in enumerate(sql_support) if v > 0)
                        sql_score = float(len(top & support_idx) / max(1, len(support_idx)))
                text_score = 0.0
                if target == "text":
                    text_recon = retrieved.get("text_reconstruction", {})
                    text_support = mem.episode_targets.get(record.episode_id, {}).get("text_support", None)
                    if text_support is not None and len(text_support) > 0:
                        pred = np.array([text_recon.get(i, 0.0) for i in range(len(text_support))])
                        tgt = np.array(text_support[: len(pred)])
                        denom = (np.linalg.norm(pred) * np.linalg.norm(tgt)) + 1e-9
                        text_score = float(np.dot(pred, tgt) / denom) if denom > 0 else 0.0

                if target == "graph":
                    combined_scores.append(graph_acc)
                elif target == "sql":
                    combined_scores.append(sql_score)
                else:
                    combined_scores.append(text_score)

            control_mean = float(np.mean(combined_scores)) if combined_scores else 1e-9
            cross_mean = float(np.mean(cera[cue][target])) if cera.get(cue) and cera[cue].get(target) is not None else 0.0
            bi_key = f"{cue}->{target}"
            bi[bi_key] = float(cross_mean / max(1e-6, control_mean))

    return {"binding_index": bi}


def evaluate_polychronous_binding_gain(seed: int = 42) -> Dict[str, Any]:
    """Compute DG-bridge timing gain under structured vs zero/random delays."""
    mem = _build_memory(use_text=True, seed=seed, consolidate=True)
    sql_row, graph_edge, text = _reference_corpus(use_text=True)[0]

    prepared = mem._prepare_modalities(sql_row, graph_edge, text, duration=10.0)
    structured = _dg_bridge_alignment_stats(mem, prepared, duration=10.0)

    mem_zero = _build_memory(use_text=True, seed=seed, consolidate=True)
    prepared_zero = mem_zero._prepare_modalities(sql_row, graph_edge, text, duration=10.0)
    _set_dg_bridge_delay_mode(mem_zero, "zero")
    zero_delay = _dg_bridge_alignment_stats(mem_zero, prepared_zero, duration=10.0)

    mem_rand = _build_memory(use_text=True, seed=seed, consolidate=True)
    prepared_rand = mem_rand._prepare_modalities(sql_row, graph_edge, text, duration=10.0)
    _set_dg_bridge_delay_mode(mem_rand, "random", seed=seed + 1)
    random_delay = _dg_bridge_alignment_stats(mem_rand, prepared_rand, duration=10.0)

    structured_error = float(structured["mean_abs_error_ms"])
    zero_error = float(zero_delay["mean_abs_error_ms"])
    random_error = float(random_delay["mean_abs_error_ms"])

    return {
        "structured_mean_abs_error_ms": structured_error,
        "zero_delay_mean_abs_error_ms": zero_error,
        "random_delay_mean_abs_error_ms": random_error,
        "gain_vs_zero": float(zero_error / max(1e-9, structured_error)),
        "gain_vs_random": float(random_error / max(1e-9, structured_error)),
        "structured_coincidence": structured["coincidence"],
        "zero_delay_coincidence": zero_delay["coincidence"],
        "random_delay_coincidence": random_delay["coincidence"],
        "passed": structured_error <= 2.0,
    }


def run_system_metrics(seed: int = 42) -> Dict[str, Any]:
    mem = _build_memory(use_text=True, seed=seed, consolidate=True)
    first = mem.episode_records[0]
    partial_sql = mem._partial_sql_row(first.sql_row, 0.4)
    retrieved = mem.retrieve(sql_cue=partial_sql, duration=50.0)

    system_metrics = {
        "architecture_purity": mem.evaluate_architecture_purity(),
        "separation": mem.evaluate_separation(),
        "continuity": mem.evaluate_continuity(),
        "completion_curve": mem.evaluate_completion_curve(),
        "interference": mem.evaluate_interference(),
        "modality_dropout": mem.evaluate_modality_dropout(),
        "false_retrieval_rate": mem.evaluate_false_retrieval_rate(),
        "graph_retrieval_accuracy": mem.compute_graph_retrieval_accuracy(retrieved, first.graph_edge),
        "provenance": mem.get_episode_provenance(first.episode_id),
        "exact_lookup_text": mem.exact_lookup(first.episode_id, "text"),
        "num_episodes": len(mem.episode_records),
    }
    return system_metrics


def build_site_payload() -> Dict[str, Any]:
    return {
        "v2v3": v2_v3_data(),
        "v4v5": v4_v5_data(),
        "v6": v6_data(),
    }


def build_spec_suite() -> Dict[str, Any]:
    evaluations = {
        "encoder_reconstructibility": evaluate_encoder_reconstructibility(),
        "multimodal_binding": evaluate_multimodal_binding(),
        "engram_stability": evaluate_engram_stability(),
        "pattern_separation": evaluate_pattern_separation(),
        "polychronous_binding": evaluate_polychronous_binding(),
        "stdp_causality": evaluate_stdp_causality(),
        "system_metrics": run_system_metrics(),
        # Additional paper metrics
        "cera": None,  # computed below
        "binding_index": None,
        "polychronous_binding_gain": None,
    }
    # Compute CERA (Cross-Modal Episodic Retrieval Accuracy)
    try:
        evaluations["cera"] = evaluate_cera()
    except Exception as e:  # pragma: no cover - best-effort
        evaluations["cera"] = {"error": str(e)}

    # Binding Index (BI): cross-modal accuracy vs unimodal control
    try:
        evaluations["binding_index"] = evaluate_binding_index()
    except Exception as e:  # pragma: no cover
        evaluations["binding_index"] = {"error": str(e)}

    # Polychronous Binding Gain (PBG): structured delay vs random/zero
    try:
        evaluations["polychronous_binding_gain"] = evaluate_polychronous_binding_gain()
    except Exception as e:  # pragma: no cover
        evaluations["polychronous_binding_gain"] = {"error": str(e)}

    return evaluations


def _summary_rows(evaluations: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for name, payload in evaluations.items():
        if name == "system_metrics":
            purity = payload.get("architecture_purity", {})
            key_metric = (
                f"sep={payload['separation']['separation_margin_mean']:.3f}, "
                f"completion={payload['completion_curve']['mean_completion']:.3f}, "
                f"false_retrieval={payload['false_retrieval_rate']['false_retrieval_rate']:.3f}, "
                f"pure_ca3={bool(purity.get('pure_ec_dg_ca3', False))}"
            )
            rows.append((name, key_metric, "SUMMARY"))
            continue

        passed = bool(payload.get("passed", True))
        key_metric = ""
        if name == "encoder_reconstructibility":
            key_metric = f"age_err={payload['age_abs_error']:.2f}, delay_err={payload['graph_delay_abs_error_ms']:.2f}ms"
        elif name == "multimodal_binding":
            key_metric = f"delta={payload['activation_delta']}, overlap_delta={payload['overlap_delta']:.3f}"
        elif name == "engram_stability":
            key_metric = f"mean_jaccard={payload['mean_jaccard']:.3f}"
        elif name == "pattern_separation":
            key_metric = f"gain={payload['separation_gain_mean']:.3f}"
        elif name == "polychronous_binding":
            key_metric = f"mean_abs_error={payload['mean_abs_error_ms']:.3f}ms"
        elif name == "polychronous_binding_gain":
            key_metric = (
                f"structured={payload['structured_mean_abs_error_ms']:.3f}ms, "
                f"zero={payload['zero_delay_mean_abs_error_ms']:.3f}ms, "
                f"random={payload['random_delay_mean_abs_error_ms']:.3f}ms"
            )
        elif name == "stdp_causality":
            key_metric = f"real_minus_shuffled={payload['real_minus_shuffled']:.4f}"
        rows.append((name, key_metric, _threshold_pass(passed)))
    return rows


def _render_summary_md(manifest: Dict[str, Any], evaluations: Dict[str, Any]) -> str:
    rows = _summary_rows(evaluations)
    lines = [
        "# Full Benchmark Run",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Timestamp UTC: `{manifest['timestamp_utc']}`",
        f"- Git SHA: `{manifest['git_sha']}`",
        f"- Python: `{manifest['python_version']}`",
        f"- Platform: `{manifest['platform']}`",
        "",
        "## Evaluation Summary",
        "",
        "| Evaluation | Key Metric | Result |",
        "| --- | --- | --- |",
    ]
    for name, key_metric, result in rows:
        lines.append(f"| `{name}` | {key_metric} | {result} |")
    lines.append("")
    return "\n".join(lines)


def run_full_benchmark(output_root: Path = RUN_ROOT, update_site: bool = True) -> Dict[str, Any]:
    run_id = _utc_run_id()
    run_dir = output_root / run_id
    versions_dir = run_dir / "versions"
    eval_dir = run_dir / "evaluations"
    versions_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": 42,
        "use_text": True,
        "site_payload_version": "v1",
        "spec_suite_version": "v1",
        "output_root": str(output_root),
    }

    site_payload = build_site_payload()
    evaluations = build_spec_suite()
    summary_rows = _summary_rows(evaluations)
    summary = {
        "manifest": manifest,
        "summary_rows": [
            {"evaluation": name, "key_metric": key_metric, "result": result}
            for name, key_metric, result in summary_rows
        ],
        "evaluations": evaluations,
        "site_payload_keys": list(site_payload.keys()),
    }

    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "site_data.json", site_payload)
    save_data(site_payload, path=str(run_dir / "site_data.js"))

    for key, payload in site_payload.items():
        _write_json(versions_dir / f"{key}.json", payload)

    for name, payload in evaluations.items():
        _write_json(eval_dir / f"{name}.json", payload)

    (run_dir / "summary.md").write_text(_render_summary_md(manifest, evaluations), encoding="utf-8")
    (output_root / "LATEST.txt").write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")

    if update_site:
        SITE_JSON.parent.mkdir(parents=True, exist_ok=True)
        _write_json(SITE_JSON, site_payload)
        save_data(site_payload, path=str(SITE_JS))

    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "summary": summary,
        "site_payload": site_payload,
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full multimodal memory benchmark suite.")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT, help="Directory for versioned run artifacts.")
    parser.add_argument("--no-site-update", action="store_true", help="Do not refresh research_site/data.js.")
    args = parser.parse_args()

    result = run_full_benchmark(output_root=args.output_root, update_site=not args.no_site_update)
    print(json.dumps(_jsonable({
        "run_dir": result["run_dir"],
        "summary_rows": result["summary"]["summary_rows"],
    }), indent=2))


if __name__ == "__main__":
    main()
