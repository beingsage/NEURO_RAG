import json
from pprint import pprint
from typing import Any, Dict, List, Tuple

import numpy as np
from spiking_multimodal_memory import MultiModalMemory


def stage_example():
    mem = MultiModalMemory(use_text=True, seed=42)
    sql = {'age': 30, 'salary': 85000, 'city': 'NYC', 'role': 'Engineer', 'dept': 'AI'}
    graph = (0, 'WORKS_AT', 1)
    text = 'Alice works at Google'
    prepared = mem._prepare_modalities(sql, graph, text, duration=10.0)
    mem.dg.update(prepared['x_ec'], prepared['dg_active'])

    mem.ca3.reset()
    stage1_spikes = mem.ca3.run(10.0, prepared['ca3_inputs'], M=0.0)
    stage1_active = mem.ca3.get_active_neurons(threshold=1)
    familiar, match_idx = mem._compute_familiarity(stage1_active)
    M = 1.0 if familiar else 2.0

    stage2_spikes = mem.ca3.run(50.0, prepared['ca3_inputs'], M=M, apply_homeostasis=True)
    stage2_active = mem.ca3.get_active_neurons(threshold=1)
    mem.ca3.reinforce_assembly(stage2_active, prepared['ca3_inputs'], modulation=M)
    mem._reinforce_dg_bridge(prepared['dg_active'], stage2_active, modulation=M)
    replay_spikes = mem.ca3.run(5.0, prepared['ca3_inputs'], M=0.0, apply_homeostasis=False)
    stage2_replay_active = mem.ca3.get_active_neurons(threshold=1)

    return {
        'input': {'sql': sql, 'graph': graph, 'text': text},
        'energy': float(prepared['energy']),
        'M': float(M),
        'familiar': bool(familiar),
        'stage1_ca3_active': sorted(stage1_active),
        'stage2_ca3_active': sorted(stage2_active),
        'stage2_replay_active': sorted(stage2_replay_active),
    }


def v2_v3_data():
    mem = MultiModalMemory(use_text=True, seed=42)
    sql = {'age': 30, 'salary': 85000, 'city': 'NYC', 'role': 'Engineer', 'dept': 'AI'}
    graph = (0, 'WORKS_AT', 1)
    text = 'Alice works at Google'
    mem.encode_episode(sql, graph, text=text)

    partial_cue = {'age': 30, 'salary': 85000}
    retrieved = mem.retrieve(sql_cue=partial_cue, duration=50.0)
    text_retrieved = mem.retrieve(text_cue='Alice is employed by Google', duration=50.0)

    return {
        'episode': stage_example(),
        'retrieval_partial': {
            'ca3_active': sorted(retrieved['ca3_active']),
            'relation_prediction': retrieved['relation_prediction'],
            'relation_confidence': retrieved['relation_confidence'],
            'sql_reconstruction': {k: float(v) for k, v in retrieved['sql_reconstruction'].items()},
            'graph_reconstruction': {k: float(v) for k, v in retrieved['graph_reconstruction'].items()},
        },
        'retrieval_text': {
            'text': 'Alice is employed by Google',
            'relation_prediction': text_retrieved['relation_prediction'],
            'relation_confidence': text_retrieved['relation_confidence'],
            'sql_reconstruction': {k: float(v) for k, v in text_retrieved['sql_reconstruction'].items()},
        },
        'completion_curve': mem.evaluate_completion_curve(),
    }


def v4_v5_data():
    mem = MultiModalMemory(use_text=False, seed=7)
    sequence = [
        ({'age': 30, 'salary': 85000, 'city': 'NYC', 'role': 'Engineer', 'dept': 'Search'}, (0, 'WORKS_AT', 1), 'Alice joins Google'),
        ({'age': 31, 'salary': 92000, 'city': 'NYC', 'role': 'Manager', 'dept': 'Search'}, (0, 'MANAGES', 1), 'Alice becomes manager'),
        ({'age': 32, 'salary': 110000, 'city': 'Seattle', 'role': 'Engineer', 'dept': 'Cloud'}, (0, 'WORKS_AT', 2), 'Alice moves to Microsoft'),
        ({'age': 34, 'salary': 180000, 'city': 'Seattle', 'role': 'CTO', 'dept': 'Cloud'}, (0, 'MANAGES', 2), 'Alice becomes CTO'),
    ]
    for episode_time, (sql_row, edge, text) in enumerate(sequence):
        mem.encode_episode(sql_row, edge, text=text, episode_time=float(episode_time))

    episodes = [
        {
            'episode_id': record.episode_id,
            'text': record.text,
            'predecessor_episode_id': record.predecessor_episode_id,
            'successor_episode_id': record.successor_episode_id,
        }
        for record in mem.episode_records
    ]

    metrics = {
        'interference': mem.evaluate_interference(),
        'separation': mem.evaluate_separation(),
        'continuity': mem.evaluate_continuity(),
        'completion_curve': mem.evaluate_completion_curve(),
    }
    return {'episodes': episodes, 'metrics': metrics}


def v6_data():
    def micro_benchmark(use_text: bool, ca3_exc: int, consolidate: bool):
        model = MultiModalMemory(use_text=use_text, seed=11, ca3_exc=ca3_exc)
        corpus = [
            ({'age': 30, 'salary': 85000, 'city': 'NYC', 'role': 'Engineer', 'dept': 'AI'}, (0, 'WORKS_AT', 1), 'Alice works at Google' if use_text else None),
            ({'age': 33, 'salary': 98000, 'city': 'Boston', 'role': 'Manager', 'dept': 'Research'}, (1, 'FRIENDS_WITH', 2), 'Bob collaborates with Carol' if use_text else None),
            ({'age': 36, 'salary': 126000, 'city': 'Seattle', 'role': 'Lead', 'dept': 'Cloud'}, (2, 'MANAGES', 3), 'Carol manages Dave' if use_text else None),
        ]
        def bridge_mean_weight(memory):
            weights = [syn.w for syn in memory.ca3.synapses if isinstance(syn.pre, str) and syn.pre.startswith('dg:')]
            return float(np.mean(weights)) if weights else 0.0

        def homeostatic_offset_mean(memory):
            offsets = [neuron.homeostatic_offset for neuron in memory.ca3.neurons[:memory.ca3.n_e]]
            return float(np.mean(offsets)) if offsets else 0.0

        bridge_before = bridge_mean_weight(model)
        homeostasis_before = homeostatic_offset_mean(model)
        for sql_row, edge, text in corpus:
            model.encode_episode(sql_row, edge, text=text, consolidate=consolidate)

        completion = model.evaluate_completion_curve()
        interference = model.evaluate_interference()
        separation = model.evaluate_separation()
        continuity = model.evaluate_continuity()
        bridge_after = bridge_mean_weight(model)
        homeostasis_after = homeostatic_offset_mean(model)

        return {
            'use_text': use_text,
            'ca3_exc': ca3_exc,
            'consolidate': consolidate,
            'mean_completion': float(completion['mean_completion']),
            'mean_retention': float(interference['mean_retention']),
            'separation_margin': float(separation['separation_margin_mean']),
            'separation_top1': float(separation['separation_top1_accuracy']),
            'continuity_margin': float(continuity['continuity_margin_mean']),
            'continuity_links': float(continuity['link_consistency']),
            'bridge_weight_delta': float(bridge_after - bridge_before),
            'homeostatic_offset_mean': float(homeostasis_after),
            'final_assembly_size': int(len(model.episode_records[-1].ca3_assembly)),
        }

    benchmarks = [
        micro_benchmark(use_text=True, ca3_exc=240, consolidate=True),
        micro_benchmark(use_text=False, ca3_exc=240, consolidate=True),
        micro_benchmark(use_text=True, ca3_exc=480, consolidate=True),
        micro_benchmark(use_text=True, ca3_exc=240, consolidate=False),
    ]
    return {'benchmarks': benchmarks}


def save_data(data: Dict[str, Any], path: str = 'research_site/data.js'):
    content = 'const SIM_DATA = ' + json.dumps(data, indent=2) + ';\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == '__main__':
    payload = {
        'v2v3': v2_v3_data(),
        'v4v5': v4_v5_data(),
        'v6': v6_data(),
    }
    save_data(payload)
    print('Generated research site data in research_site/data.js')
