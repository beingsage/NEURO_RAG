# Novelty / Causality Claim Framework

This repository can support a mechanistic memory story, but only if the claims stay
within what the experiments actually measure.

## What The Current System Claims

The model is organized as a clean `EC -> DG -> CA3 -> CA1` path, with direct
SQL/graph/text-to-CA3 shortcut wiring removed from the live recall path.

The strongest supported claims are:

- pattern completion from partial cues
- cross-modal retrieval across SQL, graph, and text
- learned temporal binding through DG->CA3 bridge delays and plasticity
- separation and interference behavior that can be studied at scale
- ablation-driven evidence for which mechanism changes which metric

## What Still Requires Careful Language

Do not claim full completion unless the large-scale runs support it.

In particular, the following need direct experimental evidence:

- 100K+ sequential episode retention
- robust relation accuracy at scale
- statistically significant reduction in false retrieval
- clear advantage of learned bridge delays over shuffled or zero-delay controls

## Evidence Map

Use the metrics below as the basis for the story:

- Architecture purity: `pure_ec_dg_ca3`, `ca3_input_sources`
- Pattern completion: retention vs cue fraction, CA3 Jaccard overlap
- Cross-modal retrieval: CERA matrix and off-diagonal mean
- Graph readout: relation accuracy, graph edge accuracy, structure accuracy
- Separation: separation margin, best impostor overlap, pairwise assembly overlap
- False retrieval: false retrieval rate under matched retrieval cues
- Causality: paired tests vs `full_model` for `no_delay`, `random_delay`,
  `no_stdp`, `no_dg`, and `no_ca3_recurrence`

## Recommended Protocol

Run the top-level experiment runner:

```bash
python run_experiments.py capacity --checkpoints 100 100000 --preset fast
python run_experiments.py ablation --seeds 0 1 2 3 4 5 6 7 8 9 --preset tuned
python run_experiments.py all
```

For explicit sweep control, override the memory config at the command line:

```bash
python run_experiments.py ablation \
  --override dg_bridge_lr=0.01 \
  --override dg_target_sparsity=0.01 \
  --override dg_output_dim=2000 \
  --override ca1_train_epochs=24 \
  --override ca1_relation_lr=0.08
```

The runner writes a markdown report, JSON metrics, and plots to a timestamped
output directory. On Kaggle it defaults to `/kaggle/working/research_experiments`.

## Interpretation Standard

The mechanism story is only strong if:

- the full model beats the no-delay and random-delay controls
- the no-DG and no-CA3-recurrence ablations degrade completion and retrieval
- relation accuracy and CERA improve after tightening CA1 readout
- false retrieval drops when bridge plasticity and DG sparsity are improved

If those conditions are not met, the right conclusion is that the model is a
promising prototype, not yet a completed 10/10 system.
