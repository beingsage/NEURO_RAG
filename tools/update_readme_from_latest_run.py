from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.MD"
LATEST_RUN_FILE = ROOT / "outputs" / "eval_runs" / "LATEST.txt"


def read_latest_run_dir() -> Path:
    if not LATEST_RUN_FILE.exists():
        raise FileNotFoundError(f"Missing latest run marker: {LATEST_RUN_FILE}")
    target = LATEST_RUN_FILE.read_text(encoding="utf-8").strip()
    run_dir = (ROOT / target).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Latest run directory not found: {run_dir}")
    return run_dir


def load_summary(run_dir: Path) -> dict:
    summary_file = run_dir / "summary.json"
    return json.loads(summary_file.read_text(encoding="utf-8"))


def fmt_metric_value(value):
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def metric_row(name: str, metric_value: str, result: str) -> str:
    return f"| {name} | {metric_value} | {result} |"


def build_overview(run_dir: Path, summary: dict) -> str:
    evaluations = summary.get("evaluations", {})

    def metric(name: str, key: str):
        entry = evaluations.get(name, {})
        if not entry:
            return None
        result = entry.get("passed")
        if result is None and isinstance(entry, dict):
            for nested_key, nested_value in entry.items():
                if isinstance(nested_value, dict):
                    if isinstance(nested_value.get("passed"), bool):
                        result = nested_value.get("passed")
                        break
        return result

    pattern_completion = evaluations.get("pattern_completion", {})
    engram_stability = evaluations.get("engram_stability", {})
    polychronous_binding = evaluations.get("polychronous_binding", {})
    stdp_causality = evaluations.get("stdp_causality", {})
    encoder = evaluations.get("encoder_reconstructibility", {})

    rows = [
        metric_row("Pattern completion", f"acc={pattern_completion.get('accuracy', 'n/a')}, overlap={pattern_completion.get('mean_target_overlap', 'n/a')}", "PASS" if pattern_completion.get("passed") else "FAIL"),
        metric_row("Engram stability", f"mean_jaccard={engram_stability.get('mean_jaccard', 'n/a')}", "PASS" if engram_stability.get("passed") else "FAIL"),
        metric_row("Polychronous binding", f"mean_abs_error_ms={polychronous_binding.get('mean_abs_error_ms', 'n/a')}", "PASS" if polychronous_binding.get("passed") else "FAIL"),
        metric_row("STDP causality", f"real_minus_shuffled={stdp_causality.get('real_minus_shuffled', 'n/a')}", "PASS" if stdp_causality.get("passed") else "FAIL"),
        metric_row("Encoder reconstructibility", f"age_err={encoder.get('age_abs_error', 'n/a')}, delay_err={encoder.get('graph_delay_abs_error_ms', 'n/a')}ms", "PASS" if encoder.get("passed") else "FAIL"),
    ]

    return "\n".join([
        "## Latest benchmark results",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Date: `{json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8')).get('timestamp_utc', 'unknown')}`",
        f"- Git SHA: `{json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8')).get('git_sha', 'unknown')}`",
        "",
        "| Evaluation | Key metric | Result |",
        "| --- | --- | --- |",
        *rows,
        "",
    ])


def build_image_gallery(run_dir: Path) -> str:
    figures_dir = run_dir / "figures"
    if not figures_dir.exists():
        return "## Latest figures\n\nNo figures available for the latest run.\n"

    preferred = [
        "summary_bars.png",
        "completion_curve.png",
        "pbg_relation_acc.png",
        "binding_index.png",
        "cera_heatmap.png",
    ]
    files = []
    seen = set()
    for name in preferred:
        path = figures_dir / name
        if path.exists():
            files.append(path)
            seen.add(name)
    for path in sorted(figures_dir.glob("*.png")):
        if path.name not in seen:
            files.append(path)

    if not files:
        return "## Latest figures\n\nNo PNG charts were generated for the latest run.\n"

    gallery = ["## Latest figures", ""]
    for path in files[:5]:
        rel = path.relative_to(ROOT).as_posix()
        gallery.append(f"![{path.name}]({rel})")
    gallery.append("")
    return "\n".join(gallery)


def build_readme() -> str:
    run_dir = read_latest_run_dir()
    summary = load_summary(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    top = """# Polychronous Multimodal Spiking Memory

A biologically inspired research project for episodic memory and multimodal binding using spiking neural dynamics, recurrent attractor networks, and polychronous spike timing behavior.

This repository explores how structured data (SQL), relational knowledge (graph), and text can be bound into a unified memory representation using a CA3-like recurrent architecture with delay-based synapses, sparse coding, and STDP-driven consolidation.

## Demo video

https://github.com/user-attachments/assets/df940053-5017-4940-bec4-19834507f77e

## Project overview

The system models a spiking-memory pipeline inspired by hippocampal computation:

- SQL records are encoded as sparse spike patterns
- Graph relationships are processed as relational edge events
- Text inputs are converted into semantic activation patterns
- Entorhinal and dentate-like layers perform convergence and sparse separation
- CA3 recurrent dynamics bind the multimodal signals into a shared episodic engram
- CA1-like readout reconstructs or retrieves associated representations

The implementation is centered around the main model in `spiking_multimodal_memory.py`, with supporting research notes, benchmarks, and evaluation runs in the repository.

## Key capabilities

- Multimodal memory binding across structured, relational, and textual inputs
- Sparse and recurrent attractor dynamics in CA3
- Delay-based synaptic integration and polychronous activation patterns
- Engram storage, familiarity detection, and replay-style memory updates
- Evaluation scripts for stability, reconstruction quality, and binding performance
- Research outputs and result summaries under the `outputs/` directory

<!-- AUTO-GENERATED:START -->
"""
    bottom = """
<!-- AUTO-GENERATED:END -->

## Repository structure

- `spiking_multimodal_memory.py` — core memory model and neural implementation
- `benchmark.py` — benchmark and evaluation utilities
- `run_experiments.py` — experiment orchestration and evaluation pipeline
- `SPIKING_MEMORY_SPEC.md` — engineering specification and design notes
- `misc/` — research documents and video assets
- `outputs/` — experiment summaries, figures, and JSON metrics
- `tools/` — helper scripts for sweeps and visualization
- `web/` — browser-based visualization assets

## Quick start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install numpy matplotlib seaborn
```

### 3. Run the benchmark or experiments

```bash
python benchmark.py
python run_experiments.py
```

### 4. Visualize results

```bash
python tools/visualize_latest_run.py
```

## Research notes

A set of technical and conceptual notes are available in the `misc/` folder, including:

- `misc/research.md`
- `misc/polychronous_multimodal_memory_extracted_methodology.md`
- `misc/NOVEL_RESEARCH_REFRAMING.md`
- `misc/NOVELTY_CAUSALITY.md`
- `misc/COMMANDS.MD`

These documents explain the design rationale, experimental framing, and method-level details behind the spiking memory model.

## Outputs

The `outputs/` directory contains experiment run summaries, JSON metrics, manifest files, and generated figures. These are useful for comparing runs, validating engram stability, and exploring the memory system’s behavioral trends.

## Notes

This project is primarily a research prototype and an experimental architecture for multimodal episodic memory. It is intended for exploration, evaluation, and extension rather than as a production-grade application system.

## License

This repository does not currently declare a license file. Please check with the project owner before reusing or redistributing the code or research artifacts for external publication or commercial use.
"""

    generated_section = "\n".join([
        build_overview(run_dir, summary),
        "",
        build_image_gallery(run_dir),
        "",
        f"- Latest run directory: `{manifest.get('run_id', run_dir.name)}`",
        "- Regenerate this section with: `python tools/update_readme_from_latest_run.py`",
        "",
    ])

    return top + generated_section + bottom


def main() -> None:
    updated = build_readme()
    README_PATH.write_text(updated, encoding="utf-8")
    print(f"README updated from latest run: {read_latest_run_dir().name}")


if __name__ == "__main__":
    main()
