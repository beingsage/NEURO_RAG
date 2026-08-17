#!/usr/bin/env python3
"""
Visualize the latest benchmark run under `outputs/eval_runs/`.

Generates PNG figures in the run's `figures/` directory and a small
`index.html` to review them.

Usage: python3 tools/visualize_latest_run.py
"""
import json
from pathlib import Path
import math
import sys

try:
    import matplotlib.pyplot as pltgh
    import seaborn as sns
    import numpy as np
except Exception as e:  # pragma: no cover - plotting requires deps
    print("Missing plotting dependencies. Install matplotlib, seaborn, numpy.")
    raise


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "outputs" / "eval_runs" / "LATEST.txt"


def find_latest_run() -> Path:
    if not LATEST.exists():
        # fallback: choose newest directory under outputs/eval_runs
        base = ROOT / "outputs" / "eval_runs"
        runs = [p for p in base.iterdir() if p.is_dir()]
        if not runs:
            raise FileNotFoundError("No eval_runs found")
        runs.sort()
        return runs[-1]
    path = LATEST.read_text(encoding="utf-8").strip()
    return Path(path)


def load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def ensure_fig_dir(run_dir: Path) -> Path:
    fig = run_dir / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    return fig


def plot_summary_bars(evals: dict, out: Path):
    rows = []
    for k, v in evals.items():
        if k == "system_metrics":
            continue
        # try to find single scalar key for summary
        if isinstance(v, dict):
            if "accuracy" in v:
                val = float(v["accuracy"])
            elif "mean_jaccard" in v:
                val = float(v["mean_jaccard"])
            elif "mean_abs_error_ms" in v:
                # lower is better; invert for visualization
                val = -float(v["mean_abs_error_ms"])
            elif "separation_gain_mean" in v:
                val = float(v["separation_gain_mean"])
            elif "real_minus_shuffled" in v:
                val = float(v["real_minus_shuffled"])
            else:
                # fallback: try numeric fields
                nums = [float(x) for x in v.values() if isinstance(x, (int, float))]
                val = float(nums[0]) if nums else 0.0
        else:
            val = float(v) if isinstance(v, (int, float)) else 0.0
        rows.append((k, val))

    if not rows:
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    plt.figure(figsize=(10, max(3, 0.3 * len(names))))
    sns.barplot(x=vals, y=names, palette="viridis")
    plt.title("Evaluation summary (numeric proxies)")
    plt.tight_layout()
    plt.savefig(out / "summary_bars.png", dpi=150)
    plt.close()


def plot_cera_heatmap(evals: dict, out: Path):
    cera = None
    if "cera" in evals and isinstance(evals["cera"], dict):
        cera = evals["cera"].get("cera_matrix") or evals["cera"].get("cera")
    if not cera:
        return

    # cera: cue -> {target: value}
    cues = sorted(cera.keys())
    targets = sorted(next(iter(cera.values())).keys())
    mat = np.zeros((len(cues), len(targets)))
    for i, c in enumerate(cues):
        for j, t in enumerate(targets):
            mat[i, j] = float(cera.get(c, {}).get(t, 0.0))

    plt.figure(figsize=(6 + len(targets) * 0.6, 4 + len(cues) * 0.3))
    sns.heatmap(mat, xticklabels=targets, yticklabels=cues, annot=True, fmt=".2f", cmap="magma")
    plt.xlabel("Target modality")
    plt.ylabel("Cue modality")
    plt.title("CERA (cross-modal retrieval matrix)")
    plt.tight_layout()
    plt.savefig(out / "cera_heatmap.png", dpi=150)
    plt.close()


def plot_binding_index(evals: dict, out: Path):
    bi = None
    if "binding_index" in evals and isinstance(evals["binding_index"], dict):
        bi = evals["binding_index"].get("binding_index") or evals["binding_index"]
    if not bi:
        return
    keys = sorted(bi.keys())
    vals = [float(bi[k]) for k in keys]
    plt.figure(figsize=(8, max(3, 0.35 * len(keys))))
    sns.barplot(x=vals, y=keys, palette="coolwarm")
    plt.xlabel("Binding Index (higher = stronger binding)")
    plt.tight_layout()
    plt.savefig(out / "binding_index.png", dpi=150)
    plt.close()


def plot_pbg(evals: dict, out: Path):
    pbg = evals.get("polychronous_binding_gain")
    if not pbg or not isinstance(pbg, dict):
        return
    labels = ["structured", "zero_delay", "random_delay"]
    vals = [
        float(pbg.get("structured_mean_abs_error_ms", pbg.get("structured_mean_relation_acc", 0.0))),
        float(pbg.get("zero_delay_mean_abs_error_ms", pbg.get("zero_delay_mean_relation_acc", 0.0))),
        float(pbg.get("random_delay_mean_abs_error_ms", pbg.get("random_delay_mean_relation_acc", 0.0))),
    ]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=vals, palette="Spectral")
    plt.ylabel("Mean bridge arrival error (ms)")
    plt.title("DG bridge timing: structured vs zero/random delays")
    plt.tight_layout()
    plt.savefig(out / "pbg_relation_acc.png", dpi=150)
    plt.close()


def plot_completion_curve(evals: dict, out: Path):
    sysm = evals.get("system_metrics")
    curve = None
    if sysm and isinstance(sysm, dict):
        cc = sysm.get("completion_curve")
        if isinstance(cc, dict):
            curve = cc.get("curve") or cc

    if not curve:
        return

    xs = sorted([float(k) for k in curve.keys()])
    ys = [float(curve[str(x)]) if str(x) in curve else float(curve[x]) for x in xs]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Cue fraction")
    plt.ylabel("Mean retrieval Jaccard")
    plt.title("Completion curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "completion_curve.png", dpi=150)
    plt.close()


def make_index_html(figs: list, out: Path):
    html = ["<html><head><meta charset='utf-8'><title>Run Figures</title></head><body>"]
    html.append(f"<h1>Figures for run: {out.parent.name}</h1>")
    for f in figs:
        html.append(f"<div style='margin:16px'><h3>{f.name}</h3><img src='{f.name}' style='max-width:100%'></div>")
    html.append("</body></html>")
    (out / "index.html").write_text('\n'.join(html), encoding="utf-8")


def main():
    run_dir = find_latest_run()
    print("Using run:", run_dir)
    summary = load_summary(run_dir)
    evals = summary.get("evaluations") or summary.get("summary", {}).get("evaluations") or {}
    figdir = ensure_fig_dir(run_dir)

    figs = []
    try:
        plot_summary_bars(evals, figdir)
        figs.append(figdir / "summary_bars.png")
    except Exception as e:
        print("Failed to plot summary bars:", e)
    try:
        plot_cera_heatmap(evals, figdir)
        figs.append(figdir / "cera_heatmap.png")
    except Exception as e:
        print("Failed to plot CERA:", e)
    try:
        plot_binding_index(evals, figdir)
        figs.append(figdir / "binding_index.png")
    except Exception as e:
        print("Failed to plot Binding Index:", e)
    try:
        plot_pbg(evals, figdir)
        figs.append(figdir / "pbg_relation_acc.png")
    except Exception as e:
        print("Failed to plot PBG:", e)
    try:
        plot_completion_curve(evals, figdir)
        figs.append(figdir / "completion_curve.png")
    except Exception as e:
        print("Failed to plot completion curve:", e)

    # filter existing
    figs_existing = [f for f in figs if f.exists()]
    if figs_existing:
        make_index_html(figs_existing, figdir)
        print(f"Wrote {len(figs_existing)} figures to {figdir}")
        print("Open:", figdir / "index.html")
    else:
        print("No figures generated.")


if __name__ == "__main__":
    main()
