"""
Scaffold runners for public datasets (FB15k-237, HippoVlog).
These are lightweight scaffolds that check for dataset presence and
outline the steps to run them with the existing codebase.

They do NOT download datasets automatically; place datasets under
`data/FB15k-237` and `data/HippoVlog` respectively.
"""
from pathlib import Path
from typing import Dict

DATA_ROOT = Path("data")


def run_fb15k237_scaffold(path: Path = DATA_ROOT / "FB15k-237") -> Dict[str, str]:
    """Check dataset folder and provide next-step commands for running graph experiments.

    Expected layout:
      data/FB15k-237/train.txt
      data/FB15k-237/valid.txt
      data/FB15k-237/test.txt

    This scaffold describes how to convert triples into the network's episodic format
    and where to plug in the graph-only evaluation.
    """
    if not path.exists():
        return {"status": "missing", "message": f"Place FB15k-237 files under: {path.resolve()}"}

    msg = (
        "FB15k-237 seems present. Next steps:\n"
        "1. Convert triples to episode tuples (sql_row, graph_edge, text) where `text` can be empty.\n"
        "2. Use `spiking_multimodal_memory.MultiModalMemory.ingest_event_stream` to ingest episodes.\n"
        "3. Run `benchmark.py` or call `compute_graph_retrieval_accuracy` on held-out test episodes.\n"
    )
    return {"status": "ok", "message": msg, "path": str(path.resolve())}


def run_hippovlog_scaffold(path: Path = DATA_ROOT / "HippoVlog") -> Dict[str, str]:
    """Check HippoVlog dataset location and outline steps.

    Expected artifacts:
      - Video files and subtitles/transcripts
      - QA JSON with timestamps

    Recommended approach: precompute modality encodings (semantic text vectors, graph facts per question),
    and then feed episodes to `MultiModalMemory.ingest_event_stream`. Use `benchmark.py` for evaluations.
    """
    if not path.exists():
        return {"status": "missing", "message": f"Place HippoVlog dataset under: {path.resolve()}"}

    msg = (
        "HippoVlog present. Next steps:\n"
        "1. Extract transcripts and align QA timestamps.\n"
        "2. Convert segments to episodes with `sql_row` (metadata), `graph_edge` (if any), and `text` (transcript).\n"
        "3. Ingest with `MultiModalMemory.ingest_event_stream` and run retrieval evaluations.\n"
    )
    return {"status": "ok", "message": msg, "path": str(path.resolve())}


if __name__ == "__main__":
    import json
    print(json.dumps(run_fb15k237_scaffold(), indent=2))
