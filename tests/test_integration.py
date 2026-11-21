"""End-to-end smoke test with the real models. Slow and needs the built data, so it is opt-in:

    CARAG_RUN_MODELS=1 pytest tests/test_integration.py
"""
import os
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "nq_conflict"
pytestmark = pytest.mark.skipif(
    os.environ.get("CARAG_RUN_MODELS") != "1" or not (DATA / "corpus.jsonl").exists(),
    reason="set CARAG_RUN_MODELS=1 and build data/nq_conflict to run the real-model smoke test",
)


def test_real_pipeline_trace():
    import argparse

    from carag import cli

    ap = argparse.ArgumentParser()
    cli._add_model_args(ap)
    rag = cli._build_system(ap.parse_args(["--data", str(DATA)]))
    out, a = rag.ask("who is given credit for inventing the aqua lung")
    assert len(a.retrieved) == 5 and a.arbitration.decision in {"answer", "disagree", "abstain"}
    assert "cousteau" in out.lower()
    # the benchmark plants a wrong answer in two low-reliability sources; arbitration must not adopt it
    assert a.arbitration.best is not None and "cousteau" in a.arbitration.best.answer.lower()
