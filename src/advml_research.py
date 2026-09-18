"""Small, offline research-evidence teaching utilities; no platform integration.

The fixture is fictional. An audit checks declared metadata, not truth, permissions,
scientific validity or paper readiness. Inputs are never written to disk/network.
"""
from __future__ import annotations
import json
import math
import statistics

REQUIRED = ("question", "dataset_version", "split_id", "code_revision",
            "environment", "metric", "artifact", "limitations")


def synthetic_runs() -> list[dict]:
    """Return five fresh fictional paired rows (seed:int, a/b:float error).

    Lower error is better. Deterministic fixture, no random state or downloads.
    Each seed denotes a matched condition, not an independent real-world dataset.
    """
    return [dict(seed=i, a=a, b=b) for i, (a, b) in enumerate(
        [(0.30, 0.28), (0.25, 0.24), (0.32, 0.33), (0.28, 0.25), (0.35, 0.31)])]


def paired_summary(rows: list[dict]) -> dict:
    """Summarize matched rows using d=a-b, so positive values favor B.

    Require >=2 unique integer seeds and finite nonnegative numerical errors.
    Return n, mean_difference, sample_sd, wins_b and differences. No CI/p-value
    or population generalization is inferred. Raise ValueError on malformed data.
    Does not mutate rows. Example: paired_summary(synthetic_runs())[\"n\"] == 5.
    """
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Provide at least two paired runs.")
    seeds, diffs = set(), []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("seed")) is not int:
            raise ValueError("Each row needs an integer seed.")
        if row["seed"] in seeds:
            raise ValueError("Duplicate seed: pairing is ambiguous.")
        seeds.add(row["seed"])
        values = [row.get("a"), row.get("b")]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Errors must be finite nonnegative numbers.")
        diffs.append(values[0] - values[1])
    return dict(n=len(diffs), mean_difference=statistics.mean(diffs),
                sample_sd=statistics.stdev(diffs), wins_b=sum(d > 0 for d in diffs),
                differences=diffs)


def example_record() -> dict:
    """Return fresh fictional metadata and rows; identifiers are teaching labels.

    No external artifacts are asserted to exist. Use this to exercise the audit,
    not as provenance for a real experiment. No arguments or side effects.
    """
    return dict(question="Does B reduce error in this fictional regime?",
                dataset_version="fictional-synthetic-v1", split_id="fictional-held-out-v1",
                code_revision="teaching-fixture-only", environment="fictional measurement fixture",
                metric="error (lower is better)", artifact="embedded fictional rows",
                limitations="Five synthetic pairs; no real data or external validity test.",
                runs=synthetic_runs())


def audit_record(record: dict) -> dict:
    """Check required nonempty string fields and summarize record['runs'].

    Returns metadata_complete, missing_fields, comparison, limitations. Structural
    completeness does not verify the declared values. Raises ValueError for a
    non-object input or invalid paired rows. Never modifies the input.
    """
    if not isinstance(record, dict):
        raise ValueError("The record must be a JSON object.")
    missing = [key for key in REQUIRED
               if not isinstance(record.get(key), str) or not record[key].strip()]
    return dict(metadata_complete=not missing, missing_fields=missing,
                comparison=paired_summary(record.get("runs")),
                limitations="Structure and arithmetic only; source truth, leakage, novelty and validity are not verified.")


def audit_view(text: str) -> dict:
    """UI callback: parse JSON text and return audit or a readable error object.

    Handles malformed JSON and invalid records without pretending success.
    No credentials, platform calls, file writes or network operations.
    """
    try:
        return audit_record(json.loads(text))
    except (ValueError, TypeError) as exc:
        return {"error": str(exc), "metadata_complete": False}


def build_audit_app():
    """Construct (do not launch) a Gradio Blocks app for fictional JSON records.

    Requires Gradio. Returns a Blocks object; caller decides whether/where to
    launch. Includes complete and incomplete example records. No external access.
    """
    import gradio as gr
    complete = example_record()
    incomplete = example_record()
    incomplete.pop("dataset_version")
    with gr.Blocks() as app:
        gr.Markdown("# Research evidence audit\nFictional teaching data. Passing checks does not certify scientific validity.")
        entry = gr.Textbox(value=json.dumps(complete, indent=2), lines=16, label="Experiment record (JSON)")
        button = gr.Button("Audit evidence record")
        result = gr.JSON(label="Findings and limits")
        button.click(audit_view, inputs=entry, outputs=result)
        gr.Examples([[json.dumps(complete)], [json.dumps(incomplete)], ["not JSON"]], inputs=entry)
    return app
