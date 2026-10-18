"""Small wrapper around the OpenLineage client so each pipeline step shows up in Marquez."""

import json
import os
from pathlib import Path

import pandas as pd
from openlineage.client import OpenLineageClient
from openlineage.client.event_v2 import (
    InputDataset,
    Job,
    OutputDataset,
    Run,
    RunEvent,
    RunState,
)
from openlineage.client.facet_v2 import (
    documentation_dataset,
    documentation_job,
    job_type_job,
    schema_dataset,
)
from openlineage.client.serde import Serde
from openlineage.client.transport.http import HttpConfig, HttpTransport
from openlineage.client.uuid import generate_new_uuid

JOB_NAMESPACE = "airbnb-chicago-reporting"
DATA_NAMESPACE = "file://local"
PRODUCER = "https://github.com/yinw5/MLOps-Data-Versioning-"
EVENTS_LOG = Path("partb/output/lineage_events.jsonl")

MARQUEZ_URL = os.getenv("MARQUEZ_URL", "http://localhost:9000")
_client = OpenLineageClient(transport=HttpTransport(HttpConfig(url=MARQUEZ_URL)))


def _schema(path):
    """column names + dtypes, only reads a few rows"""
    sample = pd.read_csv(path, nrows=50)
    fields = [schema_dataset.SchemaDatasetFacetFields(name=c, type=str(t))
              for c, t in sample.dtypes.items()]
    return schema_dataset.SchemaDatasetFacet(fields=fields)


def _dataset(cls, path, note):
    facets = {"documentation": documentation_dataset.DocumentationDatasetFacet(description=note)}
    if Path(path).suffix == ".csv" and Path(path).exists():
        facets["schema"] = _schema(path)
    return cls(namespace=DATA_NAMESPACE, name=str(path), facets=facets)


def _emit(event):
    EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_LOG.open("a", encoding="utf-8") as f:
        f.write(Serde.to_json(event) + "\n")
    try:
        _client.emit(event)
    except Exception as err:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        # keep going if marquez isn't running, the events are still saved to the jsonl
        print(f"  (couldn't reach marquez: {err.__class__.__name__})")


def run_step(name, description, inputs, outputs, func):
    """run func() and send START / COMPLETE (or FAIL) events for it

    inputs / outputs are dicts of {path: short description}
    """
    run = Run(runId=str(generate_new_uuid()))
    job = Job(
        namespace=JOB_NAMESPACE,
        name=name,
        facets={
            "documentation": documentation_job.DocumentationJobFacet(description=description),
            "jobType": job_type_job.JobTypeJobFacet(
                processingType="BATCH", integration="PYTHON", jobType="TASK"
            ),
        },
    )
    ins = [_dataset(InputDataset, p, n) for p, n in inputs.items()]

    def event(state, outs):
        return RunEvent(eventType=state, eventTime=pd.Timestamp.now(tz="UTC").isoformat(),
                        run=run, job=job, producer=PRODUCER, inputs=ins, outputs=outs)

    print(f"[{name}] running")
    _emit(event(RunState.START, []))
    try:
        func()
    except Exception:
        _emit(event(RunState.FAIL, []))
        raise
    # schemas are read after the step so the output facets match what was written
    outs = [_dataset(OutputDataset, p, n) for p, n in outputs.items()]
    _emit(event(RunState.COMPLETE, outs))
    print(f"[{name}] done -> {', '.join(map(str, outputs))}")


def reset_log():
    """start a fresh events file each run"""
    if EVENTS_LOG.exists():
        EVENTS_LOG.unlink()


def count_events():
    """how many events were logged (for the summary at the end)"""
    if not EVENTS_LOG.exists():
        return 0
    return sum(1 for line in EVENTS_LOG.read_text(encoding="utf-8").splitlines()
               if json.loads(line))
