"""Experiment tracking — remember every run so you can compare them later.

Plain idea: a lab notebook for your model runs. Each time you train or tune,
you jot down *what settings you used* (params) and *how well it did* (metrics).
Later you can look back and say "run 7 was best — what did it use?"

Uses **MLflow** if it's installed (browse the runs with `mlflow ui`), and also
always writes a plain-text `experiments/runs.jsonl` so it works with zero setup.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from . import config

RUNS_DIR = config.ROOT / "experiments"
JSONL = RUNS_DIR / "runs.jsonl"

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:            # pragma: no cover
    _HAS_MLFLOW = False


def log_run(run_name: str, params: Dict[str, Any], metrics: Dict[str, float],
            tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Record one experiment run (its params + metrics + optional tags)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "run_name": run_name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": params,
        "metrics": metrics,
        "tags": tags or {},
    }
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    if _HAS_MLFLOW:
        mlflow.set_tracking_uri((config.ROOT / "mlruns").as_uri())
        mlflow.set_experiment("car-pricing")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            for k, v in (tags or {}).items():
                mlflow.set_tag(k, v)
    return record


def load_runs():
    """Return all logged runs as a flat DataFrame (params_* / metrics_* columns)."""
    import pandas as pd
    if not JSONL.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in JSONL.read_text(encoding="utf-8").splitlines() if line]
    flat = []
    for r in rows:
        d = {"run_name": r["run_name"], "time": r["time"]}
        d.update({f"param.{k}": v for k, v in r["params"].items()})
        d.update({f"metric.{k}": v for k, v in r["metrics"].items()})
        flat.append(d)
    return pd.DataFrame(flat)


if __name__ == "__main__":
    log_run("smoke-test", {"demo": True}, {"value": 1.0}, {"note": "hello"})
    print(load_runs())
    print(f"\nMLflow {'ON — run `mlflow ui`' if _HAS_MLFLOW else 'off (JSONL only)'}")
