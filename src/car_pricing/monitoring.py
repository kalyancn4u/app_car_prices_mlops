"""Monitoring — watch the model in the wild after it ships.

Plain idea: once a model is live, you want to know (1) what it's being asked and
what it answered, and (2) once the *real* prices come in later, how far off it
was. This logs every prediction to monitoring/predictions.jsonl; when you learn
the true price, you attach it and can compute the live error.

Dependency-free. In a real system these logs would flow to a dashboard
(Grafana / an MLflow or Evidently monitoring service). See docs/MLOPS_GUIDE.md.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from . import config

MON_DIR = config.ROOT / "monitoring"
PRED_LOG = MON_DIR / "predictions.jsonl"


def log_prediction(inputs: Dict[str, Any], predicted_lakhs: float,
                   actual_lakhs: Optional[float] = None) -> Dict[str, Any]:
    """Append one served prediction (and optionally the known true price)."""
    MON_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": inputs,
        "predicted_lakhs": round(float(predicted_lakhs), 2),
        "actual_lakhs": actual_lakhs,
    }
    with open(PRED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_predictions():
    import pandas as pd
    if not PRED_LOG.exists():
        return pd.DataFrame()
    rows = [json.loads(x) for x in PRED_LOG.read_text(encoding="utf-8").splitlines() if x]
    return pd.DataFrame(rows)


def live_metrics() -> Dict[str, float]:
    """MAE over logged predictions that have a known actual price yet."""
    import numpy as np
    df = load_predictions()
    if df.empty or "actual_lakhs" not in df or df["actual_lakhs"].notna().sum() == 0:
        return {"n_predictions": int(len(df)), "n_with_actual": 0, "live_mae_lakhs": None}
    labelled = df.dropna(subset=["actual_lakhs"])
    mae = float(np.mean(np.abs(labelled["predicted_lakhs"] - labelled["actual_lakhs"])))
    return {
        "n_predictions": int(len(df)),
        "n_with_actual": int(len(labelled)),
        "live_mae_lakhs": round(mae, 3),
        "kpi_breached": mae > config.KPI["max_mae_lakhs"],
    }


if __name__ == "__main__":  # `python -m car_pricing.monitoring`
    print(live_metrics())
