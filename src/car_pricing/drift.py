"""Drift detection — has new data drifted away from the training data?

Plain idea: a model learned from *last year's* cars. If *this year's* cars are
systematically different (older, higher mileage, new models), the model's guesses
get worse even though the code never changed. "Drift" is that slow mismatch — and
it's the signal to retrain.

We measure it two dependency-free ways per numeric feature:
  * **PSI** (Population Stability Index): how much a feature's distribution moved.
      < 0.1 = no real drift · 0.1–0.2 = moderate · > 0.2 = significant.
  * **KS-test** (scipy): a statistical test for "are these two samples different?".

**Evidently** is the richer upgrade — it produces full drift *reports/dashboards*
(and it's installed here). See docs/MLOPS_GUIDE.md.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config

PSI_THRESHOLD = 0.2


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, edges)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame,
                 features: Optional[List[str]] = None,
                 psi_threshold: float = PSI_THRESHOLD) -> Dict:
    """Compare `current` data to a `reference` (usually the training data).

    Returns {'drift_detected', 'n_drifted', 'report'} where report is a per-feature
    DataFrame of PSI / KS p-value / a drift flag.
    """
    features = features or config.NUMERIC
    rows = []
    for c in features:
        if c not in reference or c not in current:
            continue
        ref = pd.to_numeric(reference[c], errors="coerce").dropna().values
        cur = pd.to_numeric(current[c], errors="coerce").dropna().values
        p = psi(ref, cur)
        ks_p = float(ks_2samp(ref, cur).pvalue)
        rows.append({"feature": c, "psi": round(p, 3),
                     "ks_pvalue": round(ks_p, 4),
                     "drift": bool(p > psi_threshold or ks_p < 0.05)})
    report = pd.DataFrame(rows)
    return {
        "drift_detected": bool(report["drift"].any()) if not report.empty else False,
        "n_drifted": int(report["drift"].sum()) if not report.empty else 0,
        "report": report,
    }


if __name__ == "__main__":  # `python -m car_pricing.drift`
    from . import data
    df = data.clean(data.load_raw())
    ref = df.sample(frac=0.5, random_state=1)
    shifted = df.sample(frac=0.5, random_state=2).copy()
    shifted["age"] = shifted["age"] + 8          # simulate "older cars this year"
    out = detect_drift(ref, shifted)
    print(f"drift_detected={out['drift_detected']} (n_drifted={out['n_drifted']})")
    print(out["report"].to_string(index=False))
