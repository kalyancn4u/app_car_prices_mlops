"""Model registry — version every model so you can always roll back.

Plain idea: like "Save As… v1, v2, v3" for your model. Each time you register
the current model, we copy it into models/registry/<version>/ with its metrics
and a short content-hash (a fingerprint), and record it in an index. You can
mark one version as "production" and, if a new one misbehaves, roll back.

Dependency-free. The heavier tools are the **MLflow Model Registry** and **DVC**
(data/model version control) — see docs/MLOPS_GUIDE.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config

REGISTRY = config.MODELS_DIR / "registry"
INDEX = REGISTRY / "index.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def load_index() -> List[Dict]:
    return json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else []


def _save_index(entries: List[Dict]) -> None:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def register(pipeline_path: Optional[Path] = None, metrics: Optional[Dict] = None,
             note: str = "", stage: str = "staging") -> Dict:
    """Copy the current model into the registry as a new immutable version."""
    pipeline_path = Path(pipeline_path or config.PIPELINE_PATH)
    if metrics is None:
        metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))

    entries = load_index()
    version = f"v{len(entries) + 1}-{time.strftime('%Y%m%d-%H%M%S')}"
    dest = REGISTRY / version
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pipeline_path, dest / "price_pipeline.pkl")
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    entry = {
        "version": version,
        "hash": _hash(pipeline_path),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage,
        "note": note,
        "metrics": {k: metrics.get(k) for k in
                    ("winner", "price_r2", "price_mae_lakhs", "band_accuracy")},
    }
    entries.append(entry)
    _save_index(entries)
    return entry


def promote(version: str, stage: str = "production") -> Dict:
    """Mark a version as the given stage; demote any other in that stage."""
    entries = load_index()
    found = None
    for e in entries:
        if e["stage"] == stage and e["version"] != version:
            e["stage"] = "archived"
        if e["version"] == version:
            e["stage"] = stage
            found = e
    if found is None:
        raise KeyError(f"version {version!r} not in registry")
    _save_index(entries)
    return found


def latest(stage: Optional[str] = None) -> Optional[Dict]:
    entries = [e for e in load_index() if stage is None or e["stage"] == stage]
    return entries[-1] if entries else None


if __name__ == "__main__":  # `python -m car_pricing.registry`
    entry = register(note="demo registration")
    print("registered:", entry["version"], entry["metrics"])
