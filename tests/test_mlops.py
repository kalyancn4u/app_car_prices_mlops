"""Tests for the MLOps toolkit modules (validation, drift, registry, monitoring,
explainability). Fast + dependency-light; run with `pytest -q`."""

import numpy as np
import pandas as pd
import pytest

from car_pricing import config, data, drift, explain, monitoring, registry, validation


# --- Data validation -------------------------------------------------------
def test_validation_passes_on_clean_data():
    """The cleaned dataset passes every validation rule."""
    rep = validation.validate_dataframe(data.clean(data.load_raw()))
    assert rep.ok, rep.summary()


def test_validation_flags_nonpositive_price():
    """A non-positive price is reported as a hard validation failure."""
    df = pd.DataFrame({config.TARGET: [-1.0], "make": ["X"], "model": ["Y"],
                       **{c: [1] for c in config.NUMERIC}})
    rep = validation.validate_dataframe(df)
    assert not rep.ok


def test_validation_raises_on_missing_column():
    """A missing required column raises DataValidationError when raise_on_error=True."""
    with pytest.raises(validation.DataValidationError):
        validation.validate_dataframe(pd.DataFrame({"make": ["X"]}), raise_on_error=True)


# --- Drift detection -------------------------------------------------------
def test_psi_is_zero_for_identical_samples():
    """PSI of a sample against itself is ~0 (no drift)."""
    x = np.arange(1000.0)
    assert drift.psi(x, x) < 1e-6


def test_no_drift_on_same_data():
    """Comparing data to an identical copy reports no drift."""
    df = data.clean(data.load_raw())
    assert not drift.detect_drift(df, df.copy())["drift_detected"]


def test_drift_flagged_on_shifted_data():
    """Shifting 'age' by 15 years is detected as drift in at least one feature."""
    df = data.clean(data.load_raw())
    shifted = df.copy()
    shifted["age"] = shifted["age"] + 15
    out = drift.detect_drift(df, shifted)
    assert out["drift_detected"] and out["n_drifted"] >= 1


# --- Model registry --------------------------------------------------------
def test_registry_register_and_promote(tmp_path, monkeypatch):
    """register() writes a versioned copy; promote() marks it 'production'."""
    monkeypatch.setattr(registry, "REGISTRY", tmp_path / "registry")
    monkeypatch.setattr(registry, "INDEX", tmp_path / "registry" / "index.json")
    entry = registry.register(note="test")
    assert (tmp_path / "registry" / entry["version"] / "price_pipeline.pkl").exists()
    registry.promote(entry["version"], "production")
    assert registry.latest("production")["version"] == entry["version"]


# --- Monitoring ------------------------------------------------------------
def test_monitoring_logs_and_computes_live_mae(tmp_path, monkeypatch):
    """A logged prediction with a known actual yields the expected live MAE."""
    monkeypatch.setattr(monitoring, "MON_DIR", tmp_path)
    monkeypatch.setattr(monitoring, "PRED_LOG", tmp_path / "pred.jsonl")
    monitoring.log_prediction({"make": "MARUTI"}, 5.0, actual_lakhs=4.5)
    m = monitoring.live_metrics()
    assert m["n_with_actual"] == 1 and abs(m["live_mae_lakhs"] - 0.5) < 1e-6


# --- Explainability --------------------------------------------------------
def test_importances_cover_all_features():
    """Built-in importances return exactly one value per model feature."""
    imp = explain.builtin_importances()
    assert len(imp) == len(config.FEATURES)
