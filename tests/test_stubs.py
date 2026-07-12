"""Guided test stubs — a ladder from complete novice to mastery. 🧗

HOW TO USE
    1. Pick a stub (start at Level 1).
    2. Delete its `@pytest.mark.skip(...)` line.
    3. Replace `pytest.fail(...)` with real `assert` statements (the docstring says what).
    4. Run `pytest -v` until it's green, then climb to the next level.

    pytest -v            # working tests pass; these stubs show as SKIPPED

Reference patterns live in tests/test_pipeline.py and tests/test_mlops.py.
Difficulty: 🟢 novice · 🟡 pure functions · 🟠 edge cases · 🔴 integration · 🟣 mastery.
"""

import pytest

from car_pricing import config, data, features   # noqa: F401 (used once you implement)

TODO = "stub — delete this @skip, then implement (see docstring)"


# ── Level 1 · 🟢 First steps — confidence + how to run pytest ───────────────
@pytest.mark.skip(reason=TODO)
def test_feature_set_is_compact():
    """The production feature set is target-encoded, not one-hot.
    Assert `len(config.FEATURES)` == 16 (2 encoded + 5 numeric + 9 flags)."""
    pytest.fail("TODO")


@pytest.mark.skip(reason=TODO)
def test_kpi_thresholds_exist():
    """Assert `config.KPI` has the three gates: max_mae_lakhs, min_r2, min_band_accuracy."""
    pytest.fail("TODO")


# ── Level 2 · 🟡 Pure functions — arrange, act, assert ──────────────────────
@pytest.mark.skip(reason=TODO)
def test_price_to_band_maps_low_mid_high():
    """`features.price_to_band` buckets a price using the tercile edges.
    Compute `edges = features.band_edges(y)` on the real target, then assert a low,
    a middle, and a high price map to 'Low', 'Medium', 'High' respectively."""
    pytest.fail("TODO")


@pytest.mark.skip(reason=TODO)
def test_clean_uppercases_make_and_model():
    """Feed a tiny DataFrame with `' maruti '` / `'swift vxi'` through `data.clean`
    and assert they come out `'MARUTI'` / `'SWIFT VXI'`."""
    pytest.fail("TODO")


# ── Level 3 · 🟠 Edge cases & errors — think defensively ────────────────────
@pytest.mark.skip(reason=TODO)
def test_drift_psi_is_symmetricish_and_zero_on_identical():
    """From `car_pricing.drift`: assert `psi(x, x)` ≈ 0 for any array x, and that
    `psi` grows when you shift the second sample. Use numpy arrays."""
    pytest.fail("TODO")


@pytest.mark.skip(reason=TODO)
def test_validation_rejects_missing_required_column():
    """`car_pricing.validation.validate_dataframe(df, raise_on_error=True)` should
    raise `DataValidationError` when a required column is absent. Use `pytest.raises`."""
    pytest.fail("TODO")


# ── Level 4 · 🔴 Integration — wire the pieces together ─────────────────────
@pytest.mark.skip(reason=TODO)
def test_serve_predict_endpoint_returns_200():
    """Use the Flask test client to POST to the serving API:
        from car_pricing.serve import app
        client = app.test_client()
        r = client.post('/predict', json={'make':'MARUTI','model':'SWIFT VXI'})
    Assert status 200 and that the JSON has 'predicted_price_lakhs' and 'price_band'."""
    pytest.fail("TODO")


@pytest.mark.skip(reason=TODO)
def test_registry_rollback(tmp_path, monkeypatch):
    """Register two models, promote v1 to production, then 'roll back' by promoting v1
    again after v2 — assert `registry.latest('production')` is the one you expect.
    (Monkeypatch `registry.REGISTRY` and `registry.INDEX` to `tmp_path` first.)"""
    pytest.fail("TODO")


# ── Level 5 · 🟣 Mastery — parametrize, properties, fixtures ────────────────
@pytest.mark.parametrize("make,model,floor", [
    ("MARUTI", "SWIFT VXI", 2.0),
    ("BMW", "X5", 10.0),
])
@pytest.mark.skip(reason=TODO)
def test_predictions_are_in_plausible_ranges(make, model, floor):
    """Parametrized: `predict({...})['predicted_price_lakhs']` should exceed `floor`
    for each car. (This is how you test many cases without repeating yourself.)"""
    pytest.fail("TODO")


@pytest.mark.skip(reason=TODO)
def test_property_premium_costs_more_than_hatchback():
    """A *property* the model must always satisfy: a BMW X5 predicts higher than a
    Maruti Swift. Assert it. (Property tests encode domain truths, not exact numbers.)"""
    pytest.fail("TODO")


# ── 🐞 Debugging drills — write a regression test for a REAL past bug ────────
@pytest.mark.skip(reason=TODO)
def test_regression_shipped_model_is_servable_in_a_pipeline():
    """History (docs/XGBOOST_SERVABILITY.md): XGBoost couldn't `.predict()` inside a
    sklearn Pipeline on xgboost 2.1 x sklearn 1.6. Write the regression test that would
    catch a re-break: load `models/price_pipeline.pkl` and assert `.predict(X)` works
    on a small sample. (This is the essence of debugging: turn a fixed bug into a test.)"""
    pytest.fail("TODO")


@pytest.mark.skip(reason=TODO)
def test_regression_band_never_contradicts_price():
    """History: an earlier design let the band disagree with the ₹ figure. Here the band
    is *derived* from the price, so they can't. Prove it: for several prices, assert the
    predicted band's rupee interval actually contains the predicted price."""
    pytest.fail("TODO")
