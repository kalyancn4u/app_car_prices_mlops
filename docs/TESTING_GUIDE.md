# 🧪 Testing & Debugging Guide — novice → mastery

This guide teaches you to **test, debug, and troubleshoot** code using this repo as
a practice ground. No prior testing experience needed. The exercises live in
[`tests/test_stubs.py`](../tests/test_stubs.py) — a ladder you climb one rung at a time.

---

## 1. What is a test, really?

A **test** is a small piece of code that checks *"does this behave the way I expect?"*
— automatically. Instead of eyeballing output, you write:

```python
def test_addition():
    assert 2 + 2 == 4          # if this is false, the test FAILS and tells you
```

`assert X` means "I claim X is true." If it isn't, the test fails loudly and points
at the line. Tests are how you change code *without fear* — run them, and you know in
seconds whether you broke anything.

## 2. Running the tests

```bash
pip install -r requirements.txt && pip install -e .
pytest                 # run everything (quiet)
pytest -v              # verbose: one line per test
pytest -k drift        # only tests whose name contains "drift"
pytest -x              # stop at the first failure
pytest tests/test_stubs.py::test_kpi_thresholds_exist   # one specific test
```

You'll see dots (`.` = pass), `s` = skipped (that's our stubs, waiting for you), and
`F` = fail (with a traceback showing exactly what went wrong).

## 3. The three-line shape of every good test — **Arrange, Act, Assert**

```python
def test_clean_drops_bad_prices():
    df = pd.DataFrame({"selling_price": [5.0, -1.0], ...})   # Arrange: set up inputs
    out = data.clean(df)                                     # Act: run the thing
    assert (out["selling_price"] > 0).all()                 # Assert: check the result
```

Memorise **A-A-A** and you can write almost any test.

## 4. The difficulty ladder (in `tests/test_stubs.py`)

| Level | Focus | Skill it builds |
| :---- | :---- | :-------------- |
| 🟢 **1 — First steps** | trivial asserts, imports | how to run pytest; confidence |
| 🟡 **2 — Pure functions** | one function in, one value out | Arrange-Act-Assert |
| 🟠 **3 — Edge cases & errors** | bad input, boundaries, `pytest.raises` | defensive thinking |
| 🔴 **4 — Integration** | end-to-end: predict / the API / the registry | how the pieces connect |
| 🟣 **5 — Mastery** | `@parametrize`, property tests, fixtures, `monkeypatch` | professional testing |
| 🐞 **Debugging drills** | turn a *real past bug* into a regression test | the core debugging loop |

**To complete a stub:** delete its `@pytest.mark.skip(...)` line, replace the
`pytest.fail("TODO")` with real `assert`s, and run `pytest`.

## 5. How to debug when a test goes red

1. **Read the traceback bottom-up.** The last lines say *what* failed and *where*.
2. **Reproduce it small.** Run just that test: `pytest path::test_name -x`.
3. **Print or breakpoint.** Add `print(value)` or drop `breakpoint()` into the code and
   run `pytest -s` (the `-s` lets prints show).
4. **Check your assumptions.** Nine times in ten the *test's* expectation is wrong, not
   the code — verify what the function actually returns.
5. **Fix, re-run, repeat** until green. Then leave the test in place — it now guards
   against that bug forever. *That* is what "a regression test" means.

## 6. Troubleshooting cheat-sheet

| Symptom | Likely cause & fix |
| :------ | :----------------- |
| `ModuleNotFoundError: car_pricing` | Run `pip install -e .` (installs the package), or run pytest from the repo root. |
| `FileNotFoundError: price_pipeline.pkl` | Run `python -m car_pricing.train` once to create the model. |
| A stub "passes" but does nothing | You deleted the skip but left `pytest.fail` / no asserts — add real checks. |
| Test pollutes the repo with files | Use the `tmp_path` fixture + `monkeypatch` to redirect paths (see the registry stub). |
| Flaky test (passes sometimes) | You depend on randomness/time/order — fix a seed (`config.RANDOM_STATE`) or isolate state. |

## 7. In THIS repo — what's worth testing

The **debugging drills** reference this project's real history — the best way to learn:

- **XGBoost servability** (`docs/XGBOOST_SERVABILITY.md`): a library-version bug meant a
  model couldn't serve. Turn it into a regression test that loads the shipped pipeline and
  asserts `.predict()` works.
- **Band-vs-price consistency**: the band is *derived* from the price, so they can never
  disagree — write the test that proves it.
- The MLOps modules (`validation`, `drift`, `registry`, `monitoring`, `tuning`, `explain`)
  are all small and pure — ideal Level 2–4 practice.

> **The mastery mindset:** every bug you fix should leave behind a test that would have
> caught it. Do that, and your code only ever gets *more* reliable over time.
