# 🧭 The MLOps Guide — all the stages, explained for a complete beginner

Training a model is maybe **10%** of the job. The other **90%** is everything
around it: making sure the data is clean, remembering what you tried, explaining
the model, versioning it, shipping it, and watching it in the wild. That "other
90%" is **MLOps** (Machine-Learning Operations).

This guide walks through **every** MLOps stage this repo now implements — each in
plain English, with the exact command to run it. Nothing here assumes prior
DevOps or ML-infra experience.

> **Everything here really runs.** Where a famous tool isn't installed, we use a
> simple built-in version so you can see it work today, and we name the popular
> upgrade for later. All the code lives in `src/car_pricing/`.

---

## The map (11 stages)

| Stage | What it answers | Module / file | Upgrade tool |
| :---- | :-------------- | :------------ | :----------- |
| 1. Data validation | "Is the incoming data sane?" | `validation.py` | pandera / Great Expectations |
| 2. Hyperparameter tuning | "What model settings are best?" | `tuning.py` | Optuna |
| 3. Experiment tracking | "What did I try, and how did it do?" | `tracking.py` | MLflow *(used here)* |
| 4. Explainability | "Why did it predict that?" | `explain.py` | SHAP |
| 5. Data/model versioning | "Can I roll back to an old model?" | `registry.py` | MLflow Registry / DVC |
| 6. CI (Continuous Integration) | "Did my change break anything?" | `.github/workflows/ci.yml` | — |
| 7. CD (Continuous Deployment) | "Ship the new version automatically" | `.github/workflows/cd.yml` + `Dockerfile` | — |
| 8. CT (Continuous Training) | "Retrain when data changes" | `.github/workflows/ct.yml` | — |
| 9. Deployment / serving | "Let other apps use the model" | `serve.py` + `Dockerfile` | FastAPI / cloud |
| 10. Monitoring | "Is it still accurate in production?" | `monitoring.py` | Grafana / Evidently service |
| 11. Drift detection | "Has the world changed under the model?" | `drift.py` | Evidently *(installed)* |
| — Governance | "Who approved this model, and can we audit it?" | [`GOVERNANCE.md`](GOVERNANCE.md) | Model registry + policy |

**One-command tour** (runs the built-in versions of most stages):

```bash
pip install -r requirements.txt && pip install -e .
python -m car_pricing.validation      # stage 1
python -m car_pricing.tuning          # stage 2 (+ logs to tracking, stage 3)
python -m car_pricing.explain         # stage 4
python -m car_pricing.registry        # stage 5
python -m car_pricing.drift           # stage 11
python -m car_pricing.serve           # stage 9 (Ctrl-C to stop)
```

---

## 1. Data validation — *a bouncer for your data*

**What.** Before data trains a model (or reaches a prediction), check it's the
shape we expect: right columns, no impossible values (a car can't be 500 years
old or have a negative price).

**Why.** "Garbage in, garbage out." A silent bad row can quietly wreck a model.

**How here.** `validation.py` runs schema + range checks and gives a clear
pass/fail report:

```bash
python -m car_pricing.validation
# -> Data validation on 19,820 rows: PASS
```

**Upgrade:** [`pandera`](https://pandera.readthedocs.io) or **Great Expectations**
let you declare the schema as data and generate validation reports.

---

## 2. Hyperparameter tuning — *let the computer turn the knobs*

**What.** A model has settings ("how many trees? how deep? learning rate?").
Instead of guessing, try many combinations and keep the best (scored fairly with
cross-validation).

**Why.** Free accuracy: the same model, better-configured, predicts better.

**How here.** `tuning.py` uses scikit-learn's `RandomizedSearchCV`:

```bash
python -m car_pricing.tuning
# -> Best CV MAE for LightGBM: Rs 0.69x Lakhs   (beats the untuned default)
```

**Upgrade:** [`Optuna`](https://optuna.org) searches *smartly* — it learns which
settings to try next instead of picking at random.

---

## 3. Experiment tracking — *a lab notebook for your runs*

**What.** Every training/tuning run records *what settings* it used and *how well*
it did, so you can compare runs later and reproduce the best.

**Why.** Without it you forget which of 40 experiments was the good one.

**How here.** `tracking.py` uses **MLflow** (installed) and also writes a plain
`experiments/runs.jsonl`. Browse the runs in a web UI:

```bash
python -m car_pricing.tracking     # logs a demo run
mlflow ui                          # open http://localhost:5000
```

**Upgrade:** MLflow *is* the industry standard here; a hosted MLflow or
Weights & Biases adds team sharing.

---

## 4. Explainability — *make the model show its work*

**What.** Tell *which inputs* drove a prediction. A believable price comes with a
"because…".

**Why.** Trust, debugging, and catching a model that's "right for the wrong reason."

**How here.** `explain.py` gives the model's built-in feature importances and
model-agnostic **permutation importance** (shuffle a column, see how much worse it
gets):

```bash
python -m car_pricing.explain
# -> top features: model, km_driven, make, ...
```

**Upgrade:** [`SHAP`](https://shap.readthedocs.io) explains *individual*
predictions ("this car is +₹2L because it's a diesel automatic").

---

## 5. Data / model versioning — *Save As… v1, v2, v3*

**What.** Keep old models around with their scores, so you can roll back if a new
one misbehaves, and always know exactly which model is live.

**Why.** "The new model is worse — put yesterday's back" must be one command.

**How here.** `registry.py` copies each model into `models/registry/<version>/`
with its metrics and a fingerprint, and lets you mark one as `production`:

```bash
python -m car_pricing.registry     # register the current model
```

**Upgrade:** the **MLflow Model Registry** (stages + approvals) and **DVC**
(git-like versioning for big data/model files).

---

## 6–8. CI / CD / CT — *robots that test, ship, and retrain*

These are **GitHub Actions** workflows (`.github/workflows/`). A robot wakes up on
an event and runs steps for you.

- **CI — Continuous Integration** (`ci.yml`): on every push it installs, validates
  the data, runs the tests, and checks the shipped model still passes the KPI gate.
  A red ❌ blocks broken code from merging.
- **CD — Continuous Deployment** (`cd.yml`): when you tag a release (`v1.0`), it
  builds the serving Docker image and publishes it, ready to deploy.
- **CT — Continuous Training** (`ct.yml`): on a schedule (or when `data/` changes),
  it **retrains**. Because `train.py` enforces the KPI gate, a worse model can't be
  shipped — it just fails and keeps the old one.

> **CT is the ML-specific one.** Normal apps redeploy when *code* changes; ML apps
> must also refresh when *data* changes.

---

## 9. Deployment / serving — *put the model behind a web address*

**What.** Wrap `predict()` in a tiny web server so any app can POST a car and get a
price back over HTTP.

**How here.** `serve.py` is a Flask API (`/health`, `/predict`); the `Dockerfile`
packages it with gunicorn.

```bash
python -m car_pricing.serve        # http://localhost:8000
# in another terminal:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"make":"MARUTI","model":"SWIFT VXI","age":5,"km_driven":40000}'

# or containerised:
docker build -t car-pricing-api . && docker run -p 8000:8000 car-pricing-api
```

**Upgrade:** FastAPI (async + auto docs); deploy the image to AWS ECS/Fly.io — the
sibling **[app_car_prices_flask](https://github.com/kalyancn4u/app_car_prices_flask)**
has a full AWS ECS/Fargate walkthrough.

---

## 10. Monitoring — *watch it after it ships*

**What.** Log every served prediction; once the *real* prices arrive later, measure
the live error and alert if it crosses the KPI.

**Why.** A model silently getting worse in production is the #1 real-world failure.

**How here.** `serve.py` logs each prediction via `monitoring.py`; when you attach
the true price, `live_metrics()` reports the live MAE:

```bash
python -m car_pricing.monitoring   # -> {'n_predictions': .., 'live_mae_lakhs': ..}
```

**Upgrade:** ship these logs to a dashboard (Grafana, or an Evidently/MLflow
monitoring service) with automatic alerts.

---

## 11. Drift detection — *has the world changed under the model?*

**What.** Compare *new* data to the *training* data. If cars this year are
systematically older / higher-mileage / new models, the model drifts out of date —
even though its code never changed. Drift is the trigger for CT (retraining).

**How here.** `drift.py` computes **PSI** and a **KS-test** per feature:

```bash
python -m car_pricing.drift
# -> drift_detected=True (n_drifted=1)   # after we artificially age the cars
```

**Upgrade:** [`Evidently`](https://www.evidentlyai.com) (installed) generates full
interactive drift **reports/dashboards**.

---

## Governance

Who's allowed to ship a model, how it's approved, and how you'd audit it later —
the "paper trail." Covered in **[GOVERNANCE.md](GOVERNANCE.md)**: the KPI gate as an
automatic control, the model registry as the record, and rollback/lineage.

---

## The MLOps loop, in one picture

```
   new data ─► [validate] ─► [train + tune] ─► [KPI gate] ─pass─► [register] ─► [deploy/serve]
      ▲            (1)          (2,3)             (6, gov)          (5)            (7,9)
      │                                                                             │
   [drift?] ◄───────────────── [monitor live accuracy] ◄────────── served predictions
     (11)                            (10)
```

When monitoring or drift raises a flag, **CT** (8) kicks off a retrain, and the
loop goes round again — automatically, with the KPI gate guarding every release.
