# 📗 API Examples — the `car_pricing` package by example

Copy-paste recipes for every part of the package. Each shows the **import**, the
**call**, and the **expected output**. Run the notebook version in
[`notebooks/api_examples.ipynb`](../notebooks/api_examples.ipynb).

Setup once:

```bash
pip install -r requirements.txt && pip install -e .
```

---

## Serving — `predict()`  (the thing you'll use most)

```python
from car_pricing.predict import predict

predict({"make": "MARUTI", "model": "SWIFT VXI", "age": 5, "km_driven": 40000})
# -> {'predicted_price_lakhs': 5.74,
#     'predicted_price_display': '₹5.74 Lakhs',
#     'price_band': 'Medium'}
```

Only `make` + `model` are required; everything else is auto-filled from that model's
typical values:

```python
predict({"make": "BMW", "model": "X5"})            # -> ~₹18–20 Lakhs, band 'High'
```

## Data — load, clean, validate

```python
from car_pricing import data, validation

df = data.clean(data.load_raw())                   # 19,820 rows, cleaned
print(df.shape)                                    # (19820, 18)

report = validation.validate_dataframe(df)
print(report.summary())                            # "Data validation on 19,820 rows: PASS"
```

## Features — encoding & price bands

```python
from car_pricing import features

X, y = features.split_xy(df)                        # X: 16 feature columns
edges = features.band_edges(y)                      # [0.3, 3.99, 6.75, 20.9]
features.price_to_band([2.0, 5.0, 25.0], edges)     # array(['Low','Medium','High'])
```

## Modelling — the zoo, the pipeline, training

```python
from car_pricing import models
from car_pricing.pipeline import make_pipeline

list(models.model_zoo())          # ['Ridge (linear)', 'Decision Tree', ... 'LightGBM']
pipe = make_pipeline(models.model_zoo()["LightGBM"])   # preprocessor + model

# Full training run (CV bake-off -> pick servable winner -> KPI gate -> save):
from car_pricing.train import main as train
result = train()                  # writes models/*.json + price_pipeline.pkl
result["metrics"]["price_mae_lakhs"]   # ~0.66
```

## Hyperparameter tuning

```python
from car_pricing import tuning
res = tuning.tune("LightGBM", n_iter=10)
print(res["best_cv_mae_lakhs"], res["best_params"])
# -> 0.69x  {'learning_rate': ..., 'n_estimators': ..., 'num_leaves': ...}
```

## Experiment tracking

```python
from car_pricing import tracking
tracking.log_run("my-run", params={"lr": 0.05}, metrics={"mae": 0.71})
tracking.load_runs().tail()        # a DataFrame of past runs
# then browse the MLflow UI:  mlflow ui   (http://localhost:5000)
```

## Explainability

```python
from car_pricing import explain
explain.builtin_importances().head(5)         # top features by model importance
explain.permutation_importances(n_repeats=5)  # model-agnostic (shuffle-and-measure)
```

## Model registry (versioning + rollback)

```python
from car_pricing import registry
entry = registry.register(note="v1 for prod")        # copies model -> models/registry/
registry.promote(entry["version"], "production")
registry.latest("production")["version"]             # the live version
```

## Monitoring

```python
from car_pricing import monitoring
monitoring.log_prediction({"make": "MARUTI", "model": "SWIFT VXI"}, 5.74, actual_lakhs=5.5)
monitoring.live_metrics()
# -> {'n_predictions': 1, 'n_with_actual': 1, 'live_mae_lakhs': 0.24, 'kpi_breached': False}
```

## Drift detection

```python
from car_pricing import drift
ref = df.sample(frac=0.5, random_state=1)
cur = df.sample(frac=0.5, random_state=2).copy(); cur["age"] += 8   # "older cars this year"
out = drift.detect_drift(ref, cur)
out["drift_detected"], out["n_drifted"]     # (True, 1)
out["report"]                               # per-feature PSI / KS table
```

## Serving over HTTP

```bash
python -m car_pricing.serve      # http://localhost:8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"make":"MARUTI","model":"SWIFT VXI","age":5,"km_driven":40000}'
```

Or in a test (no server needed):

```python
from car_pricing.serve import app
client = app.test_client()
client.post("/predict", json={"make": "MARUTI", "model": "SWIFT VXI"}).get_json()
```

> Deep dives: [`MLOPS_GUIDE.md`](MLOPS_GUIDE.md) (every stage explained) ·
> [`PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md) (how the code is organised).
