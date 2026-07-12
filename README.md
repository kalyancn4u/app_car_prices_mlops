# 🚗🔬 Car Price MLOps — End-to-End ML Lifecycle

> The **complete SDLC** for used-car price prediction — from **business KPIs** to a
> **deployable, KPI-gated pipeline** — walked phase by phase in Jupyter notebooks,
> backed by a production Python package, and documented every step of the way.

This is one of the **Car Prices Quartet**, four projects built on the same Cars24 dataset:

| Repo | What it is |
| :--- | :--------- |
| `app_car_prices_streamlit` | An interactive Streamlit app |
| `app_car_prices_flask` | A containerised Flask REST API (Docker + AWS ECS) |
| **`app_car_prices_mlops`** *(this repo)* | The **full ML lifecycle**: KPIs → data → features → model bake-off → evaluation → production pipeline |
| `app_car_prices_pipeline` | A beginner-friendly starter (guided docs + test stubs) |

Where the sibling apps *serve* the model, this repo is the **process**: how you'd
actually build the model those apps serve, done properly and reproducibly.

---

## 📑 Contents

1. [Headline results](#-headline-results)
2. [The lifecycle (notebooks)](#-the-lifecycle-notebooks)
3. [Quick start](#-quick-start)
4. [How the pipeline works](#-how-the-pipeline-works)
5. [Model comparison (DT vs RF vs XGBoost vs LightGBM)](#-model-comparison-dt-vs-rf-vs-xgboost-vs-lightgbm)
6. [Data formats: CSV vs Parquet vs Feather](#-data-formats-csv-vs-parquet-vs-feather)
7. [Project structure](#-project-structure)
8. [MLOps stages (implemented)](#-mlops-stages-implemented)
9. [How it compares to the sibling projects](#-how-it-compares-to-the-sibling-projects)
10. [Documentation index](#-documentation-index)

---

## 🎯 Headline results

Trained on 19,820 listings (41 makes, 3,233 models), held-out 20 % test set:

| Metric | Result | KPI gate |
| :----- | :----- | :------- |
| **Price R²** | **0.957** | ≥ 0.85 ✅ |
| **Price MAE** | **₹0.66 Lakhs** (≈ ₹66,000) | ≤ ₹1.0 L ✅ |
| **Band accuracy** (derived) | **85.8 %** | ≥ 70 % ✅ |
| **Shipped artifact** | **0.92 MB** | — |

Shipped model: **LightGBM** in a scikit-learn `Pipeline`. The pipeline **refuses to
ship** a model that fails any KPI gate — "good enough?" is a reproducible check, not
a judgement call.

---

## 🧭 The lifecycle (notebooks)

Every SDLC phase is a runnable, **already-executed** notebook (outputs + charts
embedded). They import the `car_pricing` package rather than duplicating logic.

| # | Notebook | Phase | You'll see |
| :- | :------- | :---- | :--------- |
| 01 | [`01_business_understanding`](notebooks/01_business_understanding.ipynb) | Business KPIs | Problem framing, stakeholders, the ship/no-ship gate |
| 02 | [`02_data_understanding_eda`](notebooks/02_data_understanding_eda.ipynb) | Data & EDA | Distributions, missingness, cardinality, price drivers |
| 03 | [`03_data_format_benchmarks`](notebooks/03_data_format_benchmarks.ipynb) | Storage | Measured CSV vs Parquet vs Feather (size/speed) |
| 04 | [`04_feature_engineering`](notebooks/04_feature_engineering.ipynb) | Features | Target encoding vs one-hot; band edges |
| 05 | [`05_model_comparison`](notebooks/05_model_comparison.ipynb) | Modelling | The DT/RF/XGB/LGBM bake-off, cross-validated |
| 06 | [`06_evaluation_and_selection`](notebooks/06_evaluation_and_selection.ipynb) | Evaluation | Held-out metrics, residuals, KPI gate, ship decision |
| 07 | [`07_productionisation`](notebooks/07_productionisation.ipynb) | Deployment | The one-Pipeline artifact + serving contract |
| 08 | [`08_xgboost_deep_dive`](notebooks/08_xgboost_deep_dive.ipynb) | Appendix | Why XGBoost couldn't ship, and the XGBoost-vs-LightGBM head-to-head |
| 09 | [`09_mlops_toolkit`](notebooks/09_mlops_toolkit.ipynb) | **MLOps** | Validation, tuning, tracking, explainability, versioning, monitoring, drift — running |

---

## ⚡ Quick start

```bash
# 1. Environment
python -m venv venv && venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
pip install -e .                                      # makes `car_pricing` importable

# 2. Reproduce the whole model pipeline (bake-off -> select -> evaluate -> save)
python -m car_pricing.train

# 3. Predict
python -c "from car_pricing.predict import predict; print(predict({'make':'MARUTI','model':'SWIFT VXI','age':5,'km_driven':40000}))"

# 4. Explore the lifecycle
jupyter lab notebooks/      # 01 -> 07

# 5. Tests
pytest -q
```

> The trained artifacts are **committed** (they're ~1 MB total), so steps 3–5 work
> immediately on a fresh clone; step 2 only re-creates them.

Example output:

```python
{'predicted_price_lakhs': 4.81, 'predicted_price_display': '₹4.81 Lakhs', 'price_band': 'Medium'}
```

---

## 🔧 How the pipeline works

```
   data/raw/*.csv
        │  clean
        ▼
   ┌─────────────────────────── one sklearn Pipeline ───────────────────────────┐
   │  ColumnTransformer                                     Regressor            │
   │   ├─ TargetEncoder(make, model)   → 2 dense cols   ┐                        │
   │   └─ passthrough(numeric + flags) → 14 cols        ├─►  LightGBM  → ₹ price │
   └────────────────────────────────────────────────────┘         │            │
                                                                   ▼            │
                                          price → tercile edges → Low/Med/High band
```

The **whole thing serialises to one 0.92 MB `.pkl`** — the fitted target-encoder
rides along with the model, so training and serving can't drift apart. Full
rationale in [`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md).

**Key production decisions:**

- **Target-encode `make`/`model`** instead of one-hot → 16 features (not ~3,200), a
  ~15× smaller model, *and* higher accuracy.
- **Derive the band from the predicted price** → the band and the ₹ figure can never
  disagree, and there's no second model to train or drift.
- **KPI-gated selection** → the pipeline ships the **best model that also deploys**
  cleanly (see the model comparison below).

---

## 📊 Model comparison (DT vs RF vs XGBoost vs LightGBM)

3-fold cross-validated MAE (Lakhs), preprocessing fit inside each fold — full table
and discussion in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and
[notebook 05](notebooks/05_model_comparison.ipynb):

| Model | CV MAE | Note |
| :---- | -----: | :--- |
| Ridge (linear) | 1.184 | underfits — price is non-linear |
| Decision Tree | 0.860 | high variance |
| Random Forest | 0.745 | strong, low-tuning |
| HistGradientBoosting | 0.732 | fast boosting |
| **XGBoost** | **0.708** | best CV — but not servable in a Pipeline here* |
| **LightGBM** | **0.715** | statistical tie → **shipped** (deploys cleanly) |

\* The installed `xgboost 2.1 × scikit-learn 1.6` can't serialise/serve XGBoost
through a portable Pipeline. Since LightGBM is a tie and serves with no version
shim, `train.py` ships it automatically — **operational robustness over a
0.007-Lakh difference**. That's a real MLOps trade-off, documented, not hidden.
Evaluated head-to-head on the test set, XGBoost is **no better** (LightGBM is even
marginally ahead on MAE) → **zero accuracy cost**. Full analysis:
[`docs/XGBOOST_SERVABILITY.md`](docs/XGBOOST_SERVABILITY.md) ·
[notebook 08](notebooks/08_xgboost_deep_dive.ipynb).

---

## 🗃️ Data formats: CSV vs Parquet vs Feather

Measured on this dataset (full table + the "why does it open in Excel?" explainer
in [`docs/FORMAT_BENCHMARKS.md`](docs/FORMAT_BENCHMARKS.md) and
[notebook 03](notebooks/03_data_format_benchmarks.ipynb)):

| Format | Size | Read | Best for |
| :----- | ---: | ---: | :------- |
| CSV (raw) | 1,535 KB | 23 ms | Humans, Excel, tiny files |
| CSV + gzip | 290 KB | 26 ms | Low-friction repo shrink (pandas reads it directly) |
| Parquet (zstd) | 227 KB | **8 ms** | Real pipelines & large data (columnar, keeps dtypes) |
| Feather | 867 KB | **5 ms** | Fastest read, temporary local hand-offs |

`data/raw/` stays a plain CSV for transparency; the pipeline can cache a Parquet
copy for fast, type-safe reloads.

---

## 📁 Project structure

```
app_car_prices_mlops/
├── README.md
├── requirements.txt · pyproject.toml · Makefile · .gitignore
├── Dockerfile · .dockerignore      # serving image (car_pricing.serve)
├── .github/workflows/              # ci.yml · cd.yml · ct.yml  (CI / CD / CT)
├── data/
│   ├── raw/cars24-car-price-cleaned-new.csv.gz # the dataset, gzip-compressed (committed)
│   └── processed/                               # generated (gitignored)
├── notebooks/                # 01–09, executed with outputs & charts (09 = MLOps toolkit)
├── src/car_pricing/          # the production package
│   ├── config.py · data.py · features.py · models.py · pipeline.py
│   ├── train.py · predict.py
│   └── validation.py · tuning.py · tracking.py · explain.py     ← MLOps toolkit
│       · registry.py · monitoring.py · drift.py · serve.py
├── models/                   # trained artifacts (committed, ~1 MB)
│   ├── price_pipeline.pkl · metrics.json · model_comparison.json · serving_metadata.json
│   └── registry/             # local model registry (gitignored)
├── docs/                     # MLOPS_GUIDE · GOVERNANCE · BUSINESS_CASE · DATA_DICTIONARY
│   │                         # MODEL_CARD · PIPELINE_DESIGN · FORMAT_BENCHMARKS · XGBOOST_SERVABILITY
├── tests/                    # test_pipeline.py + test_mlops.py  (15 tests)
└── tools/build_notebooks.py  # regenerates the notebooks deterministically
```

---

## 🧭 MLOps stages (implemented)

Building the model is ~10% of the job; the rest is operating it. Every MLOps stage
is implemented as a small, **runnable** module — each with a beginner explanation and
a one-line command in **[`docs/MLOPS_GUIDE.md`](docs/MLOPS_GUIDE.md)**.

| Stage | Module / file | Run it |
| :---- | :------------ | :----- |
| Data validation | `car_pricing.validation` | `python -m car_pricing.validation` |
| Hyperparameter tuning | `car_pricing.tuning` | `python -m car_pricing.tuning` |
| Experiment tracking (MLflow) | `car_pricing.tracking` | `python -m car_pricing.tracking` → `mlflow ui` |
| Explainability | `car_pricing.explain` | `python -m car_pricing.explain` |
| Model versioning / registry | `car_pricing.registry` | `python -m car_pricing.registry` |
| CI / CD / CT | `.github/workflows/{ci,cd,ct}.yml` | on push / tag / schedule |
| Deployment / serving | `car_pricing.serve` + `Dockerfile` | `python -m car_pricing.serve` |
| Monitoring | `car_pricing.monitoring` | `python -m car_pricing.monitoring` |
| Drift detection | `car_pricing.drift` | `python -m car_pricing.drift` |
| Governance | [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) | KPI gate + registry + model card |

See it all run in **[notebook 09](notebooks/09_mlops_toolkit.ipynb)**.

---

## 🔗 How it compares to the sibling projects

Same data, a deliberately different (production) approach:

| | This (MLOps) | Streamlit / Flask siblings |
| :-- | :-- | :-- |
| Encoding | Target encoding (2 cols) | One-hot make+model (~3,200 cols) |
| Model artifact | **0.92 MB** | 8.9–13.5 MB |
| Price R² | **0.957** | 0.950 |
| Band | **Derived** from price (85.8 %) | Separate classifier (76.9 %) |
| Focus | The **lifecycle** & pipeline | Serving the model to users |

The siblings answer *"how do users interact with the model?"*; this repo answers
*"how do you build, compare, and ship the model responsibly?"*

---

## 📚 Documentation index

- 🧭 [`docs/MLOPS_GUIDE.md`](docs/MLOPS_GUIDE.md) — **all 11 MLOps stages, explained for a beginner**
- 🏛️ [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) — approval (KPI gate), registry, lineage, rollback
- 🧪 [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md) — learn testing & debugging by completing the graded stubs in [`tests/test_stubs.py`](tests/test_stubs.py)
- 💼 [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) — problem, value, KPIs, scope
- 📖 [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — every column, the baselines, cleaning
- 🃏 [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — the bake-off, selection, performance, limits
- 🧩 [`docs/XGBOOST_SERVABILITY.md`](docs/XGBOOST_SERVABILITY.md) — why XGBoost didn't ship, and the head-to-head
- 🏗️ [`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md) — package layout, leakage control, MLOps loop
- 🗃️ [`docs/FORMAT_BENCHMARKS.md`](docs/FORMAT_BENCHMARKS.md) — CSV vs Parquet vs Feather, the Excel trap

---

> ⚠️ **Disclaimer:** predictions are statistical estimates from historical Cars24
> listings — decision support, not a guaranteed sale price.

<sub>Built with scikit-learn · LightGBM · XGBoost · pandas · pyarrow. Part of the
`*_car_prices` quartet.</sub>

---

### 🔗 The Car Prices Quartet

Four sibling projects built on the same Cars24 dataset:

- 🎛️ **[Streamlit web app →](https://github.com/kalyancn4u/app_car_prices_streamlit)** — interactive price-predictor UI
- 🐳 **[Flask REST API →](https://github.com/kalyancn4u/app_car_prices_flask)** — containerised API (Docker + AWS ECS/Fargate)
- 🔬 **MLOps lifecycle** — full SDLC: notebooks → production pipeline · _you are here_
- 🛠️ **[Pipeline starter →](https://github.com/kalyancn4u/app_car_prices_pipeline)** — beginner-friendly guide + test stubs to extend
