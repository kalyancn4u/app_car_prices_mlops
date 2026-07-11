# 🏛️ Model Governance

**Governance = the rules and paper trail** that make a model safe and accountable
to ship: *who* approved it, *why* it was allowed out, and *how* you'd audit or roll
it back later. In plain terms — the model's "compliance file."

This repo builds governance in from small, concrete pieces (not a heavyweight
process). Here's each control and where it lives.

## 1. The KPI gate — an *automatic* approval rule

A model is allowed to ship **only if** it clears the business thresholds defined in
`car_pricing.config.KPI`:

| KPI | Threshold | Why |
| :-- | :-------- | :-- |
| MAE | ≤ ₹1.0 Lakh | Typical error must be small enough to trust |
| R² | ≥ 0.85 | Must explain most of the price variation |
| Band accuracy | ≥ 70 % | Budget bucket must usually be right |

`train.py` checks these on a held-out test set and records the verdict in
`models/metrics.json` (`kpi_gate.all_pass`). **CI re-checks it on every push**, so a
regressed model can't quietly merge. This turns "is it good enough?" from an opinion
into a **reproducible, enforced rule**.

## 2. The model registry — the record of every version

`car_pricing.registry` keeps an immutable copy of each shipped model under
`models/registry/<version>/` with its metrics, a content fingerprint (hash), and a
stage (`staging` → `production` → `archived`). This answers:

- *Which model is live right now?* → `registry.latest("production")`
- *What did the model we shipped last month score?* → its `metrics.json`
- *Roll back!* → `registry.promote("<older-version>", "production")`

## 3. Lineage — how to reproduce any model

Every model is reproducible because the inputs are all versioned in git:

- **Data:** `data/raw/*.csv.gz` (committed) + `validation.py` proves it was sane.
- **Code:** the exact `src/car_pricing/` at that commit.
- **Config:** `config.RANDOM_STATE` fixes the split, CV folds and every estimator,
  so `python -m car_pricing.train` reproduces the metrics bit-for-bit.
- **Experiment log:** `tracking.py` / MLflow records what was tried.

Given a registry entry's hash + the git commit, anyone can rebuild that exact model.

## 4. Transparency — the model card

[`MODEL_CARD.md`](MODEL_CARD.md) documents, in one place: what the model is, how it
was chosen (the bake-off), its measured performance, and — importantly — its
**limitations and ethical notes** (no condition/accident data; thin data on rare
premium cars; it's decision *support*, not a guaranteed price).

## 5. Monitoring & rollback — accountability after release

`monitoring.py` logs live predictions and computes live error; if it breaches the
KPI, that's the signal to investigate, retrain (**CT**), or **roll back** via the
registry. Nothing ships-and-is-forgotten.

## Roles (a simple RACI, for a real team)

| Activity | Who typically owns it |
| :------- | :-------------------- |
| Approve a model for production | Data science lead (reviews the KPI gate + model card) |
| Run/maintain CI/CD/CT | ML engineer |
| Watch monitoring & drift, trigger rollback | On-call / ML engineer |
| Sign off on fairness/limitations | Product + a domain reviewer |

## The one-paragraph summary

A model here can only reach production if it **passes the KPI gate** (enforced by CI),
it is **recorded in the registry** with its metrics and a fingerprint, it is
**reproducible** from committed data + code + a fixed seed, its behaviour and limits
are **documented in the model card**, and it is **monitored** in production with a
clear **rollback** path. That chain — gate → record → reproduce → document → monitor
→ roll back — *is* the governance.
