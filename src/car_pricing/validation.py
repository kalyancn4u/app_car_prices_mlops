"""Data validation — catch bad data *before* it reaches the model.

Plain idea: a bouncer for your data. Every time data comes in (for training or a
prediction), we check it looks the way we expect — right columns, sensible values,
no impossible numbers. If something's off, we say so loudly instead of quietly
training on (or predicting from) garbage.

This is dependency-free (just pandas). The popular heavier-duty tools are
**pandera** and **Great Expectations** — see docs/MLOPS_GUIDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when data fails a *hard* rule (e.g. a required column is missing)."""


# Sanity ranges — the physically-plausible bounds for each numeric column.
# (These are wider than the model's slider bounds; they only catch nonsense.)
RANGES = {
    "selling_price": (0.01, 500.0),   # Lakhs
    "age": (0, 60),
    "km_driven": (0, 1_000_000),
    "mileage": (0.0, 60.0),
    "engine": (0, 8000),
    "max_power": (0.0, 2000.0),
}


@dataclass
class ValidationReport:
    ok: bool
    n_rows: int = 0
    errors: List[str] = field(default_factory=list)      # hard failures
    warnings: List[str] = field(default_factory=list)    # soft issues

    def summary(self) -> str:
        head = f"Data validation on {self.n_rows:,} rows: {'PASS' if self.ok else 'FAIL'}"
        lines = [head]
        for e in self.errors:
            lines.append(f"  [error]   {e}")
        for w in self.warnings:
            lines.append(f"  [warning] {w}")
        if not self.errors and not self.warnings:
            lines.append("  (no issues)")
        return "\n".join(lines)


def validate_dataframe(df: pd.DataFrame, *, raise_on_error: bool = False) -> ValidationReport:
    """Check a DataFrame against the expected schema and value ranges.

    Hard rules (errors): required columns present, target strictly positive.
    Soft rules (warnings): nulls in required columns, out-of-range numbers.
    """
    errors: List[str] = []
    warnings: List[str] = []

    required = [config.TARGET] + config.TARGET_ENCODE + config.NUMERIC
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing required column(s): {missing}")

    for c in required:
        if c in df.columns:
            n_null = int(df[c].isna().sum())
            if n_null:
                warnings.append(f"'{c}' has {n_null:,} null value(s)")

    for c, (lo, hi) in RANGES.items():
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            n_bad = int(((s < lo) | (s > hi)).sum())
            if n_bad:
                warnings.append(f"'{c}' has {n_bad:,} value(s) outside [{lo}, {hi}]")

    if config.TARGET in df.columns:
        s = pd.to_numeric(df[config.TARGET], errors="coerce")
        if (s <= 0).any():
            errors.append(f"'{config.TARGET}' contains non-positive prices")

    report = ValidationReport(ok=not errors, n_rows=len(df), errors=errors, warnings=warnings)
    if raise_on_error and errors:
        raise DataValidationError("; ".join(errors))
    return report


if __name__ == "__main__":  # `python -m car_pricing.validation`
    from . import data
    rep = validate_dataframe(data.clean(data.load_raw()))
    print(rep.summary())
