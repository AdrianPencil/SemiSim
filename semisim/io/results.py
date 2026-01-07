# -*- coding: utf-8 -*-
"""
Results IO helpers.

Write:
  * metrics.json  (device-level KPIs)
  * fields.h5     (sparse fields, optional)
  * artifacts/    (plots, CSVs)

This keeps on-disk layout stable for post-processing and reports.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def write_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

def save_fields_npz(run_dir: Path, **arrays) -> Path:
    """
    Save arrays for viz (e.g., x, Ec, Ev, Efn, Efp, phi, n, p, Efield, T, Vg_sweep, Ig, Qg, C).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "fields.npz"
    np.savez_compressed(out, **arrays)
    return out
