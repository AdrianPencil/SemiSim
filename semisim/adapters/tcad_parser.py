# -*- coding: utf-8 -*-
"""
Parse TCAD outputs into a common schema.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def parse_outputs(out_dir: Path) -> dict:
    d = {}
    iv = out_dir / "IV.csv"
    cv = out_dir / "CV.csv"
    if iv.exists():
        d["IV"] = pd.read_csv(iv)
    if cv.exists():
        d["CV"] = pd.read_csv(cv)
    return d
