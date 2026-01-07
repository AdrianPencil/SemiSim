# -*- coding: utf-8 -*-
"""
Load experimental datasets (C–V / I–V) with metadata for validation.

CSV schema examples:
  CV: Vg [V], C [F], freq [Hz], T [K], sample_id
  IV: V [V], I [A], T [K], sample_id
"""
from pathlib import Path
import pandas as pd

def load_cv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert {"Vg","C"}.issubset(df.columns), "CV CSV must have Vg,C columns"
    return df

def load_iv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert {"V","I"}.issubset(df.columns), "IV CSV must have V,I columns"
    return df
