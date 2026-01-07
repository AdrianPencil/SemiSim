# -*- coding: utf-8 -*-
"""
Parameter sweep / DOE.

We reuse config overrides to build grids quickly.
"""
from __future__ import annotations
from pathlib import Path
from itertools import product
from semisim.io.config import load_config, apply_overrides

def run_sweep(cfg_path: Path, overrides: list[str]):
    base = load_config(cfg_path)
    print(f"[sweep] base={cfg_path}, overrides={overrides}")
    # Example: no real sweep — just demonstrates override machinery.
    apply_overrides(base, overrides)
    print("[sweep] (placeholder) run variants…")
