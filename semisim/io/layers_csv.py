# semisim/io/layers_csv.py
# -*- coding: utf-8 -*-
"""
Layer stack CSV ingest → StackSpec/LayerSpec for geometry builder.

CSV columns (header, case-sensitive):
  name, role, material, thickness_nm, T_K,
  system (optional), comp_mode (optional: constant|linear), x0 (optional), x1 (optional)

Example rows:
  metal,metal,Al,5,300,,,
  AlGaN,semiconductor,AlN,20,300,AlGaN,constant,0.25,
  GaN,semiconductor,GaN,500,300,,,
  buffer,semiconductor,GaN,2000,300,,,

Units:
  thickness [nm] → [m]
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from semisim.geometry.builder import (
    LayerSpec, CompositionSpec, StackSpec
)

@dataclass
class CSVStack:
    stack: StackSpec

def _parse_float_or_none(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    return float(s)

def load_stack_from_csv(csv_path: Path) -> CSVStack:
    layers: list[LayerSpec] = []
    T_seen: float | None = None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            role = row["role"].strip()
            material = row["material"].strip()
            thickness_m = float(row["thickness_nm"]) * 1e-9
            T_K = float(row.get("T_K", 300.0) or 300.0)
            T_seen = T_K if T_seen is None else T_seen  # first row decides, unless you prefer per-layer T

            system = row.get("system") or None
            comp_mode = (row.get("comp_mode") or "").strip().lower()
            x0 = _parse_float_or_none(row.get("x0"))
            x1 = _parse_float_or_none(row.get("x1"))

            comp = None
            if system:
                if comp_mode not in ("constant", "linear"):
                    comp_mode = "constant"
                if comp_mode == "constant":
                    comp = CompositionSpec(mode="constant", x0=float(x0 or 0.0))
                else:
                    comp = CompositionSpec(mode="linear", x0=float(x0 or 0.0), x1=float(x1 or x0 or 0.0))

            layers.append(
                LayerSpec(
                    name=name,
                    role=role,                 # "semiconductor" | "oxide" | "metal" | "void"
                    thickness=thickness_m,
                    material=material,
                    system=system,
                    comp=comp,
                )
            )

    if not layers:
        raise ValueError("CSV contains no layers")

    return CSVStack(stack=StackSpec(layers=tuple(layers), T=float(T_seen or 300.0)))
