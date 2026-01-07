# semisim/io/config.py
# -*- coding: utf-8 -*-
"""
YAML → StackSpec and MeshSpec helpers.

Schema (minimal, example):

geometry:
  T_K: 300
  layers:
    - { name: metal, role: metal, material: Al, thickness_nm: 5 }
    - { name: AlGaN, role: semiconductor, material: AlN, thickness_nm: 20, system: AlGaN, comp_mode: constant, x0: 0.25 }
    - { name: GaN, role: semiconductor, material: GaN, thickness_nm: 500 }
    - { name: buffer, role: semiconductor, material: GaN, thickness_nm: 2000 }

mesh:
  per_layer_N: [10, 25, 50, 50]
  refine_interfaces: true
  stretch_ratio: 1.0
  stretch_cells: 0
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from semisim.geometry.builder import (
    LayerSpec, CompositionSpec, StackSpec, MeshSpec
)

@dataclass
class RunConfig:
    raw: dict
    path: Path

def load_config(path: Path) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping")
    _validate_minimum(data)
    return RunConfig(raw=data, path=Path(path))

def build_stack(cfg: RunConfig) -> StackSpec:
    g = cfg.raw["geometry"]
    T = float(g.get("T_K", 300.0))
    layers_yaml = g.get("layers", [])
    if not layers_yaml:
        raise ValueError("geometry.layers is empty")

    layers: list[LayerSpec] = []
    for row in layers_yaml:
        name = str(row["name"])
        role = str(row["role"])
        material = str(row["material"])
        thickness_m = float(row["thickness_nm"]) * 1e-9

        system = row.get("system")
        comp_mode = (row.get("comp_mode") or "").lower()
        x0 = row.get("x0", None)
        x1 = row.get("x1", None)

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
                name=name, role=role,
                thickness=thickness_m,
                material=material,
                system=system,
                comp=comp,
            )
        )
    return StackSpec(layers=tuple(layers), T=T)

def build_mesh(cfg: RunConfig) -> MeshSpec:
    m = cfg.raw["mesh"]
    per_layer_N = m.get("per_layer_N")
    hz_max = m.get("hz_max")
    N_total = m.get("N_total")
    refine = bool(m.get("refine_interfaces", True))
    r = float(m.get("stretch_ratio", 1.0))
    cells = int(m.get("stretch_cells", 0))

    return MeshSpec(
        N_total=int(N_total) if N_total is not None else None,
        hz_max=float(hz_max) if hz_max is not None else None,
        per_layer_N=list(map(int, per_layer_N)) if per_layer_N is not None else None,
        refine_interfaces=refine,
        stretch_ratio=r,
        stretch_cells=cells,
    )

def _validate_minimum(cfg: dict) -> None:
    for key in ("geometry", "mesh"):
        if key not in cfg:
            raise ValueError(f"Missing top-level key: {key}")
