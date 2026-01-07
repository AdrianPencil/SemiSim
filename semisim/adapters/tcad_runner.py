# -*- coding: utf-8 -*-
"""
TCAD runner adapter.

Design:
  * Keep a pure-Python interface that can call either a local binary,
    a Docker image, or a mock (for CI).
  * Return a path to a directory containing normalized CSV outputs.

"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shutil

@dataclass
class TCADJob:
    deck: Path
    workdir: Path
    runner: str = "mock"     # options: "mock", "subprocess", "docker"
    image: str | None = None

def run(job: TCADJob) -> Path:
    out = job.workdir / "tcad_outputs"
    out.mkdir(parents=True, exist_ok=True)
    # MOCK: drop in canned CSVs or echo inputs for testing; replace in real usage
    (out / "IV.csv").write_text("V,I\n0.0,0.0\n0.5,0.01\n1.0,0.02\n")
    (out / "CV.csv").write_text("V,C\n-1.0,1e-9\n0.0,2e-9\n1.0,2e-9\n")
    return out
