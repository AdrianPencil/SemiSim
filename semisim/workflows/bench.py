# -*- coding: utf-8 -*-
"""
Benchmark automation: run our sim + TCAD, compare metrics, write a report.
"""
from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from semisim.workflows.run_dc import run_from_config
from semisim.adapters.tcad_runner import TCADJob, run as run_tcad
from semisim.adapters.tcad_parser import parse_outputs
from semisim.io.results import write_metrics

def bench_case(case_yaml: Path):
    spec = yaml.safe_load(open(case_yaml))
    # 1) run our solver
    run_from_config(Path(spec["our_config"]))
    # 2) run TCAD
    workdir = Path("runs") / Path(spec["device"]).stem / "tcad"
    out = run_tcad(TCADJob(deck=Path(spec["tcad"]["deck"]), workdir=workdir))
    parsed = parse_outputs(out)

    # 3) compare IV slope as an example metric
    ours = pd.read_json(Path("runs")/Path(spec["our_config"]).stem/"metrics.json")
    tc_iv = parsed["IV"]
    ron_tcad = 1.0 / np.polyfit(tc_iv["V"], tc_iv["I"], 1)[0]
    comparison = {"Ron_ours": float(ours["Ron_ohm"]), "Ron_tcad": float(ron_tcad)}
    write_metrics(workdir, comparison)
    print(f"[bench] comparison saved at: {workdir/'metrics.json'}")
