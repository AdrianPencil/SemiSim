# -*- coding: utf-8 -*-
"""
Compare simulated C–V against experimental data and summarize errors.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from semisim.io.experimental_csv import load_cv
from semisim.io.results import write_metrics

def validate_cv(sim_V: np.ndarray, sim_C: np.ndarray, exp_csv: Path, out_dir: Path):
    exp = load_cv(exp_csv).sort_values("Vg")
    # Interpolate sim → exp Vg grid
    C_sim_on_exp = np.interp(exp["Vg"].values, sim_V, sim_C)
    # Metrics
    rmse = float(np.sqrt(np.mean((C_sim_on_exp - exp["C"].values)**2)))
    mae  = float(np.mean(np.abs(C_sim_on_exp - exp["C"].values)))
    write_metrics(out_dir, {"CV_RMSE": rmse, "CV_MAE": mae})
    return rmse, mae
