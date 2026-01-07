from pathlib import Path
from semisim.io.config import load_config
from semisim.io.results import write_metrics, save_fields_npz

def run_from_config(cfg_path: Path):
    cfg = load_config(cfg_path).raw
    out_dir = Path("runs") / cfg_path.stem

    # ... solve physics here ...
    # placeholder demo arrays:
    x = np.linspace(0, 1e-7, 400)
    Ec = 1.0e-19 + 1.0e-20*np.sin(2*np.pi*x/x[-1])
    Ev = Ec - 1.12*1.602e-19
    Ef0 = 0.0
    V = np.linspace(0, 1, 101); I = 1e-3*V
    Vg = np.linspace(-1, 1, 201); C = 1e-9*(Vg>0)

    save_fields_npz(out_dir, x=x, Ec=Ec, Ev=Ev, Ef0=Ef0, V=V, I=I, Vg=Vg, C=C,
                    Efield=np.gradient(-Ec, x), T=300+10*(x/x[-1]))
    write_metrics(out_dir, {"Ron_ohm": float(1/np.polyfit(V, I, 1)[0])})
    print(f"[run] saved results for viz at {out_dir}")


# -*- coding: utf-8 -*-
"""
Single-run workflow wiring config → physics → solver → metrics.
"""
from __future__ import annotations
from pathlib import Path
from semisim.io.config import load_config
from semisim.io.results import write_metrics

def run_from_config(cfg_path: Path):
    cfg = load_config(cfg_path).raw
    # TODO: build mesh, materials, layers, assemble, solve...
    metrics = {"Ron_ohm": 42.0, "Rth_K_per_W": 0.123}  # placeholder
    out_dir = Path("runs") / cfg_path.stem
    write_metrics(out_dir, metrics)
    print(f"[run] wrote metrics to: {out_dir/'metrics.json'}")
