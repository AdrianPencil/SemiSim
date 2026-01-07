# semisim/postprocess/visualization.py
"""
Lightweight plotting helpers for all 1-D device output types.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---- constants (SI) ----
Q = 1.602176634e-19  # C

__all__ = ["plot_band_diagram"]


def _c64(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _to_eV(E_J: np.ndarray | float) -> np.ndarray:
    return _c64(E_J) / Q


def _to_um(z_m: np.ndarray | float) -> np.ndarray:
    return _c64(z_m) * 1e6


def plot_band_diagram(
    z_m: Iterable[float],
    E_C_J: Iterable[float],
    E_V_J: Iterable[float],
    E_F_J: float | Iterable[float],
    *,
    ax: plt.Axes | None = None,
    title: str | None = "PN Junction Band Diagram",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot EC, EV, and EF across z with axes in µm/eV.

    Parameters
    ----------
    z_m : iterable of float
        Node coordinates [m].
    E_C_J, E_V_J : iterable of float
        Conduction/valence band edges [J] at nodes.
    E_F_J : float or array-like
        Fermi level [J]. Scalar is drawn as a flat line.
    ax : matplotlib Axes, optional
        If provided, plot into this axes; otherwise create a new figure.
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    z_um = _to_um(z_m)
    EC_eV = _to_eV(E_C_J)
    EV_eV = _to_eV(E_V_J)

    # --- Make EF shape-safe (broadcast to EC/EV length if needed) ---
    EF_raw = np.asarray(E_F_J)
    if EF_raw.ndim == 0:                      # scalar
        EF_eV = np.full_like(EC_eV, float(EF_raw) / Q, dtype=np.float64)
    else:
        EF_raw = EF_raw.astype(np.float64, copy=False) / Q
        if EF_raw.size == 1:                  # length-1 array/list
            EF_eV = np.full_like(EC_eV, float(EF_raw.ravel()[0]), dtype=np.float64)
        elif EF_raw.shape != EC_eV.shape:     # try to coerce
            EF_eV = np.resize(EF_raw, EC_eV.shape)
        else:
            EF_eV = EF_raw

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    else:
        fig = ax.figure

    ax.plot(z_um, EC_eV, label=r"$E_C$", linewidth=1.8)
    ax.plot(z_um, EV_eV, label=r"$E_V$", linewidth=1.8)
    ax.plot(z_um, EF_eV, label=r"$E_F$", linestyle="--", linewidth=1.4)

    ax.set_xlabel("z (µm)")
    ax.set_ylabel("Energy (eV)")
    if title:
        ax.set_title(title)

    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="best")

    ymin = float(min(EC_eV.min(), EV_eV.min(), EF_eV.min()))
    ymax = float(max(EC_eV.max(), EV_eV.max(), EF_eV.max()))
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    if z_um[0] > z_um[-1]:
        ax.invert_xaxis()

    return fig, ax

