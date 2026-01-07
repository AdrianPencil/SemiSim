# semisim/postprocess/structure.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from semisim.geometry.builder import Geometry1D

__all__ = ["plot_device_structure"]

def plot_device_structure(geom: Geometry1D, ND_m3: np.ndarray, NA_m3: np.ndarray):
    """
    Figure 1 style: left = colored p/n blocks + vertical cut;
                    right = 1D doping profile ND, NA along z.

    Changes:
    - Draw **white** dashed line at the p|n interface(s).
    - Draw **white** solid lines at the left and right stack edges so you
      see the “white lines on the blue and red blocks”, matching the reference fig.
    """
    z = geom.z
    y = np.linspace(0.0, 1.0, 3)
    Z, Y = np.meshgrid(z, y)

    # Color by type: p (red, -1), n (blue, +1)
    typ = np.zeros_like(z)
    typ[geom.layer_id == 0] = -1.0   # p-side
    typ[geom.layer_id == 1] = +1.0   # n-side
    TYP = np.tile(typ, (y.size, 1))

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # --- Left: block view ---
    axA.pcolormesh(Z * 1e6, Y, TYP, shading="nearest", cmap="bwr", vmin=-1, vmax=+1)
    axA.set_title("PN junction stack (Si)")
    axA.set_xlabel("z (µm)")
    axA.set_yticks([])
    axA.set_xlim(z[0] * 1e6, z[-1] * 1e6)

    # White edge outlines (left/right of the stack)
    axA.axvline(z[0] * 1e6, color="white", linewidth=2.2)
    axA.axvline(z[-1] * 1e6, color="white", linewidth=2.2)

    # White dashed interface(s)
    if geom.interfaces.size:
        for (zi, _, _) in geom.interfaces:
            axA.axvline(zi * 1e6, color="white", linestyle="--", linewidth=2.0)

    # --- Right: 1D doping cut (like their “1D cut”)
    axB.plot(z * 1e6, ND_m3 / 1e6, label=r"$N_D$ [cm$^{-3}$]")
    axB.plot(z * 1e6, NA_m3 / 1e6, label=r"$N_A$ [cm$^{-3}$]")
    axB.set_xlabel("z (µm)")
    axB.set_ylabel("Doping (cm$^{-3}$)")
    axB.set_yscale("log")
    axB.grid(True, alpha=0.3, which="both")
    axB.legend()

    fig.tight_layout()
    return fig, (axA, axB)
