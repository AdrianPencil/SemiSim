# semisim/physics/transport.py
"""
Low-field transport utilities: mobility kernel (Matthiessen) and diffusion.

- Channelized mobility model driven by a JSON/YAML-like config.
- Each channel returns an *inverse mobility* contribution μ_i^{-1}(z) (Coulomb/phonon/etc.).
- Combined via Matthiessen: μ^{-1} = sum_i μ_i^{-1}; per-carrier-type outputs μ_n, μ_p.
- Degenerate Einstein relation:
      D = μ * (k_B T / q) * (F_{-1/2}(η) / F_{1/2}(η)).

This file uses *placeholder scaling* (non-credible constants) but preserves
the *correct functional dependencies* and units plumbing, so you can tune
the K_* coefficients to data later.

Public API:
    MobilityChannel
    MobilityKernel
    TransportInputs
    mobility_from_kernel(inputs, kernel) -> (mu_n, mu_p)
    diffusion_from_mobility(mu, T, eta) -> D
"""
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .carriers.statistics import fermi_dirac_F12, fermi_dirac_Fm12

# ---- constants (SI) ----
Q = 1.602176634e-19     # C
K_B = 1.380649e-23      # J/K
EPS0 = 8.8541878128e-12 # F/m
PI = np.pi

__all__ = [
    "MobilityChannel",
    "MobilityKernel",
    "TransportInputs",
    "mobility_from_kernel",
    "diffusion_from_mobility",
]


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass(slots=True)
class MobilityChannel:
    """
    One mobility channel; 'name' dispatches to a kernel function below.

    params: free-form dict with SI units; include a *tunable* 'K' coefficient
            for magnitude calibration (dimension chosen by each kernel).
    enabled: if False, the channel is skipped.
    carrier: "n", "p", or "both" — which carrier(s) this channel affects.
    """
    name: str
    params: Dict[str, float]
    enabled: bool = True
    carrier: Literal["n", "p", "both"] = "both"


@dataclass(slots=True)
class MobilityKernel:
    """
    Container for multiple channels and combination rule (Matthiessen).
    """
    channels: List[MobilityChannel]
    combine_rule: Literal["matthiessen"] = "matthiessen"


@dataclass(slots=True)
class TransportInputs:
    """
    Inputs required to evaluate mobilities on a 1D mesh.

    Attributes
    ----------
    T_K : float
        Temperature [K].
    eps_r : np.ndarray
        Relative permittivity per node (-).
    n_m3, p_m3 : np.ndarray
        Carrier densities per node [1/m^3].
    Nd_m3, Na_m3 : np.ndarray
        Ionized dopants [1/m^3] (if neutral/partial ionization is needed, provide the ionized portion).
    x_alloy : np.ndarray
        Alloy mole fraction x at nodes (NaN if not alloyed).
    z_m : np.ndarray
        Node coordinates [m] (used by some kernels).
    """
    T_K: float
    eps_r: np.ndarray
    n_m3: np.ndarray
    p_m3: np.ndarray
    Nd_m3: np.ndarray
    Na_m3: np.ndarray
    x_alloy: np.ndarray
    z_m: np.ndarray


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def _safe(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _nz(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.where(x == 0.0, eps, x)


def diffusion_from_mobility(mu: np.ndarray, T_K: float | np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    Degenerate Einstein relation:
        D = μ * (k_B T / q) * (F_{-1/2}(η) / F_{1/2}(η)).
    For MB (nondegenerate), you may pass eta << 0, which makes the ratio ~ 1.
    """
    T = np.asarray(T_K, dtype=np.float64)
    eta = _safe(eta)
    factor = fermi_dirac_Fm12(eta) / _nz(fermi_dirac_F12(eta))
    return _safe(mu) * (K_B * T / Q) * factor


# ---------------------------------------------------------------------
# Channel kernels (placeholder scalings; tune 'K' later)
# ---------------------------------------------------------------------


def _mu_inv_background_impurities(inp: TransportInputs, params: Dict[str, float]) -> np.ndarray:
    """
    Ionized impurity scattering (Brooks–Herring-inspired trend, simplified):
        μ^{-1} ~ K * N_imp / (eps_r^2 * T^{3/2}).
    params:
        K              : calibration [m^3 V s / ???] (tunable)
        Nimp_m3        : [1/m^3] (if omitted, use Nd+Na)
    """
    K = float(params.get("K", 1.0))
    Nimp = _safe(params.get("Nimp_m3", 0.0)) + inp.Nd_m3 + inp.Na_m3
    T = inp.T_K
    return K * _safe(Nimp) / (_nz(inp.eps_r) ** 2 * (T ** 1.5))


def _mu_inv_dislocation_core(inp: TransportInputs, params: Dict[str, float]) -> np.ndarray:
    """
    Charged dislocations (line charge): trend ~ proportional to line density.
        μ^{-1} ~ K * (N_dis * f_occ)
    params:
        K              : calibration [m^2 V s] (tunable)
        Ndis_m2        : threading dislocation density [1/m^2]
        f_occ          : charged-core occupancy 0..1
    """
    K = float(params.get("K", 1.0))
    Ndis = float(params.get("Ndis_m2", 0.0))
    focc = float(params.get("f_occ", 1.0))
    return K * (Ndis * focc) * np.ones_like(inp.n_m3)


def _mu_inv_remote_interface(inp: TransportInputs, params: Dict[str, float]) -> np.ndarray:
    """
    Remote/interface charge scattering: trend increases with |sigma| and decreases with spacing z0, eps_r.
        μ^{-1} ~ K * |sigma| / (eps_r^2 * max(z0, z_min))
    params:
        K              : calibration [m V s / C]
        sigma_Cm2      : fixed sheet magnitude [C/m^2] (effective)
        z0_m           : separation [m] (oxide or barrier thickness)
        z_min_m        : floor distance to avoid blow-ups (default 1e-10 m)
    """
    K = float(params.get("K", 1.0))
    sigma = abs(float(params.get("sigma_Cm2", 0.0)))
    z0 = float(params.get("z0_m", 1e-9))
    zmin = float(params.get("z_min_m", 1e-10))
    return K * sigma / (_nz(inp.eps_r) ** 2 * max(z0, zmin)) * np.ones_like(inp.n_m3)


def _mu_inv_acoustic_phonon(inp: TransportInputs, params: Dict[str, float]) -> np.ndarray:
    """
    Acoustic phonon (deformation potential) rough trend:
        μ^{-1} ~ K * T   (placeholder; real law often ~ T^(3/2)/ρ v_s^2 ...)
    params:
        K : calibration [V s / m^2 K] (tunable)
    """
    K = float(params.get("K", 1.0))
    return K * inp.T_K * np.ones_like(inp.n_m3)


def _mu_inv_polar_optical_phonon(inp: TransportInputs, params: Dict[str, float]) -> np.ndarray:
    """
    Polar optical phonon trend:
        μ^{-1} ~ K * (1 + 2 n_LO),   n_LO = 1/(exp(ħω_LO / kT) - 1)
    params:
        K               : calibration [V s / m^2]
        hwLO_J          : LO phonon energy [J]
    """
    K = float(params.get("K", 1.0))
    hw = float(params.get("hwLO_J", 0.09 * Q))
    nLO = 1.0 / (np.exp(hw / (K_B * inp.T_K)) - 1.0)
    return K * (1.0 + 2.0 * nLO) * np.ones_like(inp.n_m3)


def _mu_inv_alloy_disorder(inp: TransportInputs, params: Dict[str, float]) -> np.ndarray:
    """
    Alloy disorder trend:
        μ^{-1} ~ K * x (1 - x)
    params:
        K : calibration [V s / m^2]
    """
    K = float(params.get("K", 1.0))
    x = _safe(inp.x_alloy)
    x = np.where(np.isnan(x), 0.0, x)
    return K * (x * (1.0 - x))


def _mu_inv_interface_roughness(inp: TransportInputs, params: Dict[str, float]) -> np.ndarray:
    """
    Interface roughness trend:
        μ^{-1} ~ K * (Delta_rms^2 / L_corr)
    params:
        K            : calibration [V s / m^2 / Å]
        Delta_rms_m  : RMS height [m]
        L_corr_m     : correlation length [m]
    """
    K = float(params.get("K", 1.0))
    Drms = float(params.get("Delta_rms_m", 2.5e-10))
    Lc = float(params.get("L_corr_m", 2.0e-9))
    return K * (Drms * Drms / _nz(Lc)) * np.ones_like(inp.n_m3)


def _mu_inv_dipole_scattering(inp: TransportInputs, params: Dict[str, float]) -> np.ndarray:
    """
    Polarization–alloy dipole (nitride-specific placeholder):
        μ^{-1} ~ K * (|dx/dz|)  (strength grows with composition gradient)
    params:
        K : calibration [V s / m]
    """
    K = float(params.get("K", 1.0))
    x = _safe(inp.x_alloy)
    x = np.where(np.isnan(x), 0.0, x)
    # crude derivative |dx/dz|
    z = _safe(inp.z_m)
    dx = np.empty_like(x)
    dx[0] = abs((x[1] - x[0]) / (z[1] - z[0]))
    dx[-1] = abs((x[-1] - x[-2]) / (z[-1] - z[-2]))
    dx[1:-1] = np.abs((x[2:] - x[:-2]) / (z[2:] - z[:-2]))
    return K * dx


# Channel dispatch table
_CHANNELS = {
    "background_ionized_impurities": _mu_inv_background_impurities,
    "dislocation_core_charges": _mu_inv_dislocation_core,
    "remote_interface_charges": _mu_inv_remote_interface,
    "acoustic_deformation": _mu_inv_acoustic_phonon,
    "polar_optical_phonon": _mu_inv_polar_optical_phonon,
    "alloy_disorder": _mu_inv_alloy_disorder,
    "interface_roughness": _mu_inv_interface_roughness,
    "dipole_scattering": _mu_inv_dipole_scattering,
}


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def _sum_mu_inverse(
    inp: TransportInputs, kernel: MobilityKernel, carrier: Literal["n", "p"]
) -> np.ndarray:
    """
    Sum μ^{-1} contributions across enabled channels for a given carrier.
    """
    out = np.zeros_like(inp.n_m3, dtype=np.float64)
    for ch in kernel.channels:
        if not ch.enabled:
            continue
        if ch.carrier not in (carrier, "both"):
            continue
        fn = _CHANNELS.get(ch.name)
        if fn is None:
            continue
        out += fn(inp, ch.params)
    return out


def mobility_from_kernel(
    inp: TransportInputs, kernel: MobilityKernel
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute low-field mobilities μ_n, μ_p [m^2/V/s] via Matthiessen's rule.

    Returns
    -------
    mu_n, mu_p : arrays shaped like inp.n_m3
    """
    if kernel.combine_rule != "matthiessen":
        raise ValueError(f"unsupported combine_rule: {kernel.combine_rule!r}")

    mu_inv_n = _sum_mu_inverse(inp, kernel, "n")
    mu_inv_p = _sum_mu_inverse(inp, kernel, "p")

    # Guard against zero (no channels): produce a large mobility
    # You can inject a 'floor' channel if desired.
    mu_n = 1.0 / _nz(mu_inv_n)
    mu_p = 1.0 / _nz(mu_inv_p)
    return mu_n, mu_p

# ---------------------------------------------------------------------
# Config → Kernel builder (lightweight)
# ---------------------------------------------------------------------
def build_kernel_from_config(cfg: Dict) -> MobilityKernel:
    """
    Build a MobilityKernel from a JSON/YAML-like dictionary.

    Expected shape:
      {
        "combine": "matthiessen",
        "channels": [
          { "name": "acoustic_deformation", "params": {"K": 1.0}, "enabled": true, "carrier": "both" },
          { "name": "background_ionized_impurities", "params": {"K": 1e-25}, "enabled": true, "carrier": "both" },
          ...
        ]
      }
    """
    combine = str(cfg.get("combine", "matthiessen")).lower()
    channels_cfg = cfg.get("channels", []) or []
    channels: List[MobilityChannel] = []
    for ch in channels_cfg:
        if not isinstance(ch, dict):
            continue
        name = str(ch.get("name", ""))
        params = dict(ch.get("params", {}))
        enabled = bool(ch.get("enabled", True))
        carrier = ch.get("carrier", "both")
        if carrier not in ("n", "p", "both"):
            carrier = "both"
        channels.append(MobilityChannel(name=name, params=params, enabled=enabled, carrier=carrier))  # type: ignore[arg-type]
    return MobilityKernel(channels=channels, combine_rule=combine)  # type: ignore[arg-type]

