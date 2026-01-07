# semisim/materials/database.py
"""
Materials database (GaN/AlN/AlGaN, extensible).

- SI units throughout.
- Provides T-dependent properties via lightweight models (Varshni, linear mixes).
- Composition mixing for binaries/ternaries with optional bowing.
- No 'credible' values here; placeholders only.
- Designed to be backend-agnostic and cache-friendly.

Public API (stable):
    get_material(name: str, T: float = 300.0) -> MaterialPropsT
    get_alloy(system: str, x: float, T: float = 300.0) -> MaterialPropsT
    list_materials() -> list[str]
    list_systems() -> list[str]

Notes:
- 'system' examples: "AlGaN" with mole fraction x ≡ Al content in Al_x Ga_{1-x} N.
- Band offsets are NOT handled here; see carriers/bands.py for alignment.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Literal, Optional, Tuple
import numpy as np


__all__ = [
    "MaterialBase",
    "MaterialPropsT",
    "AlloyRule",
    "get_material",
    "get_alloy",
    "list_materials",
    "list_systems",
]

# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass(slots=True)
class MaterialBase:
    """
    T=0 / reference parameters + T-laws (model coefficients).

    Placeholders only; numbers are not intended to be accurate.

    Attributes
    ----------
    name : str
        Canonical material key, e.g., "GaN".
    lattice : Literal["wurtzite", "zincblende", "rocksalt"]
        Crystal family tag (affects polarization usage downstream).
    # Electronic base (reference ~0 K or 300 K depending on model):
    Eg0_J : float
        Bandgap reference [J].
    varshni_alpha_J_per_K : float
        Varshni α [J/K].
    varshni_beta_K : float
        Varshni β [K].
    chi_J : float
        Electron affinity reference [J].
    me_dos_rel : float
        Electron DOS effective mass relative to m0 (dimensionless).
    mh_dos_rel : float
        Hole DOS effective mass relative to m0 (dimensionless).
    eps_r_300K : float
        Relative permittivity at 300 K (static).
    depsr_dT_per_K : float
        Linear slope of eps_r with T [1/K], placeholder.

    # Wurtzite piezo/spontaneous (used only if lattice == "wurtzite"):
    e31_C_per_m2 : float
    e33_C_per_m2 : float
    c13_Pa : float
    c33_Pa : float
    Psp_C_per_m2 : float
    """

    name: str
    lattice: Literal["wurtzite", "zincblende", "rocksalt"]
    Eg0_J: float
    varshni_alpha_J_per_K: float
    varshni_beta_K: float
    chi_J: float
    me_dos_rel: float
    mh_dos_rel: float
    eps_r_300K: float
    depsr_dT_per_K: float
    e31_C_per_m2: float = 0.0
    e33_C_per_m2: float = 0.0
    c13_Pa: float = 0.0
    c33_Pa: float = 0.0
    Psp_C_per_m2: float = 0.0


@dataclass(slots=True)
class MaterialPropsT:
    """
    T-dependent properties (resolved values at the query T).

    These are the quantities other modules should consume.
    """

    name: str
    T: float

    # Electronic
    Eg_J: float          # bandgap at T
    chi_J: float         # affinity (may be weakly T-dependent; here constant)
    me_dos_rel: float    # DOS mass rel. m0
    mh_dos_rel: float

    # Dielectric
    eps_r: float

    # Wurtzite piezo/spontaneous (unchanged with T here; models could add T-laws)
    e31_C_per_m2: float
    e33_C_per_m2: float
    c13_Pa: float
    c33_Pa: float
    Psp_C_per_m2: float

    # Metadata
    lattice: Literal["wurtzite", "zincblende", "rocksalt"]

    source: str          # "registry", "alloy(AlGaN,x=...)", etc.


# Alloy mixing rule tags
AlloyRule = Literal["linear", "bowing"]


# ---------------------------------------------------------------------
# Registry (placeholder values; non-credible!)
# ---------------------------------------------------------------------

# Minimal in-memory registry; real projects can extend via CSV/YAML ingest.
_REGISTRY: Dict[str, MaterialBase] = {
    "Si": MaterialBase(
        name="Si", lattice="zincblende",
        # Use Eg(0 K) ≈ 1.17 eV with Varshni (α=4.73e-4 eV/K, β=636 K)
        # so that Eg(300 K) ≈ 1.12 eV (matches the textbook value).
        Eg0_J=1.17 * 1.602e-19,           # Eg(0 K) ~ 1.17 eV
        varshni_alpha_J_per_K=4.73e-4 * 1.602e-19,
        varshni_beta_K=636.0,
        chi_J=4.05 * 1.602e-19,           # 4.05 eV affinity
        me_dos_rel=1.08,                  # DOS masses (typical)
        mh_dos_rel=0.56,
        eps_r_300K=11.7, depsr_dT_per_K=0.0
    ),
    "GaN": MaterialBase(
        name="GaN", lattice="wurtzite", Eg0_J=3.4 * 1.602e-19,
        varshni_alpha_J_per_K=9.0e-4 * 1.602e-19, varshni_beta_K=800.0,
        chi_J=4.1 * 1.602e-19, me_dos_rel=0.20, mh_dos_rel=1.00,
        eps_r_300K=9.5, depsr_dT_per_K=-1.0e-3, e31_C_per_m2=-0.49,
        e33_C_per_m2=0.73, c13_Pa=1.06e11, c33_Pa=3.98e11, Psp_C_per_m2=-0.029,
    ),
    "AlN": MaterialBase(
        name="AlN", lattice="wurtzite", Eg0_J=6.2 * 1.602e-19,
        varshni_alpha_J_per_K=1.0e-3 * 1.602e-19, varshni_beta_K=1200.0,
        chi_J=0.6 * 1.602e-19, me_dos_rel=0.30, mh_dos_rel=1.50,
        eps_r_300K=8.7, depsr_dT_per_K=-1.0e-3, e31_C_per_m2=-0.60,
        e33_C_per_m2=1.55, c13_Pa=1.08e11, c33_Pa=3.73e11, Psp_C_per_m2=-0.081,
    ),
    "Al": MaterialBase(
        name="Al", lattice="rocksalt", Eg0_J=0.0, varshni_alpha_J_per_K=0.0,
        varshni_beta_K=0.0, chi_J=0.0, me_dos_rel=1.0, mh_dos_rel=1.0,
        eps_r_300K=1.0, depsr_dT_per_K=0.0,
    ),
}

_SYSTEMS: Dict[str, Tuple[str, str]] = {"AlGaN": ("AlN", "GaN")}



# ---------------------------------------------------------------------
# Temperature and mixing models
# ---------------------------------------------------------------------


def _varshni_gap(Eg0_J: float, alpha_J_per_K: float, beta_K: float, T: float) -> float:
    """
    Varshni model for bandgap: Eg(T) = Eg0 - alpha*T^2/(T + beta)
    Parameters are in SI (J, J/K, K). Numerically stable for T>=0.
    """
    if T <= 0.0:
        return Eg0_J
    return Eg0_J - alpha_J_per_K * (T * T) / (T + beta_K)


def _linear_eps_r(eps_r_300K: float, depsr_dT_per_K: float, T: float) -> float:
    """Linear-in-T relative permittivity (placeholder)."""
    return eps_r_300K + depsr_dT_per_K * (T - 300.0)


def _mix_linear(a: float, b: float, x: float) -> float:
    """Linear composition mixing: x*A + (1-x)*B."""
    return x * a + (1.0 - x) * b


def _mix_bowing(a: float, b: float, x: float, bow: float) -> float:
    """Bowing mix: x*A + (1-x)*B - bow*x*(1-x)."""
    return _mix_linear(a, b, x) - bow * x * (1.0 - x)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


@lru_cache(maxsize=256)
def get_material(name: str, T: float = 300.0) -> MaterialPropsT:
    """
    Resolve T-dependent properties for a pure material.

    Parameters
    ----------
    name : str
        Key in the registry (case-sensitive).
    T : float
        Temperature [K].

    Returns
    -------
    MaterialPropsT
    """
    base = _REGISTRY.get(name)
    if base is None:
        raise KeyError(f"material '{name}' not found")

    Eg = _varshni_gap(base.Eg0_J, base.varshni_alpha_J_per_K, base.varshni_beta_K, T)
    eps_r = _linear_eps_r(base.eps_r_300K, base.depsr_dT_per_K, T)

    return MaterialPropsT(
        name=name,
        T=T,
        Eg_J=Eg,
        chi_J=base.chi_J,
        me_dos_rel=base.me_dos_rel,
        mh_dos_rel=base.mh_dos_rel,
        eps_r=eps_r,
        e31_C_per_m2=base.e31_C_per_m2,
        e33_C_per_m2=base.e33_C_per_m2,
        c13_Pa=base.c13_Pa,
        c33_Pa=base.c33_Pa,
        Psp_C_per_m2=base.Psp_C_per_m2,
        lattice=base.lattice,
        source="registry",
    )


@lru_cache(maxsize=512)
def get_alloy(
    system: str,
    x: float,
    T: float = 300.0,
    *,
    rule: AlloyRule = "bowing",
    bow_Eg_J: float = 0.0,
    bow_eps_r: float = 0.0,
    linearize_piezo: bool = True,
) -> MaterialPropsT:
    """
    Resolve T-dependent properties for a binary alloy A_x B_{1-x}.

    Parameters
    ----------
    system : str
        Alloy key, e.g., "AlGaN".
    x : float
        Mole fraction of the first endmember in _SYSTEMS[system], 0 <= x <= 1.
    T : float
        Temperature [K].
    rule : AlloyRule
        Mixing rule for scalar properties ("linear" or "bowing").
    bow_Eg_J : float
        Bowing parameter for Eg (J). Used if rule == "bowing".
    bow_eps_r : float
        Bowing parameter for eps_r (dimensionless). Used if rule == "bowing".
    linearize_piezo : bool
        If True, mix e31, e33, c13, c33, Psp linearly in composition.

    Returns
    -------
    MaterialPropsT
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must be within [0, 1]")

    try:
        name_A, name_B = _SYSTEMS[system]
    except KeyError as exc:
        raise KeyError(f"system '{system}' not found") from exc

    A = get_material(name_A, T)
    B = get_material(name_B, T)

    # --- Electronic: Eg, chi, masses
    if rule == "bowing":
        Eg_J = _mix_bowing(A.Eg_J, B.Eg_J, x, bow_Eg_J)
        eps_r = _mix_bowing(A.eps_r, B.eps_r, x, bow_eps_r)
    elif rule == "linear":
        Eg_J = _mix_linear(A.Eg_J, B.Eg_J, x)
        eps_r = _mix_linear(A.eps_r, B.eps_r, x)
    else:
        raise ValueError(f"unknown mixing rule: {rule!r}")

    chi_J = _mix_linear(A.chi_J, B.chi_J, x)
    me_dos_rel = _mix_linear(A.me_dos_rel, B.me_dos_rel, x)
    mh_dos_rel = _mix_linear(A.mh_dos_rel, B.mh_dos_rel, x)

    # --- Piezo/spontaneous (linear as a reasonable first pass)
    if linearize_piezo:
        e31 = _mix_linear(A.e31_C_per_m2, B.e31_C_per_m2, x)
        e33 = _mix_linear(A.e33_C_per_m2, B.e33_C_per_m2, x)
        c13 = _mix_linear(A.c13_Pa, B.c13_Pa, x)
        c33 = _mix_linear(A.c33_Pa, B.c33_Pa, x)
        Psp = _mix_linear(A.Psp_C_per_m2, B.Psp_C_per_m2, x)
    else:
        # Keep endmember values by default (or plug a more advanced model here).
        e31, e33, c13, c33, Psp = A.e31_C_per_m2, A.e33_C_per_m2, A.c13_Pa, A.c33_Pa, A.Psp_C_per_m2

    # Lattice tag: inherit from A if same family; else prefer "wurtzite" if either is.
    if A.lattice == B.lattice:
        lattice = A.lattice
    else:
        lattice = "wurtzite" if ("wurtzite" in (A.lattice, B.lattice)) else A.lattice

    return MaterialPropsT(
        name=f"{system}(x={x:.3f})",
        T=T,
        Eg_J=Eg_J,
        chi_J=chi_J,
        me_dos_rel=me_dos_rel,
        mh_dos_rel=mh_dos_rel,
        eps_r=eps_r,
        e31_C_per_m2=e31,
        e33_C_per_m2=e33,
        c13_Pa=c13,
        c33_Pa=c33,
        Psp_C_per_m2=Psp,
        lattice=lattice,
        source=f"alloy({system}, x={x:.3f}, rule={rule})",
    )


def list_materials() -> list[str]:
    """Return registry material keys."""
    return list(_REGISTRY.keys())


def list_systems() -> list[str]:
    """Return available alloy system keys."""
    return list(_SYSTEMS.keys())


# ---------------------------------------------------------------------
# Optional: hook for external ingestion (CSV/YAML) — outline only
# ---------------------------------------------------------------------


def _register_materials_from_table(_path: Optional[str] = None) -> None:
    """
    Placeholder for future CSV/YAML ingestion.
    Expected columns (SI units): name,lattice,Eg0_J,alpha_J_per_K,beta_K,
        chi_J,me_dos_rel,mh_dos_rel,eps_r_300K,depsr_dT_per_K,
        e31_C_per_m2,e33_C_per_m2,c13_Pa,c33_Pa,Psp_C_per_m2
    """
    # Implement as needed; keep parsing outside hot paths.
    return
