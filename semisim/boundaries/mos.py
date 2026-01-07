# semisim/boundaries/mos.py
"""
MOS boundary & interface bookkeeping for 1D stacks.

- Identifies the oxide/semiconductor interface in Geometry1D (by 'role').
- Composes total interface sheet:
      sigma_int = sigma_pol + sigma_fix + sigma_it   [C/m^2]
  mapped to the correct node index for Poisson's jump.
- Provides PoissonBC for left Dirichlet (surface) and selectable right BC.
- Small helpers for oxide field / gate-charge checks:
      V_ox = V_GB - Phi_ms - psi_s,  E_ox = V_ox / t_ox,  Q_G = eps_ox * E_ox.

Usage pattern:
    1) Find ox/semi interface node.
    2) Compute polarization sheets (physics/polarization.py) and/or traps sheets elsewhere.
    3) Compose sigma_int and pass (sheet_nodes, sheet_sigma) to PoissonSetup.
    4) Build PoissonBC(phi_s, right=..., phi_b=...).

Note:
- Enforcing V_GB partition exactly requires an outer scalar loop for psi_s; here we
  provide the formulas and leave psi_s choice to the workflow (Dirichlet at z=0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from ..geometry.builder import Geometry1D
from ..materials.database import EPS0  # reuse constant
from ..physics.poisson import PoissonBC  # type: ignore  # local import path if needed

__all__ = [
    "MOSStack",
    "find_oxide_semiconductor_interface",
    "compose_interface_sheets",
    "make_poisson_bc",
    "oxide_field_and_gate_charge",
]


@dataclass(slots=True)
class MOSStack:
    """
    Minimal MOS stack description around the oxide/semiconductor interface.
    """
    t_ox_m: float              # oxide thickness [m]
    eps_ox_rel: float          # oxide relative permittivity (-)
    Phi_ms_V: float            # metal-semiconductor work-function diff [V]
    V_GB_V: float              # applied gate-to-bulk bias [V]

    # Optional fixed/process charges at the interface (C/m^2)
    sigma_fix_Cm2: float = 0.0


def find_oxide_semiconductor_interface(geom: Geometry1D) -> Tuple[int, float]:
    """
    Return (node_index, z_interface) for the oxide/semiconductor boundary.

    Strategy:
      - Scan along nodes for a transition in role_id from oxide->semiconductor or vice versa.
      - Choose the node representing the interface (left-side node in our grid build).
    """
    role = geom.role_id
    z = geom.z
    # role ids: 0=semi, 1=oxide, 2=metal, 3=void (see builder)
    for i in range(1, role.size):
        a, b = role[i - 1], role[i]
        # detect a boundary involving oxide and semiconductor
        if (a == 1 and b == 0) or (a == 0 and b == 1):
            # interface located near z[i]; use left node index (i-1)
            idx = i - 1
            return idx, float(z[idx])
    raise RuntimeError("oxide/semiconductor interface not found in geometry")


def compose_interface_sheets(
    *,
    geom: Geometry1D,
    pol_sheet_nodes: Optional[np.ndarray] = None,
    pol_sigma_Cm2: Optional[np.ndarray] = None,
    sigma_fix_Cm2: float = 0.0,
    sigma_it_Cm2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose total interface sheets and map to node indices for Poisson.

    Parameters
    ----------
    geom : Geometry1D
        Geometry with interfaces.
    pol_sheet_nodes, pol_sigma_Cm2 : arrays from polarization (optional)
    sigma_fix_Cm2 : float
        Fixed/process charge at ox/semi interface [C/m^2].
    sigma_it_Cm2 : float
        Interface-trap *occupied* charge sheet [C/m^2] (from traps module).

    Returns
    -------
    sheet_nodes, sheet_sigma : arrays suitable for PoissonSetup
    """
    nodes = []
    sigma = []

    # Start with polarization sheets if provided
    if pol_sheet_nodes is not None and pol_sigma_Cm2 is not None:
        nodes.extend(list(np.asarray(pol_sheet_nodes, dtype=int)))
        sigma.extend(list(np.asarray(pol_sigma_Cm2, dtype=np.float64)))

    # Add (or merge) fixed + interface-trap sheet at the *oxide/semiconductor* boundary
    idx_if, _ = find_oxide_semiconductor_interface(geom)
    sigma_int = float(sigma_fix_Cm2 + sigma_it_Cm2)
    if abs(sigma_int) > 0.0:
        nodes.append(int(idx_if))
        sigma.append(sigma_int)

    if not nodes:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float64)

    # If multiple sheets land on the same node, accumulate
    nodes = np.asarray(nodes, dtype=int)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Consolidate by node index
    unique_nodes = np.unique(nodes)
    out_nodes = []
    out_sigma = []
    for i in unique_nodes:
        out_nodes.append(int(i))
        out_sigma.append(float(np.sum(sigma[nodes == i])))

    return np.asarray(out_nodes, dtype=int), np.asarray(out_sigma, dtype=np.float64)


def make_poisson_bc(
    phi_surf_V: float,
    *,
    right: Literal["Dirichlet", "NeumannZero"] = "Dirichlet",
    phi_bulk_V: float = 0.0,
) -> PoissonBC:
    """
    Build PoissonBC for the Poisson solver (Dirichlet at z=0).
    """
    return PoissonBC(phi_s_V=float(phi_surf_V), right=right, phi_b_V=float(phi_bulk_V))


def oxide_field_and_gate_charge(
    stack: MOSStack,
    psi_surf_V: float,
) -> Tuple[float, float, float]:
    """
    Given psi_s (semiconductor surface potential), compute oxide quantities:

        V_ox = V_GB - Phi_ms - psi_s,
        E_ox = V_ox / t_ox,
        Q_G = eps_ox * E_ox.

    Returns
    -------
    V_ox, E_ox, Q_G  (units: V, V/m, C/m^2)
    """
    V_ox = float(stack.V_GB_V - stack.Phi_ms_V - psi_surf_V)
    E_ox = V_ox / float(stack.t_ox_m)
    eps_ox = EPS0 * float(stack.eps_ox_rel)
    Q_G = eps_ox * E_ox
    return V_ox, E_ox, Q_G


#from semisim.physics.polarization import PolarizationSetup, compute_polarization

#pol = compute_polarization(
    #PolarizationSetup(
        #geom=geom,
        #mat=mat_fields,         # from resolve_material_fields()
       # eps_parallel=0.0,       # or per-node array
        #orientation="+c",
    #)
#)

#rho_extra = pol.rho_pol_Cm3
#sheet_nodes = pol.sheet_nodes
#heet_sigma = pol.sigma_pol_Cm2
#rom semisim.boundaries.mos import MOSStack, compose_interface_sheets, make_poisson_bc

# (optionally) add fixed/interface-trap sheets at the ox/semi interface
#sheet_nodes, sheet_sigma = compose_interface_sheets(
    #geom=geom,
    #pol_sheet_nodes=sheet_nodes,
    #pol_sigma_Cm2=sheet_sigma,
    #sigma_fix_Cm2=+0.0,         # process charge if any
    #igma_it_Cm2=+0.0,          # from traps (occupied)
#)

#bc = make_poisson_bc(phi_surf_V=<your surface potential>, right="Dirichlet", phi_bulk_V=0.0)
#from semisim.physics.poisson import PoissonSetup, solve_poisson_1d

#setup = PoissonSetup(
    #geom=geom,
    #mat=mat_fields,
    #bands=band_params,
    #bc=bc,
    #T_K=T,
    #mu_J=mu_ref,
    #ND_m3=ND, NA_m3=NA,
    #rho_extra_Cm3=rho_extra,
    #sheet_nodes=sheet_nodes,
    #sheet_sigma_Cm2=sheet_sigma,
#)
#res = solve_poisson_1d(setup)
